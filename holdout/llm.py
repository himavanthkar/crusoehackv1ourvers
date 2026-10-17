"""Crusoe Managed Inference client: concurrency cap, backoff on 429/5xx, disk cache, usage + cached-token logging."""
from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from dataclasses import dataclass

import openai
from openai import OpenAI

from . import config as C

_client: OpenAI | None = None
_sem = threading.BoundedSemaphore(C.LLM_CONCURRENCY)
_lock = threading.Lock()
_key_locks: dict[str, threading.Lock] = {}
LOG_FILE = C.RUNS_DIR / "logs" / "llm_calls.jsonl"


class BudgetExceeded(RuntimeError):
    pass


class Totals:
    calls = 0
    cache_hits = 0
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    usd = 0.0

    @classmethod
    def as_dict(cls) -> dict:
        return {k: getattr(cls, k) for k in ("calls", "cache_hits", "prompt_tokens", "completion_tokens", "cached_tokens", "usd")}


@dataclass
class Reply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None
    latency_s: float
    from_cache: bool
    model: str


def client() -> OpenAI:
    global _client
    with _lock:
        if _client is None:
            if not C.CRUSOE_API_KEY:
                raise RuntimeError("CRUSOE_API_KEY missing: copy .env.example to .env")
            _client = OpenAI(base_url=C.CRUSOE_BASE_URL, api_key=C.CRUSOE_API_KEY, max_retries=0, timeout=120)
        return _client


def cost(model: str, prompt: int, completion: int, cached: int | None) -> float:
    p, c, cr = C.PRICING.get(model, (0.3, 2.4, 0.15))
    cached = cached or 0
    return ((prompt - cached) * p + cached * cr + completion * c) / 1e6


def _hash(model: str, messages: list[dict], temperature: float) -> str:
    return hashlib.sha256(json.dumps([model, messages, temperature], sort_keys=True).encode()).hexdigest()


def _log(row: dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(LOG_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")


def chat(model: str, messages: list[dict], *, temperature: float, max_tokens: int = 4096,
         role: str = "", tag: str = "", use_cache: bool = True, fallback: str | None = None) -> Reply:
    h = _hash(model, messages, temperature)
    path = C.CACHE_DIR / f"{h}.json"
    with _lock:
        klock = _key_locks.setdefault(h, threading.Lock())
    with klock:  # identical in-flight requests (e.g. both arms in round 1) are made once
        if use_cache and path.exists():
            d = json.loads(path.read_text())
            with _lock:
                Totals.cache_hits += 1
            _log({"ts": time.time(), "role": role, "tag": tag, "model": model, "hash": h[:16], "from_cache": True})
            return Reply(d["text"], d["prompt_tokens"], d["completion_tokens"], d.get("cached_tokens"), 0.0, True, model)
        if Totals.usd > C.BUDGET_USD:
            raise BudgetExceeded(f"estimated spend ${Totals.usd:.2f} passed BUDGET_USD=${C.BUDGET_USD}")
        r = _call(model, messages, temperature, max_tokens, fallback=fallback)
        usd = cost(r.model, r.prompt_tokens, r.completion_tokens, r.cached_tokens)
        with _lock:
            Totals.calls += 1
            Totals.prompt_tokens += r.prompt_tokens
            Totals.completion_tokens += r.completion_tokens
            Totals.cached_tokens += r.cached_tokens or 0
            Totals.usd += usd
        _log({"ts": time.time(), "role": role, "tag": tag, "model": r.model, "requested_model": model, "hash": h[:16], "from_cache": False,
              "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
              "cached_tokens": r.cached_tokens, "latency_s": round(r.latency_s, 2), "usd": round(usd, 6)})
        if use_cache and r.text.strip():
            C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": r.text, "prompt_tokens": r.prompt_tokens,
                                        "completion_tokens": r.completion_tokens, "cached_tokens": r.cached_tokens}))
        return r


class _Pacer:
    """Adaptive per-model pacing: spaces request starts, widens the gap on 429, narrows it on success.
    The Crusoe key is shared with another session, so we cannot know the remaining quota in advance."""

    def __init__(self):
        self.gap = {}
        self.next = {}
        self.lock = threading.Lock()

    def wait(self, model: str):
        with self.lock:
            gap = self.gap.setdefault(model, 0.5)
            now = time.time()
            t = max(now, self.next.get(model, 0.0))
            self.next[model] = t + gap
        if t > now:
            time.sleep(t - now)

    def ok(self, model: str):
        with self.lock:
            self.gap[model] = max(0.5, self.gap.get(model, 0.5) * 0.9)

    def throttled(self, model: str):
        with self.lock:
            self.gap[model] = min(6.0, self.gap.get(model, 0.5) * 1.6 + 0.5)
            self.next[model] = time.time() + 3 * self.gap[model]


_pacer = _Pacer()


def _call(model: str, messages: list[dict], temperature: float, max_tokens: int, tries: int = 12,
          fallback: str | None = None) -> Reply:
    """Retries with backoff. If `fallback` is given and the model is rate-limited 3 times (the key is shared with
    another session), switch to the fallback model, which has its own per-model limit. Reply.model says who answered."""
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if "emotron" in model else {}
    last = None
    throttled = 0
    for attempt in range(tries):
        if fallback and throttled >= 3 and model != fallback:
            model = fallback
        _pacer.wait(model)
        with _sem:
            t0 = time.time()
            try:
                resp = client().chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                        max_tokens=max_tokens, extra_body=extra or None)
                dt = time.time() - t0
            except openai.BadRequestError as e:
                if extra:
                    extra, last = {}, e
                    continue
                raise
            except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError,
                    openai.APITimeoutError) as e:
                last, dt = e, None
                if isinstance(e, openai.RateLimitError):
                    _pacer.throttled(model)
                    throttled += 1
        if dt is None:
            time.sleep(min(45.0, 1.5 * 2 ** attempt) + random.uniform(0, 2.0))
            continue
        _pacer.ok(model)
        msg = resp.choices[0].message
        text = msg.content or ""
        if not text.strip():
            d = msg.model_dump()
            text = d.get("reasoning_content") or d.get("reasoning") or ""
        u = resp.usage
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details else None
        return Reply(text, u.prompt_tokens or 0, u.completion_tokens or 0, cached, dt, False, model)
    raise RuntimeError(f"{model} failed after {tries} tries: {last}")


# ---------------------------------------------------------------- parsing helpers

_FENCE = re.compile(r"```([a-zA-Z0-9_+-]*)[ \t]*\n(.*?)```", re.S)


def code_blocks(text: str, lang: str = "python") -> list[str]:
    return [b for tag, b in _FENCE.findall(text or "") if tag.lower() in (lang, "py", "python3")]


def extract_json(text: str):
    """First parseable JSON object in text (tolerates code fences and prose)."""
    text = text or ""
    for tag, body in _FENCE.findall(text):
        if tag.lower() in ("json", ""):
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                pass
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None
