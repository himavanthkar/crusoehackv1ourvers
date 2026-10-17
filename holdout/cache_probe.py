"""MemoryAlloy probe: does a byte-identical prompt prefix get served from Crusoe's KV cache?

For each model: K calls sharing one long prefix (different short suffixes) vs K calls whose prefix is
unique (same length, shuffled). Streams each call and records time-to-first-token (TTFT) and any
`prompt_tokens_details.cached_tokens`. Calls are sequential to keep load on the shared key low.
`python -m holdout.cache_probe` -> results/cache_probe.json
"""
from __future__ import annotations

import json
import random
import statistics
import time

from . import config as C
from .llm import client
from .researcher import SYSTEM


def _prefix(seed: int | None) -> str:
    words = ("feature model tuning ensemble playbook leaderboard accuracy holdout ladder curator researcher "
             "gradient boosting logistic regularization cross validation categorical numeric missing").split()
    rng = random.Random(seed if seed is not None else 12345)
    body = "\n".join(f"{i}. " + " ".join(rng.choice(words) for _ in range(18)) for i in range(140))
    return SYSTEM + "\n\n=== SHARED CONTEXT ===\n" + body


def _one(model: str, prefix: str, suffix: str) -> dict:
    import openai
    for attempt in range(8):
        try:
            return _one_try(model, prefix, suffix)
        except (openai.RateLimitError, openai.InternalServerError):
            time.sleep(2 + 2 * attempt)
    return _one_try(model, prefix, suffix)


def _one_try(model: str, prefix: str, suffix: str) -> dict:
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    ttft, usage = None, None
    stream = client().chat.completions.create(
        model=model, messages=[{"role": "user", "content": prefix + "\n" + suffix}], temperature=0, max_tokens=8,
        stream=True, stream_options={"include_usage": True}, extra_body=extra)
    for ch in stream:
        if ttft is None and ch.choices and (ch.choices[0].delta.content or "").strip():
            ttft = time.time() - t0
        if getattr(ch, "usage", None):
            usage = ch.usage
    det = getattr(usage, "prompt_tokens_details", None) if usage else None
    return {"ttft_s": round(ttft or (time.time() - t0), 3), "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "cached_tokens": getattr(det, "cached_tokens", None) if det else None}


def probe(model: str, k: int = 6) -> dict:
    shared = _prefix(None)
    warm = _one(model, shared, "Say OK. (warm-up)")
    same = [_one(model, shared, f"Researcher {i}: say OK.") for i in range(k)]
    uniq = [_one(model, _prefix(1000 + i + int(time.time()) % 997), f"Researcher {i}: say OK.") for i in range(k)]
    med = lambda xs: round(statistics.median(xs), 3)
    return {"model": model, "k": k, "warmup": warm, "shared_prefix": same, "unique_prefix": uniq,
            "median_ttft_shared_s": med([x["ttft_s"] for x in same]),
            "median_ttft_unique_s": med([x["ttft_s"] for x in uniq]),
            "cached_tokens_shared": [x["cached_tokens"] for x in same],
            "cached_tokens_unique": [x["cached_tokens"] for x in uniq]}


if __name__ == "__main__":
    out = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "models": [probe(C.RESEARCHER_MODEL), probe(C.CURATOR_MODEL)]}
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "cache_probe.json").write_text(json.dumps(out, indent=1))
    for m in out["models"]:
        print(m["model"], "prompt_tokens", m["shared_prefix"][0]["prompt_tokens"], "TTFT shared", m["median_ttft_shared_s"],
              "unique", m["median_ttft_unique_s"], "cached shared", m["cached_tokens_shared"], "unique", m["cached_tokens_unique"])
