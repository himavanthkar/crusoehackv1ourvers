"""Two arms, one difference: the leaderboard mechanism (Kaggle vs Ladder) and the curator's admission rule.

`python -m holdout.arena --run-id m2 --comps credit-g,diabetes,phoneme --rounds 4`
writes runs/<run_id>/{events.jsonl, arena.json} and runs/<run_id>/raw/ (LLM replies, predictions; gitignored).
"""
from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

from . import config as C
from .curator import curate
from .data import Competition, make_competition
from .gate import admit, render
from .leaderboard import BOARDS
from .llm import BudgetExceeded, Totals, chat
from .researcher import build_messages, parse_action, researcher_suffix, shared_prefix
from .sandbox import run_solution

ABBR = {"credit-g": "cg", "diabetes": "db", "phoneme": "ph", "qsar-biodeg": "qs",
        "blood-transfusion-service-center": "bt"}
ARMS = {"naive": ("kaggle", "naive"), "holdout": ("ladder", "holdout")}


@dataclass
class Sub:
    id: str
    arm: str
    comp: str
    round: int
    researcher: str
    step: int
    kind: str = "invalid"
    idea: str = ""
    hint: str = ""
    ok: bool = False
    error: str = ""
    code: str = ""
    ensemble_ids: list = field(default_factory=list)
    reported: float | None = None
    accepted: bool = False
    true_public: float | None = None   # sealed
    private: float | None = None       # sealed
    order: int = -1
    sandbox_s: float = 0.0
    llm_tokens: int = 0


def default_structure() -> dict:
    ids = [chr(ord("A") + i) for i in range(C.N_RESEARCHERS)]
    return {"researchers": ids, "hints": {r: C.DIRECTION_HINTS[i % len(C.DIRECTION_HINTS)] for i, r in enumerate(ids)},
            "subs": C.SUBS_PER_RESEARCHER}


class Arm:
    def __init__(self, name: str, mechanism: str, gate_mode: str, run_dir, emit, playbook=None):
        self.name, self.mechanism, self.gate_mode = name, mechanism, gate_mode
        self.run_dir, self._emit = run_dir, emit
        self.playbook: list = playbook or []
        self.structure = default_structure()
        self.subs: dict[str, Sub] = {}
        self.preds: dict[str, np.ndarray] = {}
        self.accepted_ids: set = set()
        self.lock = threading.Lock()
        self.gate_log: list = []
        self.structure_log: list = []
        self.playbook_history: list = [{"comp": None, "round": 0, "rules": []}]
        self.per_round: list = []
        self.finals: dict = {}
        self._order = 0

    def emit(self, type_: str, **kw):
        self._emit({"arm": self.name, "type": type_, **kw})

    # ------------------------------------------------------------ prompt pieces (no sealed fields)
    def board_text(self, comp: Competition) -> str:
        rows = [s for s in self.subs.values() if s.comp == comp.name and s.ok]
        rows.sort(key=lambda s: s.order)
        if not rows:
            return "(no submissions yet)"
        lines = ["id | researcher | kind | reported public score | new best? | idea"]
        for s in rows:
            lines.append(f"{s.id} | {s.researcher} | {s.kind} | {s.reported:.5f} | {'NEW BEST' if s.accepted else '-'} | {s.idea[:140]}")
        return "\n".join(lines)

    def own_history(self, comp: Competition, rid: str) -> str:
        mine = sorted((s for s in self.subs.values() if s.comp == comp.name and s.researcher == rid), key=lambda s: s.order)
        if not mine:
            return ""
        out = []
        for s in mine:
            res = f"reported public {s.reported:.5f}{' (NEW BEST)' if s.accepted else ''}" if s.ok else f"FAILED: {s.error[-300:]}"
            out.append(f"- {s.id} [{s.kind}] {s.idea[:160]} -> {res}")
        last_code = next((s.code for s in reversed(mine) if s.kind == "code" and s.code), "")
        if last_code:
            out.append(f"Your most recent code:\n```python\n{last_code[:2500]}\n```")
        return "\n".join(out)

    def round_table(self, comp: Competition, round_no: int) -> str:
        rows = sorted((s for s in self.subs.values() if s.comp == comp.name and s.round == round_no), key=lambda s: s.order)
        lines = ["id | researcher | direction | kind | reported public score | new best? | idea"]
        for s in rows:
            score = f"{s.reported:.5f}" if s.ok else "FAILED"
            ens = f" (ensemble of {', '.join(s.ensemble_ids)})" if s.kind == "ensemble" else ""
            lines.append(f"{s.id} | {s.researcher} | {s.hint[:40]} | {s.kind}{ens} | {score} | "
                         f"{'NEW BEST' if s.accepted else '-'} | {s.idea[:160]}")
        return "\n".join(lines)

    # ------------------------------------------------------------ one researcher, one round
    def researcher_round(self, comp, board, round_no, rounds, rid, prefix):
        hint = self.structure["hints"][rid]
        steps = self.structure["subs"]
        for step in range(1, steps + 1):
            sid = f"{ABBR.get(comp.name, comp.name[:2])}.R{round_no}.{rid}{step}"
            sub = Sub(sid, self.name, comp.name, round_no, rid, step, hint=hint)
            msgs = build_messages(prefix, researcher_suffix(rid, hint, self.own_history(comp, rid), step, steps))
            try:
                reply = chat(C.RESEARCHER_MODEL, msgs, temperature=C.RESEARCHER_TEMP, max_tokens=3000,
                             role="researcher", tag=f"{self.name}/{sid}")
            except BudgetExceeded:
                raise
            except Exception as e:  # an API failure costs this submission, not the whole arm
                sub.error = f"LLM call failed: {repr(e)[:300]}"
                with self.lock:
                    sub.order = self._order
                    self._order += 1
                    self.subs[sid] = sub
                self.emit("submission", comp=comp.name, round=round_no, id=sid, researcher=rid, kind="invalid",
                          idea="", ok=False, error=sub.error, reported=None, accepted=False, true_public=None,
                          private=None, ensemble_ids=[], mechanism=self.mechanism)
                continue
            sub.llm_tokens = reply.prompt_tokens + reply.completion_tokens
            (self.run_dir / "raw").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "raw" / f"{self.name}_{sid}.txt").write_text(reply.text)
            act = parse_action(reply.text)
            sub.kind, sub.idea = act.kind, act.idea
            preds = None
            if act.kind == "code":
                sub.code = act.code
                rr = run_solution(act.code, comp.X_train, comp.y_train, comp.X_test)
                sub.sandbox_s = round(rr.duration_s, 2)
                if rr.ok:
                    preds = rr.preds
                else:
                    sub.error = rr.error or "sandbox error"
            elif act.kind == "ensemble":
                sub.ensemble_ids = list(act.ids)
                with self.lock:
                    good = [i for i in act.ids if i in self.preds and self.subs[i].comp == comp.name]
                if len(good) >= 2:
                    stack = np.stack([self.preds[i] for i in good])
                    m = stack.mean(0)
                    preds = np.where(m == 0.5, stack[0], (m > 0.5).astype(int))
                    sub.ensemble_ids = good
                else:
                    sub.error = f"ensemble needs >=2 valid earlier submission ids from this competition, got {list(act.ids)}"
            else:
                sub.error = "could not parse a code block or ENSEMBLE line"
            with self.lock:
                sub.order = self._order
                self._order += 1
                if preds is not None:
                    v = board.submit(preds, sid)
                    sub.ok, sub.reported, sub.accepted = True, v.reported, v.accepted
                    sub.true_public = round(v.true_public, 5)
                    pi = comp.private_idx
                    sub.private = round(float(np.mean(preds[pi] == comp.y_test[pi])), 5)
                    self.preds[sid] = preds
                    if v.accepted:
                        self.accepted_ids.add(sid)
                self.subs[sid] = sub
            self.emit("submission", comp=comp.name, round=round_no, id=sid, researcher=rid, kind=sub.kind,
                      idea=sub.idea, ok=sub.ok, error=sub.error[-200:], reported=sub.reported, accepted=sub.accepted,
                      true_public=sub.true_public, private=sub.private, ensemble_ids=sub.ensemble_ids,
                      mechanism=self.mechanism)

    # ------------------------------------------------------------ structure changes
    def apply_structure(self, change) -> dict | None:
        if not isinstance(change, dict):
            return None
        s = copy.deepcopy(self.structure)
        t = change.get("type")
        try:
            if t == "reassign_hints":
                asg = {k: str(v)[:200] for k, v in (change.get("assignment") or {}).items() if k in s["hints"] and str(v).strip()}
                if not asg:
                    return None
                s["hints"].update(asg)
            elif t == "set_subs":
                v = int(change.get("value"))
                if v not in (1, 2) or v == s["subs"] or v * len(s["researchers"]) > 8:
                    return None
                s["subs"] = v
            elif t == "add_researcher":
                if len(s["researchers"]) >= 5 or (len(s["researchers"]) + 1) * s["subs"] > 8:
                    return None
                rid = chr(ord("A") + len(s["researchers"]))
                while rid in s["hints"]:
                    rid = chr(ord(rid) + 1)
                s["researchers"].append(rid)
                s["hints"][rid] = str(change.get("hint") or C.DIRECTION_HINTS[0])[:200]
            elif t == "remove_researcher":
                rid = str(change.get("id"))
                if rid not in s["researchers"] or len(s["researchers"]) <= 2:
                    return None
                s["researchers"].remove(rid)
            else:
                return None
        except (TypeError, ValueError):
            return None
        return s

    # ------------------------------------------------------------ a competition
    def run_competition(self, comp: Competition, rounds: int, curate_rounds: bool = True):
        board = BOARDS[self.mechanism](comp.y_test[comp.public_idx], comp.public_idx)
        pending = None  # (old_structure, change, round_applied)
        for rnd in range(1, rounds + 1):
            self.emit("round_start", comp=comp.name, round=rnd, playbook=render(self.playbook),
                      structure=copy.deepcopy(self.structure))
            prefix = shared_prefix(comp.card(), render(self.playbook), self.board_text(comp), rnd, rounds)
            with ThreadPoolExecutor(len(self.structure["researchers"])) as ex:
                futs = [ex.submit(self.researcher_round, comp, board, rnd, rounds, rid, prefix)
                        for rid in self.structure["researchers"]]
                for f in futs:
                    f.result()
            round_subs = [s for s in self.subs.values() if s.comp == comp.name and s.round == rnd]
            best_id = self.selection(comp)
            self.per_round.append({"comp": comp.name, "round": rnd, "best_id": best_id,
                                   "best_reported": self.subs[best_id].reported if best_id else None,
                                   "best_private": self.subs[best_id].private if best_id else None,
                                   "accepted_this_round": sum(s.accepted for s in round_subs),
                                   "submissions": len(round_subs), "failed": sum(not s.ok for s in round_subs)})
            # evaluate a structure change made after the previous round
            if pending:
                old, change, _ = pending
                improved = any(s.accepted for s in round_subs)
                verdict = "kept" if improved else "reverted"
                if not improved:
                    self.structure = old
                self.structure_log.append({"comp": comp.name, "round": rnd, "change": change, "verdict": verdict,
                                           "reason": ("round produced a leaderboard-accepted improvement" if improved
                                                      else "no leaderboard-accepted improvement in the next round")})
                self.emit("structure_verdict", comp=comp.name, round=rnd, change=change, verdict=verdict)
                pending = None
            if not curate_rounds:
                continue
            known = {s.id: s.reported for s in self.subs.values() if s.ok}
            try:
                proposal, _ = curate(self.name, comp.name, rnd, self.playbook, self.round_table(comp, rnd),
                                     self.structure, allow_structure=rnd < rounds)
            except BudgetExceeded:
                raise
            except Exception as e:  # curator failure keeps the old playbook
                proposal = {"rules": None, "error": repr(e)[:200]}
            if isinstance(proposal.get("rules"), list):
                admitted, log = admit(proposal["rules"], self.gate_mode, self.accepted_ids, known)
                for row in log:
                    row.update(comp=comp.name, round=rnd)
                self.gate_log.extend(log)
                old_rules = {r["rule"] for r in self.playbook}
                self.playbook = admitted
                self.playbook_history.append({"comp": comp.name, "round": rnd, "rules": copy.deepcopy(admitted)})
                self.emit("playbook", comp=comp.name, round=rnd, gate=log, rules=[r["rule"] for r in admitted],
                          added=[r["rule"] for r in admitted if r["rule"] not in old_rules],
                          rationale=str(proposal.get("rationale", ""))[:400])
            change = proposal.get("structure_change") if rnd < rounds else None
            new = self.apply_structure(change)
            if new is not None:
                pending = (self.structure, change, rnd)
                self.structure = new
                self.emit("structure_change", comp=comp.name, round=rnd, change=change)
            elif change:
                self.structure_log.append({"comp": comp.name, "round": rnd, "change": change, "verdict": "invalid",
                                           "reason": "rejected by structure validator"})
        best = self.selection(comp)
        self.finals[comp.name] = best
        self.emit("final", comp=comp.name, id=best, reported=self.subs[best].reported if best else None,
                  private=self.subs[best].private if best else None,
                  true_public=self.subs[best].true_public if best else None)

    def selection(self, comp: Competition) -> str | None:
        """Final selection: naive = best reported public score (earliest on ties); HOLDOUT = the Ladder's best."""
        ok = sorted((s for s in self.subs.values() if s.comp == comp.name and s.ok), key=lambda s: s.order)
        if not ok:
            return None
        if self.mechanism == "kaggle":
            return max(ok, key=lambda s: (s.reported, -s.order)).id
        acc = [s for s in ok if s.accepted]
        return acc[-1].id

    def dump(self) -> dict:
        return {"name": self.name, "mechanism": self.mechanism, "gate_mode": self.gate_mode,
                "subs": [asdict(s) for s in sorted(self.subs.values(), key=lambda s: (s.comp, s.order))],
                "finals": self.finals, "per_round": self.per_round, "gate_log": self.gate_log,
                "structure_log": self.structure_log, "playbook_final": self.playbook,
                "playbook_history": self.playbook_history, "structure_final": self.structure}



class Recorder:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.f = open(run_dir / "events.jsonl", "w")
        self.listeners = []

    def __call__(self, ev: dict):
        ev = {"t": round(time.time() - self.t0, 2), **ev}
        with self.lock:
            self.f.write(json.dumps(ev, default=str) + "\n")
            self.f.flush()
        for fn in self.listeners:
            try:
                fn(ev)
            except Exception:
                pass


def run_arms(run_id: str, comps: list[str], rounds: int, seed: int = 0, arms=("naive", "holdout"), on_event=None) -> dict:
    run_dir = C.RUNS_DIR / run_id
    rec = Recorder(run_dir)
    if on_event:
        rec.listeners.append(on_event)
    competitions = [make_competition(c, seed) for c in comps]
    rec({"arm": None, "type": "run_start", "run_id": run_id, "comps": comps, "rounds": rounds, "seed": seed,
         "researcher_model": C.RESEARCHER_MODEL, "curator_model": C.CURATOR_MODEL,
         "public_n": {c.name: int(c.is_public.sum()) for c in competitions},
         "private_n": {c.name: int((~c.is_public).sum()) for c in competitions}})
    objs = {a: Arm(a, ARMS[a][0], ARMS[a][1], run_dir, rec) for a in arms}
    errors = {}

    def go(arm: Arm):
        try:
            for comp in competitions:
                arm.run_competition(comp, rounds)
        except Exception as e:
            errors[arm.name] = repr(e)
            rec({"arm": arm.name, "type": "error", "error": repr(e)[:500]})

    threads = [threading.Thread(target=go, args=(a,)) for a in objs.values()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out = {"run_id": run_id, "seed": seed, "comps": comps, "rounds": rounds,
           "models": {"researcher": C.RESEARCHER_MODEL, "curator": C.CURATOR_MODEL},
           "public_n": {c.name: int(c.is_public.sum()) for c in competitions},
           "private_n": {c.name: int((~c.is_public).sum()) for c in competitions},
           "arms": {a: o.dump() for a, o in objs.items()}, "llm_totals": Totals.as_dict(), "errors": errors,
           "wall_s": round(time.time() - rec.t0, 1)}
    (run_dir / "arena.json").write_text(json.dumps(out, indent=1, default=str))
    raw = run_dir / "raw"
    raw.mkdir(exist_ok=True)
    for a, o in objs.items():
        np.savez_compressed(raw / f"preds_{a}.npz", **o.preds)
    rec({"arm": None, "type": "run_end", "llm_totals": Totals.as_dict(), "errors": errors})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="m2")
    ap.add_argument("--comps", default="credit-g,diabetes,phoneme")
    ap.add_argument("--rounds", type=int, default=C.ROUNDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="naive,holdout")
    a = ap.parse_args()

    def show(ev):
        if ev["type"] == "submission":
            s = f"{ev['reported']:.3f}" if ev["ok"] else "FAIL"
            print(f"[{ev['t']:6.1f}s] {ev['arm']:7s} {ev['id']:12s} {ev['kind']:8s} {s:5s} "
                  f"{'NEW BEST' if ev['accepted'] else '':8s} priv={ev['private']}  {ev['idea'][:70]}", flush=True)
        elif ev["type"] in ("playbook",):
            adm = sum(r["admitted"] for r in ev["gate"])
            print(f"[{ev['t']:6.1f}s] {ev['arm']:7s} CURATOR {ev['comp']} R{ev['round']}: {adm}/{len(ev['gate'])} rules admitted "
                  f"| $ {Totals.usd:.3f} calls {Totals.calls} cache {Totals.cache_hits}", flush=True)
        elif ev["type"] in ("final", "structure_change", "structure_verdict", "error", "round_start"):
            print(f"[{ev['t']:6.1f}s] {ev.get('arm')} {ev['type']} {json.dumps({k: v for k, v in ev.items() if k not in ('t', 'arm', 'type', 'playbook')}, default=str)[:300]}", flush=True)

    out = run_arms(a.run_id, a.comps.split(","), a.rounds, a.seed, tuple(a.arms.split(",")), on_event=show)
    print(json.dumps(out["llm_totals"]), "errors:", out["errors"], "wall", out["wall_s"])
