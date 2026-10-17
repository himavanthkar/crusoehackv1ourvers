"""Playbook admission gate. Plain Python, not an LLM.

HOLDOUT arm: a rule may only cite submissions the Ladder accepted; uncited/unaccepted evidence is stripped and a
rule left with no accepted evidence is rejected. Naive arm: any rule citing a real submission is admitted.
Both arms: at most PLAYBOOK_MAX_RULES rules and ~PLAYBOOK_MAX_TOKENS tokens.
"""
from __future__ import annotations

import re

from .config import PLAYBOOK_MAX_RULES, PLAYBOOK_MAX_TOKENS

_ID = re.compile(r"[A-Za-z0-9-]+\.R\d+\.[A-Z]\d+")


def evidence_ids(raw) -> list[str]:
    """Curators often copy ids with decorations, e.g. 'cg.R1.A1 (0.69000)'. Extract the bare ids, in order."""
    out = []
    for e in raw if isinstance(raw, list) else [raw]:
        for m in _ID.findall(str(e)):
            if m not in out:
                out.append(m)
    return out


def approx_tokens(s: str) -> int:
    return len(s) // 4 + 1


def admit(proposed: list, mode: str, accepted_ids: set, known: dict) -> tuple[list, list]:
    """known: id -> reported score. Returns (admitted rules, log rows)."""
    admitted, log = [], []
    for r in proposed if isinstance(proposed, list) else []:
        if not isinstance(r, dict):
            continue
        text = str(r.get("rule", "")).strip()[:400]
        ev = evidence_ids(r.get("evidence") or [])
        if not text:
            continue
        real = [e for e in ev if e in known]
        if mode == "holdout":
            keep = [e for e in real if e in accepted_ids]
            reason = "cites Ladder-accepted evidence" if keep else "no Ladder-accepted evidence (gain within noise)"
        else:
            keep = real
            reason = "cites reported results" if keep else "cites no real submission"
        row = {"rule": text, "evidence_proposed": ev, "evidence_kept": keep,
               "stripped": [e for e in ev if e not in keep], "admitted": bool(keep), "reason": reason}
        if keep and len(admitted) >= PLAYBOOK_MAX_RULES:
            row.update(admitted=False, reason=f"playbook full ({PLAYBOOK_MAX_RULES} rules)")
        log.append(row)
        if row["admitted"]:
            admitted.append({"rule": text, "evidence": [{"id": e, "reported": known[e]} for e in keep]})
    while admitted and approx_tokens(render(admitted)) > PLAYBOOK_MAX_TOKENS:
        dropped = admitted.pop()
        log.append({"rule": dropped["rule"], "admitted": False, "reason": f"over {PLAYBOOK_MAX_TOKENS}-token cap"})
    return admitted, log


def render(rules: list) -> str:
    if not rules:
        return "(empty: generation 0, no rules yet)"
    out = []
    for i, r in enumerate(rules, 1):
        ev = ", ".join(f"{e['id']} ({e['reported']:.5f})" for e in r["evidence"])
        out.append(f"{i}. {r['rule']}  [evidence: {ev}]")
    return "\n".join(out)
