"""Curator: reads a round's results as reported by the arm's leaderboard, rewrites the playbook,
and may propose one team-structure change. Runs on the larger Nemotron (orchestrator tier)."""
from __future__ import annotations

from . import config as C
from .gate import render
from .llm import chat, extract_json

SYSTEM = """You are the curator of an AI Kaggle team that competes in a series of tabular binary-classification
competitions. After every round you rewrite the TEAM PLAYBOOK: the shared rules every researcher reads next round
and in FUTURE, DIFFERENT competitions. Good rules are general, transferable modelling advice (no dataset-specific
column names). Each rule must cite as evidence the submission ids whose reported leaderboard results support it.
Keep at most 12 rules, each one or two sentences. Keep, edit, merge or drop existing rules. When you keep or edit an
existing rule, carry over its existing evidence ids (earlier rounds and earlier competitions still count) and add
new ids only if this round's results support it.

You may also propose ONE change to the team structure (or null):
- {"type": "reassign_hints", "assignment": {"<researcher id>": "<new research direction, one sentence>"}}
- {"type": "set_subs", "value": 1 or 2}            (submissions per researcher per round)
- {"type": "add_researcher", "hint": "<direction>"}  (max 5 researchers, max 8 submissions per round)
- {"type": "remove_researcher", "id": "<researcher id>"}  (min 2 researchers)

Reply with JSON only:
{"rules": [{"rule": "...", "evidence": ["<submission id>", ...]}], "structure_change": null or {...}, "rationale": "<two sentences>"}"""


def curate(arm_name: str, comp_name: str, round_no: int, playbook: list, round_table: str, structure: dict,
           allow_structure: bool) -> tuple[dict, object]:
    team = "\n".join(f"- researcher {rid}: {structure['hints'][rid]}" for rid in structure["researchers"])
    user = (f"=== CURRENT PLAYBOOK ===\n{render(playbook)}\n\n"
            f"=== TEAM STRUCTURE ===\n{team}\nSubmissions per researcher per round: {structure['subs']}\n\n"
            f"=== ROUND {round_no} RESULTS ({comp_name}), as reported by the public leaderboard ===\n{round_table}\n\n"
            + ("Propose a structure_change or null." if allow_structure else
               "This is the last round of this competition: set structure_change to null.")
            + "\nRewrite the playbook now. JSON only.")
    reply = chat(C.CURATOR_MODEL, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                 temperature=C.CURATOR_TEMP, max_tokens=2500, role="curator", tag=f"{arm_name}/{comp_name}/R{round_no}")
    data = extract_json(reply.text) or {}
    if not isinstance(data, dict):
        data = {}
    return data, reply
