"""One research trajectory: reads the shared round prefix + its own history, writes solution.py or calls ensemble().

Prompt layout for MemoryAlloy prefix reuse: every researcher in a round sends a byte-identical prefix
[system prompt + tool schema + playbook + competition card + leaderboard snapshot at round start];
researcher-specific content (id, direction hint, own history) goes at the END.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SYSTEM = """You are a researcher on an AI Kaggle team competing in tabular binary-classification competitions.
Your job: submit predictions that will score well on the PRIVATE leaderboard (hidden test rows), not just the public one.

Write a complete solution.py that defines:
    def fit_predict(X_train, y_train, X_test) -> np.ndarray   # 0/1 labels, one per row of X_test
- X_train, X_test: pandas DataFrames (categorical columns use pandas 'category' dtype; there may be missing values).
- y_train: numpy int array of 0/1.
- Allowed: numpy, pandas, scikit-learn, scipy, lightgbm (xgboost and catboost are NOT installed). Put every import
  at the top of the file. Variables must be defined before use. No network, no file reads or writes, no printing large output.
- It must finish in under 60 seconds on ONE cpu core: use n_jobs=1, keep grids small, at most ~5-fold CV.
- You may use cross-validation on the training data to choose models. You never see test labels.

TOOL (instead of writing code): ensemble(submission_ids) majority-votes the test predictions of earlier
submissions listed on the leaderboard. Use an odd number (3, 5 or 7) of ids that scored successfully.

Respond in exactly this format and nothing else:
IDEA: <one sentence: what you are trying and why>
then EITHER one ```python code block containing the full solution.py
OR one line: ENSEMBLE: <id>, <id>, <id>"""


def shared_prefix(card: str, playbook_text: str, board_text: str, round_no: int, rounds: int) -> str:
    return (f"=== TEAM PLAYBOOK ===\n{playbook_text}\n\n"
            f"=== COMPETITION ===\n{card}\n\n"
            f"=== PUBLIC LEADERBOARD (team submissions so far, as reported by the leaderboard) ===\n{board_text}\n\n"
            f"Round {round_no} of {rounds}.\n")


def researcher_suffix(rid: str, hint: str, own_history: str, step: int, steps: int) -> str:
    return (f"\n=== YOU ===\nYou are researcher {rid}. Submission {step} of {steps} for you this round.\n"
            f"Your own history in this competition:\n{own_history or '(none yet)'}\n\n"
            f"Your research direction this round: {hint}\n"
            f"Follow the playbook where it applies. Respond in the required format.")


def build_messages(prefix: str, suffix: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prefix + suffix}]


@dataclass
class Action:
    kind: str           # code | ensemble | invalid
    idea: str
    code: str = ""
    ids: tuple = ()


_IDEA = re.compile(r"IDEA:\s*(.+)")
_ENS = re.compile(r"ENSEMBLE\s*[:(]\s*\[?([A-Za-z0-9_.,'\"\s-]+)\]?\)?", re.I)
_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def parse_action(text: str) -> Action:
    text = text or ""
    m = _IDEA.search(text)
    idea = (m.group(1).strip() if m else text.strip().splitlines()[0] if text.strip() else "")[:240]
    blocks = [b for b in _FENCE.findall(text) if "def fit_predict" in b]
    if blocks:
        return Action("code", idea, code=blocks[0])
    e = _ENS.search(text)
    if e:
        ids = tuple(x.strip(" '\"") for x in e.group(1).split(",") if x.strip(" '\""))
        if ids:
            return Action("ensemble", idea, ids=ids)
    if "def fit_predict" in text:
        start = min(i for i in (text.find("import "), text.find("from "), text.find("def fit_predict")) if i >= 0)
        return Action("code", idea, code=text[start:])
    return Action("invalid", idea)
