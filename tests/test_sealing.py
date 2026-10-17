"""Honesty rule: no agent prompt may contain private labels/scores, sealed public truth, or unseen-competition data."""
import numpy as np

from holdout.arena import Arm, Sub
from holdout.data import DEV, UNSEEN, make_competition
from holdout.gate import render
from holdout.researcher import SYSTEM, build_messages, researcher_suffix, shared_prefix


def _arm_with_history(comp):
    arm = Arm("holdout", "ladder", "holdout", run_dir=None, emit=lambda ev: None)
    for i, rid in enumerate("AB"):
        s = Sub(f"x.R1.{rid}1", "holdout", comp.name, 1, rid, 1, kind="code", idea="try lightgbm", ok=True,
                code="def fit_predict(a, b, c):\n    return [0]*len(c)\n", reported=0.71, accepted=(i == 0),
                true_public=0.123457, private=0.987653, order=i)
        arm.subs[s.id] = s
    return arm


def _all_prompts(comp, arm):
    prefix = shared_prefix(comp.card(), render(arm.playbook), arm.board_text(comp), 2, 4)
    msgs = build_messages(prefix, researcher_suffix("A", "model choice", arm.own_history(comp, "A"), 1, 2))
    return "\n".join(m["content"] for m in msgs) + "\n" + arm.round_table(comp, 1)


def test_no_sealed_scores_in_prompts():
    comp = make_competition("diabetes", 0)
    text = _all_prompts(comp, _arm_with_history(comp))
    assert "0.98765" not in text and "0.12345" not in text and "0.12346" not in text
    assert "private score" not in text.lower()


def test_no_private_labels_in_prompts():
    comp = make_competition("credit-g", 0)
    text = _all_prompts(comp, _arm_with_history(comp))
    labels = "".join(map(str, comp.y_test[comp.private_idx][:40]))
    assert labels not in text.replace(" ", "")


def test_no_unseen_competition_data_in_dev_prompts():
    unseen_cols = set()
    for name in UNSEEN:
        uc = make_competition(name, 0)
        unseen_cols |= {c for c in uc.X_train.columns if len(c) > 6}
    for name in DEV:
        comp = make_competition(name, 0)
        text = _all_prompts(comp, _arm_with_history(comp)) + SYSTEM
        for u in UNSEEN:
            assert u not in text
        dev_cols = set(comp.X_train.columns)
        for c in unseen_cols - dev_cols:
            assert c not in text, c
