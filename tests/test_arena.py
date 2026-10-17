from holdout.arena import Arm, Sub


def _arm(mech="ladder"):
    return Arm("t", mech, "holdout" if mech == "ladder" else "naive", run_dir=None, emit=lambda ev: None)


def test_structure_validator():
    a = _arm()
    assert a.apply_structure({"type": "set_subs", "value": 1})["subs"] == 1
    assert a.apply_structure({"type": "set_subs", "value": 5}) is None
    assert a.apply_structure({"type": "add_researcher", "hint": "x"}) is None  # 5 x 2 > 8 submissions
    assert a.apply_structure({"type": "remove_researcher", "id": "D"})["researchers"] == ["A", "B", "C"]
    s = a.apply_structure({"type": "reassign_hints", "assignment": {"A": "try kNN", "Z": "ignored"}})
    assert s["hints"]["A"] == "try kNN" and "Z" not in s["hints"]
    assert a.apply_structure({"type": "teleport"}) is None
    assert a.apply_structure(None) is None


class _C:
    name = "c"


def test_final_selection_rules():
    for mech, expect in (("kaggle", "b"), ("ladder", "a")):
        a = _arm(mech)
        a.subs["a"] = Sub("a", "t", "c", 1, "A", 1, ok=True, reported=0.70, accepted=True, order=0)
        # Kaggle: a noisy higher public score wins. Ladder: rejected as within noise, it reports 0.70 again.
        a.subs["b"] = Sub("b", "t", "c", 1, "B", 1, ok=True, reported=0.74 if mech == "kaggle" else 0.70,
                          accepted=mech == "kaggle", order=1)
        assert a.selection(_C()) == expect
