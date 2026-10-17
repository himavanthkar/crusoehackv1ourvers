from holdout.gate import admit, render


KNOWN = {"cg.R1.A1": 0.78, "cg.R1.B1": 0.80, "cg.R1.C1": 0.81}


def test_holdout_rejects_rules_without_ladder_accepted_evidence():
    rules = [{"rule": "use LightGBM", "evidence": ["cg.R1.B1", "cg.R1.C1"]},
             {"rule": "ensemble everything", "evidence": ["cg.R1.C1"]}]
    admitted, log = admit(rules, "holdout", accepted_ids={"cg.R1.B1"}, known=KNOWN)
    assert [r["rule"] for r in admitted] == ["use LightGBM"]
    assert admitted[0]["evidence"] == [{"id": "cg.R1.B1", "reported": 0.80}]
    assert log[0]["stripped"] == ["cg.R1.C1"]
    assert not log[1]["admitted"] and "noise" in log[1]["reason"]


def test_naive_admits_any_real_citation_but_not_fabricated_ids():
    rules = [{"rule": "a", "evidence": ["cg.R1.C1"]}, {"rule": "b", "evidence": ["made.up.id"]}, {"rule": "c"}]
    admitted, _ = admit(rules, "naive", accepted_ids=set(), known=KNOWN)
    assert [r["rule"] for r in admitted] == ["a"]


def test_playbook_caps():
    rules = [{"rule": f"rule {i} " + "x" * 300, "evidence": ["cg.R1.A1"]} for i in range(20)]
    admitted, _ = admit(rules, "naive", accepted_ids=set(), known=KNOWN)
    assert len(admitted) <= 12
    assert len(render(admitted)) // 4 <= 1200


def test_decorated_evidence_ids_are_normalized():
    rules = [{"rule": "a", "evidence": ["cg.R1.B1 (0.80000)", "cg.R1.C1 (0.81000), cg.R1.A1"]}]
    admitted, log = admit(rules, "naive", accepted_ids=set(), known=KNOWN)
    assert log[0]["evidence_proposed"] == ["cg.R1.B1", "cg.R1.C1", "cg.R1.A1"]
    assert len(admitted) == 1
