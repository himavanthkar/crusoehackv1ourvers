"""Transfer test: do the evolved playbooks help on competitions the team has never seen?

Three playbooks (gen-0 empty, naive-final, HOLDOUT-final) x unseen competitions. One fresh round each
(same researchers and budget), all scored under the Ladder for a fair comparison. Report private accuracy of
each final selection. `python -m holdout.transfer --from-run m2`
"""
from __future__ import annotations

import argparse
import json
import threading
import time

from . import config as C
from .arena import Arm, Recorder
from .data import UNSEEN, make_competition
from .llm import Totals


def run(from_run: str, run_id: str = "transfer", seeds=(0,), comps=None) -> dict:
    src = json.loads((C.RUNS_DIR / from_run / "arena.json").read_text())
    playbooks = {"gen-0": [], "naive-final": src["arms"]["naive"]["playbook_final"],
                 "holdout-final": src["arms"]["holdout"]["playbook_final"]}
    comps = comps or list(UNSEEN)
    run_dir = C.RUNS_DIR / run_id
    rec = Recorder(run_dir)
    rec({"arm": None, "type": "run_start", "run_id": run_id, "from_run": from_run, "comps": comps, "seeds": list(seeds),
         "playbooks": {k: [r["rule"] for r in v] for k, v in playbooks.items()}})
    results, errors = [], {}
    lock = threading.Lock()

    def go(pb_name, comp_name, seed):
        comp = make_competition(comp_name, seed)
        arm = Arm(pb_name, "ladder", "holdout", run_dir, rec, playbook=[dict(r) for r in playbooks[pb_name]])
        try:
            arm.run_competition(comp, rounds=1, curate_rounds=False)
        except Exception as e:
            errors[f"{pb_name}/{comp_name}/{seed}"] = repr(e)[:300]
        d = arm.dump()
        fid = d["finals"].get(comp_name)
        subs = [s for s in d["subs"] if s["comp"] == comp_name]
        sel = next((s for s in subs if s["id"] == fid), None)
        ok = [s for s in subs if s["ok"]]
        with lock:
            results.append({"playbook": pb_name, "comp": comp_name, "seed": seed, "final_id": fid,
                            "final_reported": sel["reported"] if sel else None,
                            "final_private": sel["private"] if sel else None,
                            "mean_private_all": round(sum(s["private"] for s in ok) / len(ok), 5) if ok else None,
                            "n_subs": len(subs), "n_failed": len(subs) - len(ok), "subs": subs})

    threads = [threading.Thread(target=go, args=(pb, c, s)) for s in seeds for c in comps for pb in playbooks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out = {"run_id": run_id, "from_run": from_run, "comps": comps, "seeds": list(seeds), "mechanism": "ladder",
           "playbooks": playbooks, "results": sorted(results, key=lambda r: (r["comp"], r["seed"], r["playbook"])),
           "errors": errors, "llm_totals": Totals.as_dict()}
    (run_dir / "transfer.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", default="m2")
    ap.add_argument("--run-id", default="transfer")
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    t0 = time.time()
    out = run(a.from_run, a.run_id, tuple(int(s) for s in a.seeds.split(",")))
    for r in out["results"]:
        print(f"{r['comp']:34s} seed={r['seed']} {r['playbook']:14s} final={r['final_id']} "
              f"reported={r['final_reported']} private={r['final_private']} mean_private_all={r['mean_private_all']} "
              f"failed={r['n_failed']}/{r['n_subs']}")
    print("errors", out["errors"], "llm", out["llm_totals"], f"{time.time() - t0:.0f}s")
