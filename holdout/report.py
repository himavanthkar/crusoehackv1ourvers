"""results.json, charts and markdown tables, computed only from logged run data.

`python -m holdout.report --arena m2 --transfer transfer`
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from . import config as C

RESAMPLE_RUNS: list[str] = []
ARM_LABEL = {"naive": "Naive (Kaggle board)", "holdout": "HOLDOUT (Ladder + gate)"}
COLORS = {"naive": "#d62728", "holdout": "#2ca02c", "gen-0": "#7f7f7f", "naive-final": "#d62728", "holdout-final": "#2ca02c"}


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def shakeup(arena: dict) -> dict:
    out = {}
    for arm, d in arena["arms"].items():
        rows = {}
        for comp in arena["comps"]:
            subs = [s for s in d["subs"] if s["comp"] == comp]
            ok = [s for s in subs if s["ok"]]
            fid = d["finals"].get(comp)
            sel = next((s for s in ok if s["id"] == fid), None)
            if not sel:
                rows[comp] = {"final_id": None}
                continue
            pub = np.array([s["true_public"] for s in ok])
            priv = np.array([s["private"] for s in ok])
            rho = float(spearmanr(pub, priv).statistic) if len(ok) > 2 and pub.std() > 0 and priv.std() > 0 else None
            # rank shake-up: top 10 by public score, with private rank
            order_pub = sorted(ok, key=lambda s: (-s["true_public"], s["order"]))
            priv_rank = {s["id"]: i + 1 for i, s in enumerate(sorted(ok, key=lambda s: (-s["private"], s["order"])))}
            top = [{"public_rank": i + 1, "id": s["id"], "kind": s["kind"], "public": s["true_public"],
                    "reported": s["reported"], "private": s["private"], "private_rank": priv_rank[s["id"]],
                    "moved": i + 1 - priv_rank[s["id"]]} for i, s in enumerate(order_pub[:10])]
            rows[comp] = {
                "final_id": fid, "final_kind": sel["kind"], "reported_public": sel["reported"],
                "true_public": sel["true_public"], "private": sel["private"],
                "inflation": round(sel["reported"] - sel["private"], 5),
                "best_private_any": max(priv), "mean_private_all": round(float(priv.mean()), 5),
                "final_private_rank": priv_rank[fid], "n_ok": len(ok), "n_subs": len(subs),
                "n_failed": len(subs) - len(ok), "n_ensembles": sum(s["kind"] == "ensemble" for s in ok),
                "n_accepted": sum(s["accepted"] for s in ok), "spearman_public_private": rho, "top10": top,
            }
        finals = [r for r in rows.values() if r.get("final_id")]
        agg = {k: round(float(np.mean([r[k] for r in finals])), 5) for k in
               ("reported_public", "true_public", "private", "inflation")} if finals else {}
        out[arm] = {"per_comp": rows, "mean": agg}
    return out


def gate_summary(arena: dict) -> dict:
    out = {}
    for arm, d in arena["arms"].items():
        log = [r for r in d["gate_log"] if "evidence_proposed" in r]
        adm = [r for r in log if r["admitted"]]
        rej = [r for r in log if not r["admitted"]]
        out[arm] = {"proposed": len(log), "admitted": len(adm), "rejected": len(rej),
                    "reasons": dict(Counter(r["reason"] for r in rej)),
                    "evidence_stripped": sum(len(r["stripped"]) for r in log),
                    "examples_rejected": [{"comp": r["comp"], "round": r["round"], "rule": r["rule"],
                                           "evidence": r["evidence_proposed"], "reason": r["reason"]} for r in rej[:6]],
                    "structure_log": d["structure_log"],
                    "playbook_final": [r["rule"] for r in d["playbook_final"]]}
    return out


def transfer_summary(tr: dict | None) -> dict | None:
    if not tr:
        return None
    rows = [{k: v for k, v in r.items() if k != "subs"} for r in tr["results"]]
    by_pb = {}
    for r in rows:
        if r["final_private"] is not None:
            by_pb.setdefault(r["playbook"], []).append(r["final_private"])
    return {"rows": rows, "mean_final_private": {k: round(float(np.mean(v)), 5) for k, v in by_pb.items()},
            "mechanism": tr["mechanism"], "seeds": tr["seeds"], "playbooks": {k: [x["rule"] for x in v] for k, v in tr["playbooks"].items()}}


def efficiency(arena: dict, tr: dict | None, sh: dict | None = None) -> dict:
    """Per-run totals come from each run's own counters; the project total is every live call logged today
    (including smoke tests and aborted runs) from runs/logs/llm_calls.jsonl."""
    log = C.RUNS_DIR / "logs" / "llm_calls.jsonl"
    calls = [json.loads(l) for l in open(log)] if log.exists() else []
    live = [c for c in calls if not c.get("from_cache")]
    by_model = {}
    for c in live:
        r = by_model.setdefault(c["model"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                             "cached_tokens": 0, "calls_reporting_cached": 0, "usd": 0.0})
        r["calls"] += 1
        r["prompt_tokens"] += c.get("prompt_tokens", 0)
        r["completion_tokens"] += c.get("completion_tokens", 0)
        if c.get("cached_tokens") is not None:
            r["calls_reporting_cached"] += 1
            r["cached_tokens"] += c["cached_tokens"]
        r["usd"] = round(r["usd"] + c.get("usd", 0.0), 6)
    tot_prompt = sum(r["prompt_tokens"] for r in by_model.values())
    tot_cached = sum(r["cached_tokens"] for r in by_model.values())
    arena_t = arena.get("llm_totals", {})
    tr_t = (tr or {}).get("llm_totals", {})
    out = {"arena_run": {k: arena_t.get(k) for k in ("calls", "prompt_tokens", "completion_tokens", "cached_tokens", "usd")},
           "transfer_run": {k: tr_t.get(k) for k in ("calls", "prompt_tokens", "completion_tokens", "cached_tokens", "usd")} if tr_t else None,
           "project_total": {"live_calls": len(live), "replayed_from_local_cache": len(calls) - len(live),
                             "total_tokens": tot_prompt + sum(r["completion_tokens"] for r in by_model.values()),
                             "usd": round(sum(r["usd"] for r in by_model.values()), 4),
                             "cached_token_share_reported": round(tot_cached / tot_prompt, 4) if tot_prompt else None,
                             "by_model": by_model}}
    if sh:
        # $ per point of private accuracy: arena cost / (HOLDOUT mean private - mean private of a plain LR baseline)
        base = baseline_private(arena)
        hp = sh["holdout"]["mean"].get("private")
        if base is not None and hp is not None:
            gain = (hp - base) * 100
            out["baseline_lr_mean_private"] = round(base, 5)
            out["holdout_private_gain_points_vs_lr"] = round(gain, 2)
            out["usd_per_private_point"] = round(arena_t.get("usd", 0) / gain, 4) if gain > 0 else None
    return out


def baseline_private(arena: dict) -> float | None:
    """Private accuracy of the plain logistic-regression baseline (holdout/sandbox.BASELINE) on the same splits."""
    from .data import make_competition
    from .sandbox import BASELINE, run_solution
    vals = []
    for comp in arena["comps"]:
        c = make_competition(comp, arena["seed"])
        r = run_solution(BASELINE, c.X_train, c.y_train, c.X_test)
        if r.ok:
            vals.append(float(np.mean(r.preds[c.private_idx] == c.y_test[c.private_idx])))
    return float(np.mean(vals)) if vals else None


def resampled_selection(arena_ids: list[str], draws: int = 2000, seed: int = 7) -> dict | None:
    """Hold each team's real submissions fixed (in their real order) and redraw the n=100 public set many times.
    For every draw, apply the Kaggle selection rule (max public, earliest on ties) and the Ladder rule (run the
    parameter-free Ladder over the submissions in order, take its best). Report reported-public, private and
    inflation of each pick. This isolates the selection mechanism from one lucky split. It ignores adaptivity
    (the agents chose submissions after seeing the real public scores), so it understates naive overfitting."""
    from .data import make_competition
    from .leaderboard import KaggleBoard, LadderBoard
    rng = np.random.default_rng(seed)
    rows = []
    for rid in arena_ids:
        a = _load(C.RUNS_DIR / rid / "arena.json")
        if not a:
            continue
        for arm in ("naive", "holdout"):
            pz = C.RUNS_DIR / rid / f"preds_{arm}.npz"
            if not pz.exists():
                pz = C.RUNS_DIR / rid / "raw" / f"preds_{arm}.npz"
            if not pz.exists():
                continue
            preds = np.load(pz)
            for comp in a["comps"]:
                c = make_competition(comp, a["seed"])
                ok = [x for x in a["arms"][arm]["subs"] if x["comp"] == comp and x["ok"] and x["id"] in preds.files]
                ok.sort(key=lambda x: x["order"])
                if len(ok) < 3:
                    continue
                P = np.stack([preds[x["id"]] for x in ok])
                y = c.y_test
                m = len(y)
                res = {"kaggle": [], "ladder": []}
                for _ in range(draws):
                    pub = rng.choice(m, c.public_idx.size, replace=False)
                    mask = np.zeros(m, bool)
                    mask[pub] = True
                    priv_acc = (P[:, ~mask] == y[~mask]).mean(1)
                    kb, lb = KaggleBoard(y[pub], pub), LadderBoard(y[pub], pub)
                    for i in range(len(ok)):
                        kb.submit(P[i], i)
                        lb.submit(P[i], i)
                    res["kaggle"].append((kb.best_reported, priv_acc[kb.best_id]))
                    res["ladder"].append((lb.best_reported, priv_acc[lb.best_id]))
                for rule, v in res.items():
                    v = np.array(v)
                    rows.append({"run": rid, "seed": a["seed"], "pool": arm, "comp": comp, "rule": rule, "n_subs": len(ok),
                                 "reported": round(float(v[:, 0].mean()), 5), "private": round(float(v[:, 1].mean()), 5),
                                 "inflation": round(float((v[:, 0] - v[:, 1]).mean()), 5)})
    if not rows:
        return None
    pooled = {}
    for rule in ("kaggle", "ladder"):
        rr = [r for r in rows if r["rule"] == rule]
        pooled[rule] = {k: round(float(np.mean([r[k] for r in rr])), 5) for k in ("reported", "private", "inflation")}
        pooled[rule]["n_pools"] = len(rr)
    return {"draws": draws, "rows": rows, "pooled": pooled}


# ---------------------------------------------------------------- charts

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def chart_shakeup(sh: dict, comps: list, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.2
    x = np.arange(len(comps))
    for i, arm in enumerate(("naive", "holdout")):
        rows = sh[arm]["per_comp"]
        rep = [rows[c].get("reported_public") or 0 for c in comps]
        prv = [rows[c].get("private") or 0 for c in comps]
        ax.bar(x + (2 * i - 1.5) * w, rep, w, color=COLORS[arm], alpha=0.35, label=f"{ARM_LABEL[arm]}: reported public")
        ax.bar(x + (2 * i - 0.5) * w, prv, w, color=COLORS[arm], label=f"{ARM_LABEL[arm]}: private")
        for j in range(len(comps)):
            ax.annotate(f"{rep[j] - prv[j]:+.3f}", (x[j] + (2 * i - 1) * w, max(rep[j], prv[j]) + 0.005),
                        ha="center", fontsize=8, color=COLORS[arm])
    ax.set_xticks(x, comps)
    lo = min(min(sh[a]["per_comp"][c].get("private") or 1 for c in comps) for a in sh) - 0.08
    ax.set_ylim(max(0.4, lo), 1.0)
    ax.set_ylabel("accuracy of the final selection")
    ax.set_title("Shake-up: final selection, reported public vs private (labels = inflation)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def chart_rounds(arena: dict, path):
    plt = _plt()
    comps = arena["comps"]
    fig, axes = plt.subplots(1, len(comps), figsize=(4 * len(comps), 3.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, comp in zip(axes, comps):
        for arm in ("naive", "holdout"):
            pr = [r for r in arena["arms"][arm]["per_round"] if r["comp"] == comp]
            rr = [r["round"] for r in pr]
            ax.plot(rr, [r["best_reported"] for r in pr], "-o", color=COLORS[arm], label=f"{arm}: best reported public")
            ax.plot(rr, [r["best_private"] for r in pr], "--s", color=COLORS[arm], alpha=0.7, label=f"{arm}: its private")
        ax.set_title(comp)
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=7)
    fig.suptitle("Per round: the team's best reported public score vs the private score of that same submission", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def chart_transfer(ts: dict, path):
    plt = _plt()
    rows = ts["rows"]
    comps = sorted({r["comp"] for r in rows})
    pbs = ["gen-0", "naive-final", "holdout-final"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(comps))
    w = 0.26
    for i, pb in enumerate(pbs):
        vals = []
        for c in comps:
            v = [r["final_private"] for r in rows if r["comp"] == c and r["playbook"] == pb and r["final_private"] is not None]
            vals.append(float(np.mean(v)) if v else 0)
        b = ax.bar(x + (i - 1) * w, vals, w, color=COLORS[pb], label=pb)
        ax.bar_label(b, fmt="%.3f", fontsize=8)
    ax.set_xticks(x, comps, fontsize=8)
    allv = [r["final_private"] for r in rows if r["final_private"]]
    ax.set_ylim(max(0.4, min(allv) - 0.08) if allv else 0.4, 1.0)
    ax.set_ylabel("private accuracy of final selection")
    ax.set_title("Transfer to unseen competitions (one fresh round, Ladder board for all)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def markdown(res: dict) -> str:
    L = []
    sh = res["shakeup"]
    L.append("### Shake-up: final selection per arm (seed %s)\n" % res["arena_seed"])
    L.append("| Competition | Arm | Final pick | Reported public | Private | Inflation | Private rank of pick | Spearman(public, private) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for comp in res["comps"]:
        for arm in ("naive", "holdout"):
            r = sh[arm]["per_comp"][comp]
            if not r.get("final_id"):
                continue
            rho = "n/a" if r["spearman_public_private"] is None else f"{r['spearman_public_private']:.2f}"
            L.append(f"| {comp} | {ARM_LABEL[arm]} | {r['final_id']} ({r['final_kind']}) | {r['reported_public']:.3f} | "
                     f"{r['private']:.3f} | {r['inflation']:+.3f} | {r['final_private_rank']}/{r['n_ok']} | {rho} |")
    for arm in ("naive", "holdout"):
        m = sh[arm]["mean"]
        if m:
            L.append(f"| **mean** | {ARM_LABEL[arm]} | | {m['reported_public']:.3f} | {m['private']:.3f} | {m['inflation']:+.3f} | | |")
    g = res["gate"]
    L.append("\n### Gate log\n")
    L.append("| Arm | Rules proposed | Admitted | Rejected | Evidence ids stripped | Structure changes (kept / reverted / invalid) |")
    L.append("|---|---|---|---|---|---|")
    for arm in ("naive", "holdout"):
        s = g[arm]
        c = Counter(x["verdict"] for x in s["structure_log"])
        L.append(f"| {ARM_LABEL[arm]} | {s['proposed']} | {s['admitted']} | {s['rejected']} | {s['evidence_stripped']} | "
                 f"{c.get('kept', 0)} / {c.get('reverted', 0)} / {c.get('invalid', 0)} |")
    if res.get("transfer"):
        t = res["transfer"]
        L.append("\n### Transfer to unseen competitions (Ladder board for all three)\n")
        L.append("| Competition | Seed | gen-0 (empty) | naive-final | HOLDOUT-final |")
        L.append("|---|---|---|---|---|")
        keys = sorted({(r["comp"], r["seed"]) for r in t["rows"]})
        for c, s in keys:
            v = {r["playbook"]: r["final_private"] for r in t["rows"] if r["comp"] == c and r["seed"] == s}
            f = lambda x: "n/a" if x is None else f"{x:.3f}"
            L.append(f"| {c} | {s} | {f(v.get('gen-0'))} | {f(v.get('naive-final'))} | {f(v.get('holdout-final'))} |")
        m = t["mean_final_private"]
        L.append(f"| **mean** | | {m.get('gen-0', 0):.3f} | {m.get('naive-final', 0):.3f} | {m.get('holdout-final', 0):.3f} |")
    rs = res.get("resampled_selection")
    if rs:
        L.append(f"\n### Same submissions, two selection rules ({rs['draws']} redraws of the n=100 public set)\n")
        L.append("Each team's real submissions are held fixed; only which 100 test rows are public changes. "
                 "This ignores adaptivity, so it understates the naive team's overfitting.\n")
        L.append("| Submission pool | Competition | Seed | Subs | Kaggle rule: reported / private / inflation | Ladder rule: reported / private / inflation |")
        L.append("|---|---|---|---|---|---|")
        keys = sorted({(r["pool"], r["comp"], r["seed"], r["run"]) for r in rs["rows"]})
        for pool, comp, sd, rid in keys:
            k = next(r for r in rs["rows"] if (r["pool"], r["comp"], r["seed"], r["run"], r["rule"]) == (pool, comp, sd, rid, "kaggle"))
            l = next(r for r in rs["rows"] if (r["pool"], r["comp"], r["seed"], r["run"], r["rule"]) == (pool, comp, sd, rid, "ladder"))
            L.append(f"| {pool} team | {comp} | {sd} | {k['n_subs']} | {k['reported']:.3f} / {k['private']:.3f} / {k['inflation']:+.3f} | "
                     f"{l['reported']:.3f} / {l['private']:.3f} / {l['inflation']:+.3f} |")
        pk, pl = rs["pooled"]["kaggle"], rs["pooled"]["ladder"]
        L.append(f"| **mean** | | | | {pk['reported']:.3f} / {pk['private']:.3f} / {pk['inflation']:+.3f} | "
                 f"{pl['reported']:.3f} / {pl['private']:.3f} / {pl['inflation']:+.3f} |")
    b = res.get("boosting_bot")
    if b:
        L.append("\n### Boosting attack (random labels, no skill)\n")
        L.append(f"{b['competition']}, public n={b['public_n']}, private n={b['private_n']}, k={b['k_max']} random submissions, "
                 f"mean of {b['reps']} runs.\n")
        L.append("| Board | Reported public | Private |")
        L.append("|---|---|---|")
        for name in ("kaggle", "ladder"):
            bb = b["boards"][name]
            L.append(f"| {name} | {bb['reported_public']['mean'][-1]:.3f} | {bb['private']['mean'][-1]:.3f} |")
    e = res["efficiency"]
    a, pt = e["arena_run"], e["project_total"]
    L.append("\n### Efficiency (Crusoe Managed Inference, all Nemotron)\n")
    L.append("| Scope | Live LLM calls | Prompt tokens | Completion tokens | Est. cost |")
    L.append("|---|---|---|---|---|")
    L.append(f"| Arena run `{res['arena_run']}` (both arms, 3 competitions x 4 rounds) | {a['calls']} | {a['prompt_tokens']:,} | {a['completion_tokens']:,} | ${a['usd']:.3f} |")
    if e.get("transfer_run"):
        t = e["transfer_run"]
        L.append(f"| Transfer run | {t['calls']} | {t['prompt_tokens']:,} | {t['completion_tokens']:,} | ${t['usd']:.3f} |")
    L.append(f"| Everything today (incl. smoke tests, aborted runs) | {pt['live_calls']} | | | ${pt['usd']:.3f} |")
    L.append(f"\nCached-token share reported by Crusoe in `usage.prompt_tokens_details`: {pt['cached_token_share_reported']} "
             f"(Lightning returns `null` for this field, so this undercounts; see the cache probe).")
    if e.get("usd_per_private_point") is not None:
        L.append(f"HOLDOUT final picks beat a plain logistic-regression baseline by {e['holdout_private_gain_points_vs_lr']:.2f} "
                 f"private-accuracy points on average (baseline {e['baseline_lr_mean_private']:.3f}), at "
                 f"${e['usd_per_private_point']:.4f} of arena inference per point.")
    elif e.get("holdout_private_gain_points_vs_lr") is not None:
        L.append(f"HOLDOUT final picks vs a plain logistic-regression baseline: {e['holdout_private_gain_points_vs_lr']:+.2f} private points.")
    cp = res.get("cache_probe")
    if cp:
        for m in cp["models"]:
            L.append(f"- Cache probe `{m['model']}`: median time-to-first-token {m['median_ttft_shared_s']:.3f} s with a shared "
                     f"~{m['shared_prefix'][0]['prompt_tokens']}-token prefix vs {m['median_ttft_unique_s']:.3f} s with unique prefixes; "
                     f"cached_tokens reported: shared {m['cached_tokens_shared']}, unique {m['cached_tokens_unique']}.")
    return "\n".join(L)


def build(arena_id: str = "m2", transfer_id: str | None = "transfer") -> dict:
    arena = _load(C.RUNS_DIR / arena_id / "arena.json")
    tr = _load(C.RUNS_DIR / transfer_id / "transfer.json") if transfer_id and transfer_id != "none" else None
    bb = _load(C.RESULTS_DIR / "boosting_bot.json")
    probe = _load(C.RESULTS_DIR / "cache_probe.json")
    res = {"arena_run": arena_id, "arena_seed": arena["seed"], "comps": arena["comps"], "rounds": arena["rounds"],
           "models": arena["models"], "public_n": arena["public_n"], "private_n": arena["private_n"],
           "shakeup": shakeup(arena), "gate": gate_summary(arena), "per_round": {a: d["per_round"] for a, d in arena["arms"].items()},
           "transfer": transfer_summary(tr), "boosting_bot": bb, "cache_probe": probe,
           "arena_errors": arena.get("errors"), "arena_wall_s": arena.get("wall_s")}
    res["efficiency"] = efficiency(arena, tr, res["shakeup"])
    res["resampled_selection"] = resampled_selection(RESAMPLE_RUNS or [arena_id])
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "results.json").write_text(json.dumps(res, indent=1, default=str))
    chart_shakeup(res["shakeup"], arena["comps"], C.RESULTS_DIR / "shakeup.png")
    chart_rounds(arena, C.RESULTS_DIR / "per_round.png")
    if res["transfer"]:
        chart_transfer(res["transfer"], C.RESULTS_DIR / "transfer.png")
    md = markdown(res)
    (C.RESULTS_DIR / "results.md").write_text(md + "\n")
    return res


def extra_seeds(run_ids: list[str]) -> dict | None:
    """Final-selection shake-up for additional arena runs (other split seeds), pooled with nothing typed by hand."""
    rows = []
    for rid in run_ids:
        a = _load(C.RUNS_DIR / rid / "arena.json")
        if not a:
            continue
        sh = shakeup(a)
        for arm in ("naive", "holdout"):
            for comp, r in sh[arm]["per_comp"].items():
                if r.get("final_id"):
                    rows.append({"run": rid, "seed": a["seed"], "arm": arm, "comp": comp, "reported_public": r["reported_public"],
                                 "private": r["private"], "inflation": r["inflation"]})
    if not rows:
        return None
    agg = {}
    for arm in ("naive", "holdout"):
        rr = [r for r in rows if r["arm"] == arm]
        agg[arm] = {k: {"mean": round(float(np.mean([r[k] for r in rr])), 5), "sd": round(float(np.std([r[k] for r in rr])), 5)}
                    for k in ("reported_public", "private", "inflation")} | {"n": len(rr)}
    return {"rows": rows, "pooled": agg}


def inject_readme(md: str):
    p = C.ROOT / "README.md"
    if not p.exists():
        return
    s = p.read_text()
    a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if a in s and b in s:
        s = s[: s.index(a) + len(a)] + "\n" + md + "\n" + s[s.index(b):]
        p.write_text(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", default="m2")
    ap.add_argument("--transfer", default="transfer")
    ap.add_argument("--seeds-runs", default="", help="comma-separated arena run ids to pool (e.g. m2,m2s1,m2s2)")
    a = ap.parse_args()
    if a.seeds_runs:
        RESAMPLE_RUNS.extend(a.seeds_runs.split(","))
    res = build(a.arena, a.transfer)
    md = (C.RESULTS_DIR / "results.md").read_text()
    if a.seeds_runs:
        es = extra_seeds(a.seeds_runs.split(","))
        if es:
            res["seeds_pooled"] = es
            (C.RESULTS_DIR / "results.json").write_text(json.dumps(res, indent=1, default=str))
            lines = ["\n### Pooled over split seeds (final selection, every dev competition x seed)\n",
                     "| Arm | n | Reported public | Private | Inflation (mean ± sd) |", "|---|---|---|---|---|"]
            for arm in ("naive", "holdout"):
                g = es["pooled"][arm]
                lines.append(f"| {ARM_LABEL[arm]} | {g['n']} | {g['reported_public']['mean']:.3f} | {g['private']['mean']:.3f} | "
                             f"{g['inflation']['mean']:+.3f} ± {g['inflation']['sd']:.3f} |")
            md += "\n".join(lines) + "\n"
            (C.RESULTS_DIR / "results.md").write_text(md)
    inject_readme(md)
    print(md)
