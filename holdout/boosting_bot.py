"""Blum and Hardt boosting attack (Section 5): climb a public leaderboard with zero skill.

Submit k uniformly random label vectors; keep those the board says are at least 50% accurate; submit their
coordinate-wise majority. Against the Ladder, keep the vectors the Ladder accepted (the paper's adapted attack).
The attacker's leaderboard entry is its best reported score; we track the private accuracy of that same entry.
`python -m holdout.boosting_bot` writes results/boosting_bot.json and results/boosting_bot.png
"""
from __future__ import annotations

import copy
import json

import numpy as np

from .config import RESULTS_DIR
from .leaderboard import BOARDS


def attack(board_name: str, y_test: np.ndarray, is_public: np.ndarray, k_max: int, checkpoints, rng) -> list[dict]:
    pub_idx = np.flatnonzero(is_public)
    priv = ~is_public
    board = BOARDS[board_name](y_test[pub_idx], pub_idx)
    m = len(y_test)
    kept_sum = np.zeros(m)
    kept_n = 0
    subs = []  # (preds, reported)
    out = []
    for j in range(1, k_max + 1):
        u = rng.integers(0, 2, m)
        v = board.submit(u, sub_id=str(j))
        subs.append((u, v.reported))
        keep = v.reported >= 0.5 if board_name == "kaggle" else v.accepted
        if keep:
            kept_sum += u
            kept_n += 1
        if j in checkpoints:
            # final step of an attack with k=j queries: submit the majority vote to a copy of the board
            b = copy.deepcopy(board)
            maj = (kept_sum > kept_n / 2).astype(int) if kept_n else rng.integers(0, 2, m)
            vm = b.submit(maj, sub_id="maj")
            if board_name == "kaggle":
                cands = subs + [(maj, vm.reported)]
                sel, rep = max(cands, key=lambda t: t[1])
            else:
                sel = maj if b.best_id == "maj" else subs[int(b.best_id) - 1][0]
                rep = b.best_reported
            out.append({"k": j, "reported_public": rep,
                        "true_public": float(np.mean(sel[pub_idx] == y_test[pub_idx])),
                        "private": float(np.mean(sel[priv] == y_test[priv]))})
    return out


def run(k_max: int = 400, reps: int = 30, seed: int = 0, comp_name: str = "credit-g") -> dict:
    from .data import make_competition
    comp = make_competition(comp_name, seed)
    checkpoints = sorted(set([1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400]) & set(range(1, k_max + 1)))
    res = {"competition": comp_name, "public_n": int(comp.is_public.sum()), "private_n": int((~comp.is_public).sum()),
           "k_max": k_max, "reps": reps, "checkpoints": checkpoints, "boards": {}}
    for name in ("kaggle", "ladder"):
        runs = [attack(name, comp.y_test, comp.is_public, k_max, set(checkpoints), np.random.default_rng(1000 * seed + r))
                for r in range(reps)]
        agg = {}
        for key in ("reported_public", "true_public", "private"):
            arr = np.array([[row[key] for row in rr] for rr in runs])
            agg[key] = {"mean": arr.mean(0).round(4).tolist(), "std": arr.std(0).round(4).tolist()}
        res["boards"][name] = agg
    return res


def plot(res: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = res["checkpoints"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, title) in zip(axes, (("kaggle", "Kaggle mechanism (naive)"), ("ladder", "Ladder mechanism (HOLDOUT)"))):
        b = res["boards"][name]
        for key, label, color in (("reported_public", "public score the board reports", "#d62728"),
                                  ("private", "private score (sealed)", "#1f77b4")):
            m, s = np.array(b[key]["mean"]), np.array(b[key]["std"])
            ax.plot(ks, m, label=label, color=color, lw=2)
            ax.fill_between(ks, m - s, m + s, color=color, alpha=0.15)
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("random submissions k")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy of the attacker's leaderboard entry")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Boosting attack with random labels: {res['competition']}, public n={res['public_n']}, "
                 f"private n={res['private_n']}, mean ± sd over {res['reps']} runs", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    res = run()
    (RESULTS_DIR / "boosting_bot.json").write_text(json.dumps(res, indent=1))
    plot(res, RESULTS_DIR / "boosting_bot.png")
    for name, b in res["boards"].items():
        print(f"{name:7s} k={res['checkpoints'][-1]}: reported public {b['reported_public']['mean'][-1]:.3f}  "
              f"private {b['private']['mean'][-1]:.3f}")
