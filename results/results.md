### Shake-up: final selection per arm (seed 0)

| Competition | Arm | Final pick | Reported public | Private | Inflation | Private rank of pick | Spearman(public, private) |
|---|---|---|---|---|---|---|---|
| credit-g | Naive (Kaggle board) | cg.R2.B2 (code) | 0.700 | 0.715 | -0.015 | 16/19 | -0.60 |
| credit-g | HOLDOUT (Ladder + gate) | cg.R1.A1 (code) | 0.690 | 0.720 | -0.030 | 13/16 | -0.27 |
| diabetes | Naive (Kaggle board) | db.R1.B2 (code) | 0.870 | 0.746 | +0.124 | 4/20 | 0.53 |
| diabetes | HOLDOUT (Ladder + gate) | db.R1.C1 (code) | 0.840 | 0.729 | +0.111 | 11/11 | -0.23 |
| phoneme | Naive (Kaggle board) | ph.R2.D2 (code) | 0.840 | 0.892 | -0.052 | 1/12 | 1.00 |
| phoneme | HOLDOUT (Ladder + gate) | ph.R1.C2 (code) | 0.860 | 0.872 | -0.012 | 6/20 | 0.88 |
| **mean** | Naive (Kaggle board) | | 0.803 | 0.785 | +0.019 | | |
| **mean** | HOLDOUT (Ladder + gate) | | 0.797 | 0.774 | +0.023 | | |

### Gate log

| Arm | Rules proposed | Admitted | Rejected | Evidence ids stripped | Structure changes (kept / reverted / invalid) |
|---|---|---|---|---|---|
| Naive (Kaggle board) | 60 | 60 | 0 | 1 | 0 / 1 / 0 |
| HOLDOUT (Ladder + gate) | 26 | 12 | 14 | 45 | 0 / 0 / 0 |

### Same submissions, two selection rules (2000 redraws of the n=100 public set)

Each team's real submissions are held fixed; only which 100 test rows are public changes. This ignores adaptivity, so it understates the naive team's overfitting.

| Submission pool | Competition | Seed | Subs | Kaggle rule: reported / private / inflation | Ladder rule: reported / private / inflation |
|---|---|---|---|---|---|
| holdout team | credit-g | 0 | 16 | 0.779 / 0.737 / +0.042 | 0.767 / 0.727 / +0.040 |
| holdout team | diabetes | 0 | 11 | 0.796 / 0.770 / +0.026 | 0.789 / 0.768 / +0.021 |
| holdout team | phoneme | 0 | 20 | 0.895 / 0.877 / +0.018 | 0.890 / 0.876 / +0.013 |
| naive team | credit-g | 0 | 19 | 0.759 / 0.718 / +0.041 | 0.746 / 0.715 / +0.031 |
| naive team | diabetes | 0 | 20 | 0.791 / 0.768 / +0.023 | 0.784 / 0.765 / +0.019 |
| naive team | phoneme | 0 | 12 | 0.891 / 0.890 / +0.001 | 0.891 / 0.890 / +0.001 |
| **mean** | | | | 0.818 / 0.793 / +0.025 | 0.811 / 0.790 / +0.021 |

### Boosting attack (random labels, no skill)

credit-g, public n=100, private n=400, k=400 random submissions, mean of 30 runs.

| Board | Reported public | Private |
|---|---|---|
| kaggle | 0.871 | 0.516 |
| ladder | 0.636 | 0.529 |

### Efficiency (Crusoe Managed Inference, all Nemotron)

| Scope | Live LLM calls | Prompt tokens | Completion tokens | Est. cost |
|---|---|---|---|---|
| Arena run `m2` (both arms, 3 competitions x 4 rounds) | 202 | 445,584 | 125,231 | $0.082 |
| Everything today (incl. smoke tests, aborted runs) | 585 | | | $0.206 |

Cached-token share reported by Crusoe in `usage.prompt_tokens_details`: 0.003 (Lightning returns `null` for this field, so this undercounts; see the cache probe).
HOLDOUT final picks beat a plain logistic-regression baseline by 3.22 private-accuracy points on average (baseline 0.741), at $0.0255 of arena inference per point.
