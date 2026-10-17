### Key findings (generated from results.json)

1. **The mechanism works against a zero-skill attacker.** 400 random label vectors: the Kaggle board reports 0.871 public while private is 0.516 (inflation +0.355); the Ladder reports 0.636 vs 0.529 (+0.108). With n=100 the Ladder limits the climb but does not remove it.
2. **The gate changes what the team learns.** Over 2 split seeds the naive curator got 135/135 proposed playbook rules admitted; the HOLDOUT gate admitted 43/76 and stripped 114 citations of submissions whose "gain" the Ladder judged to be noise.
3. **Learned noise hurts on unseen competitions.** One fresh round on 2 unseen competitions x 2 seeds: submissions that ran: gen-0 9/32, naive-final 2/32, HOLDOUT-final 16/32. Mean private accuracy of the final pick: gen-0 0.780 (4/4 cells with a valid pick), naive-final 0.855 (2/4), HOLDOUT-final 0.814 (4/4). The naive playbook (every reported gain admitted) steered researchers to HistGradientBoosting with native categorical handling; on all-numeric blood-transfusion every naive-final submission failed in the sandbox, in both seeds.
4. **The final-pick shake-up with real LLM teams was small and mixed** (reported honestly). Pooled over 6 competition x seed cells, inflation of the final pick was +0.028 ± 0.061 (naive) vs +0.033 ± 0.055 (HOLDOUT): no clear difference at this N. Holding the submissions fixed and redrawing the public set 2000 times, the Ladder selection rule cut inflation from +0.027 to +0.021 at a private cost of -0.1 points. LLM submissions are highly correlated, so they are far less adversarial than the boosting attack.
5. **Cheap on Crusoe.** The full two-arm arena run cost $0.082 (202 live Nemotron calls); everything today including smoke tests $0.23. MemoryAlloy: with a shared prefix, Nemotron Super reported 3456 of 4118 prompt tokens cached (0 with unique prefixes); Lightning's median time-to-first-token was 0.325 s shared vs 0.408 s unique.

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

### Transfer to unseen competitions (Ladder board for all three)

Cells: private accuracy of the final selection (submissions that ran / submitted). 'no valid submission' means every researcher's code failed in the sandbox.

| Competition | Seed | gen-0 (empty) | naive-final | HOLDOUT-final |
|---|---|---|---|---|
| blood-transfusion-service-center | 0 | 0.748 (3/8 ran) | no valid submission (0/8 ran) | 0.781 (5/8 ran) |
| blood-transfusion-service-center | 1 | 0.646 (2/8 ran) | no valid submission (0/8 ran) | 0.755 (4/8 ran) |
| qsar-biodeg | 0 | 0.862 (2/8 ran) | 0.848 (1/8 ran) | 0.860 (3/8 ran) |
| qsar-biodeg | 1 | 0.864 (2/8 ran) | 0.862 (1/8 ran) | 0.860 (4/8 ran) |
| **mean of valid picks** | | 0.780 over 4/4 cells | 0.855 over 2/4 cells | 0.814 over 4/4 cells |
| **submissions that ran** | | 9/32 | 2/32 | 16/32 |

### Same submissions, two selection rules (2000 redraws of the n=100 public set)

Each team's real submissions are held fixed; only which 100 test rows are public changes. This ignores adaptivity, so it understates the naive team's overfitting.

| Submission pool | Competition | Seed | Subs | Kaggle rule: reported / private / inflation | Ladder rule: reported / private / inflation |
|---|---|---|---|---|---|
| holdout team | credit-g | 0 | 16 | 0.779 / 0.737 / +0.042 | 0.767 / 0.727 / +0.040 |
| holdout team | credit-g | 1 | 15 | 0.762 / 0.717 / +0.044 | 0.750 / 0.720 / +0.030 |
| holdout team | diabetes | 0 | 11 | 0.796 / 0.770 / +0.026 | 0.789 / 0.768 / +0.021 |
| holdout team | diabetes | 1 | 21 | 0.769 / 0.733 / +0.035 | 0.761 / 0.743 / +0.018 |
| holdout team | phoneme | 0 | 20 | 0.895 / 0.877 / +0.018 | 0.890 / 0.876 / +0.013 |
| holdout team | phoneme | 1 | 14 | 0.919 / 0.898 / +0.021 | 0.914 / 0.896 / +0.018 |
| naive team | credit-g | 0 | 19 | 0.759 / 0.718 / +0.041 | 0.746 / 0.715 / +0.031 |
| naive team | credit-g | 1 | 23 | 0.765 / 0.733 / +0.032 | 0.754 / 0.727 / +0.027 |
| naive team | diabetes | 0 | 20 | 0.791 / 0.768 / +0.023 | 0.784 / 0.765 / +0.019 |
| naive team | diabetes | 1 | 22 | 0.747 / 0.723 / +0.023 | 0.739 / 0.716 / +0.023 |
| naive team | phoneme | 0 | 12 | 0.891 / 0.890 / +0.001 | 0.891 / 0.890 / +0.001 |
| naive team | phoneme | 1 | 17 | 0.903 / 0.888 / +0.015 | 0.899 / 0.891 / +0.008 |
| **mean** | | | | 0.814 / 0.788 / +0.027 | 0.807 / 0.786 / +0.021 |

### NVIDIA OpenShell run (`runs/openshell_r1`)

One full round on diabetes, both arms, with every agent-written solution executed inside an OpenShell sandbox: 16 submissions, 10 ran successfully (the rest are researcher code errors), mean 2.35 s per sandbox execution, 19.5 s wall time.

| Arm | Final pick | Reported public | Private |
|---|---|---|---|
| naive | db.R1.B2 | 0.890 | 0.739 |
| holdout | db.R1.D1 | 0.830 | 0.771 |

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
| Transfer run | 67 | 107,983 | 39,632 | $0.013 |
| Everything today (incl. smoke tests, aborted runs) | 639 | | | $0.232 |

Cached-token share reported by Crusoe in `usage.prompt_tokens_details`: 0.0027 (Lightning returns `null` for this field, so this undercounts; see the cache probe).
HOLDOUT final picks beat a plain logistic-regression baseline by 3.22 private-accuracy points on average (baseline 0.741), at $0.0255 of arena inference per point.
- Cache probe `nvidia/Nemotron-3.5-Lightning-30B-A3B`: median time-to-first-token 0.325 s with a shared ~4118-token prefix vs 0.408 s with unique prefixes; cached_tokens reported: shared [None, None, None, None, None, None], unique [None, None, None, None, None, None].
- Cache probe `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`: median time-to-first-token 0.536 s with a shared ~4118-token prefix vs 0.566 s with unique prefixes; cached_tokens reported: shared [3456, 3456, 3456, 3456, 3456, 3456], unique [0, 0, 0, 0, 0, 0].

### Pooled over split seeds (final selection, every dev competition x seed)

| Arm | n | Reported public | Private | Inflation (mean ± sd) |
|---|---|---|---|---|
| Naive (Kaggle board) | 6 | 0.808 | 0.780 | +0.028 ± 0.061 |
| HOLDOUT (Ladder + gate) | 6 | 0.810 | 0.777 | +0.033 ± 0.055 |
