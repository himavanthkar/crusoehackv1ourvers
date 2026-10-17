# 1-minute video script (voiceover + screen)

Record: `python demo.py --intro --replay runs/m2 --comp diabetes --speed 2` in a full-screen terminal,
then the transfer chart. Numbers below are from `results/results.md` (seed 0 arena, 2-seed transfer).

| Time | Screen | Voiceover |
|---|---|---|
| 0:00-0:10 | Title card from `demo.py --intro` | "Self-improving agents keep a change when their score goes up. That's how Kaggle teams overfit the public leaderboard, and how agent teams learn the scoreboard instead of the task." |
| 0:10-0:25 | Boosting chart (opens automatically), then the replay starts: two teams, leaderboards filling | "We built an AI Kaggle team on Crusoe, all Nemotron: four researchers write code in a sandbox, a curator rewrites the team playbook every round. Two identical teams, one difference. The naive team keeps any gain. HOLDOUT scores against the Ladder, which only releases a significant gain. Random guessing climbs the normal board to 0.87 with zero skill. The Ladder holds it to 0.64." |
| 0:25-0:45 | Ticker: `LADDER: rejected, gain within noise`, `GATE rejected`; press Enter, REVEAL PRIVATE re-sorts with arrows | "Watch the gate: rules backed by noise get rejected. Across two seeds the naive curator admitted 135 of 135 rules; HOLDOUT admitted 43 of 76. Now the reveal: on diabetes the naive pick drops from 0.87 public to 0.75 private." |
| 0:45-0:55 | `results/transfer.png` | "Then we test the evolved playbooks on competitions the team never saw. The naive playbook's code ran 2 times out of 32. HOLDOUT's ran 16 of 32 and beat the empty playbook on blood transfusion, 0.77 to 0.70." |
| 0:55-1:00 | Results panel | "Everyone builds the optimizer. We built the thing that catches it lying. HOLDOUT, on Crusoe." |

Honesty note for Q&A: the final-pick shake-up between arms was small and mixed at our N (pooled inflation +0.028
naive vs +0.033 HOLDOUT). The robust effects are the boosting calibration, the gate, and transfer. Say so if asked.

---

# 3-minute live pitch (Round 1)

1. **0:00** "Self-improving agents keep a change when their score goes up. That is exactly the Kaggle public-leaderboard rule, and Kaggle teams famously overfit it." (`python demo.py --intro --replay runs/m2 --comp diabetes`)
2. **0:20** Boosting chart: random label vectors, zero skill. Kaggle board reports 0.871 while private is 0.516; the Ladder (Blum and Hardt 2015) reports 0.636. "This is our calibration: it proves the boards are implemented right."
3. **0:45** Replay: four Nemotron Lightning researchers, a Nemotron Super curator, sandboxed code. Point at `LADDER: rejected, gain within noise` and `GATE rejected`. "The gate is plain Python: a rule can only cite submissions the Ladder accepted."
4. **1:45** Press Enter, REVEAL PRIVATE. Arrows show the re-sort. Be honest: "On diabetes the naive pick drops 0.87 to 0.75; on phoneme it doesn't. Pooled, the final-pick difference is not significant at our N."
5. **2:15** Transfer chart: "What did each team learn? On unseen competitions, naive-final code ran 2/32 times, HOLDOUT-final 16/32, gen-0 9/32."
6. **2:40** Crusoe: "$0.08 for the whole two-arm run. Shared-prefix prompts: Nemotron Super reported 3,456 of 4,118 prompt tokens served from MemoryAlloy's cache." Close: "Everyone builds the optimizer. We built the thing that catches it lying."

Live fallback: `python demo.py --live --rounds 1 --comp diabetes` runs one real round on Crusoe (about 1 minute).
If the Wi-Fi is bad, everything above replays offline from `runs/`.
