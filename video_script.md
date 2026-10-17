# 1-minute video: rendered video + timed voiceover

**Video:** `video/holdout_demo.mp4` (1920x1080, 60.0 s, silent). Rebuild with `python video/make_video.py`.
Every frame is a real render of `demo.py` from recorded runs (`runs/openshell_r1` for the replay) and `results/`.

## How to record your voice

Easiest: open `video/holdout_demo.mp4` in QuickTime, start **File > New Screen Recording** with your microphone on,
play the video full screen, and read the lines below at the cue times. Or record audio only (Voice Memos / QuickTime
audio), then merge:

```bash
ffmpeg -i video/holdout_demo.mp4 -i voice.m4a -map 0:v -map 1:a -c:v copy -c:a aac -shortest video/holdout_final.mp4
```

Pace: about 2.5 words per second. Each block fits its scene with a little slack; if you run long, drop the words in [brackets].

## Script (read at these times)

| Time | On screen | Say this |
|---|---|---|
| **0:00 – 0:10** | Title card | "Self-improving agents keep a change when their score goes up. That's exactly how Kaggle teams overfit the public leaderboard. We built HOLDOUT to catch it." |
| **0:10 – 0:17** | Boosting-attack chart | "Calibration: random guesses reach 0.87 on a normal leaderboard. The Ladder holds them to 0.64." |
| **0:17 – 0:34** | Two leaderboards filling, gate verdicts in the ticker | "Now a real AI Kaggle team on Crusoe, all Nemotron. Researchers write code that runs inside NVIDIA OpenShell. Two identical teams, one difference: the naive team trusts every score. HOLDOUT's Ladder rejects gains within noise, [and its gate rejects playbook rules built on them.]" |
| **0:34 – 0:40** | REVEAL PRIVATE, boards re-sort with arrows | "Private reveal: naive's pick drops to 0.74. HOLDOUT's ranks first, at 0.77." |
| **0:40 – 0:52** | Transfer chart | "Then the real test: playbooks on competitions they've never seen. Code from the naive playbook ran 2 times out of 32. HOLDOUT's ran 16 out of 32, [and won on blood transfusion.]" |
| **0:52 – 1:00** | Closing card | "The whole two-arm run cost eight cents on Crusoe. Everyone builds the optimizer. We built the thing that catches it lying." |

Where the numbers come from: boosting 0.871 / 0.636 (`results/boosting_bot.json`); reveal 0.890 to 0.739 and
0.830 to 0.771, HOLDOUT's pick private rank 1 (`runs/openshell_r1`, one recorded round); transfer 2/32, 16/32 and
blood-transfusion 0.768 vs gen-0 0.697 (`runs/transfer`); cost $0.08 (`results/results.json`).

Honesty note for Q&A: the reveal is one recorded round. Across all runs the final-pick difference between arms was
small and not significant (README, finding 4). The robust effects are the boosting calibration, the gate, and transfer.

---

# 3-minute live pitch (Round 1)

1. **0:00** "Self-improving agents keep a change when their score goes up. That is exactly the Kaggle public-leaderboard rule, and Kaggle teams famously overfit it." (`python demo.py --intro --replay runs/m2 --comp diabetes`)
2. **0:20** Boosting chart: random label vectors, zero skill. Kaggle board reports 0.871 while private is 0.516; the Ladder (Blum and Hardt 2015) reports 0.636. "This is our calibration: it proves the boards are implemented right."
3. **0:45** Replay: four Nemotron Lightning researchers, a Nemotron Super curator, sandboxed code. Point at `LADDER: rejected, gain within noise` and `GATE rejected`. "The gate is plain Python: a rule can only cite submissions the Ladder accepted."
4. **1:45** Press Enter, REVEAL PRIVATE. Arrows show the re-sort. Be honest: "On diabetes the naive pick drops 0.87 to 0.75; on phoneme it doesn't. Pooled, the final-pick difference is not significant at our N."
5. **2:15** Transfer chart: "What did each team learn? On unseen competitions, naive-final code ran 2/32 times, HOLDOUT-final 16/32, gen-0 9/32."
6. **2:40** Crusoe: "$0.08 for the whole two-arm run. Shared-prefix prompts: Nemotron Super reported 3,456 of 4,118 prompt tokens served from MemoryAlloy's cache." Close: "Everyone builds the optimizer. We built the thing that catches it lying."

Live fallback: `python demo.py --live --rounds 1 --comp diabetes` runs one real round on Crusoe (about 30 seconds);
prefix it with `SANDBOX=openshell` to run the agents' code inside NVIDIA OpenShell.
If the Wi-Fi is bad, everything above replays offline from `runs/`.
