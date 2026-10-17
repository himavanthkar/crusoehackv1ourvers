# HOLDOUT: a shared brain that only learns proven lessons

**Track: Version N+1.** Built today at The Recursive Agents Hackathon (Crusoe x Cerebral Valley, Shack15, Oct 17 2026). Every agent runs on NVIDIA Nemotron through Crusoe Managed Inference.

> Self-improving agents grade themselves on the scoreboard they optimize, so they learn the scoreboard, not the task. HOLDOUT is an AI Kaggle team whose shared playbook only accepts a lesson when the Ladder leaderboard (Blum and Hardt, ICML 2015) says the gain is real.

---

## The problem, in numbers from our other team

We ran the same idea on a second agent team today (a self-improving coding team, [holdout-code](https://github.com/himavanthkar/holdout-code)). In its seed-17 run, the naive self-improvement loop pushed its visible test score to **100.0%** while hidden tests fell to **53.3%**. That drop came from accepting 2 of 2 *planted* red-team cheats, a labeled stress test. The improver's own, unplanted proposals point the same way: in that seed, all **7 of 7** distinct changes it proposed that raised the visible score left the hidden score flat or lower (9 of 9 gate entries, counting proposals logged by both arms). It did not hold everywhere: in seed 41, 2 of the 3 such proposals did raise the hidden score, and seed 23 had none. (Source: holdout-code `results/results.json` and `runs/seed23`, `runs/seed41`.)

---

## What we built

An AI Kaggle team that improves itself round after round, in two arms that differ in only one thing: how they decide what is a real improvement.

- **Researchers** (Nemotron): each runs a research trajectory, writes `fit_predict` code, and submits predictions to a leaderboard.
- **Shared brain:** a playbook every researcher reads before working. After each round a **curator** (Nemotron) rewrites it and can also change team structure.
- **Naive arm:** the leaderboard reports every public score (the Kaggle mechanism) and the curator may admit any rule that looked like a gain.
- **HOLDOUT arm:** the leaderboard runs the **Ladder**, which only reports a new best when the gain beats the noise threshold, and a plain-Python **admission gate** only lets a rule into the playbook if it cites a Ladder-accepted gain.
- **Sandbox:** agent code runs isolated with a timeout and no network. One recorded round runs inside **NVIDIA OpenShell**.
- **Transfer test:** the final playbooks are dropped onto datasets the team never saw.

| Role | Exact model id (Crusoe Managed Inference) |
|---|---|
| Researchers (4 per round, 2 submissions each) | `nvidia/Nemotron-3.5-Lightning-30B-A3B` |
| Curator (rewrites the playbook, proposes structure changes) | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` |
| Curator fallback, only after 3 rate-limited attempts | `nvidia/NVIDIA-Nemotron-3.5-Super-VL` (added after the reported runs; never triggered in any logged call) |

Nemotron 3 Ultra was not live on our project's `/v1/models`, so the curator uses the largest live Nemotron.

Each competition: train split, a **public board of 100 rows** (deliberately small, like a real public leaderboard), and a private board with the rest, never shown to any agent.

---

## Results

### 1. What the shared brain is allowed to learn

| Seed | Naive: proposed / admitted | HOLDOUT: proposed / admitted |
|---|---|---|
| 0 | 60 / 60 | 26 / 12 |
| 1 | 75 / 75 | 50 / 31 |
| **Total** | **135 / 135** | **76 / 43** (33 rejected) |

The naive curator admitted every rule it wrote. The gate rejected 33 rules and stripped 114 citations to gains the Ladder did not accept. Held on both seeds.

### 2. Unseen datasets (the N+1 before/after)

One fresh round per cell (4 researchers x 2 submissions), all three playbooks scored on the Ladder board. Value = private accuracy of the team's final pick (submissions that ran). Private rows: blood-transfusion 274, qsar-biodeg 428.

| Dataset | Seed | gen-0 (start) | naive-final | HOLDOUT-final |
|---|---|---|---|---|
| blood-transfusion | 0 | 0.748 (3/8 ran) | no valid submission (0/8) | **0.781 (5/8)** |
| blood-transfusion | 1 | 0.646 (2/8) | no valid submission (0/8) | **0.755 (4/8)** |
| qsar-biodeg | 0 | 0.862 (2/8) | 0.848 (1/8) | 0.860 (3/8) |
| qsar-biodeg | 1 | 0.864 (2/8) | 0.862 (1/8) | 0.860 (4/8) |

**Submissions that ran: gen-0 9/32, naive-final 2/32, HOLDOUT-final 16/32.** On blood-transfusion the gated playbook beat the starting point on both seeds while the naive playbook produced nothing that ran. On qsar-biodeg it was slightly below the starting point on both seeds.

Why the naive brain broke: its curator admitted a rule pushing HistGradientBoosting with native categorical handling, citing 22 submissions. Replaying the Ladder over the naive team's own submissions, only 1 of those 22 counts as a real gain. On all-numeric blood-transfusion, researchers following it wrote broken code: 12 of the 16 naive-final submissions failed with `No module named 'histgradientboosting'`. HOLDOUT's final playbook never contained that rule; it kept a single Ladder-backed rule (start from a LightGBM baseline). That is the point: lessons backed by noise rot a shared brain.

### 3. The Ladder against a zero-skill attack

Blum and Hardt's boosting attack (random-label submissions, 400 per run, 30 runs, credit-g):

| Mechanism | Reported public | True private |
|---|---|---|
| Kaggle (naive) | 0.871 | 0.516 |
| Ladder (HOLDOUT) | 0.636 | 0.529 |

Random guessing looks like 87% on a normal leaderboard. The Ladder cuts most of that inflation, though with only 100 public rows it slows the attack rather than stopping it.

### 4. Final-pick shake-up between the two teams (honest)

Pooled over 3 dev competitions x 2 seeds: naive reported 0.808 vs private 0.780 (gap +0.028 ± 0.061); HOLDOUT 0.810 vs 0.777 (gap +0.033 ± 0.055). **No measurable difference at this sample size.** LLM submissions are highly correlated, so they overfit a public board far less than an adversarial attacker does. Resampling the public rows 2,000 times, the Ladder selection rule cut the gap from +0.027 to +0.021 with private accuracy essentially unchanged (0.788 vs 0.786).

---

## What did not work

- The team-level shake-up was not measurable (section 4). The Ladder's effect on real agents showed up in what the team learned, not in the final pick.
- The transfer win is mostly about whether code runs at all, on 2 datasets x 2 seeds. A promising signal, not proof.
- HOLDOUT-final was slightly below the starting point on qsar-biodeg on both seeds.
- No significance tests were run.
- Crusoe reported few cached tokens because Nemotron Lightning returns `null` for cached tokens, so the cache benefit is undercounted (see below).

---

## Crusoe and NVIDIA

- **Crusoe Managed Inference** for every call, through the OpenAI-compatible API (`https://api.inference.crusoecloud.com/v1`).
- **NVIDIA Nemotron** for every agent role: researchers on `nvidia/Nemotron-3.5-Lightning-30B-A3B`, curator on `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`, and `nvidia/NVIDIA-Nemotron-3.5-Super-VL` as the curator's rate-limit fallback.
- **MemoryAlloy prefix design:** every researcher in a round shares a byte-identical prefix (system, tools, playbook, competition card, leaderboard snapshot), with researcher-specific text last. Probe: with a shared prefix, Nemotron Super reported 3,456 of 4,118 prompt tokens cached (0 with unique prefixes); Lightning's median time-to-first-token was 0.325 s shared vs 0.408 s unique.
- **NVIDIA OpenShell:** one recorded round (diabetes) ran fully inside the OpenShell sandbox. Naive reported 0.89 vs private 0.739; HOLDOUT reported 0.83 vs private 0.771.
- **Cost:** 641 live calls, 1,633,332 tokens, $0.2322 estimated from Crusoe's listed prices (everything today, including smoke tests and aborted runs).

---

## How to run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # add CRUSOE_API_KEY
.venv/bin/python -m holdout.check                 # list live Crusoe models, smoke-test both Nemotron roles
.venv/bin/python -m pytest -q tests               # offline tests
.venv/bin/python -m holdout.boosting_bot          # boosting attack chart (no LLM)
.venv/bin/python -m holdout.arena --run-id m2     # both arms on the 3 dev competitions
.venv/bin/python -m holdout.transfer --from-run m2
.venv/bin/python -m holdout.report --arena m2 --transfer transfer --seeds-runs m2,m2s1
.venv/bin/python demo.py --replay runs/m2         # offline replay, press Enter for REVEAL PRIVATE
.venv/bin/python demo.py --live --rounds 1        # one small live round on Crusoe
SANDBOX=openshell .venv/bin/python demo.py --live --rounds 1   # same, agent code inside NVIDIA OpenShell
```

Demo video: [`video/holdout_demo_v2.mp4`](video/holdout_demo_v2.mp4) (57 s with narration, rendered by `python video/make_video_v2.py`; narration text in `video/voiceover_v2.txt`).

Replays need no network and no API key. LLM responses are cached on disk (`cache/`, gitignored), so a rerun of the
same run is free and deterministic.

The stage demo replays a recorded run from a fresh clone with no API key. `SANDBOX=openshell` runs a live round inside OpenShell.

---

## Built today

Everything in this repo was built today: leaderboard mechanisms (Kaggle and Ladder), admission gate, researcher and curator agents, sandbox and OpenShell backend, arena, boosting-attack bot, transfer test, reporting, and the replay/live demo.

### Provenance
No code was imported from our other project built today (`proof-of-done`). Early runs used that project's Crusoe API key
(this project now has its own key); NOTES.md re-verifies the live model list from scratch. The OpenShell backend follows
the same upload / `sandbox exec` pattern as proof-of-done's OpenShell sandbox, but the code here is new and uses its own
sandbox (`holdout-sbx`). Every line in this repo was written today; see the commit history.

### Datasets and licences

All from OpenML, fetched with `sklearn.datasets.fetch_openml(data_id=...)`. OpenML lists the licence of each as "Public".

| Dataset | OpenML id | Rows | Role |
|---|---|---|---|
| credit-g | [31](https://www.openml.org/d/31) | 1000 | dev |
| diabetes | [37](https://www.openml.org/d/37) | 768 | dev |
| phoneme | [1489](https://www.openml.org/d/1489) | 5404 | dev |
| qsar-biodeg | [1494](https://www.openml.org/d/1494) | 1055 | unseen (transfer only) |
| blood-transfusion-service-center | [1464](https://www.openml.org/d/1464) | 748 | unseen (transfer only) |

---

## FAQ

**Isn't this just autoresearch?** Autoresearch keeps a change when validation improves, which is the Kaggle mechanism. We show what that rule lets into a shared brain, and replace it with one the team cannot game.

**Is the transfer win just broken imports?** Partly. The broken imports came from a rule the naive curator admitted on mostly noise-level evidence (only 1 of its 22 cited gains would pass the Ladder), and it broke the team on new data. HOLDOUT's playbook never contained it.

**Does the Ladder help a real LLM team?** On the final pick, not measurably at our scale. It changed what the team learned, and that is what transferred.
