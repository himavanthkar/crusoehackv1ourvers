# HOLDOUT: Shake-Up. Build brief for Claude Code

You are building a hackathon project TODAY at The Recursive Agents Hackathon (Crusoe x Cerebral Valley, Shack15, San Francisco, Saturday Oct 17 2026). Submissions close at 5:00 PM PT. Read this whole file before writing code. Ask the human only for things you cannot find or default, and batch every question into ONE message.

First action: if this file is not already named CLAUDE.md, rename it to CLAUDE.md so it persists as project memory.

---

## 1. The pitch

**One line:** Self-improving agents grade themselves on the scoreboard they optimize, so they learn the scoreboard, not the task. We built an AI Kaggle team, show the leaderboard shake-up when its "improvements" meet the private leaderboard, and fix it with a principled gate (the Ladder) so the team only keeps changes that are real.

**Track (pick exactly one on the form): Version N+1.** The brief asks for a team that reviews each run, rewrites its own prompts/playbooks/tools/structure, keeps a change only if it measurably helps, and proves it with a before/after on tasks it has not seen. Map every feature to those words in the README.

**Why it matters (context for README, paraphrase, do not copy text):**
- Karpathy's autoresearch (March 2026) popularized the loop: agent edits code, runs an experiment, checks a validation score, keeps or discards. Swarm forks share results through shared memory. That "keep it if validation went up" rule is exactly the Kaggle public-leaderboard mechanism.
- Sakana's Darwin Gödel Machine (2025) documented "objective hacking": a self-improving agent raised its score by removing the logging its hallucination checker relied on.
- Kaggle "shake-ups": teams overfit the small public leaderboard and drop when the private leaderboard is revealed.
- Blum and Hardt, "The Ladder: A Reliable Leaderboard for Machine Learning Competitions" (ICML 2015, arXiv 1502.04585): showed a boosting attack that climbs a public leaderboard with no real skill, and introduced the Ladder mechanism that resists adaptive overfitting. Nobody applies this to self-improving agent teams. We do.

**Research angle (for Q&A only, not the main pitch):** the builder's MS thesis studies the generation-verification gap in LLMs. Here the swarm out-generates its verifier (the public leaderboard). HOLDOUT closes that gap with a verifier the swarm cannot game.

---

## 2. Hackathon rules (hard, non-negotiable)

1. **Public GitHub repo.** Judges check commits to confirm work was done today. Commit after every milestone with clear messages. Never commit `.env` or keys (add to `.gitignore` in the first commit).
2. **New work only.** Everything in this repo is built today. If any utility is copied from the builder's other repo built today (`proof-of-done`), do it in one commit titled `import utilities from proof-of-done (also built today at this hackathon)` and list it in the README Provenance section.
3. **Demo must only highlight what was built today.** README needs a "Built today" section listing every component.
4. **Banned project types:** no Streamlit, no dashboard as the main feature, no basic RAG. Our UI is a `rich` terminal view of the agents working, plus matplotlib charts saved as PNG. The hero is agent behavior and the gate, never a dashboard.
5. **Must run on Crusoe Managed Inference** (OpenAI-compatible API). All LLM calls go through Crusoe.
6. **Data/code rights:** use only openly licensed datasets (OpenML datasets, sklearn built-ins). Record each dataset's name, OpenML id and license in the README.
7. **Submission (human does this, you prepare everything):** public repo URL, 1-minute demo video with voiceover showing only today's work, project description, Crusoe feedback, all teammates added.
8. **Judging:** Round 1 is a ~3 minute live demo + 1-2 min Q&A. Criteria: Technicality 40%, Live Demo 20%, Creativity 25%, Future Potential 15%. Top 6 go to stage with equal weights. The live demo must not depend on a flaky network: build a replay mode.
9. **Side prizes:** Best Use of Nemotron (use Nemotron for the main agent roles), Best Use of NemoClaw/OpenShell (optional stretch: run experiment code inside an OpenShell sandbox).

---

## 3. System design

### 3.1 Competitions (the task family)
- Pick 5 binary classification datasets from OpenML via `sklearn.datasets.fetch_openml(name=..., version=1, as_frame=True)`, 700 to 6000 rows, where a quick logistic regression gets roughly 65 to 90% accuracy (not saturated). Candidates: `credit-g`, `diabetes`, `blood-transfusion-service-center`, `phoneme`, `kc1`, `qsar-biodeg`, `ilpd`, `pc1`, `ozone-level-8hr`, `climate-model-simulation-crashes`, `spambase`. Check each quickly; fall back to sklearn built-ins if OpenML fails.
- 3 are **dev competitions** (the team learns on these). 2 are **unseen competitions** (opened only for the final transfer test).
- Each competition, per seed: stratified split into **train 50%**, **public leaderboard 100 rows (fixed n=100)**, **private leaderboard = all remaining rows**. The small public set is deliberate and realistic: it is what makes adaptive overfitting visible. State this openly in the README.
- Agents receive `X_train, y_train` and the unlabeled `X_test` (public and private rows shuffled together, like Kaggle). They submit predictions for all of `X_test`. Only the public subset is scored on the leaderboard. Private scores are computed and stored but never shown to any agent.
- Metric: accuracy (0/1 loss), matching the Ladder paper.

### 3.2 Leaderboard mechanisms (the core contribution)
Implement both in `holdout/leaderboard.py` with unit tests.

- **Kaggle mechanism (naive arm):** every submission gets its public accuracy reported, rounded to 5 decimals. The team sees every score.
- **Ladder mechanism (HOLDOUT arm):** implement the parameter-free Ladder from Blum and Hardt (arXiv 1502.04585). Read the paper's algorithm section and implement it exactly. Summary to verify against the paper: keep the per-example loss vector of the current best submission on the public set; for a new submission compute paired loss differences against the best; accept it as the new best only if its public loss is lower than the best's by more than the threshold (sample standard deviation of the paired differences divided by sqrt(n)); the reported score is the new (rounded) score if accepted, otherwise the previous best is reported again. Unit-test it: identical submissions are never accepted; a clearly better submission is accepted; random noise submissions are rarely accepted.

### 3.3 The agent team (identical in both arms)
- **Researchers (N=4 per round):** each is one research trajectory with a budget of up to 2 submissions per round. Each writes `solution.py` defining `fit_predict(X_train, y_train, X_test) -> np.ndarray` using scikit-learn (plus xgboost/lightgbm if installed). They may do internal cross-validation on train. They see: the competition card (rows, feature types, class balance), the shared playbook, the leaderboard history as reported by their arm's mechanism, and their own previous code and scores.
- **Diversity:** append a different direction hint to each researcher at the END of its prompt: feature engineering, model choice, regularization and tuning, ensembling.
- **Tool `ensemble(submission_ids)`:** majority-votes the test predictions of earlier submissions. Realistic Kaggle behavior, and the natural way a team overfits a public leaderboard (it is a soft version of the boosting attack). Both arms have it. Under Kaggle it gets rewarded by noise; under the Ladder it usually does not.
- **Curator (1 per round):** reads the round's results as reported by the arm's mechanism and rewrites the **playbook**: max 12 rules, max 1200 tokens, each rule with its evidence (submission ids and reported gains). This is the shared brain the next round reads.
  - **HOLDOUT arm admission gate (plain Python, not an LLM):** a rule may only cite submissions the Ladder accepted. Rules with no accepted evidence are rejected and logged.
  - **Naive arm:** the curator may cite any reported improvement.
- **Team structure change (N+1):** the curator may also propose one structural change per round (reassign direction hints, change submissions per researcher, add/remove a researcher). In the HOLDOUT arm it is kept only if the next round produces a Ladder-accepted improvement, otherwise reverted. Log every change.
- **Sandbox:** run each `solution.py` in a subprocess with a timeout (60 s), no network, in a temp dir, returning only predictions and a short log. Never let agent code touch labels of the test rows. Stretch: run inside NVIDIA OpenShell via NemoClaw.

### 3.4 Two arms, one difference
Both arms use the same models, prompts, researchers, rounds and budgets. **The only differences are the leaderboard mechanism and the curator's admission rule.** That clean causal story is the point.

- Rounds per competition: R=4. Researchers: N=4. Submissions: up to 2 per researcher per round (at most 32 per arm per competition).
- **Final selection:** each arm submits the submission with its best reported public score (naive: best Kaggle score; HOLDOUT: Ladder's current best).
- **Reveal:** compute private accuracy of each arm's final selection and of every submission.

### 3.5 Boosting-attack bot (calibration and guaranteed demo moment)
A non-LLM bot that reproduces the Blum and Hardt boosting attack: submit k random label vectors, keep those scoring above 50% on the public board, submit their majority vote. Run it against both mechanisms. Expected: under Kaggle the public score climbs well above 50% while private stays near 50%; under the Ladder it does not climb. This validates the implementation and gives a chart that is guaranteed to show the effect. Implement before the LLM arms.

### 3.6 Transfer test (the N+1 "unseen tasks" proof)
After the dev competitions, take three playbooks: **gen-0 (empty)**, **naive-final**, **HOLDOUT-final**. On each of the 2 unseen competitions, run one fresh round (N=4 researchers, same budget) under the Ladder mechanism for all three (fair comparison). Report private accuracy of each final selection. This answers: did the team actually learn something that transfers, and did the naive team learn noise?

### 3.7 Crusoe-native efficiency (MemoryAlloy)
Crusoe's inference engine uses MemoryAlloy, a cluster-wide KV cache that reuses repeated prompt prefixes. Design prompts so every researcher in a round shares a byte-identical prefix: `[system prompt + tool schemas + playbook + competition card + leaderboard snapshot at round start]`. Put researcher-specific content (id, direction hint, own history) at the END. Log `usage` from every response, including `prompt_tokens_details.cached_tokens` if Crusoe returns it. Report the cached-token share and cost. If cached tokens are not reported, measure time-to-first-token for prefix-sharing vs shuffled-prefix calls on a small sample and report that instead.

---

## 4. Models (Crusoe Managed Inference)

- Base URL: `https://api.inference.crusoecloud.com/v1` (OpenAI-compatible). Key in `.env` as `CRUSOE_API_KEY`.
- **Step 0:** call `GET /v1/models` and print the exact model ids. Model id strings differ between Crusoe docs and blogs, so never guess. Smoke-test each chosen model with one tiny call.
- **Researchers:** `nvidia/nemotron-3.5-lightning-30b-a3b` (fallback: `nvidia/Nemotron-3-Nano-30B-A3B`). Cheap and fast.
- **Curator:** Nemotron 3 Ultra (exact id from `/v1/models`). This matches Crusoe's recommended tiering (Ultra orchestrates, small Nemotron runs subagents). A final-round judge wrote that tiering post, so call it out.
- Using Nemotron for every agent role keeps us eligible for Best Use of Nemotron.
- **Rate limits and budget:** the builder runs a second Claude Code session on another project with the SAME Crusoe key. Cap concurrency at 6 requests, use exponential backoff on 429s, and cache every LLM response on disk keyed by a hash of (model, messages, temperature). Use temperature 0.7 for researchers (diversity) and 0.2 for the curator. Print running token totals and estimated cost. Stop and tell the human if estimated spend passes $15.

---

## 5. Repo layout

```
holdout-shakeup/
  CLAUDE.md
  README.md
  .env.example            # CRUSOE_API_KEY=, CRUSOE_BASE_URL=https://api.inference.crusoecloud.com/v1
  .gitignore              # .env, cache/, __pycache__, runs/*/raw (keep summaries)
  requirements.txt
  holdout/
    config.py             # models, budgets, seeds, sizes
    llm.py                # Crusoe client: retry/backoff, disk cache, usage + cached-token logging
    data.py               # competitions, splits, licenses
    leaderboard.py        # Kaggle + Ladder mechanisms
    sandbox.py            # subprocess runner with timeout, no network
    researcher.py         # one trajectory
    curator.py            # playbook rewrite + structure change proposals
    gate.py               # admission rules (plain Python)
    arena.py              # rounds, arms, final selection, reveal
    boosting_bot.py       # Blum and Hardt attack
    transfer.py           # unseen-competition test
    report.py             # results.json, charts, tables
  demo.py                 # rich terminal: live/replay leaderboard, REVEAL PRIVATE, charts
  tests/                  # leaderboard, gate, sandbox tests (offline)
  runs/                   # recorded runs for replay
  results/                # results.json, PNG charts
```

---

## 6. Milestones and deadlines (PT)

| By | Milestone | Done when |
|---|---|---|
| 2:50 | **M0 setup** | repo + `.gitignore` committed; models listed and smoke-tested; 5 datasets load; splits built |
| 3:15 | **M1 core** | Kaggle + Ladder implemented with passing tests; boosting bot chart produced; sandbox runs a baseline `fit_predict` |
| 3:45 | **M2 arms** | both arms run end to end on all 3 dev competitions for 1 seed; final selection + private reveal; `results.json` |
| 4:00 | **M3 proof** | transfer test on 2 unseen competitions; charts; `demo.py --replay`; README with real numbers |
| 4:10 | **Freeze** | final push; repo public; replay tested from a clean clone |
| 4:10 to 4:45 | Human | record 1-minute video with voiceover; submit by 4:50 |

**If behind, cut in this order (never cut the gate or the honesty rules):** extra seeds, then structure changes, then You.com, then OpenShell, then transfer to 1 unseen competition. The boosting-bot chart plus one dev competition with both arms is the minimum viable proof.

**If ahead (full version):** 3 seeds per competition with mean and spread; You.com search tool for researchers with a contamination guard (block queries containing the dataset name, "kaggle", "openml"; log every block); OpenShell sandbox; Thresholdout (Dwork et al. 2015) as a second gate for comparison; more rounds to show the naive arm's public score keep climbing while private flattens.

---

## 7. Metrics and charts (all computed from logged data, never typed by hand)

1. **Shake-up table per arm:** final selection's reported public score, true public score, private score, and inflation (public minus private).
2. **Rank shake-up:** top 10 submissions by reported public with their private rank and an up/down arrow; Spearman correlation of public vs private across all submissions per arm.
3. **Boosting-bot chart:** public vs private accuracy over submissions, Kaggle vs Ladder.
4. **Per-round chart per arm:** best reported public vs private of that best, across rounds.
5. **Transfer bars:** private accuracy on unseen competitions with gen-0, naive-final, HOLDOUT-final playbooks.
6. **Gate log:** rules proposed vs admitted vs rejected in each arm, with reasons.
7. **Efficiency:** total tokens, cached-token share (or TTFT comparison), estimated $, $ per point of private improvement.

---

## 8. Demo assets

**`demo.py --replay runs/<id>`** plays a recorded run: a live `rich` leaderboard that fills as researchers submit, gate verdict lines (`LADDER: rejected, gain within noise` / `KAGGLE: accepted`), curator playbook diffs, then a keypress `REVEAL PRIVATE` that re-sorts the board with arrows (the shake-up), then opens the charts. **`demo.py --live --rounds 1`** runs one small live round to prove it is real.

**3-minute live pitch outline:**
- 0:00 "Self-improving agents keep a change when their score goes up. That's how Kaggle teams overfit the public leaderboard."
- 0:20 Boosting-bot chart: random guesses climb the public board under the normal mechanism; the Ladder holds.
- 0:45 Replay: the AI Kaggle team works, the curator rewrites the playbook, the gate rejects a "gain" that is noise.
- 1:45 REVEAL PRIVATE: naive arm shakes down, HOLDOUT holds.
- 2:15 Transfer: playbooks on competitions never seen.
- 2:40 Crusoe cache share and cost; Nemotron tiering. Close: "Everyone builds the optimizer. We built the thing that catches it lying."

**1-minute video script** (write to `video_script.md`): 10 s problem, 15 s system, 20 s shake-up reveal, 10 s transfer result, 5 s close. Only show today's work.

---

## 9. README must include

Pitch line; "Built today at The Recursive Agents Hackathon" with every component listed; Provenance (anything imported from proof-of-done, also built today); architecture diagram (Mermaid); how each feature maps to the Version N+1 brief; results tables and charts from `results.json`; how to run (install, `.env`, run, replay); sponsor usage (Crusoe Managed Inference for every call, Nemotron roles, MemoryAlloy prefix design, OpenShell/You.com if done); datasets and licenses; honest limitations (small public n is deliberate, few seeds, LLM nondeterminism, accuracy only, transfer across a handful of datasets).

---

## 10. Honesty rules

- Never invent or round up a number. Every figure in the README comes from `results.json`.
- If the naive arm does not shake down on some competition, report it. The boosting-bot chart still shows the mechanism, and mixed results are fine if explained.
- Keep the private leaderboard and unseen competitions sealed until final evaluation. Add a test that fails if any agent prompt contains private labels or unseen-competition data.

---

## 11. Likely judge questions (put short answers in README FAQ)

- **"Isn't this just autoresearch?"** Autoresearch keeps a change when validation improves, which is the Kaggle mechanism. We show that rule fools an agent team, and replace it with a mechanism the team cannot game.
- **"Why not just use a bigger validation set?"** It helps, but adaptive reuse still overfits, and real agents usually have small evals. The Ladder gives guarantees under adaptive reuse.
- **"Does the gate slow real progress?"** Compare private scores of both arms and the transfer result.
- **"Where is the N+1?"** Every round the team rewrites its playbook and structure; only gated changes survive; the evolved playbook is tested on unseen competitions.
- **"Why Crusoe?"** Shared-prefix design rides MemoryAlloy's cluster-wide KV cache; Nemotron Ultra orchestrates and small Nemotron researchers work, per Crusoe's tiering.

---

## 12. Things only the human can do (ask once, batched)

1. Crusoe API key (or the path to the proof-of-done `.env` to reuse it).
2. Create a public GitHub repo `holdout-shakeup` and give the remote URL.
3. Optional: You.com API key.
4. Record the video and submit the form.
