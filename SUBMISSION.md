# Submission form helper (copy and paste)

**Project name:** HOLDOUT: Shake-Up

**Track:** Version N+1

**One-liner:** Self-improving agents grade themselves on the scoreboard they optimize, so they learn the scoreboard, not
the task. HOLDOUT is an AI Kaggle team on Crusoe (all Nemotron) that rewrites its own playbook and team structure every
round, and keeps a change only if the Ladder leaderboard (Blum and Hardt 2015) says the gain is real.

**Description (short):**
We built an AI Kaggle team on Crusoe Managed Inference: four Nemotron Lightning researchers write and submit
scikit-learn solutions in a sandbox, and a Nemotron Super curator rewrites the team playbook and structure after every
round. We run two identical teams that differ in one thing. The naive team keeps any change whose public score went up
(the autoresearch / Kaggle rule). The HOLDOUT team scores against the Ladder, which releases a new score only on a
statistically significant gain, and a plain-Python gate admits a playbook rule only if it cites a Ladder-accepted
submission. A non-LLM boosting-attack bot calibrates both boards: random guesses climb the Kaggle board but not the
Ladder. Then we reveal the sealed private leaderboard (the shake-up), and test the evolved playbooks on two competitions
the team never saw. All results are generated from logs; the terminal demo replays a recorded run offline and can run
one round live.

**Built today:** every file in the repo (see README "What we built" and the commit history).

**Sponsor tech:** Crusoe Managed Inference for every LLM call; NVIDIA Nemotron for every agent role (Lightning 30B-A3B
researchers, Super 120B curator); prompts laid out for MemoryAlloy prefix reuse; NVIDIA OpenShell sandbox for
agent-written code (`SANDBOX=openshell`, recorded round in `runs/openshell_r1`).

**Crusoe feedback (draft, edit freely):**
- The OpenAI-compatible endpoint worked with the stock `openai` SDK; `/v1/models` returning pricing (including the
  cached-read price) made cost tracking easy.
- Model ids differ from the docs and blogs (e.g. `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` vs `nvidia/Nemotron-3.5-Lightning-30B-A3B`
  casing and prefixes), and Nemotron 3 Ultra was listed on the pricing page but not live on our project. A stable alias
  per model family would help.
- We hit `429 max_requests_limit (orchestrator)` at modest load (under 30 calls/min). After moving this project to its
  own key, Nemotron Super still returned 429 while Lightning and Super-VL answered, so the limit looks endpoint-wide.
  Returning `x-ratelimit-*` headers or a `Retry-After` would let clients pace correctly instead of guessing.
- `usage.prompt_tokens_details.cached_tokens` came back `null` for Nemotron Lightning and present for Super. Reporting it
  consistently would make MemoryAlloy savings visible to users.

**Teammates:** (add all teammates on the form)
