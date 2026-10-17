# Submission form answers (copy and paste)

Every number below is from `results/results.json` (or holdout-code's results where marked).

## Team Name
HOLDOUT

## Team Members
Himavanth Karpurapu (if "hima vanth" and "himavanth" are the same person, list them once; add any other teammates)

## Project Description
Self-improving agents grade themselves on the scoreboard they optimize, so they learn the scoreboard, not the task. HOLDOUT is an AI Kaggle team on Crusoe Managed Inference, all NVIDIA Nemotron: Nemotron Lightning researchers write and submit scikit-learn solutions from a sandbox, and a Nemotron Super curator rewrites the team's shared playbook (and team structure) after every round. Two identical teams differ in one thing. The naive team keeps any change whose public score went up (the autoresearch / Kaggle rule). The HOLDOUT team scores against the Ladder leaderboard (Blum and Hardt, ICML 2015), which only releases a significant gain, and a plain-Python gate admits a playbook rule only if it cites a Ladder-accepted submission.

Results: against a zero-skill boosting attack, random guesses reach 0.871 on a normal public board but 0.636 on the Ladder (private about 0.52 for both). Across 2 seeds, the naive curator admitted 135/135 proposed rules; HOLDOUT admitted 43/76. On two datasets the team never saw, code written from the naive playbook ran 2/32 times, HOLDOUT's 16/32, and on blood-transfusion HOLDOUT's playbook beat the starting playbook on both seeds while naive's produced nothing that ran. Honest limit: the final-pick shake-up between teams was not measurably different at our sample size. Agent code runs in a sandbox, including a recorded round inside NVIDIA OpenShell. Version N+1 track.

## Public GitHub Repository
https://github.com/himavanthkar/crusoehackv1ourvers

## Public Demo Video
Upload your voiced recording (YouTube unlisted, Loom, or Google Drive set to "anyone with the link") and paste that link.
Already public (57 s, narrated with the macOS Samantha voice):
https://github.com/himavanthkar/crusoehackv1ourvers/blob/main/video/holdout_demo_v2.mp4

## Which problem statement are you submitting to?
Version N+1

## Are you submitting for NVIDIA's Best Use of Nemotron?
Yes

## Are you submitting for NVIDIA's Best Use of NemoClaw & OpenShell?
Yes (OpenShell; we did not use NemoClaw)

## Did you use NVIDIA tools for your project?
Yes. NVIDIA Nemotron for every agent role, served by Crusoe: researchers on `nvidia/Nemotron-3.5-Lightning-30B-A3B`, curator on `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`, with `nvidia/NVIDIA-Nemotron-3.5-Super-VL` as a rate-limit fallback. NVIDIA OpenShell as a sandbox backend (`SANDBOX=openshell`): agent-written code runs in a micro-VM as an unprivileged user with deny-by-default network. A recorded round (`runs/openshell_r1`) ran all 16 agent solutions inside OpenShell (10 ran successfully, mean 2.35 s per execution); the live demo can run in OpenShell too. The main experiment runs used our local sandbox.

## Did you use You.com tools for your project?
No

## Describe your use of Crusoe Managed Inference to build the submitted project
Every LLM call goes through Crusoe Managed Inference via the OpenAI-compatible endpoint (`https://api.inference.crusoecloud.com/v1`) with the stock `openai` SDK. We read exact model ids and pricing from `/v1/models` instead of trusting docs, and used Crusoe's recommended tiering: a larger Nemotron (Super 120B) orchestrates as curator, a small fast Nemotron (Lightning 30B-A3B) runs four parallel researchers. Prompts are laid out for MemoryAlloy: every researcher in a round sends a byte-identical prefix (system prompt, tool schema, playbook, competition card, leaderboard snapshot) with researcher-specific text last. A probe showed the effect: with a shared ~4.1k-token prefix, Nemotron Super reported 3,456 of 4,118 prompt tokens cached (0 with unique prefixes), and Lightning's median time-to-first-token was 0.325 s vs 0.408 s. Our client adds concurrency caps, adaptive pacing and backoff on 429s, a disk cache for exact replays, and per-call logging of tokens, cached tokens and cost. A full two-arm experiment (202 live calls) cost $0.082; everything we ran today, including smoke tests and aborted runs, was 641 live calls, 1,633,332 tokens, about $0.23.

## Provide feedback from your experience with using Crusoe products
- The OpenAI-compatible endpoint worked with the stock `openai` SDK on the first try, and `/v1/models` returning pricing (including the cached-read price) made cost tracking easy.
- Model ids differ between docs, blogs and the API (for example `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` vs `nvidia/Nemotron-3.5-Lightning-30B-A3B` casing and prefixes), and Nemotron 3 Ultra appeared in pricing but was not live on our project. A stable alias per model family would help.
- We hit `429 max_requests_limit (orchestrator)` at modest load. After moving to a separate key, Nemotron Super still returned 429 while Lightning and Super-VL answered, so the limit looks endpoint-wide. Rate-limit headers or a `Retry-After` value would let clients pace correctly instead of guessing.
- `usage.prompt_tokens_details.cached_tokens` came back `null` for Nemotron Lightning and populated for Super. Reporting it consistently would make MemoryAlloy savings visible.
- Prefix caching clearly works (see the probe above); a short doc on how to structure prompts for it would help more teams use it.

## Provide feedback from your experience of using partner products to build the submitted project
- **NVIDIA Nemotron:** Lightning is fast and very cheap, good for many parallel researchers, but as a coder it often produced code that failed in the sandbox (in our transfer test only 27 of 96 researcher submissions ran), so robust sandboxing and logging were essential. Super was reliable at structured JSON for the curator role. The `enable_thinking` toggle via `chat_template_kwargs` made latency predictable.
- **NVIDIA OpenShell:** creating a sandbox, uploading files and running `sandbox exec` from the CLI was straightforward, and the deny-by-default network plus unprivileged user is exactly what agent-written code needs. Because the sandbox has no network, we vendored pinned Linux wheels and installed them offline; the slim Python image lacks `libgomp`, so LightGBM needed the copy bundled with scikit-learn. A documented recipe for offline Python ML dependencies, or an image with them preinstalled, would save time. Execution averaged 2.35 s per solution.
- **You.com:** not used.
