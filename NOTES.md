# NOTES (Step 0 and M0 findings, Oct 17 2026)

## Crusoe Managed Inference
- Base URL `https://api.inference.crusoecloud.com/v1`, OpenAI-compatible, `Authorization: Bearer $CRUSOE_API_KEY`.
- `GET /v1/models` (14:43 PT) returned 13 live ids:
  `Qwen/Qwen3.8-27B`, `deepseek-ai/DeepSeek-V4-Pro`, `deepseek-ai/DeepSeek-V4.1-Flash`, `deepseek-ai/Deepseek-V4-Flash`,
  `google/gemma-4-31b-it`, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`,
  `nvidia/NVIDIA-Nemotron-3.5-Super-VL`, `nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B`, `nvidia/Nemotron-3.5-Lightning-30B-A3B`,
  `openai/gpt-oss-120b`, `zai-org/GLM-5.3`, `zai-org/GLM-5.3-Flash`.
- **Nemotron 3 Ultra is not live on this project.** Curator falls back to the largest live Nemotron:
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` ($0.30 in / $2.40 out / $0.15 cached-read per 1M).
  Researchers: `nvidia/Nemotron-3.5-Lightning-30B-A3B` ($0.05 / $0.20 / $0.03). Same Ultra-orchestrates /
  small-Nemotron-works tiering, one step down.
- Smoke test: both answered a JSON ping (Lightning 0.22 s, Super 0.67 s). Thinking disabled with
  `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`.
- `usage.prompt_tokens_details.cached_tokens`: Super returns it (0 on a cold prompt); Lightning returned `None` on the ping.
  We log whatever comes back and fall back to a TTFT measurement if the share stays unreported.
- `/v1/models` pricing includes an `input_cache_reads` price, so MemoryAlloy cache hits are billed lower.

## Datasets (OpenML, `fetch_openml(data_id=...)`, all licence "Public")
Quick check: logistic regression (impute + scale + one-hot), seed 0 split (train 50%, public 100, private rest).

| Dataset | OpenML id | Rows | LR public | LR private | Role |
|---|---|---|---|---|---|
| credit-g | 31 | 1000 | 0.700 | 0.723 | dev |
| diabetes | 37 | 768 | 0.830 | 0.761 | dev |
| phoneme | 1489 | 5404 | 0.730 | 0.741 | dev |
| qsar-biodeg | 1494 | 1055 | 0.890 | 0.864 | unseen |
| blood-transfusion-service-center | 1464 | 748 | 0.760 | 0.774 | unseen |

Rejected: kc1 (id 1067 not found on OpenML), pc1 / spambase / ozone / climate (LR above 0.9, saturated), ilpd (583 rows < 700).
Note diabetes seed 0: public 0.83 vs private 0.76 for the same model. That gap is pure sampling noise on n=100,
which is exactly what the Ladder is built to ignore.

## Ladder (Blum and Hardt 2015, arXiv 1502.04585, Figure 2, parameter-free)
R_0 = inf, l_0 = zeros(n). For each submission f_t: l_t = per-example 0/1 losses; s = sample std (ddof=1) of (l_t - l_{t-1});
if R_S(f_t) < R_{t-1} - s/sqrt(n): R_t = round(R_S(f_t), to multiple of 1/n), keep l_t as best; else R_t = R_{t-1}, l_t = l_{t-1}.
Boosting attack (Section 5): k random vectors, I = {i : l_i <= 1/2}, output coordinate-wise majority of {u_i : i in I}.
Against the Ladder, the attacker keeps the vectors whose submission was accepted (the paper's modified attack).
