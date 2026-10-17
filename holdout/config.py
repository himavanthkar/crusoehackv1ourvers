"""Models, budgets, seeds, sizes. Everything overridable from .env."""
from __future__ import annotations

import os
from pathlib import Path

import certifi
from dotenv import load_dotenv

# python.org macOS builds ship without a CA bundle; urllib (fetch_openml) needs one
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CRUSOE_BASE_URL = os.getenv("CRUSOE_BASE_URL", "https://api.inference.crusoecloud.com/v1")
CRUSOE_API_KEY = os.getenv("CRUSOE_API_KEY", "")

# Exact ids from GET /v1/models (Step 0, see NOTES.md). Nemotron 3 Ultra is not live on this key's
# project, so the curator uses the largest live Nemotron (Super 120B); researchers use Lightning 30B-A3B.
RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "nvidia/Nemotron-3.5-Lightning-30B-A3B")
CURATOR_MODEL = os.getenv("CURATOR_MODEL", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B")
RESEARCHER_TEMP = 0.7
CURATOR_TEMP = 0.2

# USD per 1M tokens: (prompt, completion, cached prompt). From /v1/models pricing.
PRICING = {
    "nvidia/Nemotron-3.5-Lightning-30B-A3B": (0.05, 0.20, 0.03),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.05, 0.20, 0.03),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B": (0.30, 2.40, 0.15),
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL": (0.10, 0.40, 0.08),
}

LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "6"))
SANDBOX_CONCURRENCY = int(os.getenv("SANDBOX_CONCURRENCY", "6"))
BUDGET_USD = float(os.getenv("BUDGET_USD", "15"))

# Arena sizes (spec section 3)
ROUNDS = int(os.getenv("ROUNDS", "4"))
N_RESEARCHERS = int(os.getenv("N_RESEARCHERS", "4"))
SUBS_PER_RESEARCHER = int(os.getenv("SUBS_PER_RESEARCHER", "2"))
PUBLIC_N = 100
TRAIN_FRAC = 0.5
SANDBOX_TIMEOUT_S = 60
PLAYBOOK_MAX_RULES = 12
PLAYBOOK_MAX_TOKENS = 1200
SEEDS = [int(s) for s in os.getenv("SEEDS", "0").split(",")]

DIRECTION_HINTS = [
    "feature engineering (encodings, interactions, scaling, handling skew and missing values)",
    "model choice (try a different model family than the obvious default)",
    "regularization and hyperparameter tuning (use cross-validation on the training set)",
    "ensembling (blend or stack models, or use the ensemble tool on earlier submissions)",
]

CACHE_DIR = ROOT / "cache"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
DATA_HOME = ROOT / "data_cache"
