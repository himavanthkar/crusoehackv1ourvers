"""Competitions: OpenML binary classification datasets, Kaggle-style public/private splits."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from .config import DATA_HOME, PUBLIC_N, TRAIN_FRAC

# name -> OpenML data id. Chosen in M0 by a quick logistic-regression check (see NOTES.md).
DEV = {"credit-g": 31, "diabetes": 37, "phoneme": 1489}
UNSEEN = {"qsar-biodeg": 1494, "blood-transfusion-service-center": 1464}
CANDIDATES = {**DEV, **UNSEEN, "pc1": 1068, "ilpd": 1480,
              "spambase": 44, "ozone-level-8hr": 1487, "climate-model-simulation-crashes": 1467}

_raw_cache: dict[str, tuple[pd.DataFrame, np.ndarray, dict]] = {}


def load_raw(name: str) -> tuple[pd.DataFrame, np.ndarray, dict]:
    if name in _raw_cache:
        return _raw_cache[name]
    b = fetch_openml(data_id=CANDIDATES[name], as_frame=True, data_home=str(DATA_HOME), parser="auto")
    X = b.data.copy()
    y_raw = b.target.astype(str)
    classes = sorted(y_raw.unique())
    assert len(classes) == 2, f"{name} is not binary: {classes}"
    # positive class = minority class
    pos = y_raw.value_counts().idxmin()
    y = (y_raw == pos).astype(int).to_numpy()
    X.columns = [str(c).replace(" ", "_") for c in X.columns]
    meta = {"name": name, "openml_id": CANDIDATES[name], "licence": b.details.get("licence", "unknown"),
            "version": b.details.get("version"), "rows": len(X), "features": X.shape[1], "positive_class": pos,
            "url": f"https://www.openml.org/d/{CANDIDATES[name]}"}
    _raw_cache[name] = (X, y, meta)
    return X, y, meta


@dataclass
class Competition:
    name: str
    seed: int
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame          # public + private rows, shuffled together, no labels
    # --- sealed: never shown to agents ---
    y_test: np.ndarray = field(repr=False)
    is_public: np.ndarray = field(repr=False)   # bool mask over X_test rows
    meta: dict = field(default_factory=dict)

    @property
    def public_idx(self) -> np.ndarray:
        return np.flatnonzero(self.is_public)

    @property
    def private_idx(self) -> np.ndarray:
        return np.flatnonzero(~self.is_public)

    def card(self) -> str:
        """Competition card shown to agents: shape, feature types, class balance. No test labels."""
        num = [c for c in self.X_train.columns if pd.api.types.is_numeric_dtype(self.X_train[c])]
        cat = [c for c in self.X_train.columns if c not in num]
        miss = int(self.X_train.isna().sum().sum())
        lines = [
            f"Competition: {self.name} (binary classification, metric = accuracy)",
            f"Train rows: {len(self.X_train)}  Test rows to predict: {len(self.X_test)}  Features: {self.X_train.shape[1]}",
            f"Numeric features ({len(num)}): {', '.join(num[:40])}{' ...' if len(num) > 40 else ''}",
            f"Categorical features ({len(cat)}): {', '.join(cat[:40])}{' ...' if len(cat) > 40 else ''}",
            f"Missing values in train: {miss}",
            f"Train class balance: {np.mean(self.y_train == 1):.3f} positive (label 1), {np.mean(self.y_train == 0):.3f} negative (label 0)",
            "X_train / X_test are pandas DataFrames (categoricals have pandas 'category' dtype); y_train is a numpy int array of 0/1.",
            f"The public leaderboard scores {PUBLIC_N} hidden rows of X_test; the rest are scored privately at the end.",
        ]
        return "\n".join(lines)


def make_competition(name: str, seed: int = 0) -> Competition:
    X, y, meta = load_raw(name)
    idx = np.arange(len(X))
    tr, rest = train_test_split(idx, train_size=TRAIN_FRAC, stratify=y, random_state=seed)
    pub, priv = train_test_split(rest, train_size=PUBLIC_N, stratify=y[rest], random_state=seed + 1)
    rng = np.random.default_rng(seed + 2)
    test = np.concatenate([pub, priv])
    perm = rng.permutation(len(test))
    test = test[perm]
    is_public = np.isin(test, pub)
    X_train = X.iloc[tr].reset_index(drop=True)
    X_test = X.iloc[test].reset_index(drop=True)
    return Competition(name, seed, X_train, y[tr], X_test, y[test], is_public, meta)
