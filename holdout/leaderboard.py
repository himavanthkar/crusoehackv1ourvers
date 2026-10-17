"""Leaderboard mechanisms: Kaggle (naive) and the parameter-free Ladder (Blum and Hardt 2015, arXiv 1502.04585, Fig. 2).

Both score only the public rows of X_test. Scores are reported as accuracy (1 - 0/1 loss).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Verdict:
    reported: float      # what the team sees for this submission
    accepted: bool       # Kaggle: new best reported score. Ladder: passed the significance step.
    true_public: float   # sealed: exact public accuracy (for analysis only, never shown to agents)


class Board:
    name = "base"

    def __init__(self, y_public: np.ndarray, public_idx: np.ndarray):
        self.y = np.asarray(y_public).astype(int)
        self.idx = np.asarray(public_idx)
        self.n = len(self.y)
        self.best_id: str | None = None

    def losses(self, preds: np.ndarray) -> np.ndarray:
        p = np.asarray(preds).astype(int)[self.idx]
        return (p != self.y).astype(float)

    def submit(self, preds: np.ndarray, sub_id: str | None = None) -> Verdict:
        raise NotImplementedError

    @property
    def best_reported(self) -> float | None:
        raise NotImplementedError


class KaggleBoard(Board):
    """Every submission gets its public accuracy, rounded to 5 decimals."""
    name = "kaggle"

    def __init__(self, y_public, public_idx):
        super().__init__(y_public, public_idx)
        self._best = None

    def submit(self, preds, sub_id=None) -> Verdict:
        acc = 1.0 - float(self.losses(preds).mean())
        reported = round(acc, 5)
        accepted = self._best is None or reported > self._best
        if accepted:
            self._best, self.best_id = reported, sub_id
        return Verdict(reported, accepted, acc)

    @property
    def best_reported(self):
        return self._best


class LadderBoard(Board):
    """Parameter-free Ladder, exactly as Figure 2 of the paper:

    R_0 = inf, l_0 = (0)^n. For each f_t: l_t = loss vector; s = std(l_t - l_{t-1}) (sample std, ddof=1);
    if R_S(f_t) < R_{t-1} - s/sqrt(n): R_t = [R_S(f_t)]_{1/n}, keep l_t; else R_t = R_{t-1}, l_t = l_{t-1}.
    The released value R_t is reported (as accuracy 1 - R_t) for every submission.
    """
    name = "ladder"

    def __init__(self, y_public, public_idx):
        super().__init__(y_public, public_idx)
        self.R = math.inf
        self.best_loss = np.zeros(self.n)
        self.last_threshold = None

    def submit(self, preds, sub_id=None) -> Verdict:
        l = self.losses(preds)
        r = float(l.mean())
        s = float(np.std(l - self.best_loss, ddof=1))
        self.last_threshold = s / math.sqrt(self.n)
        accepted = r < self.R - self.last_threshold
        if accepted:
            self.R = round(r * self.n) / self.n
            self.best_loss = l
            self.best_id = sub_id
        return Verdict(round(1.0 - self.R, 5), accepted, 1.0 - r)

    @property
    def best_reported(self):
        return None if math.isinf(self.R) else round(1.0 - self.R, 5)


BOARDS = {"kaggle": KaggleBoard, "ladder": LadderBoard}
