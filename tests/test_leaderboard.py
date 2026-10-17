import numpy as np

from holdout.leaderboard import KaggleBoard, LadderBoard


def _setup(n=100, m=500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, m)
    pub = rng.choice(m, n, replace=False)
    return rng, y, pub


def _noisy(y, acc, rng):
    flip = rng.random(len(y)) > acc
    return np.where(flip, 1 - y, y)


def test_kaggle_reports_exact_public_accuracy_rounded():
    rng, y, pub = _setup()
    b = KaggleBoard(y[pub], pub)
    p = _noisy(y, 0.7, rng)
    v = b.submit(p, "a")
    assert v.reported == round(float(np.mean(p[pub] == y[pub])), 5)
    assert v.accepted and b.best_id == "a"


def test_ladder_first_submission_accepted():
    rng, y, pub = _setup()
    b = LadderBoard(y[pub], pub)
    p = _noisy(y, 0.7, rng)
    v = b.submit(p, "a")
    assert v.accepted and v.reported == round(float(np.mean(p[pub] == y[pub])), 5)


def test_ladder_identical_submission_never_accepted():
    rng, y, pub = _setup()
    b = LadderBoard(y[pub], pub)
    p = _noisy(y, 0.7, rng)
    b.submit(p, "a")
    for i in range(20):
        v = b.submit(p.copy(), f"dup{i}")
        assert not v.accepted
    assert b.best_id == "a"


def test_ladder_clearly_better_submission_accepted():
    rng, y, pub = _setup()
    b = LadderBoard(y[pub], pub)
    b.submit(_noisy(y, 0.6, rng), "weak")
    v = b.submit(_noisy(y, 0.95, rng), "strong")
    assert v.accepted and b.best_id == "strong"
    assert v.reported > 0.85


def test_ladder_reports_previous_best_when_rejected():
    rng, y, pub = _setup()
    b = LadderBoard(y[pub], pub)
    b.submit(_noisy(y, 0.8, rng), "a")
    first = b.best_reported
    v = b.submit(_noisy(y, 0.5, rng), "worse")
    assert not v.accepted and v.reported == first


def test_ladder_rarely_accepts_noise_variants():
    """Perturbations of the current best that change nothing real should almost never pass the step."""
    rng, y, pub = _setup(seed=3)
    b = LadderBoard(y[pub], pub)
    base = _noisy(y, 0.75, rng)
    b.submit(base, "base")
    accepted = 0
    for i in range(300):
        p = base.copy()
        flip = rng.choice(len(y), 20, replace=False)  # same skill, random flips
        p[flip] = 1 - p[flip]
        accepted += b.submit(p, f"n{i}").accepted
    assert accepted / 300 < 0.05


def test_ladder_rarely_accepts_random_guesses_after_real_model():
    rng, y, pub = _setup(seed=4)
    b = LadderBoard(y[pub], pub)
    b.submit(_noisy(y, 0.75, rng), "model")
    acc = sum(b.submit(rng.integers(0, 2, len(y)), f"r{i}").accepted for i in range(300))
    assert acc == 0


def test_kaggle_rewards_noise():
    """Under Kaggle, the max over many random guesses drifts well above 0.5 (the shake-up fuel)."""
    rng, y, pub = _setup(seed=5)
    b = KaggleBoard(y[pub], pub)
    for i in range(200):
        b.submit(rng.integers(0, 2, len(y)), f"r{i}")
    assert b.best_reported >= 0.6
