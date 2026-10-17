import numpy as np
import pandas as pd

from holdout.sandbox import BASELINE, run_solution


def _data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200),
                      "c": pd.Categorical(rng.choice(["x", "y"], 200))})
    y = (X["a"] + 0.3 * rng.normal(size=200) > 0).astype(int).to_numpy()
    return X.iloc[:100], y[:100], X.iloc[100:].reset_index(drop=True), y[100:]


def test_baseline_runs_and_predicts():
    Xtr, ytr, Xte, yte = _data()
    r = run_solution(BASELINE, Xtr, ytr, Xte)
    assert r.ok, r.log
    assert r.preds.shape == (100,) and np.mean(r.preds == yte) > 0.8


def test_timeout_is_enforced():
    Xtr, ytr, Xte, _ = _data()
    r = run_solution("import time\ndef fit_predict(a, b, c):\n    time.sleep(30)\n", Xtr, ytr, Xte, timeout=2)
    assert not r.ok and r.timed_out


def test_network_is_blocked():
    Xtr, ytr, Xte, _ = _data()
    code = ("import urllib.request\n"
            "def fit_predict(a, b, c):\n"
            "    urllib.request.urlopen('https://example.com', timeout=5)\n"
            "    return [0] * len(c)\n")
    r = run_solution(code, Xtr, ytr, Xte, timeout=20)
    assert not r.ok


def test_no_secrets_in_env():
    Xtr, ytr, Xte, _ = _data()
    code = ("import os\n"
            "def fit_predict(a, b, c):\n"
            "    assert not any('KEY' in k for k in os.environ), sorted(os.environ)\n"
            "    return [0] * len(c)\n")
    r = run_solution(code, Xtr, ytr, Xte)
    assert r.ok, r.log


def test_bad_output_rejected():
    Xtr, ytr, Xte, _ = _data()
    r = run_solution("def fit_predict(a, b, c):\n    return [0, 1]\n", Xtr, ytr, Xte)
    assert not r.ok and "predictions" in r.log
