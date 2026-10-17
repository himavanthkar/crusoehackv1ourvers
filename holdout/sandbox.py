"""Run agent-written solution.py in a subprocess: temp dir, timeout, stripped env, no network.

The child only ever receives X_train, y_train and the unlabeled X_test. It returns predictions and a short log.
On macOS, `sandbox-exec` denies outbound IP traffic; everywhere, the runner also disables socket connects.
"""
from __future__ import annotations

import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import SANDBOX_CONCURRENCY, SANDBOX_TIMEOUT_S

_sem = threading.BoundedSemaphore(SANDBOX_CONCURRENCY)
_SBX = shutil.which("sandbox-exec")
_PROFILE = "(version 1)(allow default)(deny network-outbound (remote ip))(deny network-inbound (local ip))"

RUNNER = r'''
import os, pickle, socket, sys, traceback
def _blocked(*a, **k):
    raise OSError("network is disabled in the HOLDOUT sandbox")
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
import warnings; warnings.filterwarnings("ignore")
import numpy as np
with open("data.pkl", "rb") as f:
    X_train, y_train, X_test = pickle.load(f)
os.remove("data.pkl")
sys.path.insert(0, os.getcwd())
try:
    import solution
    preds = solution.fit_predict(X_train, y_train, X_test)
    preds = np.asarray(preds).ravel()
    if preds.shape[0] != len(X_test):
        raise ValueError(f"fit_predict returned {preds.shape[0]} predictions for {len(X_test)} test rows")
    if preds.dtype.kind == "f" and not np.all(np.isin(preds, [0.0, 1.0])):
        preds = (preds >= 0.5)
    preds = preds.astype(int)
    if not np.all(np.isin(preds, [0, 1])):
        raise ValueError("predictions must be 0/1 labels")
    np.save("preds.npy", preds)
    print("OK")
except Exception:
    traceback.print_exc()
    sys.exit(3)
'''


@dataclass
class RunResult:
    ok: bool
    preds: np.ndarray | None
    log: str
    duration_s: float
    timed_out: bool = False

    @property
    def error(self) -> str:
        return "" if self.ok else self.log[-600:]


def run_solution(code: str, X_train, y_train, X_test, timeout: float = SANDBOX_TIMEOUT_S) -> RunResult:
    if os.environ.get("SANDBOX", "local").lower() == "openshell":
        return run_solution_openshell(code, X_train, y_train, X_test, timeout)
    return run_solution_local(code, X_train, y_train, X_test, timeout)


def run_solution_local(code: str, X_train, y_train, X_test, timeout: float = SANDBOX_TIMEOUT_S) -> RunResult:
    with _sem:
        d = tempfile.mkdtemp(prefix="holdout_sbx_")
        try:
            with open(os.path.join(d, "solution.py"), "w") as f:
                f.write("import numpy as np\nimport pandas as pd\n" + code)
            with open(os.path.join(d, "runner.py"), "w") as f:
                f.write(RUNNER)
            with open(os.path.join(d, "data.pkl"), "wb") as f:
                pickle.dump((X_train, np.asarray(y_train), X_test), f)
            env = {"PATH": "/usr/bin:/bin", "HOME": d, "PYTHONPATH": d, "PYTHONHASHSEED": "0",
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "LOKY_MAX_CPU_COUNT": "1", "MPLBACKEND": "Agg"}
            cmd = [sys.executable, "runner.py"]
            if _SBX:
                cmd = [_SBX, "-p", _PROFILE] + cmd
            t0 = time.time()
            p = subprocess.Popen(cmd, cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            try:
                out, _ = p.communicate(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                out, _ = p.communicate()
                timed_out = True
            dt = time.time() - t0
            log = (out or b"").decode(errors="replace")[-2000:]
            if timed_out:
                return RunResult(False, None, f"TIMEOUT after {timeout:.0f}s\n" + log, dt, True)
            pp = os.path.join(d, "preds.npy")
            if p.returncode == 0 and os.path.exists(pp):
                return RunResult(True, np.load(pp), log, dt)
            return RunResult(False, None, log or f"exit code {p.returncode}", dt)
        finally:
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- NVIDIA OpenShell
# Same interface, selected by SANDBOX=openshell. Agent code runs inside a persistent OpenShell sandbox (micro-VM,
# unprivileged user, deny-by-default network, no credentials). Set it up once: `python -m holdout.openshell_setup`.
OPENSHELL_SANDBOX = os.environ.get("OPENSHELL_SANDBOX", "holdout-sbx")
_uploaded: set = set()
_up_lock = threading.Lock()

OS_RUNNER = RUNNER.replace('with open("data.pkl", "rb") as f:', 'with open(sys.argv[1], "rb") as f:').replace(
    'os.remove("data.pkl")', 'pass').replace(
    '    np.save("preds.npy", preds)\n    print("OK")',
    '    import base64\n    print("@@PREDS", base64.b64encode(preds.astype(np.int8).tobytes()).decode())')


def _upload_data(X_train, y_train, X_test) -> str:
    import hashlib
    blob = pickle.dumps((X_train, np.asarray(y_train), X_test))
    h = hashlib.sha256(blob).hexdigest()[:16]
    with _up_lock:
        if h not in _uploaded:
            d = tempfile.mkdtemp(prefix="holdout_up_")
            os.makedirs(os.path.join(d, h))
            with open(os.path.join(d, h, "data.pkl"), "wb") as f:
                f.write(blob)
            r = subprocess.run(["openshell", "sandbox", "upload", "--no-git-ignore", OPENSHELL_SANDBOX, os.path.join(d, h),
                                "/sandbox/data"], capture_output=True, text=True, timeout=120, cwd=tempfile.gettempdir())
            shutil.rmtree(d, ignore_errors=True)
            if r.returncode != 0:
                raise RuntimeError(f"openshell upload failed: {(r.stderr or r.stdout)[-300:]}")
            _uploaded.add(h)
    return f"/sandbox/data/{h}/data.pkl"


def run_solution_openshell(code: str, X_train, y_train, X_test, timeout: float = SANDBOX_TIMEOUT_S) -> RunResult:
    import base64
    import uuid
    with _sem:
        t0 = time.time()
        try:
            data = _upload_data(X_train, y_train, X_test)
        except Exception as e:
            return RunResult(False, None, f"openshell setup error: {e}", time.time() - t0)
        d = f"/tmp/hx_{uuid.uuid4().hex[:12]}"
        b64 = lambda x: base64.b64encode(x.encode()).decode()  # noqa: E731
        script = (f"mkdir -p {d} && cd {d} && printf %s '{b64('import numpy as np' + chr(10) + 'import pandas as pd' + chr(10) + code)}' "
                  f"| base64 -d > solution.py && printf %s '{b64(OS_RUNNER)}' | base64 -d > runner.py; "
                  f"LD_LIBRARY_PATH=/sandbox/vendor/_libs PYTHONPATH={d}:/sandbox/vendor HOME={d} OMP_NUM_THREADS=1 "
                  f"PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 timeout -s KILL {int(timeout)} python3 runner.py {data} 2>&1 "
                  f"| tail -c 200000; echo @@RC $?; cd / && rm -rf {d}")
        cmd = ["openshell", "sandbox", "exec", OPENSHELL_SANDBOX, "--no-tty", "--no-login-shell",
               "--timeout", str(int(timeout) + 30), "--", "sh", "-c", script]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60, cwd=tempfile.gettempdir())
        except subprocess.TimeoutExpired:
            return RunResult(False, None, "TIMEOUT (openshell exec)", time.time() - t0, True)
        dt = time.time() - t0
        out = p.stdout or ""
        if "@@PREDS" in out:
            enc = out.split("@@PREDS", 1)[1].split()[0]
            preds = np.frombuffer(base64.b64decode(enc), dtype=np.int8).astype(int)
            if len(preds) == len(X_test):
                return RunResult(True, preds, "OK (openshell)", dt)
        log = out.split("@@PREDS")[0][-2000:] or (p.stderr or "")[-2000:]
        timed_out = dt >= timeout and "Killed" in log or "@@RC 137" in out
        return RunResult(False, None, ("TIMEOUT\n" if timed_out else "") + log, dt, timed_out)


BASELINE = '''
import numpy as np
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def fit_predict(X_train, y_train, X_test):
    pre = make_column_transformer(
        (make_pipeline(SimpleImputer(), StandardScaler()), make_column_selector(dtype_include="number")),
        (OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_exclude="number")),
    )
    model = make_pipeline(pre, LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)
    return model.predict(X_test)
'''
