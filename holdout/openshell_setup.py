"""Create the NVIDIA OpenShell sandbox for SANDBOX=openshell:  python -m holdout.openshell_setup <wheel_dir>

The sandbox has no network, so Linux aarch64 wheels (pinned to this venv's versions, so pickled DataFrames stay
compatible) are downloaded on the host and installed offline inside it:
  pip download --only-binary=:all: --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 \
      --python-version 3.12 --implementation cp -d <wheel_dir> numpy==X pandas==X scikit-learn==X scipy==X lightgbm==X
LightGBM needs libgomp.so.1, which the slim image lacks; we link the copy scikit-learn's wheel already bundles.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile

from .sandbox import BASELINE, OPENSHELL_SANDBOX


def sh(*args: str) -> subprocess.CompletedProcess:
    print("$", " ".join(args))
    return subprocess.run(args, text=True, capture_output=True, cwd=tempfile.gettempdir())


def main(wheels: str) -> int:
    if OPENSHELL_SANDBOX not in sh("openshell", "sandbox", "list").stdout:
        print(sh("openshell", "sandbox", "create", "--name", OPENSHELL_SANDBOX, "--from", "python:3.12-slim", "--no-tty",
                 "--detach", "--", "sleep", "infinity").stdout[-500:])
    print(sh("openshell", "sandbox", "upload", "--no-git-ignore", OPENSHELL_SANDBOX, wheels, "/sandbox/wheels").stdout[-300:])
    r = sh("openshell", "sandbox", "exec", OPENSHELL_SANDBOX, "--no-tty", "--no-login-shell", "--timeout", "600", "--",
           "sh", "-c", "pip install -q --no-index --find-links /sandbox/wheels --find-links /sandbox/wheels/wheels "
           "--target /sandbox/vendor numpy pandas scikit-learn scipy lightgbm && mkdir -p /sandbox/vendor/_libs && "
           "ln -sf $(ls /sandbox/vendor/scikit_learn.libs/libgomp*.so* | head -1) /sandbox/vendor/_libs/libgomp.so.1 && "
           "LD_LIBRARY_PATH=/sandbox/vendor/_libs PYTHONPATH=/sandbox/vendor python3 -c 'import sklearn, lightgbm; print(\"ready\")'")
    print(r.stdout[-300:], r.stderr[-300:])
    import os
    os.environ["SANDBOX"] = "openshell"
    from .data import make_competition
    from .sandbox import run_solution
    c = make_competition("diabetes", 0)
    res = run_solution(BASELINE, c.X_train, c.y_train, c.X_test)
    print("smoke test:", "PASS" if res.ok else res.log[-500:])
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
