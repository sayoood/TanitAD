"""Shared deliberate-regression harness: re-run a LITERAL test file in a fresh interpreter with a mutation env var set, and report which
tests went red.  The unmutated run must be green; every mutated run must be red AND must contain the NAMED tests among its failures."""
from __future__ import annotations

import os
import subprocess
import sys

import conftest as C

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.dirname(HERE)


def run_suite(target: str, env_var: str, mutation: str, fast: str = "not e2e and not devkit and not stitch and not bridge") -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    env.pop(env_var, None)
    env.pop("R8SEL_BANK", None)
    if mutation:
        env[env_var] = mutation
    return subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider", "-m", fast, target],
                          capture_output=True, text=True, env=env, cwd=BUILD, timeout=1800)


def failed_tests(stdout: str) -> list:
    out = stdout.replace("\\", "/")
    return sorted({ln.split(" ")[1].split("::")[-1].split(" - ")[0].split("[")[0] for ln in out.splitlines() if ln.startswith("FAILED ")})


def assert_red(target, env_var, mutation, named, record_key=None):
    r = run_suite(target, env_var, mutation)
    assert r.returncode != 0, f"{mutation}: the literal suite stayed GREEN under the defect"
    failed = failed_tests(r.stdout)
    for t in named:
        assert t in failed, f"{mutation}: expected {t} to go RED; failed = {failed}"
    n_tests = r.stdout.count("passed") and r.stdout.strip().splitlines()[-1]
    C.MUTATION_RECORD[record_key or mutation] = {"red": True, "failed_tests": failed, "named_tests_all_red": True, "pytest_summary": n_tests}
    return failed
