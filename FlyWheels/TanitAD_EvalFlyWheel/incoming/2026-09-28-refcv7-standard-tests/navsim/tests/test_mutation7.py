"""DELIBERATE-REGRESSION ARMS THAT MUST GO RED (TANITAD VENV, CPU, no checkpoint).

Each case re-runs the LITERAL input tests in a fresh interpreter with ``R7_MUTATION`` set, which
reintroduces a real defect class into ``code/refcv7_bridge.py`` (see its section 5). The mutated
suite must FAIL (and the named literal tests among the failures); the unmutated suite must PASS.
A literal test that stays green under the defect it exists to catch would be the "check that shares
the defect" of CLAUDE.md -- this file is the proof it does not. The runner refuses to score while
``R7_MUTATION`` is set (``run_navsim_refcv7.py``).
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TARGET = os.path.join(HERE, "test_inputs7.py")

CASES = {
    "wrong_grid": ["test_prior_constant_acceleration_a0_is_2",
                   "test_hist_constant_acceleration_reads_2_mps2"],
    "drop_mask": ["test_bank_params_are_the_runs",
                  "test_KL_lift_navsim_camera_through_the_banks_own_call"],
    "stride16": ["test_bank_params_are_the_runs", "test_KL_lift_navsim_camera_through_the_banks_own_call"],
    "kw_drop": ["test_forward_kwargs_are_the_trainers"],
}


def _run(mutation: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8")
    if mutation:
        env["R7_MUTATION"] = mutation
    else:
        env.pop("R7_MUTATION", None)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider",
                           TARGET], capture_output=True, text=True, env=env, cwd=PKG, timeout=1200)


def test_unmutated_suite_is_green():
    r = _run("")
    assert r.returncode == 0, r.stdout[-2000:]


@pytest.mark.parametrize("mutation", sorted(CASES))
def test_mutation_goes_red(mutation):
    r = _run(mutation)
    assert r.returncode != 0, f"{mutation}: the literal suite stayed GREEN under the defect"
    for name in CASES[mutation]:
        assert f"FAILED tests/test_inputs7.py::{name}" in r.stdout.replace("\\", "/"), (
            f"{mutation}: {name} did not fail\n" + r.stdout[-3000:])


def test_the_runner_refuses_a_mutated_environment():
    env = dict(os.environ, R7_MUTATION="wrong_grid", CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([sys.executable, os.path.join(PKG, "code", "run_navsim_refcv7.py"),
                        "--ckpt", "nonexistent.pt"], capture_output=True, text=True, env=env,
                       timeout=300)
    assert r.returncode != 0 and "R7_MUTATION" in (r.stdout + r.stderr)
