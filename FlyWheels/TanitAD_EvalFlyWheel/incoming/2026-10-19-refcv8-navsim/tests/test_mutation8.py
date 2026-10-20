"""DELIBERATE-REGRESSION ARMS THAT MUST GO RED (TANITAD VENV, CPU, no checkpoint).

Each case re-runs the LITERAL input tests (``tests/test_inputs8.py``) in a fresh interpreter with ``R8_MUTATION`` set, which
reintroduces a real defect class into ``code/refcv8_bridge.py`` (its section 7).  The mutated suite must FAIL, and the named literal
tests must be among the failures; the unmutated suite must PASS.  A literal test that stays green under the defect it exists to
catch would be the "check that shares the defect" of CLAUDE.md -- this file is the proof it does not.  The runners refuse to score
while ``R8_MUTATION`` is set (``test_the_runners_refuse_a_mutated_environment``).
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TARGET = os.path.join(HERE, "test_inputs8.py")

CASES = {
    # the feed-the-WRONG-TOKEN defect: TURN_L and TURN_R swapped
    "wrong_token": ["test_nav_command_to_token_literal", "test_token_agrees_with_E2s_independent_route"],
    # a nav ARGUMENT row (known = 1) on the LEGAL row
    "nav_fed_known": ["test_r8_nav_row_is_exactly_zero", "test_unknown_row_equals_the_dropout_path_even_with_nonzero_args",
                      "test_forward_kwargs8_emits_exactly_the_contract"],
    # r8_nav never passed (R8-EVAL-NAV-FEED-1's other half)
    "drop_r8_nav": ["test_forward_kwargs8_emits_exactly_the_contract"],
    # a speed ceiling on the LEGAL row
    "legal_off": ["test_legal_speed_is_the_unknown_row"],
    # forwarded to the refcv7 modules (boot8): the 2 Hz packing / the 43-row mask / the stride-16 lift
    "wrong_grid": ["test_prior_constant_acceleration_a0_is_2", "test_hist_constant_acceleration_reads_2_mps2"],
    "drop_mask": ["test_bank_params_are_the_runs", "test_KL_lift_navsim_camera_through_the_banks_own_call"],
    "stride16": ["test_bank_params_are_the_runs", "test_KL_lift_navsim_camera_through_the_banks_own_call"],
}


def _run(mutation: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8")
    env.pop("R7_MUTATION", None)
    if mutation:
        env["R8_MUTATION"] = mutation
    else:
        env.pop("R8_MUTATION", None)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider", TARGET],
                          capture_output=True, text=True, env=env, cwd=PKG, timeout=1800)


def test_unmutated_suite_is_green():
    r = _run("")
    assert r.returncode == 0, r.stdout[-2500:]


@pytest.mark.parametrize("mutation", sorted(CASES))
def test_mutation_goes_red(mutation):
    r = _run(mutation)
    out = r.stdout.replace("\\", "/")
    assert r.returncode != 0, f"{mutation}: the literal suite stayed GREEN under the defect"
    for name in CASES[mutation]:
        assert f"FAILED tests/test_inputs8.py::{name}" in out, f"{mutation}: {name} did not fail\n" + out[-3000:]
