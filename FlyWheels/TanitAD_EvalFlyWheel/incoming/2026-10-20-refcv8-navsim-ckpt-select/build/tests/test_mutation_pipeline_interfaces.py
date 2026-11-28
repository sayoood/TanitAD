"""DELIBERATE-REGRESSION ARMS FOR THE CHAIN ``tests/test_pipeline_interfaces.py``: a defect in the PRODUCER or the CONSUMER of an interface must
turn a chain test RED, not only the producer's own fixture-based tests.

  R8BANK_MUTATION=missing_frame_standin   the bank builder drops a token with a missing jpg instead of refusing (the B7 defect)
  R8BANK_MUTATION=kb3_t0_only             the bank builder demands only the t0 frame, not the 4 history frames
  R8EXP_MUTATION=cams_not_history         the export record lists 5 camera paths per camera instead of the 4 history frames
  R8EXP_MUTATION=frame_tokens_shifted     the export record's frame_tokens are the window off by one
"""
from __future__ import annotations

import os

import pytest

import mutharness as M

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "test_pipeline_interfaces.py")
BUILDS = "test_a_sample_the_fixer_calls_readable_is_buildable_from_the_real_record_builder"
B7 = "test_what_the_fixer_excludes_for_a_missing_jpg_is_exactly_what_the_bank_builder_refuses"
CASES = {
    ("R8BANK_MUTATION", "missing_frame_standin"): [B7],
    ("R8BANK_MUTATION", "kb3_t0_only"): [BUILDS, B7],
    ("R8EXP_MUTATION", "cams_not_history"): [BUILDS, B7],
    ("R8EXP_MUTATION", "frame_tokens_shifted"): [BUILDS],
}


def test_unmutated_chain_is_green():
    r = M.run_suite(TARGET, "R8BANK_MUTATION", "")
    assert r.returncode == 0, r.stdout[-2500:]


@pytest.mark.parametrize("env_var,mutation", sorted(CASES))
def test_mutation_goes_red(env_var, mutation):
    M.assert_red(TARGET, env_var, mutation, CASES[(env_var, mutation)], record_key=f"chain:{env_var}:{mutation}")
