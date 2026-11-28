"""DELIBERATE-REGRESSION ARMS FOR ``export_navtrain_inputs.py`` -- each must turn ``tests/test_export_navtrain_inputs.py`` RED.

  stage_2                        the record says stage 2 (the two-stage branch) instead of 1
  frame_tokens_shifted           frame_tokens are the window off by one (history shifted into the future)
  cams_not_history               five camera paths per camera instead of the four history frames
  human_horizon_6                the human future is read at 6 poses, not W3's 8
  es_dtype_float32               ego statuses are narrowed through float32 instead of widened to float64
  sample_unchecked               the sample's FIXED status / sha256 are not verified before exporting
  velocity_acceleration_swapped  ego_velocity and ego_acceleration are exchanged
  refused_export_written         a refused (incomplete) export is still written to the canonical path
  prefix_tree_check              the imported navsim is accepted when its path merely STARTS WITH the named tree
"""
from __future__ import annotations

import os

import pytest

import mutharness as M

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "test_export_navtrain_inputs.py")
CASES = {
    "stage_2": ["test_record_schema_is_w3s"],
    "frame_tokens_shifted": ["test_record_frame_tokens_are_the_four_history_frames"],
    "cams_not_history": ["test_record_cams_are_the_history_frames_of_three_cameras"],
    "human_horizon_6": ["test_constants_are_the_registered_literals"],
    "es_dtype_float32": ["test_es_dict_widens_float32_exactly_and_keeps_float64"],
    "sample_unchecked": ["test_sample_must_be_fixed_for_navtrain", "test_sample_hash_must_rederive"],
    "velocity_acceleration_swapped": ["test_es_dict_keeps_velocity_and_acceleration_apart"],
    "refused_export_written": ["test_a_refused_export_leaves_no_file"],
    "prefix_tree_check": ["test_the_navsim_tree_check_is_exact_not_a_prefix"],
}


def test_unmutated_suite_is_green():
    r = M.run_suite(TARGET, "R8EXP_MUTATION", "")
    assert r.returncode == 0, r.stdout[-2500:]


def test_the_mutation_list_matches_the_module():
    import export_navtrain_inputs as X
    assert set(CASES) == set(X.MUTATIONS)


@pytest.mark.parametrize("mutation", sorted(CASES))
def test_mutation_goes_red(mutation):
    M.assert_red(TARGET, "R8EXP_MUTATION", mutation, CASES[mutation], record_key=f"export:{mutation}")
