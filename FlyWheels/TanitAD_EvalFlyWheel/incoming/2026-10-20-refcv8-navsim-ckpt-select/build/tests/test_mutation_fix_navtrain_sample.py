"""DELIBERATE-REGRESSION ARMS FOR ``fix_navtrain_sample.py`` -- each must turn ``tests/test_fix_navtrain_sample.py`` RED.

  no_sort_logs        logs are taken in the filter's order, not sorted by log_name
  stop_strict         the sample walks on until the total EXCEEDS the target (">" for ">=")
  f0_only             readable = CAM_F0 only (the registered text's too-narrow reading, AMENDMENT 1a)
  t0_frame_only       readable = the t0 frame only (no history frames)
  truncate_to_target  the stop log is cut to exactly the target instead of taken whole
  hash_unsorted       the sha256 is taken over the tokens in log order, not sorted
  underpowered_off    the < min-tokens UNDERPOWERED status is never raised
  no_overwrite_guard  an existing FIXED sample is silently replaced
  keep_unwindowed     tokens whose devkit window does not exist (no route / too close to the log end) are kept
"""
from __future__ import annotations

import os

import pytest

import mutharness as M

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "test_fix_navtrain_sample.py")
CASES = {
    "no_sort_logs": ["test_sorted_logs_all_readable_tokens_stop_at_the_first_log_reaching_the_target"],
    "stop_strict": ["test_a_log_that_lands_exactly_on_the_target_stops_the_sample"],
    "f0_only": ["test_front_camera_only_is_not_readable"],
    "t0_frame_only": ["test_history_frames_are_required_not_just_the_current_frame"],
    "truncate_to_target": ["test_sorted_logs_all_readable_tokens_stop_at_the_first_log_reaching_the_target"],
    "hash_unsorted": ["test_the_sha256_is_over_the_sorted_token_list"],
    "underpowered_off": ["test_fewer_than_min_tokens_is_underpowered", "test_front_camera_only_is_not_readable"],
    "no_overwrite_guard": ["test_an_existing_fixed_sample_is_never_replaced_with_different_content"],
    "keep_unwindowed": ["test_a_window_the_devkit_drops_is_not_readable"],
}


def test_unmutated_suite_is_green():
    r = M.run_suite(TARGET, "R8FIX_MUTATION", "")
    assert r.returncode == 0, r.stdout[-2500:]


def test_the_mutation_list_matches_the_module():
    import fix_navtrain_sample as F
    assert set(CASES) == set(F.MUTATIONS)


@pytest.mark.parametrize("mutation", sorted(CASES))
def test_mutation_goes_red(mutation):
    M.assert_red(TARGET, "R8FIX_MUTATION", mutation, CASES[mutation], record_key=f"fix:{mutation}")
