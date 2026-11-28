"""DELIBERATE-REGRESSION ARMS FOR ``build_navtrain_frames.py`` -- each must turn ``tests/test_build_navtrain_frames.py`` RED.

  missing_frame_standin      a token with a missing jpg is silently dropped instead of REFUSING the build (the B7 defect)
  kb3_t0_only                only the t0 frame is demanded (history frames are not extracted / checked)
  zero_frame_ok              the all-black-frame check (KB2, mean_px >= 1) is removed
  dup_content_ignored        two shard members with the same path but different bytes are accepted
  partial_on_stitch_failure  a frame that fails to stitch drops its tokens silently instead of refusing
  rows_shifted               the token -> rows mapping in index.json is off by one
"""
from __future__ import annotations

import os

import pytest

import mutharness as M

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "test_build_navtrain_frames.py")
CASES = {
    "missing_frame_standin": ["test_one_missing_history_jpg_refuses_and_names_every_affected_token", "test_a_missing_t0_jpg_of_one_camera_refuses",
                              "test_an_archive_that_is_absent_refuses"],
    "kb3_t0_only": ["test_one_missing_history_jpg_refuses_and_names_every_affected_token", "test_unique_frames_are_stored_once_and_rows_are_the_history_windows"],
    "zero_frame_ok": ["test_an_all_black_frame_refuses"],
    "dup_content_ignored": ["test_two_shard_members_with_the_same_path_but_different_bytes_refuse"],
    "partial_on_stitch_failure": ["test_a_frame_that_fails_to_stitch_refuses"],
    "rows_shifted": ["test_unique_frames_are_stored_once_and_rows_are_the_history_windows", "test_a_token_reads_its_own_four_frames_in_history_order"],
}


def test_unmutated_suite_is_green():
    r = M.run_suite(TARGET, "R8BANK_MUTATION", "")
    assert r.returncode == 0, r.stdout[-2500:]


def test_the_mutation_list_matches_the_module():
    import build_navtrain_frames as B
    assert set(CASES) == set(B.MUTATIONS)


@pytest.mark.parametrize("mutation", sorted(CASES))
def test_mutation_goes_red(mutation):
    M.assert_red(TARGET, "R8BANK_MUTATION", mutation, CASES[mutation], record_key=f"bank:{mutation}")
