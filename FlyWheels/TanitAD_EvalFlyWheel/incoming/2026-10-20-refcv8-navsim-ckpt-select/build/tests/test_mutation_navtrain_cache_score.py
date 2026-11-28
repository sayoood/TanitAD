"""DELIBERATE-REGRESSION ARMS FOR ``navtrain_cache_score.py`` -- each must turn ``tests/test_navtrain_cache_score.py`` RED.

  split_navtest                the overrides name ``train_test_split=navtest`` (a navtest scorer pointed at navtrain tokens)
  tokens_not_restricted        the sample's tokens are not passed: the devkit would take ALL 103,288 navtrain tokens
  log_names_not_restricted     the sample's logs are not passed
  no_cmdline_guard             an over-long Hydra command line is launched instead of refused
  maps_sample_cities_only      a maps root is accepted when it holds the SAMPLE'S cities (Thor's Las-Vegas-only state)
  pickles_not_checked          the trainval log pickles are not checked before the cache / scorer run
  binding_without_sample_sha   the binding omits the sample's sorted-token sha256
  binding_on_failed_counts     a binding is written although the scorer's counts do not say PASS
  csv_token_set_unchecked      the CSV's token set is not compared with the sample before binding
  rescore_overwrites           a label already bound to another seam / agent / sample is rescored over
  label_prefix_unchecked       labels need not start with r8sel_
  chunk_splits_a_log           the Windows chunker cuts a log across two chunks
  cache_verified_by_count_only the cache verifier compares row COUNTS, not the token SETS
  control_binding_says_navtrain a navtest control binding claims split navtrain
"""
from __future__ import annotations

import os

import pytest

import mutharness as M

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "test_navtrain_cache_score.py")
OV = "test_overrides_name_the_navtrain_split_and_exactly_the_samples_logs_and_tokens"
CASES = {
    "split_navtest": [OV],
    "tokens_not_restricted": [OV],
    "log_names_not_restricted": [OV],
    "no_cmdline_guard": ["test_windows_refuses_a_2000_token_line_and_accepts_1000", "test_linux_refuses_one_argument_over_131072_bytes"],
    "maps_sample_cities_only": ["test_a_maps_root_with_only_las_vegas_is_not_enough"],
    "pickles_not_checked": ["test_a_missing_or_empty_log_pickle_is_a_problem", "test_preflight_is_ok_only_when_nothing_is_missing"],
    "binding_without_sample_sha": ["test_a_binding_carries_the_four_things_the_decision_script_requires",
                                   "test_a_binding_from_here_is_accepted_by_the_decision_scripts_loader"],
    "binding_on_failed_counts": ["test_counts_that_are_not_pass_refuse_a_binding"],
    "csv_token_set_unchecked": ["test_a_csv_missing_a_sample_token_refuses_a_binding", "test_a_csv_with_an_extra_token_refuses_a_binding"],
    "rescore_overwrites": ["test_a_label_bound_to_different_input_is_never_rescored"],
    "label_prefix_unchecked": ["test_a_label_without_the_prefix_refuses"],
    "chunk_splits_a_log": ["test_chunks_are_whole_logs_disjoint_and_exhaustive", "test_a_log_is_never_split_across_chunks"],
    "cache_verified_by_count_only": ["test_a_swapped_token_with_the_same_count_is_reported", "test_a_pickle_on_disk_that_the_sample_does_not_name_is_reported"],
    "control_binding_says_navtrain": ["test_a_navtest_control_binding_is_rejected_by_the_decision_script"],
}


def test_unmutated_suite_is_green():
    r = M.run_suite(TARGET, "R8CS_MUTATION", "")
    assert r.returncode == 0, r.stdout[-2500:]


def test_the_mutation_list_matches_the_module():
    import navtrain_cache_score as S
    assert set(CASES) == set(S.MUTATIONS)


@pytest.mark.parametrize("mutation", sorted(CASES))
def test_mutation_goes_red(mutation):
    M.assert_red(TARGET, "R8CS_MUTATION", mutation, CASES[mutation], record_key=f"cache_score:{mutation}")
