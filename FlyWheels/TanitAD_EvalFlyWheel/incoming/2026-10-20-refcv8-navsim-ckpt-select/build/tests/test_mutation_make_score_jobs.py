"""DELIBERATE-REGRESSION ARMS FOR ``make_score_jobs.py`` -- each must turn ``tests/test_make_score_jobs.py`` RED.

  label_typo             a job label that ``select_ckpt_r8.spec_from_layout`` will not find
  prior_from_challenger  the PRIOR_ha0p reference seam is taken from the CHALLENGER checkpoint's directory
  seed_seams_swapped     the seed-0 job scores the seed-1 seam and vice versa
  missing_seam_ignored   a job list is written although a seam file does not exist
"""
from __future__ import annotations

import os

import pytest

import mutharness as M

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "test_make_score_jobs.py")
CASES = {
    "label_typo": ["test_seven_jobs_with_the_literal_labels_and_seams", "test_every_job_lands_where_the_decision_script_looks"],
    "prior_from_challenger": ["test_seven_jobs_with_the_literal_labels_and_seams", "test_every_job_lands_where_the_decision_script_looks"],
    "seed_seams_swapped": ["test_seven_jobs_with_the_literal_labels_and_seams", "test_the_job_argv_is_the_score_command_of_the_driver"],
    "missing_seam_ignored": ["test_missing_seams_refuse_and_write_nothing"],
}


def test_unmutated_suite_is_green():
    r = M.run_suite(TARGET, "R8JOBS_MUTATION", "")
    assert r.returncode == 0, r.stdout[-2500:]


def test_the_mutation_list_matches_the_module():
    import make_score_jobs as J
    assert set(CASES) == set(J.MUTATIONS)


@pytest.mark.parametrize("mutation", sorted(CASES))
def test_mutation_goes_red(mutation):
    M.assert_red(TARGET, "R8JOBS_MUTATION", mutation, CASES[mutation], record_key=f"jobs:{mutation}")
