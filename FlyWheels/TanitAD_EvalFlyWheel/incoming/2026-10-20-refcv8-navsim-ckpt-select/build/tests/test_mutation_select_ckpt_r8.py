"""DELIBERATE-REGRESSION ARMS FOR ``select_ckpt_r8.py`` -- each must turn the literal suite RED (any python with numpy + pytest).

Each case re-runs ``tests/test_select_ckpt_r8.py`` (its fast, non-e2e part) in a FRESH interpreter with ``R8SEL_MUTATION`` set, which
reintroduces one real defect class into ``code/select_ckpt_r8.py``:

  sign_flip                 delta computed as 48,852 - 40k (every sign inverted)
  seed0_only                the seed-1 positivity / separation test is dropped
  drop_floor_gate           the "margin > 2x the inference floor" clause is removed
  floor_1x                  the floor multiplier is 1 instead of 2
  margin_mean               the margin is the MEAN of the two seeds' deltas instead of the smaller
  positional_join           arms are paired by row position, not by token
  token_cluster             the bootstrap resamples TOKENS, not logs
  no_separation_needed      delta > 0 suffices; the interval is ignored
  standin_allowed           CV stand-in rows are accepted
  min_tokens_off            the 1,000-token floor is removed
  missing_token_ignored     a token missing from one arm is dropped instead of refusing
  identity_unchecked        the arm -> checkpoint identity (md5 / step / device / seed / table) is not read
  shuffle_control_identity  the shuffled-pairing control uses the identity permutation (it can no longer widen)
  controls_not_vetoing      a control that fails to read its known value no longer voids the choice
  alpha_90                  a 90 % interval instead of 95 %
  n_boot_200                B = 200 instead of 2000
  incomplete_chooses_40k    an INCOMPLETE run defaults to 40k instead of 48,852

The unmutated suite must PASS; each mutated suite must FAIL and the NAMED literal tests must be among the failures.  A literal test that
stays green under the defect it exists to catch would be the "check that shares the defect" of CLAUDE.md -- this file is the proof it
does not.  ``select_ckpt_r8.main`` refuses to produce a decision while the variable is set.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

import conftest as C

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.dirname(HERE)
TARGET = os.path.join(HERE, "test_select_ckpt_r8.py")
FAST = "not e2e"

CASES = {
    "sign_flip": ["test_constant_better_chooses_40k", "test_constant_worse_keeps_the_default", "test_files_roundtrip_chooses_40k"],
    "seed0_only": ["test_one_seed_whose_interval_straddles_zero_keeps_the_default"],
    "drop_floor_gate": ["test_margin_inside_2x_the_floor_keeps_the_default", "test_the_margin_is_the_smaller_seed_not_the_mean"],
    "floor_1x": ["test_margin_inside_2x_the_floor_keeps_the_default", "test_constants_are_the_registered_literals"],
    "margin_mean": ["test_the_margin_is_the_smaller_seed_not_the_mean"],
    "positional_join": ["test_pairing_is_by_token_not_by_row_position"],
    "token_cluster": ["test_the_cluster_is_the_log_not_the_token"],
    "no_separation_needed": ["test_one_seed_whose_interval_straddles_zero_keeps_the_default", "test_the_cluster_is_the_log_not_the_token",
                             "test_the_interval_is_95_percent"],
    "standin_allowed": ["test_a_cv_standin_row_refuses", "test_an_unverifiable_standin_status_refuses",
                        "test_a_cv_standin_in_the_seam_file_refuses"],
    "min_tokens_off": ["test_fewer_than_1000_tokens_is_underpowered"],
    "missing_token_ignored": ["test_a_token_missing_in_any_arm_refuses"],
    "identity_unchecked": ["test_a_swapped_checkpoint_directory_refuses", "test_a_cpu_arm_refuses"],
    "shuffle_control_identity": ["test_a_token_shuffled_pairing_widens_the_interval"],
    "controls_not_vetoing": ["test_a_control_that_cannot_read_its_known_value_voids_the_choice"],
    "alpha_90": ["test_the_interval_is_95_percent", "test_constants_are_the_registered_literals"],
    "n_boot_200": ["test_the_bootstrap_is_b_2000_per_seed", "test_constants_are_the_registered_literals"],
    "incomplete_chooses_40k": ["test_every_incomplete_run_keeps_the_default", "test_a_cv_standin_row_refuses"],
}


def _run(mutation: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    env.pop("R8SEL_MUTATION", None)
    env.pop("R8SEL_BANK", None)
    if mutation:
        env["R8SEL_MUTATION"] = mutation
    return subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider", "-m", FAST, TARGET],
                          capture_output=True, text=True, env=env, cwd=BUILD, timeout=1800)


def test_unmutated_suite_is_green():
    r = _run("")
    assert r.returncode == 0, r.stdout[-2500:]


def test_the_mutation_list_matches_the_module():
    import select_ckpt_r8 as S
    assert set(CASES) == set(S.MUTATIONS)


@pytest.mark.parametrize("mutation", sorted(CASES))
def test_mutation_goes_red(mutation):
    r = _run(mutation)
    out = r.stdout.replace("\\", "/")
    assert r.returncode != 0, f"{mutation}: the literal suite stayed GREEN under the defect"
    failed = [ln.split(" ")[1].split("::")[-1].split(" - ")[0] for ln in out.splitlines() if ln.startswith("FAILED ")]
    for t in CASES[mutation]:
        assert t in failed, f"{mutation}: expected {t} to go RED; failed = {failed}"
    C.MUTATION_RECORD[mutation] = {"red": True, "failed_tests": sorted(set(failed)), "named_tests_all_red": True}
