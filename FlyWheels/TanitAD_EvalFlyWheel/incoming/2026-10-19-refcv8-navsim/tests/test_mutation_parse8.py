"""DELIBERATE-REGRESSION ARMS FOR parse8 -- each must turn the literal suite RED (TANITAD VENV, CPU).

Each case re-runs ``tests/test_parse8.py`` in a FRESH interpreter with ``PARSE8_MUTATION`` set, which reintroduces one real defect
class into ``code/parse8.py``:

  swap_seed_arms        R8_LEGAL (seed 0, the BAR arm) and R8_LEGAL_s1 (seed 1, the floor arm) trade files -> every margin and the
                        paired seed-floor delta change sign / value
  drop_blind_arm        the frames-blind control arm is not read (N3's panel can then never be certified, or voided)
  skip_count_guard      a CSV beside a FAILED count guard is used (partial work enters the table)
  silent_skip_missing   a missing expected arm is skipped silently (parse7's behaviour) instead of reading UNAVAILABLE
  ignore_cv_standin     an arm whose stage-1 rows are the declared CV stand-in is reported as a real two-stage EPDMS
  s1_pair_reuses_s0     the seed-1 mirror pairs (R8_LEGAL_s1 vs STOP/CV/ECHO) silently take the SEED-0 rows (AMENDMENT 3: the bars hold
                        on both inference seeds, so a mirror that is really seed 0 doubles the evidence and hides a seed-1 failure)

The unmutated suite must PASS; the mutated suite must FAIL and the NAMED literal tests must be among the failures.  A literal test
that stays green under the defect it exists to catch would be the 'check that shares the defect' of CLAUDE.md -- this file is the
proof it does not.  ``parse8.main`` refuses to write while the variable is set.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TARGET = os.path.join(HERE, "test_parse8.py")
FAST = "not control and not navhard"            # the warmup-shaped synthetic splits + the e2e-on-synthetic bars checks (~10 s)
WITH_NAVHARD = "not control"                    # + the navhard-shaped synthetic splits (~35 s)

CASES = {
    "swap_seed_arms": (FAST, ["test_warmup_s2_epdms_u_literal", "test_seed_arms_are_not_swapped", "test_wtl_counts_literal",
                              "test_e2e_warmup_synthetic_n3_passes_and_the_blind_panel_is_ok"]),
    "drop_blind_arm": (FAST, ["test_blind_arm_is_read", "test_warmup_s2_epdms_u_literal",
                              "test_e2e_warmup_synthetic_n3_passes_and_the_blind_panel_is_ok",
                              "test_e2e_a_blind_arm_that_passes_voids_the_panel"]),
    "skip_count_guard": (FAST, ["test_count_guard_failure_refuses_the_arm"]),
    "silent_skip_missing": (WITH_NAVHARD, ["test_missing_expected_arm_reads_unavailable_never_a_value",
                                           "test_missing_floor_arm_reads_unavailable_never_a_value",
                                           "test_navhard_a_missing_floor_arm_makes_its_paired_interval_unavailable"]),
    "ignore_cv_standin": (FAST, ["test_cv_standin_stage1_is_not_reported_as_a_real_two_stage_epdms"]),
    "s1_pair_reuses_s0": (WITH_NAVHARD, ["test_warmup_seed1_mirror_pairs_are_point_deltas_only",
                                         "test_warmup_seed1_mirror_reads_unavailable_when_the_s1_arm_is_absent",
                                         "test_navhard_seed1_mirror_rows_literal",
                                         "test_navhard_seed1_mirror_rows_read_unavailable_when_the_s1_arm_is_absent"]),
}


def _run(mutation: str, kexpr: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PARSE8_MUTATION", None)
    if mutation:
        env["PARSE8_MUTATION"] = mutation
    return subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider", "-k", kexpr, TARGET],
                          capture_output=True, text=True, env=env, cwd=PKG, timeout=1800)


def test_unmutated_suite_is_green():
    r = _run("", WITH_NAVHARD)
    assert r.returncode == 0, r.stdout[-2500:]


def test_the_mutation_list_matches_the_module():
    sys.path.insert(0, os.path.join(PKG, "code"))
    import parse8
    assert set(CASES) == set(parse8.MUTATIONS)


@pytest.mark.parametrize("mutation", sorted(CASES))
def test_mutation_goes_red(mutation):
    kexpr, names = CASES[mutation]
    r = _run(mutation, kexpr)
    out = r.stdout.replace("\\", "/")
    assert r.returncode != 0, f"{mutation}: the literal suite stayed GREEN under the defect"
    for name in names:
        assert re.search(rf"FAILED .*test_parse8\.py::{re.escape(name)}", out), f"{mutation}: {name} did not fail\n" + out[-3000:]
