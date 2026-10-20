"""DELIBERATE-REGRESSION ARMS FOR THE BARS LAYER -- each must turn the literal suite RED (TANITAD VENV, CPU).

Each case re-runs ``tests/test_navsim_bars8.py`` (the fast literal part) in a FRESH interpreter with ``NAVSIM_BARS8_MUTATION`` set, which
reintroduces one real defect class into ``code/navsim_bars8.py``:

  swap_arm_labels           the bar arm and STOP trade labels (every margin changes sign)
  drop_seed_floor_gate      the inference-seed-floor gate is removed (a margin inside the floor passes)
  speed_edges_closed_right  the t0-speed bands become right-CLOSED (a token at exactly 2.0 / 8.33 / 13.89 changes band)
  positional_join           arms are paired by row position, not by token
  rt_bar_without_ship       R8_RT carries a bar although SHIP-R8 has not passed
  priv_carries_bar          R8_PRIV is given a bar
  drop_drive_conjunction    'separated' reads the log level only (the nuplan_drive level is ignored)

The unmutated suite must PASS; the mutated suite must FAIL and the NAMED literal tests must be among the failures.  A literal test that
stays green under the defect it exists to catch would be the 'check that shares the defect' of CLAUDE.md -- this file is the proof it does
not.  ``navsim_bars8.main`` refuses to score while the variable is set.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TARGET = os.path.join(HERE, "test_navsim_bars8.py")
FAST = "not e2e and not fullsplit"

CASES = {
    "swap_arm_labels": ["test_n1_and_n4_pass_literal", "test_swapped_arm_labels_would_flip_the_n1_verdict_literal"],
    "drop_seed_floor_gate": ["test_n1_inside_2x_floor_is_not_proven", "test_n4_inside_2x_floor_is_not_proven",
                             "test_n1_needs_the_floor_arm_to_pass", "test_n2_literal"],
    "speed_edges_closed_right": ["test_speed_band_edges_literal", "test_strata_partition_counts_literal"],
    "positional_join": ["test_join_is_keyed_by_token_not_by_position"],
    "rt_bar_without_ship": ["test_rt_carries_no_bar_unless_ship_r8_passed"],
    "priv_carries_bar": ["test_priv_never_carries_a_bar"],
    "drop_drive_conjunction": ["test_pair_positive_needs_both_levels"],
    "n4_floor_1x": ["test_n4_inside_2x_floor_is_not_proven"],
}


def _run(mutation: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    env.pop("NAVSIM_BARS8_MUTATION", None)
    if mutation:
        env["NAVSIM_BARS8_MUTATION"] = mutation
    return subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider", "-k", FAST, TARGET],
                          capture_output=True, text=True, env=env, cwd=PKG, timeout=1800)


def test_unmutated_suite_is_green():
    r = _run("")
    assert r.returncode == 0, r.stdout[-2500:]


def test_the_mutation_list_matches_the_module():
    sys.path.insert(0, os.path.join(PKG, "code"))
    import navsim_bars8 as B
    assert set(CASES) == set(B.MUTATIONS)


@pytest.mark.parametrize("mutation", sorted(CASES))
def test_mutation_goes_red(mutation):
    r = _run(mutation)
    out = r.stdout.replace("\\", "/")
    assert r.returncode != 0, f"{mutation}: the literal suite stayed GREEN under the defect"
    for name in CASES[mutation]:
        assert re.search(rf"FAILED .*test_navsim_bars8\.py::{re.escape(name)}", out), f"{mutation}: {name} did not fail\n" + out[-3000:]
