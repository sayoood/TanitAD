"""The GT-collision floor divides by ALL scored samples — the reference compute() convention.

⭐ Pre-registered and CONFIRMED 2026-09-26 (…/raw/nuscenes/PREREG_UNIAD_GT_FLOOR_GAP.md, sha256
08677e75…). Our UniAD GT floor read +10 / +12 / +23 % above PARA-Drive Table 8, growing with horizon.
Hypothesis H1: we divided by VALID timesteps; the reference divides by ALL samples (masked steps count 0).
Recomputed with the reference denominator: 0.3655 / 0.3821 / 0.3655 % vs PARA-Drive 0.35 / 0.38 / 0.35 %,
every horizon within 5 %, and the valid fraction fell 0.950 → 0.900 → 0.850 as H1 predicted.

Expectations are hand-countable LITERALS on a 4-sample toy, never values computed by the code under test.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TE = os.path.dirname(_HERE)
_REPO = os.path.dirname(_TE)
for _p in (os.path.join(_REPO, "stack"), _TE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from adapters import nuscenes_planning as NP              # noqa: E402

T = NP.N_FUTURE


def _kernel(valid: np.ndarray, gtcol: np.ndarray) -> "NP.KernelOutput":
    n = valid.shape[0]
    z = np.zeros((n, T), np.int8)
    return NP.KernelOutput(NP.Pipeline.UNIAD, np.zeros((n, T)), z, z.copy(), gtcol.astype(np.int8),
                           valid.astype(bool), np.ones(n, bool))


def test_default_divides_by_ALL_samples_and_the_old_convention_is_kept_labelled():
    """4 samples; sample 3 has no future at the last two steps; one GT collision at the last step.
    Last step: 1 collision / 4 samples = 25.0 %  (reference)  vs  1 / 3 valid = 33.33… % (old)."""
    valid = np.ones((4, T), bool)
    valid[3, T - 2:] = False
    gtcol = np.zeros((4, T), np.int8)
    gtcol[0, T - 1] = 1
    m = NP.per_timestep_means(_kernel(valid, gtcol))
    assert m["gt_collision_box_pct"][T - 1] == pytest.approx(25.0)
    assert m["gt_collision_box_pct_valid_steps_only"][T - 1] == pytest.approx(100.0 / 3.0)
    assert list(m["gt_valid_n"]) == [4] * (T - 2) + [3, 3]


def test_an_invalid_step_can_never_add_a_collision():
    """A masked (255-filled) step contributes 0 to the numerator under BOTH conventions."""
    valid = np.ones((2, T), bool)
    valid[1, T - 1] = False
    gtcol = np.zeros((2, T), np.int8)
    gtcol[1, T - 1] = 1                                  # a "collision" on an INVALID step
    m = NP.per_timestep_means(_kernel(valid, gtcol))
    assert m["gt_collision_box_pct"][T - 1] == 0.0
    assert m["gt_collision_box_pct_valid_steps_only"][T - 1] == 0.0


def test_with_every_step_valid_the_two_conventions_COINCIDE():
    """⭐ Why the change is safe for VAD / ST-P3: they score only full-future samples, so nothing is
    masked and both denominators equal n. (MEASURED: VAD's GT floor is unchanged by this fix.)"""
    valid = np.ones((5, T), bool)
    gtcol = np.zeros((5, T), np.int8)
    gtcol[2, 3] = 1
    m = NP.per_timestep_means(_kernel(valid, gtcol))
    assert np.array_equal(m["gt_collision_box_pct"], m["gt_collision_box_pct_valid_steps_only"])
    assert m["gt_collision_box_pct"][3] == pytest.approx(20.0)


def test_MUTATION_the_conventions_differ_exactly_when_steps_are_missing():
    """If this ever reads equal on a masked input, the default has silently reverted to valid-only."""
    valid = np.ones((4, T), bool)
    valid[0, :] = [True, True, True, False, False, False]
    gtcol = np.zeros((4, T), np.int8)
    gtcol[1, 5] = 1
    m = NP.per_timestep_means(_kernel(valid, gtcol))
    assert m["gt_collision_box_pct"][5] != m["gt_collision_box_pct_valid_steps_only"][5]
