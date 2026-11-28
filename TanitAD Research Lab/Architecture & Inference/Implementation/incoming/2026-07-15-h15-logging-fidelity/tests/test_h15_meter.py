"""Standalone tests for the H15 accumulation-window meter.

Pins the exact false-'dark-edge' failure the meter fixes: a last-micro sample
reads 0.0 while the window actually trained the imagination edge, whereas the
meter's ``h15`` field reports > 0 whenever ANY micro fired.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h15_meter import H15Meter  # noqa: E402


def test_window_with_some_fires_is_never_falsely_zero():
    """The failure mode: accum=4, last micro masked (0.0), but 2 micros fired.
    Old logger would show h15=0.0; the meter's mean-over-window is > 0."""
    m = H15Meter()
    for v in [0.61, 0.0, 0.55, 0.0]:      # last micro is 0.0 (the artifact case)
        m.update(v)
    out = m.log()
    assert out["h15"] > 0.0                # <-- can't falsely read dark
    assert out["h15"] == pytest.approx((0.61 + 0.55) / 4, abs=1e-6)   # incl zeros
    assert out["h15_fired"] == pytest.approx((0.61 + 0.55) / 2, abs=1e-6)  # cond.
    assert out["h15_fire_frac"] == pytest.approx(0.5)


def test_all_masked_window_reports_true_zero_no_div0():
    """A genuinely idle step (all micros masked) reports 0.0 cleanly — this is
    the ~6% true-idle case, distinguishable from the artifact via fire_frac=0."""
    m = H15Meter()
    for _ in range(4):
        m.update(0.0)
    out = m.log()
    assert out["h15"] == 0.0
    assert out["h15_fired"] == 0.0         # no ZeroDivisionError
    assert out["h15_fire_frac"] == 0.0


def test_all_fired_window():
    m = H15Meter()
    for v in [0.5, 0.7, 0.6, 0.4]:
        m.update(v)
    out = m.log()
    assert out["h15"] == pytest.approx(0.55, abs=1e-6)
    assert out["h15_fired"] == pytest.approx(0.55, abs=1e-6)  # == mean, all fired
    assert out["h15_fire_frac"] == pytest.approx(1.0)


def test_fire_frac_tracks_mask_prob_over_many_windows():
    """Over many accum windows the aggregate fire_frac should approximate
    mask_prob — the field that would catch a genuinely dark edge (frac -> 0)."""
    import random
    rng = random.Random(0)
    mask_prob = 0.5
    fracs = []
    for _ in range(500):
        m = H15Meter()
        for _ in range(4):
            m.update(0.6 if rng.random() < mask_prob else 0.0)
        fracs.append(m.log()["h15_fire_frac"])
    assert abs(sum(fracs) / len(fracs) - mask_prob) < 0.05


def test_update_is_chainable_and_counts():
    m = H15Meter()
    m.update(0.1).update(0.0).update(0.2)
    assert m.n == 3 and m.fired == 2
    assert m.log()["h15"] == pytest.approx(0.3 / 3, abs=1e-6)


def test_empty_meter_is_safe():
    """A meter that never saw a micro (defensive) reports zeros, no crash."""
    out = H15Meter().log()
    assert out == {"h15": 0.0, "h15_fired": 0.0, "h15_fire_frac": 0.0}
