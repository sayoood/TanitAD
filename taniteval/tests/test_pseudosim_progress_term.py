"""⛔ THE VERSIONED EGO-PROGRESS TERM — and the proof that it CAN FAIL.

THE DEFECT (MEASURED 2026-07-28, …/2026-07-28-tactical-action-input/)
---------------------------------------------------------------------
``ego_progress = clamp(along/human, 0, 1)`` charges **NOTHING** for over-travel.
v1's tactical plan over-travels on **48.80 %** of windows (p95 ratio **2.430x**),
so an intervention that cut along-track RMS **8.799 -> 1.557 m (5.65x)** and
lifted the +-5 % distance hit-rate **15.00 % -> 58.93 %** moved the composite
**+0.0078 [-0.0110, +0.0260], n.s.**

It is BLINDNESS, not low power: the identical estimator on the identical 15,981
rows SEPARATES a **3.36x DEGRADATION** of the same axis (REF-C-XL's trained-`v0`
ablation, **-0.0332 [-0.0433, -0.0243]**, replicated at base scale **-0.0461**).

Every rule below is exercised in the direction that makes it FAIL as well as the
one that makes it pass.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from taniteval import pseudosim as PS


def _pw(plan_along, human_along, n=64):
    """Rows whose along-track ratio is exactly ``plan_along / human_along``.

    dyaw = dlat = 0 and ref_yaw = 0, so ``_cross_and_along`` reduces to the
    identity and the ratio is exactly the constructed one — no model, no GPU."""
    Hh = 20
    traj = torch.zeros(n, Hh, 2)
    traj[:, :, 0] = torch.linspace(0, 1, Hh)[None, :] * float(plan_along)
    ref = torch.zeros(n, Hh + 1, 2)
    ref[:, :, 0] = torch.linspace(0, 1, Hh + 1)[None, :] * float(human_along)
    return {"traj": traj, "ref_path": ref, "ref_yaw": torch.zeros(n),
            "v0": torch.full((n,), 10.0), "pt_dyaw": torch.zeros(n),
            "pt_dlat": torch.zeros(n), "pt_dlon": torch.zeros(n),
            "eid": [str(i % 8) for i in range(n)]}


# --------------------------------------------------------------------------- #
# 1. ⛔ THE BLINDNESS, REPRODUCED — the published term scores 2x AS PERFECT     #
# --------------------------------------------------------------------------- #
def test_the_PUBLISHED_term_is_BLIND_to_over_travel():
    """⛔ THE FAILING VALUE OF THE OLD METRIC, on 2 lines of arithmetic.

    A plan that travels TWICE as far as the human scores IDENTICALLY to a
    perfect one. This is why a 5.65x along-track fix was worth +0.0078 n.s."""
    perfect = PS.score_windows(_pw(25.0, 25.0),
                               progress_term="clamp_v1")["ego_progress"]
    double = PS.score_windows(_pw(50.0, 25.0),
                              progress_term="clamp_v1")["ego_progress"]
    assert np.nanmean(perfect) == pytest.approx(1.0, abs=1e-5)
    assert np.nanmean(double) == pytest.approx(1.0, abs=1e-5), (
        "clamp_v1 must be reproduced EXACTLY, blindness included — the old "
        "value has to stay computable")
    assert np.nanmean(double) - np.nanmean(perfect) == pytest.approx(0.0,
                                                                    abs=1e-6)


def test_the_TWO_SIDED_term_SEES_the_over_travel_the_old_one_missed():
    """⭐ THE FIX, on the same rows: 2x over-travel is now scored 0."""
    perfect = PS.score_windows(_pw(25.0, 25.0),
                               progress_term="twosided_v2")["ego_progress"]
    double = PS.score_windows(_pw(50.0, 25.0),
                              progress_term="twosided_v2")["ego_progress"]
    assert np.nanmean(perfect) == pytest.approx(1.0, abs=1e-5)
    assert np.nanmean(double) == pytest.approx(0.0, abs=1e-5)


def test_twosided_is_IDENTICAL_to_clamp_v1_on_every_under_travelling_row():
    """⭐ THE JUSTIFICATION FOR THE SHAPE: it is a STRICT REFINEMENT.

    For every ratio <= 1 the two terms are bit-identical, so the published
    term's under-travel half is preserved exactly and the change is purely
    ADDITIVE INFORMATION on the side the old term could not see. That is what
    makes it the minimum edit rather than a new metric with a new opinion."""
    r = torch.linspace(0.0, 1.0, 4001)
    a = PS.progress_from_ratio(r, "clamp_v1")
    b = PS.progress_from_ratio(r, "twosided_v2")
    assert torch.allclose(a, b, atol=0.0, rtol=0.0)


def test_the_two_terms_DIVERGE_above_ratio_1_and_the_divergence_is_the_point():
    r = torch.linspace(1.0 + 1e-6, 3.0, 1000)
    a = PS.progress_from_ratio(r, "clamp_v1")
    b = PS.progress_from_ratio(r, "twosided_v2")
    assert bool((a >= b).all())
    assert float((a - b).max()) == pytest.approx(1.0, abs=1e-3)


def test_the_two_sided_term_is_symmetric_at_the_default_weight():
    """w = 1.0: r = 1-d and r = 1+d score the same. The MINIMUM-ASSUMPTION
    shape, chosen because this surface has no collision gate and therefore
    cannot measure which side is more dangerous."""
    d = torch.tensor([0.05, 0.2, 0.5, 0.9])
    lo = PS.progress_from_ratio(1.0 - d, "twosided_v2")
    hi = PS.progress_from_ratio(1.0 + d, "twosided_v2")
    assert torch.allclose(lo, hi, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. THE ASYMMETRY GRID — two-sided does NOT mean symmetric                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("term,w", [("twosided_asym_w0p5", 0.5),
                                    ("twosided_v2", 1.0),
                                    ("twosided_asym_w1p5", 1.5),
                                    ("twosided_asym_w2", 2.0),
                                    ("twosided_asym_w3", 3.0)])
def test_the_over_travel_slope_is_exactly_w(term, w):
    """The sensitivity grid must actually vary the thing it claims to vary."""
    r = torch.tensor([1.0, 1.1])
    s = PS.progress_from_ratio(r, term)
    assert float(s[0] - s[1]) == pytest.approx(0.1 * w, abs=1e-6)
    # ...and the UNDER-travel slope is 1.0 for every member of the grid
    ru = torch.tensor([1.0, 0.9])
    su = PS.progress_from_ratio(ru, term)
    assert float(su[0] - su[1]) == pytest.approx(0.1, abs=1e-6)


def test_every_term_stays_in_the_unit_interval():
    r = torch.linspace(-5.0, 20.0, 5001)
    for name in PS.PROGRESS_TERMS:
        s = PS.progress_from_ratio(r, name)
        assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0, name


# --------------------------------------------------------------------------- #
# 3. THE VERSIONING — a silent redefinition must be impossible                  #
# --------------------------------------------------------------------------- #
def test_the_metric_id_carries_the_term():
    assert PS.metric_id("clamp_v1") == "PSS_recovery_progress@clamp_v1"
    assert PS.metric_id("twosided_v2") == "PSS_recovery_progress@twosided_v2"
    assert PS.metric_id("clamp_v1") != PS.metric_id("twosided_v2")


def test_the_default_is_the_FIXED_term():
    """A capability that must be passed explicitly is a capability nobody
    passes — that is the E1 bug found by the same report."""
    assert PS.PROGRESS_TERM_DEFAULT == "twosided_v2"
    assert PS.PROGRESS_TERM_PUBLISHED == "clamp_v1"


def test_the_published_term_is_STILL_REACHABLE_and_reproduces_exactly():
    """The old value must remain computable and identifiable, forever."""
    sc = PS.score_windows(_pw(50.0, 25.0), progress_term="clamp_v1")
    assert sc["_progress_term"] == "clamp_v1"
    assert np.nanmean(sc["ego_progress"]) == pytest.approx(1.0, abs=1e-5)


def test_composite_name_follows_score_windows_without_being_told():
    """A composite built from `clamp_v1` scores must not be labelled v2."""
    sc = PS.score_windows(_pw(50.0, 25.0), progress_term="clamp_v1")
    comps = {k: sc[k] for k in ("ego_progress", "recovery", "comfort")}
    comps["_progress_term"] = sc["_progress_term"]
    ranges = PS.discriminative_range(
        {k: comps[k] for k in ("ego_progress", "recovery", "comfort")})
    ranges["ego_progress"]["admissible"] = True         # force one admissible
    comp = PS.composite(comps, ranges)
    assert comp["name"] == "PSS_recovery_progress@clamp_v1"


def test_an_UNKNOWN_term_is_REFUSED_not_defaulted():
    """⛔ THE FAILING VALUE for the version parser. Falling back to a default on
    a typo would produce a number under the wrong metric id — the exact defect."""
    with pytest.raises(PS.UnknownProgressTerm, match="unknown progress_term"):
        PS.progress_from_ratio(torch.ones(3), "twosided")
    with pytest.raises(PS.UnknownProgressTerm):
        PS.score_windows(_pw(25.0, 25.0), progress_term="clamp_V1")


def test_the_NaN_mask_is_unchanged_by_the_term():
    """Rows the published term excluded (human displacement <= 0.5 m) must stay
    excluded, or the two terms are scored on different row sets and no paired
    delta between them is valid."""
    pw = _pw(0.2, 0.2)                                  # human moves 0.2 m
    a = PS.score_windows(pw, progress_term="clamp_v1")["ego_progress"]
    b = PS.score_windows(pw, progress_term="twosided_v2")["ego_progress"]
    assert np.isnan(a).all() and np.isnan(b).all()
    pw2 = _pw(25.0, 25.0)
    a2 = PS.score_windows(pw2, progress_term="clamp_v1")["ego_progress"]
    b2 = PS.score_windows(pw2, progress_term="twosided_v2")["ego_progress"]
    assert np.array_equal(np.isnan(a2), np.isnan(b2))


def test_the_raw_unclamped_ratio_is_still_published_under_both_terms():
    """The quantity the metric throws away is the artifact behind the finding;
    it must stay in the dump regardless of which term is scored."""
    for term in ("clamp_v1", "twosided_v2"):
        sc = PS.score_windows(_pw(50.0, 25.0), progress_term=term)
        assert np.nanmean(sc["ego_progress_raw_ratio"]) == pytest.approx(
            2.0, abs=1e-3)
