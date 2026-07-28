"""THE SECOND ONE-SIDED CLAMP — ``recovery``. C45, fixed 2026-07-28.

⛔ Every rule below is driven with **the value that makes it FAIL**, not only
the value that makes it pass. The defect being fixed is precisely a metric that
was only ever exercised in the direction it was built to detect.

The defect, in one line: ``recovery = clamp(1 - r, 0, 1)`` with
``r = |xt_end| / |xt_hold|`` is ``max(1 - r, 0)`` — its ceiling is provably
never active (``r >= 0``), and its FLOOR swallowed **55.65-92.19 %** of every
arm's defined rows, so an injected lateral degradation RAISED the composite on
**8 of 8** injections, both arms, including under a ZERO-MEAN control.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from taniteval import pseudosim as PS

DENSE = torch.linspace(0.0, 50.0, 5001, dtype=torch.float64)
UNDER = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# 1. the published term is BLIND above ratio 1 — the FAILING value             #
# --------------------------------------------------------------------------- #
def test_the_PUBLISHED_recovery_term_cannot_tell_hold_from_catastrophe():
    """⛔ THE DEFECT, as a number.

    A plan that merely fails to recover (``r = 1``) and a plan that ends
    **thirty times** further off the logged path than doing nothing
    (``r = 30``, inside the MEASURED max of 34.1) are scored IDENTICALLY.
    That is the value that makes the old metric fail."""
    pub = PS.recovery_from_ratio(torch.tensor([1.0, 1.5, 3.0, 10.0, 30.0]),
                                 "clamp_v1")
    assert pub.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    # and the same rows under the shipped term are strictly ordered until the
    # (much later) floor
    new = PS.recovery_from_ratio(torch.tensor([1.0, 1.5, 3.0, 10.0, 30.0]))
    assert new[0] > new[1] > new[2]
    assert float(new[0]) == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_a_term_at_its_floor_CANNOT_BE_CHARGED_MORE_and_that_is_the_mechanism():
    """⛔ Zero gradient on the floored half is the whole failure mode.

    Pinned in BOTH directions: the published term has zero gradient on
    ``[1, inf)``; the shipped term has zero gradient only on ``[3, inf)`` — a
    RESIDUAL, not an elimination, and it is asserted rather than glossed."""
    eps = 1e-6
    for r in (1.5, 5.0, 20.0):
        a = PS.recovery_from_ratio(torch.tensor([r]), "clamp_v1")
        b = PS.recovery_from_ratio(torch.tensor([r + eps]), "clamp_v1")
        assert float(a - b) == 0.0, "clamp_v1 must be flat above 1 (the defect)"
    # the shipped term charges at r = 1.5 and r = 2.5 …
    for r in (1.0, 1.5, 2.5):
        a = PS.recovery_from_ratio(torch.tensor([r]))
        b = PS.recovery_from_ratio(torch.tensor([r + 1e-3]))
        assert float(a - b) > 0.0
    # … and is FLAT beyond its own floor. ⚠️ disclosed, not hidden.
    a = PS.recovery_from_ratio(torch.tensor([5.0]))
    b = PS.recovery_from_ratio(torch.tensor([25.0]))
    assert float(a) == float(b) == 0.0


# --------------------------------------------------------------------------- #
# 2. versioning — the old value stays computable AND identifiable              #
# --------------------------------------------------------------------------- #
def test_clamp_v1_is_BIT_identical_to_the_family_member_lin_q0():
    """The published term must sit INSIDE the swept family, exactly.

    Not "equal to tolerance" — bit-identical, over 5001 ratios. The progress
    fix drifted by 2.5e-4 on 36 % of rows when it was first written as
    ``1 - (1 - r)``; the same trap is checked here rather than assumed away."""
    a = PS.recovery_from_ratio(DENSE, "clamp_v1")
    b = PS.recovery_from_ratio(DENSE, "lin_q0")
    assert bool((a == b).all())


def test_the_metric_id_carries_the_recovery_term_only_when_it_is_NOT_published():
    """⭐ Backward compatibility is the load-bearing half.

    Every id published through 2026-07-28 must keep its EXACT string, or every
    pin, log line and report naming it silently stops resolving — while a
    non-published recovery term must ALWAYS be visible in the name."""
    assert PS.metric_id("clamp_v1", "clamp_v1") == "PSS_recovery_progress@clamp_v1"
    assert (PS.metric_id("twosided_v2", "clamp_v1")
            == "PSS_recovery_progress@twosided_v2")
    assert (PS.metric_id("twosided_v2", "twosided_v2")
            == "PSS_recovery_progress@twosided_v2+rec_twosided_v2")
    assert PS.metric_id("twosided_v2", "clamp_v1") != PS.metric_id(
        "twosided_v2", "twosided_v2")
    # ⛔ the default of `metric_id` is the PUBLISHED recovery term, so a caller
    # that names only the progress term cannot accidentally mint the new id.
    assert PS.metric_id("clamp_v1") == "PSS_recovery_progress@clamp_v1"


def test_the_parser_REFUSES_a_typo_instead_of_falling_back():
    with pytest.raises(PS.UnknownRecoveryTerm, match="unknown recovery_term"):
        PS.recovery_from_ratio(torch.tensor([0.5]), "clamp_V1")
    with pytest.raises(PS.UnknownRecoveryTerm):
        PS.metric_id("twosided_v2", "twosided")
    with pytest.raises(PS.UnknownRecoveryTerm):
        PS.score_windows(_pw(), recovery_term="lin_q05")


def test_the_default_alias_TARGET_is_pinned():
    """⛔ The shipped default is an alias; which member it points at is a
    DECISION and must not drift silently. This test is the pin."""
    assert PS.RECOVERY_TERM_DEFAULT == "twosided_v2"
    assert PS.RECOVERY_TERM_DEFAULT_TARGET == "lin_q0p6667"
    assert (PS.RECOVERY_TERMS["twosided_v2"]
            is PS.RECOVERY_TERMS["lin_q0p6667"])
    assert PS.RECOVERY_TERM_PUBLISHED == "clamp_v1"
    # the shipped term reaches 0 at three times the hold error, and says so
    assert float(PS.recovery_from_ratio(torch.tensor([3.0]))) == 0.0
    assert float(PS.recovery_from_ratio(torch.tensor([2.999]))) > 0.0


# --------------------------------------------------------------------------- #
# 3. ⭐ THE IMPOSSIBILITY — why this is not a strict refinement                 #
# --------------------------------------------------------------------------- #
def test_a_STRICT_REFINEMENT_IS_IMPOSSIBLE_and_the_family_shows_it():
    """⛔ The reason this fix does NOT copy ``twosided_v2``'s shape.

    Any bounded ``g: [0, inf) -> [0, 1]`` that agrees with the published
    ``1 - r`` on ``[0, 1]`` has ``g(1) = 0``, and being bounded below by 0 and
    non-increasing it must be CONSTANT on ``[1, inf)`` — i.e. it IS the defect.
    So: the ONLY member of the family that reproduces the published term below
    ratio 1 is the published term itself, and it is flat above 1.
    """
    pub = PS.recovery_from_ratio(UNDER, "clamp_v1")
    agreeing = [name for name in PS.RECOVERY_TERMS
                if bool(torch.allclose(PS.recovery_from_ratio(UNDER, name),
                                       pub, atol=1e-12))]
    assert set(agreeing) == {"clamp_v1", "lin_q0"}, agreeing
    for name in agreeing:
        above = PS.recovery_from_ratio(
            torch.tensor([1.0, 2.0, 7.0, 40.0]), name)
        assert len(set(above.tolist())) == 1, (
            f"{name} agrees with the published term below 1 and is therefore "
            f"forced to be constant above it — that is the impossibility")


def test_the_LINEAR_family_is_AFFINE_in_the_published_term_below_ratio_1():
    """⭐ The closest attainable analogue of a strict refinement, asserted.

    ``g(r) = q + (1 - q) * clamp_v1(r)`` for every ``r <= 1``. Consequences the
    report leans on: no pair of under-recovering rows changes order, and the
    published value is exactly invertible from the new one."""
    pub = PS.recovery_from_ratio(UNDER, "clamp_v1")
    for q, name in ((0.25, "lin_q0p25"), (0.5, "lin_q0p5"),
                    (2.0 / 3.0, "lin_q0p6667"), (0.75, "lin_q0p75")):
        g = PS.recovery_from_ratio(UNDER, name)
        assert torch.allclose(g, q + (1.0 - q) * pub, atol=1e-9), name
        # exact inversion back to the published value
        assert torch.allclose((g - q) / (1.0 - q), pub, atol=1e-9), name
    # ⛔ and the SHARE family is NOT affine — stated as a test so the trade-off
    # is checkable rather than claimed.
    g = PS.recovery_from_ratio(UNDER, "share_q0p5")
    assert not torch.allclose(g, 0.5 + 0.5 * pub, atol=1e-3)


# --------------------------------------------------------------------------- #
# 4. ⛔⛔ THE SELF-REFUTATION — "never saturates" is NOT the fix                 #
# --------------------------------------------------------------------------- #
def test_the_UNSATURATING_share_family_still_pays_for_lateral_degradation():
    """⛔⛔ MY PROVISIONAL DEFAULT, REFUTED BY ITS OWN ACCEPTANCE TEST.

    ``share_q0p5`` never floors on any row of any arm (floor fraction 0.0000)
    and still scored **0 of 8** injections in the correct direction. Two rows
    are enough to show why: degrade BOTH rows' ratio (the mean ratio rises from
    1.00 to 1.30) and the share mean goes **UP**, because its charge rate decays
    like ``r^-2`` and the row near zero is rewarded harder than the row past the
    median is charged. The linear family, whose rate is constant, goes DOWN.

    ⇒ the property that fixes a saturating metric is **a charge rate that does
    not collapse where the data lives**, not the absence of a floor."""
    before = torch.tensor([0.4, 1.6], dtype=torch.float64)
    after = torch.tensor([0.2, 2.4], dtype=torch.float64)     # strictly worse
    assert float(after.mean()) > float(before.mean())

    share = [float(PS.recovery_from_ratio(x, "share_q0p5").mean())
             for x in (before, after)]
    assert share[1] > share[0], (
        "share_q0p5 is expected to FAIL here — this test pins the failure")

    lin = [float(PS.recovery_from_ratio(x, "lin_q0p6667").mean())
           for x in (before, after)]
    assert lin[1] < lin[0]

    pub = [float(PS.recovery_from_ratio(x, "clamp_v1").mean())
           for x in (before, after)]
    assert pub[1] > pub[0], "and the published term fails it too, harder"


def test_the_charge_rate_of_the_linear_family_is_CONSTANT_and_the_share_is_not():
    """The mechanism behind the test above, measured on the slope itself."""
    eps = 1e-6
    def slope(r, name):
        a = PS.recovery_from_ratio(torch.tensor([r], dtype=torch.float64), name)
        b = PS.recovery_from_ratio(torch.tensor([r + eps], dtype=torch.float64),
                                   name)
        return float((a - b) / eps)
    lin = [slope(r, "lin_q0p6667") for r in (0.05, 0.5, 1.0, 2.0, 2.9)]
    assert max(lin) - min(lin) < 1e-3, lin
    sh = [slope(r, "share_q0p5") for r in (0.05, 1.0, 3.0)]
    assert sh[0] > 3.0 * sh[2], sh          # the rate collapses by > 3x
    # ⛔ and the published term's rate is EXACTLY ZERO at the median row
    assert slope(1.2, "clamp_v1") == 0.0


# --------------------------------------------------------------------------- #
# 5. range, monotonicity, and the row set                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(PS.RECOVERY_TERMS))
def test_every_term_stays_in_0_1_and_is_non_increasing(name):
    g = PS.recovery_from_ratio(DENSE, name)
    assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0
    assert float(g[0]) == pytest.approx(1.0, abs=1e-9)
    assert bool((g[1:] <= g[:-1] + 1e-12).all()), f"{name} is not monotone"


def test_the_share_family_REFUSES_a_degenerate_anchor():
    with pytest.raises(ValueError, match="0 < q < 1"):
        PS._recovery_share(0.0)
    with pytest.raises(ValueError):
        PS._recovery_share(1.0)


def _pw(n=48, Hh=20, dpsi_deg=8.0, v0=10.0):
    """A minimal ``score_windows``-shaped record. No model, no corpus.

    ⚠️ Each row gets a DIFFERENT recovery fraction and a different travel
    scale, so ``recovery`` and ``ego_progress`` both have real range: a fixture
    with none of it would make ``composite`` refuse (``VacuousMetric``) and the
    test would be passing on a refusal rather than on the arithmetic.
    Recovery fractions span past 1.0 on purpose, so the fixture contains rows in
    the DIVERGING half — the half the published term cannot see."""
    t = torch.arange(1, Hh + 1, dtype=torch.float32) * PS.DT
    ref = torch.zeros(n, Hh + 1, 2)
    ref[:, :, 0] = torch.cat([torch.zeros(1), v0 * t])[None].expand(n, -1)
    dpsi = float(np.deg2rad(dpsi_deg))
    hold = torch.stack([v0 * t, torch.zeros(Hh)], -1)                 # [Hh, 2]
    rec = torch.stack([np.cos(dpsi) * (v0 * t), -np.sin(dpsi) * (v0 * t)], -1)
    a = torch.linspace(-0.6, 1.4, n)[:, None, None]     # recovery fraction
    s = torch.linspace(0.6, 1.4, n)[:, None, None]      # travel scale
    traj = s * ((1.0 - a) * hold[None] + a * rec[None])
    return {"traj": traj.contiguous(), "ref_path": ref,
            "ref_yaw": torch.zeros(n), "v0": torch.full((n,), v0),
            "pt_dyaw": torch.full((n,), dpsi_deg),
            "pt_dlat": torch.zeros(n), "pt_dlon": torch.zeros(n),
            "eid": [str(i % 8) for i in range(n)]}


def test_the_ROW_SET_is_identical_across_recovery_terms():
    """⛔ Otherwise no paired delta between two terms is valid. The NaN mask is
    ``xt_hold > 0.10`` and belongs to the DENOMINATOR, not to the shape."""
    pw = _pw()
    masks = {}
    for name in sorted(PS.RECOVERY_TERMS):
        rc = PS.score_windows(pw, recovery_term=name)["recovery"]
        masks[name] = np.isfinite(rc)
    ref = masks["clamp_v1"]
    for name, m in masks.items():
        assert bool((m == ref).all()), f"{name} changed the row set"


def test_ego_progress_is_BIT_identical_across_recovery_terms():
    """The change touches ``recovery`` and nothing else — asserted, not assumed.
    A shared-array bug here would contaminate every progress-term verdict."""
    pw = _pw()
    a = PS.score_windows(pw, recovery_term="clamp_v1")["ego_progress"]
    for name in sorted(PS.RECOVERY_TERMS):
        b = PS.score_windows(pw, recovery_term=name)["ego_progress"]
        assert bool((a == b).all()), name


def test_the_raw_ratio_is_EMITTED_because_its_floored_half_is_the_defect():
    sc = PS.score_windows(_pw())
    r = np.asarray(sc["recovery_raw_ratio"], float)
    assert r.shape == np.asarray(sc["recovery"], float).shape
    assert float(np.nanmin(r)) >= 0.0
    assert sc["_recovery_term"] == PS.RECOVERY_TERM_DEFAULT


def test_composite_infers_the_recovery_term_from_score_windows():
    """And a RAW dict with no ``_recovery_term`` key falls back to the PUBLISHED
    term, never to the new default — a caller reproducing an old number must
    not have it renamed to the new metric under them."""
    sc = PS.score_windows(_pw(), progress_term="clamp_v1",
                          recovery_term="clamp_v1")
    comps = {k: sc[k] for k in ("ego_progress", "recovery", "comfort")}
    comps["no_collision"] = comps["ttc"] = None
    comps["_progress_term"] = "clamp_v1"
    comps["_recovery_term"] = "clamp_v1"
    r = PS.discriminative_range(comps)
    assert PS.composite(comps, r)["name"] == "PSS_recovery_progress@clamp_v1"
    bare = {k: v for k, v in comps.items() if not k.startswith("_")}
    assert PS.composite(bare, r)["recovery_term"] == PS.RECOVERY_TERM_PUBLISHED


# --------------------------------------------------------------------------- #
# 6. ⚠️ C45's STANDING CONSEQUENCE — saturation is reported, always            #
# --------------------------------------------------------------------------- #
def test_saturation_WARNS_and_can_fail_to_warn():
    """Driven in both directions: a floored array must warn, a spread one must
    not. A warning that always fires is as useless as one that never does."""
    floored = np.concatenate([np.zeros(900), np.linspace(0.1, 1.0, 100)])
    node = PS.saturation(floored)
    assert node["floor_frac_le_0p001"] == pytest.approx(0.90, abs=1e-9)
    assert node["SATURATION_WARNING"] and "FLOOR" in node["SATURATION_WARNING"]
    spread = np.linspace(0.05, 0.95, 1000)
    assert PS.saturation(spread)["SATURATION_WARNING"] is None
    ceil = np.concatenate([np.ones(900), np.linspace(0.0, 0.9, 100)])
    assert "CEILING" in PS.saturation(ceil)["SATURATION_WARNING"]
    # NaNs are excluded from the fractions but reported via defined_frac
    withnan = np.concatenate([np.full(500, np.nan), np.zeros(500)])
    n = PS.saturation(withnan)
    assert n["defined_frac"] == pytest.approx(0.5)
    assert n["floor_frac_le_0p001"] == pytest.approx(1.0)


def test_discriminative_range_SURFACES_the_floor_fraction_beside_the_score():
    """⛔ It COMPUTED ``floor_frac_le_0p001`` for its whole life and never
    published it beside a level. That is how a term floored on 55-92 % of its
    rows was published twenty times."""
    rng = PS.discriminative_range(
        {"x": np.concatenate([np.zeros(800), np.linspace(0.1, 1.0, 200)])})
    assert "saturation" in rng["x"]
    assert rng["x"]["saturation"]["floor_frac_le_0p001"] == pytest.approx(0.8)
    assert rng["x"]["saturation"]["SATURATION_WARNING"]
    # the GATE is deliberately looser than the WARNING: 0.80 is admissible and
    # still warns. Both facts are asserted so neither can drift into the other.
    assert rng["x"]["admissible"] is True
    assert PS.FLOOR_FRAC_MAX == 0.95 and PS.SATURATION_WARN_FRAC == 0.50


def test_the_composite_publishes_component_saturation():
    sc = PS.score_windows(_pw())
    comps = {k: sc[k] for k in ("ego_progress", "recovery", "comfort")}
    comps["no_collision"] = comps["ttc"] = None
    comps["_recovery_term"] = sc["_recovery_term"]
    comp = PS.composite(comps, PS.discriminative_range(comps))
    assert set(comp["component_saturation"]) == set(comp["weights_admitted"])
    for node in comp["component_saturation"].values():
        assert "floor_frac_le_0p001" in node
