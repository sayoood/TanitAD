"""WP-D — the BEV auxiliary target, head and loss.

⛔⛔ **EVERY EXPECTATION IN THIS FILE IS A LITERAL, AND NO CROSS-CHECK IS THE
PRODUCER'S OWN DERIVATION RE-RUN.** (CLAUDE.md `e4af94f`: four measured instances
in one night of a check that shared the defect it checked for and was therefore
green forever.) The three admissible forms are used and labelled:

* **ANALYTIC** — the geometry is computed by hand from the road/optics
  statement, not from the code. *"20 columns over 120°, column 0 at +60° on the
  LEFT"* fixes ``+60 -> 0``, ``0 -> 10``, ``+30 -> 5``, ``-59.999 -> 19``. A
  4.5 x 2.0 m car at (30 m, 0) occupies **exactly four cells**, and the four
  indices are derived from its footprint bounds and the bin width.
* **INDEPENDENTLY AUTHORED** — :func:`_inside_rect_reference` re-derives the
  footprint predicate from the definition of an oriented rectangle. It never
  calls ``bev_raster`` or ``bev_aux``.
* **MUTATION** — three defects that really happened in this programme are
  re-introduced and each must move a pinned literal: the **MIRRORED WORLD**
  (`E-DEC-18`'s build measured the ego-frame convention precisely because a sign
  error there does not crash and does not show in a loss curve), the **PINHOLE
  FOV** (92.641° where the truth is 120°, the retracted 2026-08-21 error), and
  the **TWO-STATE MERGE** (supervising occluded space as free — the third-state
  defect, on its third appearance in this programme).

⚠️ ``test_agrees_with_psg_targets_azimuth_column`` is labelled a CONSISTENCY
check on purpose: two implementations of ONE formula agreeing measures
divergence, not correctness. The correctness statements are the analytic ones
above it.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tanitad.data.bev_aux import (  # noqa: E402
    OCCLUSION_MODES, PINHOLE_TRAP_DEG, POLAR_DEFAULT, PolarBEVSpec,
    azimuth_column, build_target, polar_occupancy, range_bin, sample_lattice,
    shadow_mask, target_census)
from tanitad.refs.refc import RefCModel, param_breakdown, refc_smoke_config  # noqa: E402
from tanitad.refs.refc_bev_aux import (  # noqa: E402
    LOSS_PARITY_BAND, BEVAuxConfig, BEVAuxHead, assert_loss_parity,
    bev_aux_loss, bev_aux_metrics, constant_control_ap)

S = POLAR_DEFAULT                       # n_az 20, n_rng 24, 60 m, 120°

#: The reference scene: one 4.5 x 2.0 m car, centre 30 m dead ahead, heading
#: along +x. Its footprint is x in [27.75, 32.25], y in [-1.0, +1.0] — both
#: bounds are arithmetic on the stated size, not a code output.
CAR_AHEAD = [{"cx": 30.0, "cy": 0.0, "yaw": 0.0, "l": 4.5, "w": 2.0}]

#: ⭐ ANALYTIC. Range bins are 2.5 m: [27.75, 32.25] spans bin 11 ([27.5, 30.0))
#: and bin 12 ([30.0, 32.5)) and NO OTHER (bin 10 ends at 27.5 < 27.75; bin 13
#: starts at 32.5 > 32.25). Azimuth columns are 6.0°: the optical axis az = 0
#: falls on the 9|10 BOUNDARY, so a car dead ahead straddles columns 9 and 10
#: and there is NO CENTRE COLUMN — the same geometric property `E-DEC-18`'s
#: suite found at 8 columns, re-derived here at 20.
CAR_AHEAD_CELLS = {(11, 9), (11, 10), (12, 9), (12, 10)}


def _inside_rect_reference(px, py, cx, cy, yaw, length, width) -> bool:
    """⭐ INDEPENDENTLY AUTHORED reference: is ``(px, py)`` inside the oriented
    rectangle? Written from the definition — translate to the agent's centre,
    rotate by ``-yaw`` into the agent's own axes, compare against half-extents —
    and it calls nothing under test."""
    dx, dy = px - cx, py - cy
    c, s = math.cos(-yaw), math.sin(-yaw)
    lon = dx * c - dy * s
    lat = dx * s + dy * c
    return abs(lon) <= 0.5 * length and abs(lat) <= 0.5 * width


# =====================================================================
# 1. GEOMETRY — analytic literals
# =====================================================================

def test_azimuth_column_analytic_literals():
    """⭐ ANALYTIC. 20 columns over 120°; image column 0 is the LEFTMOST, which
    is azimuth +60° because the ego frame is +x forward / +y LEFT."""
    assert azimuth_column(60.0, S) == 0            # far left edge
    assert azimuth_column(54.0, S) == 1            # one 6° step in
    assert azimuth_column(30.0, S) == 5            # 30/6 = 5 steps from the edge
    assert azimuth_column(0.0, S) == 10            # the 9|10 boundary lands on 10
    assert azimuth_column(-0.001, S) == 10
    assert azimuth_column(0.001, S) == 9           # ⭐ NO CENTRE COLUMN
    assert azimuth_column(-59.999, S) == 19        # far right edge
    assert azimuth_column(60.001, S) == -1         # outside the field
    assert azimuth_column(-60.001, S) == -1


def test_azimuth_column_is_left_right_mirror_symmetric():
    """⭐ ANALYTIC identity: reflecting the world about the optical axis must
    map column ``j`` to ``n_az - 1 - j``. This is the statement a MIRRORED
    ego-frame convention violates, and it holds for no other column map."""
    for az in (0.5, 3.0, 17.0, 31.5, 59.0):
        assert (azimuth_column(az, S) + azimuth_column(-az, S)
                == S.n_az - 1), az


def test_range_bin_analytic_literals():
    """⭐ ANALYTIC. 24 bins over 60 m = 2.5 m each; bin i covers [2.5i, 2.5(i+1))."""
    assert range_bin(0.0, S) == 0
    assert range_bin(2.4999, S) == 0
    assert range_bin(2.5, S) == 1
    assert range_bin(27.75, S) == 11
    assert range_bin(30.0, S) == 12
    assert range_bin(59.999, S) == 23
    assert range_bin(60.0, S) == -1                # r_max is EXCLUSIVE
    assert range_bin(-1.0, S) == -1


def test_agrees_with_psg_targets_azimuth_column():
    """⚠️ CONSISTENCY, NOT CORRECTNESS. ``psg_targets.azimuth_column`` is the
    programme's existing scalar implementation of the same formula. Agreement
    proves the two have not DIVERGED; it cannot prove either is right, because
    a shared defect would be invisible. The correctness claims are the analytic
    tests above.

    ⚠️ **Both sides must be handed the SAME float.** ``psg_targets`` takes
    ``(cx, cy)`` and re-derives the angle, and the ``degrees(atan2(sin a, cos a))``
    round-trip is not the identity: at ``a = 48.0``, exactly a 6° bin boundary, it
    returns 48.00000000000001 and the two implementations then land one column
    apart. That is a real property of any FLOOR on a boundary, not a divergence —
    which is why the angle is derived once here and passed to both.
    """
    from tanitad.data.psg_targets import azimuth_column as psg_col
    for a in np.linspace(-70.0, 70.0, 281):
        cx, cy = math.cos(math.radians(a)), math.sin(math.radians(a))
        az = math.degrees(math.atan2(cy, cx))       # the SAME float, both sides
        mine = azimuth_column(az, S)
        theirs = psg_col(math.cos(math.radians(az)), math.sin(math.radians(az)),
                         n_cols=S.n_az, hfov_deg=S.hfov_deg)
        assert mine == (-1 if theirs is None else theirs), (a, az)


def test_no_cell_of_the_polar_grid_is_out_of_field():
    """⭐ The property that makes the polar space the honest one here: the
    Cartesian [120, 64] raster loses 590 of 7,680 cells to the camera field
    (``bev_raster.fov_mask``, MEASURED 92.32 % in-field), while EVERY polar cell
    is inside by construction — so ``OCC_IGNORE`` never has to carry
    out-of-field, and the only unobservable states left are NO_LABEL and
    occlusion."""
    x, y = sample_lattice(S)
    az = np.degrees(np.arctan2(y, x))
    assert float(np.abs(az).max()) <= S.hfov_deg / 2.0 + 1e-9
    assert float(np.hypot(x, y).max()) < S.r_max_m


# =====================================================================
# 2. THE TARGET — analytic literal + independent reference
# =====================================================================

def test_car_ahead_occupies_exactly_four_cells_literal():
    """⭐ ANALYTIC LITERAL, written out rather than computed from the code."""
    occ = polar_occupancy(CAR_AHEAD, S)
    got = {(int(i), int(j)) for i, j in np.argwhere(occ > 0.0)}
    assert got == CAR_AHEAD_CELLS, got
    assert occ.dtype == np.float32
    assert occ.shape == (24, 20)


def test_every_occupied_cell_is_confirmed_by_an_independent_predicate():
    """⭐ INDEPENDENTLY AUTHORED. A cell is occupied iff at least one of its
    sub-samples is inside a footprint by :func:`_inside_rect_reference`, which
    calls nothing under test. Run on a scene of five agents at deliberately
    awkward poses (off-axis, rotated, straddling the field edge)."""
    scene = [
        {"cx": 30.0, "cy": 0.0, "yaw": 0.0, "l": 4.5, "w": 2.0},
        {"cx": 12.0, "cy": 6.5, "yaw": 0.4, "l": 5.0, "w": 2.2},
        {"cx": 45.0, "cy": -9.0, "yaw": -0.9, "l": 12.0, "w": 2.6},
        {"cx": 6.0, "cy": -9.5, "yaw": 1.2, "l": 4.0, "w": 1.9},   # near edge
        {"cx": 55.0, "cy": 3.0, "yaw": 0.0, "l": 4.4, "w": 1.8},
    ]
    occ = polar_occupancy(scene, S)
    px, py = sample_lattice(S)
    ref = np.zeros(S.shape, dtype=bool)
    for i in range(S.n_rng):
        for j in range(S.n_az):
            for k in range(px.shape[2]):
                if any(_inside_rect_reference(px[i, j, k], py[i, j, k],
                                              a["cx"], a["cy"], a["yaw"],
                                              a["l"], a["w"]) for a in scene):
                    ref[i, j] = True
                    break
    assert np.array_equal(occ > 0.0, ref)
    assert int(ref.sum()) > 0, "reference found nothing — the scene is wrong"


def test_footprint_reaching_into_the_field_is_kept_even_when_its_centre_is_not():
    """⭐ Strictly better than the join's per-agent ``occ`` flag, which grades
    the agent CENTRE only. A long vehicle centred just outside +60° whose body
    crosses the boundary must still occupy cells."""
    az = math.radians(62.0)                      # centre OUTSIDE the field
    r = 14.0
    a = [{"cx": r * math.cos(az), "cy": r * math.sin(az),
          "yaw": az, "l": 12.0, "w": 2.6}]
    assert azimuth_column(62.0, S) == -1         # its centre has no column
    assert float(polar_occupancy(a, S).sum()) > 0.0


def test_empty_scene_is_all_free_and_fully_supervised():
    occ, mask = build_target([], labelled=True, spec=S)
    assert float(occ.sum()) == 0.0
    assert int(mask.sum()) == S.n_rng * S.n_az   # labelled-clear IS a label


# =====================================================================
# 3. MUTATIONS — each re-introduces a real historical defect, and must move
#    a pinned literal. A mutation that leaves the literal intact means the
#    literal was not testing what it claims to test.
# =====================================================================

def test_MUTATION_mirrored_world_moves_every_column():
    """⛔ MUTATION 1 — THE MIRRORED WORLD (+y LEFT flipped to +y right).

    MEASURED provenance: the ego-frame convention is +x forward / +y LEFT, from
    the parked-car experiment (``bev_raster.py`` / ``build_obstacle_join.py:15``).
    `E-DEC-18`'s build stated the reason for measuring it rather than assuming:
    *"a sign error here does not crash and does not show in a loss curve, it
    teaches a MIRRORED world."*

    ⭐ The invariant is EXACT and needs no fixture-specific literal: the sample
    lattice is symmetric about the optical axis (``n_az`` is even and the
    sub-sample offsets are symmetric, so ``az(n_az-1-j, sub_az-1-m) = -az(j, m)``
    identically), therefore reflecting the world in ``y`` must reflect the target
    in its column axis, **bit-for-bit**. A mirrored convention satisfies that
    identity while placing every agent on the wrong side.

    An off-axis car at (30, +5) lands in column **8** — ANALYTIC: column 8 spans
    bearings (6°, 12°], and the car's footprint (x in [27.75, 32.25],
    y in [4, 6]) contains bearings 7.1°-12.2° at those radii, of which only
    column 8's band is sampled inside the box. Mirroring must put it in
    ``19 - 8 = 11``.
    """
    car = [{"cx": 30.0, "cy": 5.0, "yaw": 0.0, "l": 4.5, "w": 2.0}]
    cols = {int(j) for _, j in np.argwhere(polar_occupancy(car, S) > 0)}
    assert cols == {8}, cols
    mirrored = [dict(a, cy=-a["cy"]) for a in car]
    cols_m = {int(j) for _, j in np.argwhere(polar_occupancy(mirrored, S) > 0)}
    assert cols_m == {11}, cols_m
    assert cols_m == {S.n_az - 1 - c for c in cols}
    assert cols_m != cols, "the mutation did not move the literal — RED FAILED"

    # The exact reflection identity, on an asymmetric multi-agent scene.
    scene = [{"cx": 30.0, "cy": 5.0, "yaw": 0.3, "l": 4.5, "w": 2.0},
             {"cx": 12.0, "cy": -8.0, "yaw": -0.7, "l": 6.0, "w": 2.4}]
    a = polar_occupancy(scene, S)
    b = polar_occupancy([dict(x, cy=-x["cy"], yaw=-x["yaw"]) for x in scene], S)
    assert np.array_equal(b, a[:, ::-1])          # bit-for-bit reflection
    assert not np.array_equal(b, a), "the scene is symmetric — pick another"


def test_MUTATION_pinhole_fov_deletes_the_near_lateral_band():
    """⛔ MUTATION 2 — THE PINHOLE FOV (92.641° instead of the true 120°).

    The retracted 2026-08-21 error: on a CYLINDRICAL projection the column is
    linear in azimuth and the field is 120° (the rig is literally named
    ``camera_front_wide_120fov``); ``2*atan((W/2)/f)`` gives 92.641° and looks
    entirely plausible. Under the wrong field EVERY agent between 46.32° and 60°
    of bearing is silently DELETED from the target — that is the near-lateral
    band where cut-ins live, so the defect removes exactly the agents a planner
    most needs.
    """
    bad = PolarBEVSpec(n_az=S.n_az, n_rng=S.n_rng, r_max_m=S.r_max_m,
                       hfov_deg=PINHOLE_TRAP_DEG)
    az = math.radians(52.0)                      # inside 120°, outside 92.641°
    r = 11.0
    cutin = [{"cx": r * math.cos(az), "cy": r * math.sin(az),
              "yaw": az, "l": 4.5, "w": 2.0}]
    assert float(polar_occupancy(cutin, S).sum()) > 0.0
    assert float(polar_occupancy(cutin, bad).sum()) == 0.0, \
        "the pinhole spec kept the cut-in — RED FAILED"
    # And it moves a mid-field agent's column too, not only the edge.
    mid = [{"cx": 20.0, "cy": 15.0, "yaw": 0.0, "l": 4.5, "w": 2.0}]
    c_true = {int(j) for _, j in np.argwhere(polar_occupancy(mid, S) > 0)}
    c_bad = {int(j) for _, j in np.argwhere(polar_occupancy(mid, bad) > 0)}
    assert c_true and c_bad and c_true != c_bad, (c_true, c_bad)


def test_MUTATION_two_state_merge_supervises_occluded_space_as_free():
    """⛔ MUTATION 3 — THE TWO-STATE MERGE (``occlusion="none"``).

    This is the defect WP-A flagged on WP-D's critical path: *"a scored negative
    still MERGES seen-and-empty with agent-occluded"*, and supervising "empty" on
    an occupied-but-occluded cell teaches the trunk that occluded space is free.
    ⭐ It is shipped as a NAMED ARM rather than as patched-in defect code, so the
    regression the gate must catch is a thing anyone can run.

    Scene: a bus at 10 m dead ahead, a car at 40 m directly behind it. Under
    ``"mask"`` nothing beyond the bus is supervised. Under ``"none"`` the car's
    cells are supervised as OCCUPIED (asking a monocular head to hallucinate)
    AND the empty cells between and beyond are supervised as FREE.
    """
    scene = [{"cx": 10.0, "cy": 0.0, "yaw": 0.0, "l": 12.0, "w": 2.6},
             {"cx": 40.0, "cy": 0.0, "yaw": 0.0, "l": 4.5, "w": 2.0}]
    occ_m, mask_m = build_target(scene, spec=S, occlusion="mask")
    occ_n, mask_n = build_target(scene, spec=S, occlusion="none")

    assert np.array_equal(occ_m, occ_n)          # ONE variable: only the mask
    assert int((~mask_n).sum()) == 0             # two-state: nothing ignored
    assert int((~mask_m).sum()) > 0

    # THE GATE: no supervised cell may lie beyond the nearest occluder.
    def n_supervised_behind_occluder(occ, mask):
        bad = 0
        for j in range(S.n_az):
            nz = np.flatnonzero(occ[:, j] > 0.0)
            if nz.size == 0:
                continue
            i0 = int(nz[0])
            i1 = i0
            while i1 + 1 < S.n_rng and occ[i1 + 1, j] > 0.0:
                i1 += 1
            bad += int(mask[i1 + 1:, j].sum())
        return bad

    assert n_supervised_behind_occluder(occ_m, mask_m) == 0          # GREEN
    assert n_supervised_behind_occluder(occ_n, mask_n) > 0, \
        "the two-state arm supervised nothing behind the bus — RED FAILED"

    # ...and the hidden car is among the cells the two-state arm gets wrong:
    far_cells = [(i, j) for i, j in np.argwhere(occ_n > 0) if i >= 15]
    assert far_cells, "the far car did not rasterise — the scene is wrong"
    for i, j in far_cells:
        assert not mask_m[i, j]                  # correctly unobservable
        assert mask_n[i, j]                      # wrongly supervised


def test_MUTATION_no_label_treated_as_clear_road():
    """⛔ MUTATION 4 — NO_LABEL read as "road clear".

    The join's own documentation names it: *"An ABSENT (clip, frame) line is
    NO_LABEL (skip+count downstream, never 'road clear'); an EMPTY agents list
    IS a label: labelled clear."* The two states must be distinguishable at the
    target, and they are — by the MASK, not by the occupancy array (both have
    ``occ.sum() == 0``, which is exactly why a check on occupancy alone would be
    permanently green).
    """
    occ_nl, mask_nl = build_target([], labelled=False, spec=S)
    occ_cl, mask_cl = build_target([], labelled=True, spec=S)
    assert float(occ_nl.sum()) == float(occ_cl.sum()) == 0.0     # indistinguishable
    assert int(mask_nl.sum()) == 0                               # ...by the mask
    assert int(mask_cl.sum()) == S.n_rng * S.n_az                # they are not


def test_shadow_is_analytic_for_a_bus_at_a_known_range():
    """⭐ ANALYTIC LITERAL for the two centre columns.

    A 12.0 x 2.6 m bus centred at (10, 0) has footprint x in [4, 16],
    y in [-1.3, +1.3]. Along column 9's innermost sampled bearing (1°) the ray is
    essentially the x-axis, so it is inside for r in [4.0, 16.0] — range bins 1
    ([2.5, 5)) through 6 ([15, 17.5)) — and columns 9 and 10 are mirror images.
    ⇒ the first contiguous occupied run in each is bins **1-6**, the shadow is
    bins **7-23 = 17 cells**, and nothing before bin 7 is shadowed.

    ⚠️ **A Cartesian "wall" is NOT at constant range in a polar grid** — an
    earlier version of this test asserted one bin pair for every column it filled
    and failed, because at 45° the same plane sits at r = x/cos(45°). The wide
    columns here are asserted structurally instead, and the exact literal is
    claimed only where it was derived.
    """
    bus = [{"cx": 10.0, "cy": 0.0, "yaw": 0.0, "l": 12.0, "w": 2.6}]
    occ = polar_occupancy(bus, S)
    sh = shadow_mask(occ, S)
    for j in (9, 10):
        assert set(map(int, np.flatnonzero(occ[:, j] > 0))) == {1, 2, 3, 4, 5, 6}, j
        assert int(sh[:, j].sum()) == 17, (j, int(sh[:, j].sum()))
        assert not sh[:7, j].any()
        assert sh[7:, j].all()
    # left/right symmetry of the whole shadow, since the scene is symmetric
    assert np.array_equal(sh, sh[:, ::-1])
    # structural rule everywhere: the shadow starts exactly one bin past the end
    # of the FIRST contiguous occupied run, and an empty column casts none.
    for j in range(S.n_az):
        nz = np.flatnonzero(occ[:, j] > 0.0)
        if nz.size == 0:
            assert not sh[:, j].any(), j
            continue
        i1 = int(nz[0])
        while i1 + 1 < S.n_rng and occ[i1 + 1, j] > 0.0:
            i1 += 1
        assert not sh[:i1 + 1, j].any() and sh[i1 + 1:, j].all(), j


def test_occluder_is_never_self_occluded():
    """The stated convention: an agent's OWN footprint stays OCCUPIED and
    supervised; the shadow starts at its far edge. Marking a vehicle's far half
    unobservable would delete most of the positive signal to model an effect a
    2.5 m bin cannot resolve."""
    occ, mask = build_target(CAR_AHEAD, spec=S, occlusion="mask")
    for i, j in CAR_AHEAD_CELLS:
        assert occ[i, j] == 1.0 and mask[i, j]


def test_build_target_refuses_an_unknown_occlusion_mode():
    with pytest.raises(ValueError, match="occlusion must be one of"):
        build_target([], spec=S, occlusion="three-state")
    assert OCCLUSION_MODES == ("mask", "none")


def test_census_counts_are_consistent_literals():
    occ, mask = build_target(CAR_AHEAD, spec=S, occlusion="mask")
    c = target_census(occ, mask)
    assert c["n_cells"] == 480
    assert c["n_pos_all"] == 4                       # the analytic literal
    assert c["n_pos_supervised"] == 4                # none of it is self-hidden
    assert c["n_ignored"] == 22                      # 11 bins x 2 columns
    assert c["n_supervised"] == 480 - 22
    assert c["base_rate_supervised"] == 4 / 458


# =====================================================================
# 4. THE HEAD — refusals, shapes, and the gradient path
# =====================================================================

def _bcfg(**kw):
    d = dict(enable=True, kind="col", n_rng=24, n_az=20, d_tok=16, hidden=64,
             w=1.0)
    d.update(kw)
    return BEVAuxConfig(**d)


def test_head_refuses_a_column_count_that_does_not_match_the_map():
    with pytest.raises(ValueError, match="azimuth columns"):
        BEVAuxHead(64, (8, 16), _bcfg(n_az=20))


def test_config_refuses_enable_with_zero_weight():
    """⛔ The ``w_agent`` defect verbatim: a seam declared in ``config.json``
    whose loss term is silently skipped, so the run record claims a lever that
    never entered the gradient."""
    with pytest.raises(ValueError, match="w_agent"):
        BEVAuxConfig(enable=True, w=0.0)
    BEVAuxConfig(enable=False, w=0.0)                # off is fine


def test_config_refuses_an_unknown_head_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        BEVAuxConfig(kind="lss")


@pytest.mark.parametrize("kind", ["col", "xcol"])
def test_head_shapes_and_column_registration(kind):
    h = BEVAuxHead(48, (8, 20), _bcfg(kind=kind))
    out = h(torch.randn(3, 48, 8, 20))
    assert out.shape == (3, 24, 20)
    assert h.n_params() > 0


def test_col_head_has_no_cross_column_leakage():
    """⭐ The ``col`` arm is the CHEAP FLOOR and its whole claim is that azimuth
    needs no learning, because on a cylindrical corpus the column already IS a
    bearing. That claim is only true if the head cannot mix columns — so
    perturbing ONE input column must move exactly ONE output column. ``xcol``
    exists to test whether cross-column reasoning buys anything, and it is the
    arm for which this must NOT hold."""
    torch.manual_seed(0)
    h = BEVAuxHead(48, (8, 20), _bcfg(kind="col")).eval()
    x = torch.randn(1, 48, 8, 20)
    x2 = x.clone()
    x2[:, :, :, 7] += 5.0
    with torch.no_grad():
        d = (h(x) - h(x2)).abs().amax(dim=1)[0]
    assert {int(j) for j in np.flatnonzero(d.numpy() > 0)} == {7}

    torch.manual_seed(0)
    hx = BEVAuxHead(48, (8, 20), _bcfg(kind="xcol")).eval()
    with torch.no_grad():
        dx = (hx(x) - hx(x2)).abs().amax(dim=1)[0]
    assert int((dx > 0).sum()) > 1, "xcol did not mix columns"


def test_detach_trunk_is_a_real_deliberate_regression():
    """⛔ ``detach_trunk=True`` is the arm that proves the aux loss is actually
    REACHING the trunk. With it on, the head trains perfectly well, its own
    metric improves, and the encoder receives nothing — an experiment that would
    read as "the BEV aux does not help the planner" for a reason that has
    nothing to do with the hypothesis."""
    torch.manual_seed(0)
    trunk = torch.nn.Conv2d(9, 48, 3, padding=1)
    occ = torch.zeros(2, 24, 20)
    occ[:, 11, 9] = 1.0
    mask = torch.ones(2, 24, 20, dtype=torch.bool)

    for detach, expect_grad in ((False, True), (True, False)):
        for p in trunk.parameters():
            p.grad = None
        torch.manual_seed(1)
        head = BEVAuxHead(48, (8, 20), _bcfg(detach_trunk=detach))
        fmap = trunk(torch.randn(2, 9, 8, 20))
        out = bev_aux_loss(head(fmap), occ, mask, pos_weight=30.0)
        out["loss"].backward()
        got = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                  for p in trunk.parameters())
        assert got is expect_grad, (detach, got)


# =====================================================================
# 5. THE LOSS AND THE CONTROLS — every one reads a KNOWN VALUE
# =====================================================================

def test_loss_refuses_a_float_mask():
    with pytest.raises(TypeError, match="must be a bool tensor"):
        bev_aux_loss(torch.zeros(1, 24, 20), torch.zeros(1, 24, 20),
                     torch.zeros(1, 24, 20), 30.0)


def test_loss_on_an_all_no_label_batch_is_exactly_zero_and_says_why():
    out = bev_aux_loss(torch.randn(2, 24, 20), torch.zeros(2, 24, 20),
                       torch.zeros(2, 24, 20, dtype=torch.bool), 30.0)
    assert float(out["loss"]) == 0.0
    assert out["n_supervised"] == 0                  # the reason, not a NaN


def test_loss_ignores_masked_cells_exactly():
    """⭐ The mask must REMOVE cells, not down-weight them: changing a logit
    under an IGNORE cell may not change the loss by one bit."""
    torch.manual_seed(0)
    logits = torch.randn(2, 24, 20)
    occ = (torch.rand(2, 24, 20) < 0.03).float()
    mask = torch.rand(2, 24, 20) > 0.2
    a = float(bev_aux_loss(logits, occ, mask, 30.0)["loss"])
    l2 = logits.clone()
    l2[~mask] += 100.0
    b = float(bev_aux_loss(l2, occ, mask, 30.0)["loss"])
    assert a == b


def test_CONTROL_constant_score_reads_exactly_the_base_rate():
    """⭐ KNOWN-VALUE CONTROL. A constant score cannot rank, so its average
    precision is the base rate EXACTLY. ⛔ And note what its ACCURACY would be:
    on the MEASURED corpus base rate of 3.1634 % an all-zero predictor scores
    96.84 % — which is why no headline number here is an accuracy."""
    occ = np.zeros((1, 24, 20), dtype=np.float32)
    occ[0, 11, 9] = occ[0, 11, 10] = occ[0, 12, 9] = occ[0, 12, 10] = 1.0
    mask = np.ones((1, 24, 20), dtype=bool)
    base = constant_control_ap(occ, mask)
    assert base == 4 / 480                                     # the literal
    for const in (-9.0, 0.0, +9.0):
        m = bev_aux_metrics(np.full((1, 24, 20), const, dtype=np.float32),
                            occ, mask)
        assert m["ap"] == pytest.approx(base, abs=1e-12), const
        assert m["base_rate"] == pytest.approx(base, abs=1e-12)
        assert "accuracy" not in m                             # by construction


def test_CONTROL_all_zero_predictor_scores_iou_and_f1_exactly_zero():
    occ = np.zeros((1, 24, 20), dtype=np.float32)
    occ[0, 5, 5] = 1.0
    mask = np.ones((1, 24, 20), dtype=bool)
    m = bev_aux_metrics(np.full((1, 24, 20), -50.0, dtype=np.float32),
                        occ, mask)
    assert m["iou"] == 0.0 and m["f1"] == 0.0
    assert m["ap"] == pytest.approx(1 / 480, abs=1e-12)


def test_CONTROL_perfect_ranker_reads_exactly_one():
    occ = np.zeros((1, 24, 20), dtype=np.float32)
    occ[0, 5, 5] = occ[0, 6, 6] = 1.0
    mask = np.ones((1, 24, 20), dtype=bool)
    m = bev_aux_metrics(occ * 20.0 - 10.0, occ, mask)
    assert m["ap"] == 1.0
    assert m["iou"] == 1.0 and m["f1"] == 1.0


def test_metrics_score_only_supervised_cells():
    occ = np.zeros((1, 24, 20), dtype=np.float32)
    occ[0, 20, 3] = 1.0                    # the ONLY positive, and it is masked
    mask = np.ones((1, 24, 20), dtype=bool)
    mask[0, 20, 3] = False
    m = bev_aux_metrics(np.zeros((1, 24, 20), dtype=np.float32), occ, mask)
    assert m["n_pos"] == 0 and m["n_supervised"] == 479
    assert math.isnan(m["ap"])             # no positives -> undefined, not 0.0


def test_loss_parity_guard_refuses_the_E_DEC_18b_failure_shape():
    """⭐ MEASURED historical defect, turned into a launch-time refusal.
    `E-DEC-18b`: the PSG aux term sat 10-30x above the objective it was meant to
    support, at EVERY weight tested, and destroyed the encoder — visible only
    after the GPU-days were spent."""
    ok = assert_loss_parity(0.1, 0.6, 0.03 / 0.1 * 0.6 / 0.6 * 1.0)
    assert LOSS_PARITY_BAND[0] <= ok["ratio"] <= LOSS_PARITY_BAND[1]
    with pytest.raises(SystemExit, match="E-DEC-18b"):
        assert_loss_parity(1.0, 0.9, 0.03)         # ratio 30.0 — the real one
    with pytest.raises(SystemExit, match="inert in the gradient"):
        assert_loss_parity(1e-4, 0.9, 0.03)        # ratio 0.003 — declared, dead
    with pytest.raises(ValueError):
        assert_loss_parity(1.0, 0.9, 0.0)


# =====================================================================
# 6. REMOVABILITY — proven, not asserted
# =====================================================================

def _models(bev):
    cfg = refc_smoke_config()
    cfg.encoder.image_width = 640            # -> grid (2, 20): 20 azimuth cols
    if bev is not None:
        cfg.bev_aux = bev
    torch.manual_seed(1234)
    return cfg, RefCModel(cfg)


def test_shared_params_bit_identical():
    """⛔⛔ THE ONE-VARIABLE PROOF AT STEP 0. Module construction draws from the
    global RNG, so a head built anywhere but LAST would silently change every
    subsequent module's initial weights — and the "aux on vs aux off" A/B would
    then differ in the SEED as well as in the lever, invisibly, in every log."""
    _, off = _models(None)
    _, on = _models(_bcfg())
    so, sn = off.state_dict(), on.state_dict()
    shared = [k for k in so if k in sn]
    assert len(shared) == len(so) and len(sn) > len(so)
    bad = [k for k in shared if not torch.equal(so[k], sn[k])]
    assert bad == [], bad


def test_planner_output_bit_identical():
    """⛔⛔ THE REMOVABILITY PROOF. Same seed, same input, eval mode: every
    tensor the planner emits must be BIT-identical between the aux-on and
    aux-off builds, and the only difference in the output dict is the extra
    ``bev_logits`` key. This is the PI's binding rule in executable form —
    labels may use agents, INFERENCE IS VISION-ONLY — and the published
    precedent (PhyLatent PSG, "not required by the planner")."""
    cfg, off = _models(None)
    _, on = _models(_bcfg())
    off.eval(), on.eval()
    x = torch.randn(2, cfg.window, cfg.encoder.in_channels, 64, 640)
    torch.manual_seed(7)
    a = off(x)
    torch.manual_seed(7)
    b = on(x)
    assert set(b) - set(a) == {"bev_logits"}
    assert set(a) - set(b) == set()
    for k in a:
        va, vb = a[k], b[k]
        if torch.is_tensor(va):
            assert va.shape == vb.shape, k
            assert torch.equal(va, vb), k
        elif isinstance(va, dict):
            for kk in va:
                assert torch.equal(va[kk], vb[kk]), f"{k}.{kk}"
    assert b["bev_logits"].shape == (2, 24, 20)


def test_param_breakdown_carries_bev_aux_and_still_sums():
    _, off = _models(None)
    _, on = _models(_bcfg())
    bo, bn = param_breakdown(off), param_breakdown(on)
    assert bo["bev_aux"] == 0
    assert bn["bev_aux"] == on.bev_aux_head.n_params() > 0
    for bd in (bo, bn):
        assert sum(v for k, v in bd.items() if k != "total") == bd["total"]
    assert bn["total"] - bo["total"] == bn["bev_aux"]
    # every non-aux line is unchanged — the head added capacity NOWHERE else
    assert {k: v for k, v in bo.items() if k not in ("bev_aux", "total")} == \
           {k: v for k, v in bn.items() if k not in ("bev_aux", "total")}


def test_off_is_not_constructed_at_all():
    """⛔ ``enable=False`` must mean NOT CONSTRUCTED, not constructed-and-idle.
    A disabled-but-present module still consumes RNG at ``__init__`` and still
    lands in ``state_dict``, and both break the two proofs above."""
    _, off = _models(None)
    assert off.bev_aux_head is None
    assert not any(k.startswith("bev_aux_head") for k in off.state_dict())
    _, dis = _models(BEVAuxConfig(enable=False))
    assert dis.bev_aux_head is None
