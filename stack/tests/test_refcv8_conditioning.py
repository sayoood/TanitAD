"""refcv8 WP-B -- the pure pieces of `tanitad.refs.refcv8_conditioning`, against LITERAL known values and with a
MUTATION per check (a check that cannot fail is not a check). Analytic tracks are constructed here, never read from
the code under test."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
_STACK = Path(__file__).resolve().parents[1]
if str(_STACK) not in sys.path:
    sys.path.insert(0, str(_STACK))

from tanitad.refs import refcv8_conditioning as C  # noqa: E402

SLOT_T = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)


def _arc(v: float, r: float, side: int) -> torch.Tensor:
    """slots of a constant-speed circle of radius r (side +1 left / -1 right), ego frame."""
    out = []
    for t in SLOT_T:
        th = v * t / r
        out.append([r * math.sin(th), side * r * (1.0 - math.cos(th))])
    return torch.tensor(out)


def _line(speeds) -> torch.Tensor:
    """straight path along x whose segment-average speeds are `speeds` (one per slot segment)."""
    dt = [SLOT_T[0]] + [SLOT_T[i] - SLOT_T[i - 1] for i in range(1, len(SLOT_T))]
    x, out = 0.0, []
    for v, d in zip(speeds, dt):
        x += v * d
        out.append([x, 0.0])
    return torch.tensor(out)


def _tag(p, v0):
    lat, lon = C.tag_paths(p[None, None], torch.tensor([float(v0)]), SLOT_T)
    return C.LAT3[int(lat)], C.LON6[int(lon)]


# ---- vocabulary pinned against the tree ---------------------------------------------------------------------------- #
def test_vocab_is_the_frozen_v7_order() -> None:
    from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7, TACTICAL_LON_ACTIONS_V7
    assert C.LAT_V7 == tuple(TACTICAL_LAT_ACTIONS_V7)
    assert C.LON_V7 == tuple(TACTICAL_LON_ACTIONS_V7)
    assert [C.LAT_V7[i] for i in C.LAT3_V7_IDS] == ["LANE_KEEP", "TURN_L", "TURN_R"]
    assert [C.LON_V7[i] for i in C.LON6_V7_IDS] == ["FOLLOW", "CRUISE", "BRAKE_TO", "CREEP", "HOLD", "ACCELERATE"]
    assert C.PHI_DIMS == 17 and C.N_HYP == 18


# ---- the tagger: analytic tracks ------------------------------------------------------------------------------------ #
def test_tagger_reads_analytic_tracks() -> None:
    assert _tag(_line([10.0] * 8), 10.0) == ("LANE_KEEP", "CRUISE")
    assert _tag(_arc(5.0, 15.0, +1), 5.0) == ("TURN_L", "CRUISE")
    assert _tag(_arc(5.0, 15.0, -1), 5.0) == ("TURN_R", "CRUISE")
    assert _tag(_line([0.0] * 8), 0.0) == ("LANE_KEEP", "HOLD")
    assert _tag(_line([1.5] * 8), 1.5) == ("LANE_KEEP", "CREEP")
    # +1 m/s^2 from 5 m/s: segment speeds = speed at the segment mid-time
    mids = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 4.5, 5.5]
    assert _tag(_line([5.0 + m for m in mids]), 5.0) == ("LANE_KEEP", "ACCELERATE")
    assert _tag(_line([max(10.0 - 1.5 * m, 0.0) for m in mids]), 10.0) == ("LANE_KEEP", "BRAKE_TO")
    # a 90 deg turn inside a path shorter than 5 m is not identifiable -> LANE_KEEP (the R1 literal)
    assert _tag(_arc(0.5, 1.0, +1), 0.5)[0] == "LANE_KEEP"


def _check_left_arc(tagfn) -> None:
    lat, _ = tagfn(_arc(5.0, 15.0, +1)[None, None], torch.tensor([5.0]), SLOT_T)
    assert C.LAT3[int(lat)] == "TURN_L"


def test_tagger_mutation_mirrored_y_goes_red() -> None:
    _check_left_arc(C.tag_paths)

    def mirrored(p, v0, st):
        q = p.clone()
        q[..., 1] = -q[..., 1]
        return C.tag_paths(q, v0, st)
    with pytest.raises(AssertionError):
        _check_left_arc(mirrored)


# ---- constraint labels ------------------------------------------------------------------------------------------- #
def _fut_from_arc(v: float, r: float, side: int, n: int = 60):
    t = torch.arange(1, n + 1, dtype=torch.float32) * 0.1
    th = v * t / r
    f = torch.zeros(1, n, 4)
    f[0, :, 0] = r * torch.sin(th)
    f[0, :, 1] = side * r * (1 - torch.cos(th))
    f[0, :, 2] = side * th
    f[0, :, 3] = v
    return f


def test_constraint_targets_on_an_analytic_arc() -> None:
    v, r = 5.0, 15.0
    gt = _arc(v, r, +1)[None]
    c = C.constraint_targets(gt, torch.ones(1, 8, dtype=torch.bool), torch.zeros(1, 4),
                             _fut_from_arc(v, r, +1), torch.ones(1, 60, dtype=torch.bool))
    # onset: first tick with theta >= 10 deg: v t / r >= 0.17453 -> t >= 0.5236 s -> tick 0.6 s (LITERAL)
    assert abs(float(c["t_onset"]) - 0.6) < 1e-6 and bool(c["t_onset_valid"])
    # terminal (slot 5 s -> 6 s) chord heading = mid-angle of the chord = v * 5.5 / r = 1.8333 rad (LITERAL)
    assert abs(float(c["psi_term"]) - 1.833333) < 1e-4
    # 6-s arc of the 8 slot chords on R 15: sum of 2 R sin(dth/2) over the slot spans = 29.92 m, by hand from the
    # slot spans 0.5 s x4 (dth 1/6 rad) and 1.0 s x4 (dth 1/3 rad): 4 * 2*15*sin(1/12) + 4 * 2*15*sin(1/6)
    exp = 4 * 30 * math.sin(1 / 12) + 4 * 30 * math.sin(1 / 6)
    assert abs(float(c["prog6"]) - exp) < 1e-3 and abs(exp - 29.92) < 0.05
    assert abs(float(c["v_end"]) - 5.0) < 1e-6
    # a straight track has no onset (not a turn)
    f = torch.zeros(1, 60, 4)
    f[0, :, 0] = torch.arange(1, 61) * 1.0
    f[0, :, 3] = 10.0
    c2 = C.constraint_targets(_line([10.0] * 8)[None], torch.ones(1, 8, dtype=torch.bool), torch.zeros(1, 4), f,
                              torch.ones(1, 60, dtype=torch.bool))
    assert not bool(c2["t_onset_valid"]) and abs(float(c2["prog6"]) - 60.0) < 1e-4


# ---- posteriors, phi, modulation --------------------------------------------------------------------------------- #
def test_masked_classes_never_reach_the_hypothesis_posterior() -> None:
    lat = torch.zeros(1, 8)
    lon = torch.zeros(1, 8)
    a = C.hypothesis_posteriors(lat, lon)
    lat2, lon2 = lat.clone(), lon.clone()
    for i in C.LAT_MASKED_V7_IDS:
        lat2[0, i] = 50.0
    for i in C.LON_MASKED_V7_IDS:
        lon2[0, i] = 50.0
    b = C.hypothesis_posteriors(lat2, lon2)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert abs(float(a[0].exp().sum()) - 1.0) < 1e-6 and abs(float(a[1].exp().sum()) - 1.0) < 1e-6


def test_phi_layout_is_the_one_refcv8_train_reads() -> None:
    lat3 = torch.tensor([[1, 2]])
    lon6 = torch.tensor([[0, 5]])
    lp = torch.log(torch.full((1, 3), 1 / 3))
    lo = torch.log(torch.full((1, 6), 1 / 6))
    cons = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]])
    phi = C.build_phi(lat3, lon6, lp, lo, cons, torch.ones(1, 2), torch.tensor([[False, True]]))
    assert phi.shape == (1, 2, 17)
    assert torch.equal(phi[0, 0, :3], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(phi[0, 1, 3:9], torch.tensor([0.0, 0, 0, 0, 0, 1.0]))
    assert torch.allclose(phi[0, 1, 11:15], cons[0, 1]) and float(phi[0, 1, 15]) == 1.0
    assert float(phi[0, 0, 16]) == 0.0 and float(phi[0, 1, 16]) == 1.0


def test_modulation_is_the_identity_at_init_and_moves_once_trained() -> None:
    m = C.PerCandidateModulation(2, 8)
    q = torch.randn(2, 5, 8)
    phi = torch.randn(2, 5, C.PHI_DIMS)
    assert torch.equal(m(1, q, phi), q)                       # bit-identical, not approx
    with torch.no_grad():
        m.proj[1].bias[0] = 0.5                                # the deliberate regression: must move
    assert not torch.equal(m(1, q, phi), q)


# ---- allocation ------------------------------------------------------------------------------------------------- #
def test_allocation_is_proportional_with_floors_and_force() -> None:
    p = torch.full((1, C.N_HYP), 1e-6)
    p[0, 0], p[0, 1], p[0, 2], p[0, 3] = 0.5, 0.3, 0.15, 0.05
    p = p / p.sum()
    h = C.allocate(p.log(), 32, top_k=4, min_each=2)[0].tolist()
    # floors 2 each (8), remaining 24 split 0.5/0.3/0.15/0.05 -> 12 / 7.2 / 3.6 / 1.2 -> 12, 7, 4, 1 (largest
    # remainder: .6 then .2 -> +1 to the 3rd) => counts 14, 9, 6, 3 (LITERAL)
    assert [h.count(k) for k in (0, 1, 2, 3)] == [14, 9, 6, 3] and len(h) == 32
    hf = C.allocate(p.log(), 32, top_k=4, min_each=2, force=torch.tensor([17]), force_min=4)[0].tolist()
    assert hf.count(17) >= 4 and 3 not in hf and len(hf) == 32          # forced in, the least probable out


def test_source_anchor_match_levels() -> None:
    tl = torch.tensor([[0, 1, 1, 2]])
    to = torch.tensor([[1, 1, 2, 1]])
    hyp = C.joint_id(torch.tensor([[1, 1, 2, 0]]), torch.tensor([[2, 4, 1, 5]]))
    src, lvl = C.select_source_anchors(tl, to, hyp, C.R8Generator(0))
    assert lvl.tolist() == [[2, 1, 2, 1]]
    assert int(src[0, 0]) == 2 and int(src[0, 2]) == 3 and int(src[0, 1]) in (1, 2) and int(src[0, 3]) == 0


# ---- selection pieces --------------------------------------------------------------------------------------------- #
def test_factorized_selection_is_zero_at_init_and_sat_is_zero_when_satisfied() -> None:
    fs = C.FactorizedSelection()
    paths = torch.stack([_arc(5.0, 15.0, +1), _line([10.0] * 8)])[None]
    lat3 = torch.tensor([[1, 0]])
    lon6 = torch.tensor([[1, 1]])
    raw = torch.zeros(1, 2, 4)
    out = fs(lat3, lon6, torch.zeros(1, 3), torch.zeros(1, 6), paths, raw, torch.ones(1, 2))
    assert torch.equal(out, torch.zeros(1, 2))
    raw[0, 1] = torch.tensor([0.0, 0.0, 60.0, 10.0])          # the straight line's own constraint
    sp, sh = C.FactorizedSelection.sat_terms(paths, raw, torch.ones(1, 2))
    assert abs(float(sp[0, 1])) < 1e-6 and abs(float(sh[0, 1])) < 1e-6
    assert float(sp[0, 0]) < -0.5                              # 29.9 m vs 0 m -> |log1p| gap ~3.4


def test_listwise_loss_known_values() -> None:
    gt = _line([10.0] * 8)[None]
    fan = torch.stack([_line([10.0] * 8), _line([5.0] * 8)])[None]
    gv = torch.ones(1, 8, dtype=torch.bool)
    keep = torch.ones(1, 2, dtype=torch.bool)
    # equal scores -> log softmax = log(1/2) for both -> loss = ln 2 exactly, whatever the target (LITERAL)
    l = C.listwise_selection_loss(torch.zeros(1, 2), fan, gt, gv, keep)
    assert abs(float(l) - math.log(2.0)) < 1e-6
    # a single kept candidate -> loss 0
    l1 = C.listwise_selection_loss(torch.zeros(1, 2), fan, gt, gv, torch.tensor([[True, False]]))
    assert abs(float(l1)) < 1e-7
    # the target prefers the GT-equal candidate: raising its score lowers the loss
    l2 = C.listwise_selection_loss(torch.tensor([[3.0, 0.0]]), fan, gt, gv, keep)
    assert float(l2) < float(l)


def test_lsat_counts_only_the_conditioned_candidates() -> None:
    paths = torch.stack([_line([10.0] * 8), _line([10.0] * 8)])[None]
    raw = torch.tensor([[[0.0, 0.0, 60.0, 10.0], [0.0, 0.0, 10.0, 0.0]]])   # cand 1 asked to stop at 10 m
    assert float(C.constraint_satisfaction_loss(paths, raw, torch.ones(1, 2), torch.tensor([[True, False]]))) == 0.0
    assert float(C.constraint_satisfaction_loss(paths, raw, torch.ones(1, 2), torch.tensor([[False, True]]))) > 1.0


def test_prior_free_offset_cancels_the_prior_curvature() -> None:
    from tanitad.models import kinematic_prior as KP
    hz = (5, 10, 15, 20, 30, 40, 50, 60)
    v = torch.tensor([8.0])
    k0 = torch.tensor([0.05])                                  # |0.05| < the 0.12 residual cap: exact regime
    off = C.prior_free_lateral_offset(k0, v, 1, 8, control_norm_lat=3.0, alat_v_floor=4.0)
    delta = off * torch.tensor([4.0, 3.0])                     # back to vocabulary units (a = 0 anchor)
    pf = KP.roll_plan(delta, torch.zeros(1), k0, v, hz, control_units="alat")
    straight = KP.prior_path(torch.zeros(1), torch.zeros(1), v, hz)
    assert float((pf[0, 0] - straight[0]).abs().max()) < 1e-4
    biased = KP.roll_plan(torch.zeros(1, 1, 8, 2), torch.zeros(1), k0, v, hz, control_units="alat")
    assert float((biased[0, 0] - straight[0]).abs().max()) > 1.0                # the prior alone curves


def test_no_situation_output_in_the_hypothesis_space_and_the_mutation_refuses() -> None:
    C.assert_no_situation_feed()
    with pytest.raises(ValueError):
        C.assert_no_situation_feed(C.LAT3 + C.LON6 + ("TRAFFIC_LIGHT_REACT_RED",))
