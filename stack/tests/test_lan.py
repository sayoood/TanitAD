"""LAN tests — Lane-Anchored Navigation route conditioning.

Covers, in the order the design has to hold up:
(a) geometry primitives: arc-length resample refuses to extrapolate (a clamped
    point would look like a real route hop), ego-frame transform matches the
    repo's CCW-left convention,
(b) THE LEAK GUARD — no valid route anchor may lie inside the horizon the model
    must predict, and the encoding is SPEED-INVARIANT (arc-length, not time),
    which is the property that separates LAN from `refb_labels.nav_command`,
(c) the encoding distinguishes "route straight ahead" from "no route", and
    `mirror` changes topology without changing signal energy,
(d) cross-module constant pin: `refc.LAN_FEATS_PER_ANCHOR` == `lan.` — a silent
    divergence would mis-slice the route tensor,
(e) LaneCorridor: nearest-neighbour snapping, the heading gate that the NuRec
    probe named as the fix, hysteresis, and honest hop-coverage stats,
(f) route_agreement compares only commonly-valid anchors,
(g) the REF-C `graft_lan` seam: gated (absent when off), byte-identical output
    at init (zero-init cond + zero anchor gate), responsive once the gate opens,
    direction-correct under mirror, route-dropout is training-gated per-sample,
    the two route-carrying seams have LIVE gradients at init, and the route
    encoder's step-0 gradient block (the price of byte-identity) UNBLOCKS after
    one optimiser step — stated and pinned, not hidden,
(h) the route-counterfactual instrument, including its NEGATIVE CONTROLS: an
    inert model must read INERT, a nondeterministic one must read
    INSTRUMENT-FAIL, and a route-following one must read RESPONSIVE with
    compliance 1.0,
(i) a regression pin on the banked NuRec S1-vs-S2 agreement (skipped when the
    research-hub artifacts are not present).
CPU-only, synthetic data except (i).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tanitad.data.lan import (LAN_FEATS_PER_ANCHOR, LanConfig,  # noqa: E402
                              LaneCorridor, cumulative_arclength,
                              encode_route, horizon_lead_m, inert_route,
                              lan_from_future_path, lan_from_polyline,
                              mirror_route, resample_arclength,
                              route_agreement, straight_route, to_ego_frame,
                              yaw_from_path)
from tanitad.eval.route_cf import (CONTROL_TOL_M, commanded_lateral,
                                   lan_sensitivity, nav_cmd_sensitivity,
                                   route_counterfactual)
from tanitad.refs.refc import (LAN_FEATS_PER_ANCHOR as REFC_FEATS,
                               RefCModel, param_breakdown, refc_smoke_config)

HUB = Path(__file__).resolve().parents[2] / "TanitAD Research Hub"
NUREC = (HUB / "Architecture & Inference" / "Research"
         / "2026-08-02-nurec-xodr-map")


def _straight(n=201, length=200.0, lat_slope=0.0):
    x = np.linspace(0.0, length, n)
    return np.stack([x, lat_slope * x], axis=1)


# ---------- (a) geometry ------------------------------------------------------

def test_arclength_resample_refuses_to_extrapolate():
    xy = _straight(n=101, length=100.0)
    assert cumulative_arclength(xy)[-1] == pytest.approx(100.0)
    pts, valid = resample_arclength(xy, [10.0, 50.0, 100.0, 140.0])
    assert valid.tolist() == [True, True, True, False]
    assert pts[0] == pytest.approx([10.0, 0.0])
    assert pts[2] == pytest.approx([100.0, 0.0])
    # ⛔ the out-of-range point is ZEROED, never clamped to the last vertex — a
    # clamped point is indistinguishable from a real route anchor.
    assert pts[3] == pytest.approx([0.0, 0.0])


def test_degenerate_polylines_are_invalid_not_crashy():
    for xy in (np.zeros((0, 2)), np.zeros((1, 2)), np.zeros((5, 2))):
        _, valid = resample_arclength(xy, [10.0])
        assert not valid.any()


def test_ego_frame_is_x_forward_y_left_ccw():
    # Facing +y (yaw 90 deg): a world point at +y is straight AHEAD; a world
    # point at -x is to the LEFT. Matches refb_labels (CCW == left == NAV_LEFT)
    # and refc.synth_anchor_pool (positive yaw-rate integrates to positive y).
    e = to_ego_frame(np.array([[0.0, 10.0], [-4.0, 0.0]]), np.zeros(2),
                     math.pi / 2)
    assert e[0] == pytest.approx([10.0, 0.0], abs=1e-9)
    assert e[1] == pytest.approx([0.0, 4.0], abs=1e-9)


def test_yaw_from_path_has_no_silent_default_when_stationary():
    assert yaw_from_path(np.zeros((5, 2))) == 0.0
    assert yaw_from_path(_straight(n=20, length=20.0)) == pytest.approx(0.0)


# ---------- (b) the leak guard ------------------------------------------------

def test_leak_guard_is_conservative_and_masks_inside_the_horizon():
    cfg = LanConfig(min_lead_m=5.0)
    # v0 = 15 m/s over 2 s = 30 m; guard = 35 m -> the 20 m anchor is masked.
    lead = horizon_lead_m(v0=15.0, t_pred_s=2.0, cfg=cfg)
    assert lead == pytest.approx(35.0)
    r = lan_from_future_path(_straight(), np.zeros(2), 0.0, cfg, lead)
    assert r.valid.tolist() == [False, True, True, True]    # the 20 m is masked
    arc = np.asarray(cfg.arclengths_m)
    assert (arc[r.valid] >= lead).all()
    # ...and a SHORT future loses its far anchors to the coverage mask, not the
    # guard: 100 m of path cannot answer where the route is at 160 m.
    short = lan_from_future_path(_straight(n=101, length=100.0), np.zeros(2),
                                 0.0, cfg, lead)
    assert short.valid.tolist() == [False, True, True, False]

    # The guard takes the MAX of the two estimates, so it can only mask MORE.
    gt = np.stack([np.linspace(0, 60, 21), np.zeros(21)], axis=1)   # 60 m path
    assert horizon_lead_m(gt_path_ego=gt, v0=1.0, cfg=cfg) == pytest.approx(65.0)
    assert horizon_lead_m(gt_path_ego=gt[:1], v0=100.0, cfg=cfg) > 200.0


def test_no_valid_anchor_lies_inside_the_predicted_horizon():
    """The exact, non-overclaimed guarantee: every anchor the model can see is
    at an arc-length beyond the whole 2 s path it is asked to output."""
    cfg = LanConfig()
    rng = np.random.default_rng(0)
    for _ in range(50):
        v0 = float(rng.uniform(0.0, 30.0))
        lead = horizon_lead_m(v0=v0, t_pred_s=2.0, cfg=cfg)
        r = lan_from_future_path(_straight(length=400.0), np.zeros(2), 0.0,
                                 cfg, lead)
        arc = np.asarray(cfg.arclengths_m)[r.valid]
        assert (arc >= v0 * 2.0).all(), (v0, arc)


def test_encoding_is_speed_invariant_because_it_is_arc_length_not_time():
    """The defect LAN replaces: `nav_command` thresholds NET HEADING OVER TIME,
    and dyaw = kappa * v * t, so the same road yields a different label at a
    different speed. Sampling the SAME spatial path at two different speeds must
    give bit-identical LAN features."""
    cfg = LanConfig(min_lead_m=5.0)
    path = _straight(n=2001, length=400.0, lat_slope=0.2)
    slow = path[::1]          # dense time sampling == slow
    fast = path[::7]          # sparse == fast; identical GEOMETRY
    lead = horizon_lead_m(v0=0.0, cfg=cfg)      # same guard for both
    a = lan_from_future_path(slow, path[0], 0.0, cfg, lead)
    b = lan_from_future_path(fast, path[0], 0.0, cfg, lead)
    assert a.valid.tolist() == b.valid.tolist()
    np.testing.assert_allclose(a.features, b.features, atol=1e-5)


# ---------- (c) the encoding --------------------------------------------------

def test_straight_route_and_no_route_are_distinguishable():
    cfg = LanConfig()
    s = straight_route(cfg)
    z = inert_route(cfg)
    assert s.valid.all() and not z.valid.any()
    f = s.features.reshape(-1, LAN_FEATS_PER_ANCHOR)
    assert f[:, 0] == pytest.approx(1.0)          # cos bearing
    assert f[:, 1] == pytest.approx(0.0)          # sin bearing
    assert f[:, 3] == pytest.approx(1.0)          # valid
    assert not np.allclose(s.features, z.features)


def test_mirror_flips_topology_without_changing_signal_energy():
    cfg = LanConfig(min_lead_m=0.0)
    r = lan_from_future_path(_straight(length=400.0, lat_slope=0.3),
                             np.zeros(2), 0.0, cfg, 0.0)
    m = mirror_route(r)
    assert np.linalg.norm(r.features) == pytest.approx(np.linalg.norm(m.features))
    a = r.features.reshape(-1, LAN_FEATS_PER_ANCHOR)
    b = m.features.reshape(-1, LAN_FEATS_PER_ANCHOR)
    np.testing.assert_allclose(a[:, 0], b[:, 0])          # cos unchanged
    np.testing.assert_allclose(a[:, 1], -b[:, 1])         # sin flipped
    np.testing.assert_allclose(a[:, 2], -b[:, 2])         # lateral flipped
    np.testing.assert_allclose(a[:, 3], b[:, 3])          # validity unchanged


def test_encode_route_rejects_a_shape_mismatch():
    cfg = LanConfig()
    with pytest.raises(ValueError):
        encode_route(np.zeros((3, 2)), np.ones(3, dtype=bool), cfg)


def test_lan_config_guards_its_own_contract():
    for kw in ({"arclengths_m": ()}, {"arclengths_m": (10.0, -1.0)},
               {"arclengths_m": (40.0, 20.0)}, {"lat_clip": 0.0}):
        with pytest.raises(ValueError):
            LanConfig(**kw)


# ---------- (d) cross-module constant pin -------------------------------------

def test_feature_width_is_pinned_across_modules():
    assert REFC_FEATS == LAN_FEATS_PER_ANCHOR == 4
    assert LanConfig(arclengths_m=(1.0, 2.0, 3.0)).dim == 12


# ---------- (e) LaneCorridor --------------------------------------------------

def _two_lane_corridor():
    """Two parallel lanes 3.5 m apart, one running each way — the geometry the
    heading gate exists for."""
    fwd = np.stack([np.linspace(0, 100, 51), np.zeros(51)], axis=1)
    bwd = np.stack([np.linspace(100, 0, 51), np.full(51, 3.5)], axis=1)
    return LaneCorridor({"fwd": fwd, "bwd": bwd}, [("fwd", "fwd")])


def test_heading_gate_rejects_the_oncoming_lane():
    corr = _two_lane_corridor()
    p = np.array([50.0, 2.6])                # nearer the OPPOSING lane (0.9 m)
    assert corr.snap(p)[0] == "bwd"                       # ungated: wrong lane
    lid, d = corr.snap(p, heading=0.0, max_heading_dev_rad=math.radians(60))
    assert lid == "fwd" and d == pytest.approx(2.6, abs=1e-6)


def test_heading_gate_falls_back_rather_than_returning_nothing():
    corr = _two_lane_corridor()
    lid, _ = corr.snap(np.array([50.0, 1.0]), heading=math.pi / 2,
                       max_heading_dev_rad=math.radians(5))
    assert lid in ("fwd", "bwd")             # gate rejected all -> honest fallback


def test_route_polyline_reports_hop_coverage_without_claiming_routability():
    corr = _two_lane_corridor()
    track = np.stack([np.linspace(5, 95, 40), np.zeros(40)], axis=1)
    poly, seq, stats = corr.route_polyline(track)
    assert seq == ["fwd"] and poly.shape[1] == 2
    assert stats["n_hops"] == 0 and stats["hops_on_graph_frac"] == 0.0
    assert stats["snap_median_m"] == pytest.approx(0.0, abs=1e-6)


def test_corridor_refuses_a_degenerate_map():
    with pytest.raises(ValueError):
        LaneCorridor({"a": [[0.0, 0.0]]})


def test_polyline_supplier_reanchors_at_the_ego():
    cfg = LanConfig(min_lead_m=0.0)
    poly = _straight(n=401, length=400.0)
    r = lan_from_polyline(poly, np.array([100.0, 0.0]), 0.0, cfg, 0.0)
    # arc-length 0 is the foot of the ego on the polyline, so the 20 m anchor is
    # 20 m AHEAD OF THE EGO, not 20 m from the polyline's own start.
    assert r.points_ego[0] == pytest.approx([20.0, 0.0], abs=1e-6)


# ---------- (f) agreement -----------------------------------------------------

def test_route_agreement_is_zero_for_identical_routes_and_skips_half_valid():
    cfg = LanConfig(min_lead_m=0.0)
    r = lan_from_future_path(_straight(length=400.0, lat_slope=0.1),
                             np.zeros(2), 0.0, cfg, 0.0)
    a = route_agreement(r, r)
    assert a["pos_l2_m"] == pytest.approx(0.0)
    assert a["side_agree"] == pytest.approx(1.0)
    assert a["n_compared"] == cfg.k
    # An anchor valid in only one supplier is coverage, not error.
    b = route_agreement(r, inert_route(cfg))
    assert b["n_compared"] == 0 and math.isnan(b["pos_l2_m"])


# ---------- (g) the REF-C graft ----------------------------------------------

def _lan_model(**flags):
    cfg = refc_smoke_config()
    cfg.graft_lan = True
    for k, v in flags.items():
        setattr(cfg, k, v)
    return RefCModel(cfg), cfg


def _frames(cfg, b=3):
    return torch.randn(b, cfg.window, cfg.encoder.in_channels, 64, 64)


def _routes(n, cfg_lan, slopes):
    return [lan_from_future_path(_straight(length=400.0, lat_slope=s),
                                 np.zeros(2), 0.0, cfg_lan, 0.0) for s in slopes]


def test_lan_graft_is_absent_when_off_and_adds_only_its_own_keys():
    off = set(RefCModel(refc_smoke_config()).state_dict())
    assert not any("lan" in k for k in off)
    on = set(_lan_model()[0].state_dict())
    extra = on - off
    assert extra == {"lan_enc.0.weight", "lan_enc.0.bias",
                     "lan_enc.2.weight", "lan_enc.2.bias",
                     "decoder.lan_to_cond.weight", "decoder.lan_to_cond.bias",
                     "decoder.lan_gate"}
    assert param_breakdown(RefCModel(refc_smoke_config()))["lan"] == 0
    bd = param_breakdown(_lan_model()[0])
    assert bd["lan"] > 0
    assert sum(v for k, v in bd.items() if k != "total") == bd["total"]


def test_at_init_the_route_changes_nothing_bitwise():
    """Zero-init condition projection + zero anchor gate: a LAN-conditioned
    model at step 0 must decode EXACTLY what it decodes with no route at all,
    or the graft is not a graft."""
    torch.manual_seed(0)
    model, cfg = _lan_model()
    model.eval()
    fr = _frames(cfg)
    lc = LanConfig(min_lead_m=0.0)
    lan = torch.tensor(np.stack([r.features for r in
                                 _routes(3, lc, [0.0, 0.4, -0.4])]))
    with torch.no_grad():
        base = model(fr, lan=None)["traj"]
        with_route = model(fr, lan=lan)["traj"]
        mirrored = model(fr, lan=torch.tensor(np.stack(
            [mirror_route(r).features
             for r in _routes(3, lc, [0.0, .4, -.4])])))["traj"]
    assert torch.equal(base, with_route)
    assert torch.equal(base, mirrored)


def test_once_the_gate_opens_the_route_steers_the_pick_the_right_way():
    torch.manual_seed(0)
    model, cfg = _lan_model()
    model.eval()
    with torch.no_grad():                       # open the geometric anchor gate
        model.decoder.lan_gate.fill_(8.0)
    fr = _frames(cfg, b=2)
    lc = LanConfig(min_lead_m=0.0)
    left = _routes(2, lc, [0.6, 0.6])
    right = [mirror_route(r) for r in left]
    with torch.no_grad():
        yl = model(fr, lan=torch.tensor(np.stack([r.features for r in left])))
        yr = model(fr, lan=torch.tensor(np.stack([r.features for r in right])))
    assert not torch.equal(yl["traj"], yr["traj"])
    # A LEFT route must select an endpoint further left than a RIGHT route.
    assert (yl["traj"][:, -1, 1] > yr["traj"][:, -1, 1]).all()
    assert yl["lan_dir"][:, 1].gt(0).all() and yr["lan_dir"][:, 1].lt(0).all()


def test_lan_direction_reads_the_first_valid_anchor_and_never_votes_straight():
    model, _ = _lan_model()
    cfg = LanConfig()
    # No valid anchor -> valid flag 0 so the geometric prior multiplies out.
    d = model.lan_direction(torch.tensor(inert_route(cfg).features)[None], cfg.k)
    assert d.tolist() == [[1.0, 0.0, 0.0]]
    # First valid anchor wins: mask anchor 0, give anchor 1 a left bearing.
    f = np.zeros((cfg.k, LAN_FEATS_PER_ANCHOR), dtype=np.float32)
    f[1] = [0.0, 1.0, 1.0, 1.0]
    f[2] = [1.0, 0.0, 0.0, 1.0]
    d = model.lan_direction(torch.tensor(f.reshape(1, -1)), cfg.k)
    assert d[0].tolist() == [0.0, 1.0, 1.0]


def test_route_dropout_is_per_sample_and_training_gated():
    torch.manual_seed(0)
    model, cfg = _lan_model(route_dropout=1.0)
    fr = _frames(cfg, b=4)
    lc = LanConfig(min_lead_m=0.0)
    lan = torch.tensor(np.stack([r.features for r in _routes(4, lc, [.5] * 4)]))
    model.train()
    assert model(fr, lan=lan)["lan_dir"][:, 2].sum() == 0.0     # all dropped
    model.eval()
    assert model(fr, lan=lan)["lan_dir"][:, 2].sum() == 4.0     # none dropped
    torch.manual_seed(3)
    model.cfg.route_dropout = 0.5
    model.train()
    keeps = [float(model(fr, lan=lan)["lan_dir"][:, 2].sum()) for _ in range(12)]
    assert min(keeps) < 4.0 and max(keeps) > 0.0                # per-sample
    assert len(set(keeps)) > 1


def _traj_loss(model, cfg, lan, steps=1):
    out = model(_frames(cfg, b=2), lan=lan, steps=steps)
    return out["traj"].pow(2).mean() + out["anchor_logits"].pow(2).mean()


def test_the_two_seams_that_carry_the_route_have_live_gradients_at_init():
    """The seams that must move first — the condition projection and the anchor
    gate — are gated, NOT dead: both start at exactly 0 and both receive a
    finite non-zero gradient on the very first backward."""
    torch.manual_seed(0)
    # route_dropout off: it is a training regulariser, and with it on a seeded
    # 2-row batch can drop BOTH routes, which correctly zeroes the anchor-gate
    # gradient and would make this test measure the dropout, not the seam.
    model, cfg = _lan_model(route_dropout=0.0)
    model.train()
    lc = LanConfig(min_lead_m=0.0)
    lan = torch.tensor(np.stack([r.features for r in _routes(2, lc, [.5, -.5])]))
    assert float(model.decoder.lan_gate.detach()) == 0.0
    assert float(model.decoder.lan_to_cond.weight.detach().abs().sum()) == 0.0
    _traj_loss(model, cfg, lan).backward()
    for name in ("decoder.lan_to_cond.weight", "decoder.lan_gate"):
        g = dict(model.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_the_route_encoder_is_gradient_blocked_at_init_and_unblocks_after_a_step():
    """⚠️ A MEASURED property of this design, stated rather than hidden.

    Byte-identity at init and a live INPUT gradient at init are mutually
    exclusive for a zero-init gated seam: with ``lan_to_cond.weight == 0`` the
    chain rule sends exactly 0 back into ``lan_enc``. The programme's existing
    ``ctx_to_cond`` -> ``StrategicCtx`` graft has the same property, and this
    test pins the consequence that actually matters — the block is transient:
    ``lan_to_cond`` itself HAS a gradient, so one optimiser step opens the path
    and ``lan_enc`` trains from step 1. A reviewer must not read the step-0 zero
    as a dead module.
    """
    torch.manual_seed(0)
    model, cfg = _lan_model(route_dropout=0.0)
    model.train()
    lc = LanConfig(min_lead_m=0.0)
    lan = torch.tensor(np.stack([r.features for r in _routes(2, lc, [.5, -.5])]))

    _traj_loss(model, cfg, lan).backward()
    assert float(model.lan_enc[0].weight.grad.abs().sum()) == 0.0   # step 0

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt.step()                                   # one step opens lan_to_cond
    opt.zero_grad(set_to_none=True)
    assert float(model.decoder.lan_to_cond.weight.detach().abs().sum()) > 0.0
    _traj_loss(model, cfg, lan).backward()
    for name in ("lan_enc.0.weight", "lan_enc.2.weight"):
        g = dict(model.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


# ---------- (h) the instrument and its negative controls ----------------------

class _StubPredictor:
    """`follow=0` ignores the route; `follow>0` moves the endpoint toward the
    commanded lateral offset; `jitter>0` makes decoding nondeterministic."""

    def __init__(self, n, k, follow=0.0, jitter=0.0, seed=0):
        self.n, self.k, self.follow, self.jitter = n, k, follow, jitter
        self.rng = np.random.default_rng(seed)
        self.base = np.zeros((n, 4, 2))
        self.base[:, :, 0] = np.linspace(2.0, 8.0, 4)[None]

    def __call__(self, feats):
        f = np.asarray(feats, dtype=np.float64).reshape(self.n, self.k,
                                                        LAN_FEATS_PER_ANCHOR)
        y = self.base.copy()
        lat = (f[..., 2] * f[..., 3]).sum(axis=1)
        y[:, :, 1] += self.follow * lat[:, None]
        if self.jitter:
            y = y + self.rng.normal(scale=self.jitter, size=y.shape)
        return y


# slopes deliberately include 0.0: a straight route is unchanged by `mirror`,
# so that window is a TIE and must be EXCLUDED from compliance, not counted as
# agreement (counting ties is how a 74 %-straight corpus hides a dead route).
_SLOPES = [-0.5, -0.3, 0.0, 0.1, 0.3, 0.5]


def test_instrument_reads_INERT_on_a_model_that_ignores_the_route():
    cfg = LanConfig(min_lead_m=0.0)
    base = _routes(6, cfg, _SLOPES)
    treat = [mirror_route(r) for r in base]
    res = lan_sensitivity(_StubPredictor(6, cfg.k, follow=0.0), base, treat,
                          cfg.arclengths_m)
    s = res.summary()
    assert s["discriminative"] is True            # the identical-copy control
    assert s["route_sensitivity_m"] == 0.0
    assert s["verdict"].startswith("INERT")


def test_instrument_reads_RESPONSIVE_and_compliant_on_a_route_follower():
    cfg = LanConfig(min_lead_m=0.0)
    base = _routes(6, cfg, _SLOPES)
    treat = [mirror_route(r) for r in base]
    res = lan_sensitivity(_StubPredictor(6, cfg.k, follow=1.0), base, treat,
                          cfg.arclengths_m)
    s = res.summary()
    assert s["discriminative"] is True
    assert s["route_sensitivity_m"] > 0.0
    assert s["lat_compliance"] == 1.0
    assert s["n_decided"] == 5                    # the straight window is a tie
    assert s["verdict"].startswith("RESPONSIVE")


def test_instrument_fails_loudly_on_a_nondeterministic_decode():
    cfg = LanConfig(min_lead_m=0.0)
    base = _routes(6, cfg, _SLOPES)
    res = lan_sensitivity(_StubPredictor(6, cfg.k, follow=1.0, jitter=0.05),
                          base, [mirror_route(r) for r in base],
                          cfg.arclengths_m)
    s = res.summary()
    assert s["discriminative"] is False
    assert s["control_max_disp_m"] > CONTROL_TOL_M
    assert s["verdict"].startswith("INSTRUMENT-FAIL")


def test_commanded_lateral_reads_the_farthest_commonly_valid_anchor():
    cfg = LanConfig(min_lead_m=0.0)
    r = lan_from_future_path(_straight(length=400.0, lat_slope=0.25),
                             np.zeros(2), 0.0, cfg, 0.0)
    a = np.stack([r.features])
    b = np.stack([mirror_route(r).features])
    cmd = commanded_lateral(a, b, cfg.arclengths_m)
    j = int(np.flatnonzero(r.valid)[-1])
    expect = -2.0 * r.points_ego[j, 1]
    assert cmd[0] == pytest.approx(expect, rel=1e-4)


def test_route_counterfactual_rejects_misshaped_arms():
    with pytest.raises(ValueError):
        route_counterfactual(lambda c: np.zeros((3, 4)), 0, 1, "bad")


def test_nav_cmd_sweep_separates_inert_from_responsive():
    y = np.zeros((5, 4, 2))

    inert = nav_cmd_sensitivity(lambda i: y, 5)
    assert inert["discriminative"] and inert["max_pairwise_mean_m"] == 0.0
    assert inert["verdict"].startswith("INERT")

    def responsive(i):
        z = y.copy()
        z[:, :, 1] += float(i)
        return z

    live = nav_cmd_sensitivity(responsive, 5)
    assert live["discriminative"] and live["max_pairwise_mean_m"] == pytest.approx(3.0)
    assert live["verdict"].startswith("RESPONSIVE")
    assert live["pairwise_mean_m"]["0v3"] == pytest.approx(3.0)


# ---------- (h2) trainer wiring ----------------------------------------------

def test_lan_window_features_bridge_is_fail_loud():
    from tanitad.data.lan import lan_window_features
    poses = np.zeros((50, 4))
    poses[:, 0] = np.linspace(0, 200, 50)
    poses[:, 3] = 10.0
    f = lan_window_features(poses, 0)
    assert f.shape == (LanConfig().dim,) and f.dtype == np.float32
    # v0 = 10 -> guard 25 m -> the 20 m anchor is masked, not silently kept.
    assert f.reshape(-1, LAN_FEATS_PER_ANCHOR)[0, 3] == 0.0
    with pytest.raises(ValueError):
        lan_window_features(np.zeros((50, 3)), 0)
    with pytest.raises(IndexError):
        lan_window_features(poses, 999)


def test_trainer_runs_the_lan_arm_and_records_its_provenance(tmp_path):
    """`--graft-lan` alone defines the arm: the dataset emits `lan`, the model
    consumes it, the coverage stat is written, and the step log carries
    `lan_valid_frac` so a dead route input is visible in the FIRST log line
    rather than after a 30 k run."""
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import refc_train  # noqa: E402
    from test_refc import _make_cached_root  # noqa: E402

    root = _make_cached_root(tmp_path)
    out = tmp_path / "lanrun"
    metrics = refc_train.main([
        "--data-root", str(root), "--out", str(out), "--steps", "2",
        "--batch", "4", "--lr", "1e-3", "--episodes", "0", "--log-every", "1",
        "--device", "cpu", "--smoke", "--graft-lan",
        "--lan-arclengths", "20", "40", "80", "160"])
    assert np.isfinite(metrics["final"]["loss"])
    assert 0.0 <= metrics["final"]["lan_valid_frac"] <= 1.0
    conf = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert conf["cfg"]["graft_lan"] is True
    assert conf["param_breakdown"]["lan"] > 0
    lan = conf["labels"]["lan"]
    assert lan["derivation"].startswith("tanitad.data.lan")
    assert lan["stats"]["n_sampled"] > 0
    assert len(lan["stats"]["per_anchor_valid_frac"]) == 4


# ---------- (i) regression pin on the banked NuRec measurement ----------------

@pytest.mark.skipif(not (NUREC / "lane_centerlines.json").exists(),
                    reason="NuRec xodr artifacts not present in this checkout")
def test_map_free_and_map_based_route_agree_on_the_banked_nurec_scene():
    """MEASURED 2026-08-03 on scene 00040136-…: with the heading gate the two
    suppliers agree to a median 1.186 m (mean 1.459, p90 2.733) over 275
    comparable samples; the UNGATED nearest-neighbour snap gives median 4.685 /
    mean 19.801. n = 1 scene — a pin against regression, NOT a corpus claim.
    """
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import lan_probe  # noqa: E402

    cfg = LanConfig()
    res = lan_probe.agreement(NUREC / "lane_centerlines.json",
                              NUREC / "lane_graph_edges.json",
                              NUREC / "ego_track_map_frame.json", cfg)
    assert res["n_compared_samples"] > 200
    assert res["pos_l2_m"]["median"] < 2.0
    assert res["pos_l2_m"]["p90"] < 5.0
    assert res["bearing_deg"]["median"] < 5.0
    # And the gate is load-bearing, not decoration: turning it off must degrade.
    ungated = lan_probe.agreement(NUREC / "lane_centerlines.json",
                                  NUREC / "lane_graph_edges.json",
                                  NUREC / "ego_track_map_frame.json", cfg,
                                  max_heading_dev_deg=None, hysteresis_m=0.0)
    assert ungated["pos_l2_m"]["mean"] > 5.0 * res["pos_l2_m"]["mean"]
    json.dumps(res["route"])            # stats stay JSON-serialisable
