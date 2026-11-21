"""Tests for E15 — the predicted metric goal point.

⛔ The load-bearing test in this file is not a shape check: it is
``test_R_the_categorical_collapse_that_killed_E13_cannot_happen_in_the_geometric_prior``,
which REINTRODUCES the measured defect (a nav embedding whose rows have collapsed to a
common bias — refcv4b's content share 2.3 %) and proves that the categorical path is
blind to it while the param-free geometric prior is not. A guard that only INSPECTS
passes on both the fixed and the broken code; this one mutates.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tanitad.data.lan import LanConfig, horizon_lead_m
from tanitad.refs.goal_point import (GOAL_POINT_FEAT_NAMES, GOAL_POINT_FEATS,
                                     anchor_goal_prior_at_time,
                                     goal_point_label_at_time, goal_slot_index,
                                     range_rmse_m,
                                     GoalPointConditioning, GoalPointConfig,
                                     GoalPointHead, anchor_goal_prior,
                                     anchor_point_at_arc, encode_goal_point,
                                     goal_point_label, goal_point_loss,
                                     goal_point_provenance, guard_arc_m,
                                     lateral_rmse_m)


def _straight_path(n=60, ds=1.0):
    return np.stack([np.arange(1, n + 1) * ds, np.zeros(n)], axis=-1)


def _fan(b=4, n=9, s=8, arc=30.0):
    """A symmetric left/right fan: constant-curvature arcs, index 4 straight."""
    kap = torch.linspace(-0.04, 0.04, n)
    t = torch.linspace(arc / s, arc, s)
    th = kap[:, None] * t[None, :]
    x = torch.where(kap[:, None].abs() < 1e-6, t[None, :],
                    torch.sin(th) / kap[:, None].clamp(min=1e-6, max=1e9))
    y = torch.where(kap[:, None].abs() < 1e-6, torch.zeros_like(th),
                    (1 - torch.cos(th)) / kap[:, None].clamp(min=1e-6, max=1e9))
    x = torch.where(kap[:, None].abs() < 1e-6, t[None, :], torch.sin(th) / kap[:, None])
    y = torch.where(kap[:, None].abs() < 1e-6, torch.zeros_like(th),
                    (1 - torch.cos(th)) / kap[:, None])
    one = torch.stack([x, y], -1)
    return one[None].expand(b, n, s, 2).contiguous()


# --------------------------------------------------------------------------- layout

def test_feature_layout_is_pinned():
    assert GOAL_POINT_FEATS == 3
    assert GOAL_POINT_FEAT_NAMES == ("x_norm", "y_norm", "valid")


def test_invalid_goal_encodes_to_EXACTLY_zeros_including_the_flag():
    f = encode_goal_point([12.0, -3.0], False, 30.0)
    assert f.shape == (GOAL_POINT_FEATS,)
    assert np.array_equal(f, np.zeros(3, dtype=np.float32))
    # and "straight ahead" is DISTINGUISHABLE from "no goal" — the X15 rule
    g = encode_goal_point([30.0, 0.0], True, 30.0)
    assert g[2] == 1.0 and not np.array_equal(f, g)


# --------------------------------------------------------------------------- guard

def test_guard_arc_is_the_IMPORTED_lan_guard_not_a_re_derivation():
    cfg = GoalPointConfig()
    path = _straight_path(20, 1.0)
    got = guard_arc_m(v0=12.0, gt_path_ego=path, cfg=cfg)
    want = horizon_lead_m(gt_path_ego=path, v0=12.0, t_pred_s=cfg.t_pred_s,
                          cfg=LanConfig(min_lead_m=cfg.min_lead_m,
                                        lat_clip=cfg.lat_clip))
    assert got == pytest.approx(min(max(want, cfg.arc_min_m), cfg.arc_max_m))
    # ⛔ MUTATION: the plausible re-derivation "v0 * t_pred + margin", which DROPS the
    # measured-GT-arc half of the max, is a DIFFERENT number on a window whose ground
    # truth ran further than v0 * t_pred. If it were equal this test would prove nothing.
    naive = 12.0 * cfg.t_pred_s + cfg.min_lead_m
    fast_path = _straight_path(60, 1.0)                     # 60 m of GT over 2 s
    assert guard_arc_m(12.0, fast_path, cfg) != pytest.approx(naive)


def test_guard_is_clamped_into_the_registered_band():
    cfg = GoalPointConfig()
    assert guard_arc_m(0.0, None, cfg) == cfg.arc_min_m          # standstill
    assert guard_arc_m(200.0, None, cfg) == cfg.arc_max_m        # absurd speed


# --------------------------------------------------------------------------- label

def test_label_refuses_a_path_shorter_than_the_arc():
    p, ok = goal_point_label(_straight_path(10, 1.0), arc_m=30.0)
    assert ok is False and np.array_equal(p, np.zeros(2, dtype=np.float32))


def test_label_reads_a_KNOWN_value_on_a_straight_path():
    p, ok = goal_point_label(_straight_path(60, 1.0), arc_m=30.0)
    assert ok is True
    assert p[0] == pytest.approx(30.0, abs=1e-3)
    assert p[1] == pytest.approx(0.0, abs=1e-6)


def test_label_reads_a_KNOWN_value_on_a_right_turn():
    # quarter circle of radius 20 m turning RIGHT (y negative)
    th = np.linspace(0, np.pi / 2, 200)[1:]
    path = np.stack([20 * np.sin(th), -20 * (1 - np.cos(th))], -1)
    p, ok = goal_point_label(path, arc_m=20.0 * np.pi / 2 * 0.5)   # quarter of the arc
    assert ok is True and p[1] < -1.0, "a right turn must give a NEGATIVE lateral goal"


# --------------------------------------------------------------------------- geometry

def test_anchor_point_at_arc_reads_a_KNOWN_value_on_a_straight_bank():
    """⛔ The arc origin is the CAR, not waypoint 0. This bank's first slot sits at
    x = 5 m (V3_HORIZONS starts at 0.5 s), so arc 20 m must read x = 20, NOT x = 25 —
    the several-metre mismatch that would otherwise open between the anchor arc and the
    goal-label arc. The 25.0 form is asserted against as the negative."""
    bank = torch.stack([torch.linspace(5, 40, 8), torch.zeros(8)], -1)
    bank = bank[None, None].expand(3, 2, 8, 2).contiguous()
    pt = anchor_point_at_arc(bank, torch.full((3,), 20.0))
    assert torch.allclose(pt[..., 0], torch.full_like(pt[..., 0], 20.0), atol=1e-4)
    assert torch.allclose(pt[..., 1], torch.zeros_like(pt[..., 1]), atol=1e-6)
    assert not torch.allclose(pt[..., 0], torch.full_like(pt[..., 0], 25.0), atol=1e-4)


def test_invalid_goal_makes_the_prior_EXACTLY_zero():
    bank = _fan()
    s = anchor_goal_prior(torch.tensor([[25.0, 4.0]] * 4), torch.zeros(4),
                          torch.full((4,), 30.0), bank)
    assert torch.equal(s, torch.zeros_like(s))


# ------------------------------------------------- the deliberate-regression arm

def test_R_a_MIRRORED_goal_point_selects_the_MIRRORED_anchor():
    """⛔ THE DELIBERATE-REGRESSION ARM. If a corrupted goal could not change the
    ranking, a PASS on the real gate would mean nothing — E13 passed a value-flip
    trivially for exactly that reason (`os_navflip - os` = +0.0022, not separated)."""
    bank = _fan(b=6)
    arc = torch.full((6,), 30.0)
    pts = anchor_point_at_arc(bank, arc)                  # [6, 9, 2]
    goal = pts[:, 7]                                      # a decidedly LEFT anchor
    mirrored = goal.clone(); mirrored[:, 1] *= -1.0
    ok = torch.ones(6)
    a = anchor_goal_prior(goal, ok, arc, bank).argmax(-1)
    b = anchor_goal_prior(mirrored, ok, arc, bank).argmax(-1)
    assert torch.equal(a, torch.full_like(a, 7))
    assert torch.equal(b, torch.full_like(b, 1)), "the mirror must pick the mirror index"
    assert not torch.equal(a, b)


def test_R_the_categorical_collapse_that_killed_E13_cannot_happen_in_the_geometric_prior():
    """⛔⛔ MUTATION, not inspection.

    Reintroduce the MEASURED defect: an E13-style nav embedding whose four rows have
    collapsed onto a common vector (refcv4b's content share 2.3 %). Under that
    collapse the categorical conditioning is BIT-IDENTICAL for every token — which is
    exactly what `os_navflip ~ os_navshuf ~ os` measured. The param-free geometric
    prior has no such degree of freedom, and the test proves BOTH halves: without the
    first assert the second would not be evidence about a *mechanism*.
    """
    d_nav, d_tac, d_ctx = 8, 16, 4
    nav = torch.nn.Embedding(4, d_nav)
    with torch.no_grad():                                  # THE COLLAPSE
        nav.weight.copy_(torch.ones(4, d_nav) * 0.37)
    to_tac = torch.nn.Linear(d_nav, d_tac)
    outs = [to_tac(nav(torch.tensor([k]))) for k in range(4)]
    for k in range(1, 4):
        assert torch.equal(outs[0], outs[k]), (
            "the collapsed categorical edge must be blind to its token — if this "
            "fails the mutation did not take and the second half proves nothing")

    bank = _fan(b=4)
    arc = torch.full((4,), 30.0)
    pts = anchor_point_at_arc(bank, arc)
    ok = torch.ones(4)
    scores = [anchor_goal_prior(pts[:, k], ok, arc, bank) for k in (1, 4, 7)]
    for i in range(1, 3):
        assert not torch.allclose(scores[0], scores[i]), (
            "the geometric prior must MOVE when the goal moves — there is no "
            "presence-without-content regime for it to collapse into")
    assert torch.stack([s.argmax(-1) for s in scores]).unique().numel() == 3


# --------------------------------------------------------------------------- module

def test_conditioning_is_bit_inert_at_init_and_zero_on_a_withheld_goal():
    m = GoalPointConditioning(d_goal=8, d_tac=16, d_ctx=4)
    feats = torch.tensor([[0.9, 0.3, 1.0], [1.0, -0.2, 1.0], [0.0, 0.0, 0.0]])
    t, s = m(feats)
    assert torch.equal(t, torch.zeros_like(t)) and torch.equal(s, torch.zeros_like(s))
    with torch.no_grad():                                  # train the projections away
        m.to_tac.weight.normal_(); m.to_str.weight.normal_()
        m.to_tac.bias.normal_(); m.to_str.bias.normal_()
    t, s = m(feats)
    assert torch.equal(t[2], torch.zeros_like(t[2])), "a withheld goal must stay zero"
    assert torch.equal(s[2], torch.zeros_like(s[2]))
    assert not torch.equal(t[0], torch.zeros_like(t[0]))


def test_conditioning_refuses_a_wrong_width():
    m = GoalPointConditioning(8, 16, 4)
    with pytest.raises(ValueError):
        m(torch.zeros(2, 2))


def test_head_shape():
    assert GoalPointHead(12)(torch.zeros(5, 12)).shape == (5, 2)


def test_loss_masks_invalid_rows_to_EXACTLY_zero():
    pred = torch.zeros(3, 2, requires_grad=True)
    tgt = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    loss = goal_point_loss(pred, tgt, torch.tensor([0.0, 0.0, 0.0]))
    assert float(loss) == 0.0
    loss = goal_point_loss(pred, tgt, torch.tensor([1.0, 0.0, 0.0]))
    loss.backward()
    assert float(pred.grad[1].abs().sum()) == 0.0
    assert float(pred.grad[0].abs().sum()) > 0.0


def test_lateral_rmse_reads_METRES_not_normalised_units():
    pred = torch.tensor([[1.0, 0.10], [1.0, -0.10]])
    tgt = torch.tensor([[1.0, 0.00], [1.0, 0.00]])
    arc = torch.full((2,), 30.0)
    got = lateral_rmse_m(pred, tgt, torch.ones(2), arc)
    assert float(got) == pytest.approx(3.0, rel=1e-5)      # 0.10 * 30 m


def test_config_refuses_an_inverted_band():
    with pytest.raises(ValueError):
        GoalPointConfig(arc_min_m=50.0, arc_max_m=10.0)


def test_provenance_answers_the_admissibility_check_and_names_its_label_source():
    p = goal_point_provenance()
    assert p["contains_situation_classifier_output"] is False
    assert p["situation_classifier_in_graph"] is False
    assert p["reads_tactical_state_or_logits"] is False
    assert p["supplied_or_predicted"] == "predicted"
    assert "FUTURE path" in p["label_source"] and "TRAIN ONLY" in p["label_source"]
    assert "horizon_lead_m" in p["leak_guard"]
    assert p["head_requirement_lateral_m"] == 1.0
    assert p["head_requirement_range_m"] == 2.0
    assert "situation classifier" in p["admissibility_check"]


# =========================================================================== TIME MODE
# The REGISTERED form. `raw/GP_LONG_AXIS_t4.json`: a goal point at t = 4 s recovers
# 57.1 % of the LONGITUDINAL selection ceiling, while the same goal with its RANGE
# stripped (a bearing — what the shipped S6 seam predicts, and what a route command is)
# is separated WORSE by +2.3632 m. These tests pin that distinction in code.

def test_the_leak_guard_REFUSES_a_goal_inside_the_scored_horizon():
    """⛔ The guard is a CONSTRUCTOR refusal, so no arm that leaks can be built."""
    GoalPointConfig(t_goal_s=3.0, t_pred_s=2.0)                  # fine
    for bad in (2.0, 1.5, 0.5):
        with pytest.raises(ValueError, match="strictly beyond"):
            GoalPointConfig(t_goal_s=bad, t_pred_s=2.0)


def test_goal_slot_is_DERIVED_from_the_model_horizons_and_refuses_an_off_grid_time():
    from tanitad.refs.refc_v3 import V3_HORIZONS
    assert goal_slot_index(4.0) == 5
    assert V3_HORIZONS[goal_slot_index(4.0)] == 40      # ticks at 10 Hz
    with pytest.raises(ValueError):
        goal_slot_index(2.5)                            # not a horizon -> refuse


def test_time_label_reads_a_KNOWN_value_and_refuses_an_invalid_sample():
    path = _straight_path(60, 1.0)                      # sample k is at (k+1) * 0.1 s
    ok_mask = np.ones(60, dtype=bool)
    p, ok = goal_point_label_at_time(path, ok_mask, 4.0)
    assert ok is True and p[0] == pytest.approx(40.0)
    ok_mask[39] = False
    p, ok = goal_point_label_at_time(path, ok_mask, 4.0)
    assert ok is False and np.array_equal(p, np.zeros(2, dtype=np.float32))
    p, ok = goal_point_label_at_time(path[:20], np.ones(20, dtype=bool), 4.0)
    assert ok is False, "a future that ends before t_goal has NO label"


def test_time_prior_is_EXACTLY_zero_on_an_invalid_goal():
    bank = _fan()
    s = anchor_goal_prior_at_time(torch.tensor([[40.0, 2.0]] * 4), torch.zeros(4),
                                  bank, slot=5, scale_m=40.0)
    assert torch.equal(s, torch.zeros_like(s))


def test_R_the_RANGE_half_is_what_a_BEARING_CANNOT_CARRY():
    """⛔⛔ MUTATION, and the load-bearing test of the whole time-mode design.

    A longitudinal fan: five STRAIGHT anchors that differ only in how far they get by
    the goal time. A bearing is IDENTICAL for all five by construction — that is the
    measured +2.3632 m catastrophe in miniature — while the metric prior ranks them and
    picks the right one. Both halves are asserted: without the first, the second would
    not be evidence about a MECHANISM.
    """
    rng_m = torch.tensor([20.0, 30.0, 40.0, 50.0, 60.0])
    bank = torch.zeros(3, 5, 8, 2)
    for j in range(8):
        bank[:, :, j, 0] = rng_m[None, :] * (j + 1) / 8.0
    b_ = bank[:, :, 5]                                            # [3, 5, 2] at slot 5
    ab = b_ / b_.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    assert torch.allclose(ab, ab[:, :1].expand_as(ab), atol=1e-6), (
        "the bearing must be IDENTICAL across the longitudinal fan — if it is not, "
        "the mutation did not take and the second half proves nothing")

    for k, want in enumerate([0, 2, 4]):
        goal = b_[:, want]
        sc = anchor_goal_prior_at_time(goal, torch.ones(3), bank, 5, 40.0)
        assert torch.equal(sc.argmax(-1), torch.full((3,), want)), (
            f"the metric prior must pick anchor {want} when the goal is at its range")
    # ⛔ and a RANGE corruption — the longitudinal deliberate-regression arm — must move it
    goal = b_[:, 4]
    a = anchor_goal_prior_at_time(goal, torch.ones(3), bank, 5, 40.0).argmax(-1)
    b = anchor_goal_prior_at_time(goal * 0.5, torch.ones(3), bank, 5, 40.0).argmax(-1)
    assert not torch.equal(a, b)


def test_range_rmse_reads_METRES_and_is_a_DIFFERENT_axis_from_lateral():
    pred = torch.tensor([[1.10, 0.0], [0.90, 0.0]])
    tgt = torch.tensor([[1.00, 0.0], [1.00, 0.0]])
    ok = torch.ones(2)
    assert float(range_rmse_m(pred, tgt, ok, 40.0)) == pytest.approx(4.0, rel=1e-5)
    assert float(lateral_rmse_m(pred, tgt, ok, 40.0)) == pytest.approx(0.0, abs=1e-6)


def test_encode_clips_a_high_speed_goal_rather_than_emitting_a_huge_feature():
    cfg = GoalPointConfig()
    f = encode_goal_point([1000.0, 900.0], True, cfg.range_norm_m, cfg)
    assert f[0] == pytest.approx(cfg.xy_clip)
    assert f[1] == pytest.approx(cfg.lat_clip)
    assert f[2] == 1.0
    # ⚠️ and the clip must NOT bind on a realistic goal — a clip that fires on ordinary
    # data is a silent information sink, not a safety net. 40 m/s * 4 s = 160 m is the
    # corpus maximum and reads 4.0; a 5 m lateral offset reads 0.125, well inside.
    g = encode_goal_point([144.0, 5.0], True, cfg.range_norm_m, cfg)
    assert g[0] == pytest.approx(3.6) and g[1] == pytest.approx(0.125)


def test_provenance_declares_the_time_mode_and_both_head_bars():
    p = goal_point_provenance()
    assert "fixed TIME" in p["form"] and "t_goal_s=4.0" in p["form"]
    assert p["head_requirement_lateral_m"] == 1.0
    assert p["head_requirement_range_m"] == 2.0
    assert "REFUSES" in p["leak_guard_time_mode"]
