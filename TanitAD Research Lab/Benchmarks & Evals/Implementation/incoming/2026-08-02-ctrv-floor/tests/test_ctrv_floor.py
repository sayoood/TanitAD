"""Sanity tests for the CTRV floor — synthetic cases with KNOWN ground truth.

Gate G-B2: every new metric ships with an analytic-ground-truth sanity test.
Every case here is constructed so the right answer is known in closed form
(a circular arc, a straight line, a standstill), never read off a model.

Standalone: ``pytest "<this package>/tests"`` — torch only, no stack imports.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctrv_floor import (  # noqa: E402
    WP_STEPS, build_floors, ctrv_waypoints, cv_waypoints, gt_ego_waypoints,
    hold_v0_waypoints, verify_alignment, window_starts)


# --------------------------------------------------------------------------- #
# synthetic pose generators — the ground truth is the geometry, not a fit      #
# --------------------------------------------------------------------------- #
def straight_poses(T=60, step=1.0):
    """Drive along +x at a constant 1 m per tick, yaw 0. poses [T,4]."""
    x = torch.arange(T, dtype=torch.float32) * step
    return torch.stack([x, torch.zeros(T), torch.zeros(T),
                        torch.full((T,), step * 10.0)], dim=-1)


def arc_poses(T=60, radius=20.0, step=1.0):
    """Drive a perfect circle of ``radius`` at ``step`` m per tick.

    Yaw is the TANGENT heading (what a real log carries). Positions are exact
    circle points, so CTRV — whose whole model is "constant speed, constant
    yaw rate" — is the correct predictor by construction and CV is not."""
    dth = step / radius
    th = torch.arange(T, dtype=torch.float32) * dth
    x = radius * torch.sin(th)
    y = radius * (1.0 - torch.cos(th))
    return torch.stack([x, y, th, torch.full((T,), step * 10.0)], dim=-1)


def reference_arc(poses, last, wp_steps=WP_STEPS):
    """Independent CTRV implementation: integrate heading explicitly.

    Deliberately written the OTHER way round from ``ctrv_waypoints`` (an
    accumulating loop rather than a vectorised sum) so that agreement is
    evidence about the formula, not a shared expression."""
    out = []
    for b in range(len(last)):
        i = int(last[b])
        d = poses[i, :2] - poses[i - 1, :2]
        speed = float(d.norm())
        omega = float((poses[i, 2] - poses[i - 1, 2] + math.pi)
                      % (2 * math.pi) - math.pi)
        px = py = 0.0
        head = 0.0
        pts, kmax = {}, max(wp_steps)
        for k in range(1, kmax + 1):
            px += speed * math.cos(head)
            py += speed * math.sin(head)
            head += omega
            pts[k] = (px, py)
        out.append([pts[k] for k in wp_steps])
    return torch.tensor(out, dtype=torch.float32)


def ade(pred, gt):
    return torch.linalg.norm(pred - gt, dim=-1).mean(1)


# --------------------------------------------------------------------------- #
# 1. the formula is what it claims to be                                       #
# --------------------------------------------------------------------------- #
def test_matches_independent_reference_implementation():
    poses = arc_poses()
    last = torch.tensor([5, 10, 20, 30])
    got = ctrv_waypoints(poses, last, v_gate=None)
    ref = reference_arc(poses, last)
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()


# --------------------------------------------------------------------------- #
# 2. THE POINT OF THE MODULE: on a turn, CTRV is right and CV is not           #
# --------------------------------------------------------------------------- #
def test_pure_arc_ctrv_beats_cv_by_an_order_of_magnitude():
    poses = arc_poses(radius=20.0, step=1.0)      # 10 m/s, 1 rad over 2 s
    last = torch.tensor([5, 10, 15, 20, 25, 30])
    gt = gt_ego_waypoints(poses, last)
    e_ctrv = ade(ctrv_waypoints(poses, last, v_gate=None), gt).mean()
    e_cv = ade(cv_waypoints(poses, last), gt).mean()
    # CTRV reproduces the arc to well under a car width; CV cannot turn at all.
    # MEASURED here: CTRV 0.3044 m, CV 5.19 m over the 2 s horizon on a 1 rad
    # arc. CTRV's residual is NOT noise — it is the forward-Euler half-step
    # bias (see test_forward_euler_half_step_bias_is_known_and_bounded); the
    # shipped floor keeps it deliberately, to stay identical to
    # driving_diagnostic.baseline_waypoints()["constant_yaw_rate"].
    assert float(e_ctrv) < 0.35, float(e_ctrv)
    assert float(e_cv) > 3.0, float(e_cv)
    assert float(e_ctrv) < 0.1 * float(e_cv)


def test_forward_euler_half_step_bias_is_known_and_bounded():
    """The floor's residual on a PERFECT arc is a documented O(omega/2) bias.

    ``ctrv_waypoints`` integrates chords at headings 0, w, 2w, ... while the
    true chord headings are w/2, 3w/2, ... — so on an exact circle it lags by
    half a step. Rotating the prediction by ``+omega/2`` removes almost all of
    it, which is what proves the residual is that bias and not a sign, frame or
    indexing error. A midpoint-corrected CTRV would be a STRICTLY STRONGER
    floor; that is a separate proposal, and it would only strengthen any
    conclusion drawn against the current one."""
    poses = arc_poses(radius=20.0, step=1.0)
    last = torch.tensor([10, 20, 30])
    gt = gt_ego_waypoints(poses, last)
    pred = ctrv_waypoints(poses, last, v_gate=None)
    w = 1.0 / 20.0                                    # the arc's yaw rate
    c, s = math.cos(w / 2), math.sin(w / 2)
    rot = torch.stack([pred[..., 0] * c - pred[..., 1] * s,
                       pred[..., 0] * s + pred[..., 1] * c], dim=-1)
    assert float(ade(rot, gt).mean()) < 0.2 * float(ade(pred, gt).mean())


def test_straight_line_ctrv_is_identical_to_cv():
    """No turn -> the extra parameter must cost nothing. This is the guard
    against a floor that is 'better' only because it is noisier."""
    poses = straight_poses()
    last = torch.tensor([5, 10, 20, 30])
    assert torch.allclose(ctrv_waypoints(poses, last, v_gate=None),
                          cv_waypoints(poses, last), atol=1e-5)


# --------------------------------------------------------------------------- #
# 3. the low-speed yaw-noise gate (the 2026-07-15 stratification artifact)     #
# --------------------------------------------------------------------------- #
def test_speed_gate_suppresses_standstill_yaw_noise():
    """At a crawl, one-step yaw is noise; an ungated CTRV curls for free."""
    T = 60
    x = torch.arange(T, dtype=torch.float32) * 0.01     # 0.1 m/s
    yaw = torch.zeros(T)
    yaw[1::2] = 0.2                                      # +-0.2 rad jitter
    poses = torch.stack([x, torch.zeros(T), yaw, torch.full((T,), 0.1)], -1)
    last = torch.tensor([5, 11, 21, 31])
    ungated = ctrv_waypoints(poses, last, v_gate=None)
    gated = ctrv_waypoints(poses, last, v_gate=2.0)
    assert float(ungated[..., 1].abs().max()) > 1e-4     # it does curl
    assert float(gated[..., 1].abs().max()) == 0.0       # gate removes it
    # and the gate leaves a genuinely moving window untouched
    fast = arc_poses(radius=20.0, step=1.0)
    lf = torch.tensor([10, 20])
    assert torch.allclose(ctrv_waypoints(fast, lf, v_gate=2.0),
                          ctrv_waypoints(fast, lf, v_gate=None))


# --------------------------------------------------------------------------- #
# 4. hold-v0 and the enumeration contract                                      #
# --------------------------------------------------------------------------- #
def test_hold_v0_is_speed_times_time_straight_ahead():
    v0 = torch.tensor([0.0, 5.0, 13.0])
    hv = hold_v0_waypoints(v0, n=4)
    assert hv.shape == (3, 4, 2)
    assert torch.allclose(hv[..., 1], torch.zeros(3, 4))          # no lateral
    assert torch.allclose(hv[:, :, 0],
                          v0[:, None] * torch.tensor([0.5, 1.0, 1.5, 2.0]))


def test_window_starts_replicates_collect():
    """`range(0, T - window - K_MAX, stride)` — exclusive bound included."""
    assert window_starts(199) == list(range(0, 199 - 8 - 20, 8))
    assert window_starts(28) == []            # exactly at the bound -> empty
    assert window_starts(29) == [0]
    assert all(t + 8 - 1 + 20 < 199 for t in window_starts(199))


def test_ctrv_refuses_last_zero():
    """The floor needs poses[last-1]; last=0 would silently wrap to poses[-1]."""
    with pytest.raises(ValueError):
        ctrv_waypoints(straight_poses(), torch.tensor([0, 5]))


# --------------------------------------------------------------------------- #
# 5. the alignment precondition (C63: measure it, never assume it)             #
# --------------------------------------------------------------------------- #
def _fake_episodes(n_ep=3, T=199):
    return [(i, arc_poses(T=T, radius=20.0 + 5 * i), T) for i in range(n_ep)]


def test_build_floors_shapes_and_ids():
    b = build_floors(_fake_episodes())
    n_per_ep = len(window_starts(199))
    assert b["gt"].shape == (3 * n_per_ep, 4, 2)
    assert len(b["eid"]) == 3 * n_per_ep
    assert b["eid"][:2] == [0, 0] and b["eid"][-1] == 2
    for k in ("cv", "ctrv", "ctrv_gated", "holdv0"):
        assert b[k].shape == b["gt"].shape


def test_verify_alignment_passes_on_self_and_fails_on_a_one_window_shift():
    b = build_floors(_fake_episodes())
    good = verify_alignment(b, {"gt": b["gt"], "cv": b["cv"], "eid": b["eid"]})
    assert good["aligned"] and good["max_abs_diff_cv"] == 0.0

    shifted = {"gt": b["gt"].roll(1, 0), "cv": b["cv"].roll(1, 0),
               "eid": b["eid"]}
    assert not verify_alignment(b, shifted)["aligned"]

    truncated = {"gt": b["gt"][:-1], "cv": b["cv"][:-1], "eid": b["eid"][:-1]}
    bad = verify_alignment(b, truncated)
    assert not bad["aligned"] and not bad["n_equal"]


def test_verify_alignment_accepts_a_relabelled_but_identical_partition():
    """The banked v4 dumps label the same 40 episodes with packed string uids.

    Same clustering, different label values, bit-identical tensors -> aligned,
    with the literal-label difference still reported so a harness that mixes
    two id encodings is visible rather than silent."""
    b = build_floors(_fake_episodes())
    relabelled = [808464434 + 17 * e for e in b["eid"]]
    rec = verify_alignment(b, {"gt": b["gt"], "cv": b["cv"],
                               "eid": relabelled})
    assert rec["aligned"]
    assert rec["eid_partition_equal"] and not rec["eid_labels_equal"]

    # a genuinely different clustering is still refused
    scrambled = list(b["eid"])
    scrambled[0] = 999
    assert not verify_alignment(b, {"gt": b["gt"], "cv": b["cv"],
                                    "eid": scrambled})["aligned"]
