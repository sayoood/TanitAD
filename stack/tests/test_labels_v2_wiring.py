"""Wiring tests for the v2 curvature-relative labels in the FLAGSHIP path.

The v2 label FUNCTIONS themselves are pinned by tests/test_refb_labels_v2.py;
this file pins their INTEGRATION into the training dataset — the ``labels_v2``
gate on ``FailLoudWindowDataset`` (shared, REF-B) / ``FlagshipWindowDataset``
(flagship), the ``cfg.v2_labels`` flag, and the ``--v2`` / ``--labels-v2`` /
``--no-labels-v2`` threading.

Pins:
  (a) labels_v2 OFF (the default) is BYTE-IDENTICAL to the v1 path: every batch
      field matches the pre-gate behavior AND a direct v1-function recompute —
      even on curve/junction episodes where v2 WOULD differ.
  (b) labels_v2 ON drives the v2 derivation: each field matches a direct
      v2-function recompute; the distribution genuinely shifts (road-following
      curves flip from route-TURN(valid) to STRAIGHT(invalid) => lower turn
      fraction + some nav_valid=False on ambiguous windows; gentle-fast curves
      flip maneuver TURN->lane_keep); and flagship_loss runs finite on a v2 batch
      (route + maneuver CE computed, the v2 valid mask applied, grads finite).
  (c) the flag threads end-to-end: cfg.v2_labels -> the dataset (via _wrap), and
      the trainer's --v2 sets it while --labels-v2 / --no-labels-v2 override.

CPU-only, synthetic contract episodes (mirrors test_flagship4b / test_vision_levers).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refb_labels as R  # noqa: E402
import train_flagship4b as T4  # noqa: E402
from train_flagship4b import FlagshipWindowDataset, _wrap  # noqa: E402

from tanitad.config import flagship4b_smoke_config  # noqa: E402
from tanitad.data._contract import assemble_episode  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402
from tanitad.train.flagship_losses import (LossWeights, build_grounding,  # noqa: E402
                                           flagship_loss, horizon_plan)

FAST = dict(op_fwd_k=2, tac_fwd_k=3, str_fwd_k=4)     # -> max_horizon 4, maneuver_h 4


# --------------------------------------------------------------------------- #
# synthetic kinematics (unicycle) -> contract episodes                        #
# --------------------------------------------------------------------------- #
def _poses(T, dt=0.1, v0=8.0, yaw_rate=0.0, accel=0.0):
    rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, v0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.0, v + accel * dt)
    return torch.tensor(rows, dtype=torch.float32)


def _junction_poses(T=220, sign=1.0, t0=30, dur=30, v_turn=8.0, R_m=15.0, dt=0.1):
    """Straight -> brief tight (R=15 m junction) turn -> straight: TRANSIENT +
    tight, so v2 reads a genuine route turn (valid) rather than a road sweep."""
    rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, 12.0
    for t in range(T):
        rows.append([x, y, yaw, v])
        turning = t0 <= t < t0 + dur
        yr = sign * v_turn / R_m if turning else 0.0
        v = v_turn if turning else 12.0
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yr * dt
    return torch.tensor(rows, dtype=torch.float32)


def _episode(poses, eid, size=64):
    T = poses.shape[0]
    g = torch.Generator().manual_seed(100 + eid)
    frames = [torch.rand(1, size, size, generator=g) for _ in range(T)]
    return assemble_episode(frames, [p.numpy() for p in poses],
                            [0.0] * T, 0.1, eid)


# Episode roster (fixed order). GRAY = a large-radius road-following curve
# (R=100 m) that v1 mislabels a route TURN but v2 flags AMBIGUOUS(invalid);
# GENTLE = a fast large-radius curve whose 2 s window trips v1's maneuver turn
# but not v2's curvature gate.
_ORDER = ["straight", "gray", "gentle", "accel", "jleft", "jright"]
_GRAY, _GENTLE = _ORDER.index("gray"), _ORDER.index("gentle")


def _eps_list(T=220):
    d = {
        "straight": _episode(_poses(T, v0=12.0), 0),
        "gray": _episode(_poses(T, v0=10.0, yaw_rate=0.1), 1),          # R=100 m
        "gentle": _episode(_poses(T, v0=40.0, yaw_rate=0.5), 2),        # kappa 0.0125
        "accel": _episode(_poses(T, v0=6.0, accel=1.0), 3),
        "jleft": _episode(_junction_poses(T, sign=1.0), 4),
        "jright": _episode(_junction_poses(T, sign=-1.0), 5),
    }
    return [d[n] for n in _ORDER]


def _ds(eps, cfg, plan, labels_v2):
    return FlagshipWindowDataset(
        eps, window=cfg.predictor.window, max_horizon=plan.max_horizon,
        maneuver_h=plan.maneuver_h, channels=cfg.encoder.in_channels,
        labels_v2=labels_v2)


def _ep_windows(ds, e_i, k):
    """First ``k`` flat dataset indices whose window lives in episode ``e_i``
    (early windows -> the most future -> route-scale-valid)."""
    return [i for i, (e, _t) in enumerate(ds.index) if e == e_i][:k]


def _sample(ds, per_ep=6):
    out, seen = [], {}
    for i, (e, _t) in enumerate(ds.index):
        if seen.get(e, 0) < per_ep:
            out.append(i)
            seen[e] = seen.get(e, 0) + 1
    return out


# --------------------------------------------------------------------------- #
# (a) OFF (default) is byte-identical to v1                                     #
# --------------------------------------------------------------------------- #
def test_off_default_is_byte_identical_to_v1():
    eps = _eps_list()
    cfg = flagship4b_smoke_config()
    plan = horizon_plan(cfg, **FAST)
    w, mh = cfg.predictor.window, plan.maneuver_h

    ds_default = _ds(eps, cfg, plan, labels_v2=False)   # explicit False
    ds_bare = FlagshipWindowDataset(                    # no labels_v2 kwarg at all
        eps, window=w, max_horizon=plan.max_horizon, maneuver_h=mh,
        channels=cfg.encoder.in_channels)
    assert ds_bare.labels_v2 is False                   # the DEFAULT is OFF

    for i in _sample(ds_default, per_ep=6):
        a, b = ds_bare[i], ds_default[i]
        assert set(a) == set(b)
        for k in a:                                     # default == explicit-False
            if torch.is_tensor(a[k]):
                assert torch.equal(a[k], b[k]), (i, k)
            else:
                assert a[k] == b[k], (i, k)
        # ...and the emitted labels equal a direct v1-function recompute
        e_i, t = ds_default.index[i]
        poses = ds_default.episodes[e_i].poses
        t_last = t + w - 1
        cmd, valid = R.nav_command(poses, t_last)
        assert int(a["nav_cmd"]) == cmd
        assert bool(a["nav_valid"]) == valid
        assert int(a["route_target"]) == R.route_target(cmd)
        pl, p1 = poses[t_last], poses[t_last + mh]
        man = R.classify_maneuver(pl[2], p1[2], pl[3], p1[3])
        assert int(a["maneuver_label"]) == int(man)


# --------------------------------------------------------------------------- #
# (b) ON drives the v2 derivation + a genuinely shifted distribution           #
# --------------------------------------------------------------------------- #
def test_on_matches_v2_recompute_every_field():
    eps = _eps_list()
    cfg = flagship4b_smoke_config()
    plan = horizon_plan(cfg, **FAST)
    w, mh = cfg.predictor.window, plan.maneuver_h
    ds2 = _ds(eps, cfg, plan, labels_v2=True)
    assert ds2.labels_v2 is True

    for i in _sample(ds2, per_ep=8):
        it = ds2[i]
        e_i, t = ds2.index[i]
        poses = ds2.episodes[e_i].poses
        t_last = t + w - 1
        cmd2, valid2 = R.nav_command_v2(poses, t_last)
        assert int(it["nav_cmd"]) == cmd2
        assert bool(it["nav_valid"]) == valid2
        assert int(it["route_target"]) == R.route_target_v2(poses, t_last)
        fut = poses[t_last + 1: t_last + 1 + plan.max_horizon]
        man2 = R.window_maneuver_labels_v2(poses[t_last][None], fut[None],
                                           horizon=mh)[0]
        assert int(it["maneuver_label"]) == int(man2)


def test_on_shifts_distribution_vs_v1():
    """v2 vs v1 on the SAME windows: the road-following curve stops being a
    valid route turn (=> lower turn fraction + nav_valid=False on the ambiguous
    windows) and the gentle-fast curve's maneuver stops being a turn."""
    eps = _eps_list()
    cfg = flagship4b_smoke_config()
    plan = horizon_plan(cfg, **FAST)
    w = cfg.predictor.window
    ds1 = _ds(eps, cfg, plan, labels_v2=False)
    ds2 = _ds(eps, cfg, plan, labels_v2=True)

    # road-following curve: every window v1 marks a VALID route turn, v2 instead
    # flags AMBIGUOUS -> nav_valid False, route straight (the 24.5%->0% fix).
    v1_false_turns = 0
    for i in _ep_windows(ds2, _GRAY, 10):
        e_i, t = ds2.index[i]
        poses = ds2.episodes[e_i].poses
        c1, val1 = R.nav_command(poses, t + w - 1)
        if val1 and R.route_target(c1) != R.ROUTE_STRAIGHT:      # v1 says turn
            v1_false_turns += 1
            it2 = ds2[i]
            assert bool(it2["nav_valid"]) is False               # v2 excludes it
            assert int(it2["route_target"]) == R.ROUTE_STRAIGHT
            assert int(it2["nav_cmd"]) == R.NAV_FOLLOW
    assert v1_false_turns >= 3, "gray curve should give v1 false-turns to fix"

    # gentle-fast curve: v1 maneuver turn, v2 curvature-gated to lane_keep.
    man_flips = 0
    for i in _ep_windows(ds2, _GENTLE, 10):
        m1, m2 = int(ds1[i]["maneuver_label"]), int(ds2[i]["maneuver_label"])
        if m1 in (R.TURN_LEFT, R.TURN_RIGHT) and m2 == R.LANE_KEEP:
            man_flips += 1
    assert man_flips >= 3, "gentle-fast curve should flip v1-turn -> v2-lane_keep"

    # aggregate: across all sampled windows the v2 turn fraction is <= v1's
    idxs = _sample(ds2, per_ep=8)
    def turn_frac(ds):
        n = sum(int(ds[i]["maneuver_label"]) in (R.TURN_LEFT, R.TURN_RIGHT)
                for i in idxs)
        return n / len(idxs)
    assert turn_frac(ds2) <= turn_frac(ds1)


def test_flagship_loss_runs_finite_on_v2_batch():
    """The whole integration point: feeding the v2 valid mask through nav_valid,
    flagship_loss computes route + maneuver CE finitely, the valid mask is
    applied (partial nav_valid_frac), the route-from-vision aux runs on the same
    mask, and gradients are finite."""
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    cfg.v2_labels = True
    cfg.v2_route_from_vision = True                # exercise the masked aux too
    plan = horizon_plan(cfg, **FAST)
    eps = _eps_list()
    ds = _wrap(eps, cfg, plan, cfg.encoder.in_channels)   # threads cfg.v2_labels
    assert ds.labels_v2 is True

    # a batch mixing route-valid (junction/straight) and route-INVALID (gray).
    pick = (_ep_windows(ds, _ORDER.index("jleft"), 2)
            + _ep_windows(ds, _GRAY, 2)
            + _ep_windows(ds, _ORDER.index("straight"), 2)
            + _ep_windows(ds, _ORDER.index("jright"), 2))
    batch = default_collate([ds[i] for i in pick])
    n_valid = int(batch["nav_valid"].sum())
    assert 0 < n_valid < len(pick), "v2 mask must be partial (gray excluded)"

    m = WorldModel(cfg)
    grounding = build_grounding(m.state_dim, hidden=32)
    states = m.encode_window(batch["frames"])
    fut_states = m.encode_window(batch["future_frames"][:, plan.needed_fut])
    total, log, parts = flagship_loss(
        m, grounding, batch, states, fut_states, plan, cfg,
        weights=LossWeights(), sigreg_variant="full_relaxed",
        sigreg_free_dims=cfg.loss.sigreg.free_dims, pose_scale=10.0,
        fwd_step_weight=0.5, device="cpu")

    assert torch.isfinite(total)
    for name, v in parts.items():
        assert torch.isfinite(v).all(), f"non-finite part {name}"
    assert math.isfinite(log["man"]) and math.isfinite(log["route"])
    assert math.isfinite(log["route_vis"])         # aux ran on the v2 mask
    assert log["route"] > 0.0                       # valid {L,S,R} set is non-empty
    assert 0.0 < log["nav_valid_frac"] < 1.0        # mask really is partial
    total.backward()
    bad = [n for n, p in m.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads: {bad[:5]}"


# --------------------------------------------------------------------------- #
# (c) the flag threads through _wrap and the trainer CLI                        #
# --------------------------------------------------------------------------- #
def test_wrap_threads_cfg_v2_labels():
    cfg = flagship4b_smoke_config()
    plan = horizon_plan(cfg, **FAST)
    eps = _eps_list()
    assert _wrap(eps, cfg, plan, cfg.encoder.in_channels).labels_v2 is False
    cfg.v2_labels = True
    assert _wrap(eps, cfg, plan, cfg.encoder.in_channels).labels_v2 is True


def _run_main_cfg(tmp_path, name, extra):
    """Run the trainer to the config.json write (steps=0, no training loop) and
    return the persisted StackConfig dict."""
    out = tmp_path / name
    T4.main(["--data", "toy", "--config", "smoke", "--out", str(out),
             "--episodes", "6", "--steps", "0", "--batch-size", "4",
             "--op-fwd-k", "2", "--tac-fwd-k", "3", "--str-fwd-k", "4",
             *extra])
    outer = json.loads((out / "config.json").read_text(encoding="utf-8"))
    return json.loads(outer["cfg"])


def test_v2_flag_sets_v2_labels(tmp_path):
    assert _run_main_cfg(tmp_path, "v2", ["--v2"])["v2_labels"] is True


def test_no_labels_v2_overrides_v2(tmp_path):
    cfg = _run_main_cfg(tmp_path, "v2_ctrl", ["--v2", "--no-labels-v2"])
    assert cfg["v2_labels"] is False                # v2 model levers, v1 labels


def test_labels_v2_alone_without_v2(tmp_path):
    cfg = _run_main_cfg(tmp_path, "labels_only", ["--labels-v2"])
    assert cfg["v2_labels"] is True                 # v2 labels on an otherwise-v1 run
