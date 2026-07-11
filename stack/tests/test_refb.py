"""REF-B tests (tanitad/refs/refb.py + scripts/refb_train.py + refb_labels.py).

Pins the REF-B spec (REFERENCE_ARCHITECTURES + 2026-07-11 4-layer upgrade):
(a) budget-match +-2 % vs the main base250cam total (read programmatically),
(b) a training step runs with finite losses and the confidence head is fully
    detached (no gradient into encoder/heads),
(c) pseudo-label correctness on synthetic maneuvers,
(d) the FiLM intent-conditioning path is live (sensitivity check),
(e) fail-loud windowing raises on short/misaligned episodes,
(f) ckpt save/load/resume bit-exact on fixed input.
CPU-only, synthetic data.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refb_labels  # noqa: E402  (scripts/refb_labels.py)
import refb_train  # noqa: E402  (scripts/refb_train.py)
from tanitad.config import base250cam_config  # noqa: E402
from tanitad.data._contract import assemble_episode  # noqa: E402
from tanitad.data.mixing import save_episode  # noqa: E402
from tanitad.refs.refb import (RefBModel, param_breakdown,  # noqa: E402
                               refb_config, refb_smoke_config)

# ---------- synthetic kinematics ---------------------------------------------


def _poses(T: int, dt: float = 0.1, v0: float = 8.0, yaw_rate: float = 0.0,
           accel: float = 0.0, yaw0: float = 0.0) -> torch.Tensor:
    """Unicycle rollout -> contract poses [T, 4] = (x, y, yaw, v)."""
    rows, x, y, yaw, v = [], 0.0, 0.0, yaw0, v0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.0, v + accel * dt)
    return torch.tensor(rows, dtype=torch.float32)


def _drive_episode(T: int, eid: int, yaw_rate: float = 0.0,
                   accel: float = 0.0, v0: float = 8.0, size: int = 64,
                   dt: float = 0.1):
    """Kinematically consistent contract episode (1-channel, smoke-sized)."""
    g = torch.Generator().manual_seed(1000 + eid)
    frames = [torch.rand(1, size, size, generator=g) for _ in range(T)]
    poses = _poses(T, dt=dt, v0=v0, yaw_rate=yaw_rate, accel=accel)
    return assemble_episode(frames, [p.numpy() for p in poses],
                            [yaw_rate] * T, dt, eid)


def _make_cached_root(tmp_path: Path, n_train: int = 3, n_val: int = 1,
                      T: int = 40) -> Path:
    """Synthetic cached-mode root: *train*/*val* dirs of ep_*.pt."""
    specs = [(0.0, 0.0), (0.25, 0.0), (0.0, -1.2)]      # keep / left / brake
    for split, n in (("train", n_train), ("val", n_val)):
        d = tmp_path / f"toy-{split}"
        d.mkdir()
        for i in range(n):
            yr, ac = specs[i % len(specs)]
            ep = _drive_episode(T, eid=i, yaw_rate=yr, accel=ac)
            save_episode(ep, str(d / f"ep_{i:05d}.pt"))
    return tmp_path


def _batch(root: Path, cfg, n: int = 4):
    eps, _ = refb_train.load_cached_episodes(str(root), "*train*")
    max_h = max(max(cfg.tactical.waypoint_horizons),
                cfg.operative.action_seq - 1)
    ds = refb_train.FailLoudWindowDataset(
        eps, window=cfg.window, max_horizon=max_h,
        channels=cfg.encoder.in_channels)
    return torch.utils.data.default_collate([ds[i] for i in range(n)])


# ---------- (a) budget-matched within +-2 % of the MAIN model ----------------

def test_budget_matched_within_2pct():
    """Total trainable params within +-2 % of base250cam's WorldModel total —
    both read programmatically (meta device: no weight memory needed)."""
    from tanitad.models.fourbrain import WorldModel
    with torch.device("meta"):
        main = WorldModel(base250cam_config())
        refb = RefBModel(refb_config())
    n_main = sum(p.numel() for p in main.parameters() if p.requires_grad)
    bd = param_breakdown(refb)
    n_refb = sum(p.numel() for p in refb.parameters() if p.requires_grad)
    assert bd["total"] == n_refb                    # breakdown covers all
    assert bd["encoder"] + bd["operative"] + bd["tactical"] \
        + bd["strategic"] + bd["fallback"] == bd["total"]
    rel = abs(n_refb - n_main) / n_main
    assert rel <= 0.02, (f"budget mismatch: REF-B {n_refb:,} vs main "
                         f"{n_main:,} ({rel:+.3%})")
    # Structural pins: thin strategic, zero-parameter OOD monitor.
    assert bd["strategic"] < 10_000
    assert sum(p.numel() for p in refb.ood.parameters()) == 0


# ---------- (b) one training step; confidence head fully detached -------------

def test_train_step_finite_and_confidence_detached(tmp_path):
    root = _make_cached_root(tmp_path)
    cfg = refb_smoke_config()
    torch.manual_seed(0)
    model = RefBModel(cfg)
    batch = _batch(root, cfg)

    # Confidence-only backward: NO parameter outside the confidence head may
    # receive a gradient (detachment of inputs AND targets, spec item 5a).
    out = refb_train.compute_losses(model, batch)
    out["conf"].backward()
    for name, p in model.named_parameters():
        if name.startswith("confidence."):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None, f"conf grad leaked into {name}"

    # Full loss: every component finite, every parameter trained.
    model.zero_grad(set_to_none=True)
    out = refb_train.compute_losses(model, batch)
    for key in ("loss", "action", "seq", "wp", "man", "inv", "conf",
                "conf_mae", "man_acc"):
        assert torch.isfinite(out[key].detach()), key
    out["loss"].backward()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    opt.step()
    # OOD monitor never enters the graph and holds no trainable params.
    z = out["states"][:, -1].detach()
    model.ood.update(z)
    s = model.ood.score(z)
    assert s.shape == (z.shape[0],) and torch.isfinite(s).all()


# ---------- (c) pseudo-label correctness on synthetic maneuvers ---------------

H = refb_labels.LABEL_HORIZON                        # 20 steps = 2 s @ 10 Hz


def test_labels_lane_keep_straight_constant_speed():
    labs = refb_labels.maneuver_labels(_poses(40), H)
    assert (labs == refb_labels.LANE_KEEP).all()


def test_labels_turns_left_and_right():
    left = refb_labels.maneuver_labels(_poses(40, yaw_rate=0.2), H)
    right = refb_labels.maneuver_labels(_poses(40, yaw_rate=-0.2), H)
    assert (left == refb_labels.TURN_LEFT).all()     # dyaw +0.4 > 0.15
    assert (right == refb_labels.TURN_RIGHT).all()
    # gentle highway-curve drift stays lane_keep (below threshold)
    drift = refb_labels.maneuver_labels(_poses(40, yaw_rate=0.03), H)
    assert (drift == refb_labels.LANE_KEEP).all()    # dyaw 0.06 < 0.15


def test_labels_accelerate_and_brake():
    acc = refb_labels.maneuver_labels(_poses(40, accel=1.0), H)
    brk = refb_labels.maneuver_labels(_poses(40, accel=-1.0), H)
    assert (acc == refb_labels.ACCELERATE).all()     # dv +2.0 > 1.0
    assert (brk == refb_labels.BRAKE_STOP).all()     # dv -2.0 < -1.0
    # braking TO a stop counts even when |dv| is small (stop condition)
    stop = refb_labels.maneuver_labels(_poses(40, v0=1.2, accel=-0.48), H)
    assert stop[0] == refb_labels.BRAKE_STOP         # v: 1.2 -> ~0.24 < 0.3


def test_labels_turn_priority_and_yaw_wrap():
    # a braking turn is a TURN (priority order)
    both = refb_labels.maneuver_labels(_poses(40, yaw_rate=0.2, accel=-1.0), H)
    assert (both == refb_labels.TURN_LEFT).all()
    # yaw wrap across +-pi must not flip the turn direction: store WRAPPED
    # yaw (as a pose source would), so the raw difference across the +pi
    # crossing is ~ -2*pi + 0.4 and only wrap_to_pi recovers the left turn.
    p = _poses(40, yaw_rate=0.2, yaw0=math.pi - 0.05)
    p[:, 2] = refb_labels.wrap_to_pi(p[:, 2])
    assert float(p[:, 2].min()) < 0 < float(p[:, 2].max())   # crossing happened
    wrap = refb_labels.maneuver_labels(p, H)
    assert (wrap == refb_labels.TURN_LEFT).all()


def test_waypoint_targets_ego_frame_convention():
    # straight at 8 m/s with a NONZERO initial yaw: ego waypoints must be
    # (v*k*dt, ~0) — the d1_probe `_ego` rotation removes the world yaw.
    poses = _poses(40, v0=8.0, yaw0=1.0)
    horizons = (5, 10, 15, 20)
    wp = refb_labels.waypoint_targets(poses[:1], poses[1:21].unsqueeze(0),
                                      horizons)
    assert wp.shape == (1, 4, 2)
    for j, k in enumerate(horizons):
        assert abs(float(wp[0, j, 0]) - 8.0 * k * 0.1) < 1e-2, k
        assert abs(float(wp[0, j, 1])) < 1e-2, k
    # left arc: lateral ego displacement positive (+y = left)
    poses_l = _poses(40, v0=8.0, yaw_rate=0.3)
    wp_l = refb_labels.waypoint_targets(poses_l[:1], poses_l[1:21].unsqueeze(0),
                                        horizons)
    assert float(wp_l[0, -1, 1]) > 0.5
    # window-level labels agree with the episode-level function
    ep_labs = refb_labels.maneuver_labels(poses_l, H)
    win_lab = refb_labels.window_maneuver_labels(poses_l[:1],
                                                 poses_l[1:1 + H].unsqueeze(0))
    assert int(win_lab[0]) == int(ep_labs[0])


# ---------- (d) FiLM intent-conditioning path is live -------------------------

def test_intent_film_conditioning_sensitivity():
    torch.manual_seed(0)
    cfg = refb_smoke_config()
    model = RefBModel(cfg).eval()
    states = torch.randn(2, cfg.window, model.state_dim)
    i1 = torch.randn(2, cfg.tactical.d_intent)
    i2 = torch.randn(2, cfg.tactical.d_intent)
    with torch.no_grad():
        # At init FiLM is zero-init (identity start, main-stack convention):
        # intent has exactly NO effect yet — pin that too.
        assert torch.equal(model.operative(states, i1),
                           model.operative(states, i2))
        # With live FiLM weights the intent token must steer the operative
        # output (the conditioning PATH, independent of training state).
        for blk in model.operative.blocks:
            nn.init.normal_(blk.film.to_scale_shift.weight, std=0.1)
        o1 = model.operative(states, i1)
        o2 = model.operative(states, i2)
        assert float((o1 - o2).abs().max()) > 1e-4
        assert torch.equal(o1, model.operative(states, i1))  # deterministic


# ---------- (e) fail-loud windowing --------------------------------------------

@dataclass
class _RawEp:
    frames: torch.Tensor
    actions: torch.Tensor
    poses: torch.Tensor
    episode_id: int


def test_failloud_raises_on_short_episode():
    ok = _drive_episode(40, eid=0)
    short = _drive_episode(10, eid=1)                # < window+max_h+1 = 25
    with pytest.raises(ValueError, match="too short"):
        refb_train.FailLoudWindowDataset([ok, short], window=4,
                                         max_horizon=20, channels=1)
    # a valid set builds, with the shared index convention
    ds = refb_train.FailLoudWindowDataset([ok], window=4, max_horizon=20,
                                          channels=1)
    assert len(ds) == 40 - 4 - 20
    item = ds[0]
    assert item["frames"].shape == (4, 1, 64, 64)
    assert item["future_poses"].shape == (20, 4)


def test_failloud_raises_on_misalignment_and_channels():
    T = 40
    mis = _RawEp(frames=torch.zeros(T, 1, 8, 8), actions=torch.zeros(T - 3, 2),
                 poses=torch.zeros(T, 4), episode_id=9)
    with pytest.raises(ValueError, match="misaligned"):
        refb_train.FailLoudWindowDataset([mis], window=4, max_horizon=20,
                                         channels=1)
    wrong_c = _RawEp(frames=torch.zeros(T, 9, 8, 8),
                     actions=torch.zeros(T, 2), poses=torch.zeros(T, 4),
                     episode_id=10)
    with pytest.raises(ValueError, match="channels"):
        refb_train.FailLoudWindowDataset([wrong_c], window=4, max_horizon=20,
                                         channels=1)
    with pytest.raises(ValueError, match="window"):
        refb_train.build_window_index([40], 0, 20)


# ---------- (f) ckpt save/load/resume bit-exact --------------------------------

def test_ckpt_roundtrip_and_resume(tmp_path):
    root = _make_cached_root(tmp_path)
    out_dir = tmp_path / "run"
    argv = ["--data-root", str(root), "--out", str(out_dir), "--steps", "2",
            "--batch", "4", "--lr", "1e-3", "--episodes", "0",
            "--log-every", "1", "--ood-warmup", "1", "--device", "cpu",
            "--smoke"]
    metrics = refb_train.main(argv)
    assert metrics["final"]["step"] == 1
    for k in ("loss", "action", "seq", "wp", "man", "inv", "conf",
              "conf_mae", "ood_score"):
        assert np.isfinite(metrics["final"][k]), k
    assert metrics["final"]["ood_frozen"] is True    # warmup=1 < 2 steps
    ckpt_path = out_dir / "ckpt.pt"
    assert ckpt_path.exists()

    # Two fresh loads reproduce outputs bit-exactly on a fixed input.
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = refb_smoke_config()
    m1, m2 = RefBModel(cfg), RefBModel(cfg)
    m1.load_state_dict(ck["model"])
    m2.load_state_dict(ck["model"])
    m1.eval(), m2.eval()
    assert bool(m1.ood.frozen)                       # OOD stats travel in ckpt
    torch.manual_seed(123)
    fixed = torch.rand(2, cfg.window, 1, 64, 64)
    with torch.no_grad():
        o1, o2 = m1(fixed), m2(fixed)
    for key in ("action_seq", "maneuver_logits", "intent", "conf_pred"):
        assert torch.equal(o1[key], o2[key]), key
        assert torch.isfinite(o1[key]).all(), key
    assert torch.equal(m1.ood.score(o1["states"][:, -1]),
                       m2.ood.score(o2["states"][:, -1]))

    # Resume: rerun with more steps — picks up at step 2, finishes at 3,
    # frozen OOD stats bit-identical (no re-accumulation after freeze).
    metrics2 = refb_train.main(argv[:5] + ["4"] + argv[6:])
    assert metrics2["final"]["step"] == 3
    ck2 = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert ck2["step"] == 3
    for buf in ("ood.sum", "ood.sum_sq", "ood.count", "ood.frozen"):
        assert torch.equal(ck2["model"][buf], ck["model"][buf]), buf
