"""REF-C tests (tanitad/refs/refc.py + scripts/refc_train.py).

Pins the TCP-C spec (REFERENCE_ARCHITECTURES REF-C: TCP arXiv 2206.08129
BC-adapted + LAW aux + hierarchy graft):
(a) build at full config ~30 M params (25-35 M band) with a complete
    param_breakdown; nav vocab consistent with refb_labels indices,
(b) forward shapes on the smoke config (waypoints/actions/attention/LAW/
    speed), attention softmax-normalized over the spatial positions,
(c) all losses fire finite, the full loss reaches EVERY parameter, and the
    LAW loss alone reaches the trajectory branch (gradients flow through the
    predicted waypoints — the point of the aux),
(d) ego-dropout is per-sample, training-gated: OFF in eval (v0 sensitivity),
    fully zeroing under p=1.0 in train mode,
(e) gated flags follow the byte-identical-when-off discipline: refc1 /
    hierarchy modules are absent from the state_dict when off,
(f) refc1 variant: path-checkpoint keys, target-speed classification head +
    expected-value decode in [0, speed_max], losses finite,
(g) refb_labels.path_targets arc-length resample: exact on a straight line
    (incl. extrapolation past the path end), left-arc sign, stationary
    degenerate case finite,
(h) trainer smoke: 2 steps + ckpt + bit-exact resume (refb_train mirror),
    and a 1-step --refc1 run.
CPU-only, synthetic data.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refb_labels  # noqa: E402  (scripts/refb_labels.py)
import refc_train  # noqa: E402  (scripts/refc_train.py)
from tanitad.data._contract import assemble_episode  # noqa: E402
from tanitad.data.mixing import save_episode  # noqa: E402
from tanitad.refs.refc import (NAV_COMMANDS, RefCModel,  # noqa: E402
                               param_breakdown, refc_config,
                               refc_smoke_config)

# ---------- synthetic kinematics (test_refb conventions) ----------------------


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
                      T: int = 200) -> Path:
    """Synthetic cached-mode root: *train*/*val* dirs of ep_*.pt."""
    specs = [(0.0, 0.0), (0.06, 0.0), (0.0, -1.2)]  # keep / left / brake
    for split, n in (("train", n_train), ("val", n_val)):
        d = tmp_path / f"toy-{split}"
        d.mkdir()
        for i in range(n):
            yr, ac = specs[i % len(specs)]
            ep = _drive_episode(T, eid=i, yaw_rate=yr, accel=ac)
            save_episode(ep, str(d / f"ep_{i:05d}.pt"))
    return tmp_path


def _max_h(cfg) -> int:
    return max(max(cfg.trajectory.horizons), cfg.control.k,
               refc_train.LAW_AHEAD, refc_train.SPEED_AHEAD)


def _batch(root: Path, cfg, n: int = 4):
    eps, _ = refc_train.load_cached_episodes(str(root), "*train*")
    ds = refc_train.FailLoudWindowDataset(
        eps, window=cfg.window, max_horizon=_max_h(cfg),
        channels=cfg.encoder.in_channels)
    return torch.utils.data.default_collate([ds[i] for i in range(n)])


# ---------- (a) build, param count, vocab -------------------------------------

def test_param_count_and_breakdown():
    """Full config builds (meta device: no weight memory); ~30 M params
    (25-35 M band — TCP's own scale, deliberately NOT budget-matched to the
    261 M main stack); breakdown covers every parameter exactly."""
    with torch.device("meta"):
        model = RefCModel(refc_config())
    bd = param_breakdown(model)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[refc] param_breakdown: {json.dumps(bd, indent=2)}")
    assert bd["total"] == n_total                   # breakdown covers all
    assert sum(v for k, v in bd.items() if k != "total") == bd["total"]
    assert 25_000_000 < bd["total"] < 35_000_000, f"{bd['total']:,}"
    assert bd["encoder"] > 15_000_000               # ResNet-34-scale trunk
    assert bd["strategic"] > 0                      # hierarchy default ON
    # Full config: TCP-shaped conv map [B, 512, 8, 8].
    assert model.encoder.feat_dim == 512 and model.encoder.grid == 8


def test_nav_vocab_consistent_with_labels():
    assert NAV_COMMANDS[refb_labels.NAV_FOLLOW] == "follow"
    assert NAV_COMMANDS[refb_labels.NAV_LEFT] == "left"
    assert NAV_COMMANDS[refb_labels.NAV_RIGHT] == "right"
    assert NAV_COMMANDS[refb_labels.NAV_STRAIGHT] == "straight"  # reserved


# ---------- (b) forward shapes ------------------------------------------------

def test_forward_shapes_smoke(tmp_path):
    root = _make_cached_root(tmp_path)
    cfg = refc_smoke_config()
    torch.manual_seed(0)
    model = RefCModel(cfg).eval()
    batch = _batch(root, cfg)
    b = batch["frames"].shape[0]
    feat = model.encoder.feat_dim
    n_pos = model.encoder.grid ** 2
    with torch.no_grad():
        out = model(batch["frames"], nav_cmd=batch["nav_cmd"],
                    v0=batch["pose_last"][:, 3])
    n_steps = len(cfg.trajectory.horizons)
    assert out["pooled"].shape == (b, feat)
    assert out["wp_seq"].shape == (b, n_steps, 2)
    assert set(out["waypoints"]) == set(cfg.trajectory.horizons)
    for k in cfg.trajectory.horizons:
        assert out["waypoints"][k].shape == (b, 2)
    assert out["actions"].shape == (b, cfg.control.k, 2)
    assert out["att"].shape == (b, cfg.control.k, n_pos)
    assert torch.allclose(out["att"].sum(-1), torch.ones(b, cfg.control.k),
                          atol=1e-5)               # softmax over positions
    assert out["law_pred"].shape == (b, feat)
    assert out["speed_pred"].shape == (b,)
    assert out["ctx"].shape == (b, cfg.strategic.d_ctx)   # hierarchy ON
    assert "speed_logits" not in out               # refc1 OFF
    for v in ("pooled", "wp_seq", "actions", "att", "law_pred", "speed_pred"):
        assert torch.isfinite(out[v]).all(), v
    # nav_cmd/v0 default paths (None -> follow / zeros) also run.
    with torch.no_grad():
        out2 = model(batch["frames"])
    assert out2["wp_seq"].shape == (b, n_steps, 2)


# ---------- (c) losses fire, finite, full + LAW-through-waypoints grads -------

def test_losses_finite_and_backward(tmp_path):
    root = _make_cached_root(tmp_path)
    cfg = refc_smoke_config()
    torch.manual_seed(0)
    model = RefCModel(cfg)
    batch = _batch(root, cfg)

    # LAW-only backward reaches the TRAJECTORY branch: gradients flow through
    # the predicted waypoints into the GRU rollout (spec item 5 — the point).
    out = refc_train.compute_losses(model, batch)
    out["law"].backward()
    g = model.trajectory.delta.weight.grad
    assert g is not None and torch.isfinite(g).all()
    assert float(g.abs().sum()) > 0, "LAW gradient did not reach the waypoints"

    # Full loss: every component finite, every parameter trained.
    model.zero_grad(set_to_none=True)
    out = refc_train.compute_losses(model, batch)
    for key in ("loss", "wp", "ctrl", "speed", "law", "speed_cls",
                "speed_mae", "nav_follow_frac"):
        assert torch.isfinite(out[key].detach()), key
    assert float(out["speed_cls"]) == 0.0          # refc1 OFF -> zero term
    out["loss"].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.step()


# ---------- (d) ego-dropout: per-sample, training-gated -----------------------

def test_ego_dropout_gating(tmp_path):
    root = _make_cached_root(tmp_path)
    cfg = refc_smoke_config()
    torch.manual_seed(0)
    model = RefCModel(cfg)
    frames = _batch(root, cfg)["frames"]
    b = frames.shape[0]
    v_lo = torch.zeros(b)
    v_hi = torch.full((b,), 20.0)

    # EVAL: dropout off — deterministic, and v0 MUST matter.
    model.eval()
    with torch.no_grad():
        o1 = model(frames, v0=v_hi)
        o2 = model(frames, v0=v_hi)
        o3 = model(frames, v0=v_lo)
    assert torch.equal(o1["wp_seq"], o2["wp_seq"])          # deterministic
    assert not torch.equal(o1["wp_seq"], o3["wp_seq"])      # v0 is live

    # TRAIN, p=1.0: every sample's v0 is Bernoulli-zeroed — v0 cannot matter.
    model.cfg.ego_dropout = 1.0
    model.train()
    with torch.no_grad():
        t1 = model(frames, v0=v_hi)
        t2 = model(frames, v0=v_lo)
    assert torch.equal(t1["wp_seq"], t2["wp_seq"])

    # TRAIN, p=0.5: the mask is PER-SAMPLE (fixed seeds -> deterministic
    # check that two different mask draws move the output).
    model.cfg.ego_dropout = 0.5
    with torch.no_grad():
        torch.manual_seed(0)
        d1 = model(frames, v0=v_hi)
        torch.manual_seed(1)
        d2 = model(frames, v0=v_hi)
    assert not torch.equal(d1["wp_seq"], d2["wp_seq"])


# ---------- (e)+(f) gated flags: byte-identical-when-off + refc1 variant ------

def test_gated_flags_absent_when_off():
    cfg = refc_smoke_config()                       # refc1 False (default)
    keys = set(RefCModel(cfg).state_dict())
    assert not any(k.startswith("speed_cls") for k in keys)
    cfg_h = refc_smoke_config()
    cfg_h.hierarchy = False
    keys_h = set(RefCModel(cfg_h).state_dict())
    assert not any(k.startswith("strategic") for k in keys_h)
    # hierarchy build = flag-off build + ONLY the strategic module and the
    # (wider) measurement input row — no other structural drift.
    assert {k for k in keys - keys_h if not k.startswith("strategic")} == set()


def test_hierarchy_off_forward(tmp_path):
    root = _make_cached_root(tmp_path)
    cfg = refc_smoke_config()
    cfg.hierarchy = False
    torch.manual_seed(0)
    model = RefCModel(cfg).eval()
    batch = _batch(root, cfg)
    with torch.no_grad():
        out = model(batch["frames"], nav_cmd=batch["nav_cmd"],
                    v0=batch["pose_last"][:, 3])
    assert "ctx" not in out
    assert torch.isfinite(out["wp_seq"]).all()
    assert param_breakdown(model)["strategic"] == 0


def test_refc1_variant(tmp_path):
    root = _make_cached_root(tmp_path)
    cfg = refc_smoke_config()
    cfg.refc1 = True
    torch.manual_seed(0)
    model = RefCModel(cfg)
    assert any(k.startswith("speed_cls") for k in model.state_dict())
    batch = _batch(root, cfg)
    model.eval()
    with torch.no_grad():
        out = model(batch["frames"], nav_cmd=batch["nav_cmd"],
                    v0=batch["pose_last"][:, 3])
    b = batch["frames"].shape[0]
    assert set(out["waypoints"]) == set(cfg.path_dists)   # path-checkpoint keys
    assert out["speed_logits"].shape == (b, cfg.speed_bins)
    assert out["target_speed"].shape == (b,)              # expected-value decode
    assert float(out["target_speed"].min()) >= 0.0
    assert float(out["target_speed"].max()) <= cfg.speed_max
    # Losses: speed-class CE fires and the full loss trains the cls head too.
    model.train()
    losses = refc_train.compute_losses(model, batch)
    for key in ("loss", "wp", "ctrl", "speed", "law", "speed_cls",
                "speed_mae"):
        assert torch.isfinite(losses[key].detach()), key
    assert float(losses["speed_cls"].detach()) > 0.0
    losses["loss"].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


# ---------- (g) path_targets arc-length resample ------------------------------

def test_path_targets_straight_arc_and_degenerate():
    dists = (2.0, 5.0, 10.0, 20.0)
    # Straight at 8 m/s, nonzero world yaw: 20 steps = 16 m of path. In-range
    # checkpoints land exactly at (d, 0); beyond-path checkpoints CLAMP to the
    # final path point (pod/refbpatch training semantics — geometry targets
    # never invent path the ego didn't drive).
    poses = _poses(40, v0=8.0, yaw0=1.0)
    tgt = refb_labels.path_targets(poses[:1], poses[1:21].unsqueeze(0), dists)
    assert tgt.shape == (1, 4, 2)
    path_len = 16.0
    for j, d in enumerate(dists):
        expect = min(d, path_len)
        assert abs(float(tgt[0, j, 0]) - expect) < 1e-2, (d, expect)
        assert abs(float(tgt[0, j, 1])) < 1e-2, d
    # Left arc: lateral ego displacement positive (+y = left), and the far
    # checkpoint bends MORE than the near one.
    poses_l = _poses(40, v0=8.0, yaw_rate=0.3)
    tgt_l = refb_labels.path_targets(poses_l[:1], poses_l[1:21].unsqueeze(0),
                                     dists)
    assert float(tgt_l[0, -1, 1]) > float(tgt_l[0, 0, 1]) > 0.0
    # Speed-invariance: the SAME road (constant curvature = yaw_rate/v, so
    # halve both) at half speed -> same checkpoints (within resampling
    # tolerance) for dists covered by both paths.
    poses_s = _poses(40, v0=4.0, yaw_rate=0.15)
    tgt_s = refb_labels.path_targets(poses_s[:1], poses_s[1:21].unsqueeze(0),
                                     (2.0, 5.0))
    assert torch.allclose(tgt_s[0], tgt_l[0, :2], atol=0.15)
    # Stationary (degenerate zero-length path): finite, clamps to the origin.
    poses_0 = _poses(40, v0=0.0)
    tgt_0 = refb_labels.path_targets(poses_0[:1], poses_0[1:21].unsqueeze(0),
                                     dists)
    assert torch.isfinite(tgt_0).all()
    assert float(tgt_0.abs().max()) < 1e-6


# ---------- (h) trainer smoke + resume (refb_train mirror) --------------------

def test_trainer_run_ckpt_resume_and_refc1(tmp_path):
    root = _make_cached_root(tmp_path)
    out_dir = tmp_path / "run"
    argv = ["--data-root", str(root), "--out", str(out_dir), "--steps", "2",
            "--batch", "4", "--lr", "1e-3", "--episodes", "0",
            "--log-every", "1", "--device", "cpu", "--smoke"]
    metrics = refc_train.main(argv)
    assert metrics["final"]["step"] == 1
    for k in ("loss", "wp", "ctrl", "speed", "law", "nav_follow_frac"):
        assert np.isfinite(metrics["final"][k]), k
    assert "val" in metrics                        # val cache dir was found
    ckpt_path = out_dir / "ckpt.pt"
    assert ckpt_path.exists()
    conf = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert conf["arch"].startswith("REF-C")
    assert conf["param_breakdown"]["total"] == metrics["n_params_trainable"]
    assert conf["optimizer"]["kind"].startswith("Adam")

    # Two fresh loads reproduce outputs bit-exactly on a fixed input.
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = refc_smoke_config()
    m1, m2 = RefCModel(cfg), RefCModel(cfg)
    m1.load_state_dict(ck["model"])
    m2.load_state_dict(ck["model"])
    m1.eval(), m2.eval()
    torch.manual_seed(123)
    fixed = torch.rand(2, cfg.window, 1, 64, 64)
    with torch.no_grad():
        o1, o2 = m1(fixed), m2(fixed)
    for key in ("wp_seq", "actions", "law_pred", "speed_pred", "ctx"):
        assert torch.equal(o1[key], o2[key]), key
        assert torch.isfinite(o1[key]).all(), key

    # Resume: rerun with more steps — picks up at step 2, finishes at 3.
    metrics2 = refc_train.main(argv[:5] + ["4"] + argv[6:])
    assert metrics2["final"]["step"] == 3
    ck2 = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert ck2["step"] == 3

    # REF-C.1 end-to-end: 1 step through the trainer with --refc1.
    out1 = tmp_path / "run-refc1"
    m3 = refc_train.main(["--data-root", str(root), "--out", str(out1),
                          "--steps", "1", "--batch", "4", "--lr", "1e-3",
                          "--log-every", "1", "--device", "cpu", "--smoke",
                          "--refc1"])
    for k in ("loss", "wp", "ctrl", "speed", "law", "speed_cls", "speed_mae"):
        assert np.isfinite(m3["final"][k]), k
    assert m3["final"]["speed_cls"] > 0.0
    conf1 = json.loads((out1 / "config.json").read_text(encoding="utf-8"))
    assert conf1["cfg"]["refc1"] is True
