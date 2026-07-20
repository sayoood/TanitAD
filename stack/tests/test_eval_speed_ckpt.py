"""Speed-input checkpoint eval-harness tests (flagship-v2 correctness review,
HIGH finding). CPU, synthetic.

The committed eval/gate harnesses used to build the WorldModel from a stock
``action_dim=2`` config and strict-load, which CRASHES on a speed-input ckpt
(``predictor.act_emb.0.weight`` is ``[d, 3]``) — and ``strict=False`` would
silently leave 4 tensors random-init. The fix makes the CHECKPOINT self-
describing (``tanitad.eval.ckpt_compat``): infer ``action_dim`` from the ckpt
(saved ``config.json`` preferred, ``act_emb`` weight shape authoritative), build
the RIGHT shape, keep the load STRICT, and append v0 = ``pose_last[:,3]/10`` as
the constant 3rd action channel at rollout (t=0 speed only — leakage-safe).

These tests pin, for BOTH fixed scripts' detection+load+rollout path:
  (a) a 3-ch speed ckpt loads strict-clean; the eval rollout of
      ``eval_grounded_rollout_4b.collect`` AND the gate tensors of
      ``evaluate_checkpoint.build_eval_tensors`` consume 3-ch actions whose v0
      channel is the constant (past-only) speed;
  (b) a plain 2-ch ckpt still loads and rolls 2-ch — byte-identical path;
  (c) detection: config.json is preferred, the weight shape is the authoritative
      fallback (and WINS on disagreement, so pod-side v1 ckpts work);
  (d) ``append_speed_channel`` obeys the SPEED_SCALE=10 leakage-safe contract.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_grounded_rollout_4b import collect as grounded_collect  # noqa: E402
from evaluate_checkpoint import build_eval_tensors  # noqa: E402

from tanitad.config import flagship4b_smoke_config  # noqa: E402
from tanitad.data._contract import assemble_episode  # noqa: E402
from tanitad.eval.ckpt_compat import (SPEED_SCALE,  # noqa: E402
                                      adapt_config_action_dim,
                                      append_speed_channel, ckpt_action_dim,
                                      build_world_from_ckpt)
from tanitad.models.fourbrain import WorldModel  # noqa: E402
from tanitad.models.metric_dynamics import HierarchicalGrounding  # noqa: E402

WP_K = 20                       # K_MAX (2 s @ 10 Hz) — the rollout reach


# --------------------------------------------------------------------------- #
# Synthetic episodes: ACCELERATING so future speeds differ from v0 (a constant #
# v0 channel then proves no future speed leaked).                              #
# --------------------------------------------------------------------------- #
def _poses(T, dt=0.1, v0=8.0, yaw_rate=0.0, accel=0.6):
    rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, v0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.5, v + accel * dt)
    return torch.tensor(rows, dtype=torch.float32)


def _episode(T, eid, yaw_rate=0.0, accel=0.6, size=64):
    g = torch.Generator().manual_seed(100 + eid)
    frames = [torch.rand(1, size, size, generator=g) for _ in range(T)]
    poses = _poses(T, yaw_rate=yaw_rate, accel=accel)
    return assemble_episode(frames, [p.numpy() for p in poses],
                            [yaw_rate] * T, 0.1, eid)


def _episodes(n=2, T=48):
    return [_episode(T, i, yaw_rate=0.05 * (i - 0.5), accel=0.6 + 0.2 * i)
            for i in range(n)]


def _speed_cfg():
    """flagship smoke + speed_input (action_dim 3) — the trainer's wiring."""
    cfg = flagship4b_smoke_config()
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    return cfg


def _save_ckpt(tmp_path, cfg, name="ckpt.pt", with_grounding=True,
               config_json=None):
    """Build a tiny WorldModel at ``cfg`` and save a train_flagship4b-shaped
    ckpt (model + grounding + step). Optionally drop a config.json alongside."""
    torch.manual_seed(0)
    world = WorldModel(cfg)
    grounding = HierarchicalGrounding(world.state_dim)
    d = tmp_path / name
    payload = {"model": world.state_dict(), "step": 123}
    if with_grounding:
        payload["grounding"] = grounding.state_dict()
    torch.save(payload, d)
    if config_json is not None:
        (tmp_path / "config.json").write_text(json.dumps(config_json),
                                              encoding="utf-8")
    return str(d)


def _spy_predictor(world):
    """Record the ``actions`` arg of every operative-predictor forward call."""
    captured = []
    orig = world.predictor.forward

    def spy(states, actions, **kw):
        captured.append(actions.detach().clone())
        return orig(states, actions, **kw)

    world.predictor.forward = spy
    return captured


# --------------------------------------------------------------------------- #
# (c) DETECTION — config.json preferred, weight shape authoritative fallback   #
# --------------------------------------------------------------------------- #
def test_detect_from_weights_when_no_config(tmp_path):
    ck_path = _save_ckpt(tmp_path, _speed_cfg())        # 3-ch, no config.json
    ck = torch.load(ck_path, weights_only=True)
    dim, src = ckpt_action_dim(ck, ckpt_path=ck_path)
    assert dim == 3 and src == "weights"


def test_detect_prefers_config_json(tmp_path):
    cfg = _speed_cfg()
    ck_path = _save_ckpt(tmp_path, cfg, config_json={"cfg": cfg.to_json()})
    ck = torch.load(ck_path, weights_only=True)
    dim, src = ckpt_action_dim(ck, ckpt_path=ck_path)
    assert dim == 3 and src == "config.json"


def test_weight_shape_wins_on_config_disagreement(tmp_path):
    # config.json LIES (says 2) but the tensors are 3-ch: the strict load must
    # match the tensors, so the weight shape is authoritative.
    ck_path = _save_ckpt(tmp_path, _speed_cfg(),
                         config_json={"predictor": {"action_dim": 2}})
    ck = torch.load(ck_path, weights_only=True)
    dim, src = ckpt_action_dim(ck, ckpt_path=ck_path)
    assert dim == 3 and src == "weights"


def test_detect_two_channel_default(tmp_path):
    ck_path = _save_ckpt(tmp_path, flagship4b_smoke_config())     # 2-ch
    ck = torch.load(ck_path, weights_only=True)
    dim, _src = ckpt_action_dim(ck, ckpt_path=ck_path)
    assert dim == 2


def test_adapt_config_two_channel_is_untouched():
    cfg = flagship4b_smoke_config()
    pred_before, tac_before = cfg.predictor, cfg.tactical_pred
    out = adapt_config_action_dim(cfg, 2)
    assert out.predictor is pred_before and out.tactical_pred is tac_before
    assert out.speed_input is False


# --------------------------------------------------------------------------- #
# (a) 3-ch ckpt loads STRICT-clean via both scripts' shared build path         #
# --------------------------------------------------------------------------- #
def test_speed_ckpt_loads_strict_clean(tmp_path):
    ck_path = _save_ckpt(tmp_path, _speed_cfg())
    ck = torch.load(ck_path, weights_only=True)
    # base config is action_dim=2; build_world_from_ckpt must adapt to 3 and
    # STRICT-load with no missing/unexpected keys (no exception).
    world, speed_input, src = build_world_from_ckpt(flagship4b_smoke_config(),
                                                    ck, ckpt_path=ck_path)
    assert speed_input is True and src == "weights"
    assert world.predictor.act_emb[0].in_features == 3
    assert world.tactical_pred.act_emb[0].in_features == 3


# --------------------------------------------------------------------------- #
# (a) eval_grounded_rollout_4b.collect consumes 3-ch actions w/ constant v0    #
# --------------------------------------------------------------------------- #
def test_grounded_collect_rolls_three_channel_constant_v0(tmp_path):
    ck_path = _save_ckpt(tmp_path, _speed_cfg())
    ck = torch.load(ck_path, weights_only=True)
    world, speed_input, _ = build_world_from_ckpt(flagship4b_smoke_config(),
                                                  ck, ckpt_path=ck_path)
    world = world.eval()
    grounding = HierarchicalGrounding(world.state_dim).eval()
    grounding.load_state_dict(ck["grounding"])
    step_readout = grounding.step["op"]
    eps = _episodes()
    cap = _spy_predictor(world)
    data = grounded_collect(world, step_readout, eps, ["comma2k19"] * len(eps),
                            "cpu", world.predictor.cfg.window, WP_K,
                            stride=16, batch=8, speed_input=speed_input)
    assert data["pred"].shape[0] > 0 and data["pred"].shape[1:] == (4, 2)
    assert cap, "predictor never called"
    for a in cap:                       # every rolled predictor call is 3-ch
        assert a.shape[-1] == 3
        ch = a[..., 2]                  # v0 channel: constant over the window
        assert torch.allclose(ch, ch[:, :1].expand_as(ch), atol=1e-6)
        assert (ch > 0).all()           # a real (scaled) speed, not zeros


# --------------------------------------------------------------------------- #
# (a) evaluate_checkpoint.build_eval_tensors appends 3-ch to the MODEL,        #
#     but keeps the returned probe `actions` as the raw recorded 2-ch          #
# --------------------------------------------------------------------------- #
def test_gate_tensors_feed_three_channel_keep_two_channel_probe(tmp_path):
    ck_path = _save_ckpt(tmp_path, _speed_cfg(), with_grounding=False)
    ck = torch.load(ck_path, weights_only=True)
    world, speed_input, _ = build_world_from_ckpt(flagship4b_smoke_config(),
                                                  ck, ckpt_path=ck_path)
    world = world.eval()
    eps = _episodes()
    cap = _spy_predictor(world)
    k_max = max(world.predictor.cfg.horizons)
    t = build_eval_tensors(world, eps, "cpu", world.predictor.cfg.window,
                           k_max, stride=16, batch=8, speed_input=speed_input)
    assert cap and all(a.shape[-1] == 3 for a in cap)     # model sees 3-ch
    for a in cap:
        ch = a[..., 2]
        assert torch.allclose(ch, ch[:, :1].expand_as(ch), atol=1e-6)
    # the probe/spectral `actions` stay the raw recorded 2-ch (comparable
    # across checkpoints — NOT the model-fed speed-augmented tensor).
    assert t["actions"].shape[-1] == 2
    assert t["states"].shape[0] > 0


# --------------------------------------------------------------------------- #
# (b) 2-ch ckpt: byte-identical pre-change path (no v0 channel anywhere)       #
# --------------------------------------------------------------------------- #
def test_two_channel_ckpt_unchanged_grounded(tmp_path):
    ck_path = _save_ckpt(tmp_path, flagship4b_smoke_config())
    ck = torch.load(ck_path, weights_only=True)
    world, speed_input, _ = build_world_from_ckpt(flagship4b_smoke_config(),
                                                  ck, ckpt_path=ck_path)
    assert speed_input is False
    assert world.predictor.act_emb[0].in_features == 2
    world = world.eval()
    grounding = HierarchicalGrounding(world.state_dim).eval()
    grounding.load_state_dict(ck["grounding"])
    eps = _episodes()
    cap = _spy_predictor(world)
    grounded_collect(world, grounding.step["op"], eps,
                     ["comma2k19"] * len(eps), "cpu",
                     world.predictor.cfg.window, WP_K, stride=16, batch=8,
                     speed_input=speed_input)
    assert cap and all(a.shape[-1] == 2 for a in cap)


def test_two_channel_ckpt_unchanged_gate(tmp_path):
    ck_path = _save_ckpt(tmp_path, flagship4b_smoke_config(),
                         with_grounding=False)
    ck = torch.load(ck_path, weights_only=True)
    world, speed_input, _ = build_world_from_ckpt(flagship4b_smoke_config(),
                                                  ck, ckpt_path=ck_path)
    assert speed_input is False
    world = world.eval()
    eps = _episodes()
    cap = _spy_predictor(world)
    k_max = max(world.predictor.cfg.horizons)
    t = build_eval_tensors(world, eps, "cpu", world.predictor.cfg.window,
                           k_max, stride=16, batch=8, speed_input=speed_input)
    assert cap and all(a.shape[-1] == 2 for a in cap)
    assert t["actions"].shape[-1] == 2


# --------------------------------------------------------------------------- #
# (d) append_speed_channel — SPEED_SCALE + constant-expansion contract         #
# --------------------------------------------------------------------------- #
def test_append_speed_channel_contract():
    assert SPEED_SCALE == 10.0
    actions = torch.zeros(3, 5, 2)               # [B, K, 2]
    pose_speed = torch.tensor([[8.0], [12.0], [20.0]])
    v0 = pose_speed / SPEED_SCALE
    out = append_speed_channel(actions, v0)
    assert out.shape == (3, 5, 3)
    assert torch.allclose(out[..., :2], actions)             # first 2 untouched
    # 3rd channel == v0 broadcast constant across all K steps (past-only)
    assert torch.allclose(out[..., 2], v0.expand(3, 5))
    for b in range(3):
        assert torch.allclose(out[b, :, 2],
                              torch.full((5,), float(v0[b, 0])))
