"""TIME-anchored multi-anchor tactical decoder (v2 lever 8) — CPU, synthetic.

The shared 4-brain ``TacticalPolicy.wp_heads`` were UNIMODAL (the 3.4 m tactical
weakness that REF-C/REF-B's anchored decoder cures). ``v2_anchor_tactical``
REPLACES them with an :class:`AnchoredTacticalDecoder` — a DiffusionDrive-style
FPS anchor vocabulary + per-anchor conf/offset (classifier mode, REF-C steps=0) —
shared by the flagship ``WorldModel`` and REF-A (``RefAModel``).

Pins:
  (a) gating OFF -> wp_heads present, NO anchor params, forward byte-identical to
      the wp_heads path, a base model's state_dict has no new keys;
  (b) gating ON -> anchored decoder present (wp_heads REPLACED), forward emits
      anchor_traj [B,N,S,2] + anchor_logits [B,N] + traj/sel_idx + a waypoints
      shim; the H19 prior gate is a zero-init Parameter (no-op at start);
  (c) the shared anchor loss (nearest-anchor CE + WTA L1) is finite, backward
      reaches the conf/offset/maneuver-prior heads, and n_modes is computable;
  (d) WorldModel (flagship) AND RefAModel (four_brain) both thread the flag to
      the shared TacticalPolicy (exactly wp_heads-out / anchor_decoder-in), and a
      v1 (flag-off) ckpt still loads a flag-off model with no missing keys;
  (e) flagship_loss with the flag on stays finite with cls/wta/n_modes surfaced.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.config import (flagship4b_config,  # noqa: E402
                            flagship4b_smoke_config, refa4b_smoke_config)
from tanitad.models.fourbrain import (AnchoredTacticalDecoder,  # noqa: E402
                                      TacticalPolicy, WorldModel,
                                      default_time_anchors)
from tanitad.refs.refa import RefAModel  # noqa: E402
from tanitad.train.flagship_losses import (anchor_tactical_loss,  # noqa: E402
                                           build_grounding, flagship_loss,
                                           horizon_plan)

FAST = dict(op_fwd_k=2, tac_fwd_k=3, str_fwd_k=4)   # small horizons -> max_h 4


def _tac_policy(anchor: bool, cfg=None):
    """A standalone TacticalPolicy at the smoke geometry (off vs anchored)."""
    cfg = cfg or flagship4b_smoke_config()
    return TacticalPolicy(cfg.tactical_policy, state_dim=40,
                          window=cfg.predictor.window,
                          d_cond=cfg.strategic_policy.d_ctx,
                          anchor_tactical=anchor)


def _fwd_inputs(tp, cfg, B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    states = torch.randn(B, cfg.predictor.window, tp.in_proj.in_features,
                         generator=g)
    ctx = torch.randn(B, cfg.strategic_policy.d_ctx, generator=g)
    return states, ctx


# --------------------------------------------------------------------------- #
# (a) gating OFF == byte-identical pre-lever behaviour                         #
# --------------------------------------------------------------------------- #
def test_gating_off_has_wp_heads_no_anchor_params():
    torch.manual_seed(0)
    tp = _tac_policy(anchor=False)
    assert tp.wp_heads is not None and tp.anchor_decoder is None
    assert not any("anchor_decoder" in k for k in tp.state_dict())
    # the wp_heads are the per-horizon Linear(d, 2) set (unimodal, pre-lever)
    cfg = flagship4b_smoke_config()
    assert set(tp.wp_heads) == {str(k) for k in cfg.tactical_policy.waypoint_horizons}


def test_gating_off_forward_matches_wp_heads_path():
    """The OFF forward reproduces the exact wp_heads path — byte-identical."""
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    tp = _tac_policy(anchor=False, cfg=cfg).eval()
    states, ctx = _fwd_inputs(tp, cfg)
    with torch.no_grad():
        out = tp(states, ctx)
        # recompute the summary token h and the direct wp_heads output
        from tanitad.models.fourbrain import _causal_mask
        W = cfg.predictor.window
        x = tp.in_proj(states) + tp.pos[:, :W]
        cond = ctx.unsqueeze(1).expand(-1, W, -1)
        for blk in tp.blocks:
            x = blk(x, cond, _causal_mask(W, states.device))
        h = tp.norm(x[:, -1])
        for k in cfg.tactical_policy.waypoint_horizons:
            assert torch.equal(out["waypoints"][k], tp.wp_heads[str(k)](h))
    # OFF emits no anchor tensors at all
    assert "anchor_traj" not in out and "anchor_logits" not in out


# --------------------------------------------------------------------------- #
# (b) gating ON: anchored decoder present + shapes + zero-init H19 gate         #
# --------------------------------------------------------------------------- #
def test_gating_on_builds_decoder_and_emits_shapes():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    tp = _tac_policy(anchor=True, cfg=cfg).eval()
    assert tp.wp_heads is None
    assert isinstance(tp.anchor_decoder, AnchoredTacticalDecoder)
    B = 4
    S = len(cfg.tactical_policy.waypoint_horizons)
    N = tp.anchor_decoder.anchors.shape[0]
    assert N == 128 and tuple(tp.anchor_decoder.anchors.shape) == (128, S, 2)
    states, ctx = _fwd_inputs(tp, cfg, B=B)
    with torch.no_grad():
        tac = tp(states, ctx)
    assert tuple(tac["anchor_traj"].shape) == (B, N, S, 2)
    assert tuple(tac["anchor_logits"].shape) == (B, N)
    assert tuple(tac["offset"].shape) == (B, N, S, 2)
    assert tuple(tac["traj"].shape) == (B, S, 2)
    assert tuple(tac["sel_idx"].shape) == (B,)
    # the selected traj IS the anchor_traj row at sel_idx (winner-takes-all)
    for b in range(B):
        assert torch.equal(tac["traj"][b], tac["anchor_traj"][b, tac["sel_idx"][b]])
    # back-compat waypoints shim: per-horizon points off the selected traj
    assert set(tac["waypoints"]) == set(cfg.tactical_policy.waypoint_horizons)
    for i, k in enumerate(cfg.tactical_policy.waypoint_horizons):
        assert tuple(tac["waypoints"][k].shape) == (B, 2)
        assert torch.equal(tac["waypoints"][k], tac["traj"][:, i])
    # n_modes is a computable int in [1, B]
    assert isinstance(tac["n_modes"], int) and 1 <= tac["n_modes"] <= B


def test_h19_prior_gate_is_zero_init_noop():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    tp = _tac_policy(anchor=True, cfg=cfg).eval()
    g = tp.anchor_decoder.h19_gate
    assert isinstance(g, nn.Parameter) and g.requires_grad
    assert tuple(g.shape) == () and float(g.detach()) == pytest.approx(0.0)
    states, ctx = _fwd_inputs(tp, cfg)
    with torch.no_grad():
        # at gate 0 the maneuver->anchor prior contributes exactly nothing
        assert float(tp(states, ctx)["prior_norm"]) == 0.0
        tp.anchor_decoder.h19_gate.data.fill_(0.5)
        assert float(tp(states, ctx)["prior_norm"]) > 0.0     # ramps once earned


def test_default_time_anchors_deterministic_and_time_shaped():
    """The FPS vocabulary is deterministic (two builds share anchors byte-for-
    byte) and TIME-parameterized over the waypoint horizons."""
    h = (5, 10, 15, 20)
    a1 = default_time_anchors(h, 128)
    a2 = default_time_anchors(h, 128)
    assert tuple(a1.shape) == (128, len(h), 2)
    assert torch.equal(a1, a2)


# --------------------------------------------------------------------------- #
# (c) shared anchor loss: finite, backward reaches the heads, n_modes          #
# --------------------------------------------------------------------------- #
def test_anchor_loss_finite_backward_and_grad_reach():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    tp = _tac_policy(anchor=True, cfg=cfg)
    B = 6
    S = len(cfg.tactical_policy.waypoint_horizons)
    states, ctx = _fwd_inputs(tp, cfg, B=B)
    tac = tp(states, ctx)
    wp_tgt = torch.randn(B, S, 2)
    loss_wp, loss_cls, loss_wta = anchor_tactical_loss(
        tac, tp.anchor_decoder.anchors, wp_tgt, pose_scale=10.0)
    assert torch.isfinite(loss_wp) and torch.isfinite(loss_cls)
    assert torch.isfinite(loss_wta)
    assert float(loss_wta) >= 0.0 and float(loss_cls) >= 0.0
    tp.zero_grad()
    loss_wp.backward()
    # CE trains the conf head; WTA L1 trains the offset head (both must get grad)
    assert float(tp.anchor_decoder.conf_head.weight.grad.abs().sum()) > 0
    assert float(tp.anchor_decoder.offset_head.weight.grad.abs().sum()) > 0
    # n_modes (distinct argmax anchors) is computable from the logits
    n_modes = int(tac["anchor_logits"].argmax(-1).unique().numel())
    assert 1 <= n_modes <= B and n_modes == tac["n_modes"]


# --------------------------------------------------------------------------- #
# (d) WorldModel + RefAModel thread the flag to the shared TacticalPolicy       #
# --------------------------------------------------------------------------- #
def _key_diff(m_off, m_on):
    off, on = set(m_off.state_dict()), set(m_on.state_dict())
    return on - off, off - on


def test_worldmodel_threads_anchor_flag_flagship():
    torch.manual_seed(0)
    smk = flagship4b_smoke_config()
    m_off = WorldModel(smk)
    m_on = WorldModel(dataclasses.replace(smk, v2_anchor_tactical=True))
    assert m_off.tactical_policy.anchor_decoder is None
    assert m_off.tactical_policy.wp_heads is not None
    assert isinstance(m_on.tactical_policy.anchor_decoder, AnchoredTacticalDecoder)
    assert m_on.tactical_policy.wp_heads is None
    added, removed = _key_diff(m_off, m_on)
    # clean REPLACEMENT: only anchor_decoder keys added, only wp_heads keys removed
    assert added and all("tactical_policy.anchor_decoder" in k for k in added)
    assert removed and all("tactical_policy.wp_heads" in k for k in removed)
    assert "tactical_policy.anchor_decoder.h19_gate" in added
    # a v1 (flag-off) checkpoint still loads a flag-off model with NO missing keys
    miss, unexp = m_off.load_state_dict(WorldModel(smk).state_dict(), strict=True)
    assert miss == [] and unexp == []


def test_refa_four_brain_threads_anchor_flag():
    torch.manual_seed(0)
    smk = refa4b_smoke_config()
    r_off = RefAModel.from_stack_config(smk, n_tokens=64, adapter_kind="grid")
    r_on = RefAModel.from_stack_config(
        dataclasses.replace(smk, v2_anchor_tactical=True), n_tokens=64,
        adapter_kind="grid")
    assert r_off.tactical_policy.anchor_decoder is None
    assert isinstance(r_on.tactical_policy.anchor_decoder, AnchoredTacticalDecoder)
    added, removed = _key_diff(r_off, r_on)
    assert added and all("tactical_policy.anchor_decoder" in k for k in added)
    assert removed and all("tactical_policy.wp_heads" in k for k in removed)
    # the anchored REF-A tactical brain runs forward end to end on adapter states
    B, W = 2, r_on.pred_cfg.window
    states = torch.randn(B, W, r_on.state_dim)
    ctx = torch.randn(B, smk.strategic_policy.d_ctx)
    tac = r_on.tactical_policy(states, ctx)
    assert tuple(tac["anchor_logits"].shape) == (B, 128)


# --------------------------------------------------------------------------- #
# (e) flagship_loss with the anchor flag on: finite + diagnostics surfaced      #
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


def _episode(T, eid, yaw_rate=0.0, accel=0.0, size=64):
    from tanitad.data._contract import assemble_episode
    g = torch.Generator().manual_seed(100 + eid)
    frames = [torch.rand(1, size, size, generator=g) for _ in range(T)]
    poses = _poses(T, yaw_rate=yaw_rate, accel=accel)
    return assemble_episode(frames, [p.numpy() for p in poses],
                            [yaw_rate] * T, 0.1, eid)


def _anchor_batch(n=6):
    from train_flagship4b import FlagshipWindowDataset
    cfg = dataclasses.replace(flagship4b_smoke_config(), v2_anchor_tactical=True)
    plan = horizon_plan(cfg, **FAST)
    eps = [_episode(260, 0, yaw_rate=0.06), _episode(260, 1, yaw_rate=-0.06),
           _episode(260, 2, yaw_rate=0.0), _episode(260, 3, accel=-1.2)]
    ds = FlagshipWindowDataset(eps, window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    batch = default_collate([ds[i] for i in range(n)])
    return cfg, plan, batch


def test_flagship_loss_with_anchor_flag_finite_and_logs():
    torch.manual_seed(0)
    cfg, plan, batch = _anchor_batch(n=6)
    m = WorldModel(cfg)
    assert isinstance(m.tactical_policy.anchor_decoder, AnchoredTacticalDecoder)
    grounding = build_grounding(m.state_dim, hidden=32)
    from tanitad.train.flagship_losses import LossWeights
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
    # the anchor objective replaced the unimodal wp L2 (cls/wta are the parts)
    assert "cls" in parts and "wta" in parts
    assert torch.isfinite(parts["cls"]) and torch.isfinite(parts["wta"])
    # the anchored diagnostics are surfaced in the train log
    for k in ("cls", "wta", "n_modes", "conf_norm", "prior_norm"):
        assert k in log, f"missing anchored log key {k}"
    assert isinstance(log["n_modes"], int) and log["n_modes"] >= 1
    assert log["prior_norm"] == 0.0            # zero-init H19 gate -> no-op prior
    # one backward is finite through the whole 4-brain + anchored decoder
    total.backward()
    for name, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name
    # the anchored decoder's heads are in the gradient path
    assert m.tactical_policy.anchor_decoder.conf_head.weight.grad is not None
