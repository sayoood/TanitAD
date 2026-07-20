"""Part A loss-rebalance (research 2026-07-18-loss-rebalance-and-lewm-decode.md,
PART A) — CPU, synthetic.

A straight-through gradient SCALE ``alpha`` (``v2_invdyn_gradscale``) applied to
the ENCODER-latent inputs ``(z_t, fut_states)`` as they feed the metric-inverse-
dynamics REAL-PAIR term (a) of ``grounding_losses`` — and term (a) ONLY. It softly
decouples the static ego-motion probe (weight 2.0x3=6.0 of encoder-shaping mass)
from the encoder trunk WITHOUT touching:
  - the invdyn HEAD params (they read the identical forward value -> full-rate
    readout probes, unchanged gradient);
  - term (b) forward-consistency ROLLOUT (reads states/trans, never the scaled
    views) — the PROTECTED 0.033 m fwd-ADE producer, fully attached;
  - JEPA + SIGReg (outside grounding_losses).

Pins (deliverable test matrix):
  (a) alpha == 1.0  -> byte-identical loss + grads to the default (no-op);
  (b) alpha < 1.0   -> the invdyn-REAL-PAIR encoder gradient is scaled by ~alpha,
      while the invdyn HEAD grads AND the term-(b)/JEPA/SIGReg encoder grads are
      UNCHANGED;
  (c) alpha == 0.0  -> zero encoder grad from term (a), but the heads still train
      AND the metric forward-consistency term still reaches the encoder;
  (d) full --v2 forward+backward finite.

Determinism: models run in EVAL for the cross-alpha equality pins (kills any
dropout), so the ONLY difference across alpha is the backward gradient path
(grad_scale is forward-exact). The full --v2 finiteness pin runs in TRAIN mode to
exercise every lever.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_flagship4b import FlagshipWindowDataset  # noqa: E402

from tanitad.config import flagship4b_smoke_config  # noqa: E402
from tanitad.data._contract import assemble_episode  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402
from tanitad.models.metric_dynamics import (grad_scale,  # noqa: E402
                                            grounding_losses)
from tanitad.train.flagship_losses import (LossWeights,  # noqa: E402
                                           build_grounding, flagship_loss,
                                           horizon_plan)

FAST = dict(op_fwd_k=2, tac_fwd_k=3, str_fwd_k=4)
LEVELS = ("op", "tac", "str")
ALPHAS = (0.5, 0.25, 0.0)                    # the ablation sweep minus the control


# --------------------------------------------------------------------------- #
# Fixtures (mirror test_vision_levers / test_speed_input)                      #
# --------------------------------------------------------------------------- #
def _poses(T: int, dt=0.1, v0=8.0, yaw_rate=0.0, accel=0.0):
    rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, v0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.0, v + accel * dt)
    return torch.tensor(rows, dtype=torch.float32)


def _episode(T, eid, yaw_rate=0.0, accel=0.0, size=64):
    g = torch.Generator().manual_seed(100 + eid)
    frames = [torch.rand(1, size, size, generator=g) for _ in range(T)]
    poses = _poses(T, yaw_rate=yaw_rate, accel=accel)
    return assemble_episode(frames, [p.numpy() for p in poses],
                            [yaw_rate] * T, 0.1, eid)


def _batch(cfg, n=4, T=60):
    plan = horizon_plan(cfg, **FAST)
    eps = [_episode(T, 0, yaw_rate=0.06, accel=-1.0),
           _episode(T, 1, yaw_rate=-0.06, accel=0.8),
           _episode(T, 2, accel=-1.2), _episode(T, 3, accel=1.0)]
    ds = FlagshipWindowDataset(eps, window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    batch = default_collate([ds[i] for i in range(n)])
    return plan, batch


def _encode(m, plan, batch):
    """Encode the window once; return DETACHED states/fut_states so each test can
    make fresh leaves and read the encoder-bound gradient off `.grad`."""
    with torch.no_grad():
        states = m.encode_window(batch["frames"])
        fut = m.encode_window(batch["future_frames"][:, plan.needed_fut])
    return states, fut


def _leaf(t):
    return t.detach().clone().requires_grad_(True)


def _ground(grounding, m, batch, plan, states, fut_states, alpha):
    """grounding_losses with the flagship weights (invdyn 2.0, fwd 1.0) and the
    Part A gradient scale. Returns (total, parts, log)."""
    return grounding_losses(
        grounding, m.predictor, states, fut_states,
        batch["actions"], batch["future_actions"],
        batch["pose_last"].float(), batch["future_poses"].float(),
        plan.idx_of, plan.level_cfg, pose_scale=10.0,
        invdyn_weight=2.0, fwd_weight=1.0, invdyn_gradscale=alpha)


def _grads(named, clone=True):
    return {n: (p.grad.clone() if clone else p.grad)
            for n, p in named if p.grad is not None}


# --------------------------------------------------------------------------- #
# grad_scale unit contract                                                     #
# --------------------------------------------------------------------------- #
def test_grad_scale_forward_is_bit_exact_identity():
    """Forward is bit-exactly x for EVERY alpha (straight-through ordering)."""
    x = torch.randn(6, 9)
    for a in (1.0, *ALPHAS):
        assert torch.equal(grad_scale(x, a), x)


def test_grad_scale_backward_is_alpha_multiplier():
    for a in (1.0, *ALPHAS):
        x = torch.randn(4, 3, requires_grad=True)
        grad_scale(x, a).pow(2).sum().backward()          # d/dx x^2 = 2x, scaled
        assert torch.allclose(x.grad, a * 2 * x, atol=1e-6)


def test_grad_scale_alpha1_returns_same_object():
    """alpha == 1.0 short-circuits to x itself: strict, byte-identical no-op."""
    x = torch.randn(5, requires_grad=True)
    assert grad_scale(x, 1.0) is x


# --------------------------------------------------------------------------- #
# (a) alpha == 1.0 is a byte-identical no-op vs the default (unset) call        #
# --------------------------------------------------------------------------- #
def test_alpha1_matches_default_byte_identical():
    """Threading the new param and passing 1.0 reproduces the DEFAULT (no-arg,
    i.e. pre-change) call bit-for-bit: same total, same encoder-bound grads
    (states/fut_states), same invdyn+step HEAD grads. The grad_scale guard makes
    the alpha=1.0 graph identical to the code before the param existed."""
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    m = WorldModel(cfg).eval()
    plan, batch = _batch(cfg)
    states0, fut0 = _encode(m, plan, batch)
    grounding = build_grounding(m.state_dim, hidden=32)

    def run(pass_alpha):
        S, Fst = _leaf(states0), _leaf(fut0)
        m.zero_grad(set_to_none=True)
        grounding.zero_grad(set_to_none=True)
        if pass_alpha is None:                            # DEFAULT (no arg)
            total, _, _ = grounding_losses(
                grounding, m.predictor, S, Fst, batch["actions"],
                batch["future_actions"], batch["pose_last"].float(),
                batch["future_poses"].float(), plan.idx_of, plan.level_cfg,
                pose_scale=10.0, invdyn_weight=2.0, fwd_weight=1.0)
        else:
            total, _, _ = _ground(grounding, m, batch, plan, S, Fst, pass_alpha)
        total.backward()
        return (total.detach().clone(), S.grad.clone(), Fst.grad.clone(),
                _grads(grounding.named_parameters()))

    t_def, s_def, f_def, h_def = run(None)
    t_one, s_one, f_one, h_one = run(1.0)
    assert torch.equal(t_def, t_one)                      # loss byte-identical
    assert torch.equal(s_def, s_one)                      # encoder z_t path identical
    assert torch.equal(f_def, f_one)                      # encoder fut path identical
    assert h_def.keys() == h_one.keys() and h_def
    for k in h_def:                                       # every HEAD grad identical
        assert torch.equal(h_def[k], h_one[k]), k


# --------------------------------------------------------------------------- #
# Forward-identity: the loss VALUE (and every logged g_* metric) is invariant   #
# to alpha — only gradients change.                                             #
# --------------------------------------------------------------------------- #
def test_forward_loss_value_invariant_to_alpha():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    m = WorldModel(cfg).eval()
    plan, batch = _batch(cfg)
    states0, fut0 = _encode(m, plan, batch)
    grounding = build_grounding(m.state_dim, hidden=32)
    t1, _, log1 = _ground(grounding, m, batch, plan, states0, fut0, 1.0)
    for a in ALPHAS:
        ta, _, loga = _ground(grounding, m, batch, plan, states0, fut0, a)
        assert torch.equal(t1, ta), a                     # forward loss bit-identical
        assert log1 == loga, a                            # every logged metric identical


# --------------------------------------------------------------------------- #
# (b1) alpha < 1 scales ONLY term (a)'s encoder path; the invdyn HEADS unchanged #
# --------------------------------------------------------------------------- #
def _measure_mid(m, grounding, plan, batch, states0, fut0, alpha):
    """Backward the pure term-(a) mid loss; isolate the encoder-bound grads
    (S/Fst) and the invdyn / step HEAD grads."""
    S, Fst = _leaf(states0), _leaf(fut0)
    m.zero_grad(set_to_none=True)
    grounding.zero_grad(set_to_none=True)
    _, parts, _ = _ground(grounding, m, batch, plan, S, Fst, alpha)
    sum(parts[f"{lvl}_mid"] for lvl in LEVELS).backward()
    return (S.grad.clone(), Fst.grad.clone(),
            _grads(grounding.invdyn.named_parameters()),
            _grads(grounding.step.named_parameters()))


def test_alpha_scales_invdyn_encoder_path_and_heads_unchanged():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    m = WorldModel(cfg).eval()
    plan, batch = _batch(cfg)
    states0, fut0 = _encode(m, plan, batch)
    grounding = build_grounding(m.state_dim, hidden=32)

    s1, f1, inv1, step1 = _measure_mid(m, grounding, plan, batch,
                                       states0, fut0, 1.0)
    # sanity: term (a) is live (encoder + heads get gradient) and never touches
    # the step-readout heads (those belong to term (b)).
    assert float(f1.abs().sum()) > 0 and inv1
    assert not step1, "term (a) must not touch the step-readout heads"
    # z_t enters ONLY via the last window frame -> earlier frames get no term-a grad
    assert float(s1[:, :-1].abs().sum()) == 0.0

    for a in ALPHAS:
        sa, fa, inva, _ = _measure_mid(m, grounding, plan, batch,
                                       states0, fut0, a)
        # encoder-bound grad (fut = PURE term a) scales by exactly alpha
        assert abs(float(fa.norm() / f1.norm()) - a) < 1e-3, a
        assert torch.allclose(fa, a * f1, atol=1e-6, rtol=1e-4), a
        # the z_t encoder path (last frame) scales by alpha too
        assert torch.allclose(sa[:, -1], a * s1[:, -1], atol=1e-6, rtol=1e-4), a
        # the invdyn HEAD grads are IDENTICAL (forward value unchanged) — the
        # probe still trains at full rate; only the path INTO the encoder shrank.
        for k in inv1:
            assert torch.equal(inva[k], inv1[k]), (a, k)


# --------------------------------------------------------------------------- #
# (b2) term (b) forward-consistency (the 0.033 m producer) is UNTOUCHED by alpha #
# --------------------------------------------------------------------------- #
def _measure_fwd(m, grounding, plan, batch, states0, fut0, alpha):
    S, Fst = _leaf(states0), _leaf(fut0)
    m.zero_grad(set_to_none=True)
    grounding.zero_grad(set_to_none=True)
    _, parts, _ = _ground(grounding, m, batch, plan, S, Fst, alpha)
    sum(parts[f"{lvl}_fwd"] for lvl in LEVELS).backward()
    return (S.grad.clone(), Fst.grad,
            _grads(m.predictor.named_parameters()),
            _grads(grounding.step.named_parameters()),
            _grads(grounding.invdyn.named_parameters()))


def test_fwd_rollout_term_b_untouched_by_alpha():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    m = WorldModel(cfg).eval()
    plan, batch = _batch(cfg)
    states0, fut0 = _encode(m, plan, batch)
    grounding = build_grounding(m.state_dim, hidden=32)

    s1, f1, pred1, step1, inv1 = _measure_fwd(m, grounding, plan, batch,
                                              states0, fut0, 1.0)
    # term (b) reaches the encoder (via states) + predictor + the STEP heads,
    # and NEVER the fut_states pair nor the invdyn heads.
    assert float(s1.abs().sum()) > 0 and pred1 and step1
    assert f1 is None, "term (b) must not push gradient into fut_states"
    assert not inv1, "term (b) must not touch the invdyn heads"

    for a in ALPHAS:
        sa, fa, preda, stepa, inva = _measure_fwd(m, grounding, plan, batch,
                                                  states0, fut0, a)
        assert torch.equal(sa, s1), a                     # encoder grad UNCHANGED
        assert fa is None and not inva
        for k in pred1:                                   # predictor grad UNCHANGED
            assert torch.equal(preda[k], pred1[k]), (a, k)
        for k in step1:                                   # step-head grad UNCHANGED
            assert torch.equal(stepa[k], step1[k]), (a, k)


# --------------------------------------------------------------------------- #
# (b3) JEPA + SIGReg encoder gradients are UNTOUCHED by alpha (flagship_loss)    #
# --------------------------------------------------------------------------- #
def test_jepa_and_sigreg_encoder_grads_untouched_by_alpha():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    m = WorldModel(cfg).eval()
    plan, batch = _batch(cfg)
    grounding = build_grounding(m.state_dim, hidden=32)

    def enc_grads(part_key, alpha):
        cfg.v2_invdyn_gradscale = alpha
        m.zero_grad(set_to_none=True)
        grounding.zero_grad(set_to_none=True)
        states = m.encode_window(batch["frames"])         # ENCODER in the tape
        fut = m.encode_window(batch["future_frames"][:, plan.needed_fut])
        torch.manual_seed(777)                            # pin SIGReg's slices
        _, _, parts = flagship_loss(
            m, grounding, batch, states, fut, plan, cfg,
            weights=LossWeights(), sigreg_variant="full_relaxed",
            sigreg_free_dims=cfg.loss.sigreg.free_dims, pose_scale=10.0,
            fwd_step_weight=0.5, device="cpu")
        parts[part_key].backward()
        return _grads(m.encoder.named_parameters())

    for part_key in ("pred", "sigreg"):                   # JEPA-operative, SIGReg
        g1 = enc_grads(part_key, 1.0)
        assert g1, f"{part_key} produced no encoder grad (sanity)"
        for a in ALPHAS:
            ga = enc_grads(part_key, a)
            for k in g1:
                assert torch.equal(ga[k], g1[k]), (part_key, a, k)


# --------------------------------------------------------------------------- #
# (c) alpha == 0.0 detaches the encoder path but heads + term (b) stay alive     #
# --------------------------------------------------------------------------- #
def test_alpha0_detaches_encoder_but_heads_and_fwd_alive():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    m = WorldModel(cfg).eval()
    plan, batch = _batch(cfg)
    states0, fut0 = _encode(m, plan, batch)
    grounding = build_grounding(m.state_dim, hidden=32)

    # (i) term (a) at alpha=0: the encoder-bound gradient is exactly ZERO, but
    #     the invdyn HEADS still receive gradient (they keep training as probes).
    s0, f0, inv0, _ = _measure_mid(m, grounding, plan, batch, states0, fut0, 0.0)
    assert float(f0.abs().sum()) == 0.0                   # fut path fully detached
    assert float(s0[:, -1].abs().sum()) == 0.0            # z_t path fully detached
    assert inv0 and all(float(g.abs().sum()) > 0 for g in inv0.values())

    # (ii) the metric FORWARD-consistency term (b) STILL reaches the encoder at
    #      alpha=0 — the 0.033 m producer is independent of the gradient scale.
    S, Fst = _leaf(states0), _leaf(fut0)
    m.zero_grad(set_to_none=True)
    grounding.zero_grad(set_to_none=True)
    _, parts, _ = _ground(grounding, m, batch, plan, S, Fst, 0.0)
    sum(parts[f"{lvl}_fwd"] for lvl in LEVELS).backward()
    assert float(S.grad.abs().sum()) > 0


# --------------------------------------------------------------------------- #
# (d) full --v2 (incl. gradscale 0.25) forward + backward finite                #
# --------------------------------------------------------------------------- #
def _v2_cfg(rollout_k: int = 2):
    """flagship smoke config with the FULL trainer --v2 pack (all v2 levers +
    speed-input action_dim 3) INCLUDING Part A gradscale 0.25 — exactly what
    train_flagship4b.py --v2 constructs, shrunk for CPU."""
    cfg = flagship4b_smoke_config()
    cfg.v2_ego_to_planners = True
    cfg.v2_ego_dropout = 0.25
    cfg.v2_fa_dropout = 0.3
    cfg.v2_goal_decode = True
    cfg.v2_nav_dropout = 0.5
    cfg.v2_traj_jerk = 0.02
    cfg.v2_gated_intent = True
    cfg.v2_anchor_tactical = True
    cfg.v2_route_from_vision = True
    cfg.v2_encoder_ego_decorr = True
    cfg.v2_invdyn_gradscale = 0.25                        # Part A
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    cfg.train.rollout_k = rollout_k
    return cfg


def test_full_v2_forward_backward_finite():
    torch.manual_seed(0)
    cfg = _v2_cfg()
    m = WorldModel(cfg)                                   # TRAIN mode: every lever live
    plan, batch = _batch(cfg)
    grounding = build_grounding(m.state_dim, hidden=32)
    states = m.encode_window(batch["frames"])
    fut = m.encode_window(batch["future_frames"][:, plan.needed_fut])
    total, log, _ = flagship_loss(
        m, grounding, batch, states, fut, plan, cfg,
        weights=LossWeights(), sigreg_variant="full_relaxed",
        sigreg_free_dims=cfg.loss.sigreg.free_dims, pose_scale=10.0,
        fwd_step_weight=0.5, device="cpu")
    assert torch.isfinite(total)
    total.backward()
    bad = [n for n, p in m.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite model grads under full --v2: {bad[:5]}"
    gbad = [n for n, p in grounding.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not gbad, f"non-finite grounding grads under full --v2: {gbad[:5]}"
    # the invdyn heads STILL trained (nonzero grad) despite the 0.25 encoder scale
    inv_g = sum(float(p.grad.abs().sum())
                for p in grounding.invdyn.parameters() if p.grad is not None)
    assert inv_g > 0
    for lvl in LEVELS:                                    # grounding still logged
        assert math.isfinite(log[f"g_{lvl}_fwd_ade_m"])


# --------------------------------------------------------------------------- #
# Loss-side only: the lever adds NO parameters / state_dict keys                #
# --------------------------------------------------------------------------- #
def test_loss_side_only_no_param_change():
    off = flagship4b_smoke_config()
    on = dataclasses.replace(off, v2_invdyn_gradscale=0.25)
    m_off, m_on = WorldModel(off), WorldModel(on)
    assert (sum(p.numel() for p in m_off.parameters())
            == sum(p.numel() for p in m_on.parameters()))
    assert set(m_off.state_dict()) == set(m_on.state_dict())
    g_off = build_grounding(m_off.state_dim, hidden=32)
    g_on = build_grounding(m_on.state_dim, hidden=32)
    assert (sum(p.numel() for p in g_off.parameters())
            == sum(p.numel() for p in g_on.parameters()))
