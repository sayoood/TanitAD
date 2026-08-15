"""W4 — the unicycle-anchor head retrofit on the FROZEN v5f 30k trunk (v5.8f).

WHY (V58F_FUSION.md, wedge W4, pre-registered): v5f's diffusion head
(``FlagshipV4Head`` wrapping ``AnchoredDiffusionDecoder``, 256 anchors, dense
horizons 1..20 @ 10 Hz) emits a fan whose candidates are positionally strong
(oracle ADE 0.1975 m @30k) but kinematically garbage — MEASURED accel MAE
**8.11 m/s²** on the selected trajectory (four-families eval) and a step-level
feasibility census with ~97 % of fan steps violating |a| <= 4 m/s² OR
|yaw_rate| <= 0.33·v + 0.05 (free-waypoint jitter from the 2-step truncated
denoise). W4 retrains ONLY a small new module on the frozen trunk so the fan is
emitted as UNICYCLE CONTROL SEQUENCES — per candidate, per step:
(accel a_k, curvature kappa_k) — integrated by a differentiable unicycle
rollout into waypoints. Feasible BY CONSTRUCTION via bounded activations:
``a = 4.0*tanh(raw_a)``, ``kappa = 0.2*tanh(raw_kappa)`` (yaw_rate = kappa*v
<= 0.2*v, inside the 0.33*v + 0.05 census band for every v >= 0).

⛔ PRE-REGISTERED GATE (V58F_FUSION.md §3, W4): **accel MAE < 1.5 m/s² AND
oracle ADE not worse than +10 % vs 0.1975** (i.e. <= 0.21725). Both outcomes
committed in advance: PASS -> v5.8f needs no trunk retrain for a feasible fan;
FAIL -> unicycle anchors become a v6 trunk-training lever and v5.8f ships with
projection-to-manifold. The end-of-run mini-eval writes ``<out>/w4_gate.json``
with the booleans.

ONE LEVER (attribution): the SELECTION path — refined_logits, factorised
grafts, ``select()``, ``sel_idx`` — stays FROZEN and untouched. W4 changes the
fan's PARAMETERISATION, not the selector. ``selected_ade`` in the gate JSON is
the frozen selector's own pick applied to the NEW fan.

CONDITIONING (``--cond``):
  * ``feature`` (default): the per-candidate token that currently feeds the
    offset head — the query ``q`` [B, N, d] of
    ``AnchoredDiffusionDecoder._decode`` (tanitad/refs/refc.py:1193-1198),
    the same tensor read by ``conf_head`` and ``offset_head``. It is a local
    variable, so it is captured with a forward PRE-HOOK on
    ``head.decoder.offset_head`` (:class:`OffsetFeatureTap`); the LAST call per
    forward is the final denoise pass — the one whose offset produced the
    emitted ``anchor_traj``.
  * ``anchor``: the projection-to-manifold FALLBACK of V58F_FUSION.md — the
    emission conditions on ``anchor_traj.detach()`` flattened per candidate
    (K*2) + v0, i.e. an anchor-to-controls PROJECTOR.

DISCRETISATION (dt = 0.1 s, ego frame +x fwd / +y left, initial heading 0,
initial speed v0). MATCHED to the v1.6/v1.7 unicycle work — the convention of
``train_unicycle_readout.py`` ``decode()`` (2026-08-06-v1-defect-triage/tools/
train_unicycle_readout.py:247-260) composed through
``tanitad.models.metric_dynamics.accumulate_se2`` (metric_dynamics.py:114-139),
which this file CALLS rather than re-implements. Per step k (0-based):

    dx_k   = v_k * dt                      # PRE-update speed (readout line 257)
    dyaw_k = kappa_k * v_k * dt            # yaw_rate = kappa*v, pre-update v
    p_k    = p_{k-1} + Rot(psi_{k-1}) @ [dx_k, 0]   # translate at the PREVIOUS
    psi_k  = psi_{k-1} + dyaw_k            # heading, THEN turn (accumulate_se2
                                           # lines 133-137 order)
    v_{k+1} = max(0, v_k + a_k * dt)       # speed updated AFTER the row is
                                           # emitted (readout line 258)

(The brief's "update v and yaw before translating" variant was NOT chosen:
matching the banked v1.6/v1.7 convention was ruled to matter more than the
variant, and this is that convention exactly.)

LOSSES on the emitted fan (GT = ``refb_labels.waypoint_targets``, dense 1..20):
  (a) winner-takes-all L2 — squared Euclidean distance of the GT-nearest
      candidate's waypoints to GT (standard fan training, prevents collapse);
  (b) L1 speed-profile term on the winner (v1.7's measured win; eps INSIDE the
      sqrt — the stopped-path NaN-grad trap, readout lines 298-304);
  (c) small L2 on (a, kappa) magnitudes, weight 0.01 (smoothness prior).

⛔ FROZEN MEANS PROVED FROZEN (train_unicycle_readout.py contract): trunk +
head + grounding are ``requires_grad_(False)`` (done inside
``eval_flagship_v4.load_v4_from_ck``) AND the optimiser is built over the
emission's parameters only AND world+head are md5-checksummed before/after.

⚠️ POD-SIDE ONLY for the full path: this box has no GPU and no v5f checkpoint.
What IS runnable (and run) here: ``python -m py_compile`` and the standalone
:class:`UnicycleEmission` smoke test ``stack/tests/test_unicycle_emission.py``.

Usage (pod5; PYTHONPATH=/workspace/TanitAD/stack):

  python3 train_v58f_unicycle_head.py \
      --ckpt /workspace/experiments/flagship-v5f-.../ckpt_step30000.pt \
      --anchors-dense /workspace/experiments/anchors/anchors_dense_1to20.pt \
      --v2-cache  /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
      --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
      --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
      --v2-subframe 176x624 --out /workspace/experiments/w4-unicycle-head
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

DT = 0.1                     # 10 Hz tick — the dense-horizon contract
A_MAX = 4.0                  # |a| bound, m/s^2 (census criterion)
KAPPA_MAX = 0.2              # |kappa| bound, 1/m  (0.2*v <= 0.33*v + 0.05 ∀ v>=0)
SPEED_SCALE = 10.0           # hard contract with the v1 trunk (flagship_losses)
REGISTRY_V5F_ORACLE = 0.1975  # v5f 30k oracle ADE — the W4 gate reference
GATE_ACCEL_MAE = 1.5         # m/s^2, pre-registered
GATE_ORACLE_FACTOR = 1.10    # oracle not worse than +10 %


# ============================================================================
# unicycle integration — accumulate_se2 IS the integrator, not a re-derivation
# ============================================================================
def unicycle_rollout(a_ctl: Tensor, kappa: Tensor, v0: Tensor,
                     dt: float = DT) -> tuple[Tensor, Tensor]:
    """Integrate per-candidate (a, kappa) control sequences to ego waypoints.

    ``a_ctl``/``kappa`` [B, N, K], ``v0`` [B] -> ``(wp [B, N, K, 2],
    v_pre [B, N, K])`` where ``v_pre[..., k]`` is the speed ENTERING step k
    (the one dx_k and dyaw_k use). Discretisation: module docstring — the
    train_unicycle_readout.py:247-260 convention, composed through
    :func:`tanitad.models.metric_dynamics.accumulate_se2` (called, not
    re-implemented, so the SE(2) geometry is the programme's one convention).
    """
    from tanitad.models.metric_dynamics import accumulate_se2
    b, n, k = a_ctl.shape
    v = v0.to(a_ctl.dtype)[:, None].expand(b, n).reshape(b * n).clone()
    rows, v_pre = [], []
    af = a_ctl.reshape(b * n, k)
    kf = kappa.reshape(b * n, k)
    for j in range(k):
        v_pre.append(v)
        rows.append(torch.stack(
            [v * dt, torch.zeros_like(v), kf[:, j] * v * dt], dim=-1))
        v = (v + af[:, j] * dt).clamp_min(0.0)      # AFTER the row (line 258)
    wp = accumulate_se2(torch.stack(rows, dim=1))   # [B*N, K, 2]
    return wp.reshape(b, n, k, 2), torch.stack(v_pre, dim=1).reshape(b, n, k)


class UnicycleEmission(nn.Module):
    """The ONLY trainable module of W4: per-candidate feature -> (a, kappa) ->
    unicycle waypoints.

    ``forward(feat [B, N, F], v0 [B]) -> (a [B, N, K], kappa [B, N, K],
    wp [B, N, K, 2])``. ``feat`` is EITHER the offset-head query ``q``
    (``--cond feature``, F = decoder d, captured by :class:`OffsetFeatureTap`)
    OR the flattened detached ``anchor_traj`` (``--cond anchor``, F = K*2 —
    the projection-to-manifold FALLBACK of V58F_FUSION.md: an
    anchor-to-controls projector, acceptable for W4 per the wedge spec).
    ``v0/SPEED_SCALE`` is appended as one extra input column (the trunk's own
    speed-channel convention), so ``in_dim = feat_dim + 1``.

    MLP: 2 layers, hidden 256 (the W4 spec). The FINAL layer is ZERO-INIT, so
    at step 0 ``a = kappa = 0`` and the emitted fan is the constant-velocity
    straight rollout — a defined, feasible warm start (the CV baseline), not
    noise. Bounded activations make every emitted candidate feasible by
    construction.

    ⛔ **THE BOUNDING FUNCTION IS A GATED CHOICE, and ``tanh`` is the LEGACY one.**
    ``squash="tanh"`` (the default) reproduces the W4/v5.8f emission BIT-EXACTLY —
    every banked v5.8f number was measured under it, so the default may not move.
    ``squash="squash"`` uses :func:`tanitad.models.kinematic._squash`, which is what
    the same programme MEASURED to be correct and already ships in the kinematic
    module.

    Why the legacy default is a trap for any head that TRAINS here — MEASURED
    2026-08-15 on this box, float32:

    * ``d/draw tanh(raw)`` is **EXACTLY 0.0 from raw ≥ 10**, because ``tanh``
      rounds to exactly ``1.0f`` there and ``1 - 1*1`` is exactly zero. The
      ``kinematic._squash`` docstring cites ``tanh(51)`` for this; **the true
      cliff is 5× closer than that example suggests**, and ``raw ≥ 10`` is an
      ordinary pre-activation — v6's own S-W run logged two gradient-spike
      episodes, one peaking at ``gnorm 354 076``, which is exactly what pushes a
      pre-activation there. Past that point the head is DEAD and cannot learn back.
    * ``_squash`` is the IDENTITY inside the range (grad exactly 1.0 at 0.5× and
      0.9× the limit) and still carries gradient at 100× the limit (1.016e-06),
      so it has neither the dead-head cliff nor the shrink that ruled out plain
      softsign (which returns 0.0333 for a 0.04 curvature — a 16.7 % error on a
      control nowhere near its bound, and the reason a decode could not reproduce
      its own anchor).

    ⇒ **New arms should pass ``squash="squash"``.** v6 does (`V6Config.emission_squash`),
    and it costs nothing there because ``emission.`` sits in the ``planner`` group,
    which S-W does not train — so the choice lands before the head is ever fitted.
    The activation holds no parameters, so this changes no ``state_dict`` key or
    shape and cannot affect a strict resume.
    """

    def __init__(self, feat_dim: int, k: int = 20, hidden: int = 256,
                 a_max: float = A_MAX, kappa_max: float = KAPPA_MAX,
                 dt: float = DT, squash: str = "tanh"):
        super().__init__()
        if squash not in ("tanh", "squash"):
            raise ValueError(
                f"squash must be 'tanh' (legacy, bit-exact v5.8f) or 'squash' "
                f"(measured-correct, kinematic._squash), got {squash!r}")
        self.feat_dim, self.k, self.dt = int(feat_dim), int(k), float(dt)
        self.a_max, self.kappa_max = float(a_max), float(kappa_max)
        self.squash = squash
        self.net = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden), nn.GELU(),
            nn.Linear(hidden, k * 2))
        nn.init.zeros_(self.net[-1].weight)         # CV warm start
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: Tensor, v0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(f"feat dim {feat.shape[-1]} != {self.feat_dim}")
        b, n, _ = feat.shape
        vcol = (v0.to(feat.dtype) / SPEED_SCALE)[:, None, None].expand(b, n, 1)
        raw = self.net(torch.cat([feat, vcol], dim=-1)).reshape(b, n, self.k, 2)
        if self.squash == "tanh":                      # legacy: bit-exact v5.8f
            a_ctl = self.a_max * torch.tanh(raw[..., 0])
            kappa = self.kappa_max * torch.tanh(raw[..., 1])
        else:                                          # measured-correct
            from tanitad.models.kinematic import _squash  # noqa: PLC0415
            a_ctl = _squash(raw[..., 0], self.a_max)
            kappa = _squash(raw[..., 1], self.kappa_max)
        wp, _ = unicycle_rollout(a_ctl, kappa, v0, dt=self.dt)
        return a_ctl, kappa, wp


class OffsetFeatureTap:
    """Capture the per-candidate query ``q`` [B, N, d] feeding
    ``AnchoredDiffusionDecoder.offset_head`` (refc.py:1193-1198) via a forward
    PRE-hook. ``_decode`` runs 1 + diffusion_steps times per head forward; the
    LAST call is the final denoise pass whose offset produced the emitted fan,
    so :meth:`last` after :meth:`clear`-then-forward is that pass's ``q``."""

    def __init__(self, offset_head: nn.Linear):
        self._buf: list[Tensor] = []
        self._h = offset_head.register_forward_pre_hook(
            lambda _m, args: self._buf.append(args[0]))

    def clear(self) -> None:
        self._buf.clear()

    def last(self) -> Tensor:
        if not self._buf:
            raise RuntimeError("OffsetFeatureTap: no decoder pass captured — "
                               "was head.forward run after clear()?")
        return self._buf[-1]

    def n_calls(self) -> int:
        return len(self._buf)

    def remove(self) -> None:
        self._h.remove()


# ============================================================================
# kinematics — the accel-MAE instrument (float32, waypoint-derived)
# ============================================================================
def speeds_and_accels(wp: Tensor, dt: float = DT) -> tuple[Tensor, Tensor]:
    """[..., K, 2] waypoints (ego origin implicit at 0) -> (speeds [..., K],
    accels [..., K-1]) — the ``kin_metrics`` geometry of
    train_unicycle_readout.py:49-64 (origin prepended, finite differences)."""
    z = wp.new_zeros(wp.shape[:-2] + (1, 2))
    d = torch.diff(torch.cat([z, wp], dim=-2), dim=-2)
    sp = d.norm(dim=-1) / dt
    return sp, torch.diff(sp, dim=-1) / dt


def accel_mae(wp: Tensor, gt: Tensor, dt: float = DT) -> float:
    """Mean |accel_pred - accel_gt| in m/s^2, float32 — the four-families
    longitudinal instrument this gate is read against (v5f selected: 8.11)."""
    _, ap = speeds_and_accels(wp.float(), dt)
    _, ag = speeds_and_accels(gt.float(), dt)
    return float((ap - ag).abs().mean())


def emit_unicycle_fan(out: dict, fan: Tensor, horizons) -> dict:
    """⭐ THE OUTPUT-SURFACE SWAP (``--emission unicycle``): a NEW dict in which
    the emitted unicycle fan REPLACES ``anchor_traj``, with every key the head
    derives from the pick (``traj``/``wp_seq``/``waypoints``) recomputed on the
    new fan under the FROZEN selector's own ``sel_idx`` — leaving one of them
    describing the old fan is exactly how a swap silently half-lands
    (the apply_c2_selection lesson). Selection keys themselves are untouched:
    W4 changes the fan's parameterisation, not the selector."""
    new = dict(out)
    ar = torch.arange(fan.shape[0], device=fan.device)
    new["anchor_traj"] = fan
    new["traj"] = fan[ar, out["sel_idx"]]
    new["wp_seq"] = new["traj"]
    new["waypoints"] = {k: new["traj"][:, i] for i, k in enumerate(horizons)}
    return new


def module_md5(*modules) -> str:
    import hashlib
    h = hashlib.md5()
    for mod in modules:
        for n_, p in sorted(mod.named_parameters()):
            h.update(n_.encode())
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ============================================================================
# losses
# ============================================================================
def w4_losses(fan: Tensor, a_ctl: Tensor, kappa: Tensor, tgt: Tensor,
              w_speed: float, w_reg: float, dt: float = DT) -> dict:
    """Winner-takes-all L2 + winner speed-profile L1 + control-magnitude L2.

    ``fan`` [B, N, K, 2] float32, ``tgt`` [B, K, 2] float32."""
    err = (fan - tgt[:, None]).norm(dim=-1).mean(dim=-1)        # [B, N]
    with torch.no_grad():
        win = err.argmin(dim=1)                                 # GT-nearest
    ar = torch.arange(fan.shape[0], device=fan.device)
    wp_w = fan[ar, win]                                         # [B, K, 2]
    l_wta = (wp_w - tgt).pow(2).sum(-1).mean()
    # speed-profile L1 (eps INSIDE the sqrt — the stopped-path NaN-grad trap,
    # train_unicycle_readout.py:298-304)
    z = wp_w.new_zeros(wp_w.shape[0], 1, 2)
    dp = torch.diff(torch.cat([z, wp_w], dim=1), dim=1)
    dg = torch.diff(torch.cat([z, tgt], dim=1), dim=1)
    vp = (dp.pow(2).sum(-1) + 1e-12).sqrt() / dt
    vg = (dg.pow(2).sum(-1) + 1e-12).sqrt() / dt
    l_speed = (vp - vg).abs().mean()
    l_reg = a_ctl.pow(2).mean() + kappa.pow(2).mean()           # whole fan
    loss = l_wta + w_speed * l_speed + w_reg * l_reg
    return {"loss": loss, "l_wta": l_wta, "l_speed": l_speed, "l_reg": l_reg,
            "winner_ade": err[ar, win].mean().detach(),
            "winner_wp": wp_w.detach(), "win": win}


# ============================================================================
# data plumbing (pod-side; mirrors eval_flagship_v4 / train_flagship_v4 seams)
# ============================================================================
def build_train_episodes(a, *, cache_frame, train_frame):
    """The TRAIN-side twin of ``eval_flagship_v4.build_v2_val_episodes`` —
    same parity guard, same provider seam, same geometry binding, pointed at
    ``--v2-cache`` (the canonical ``physicalai-train-e438721ae894`` corpus)."""
    from tanitad.data import parity
    from tanitad.data.v2_dataset import build_v2_providers
    dirs = list(a.v2_cache)
    rec = parity.assert_v2_parity_cache(dirs, label="--v2-cache",
                                        require=bool(a.require_parity))
    slice_frame = None if train_frame == cache_frame else train_frame
    eps = build_v2_providers(dirs, lru_size=int(a.v2_lru), frame=slice_frame,
                             verbose=True)
    if not eps:
        raise SystemExit(f"[w4] no *.v2ep.pt under {dirs} — does --v2-cache "
                         f"point at the split dir?")
    binding = parity.assert_v2_geometry_matches(
        rec, train_frame, label="--v2-cache", providers=eps, parent=cache_frame)
    return eps, {"train_parity": rec, "geometry_binding": binding}


def make_sampler(ds, eps_per_batch: int, rng: random.Random):
    """Episode-grouped batch sampler: FEW episodes x MANY windows per batch.
    The I/O shape MEASURED 2026-08-06 (train_unicycle_readout.py:88-95): random
    windows from random episodes over a payload LRU on MooseFS is ~30 cold
    payload loads per batch; grouping cuts that ~8x. The mild within-batch
    correlation is an accepted, stated trade for a head-scale fine-tune."""
    ep2idx: dict[int, list[int]] = {}
    for i, (e, _t) in enumerate(ds.index):
        ep2idx.setdefault(e, []).append(i)
    ep_ids = list(ep2idx)

    def sample(bs: int) -> list[int]:
        chosen = [ep_ids[rng.randrange(len(ep_ids))]
                  for _ in range(min(eps_per_batch, len(ep_ids)))]
        out = []
        gi = 0
        while len(out) < bs:
            pool = ep2idx[chosen[gi % len(chosen)]]
            out.append(pool[rng.randrange(len(pool))])
            gi += 1
        return out

    return sample


# ============================================================================
# the frozen forward: encode -> head -> conditioning feature   (POD-SIDE)
# ============================================================================
def frozen_forward(world, head, tap, batch, device, *, probes, amp_on,
                   cond: str):
    """Run the FROZEN v5f path under no_grad (+bf16 autocast) and return
    ``(out, feat_f32, v0, tgt_f32)``. ``feat`` is detached — the emission is
    trained on it as a constant input, which is what frozen-trunk means."""
    import refb_labels
    from train_flagship_v4 import _goal_inputs, _imagination_inputs
    v0 = batch["pose_last"][:, 3].float()
    horizons = head.cfg.horizons
    tgt = refb_labels.waypoint_targets(
        batch["pose_last"].float(),
        batch["future_poses"][:, :max(horizons)].float(), horizons).float()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=amp_on):
        st = world.encode_window(batch["frames"])
        goal = _goal_inputs(head.cfg, batch, v0)
        imag = _imagination_inputs(world, head.cfg, batch, st, probes)
        tap.clear()
        out = head(st, v0, lambda_plan=1.0, **goal, **imag)
    if cond == "feature":
        feat = tap.last().detach().float()                    # [B, N, d]
    else:                                                     # "anchor" fallback
        feat = out["anchor_traj"].detach().float().flatten(2)  # [B, N, K*2]
    return out, feat, v0, tgt


# ============================================================================
# end-of-run mini-eval — SAME grid rule as eval defaults (episodes<40, stride 8)
# ============================================================================
@torch.no_grad()
def mini_eval(world, head, tap, emission, ds_val, device, *, probes, amp_on,
              cond: str, episodes: int = 40, stride: int = 8,
              batch: int = 16) -> dict:
    """oracle_ade / selected_ade / accel MAE on the emitted fan over the val
    grid ``e < episodes and t % stride == 0`` — the exact window-selection rule
    of ``train_flagship_v4.evaluate_planner`` / ``eval_flagship_v4
    collect_planner`` defaults, so the number is grid-comparable."""
    from torch.utils.data import default_collate
    from train_flagship_v4 import _to_device
    head.eval()
    horizons = head.cfg.horizons
    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < episodes and t % stride == 0]
    if not sel:
        raise SystemExit("[w4] mini-eval selected 0 windows — check "
                         "--episodes/--stride against the val cache")
    sums = {k: 0.0 for k in
            ("new_oracle", "new_selected", "new_winner",
             "orig_oracle", "orig_selected",
             "accel_mae_selected", "accel_mae_winner",
             "accel_mae_orig_selected", "viol_frac")}
    n = 0
    t0 = time.time()
    for b0 in range(0, len(sel), batch):
        idx = sel[b0:b0 + batch]
        b = _to_device(default_collate([ds_val[i] for i in idx]), device)
        out, feat, v0, tgt = frozen_forward(world, head, tap, b, device,
                                            probes=probes, amp_on=amp_on,
                                            cond=cond)
        a_ctl, kappa, fan = emission(feat, v0)
        new_out = emit_unicycle_fan(out, fan, horizons)
        bs = fan.shape[0]
        ar = torch.arange(bs, device=fan.device)
        err_new = (fan - tgt[:, None]).norm(dim=-1).mean(dim=-1)        # [B,N]
        err_old = (out["anchor_traj"].float() - tgt[:, None]).norm(
            dim=-1).mean(dim=-1)
        win = err_new.argmin(dim=1)
        sel_traj = new_out["traj"].float()                # frozen selector, new fan
        sums["new_oracle"] += float(err_new.min(dim=1).values.sum())
        sums["new_selected"] += float(err_new[ar, out["sel_idx"]].sum())
        sums["new_winner"] += float(err_new[ar, win].sum())
        sums["orig_oracle"] += float(err_old.min(dim=1).values.sum())
        sums["orig_selected"] += float(
            (out["traj"].float() - tgt).norm(dim=-1).mean(dim=-1).sum())
        sums["accel_mae_selected"] += accel_mae(sel_traj, tgt) * bs
        sums["accel_mae_winner"] += accel_mae(fan[ar, win], tgt) * bs
        sums["accel_mae_orig_selected"] += accel_mae(out["traj"], tgt) * bs
        # feasibility census on the EMITTED controls (should be ~0 by
        # construction — reported, not assumed):
        _, v_pre = unicycle_rollout(a_ctl, kappa, v0)
        yaw_rate = kappa * v_pre
        viol = ((a_ctl.abs() > A_MAX + 1e-4)
                | (yaw_rate.abs() > 0.33 * v_pre + 0.05 + 1e-4))
        sums["viol_frac"] += float(viol.float().mean()) * bs
        n += bs
    head.train()
    res = {k: round(v / n, 6) for k, v in sums.items()}
    res.update(n_windows=n, grid={"episodes": episodes, "stride": stride,
                                  "batch": batch},
               wallclock_s=round(time.time() - t0, 1))
    return res


# ============================================================================
# main (POD-SIDE: needs GPU + the v5f checkpoint + the v2 corpora)
# ============================================================================
def build_args(argv=None):
    ap = argparse.ArgumentParser("train_v58f_unicycle_head", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="v5f checkpoint "
                    "(keys: model, grounding, head[, goal_head])")
    ap.add_argument("--head-config", default=None,
                    help="run config.json (default: sibling of --ckpt)")
    ap.add_argument("--anchors-dense", default=None,
                    help="trained dense-anchor buffer (pass explicitly)")
    ap.add_argument("--probe-vocab", default=None,
                    help="probe_vocab.pt for cond_imagination heads "
                         "(default: sibling of --ckpt)")
    # corpus (v2 compressed only — the ONLY format v5 has)
    ap.add_argument("--v2-cache", required=True, nargs="+",
                    help="v2 compressed TRAIN split dir(s) — the canonical "
                         "physicalai-train-e438721ae894 build")
    ap.add_argument("--v2-val-cache", required=True, nargs="+",
                    help="v2 compressed VAL split dir(s)")
    ap.add_argument("--v2-lru", type=int, default=64)
    ap.add_argument("--v2-subframe", default=None, metavar="HxW",
                    help="centred sub-frame the model reads (e.g. 176x624) — "
                         "MUST match the run; cross-checked vs config.json")
    ap.add_argument("--require-parity", action="store_true")
    from tanitad.geometry import add_geometry_args
    add_geometry_args(ap)      # --frame-h/--frame-w/--frame-hfov/--projection
    # training
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--eps-per-batch", type=int, default=4,
                    help="episode-grouped sampling (the MooseFS I/O shape)")
    ap.add_argument("--save-every", type=int, default=500,
                    help="checkpoint + metrics.json cadence; the per-500 log "
                         "row carries winner ADE + winner accel MAE so the "
                         "gate is readable off the log")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--emission", choices=("unicycle",), default="unicycle",
                    help="fan parameterisation at the output surface. W4 has "
                         "exactly one arm; the flag names the swap "
                         "(emit_unicycle_fan) so the assembly can gate on it.")
    ap.add_argument("--cond", choices=("feature", "anchor"), default="feature",
                    help="'feature' = offset-head query q (refc.py:1193-1198) "
                         "via forward pre-hook; 'anchor' = anchor_traj.detach()"
                         " flattened + v0 (projection-to-manifold fallback)")
    ap.add_argument("--w-speed", type=float, default=0.5,
                    help="winner speed-profile L1 weight (v1.7 lever; the "
                         "VALUE is a knob, not a measured optimum)")
    ap.add_argument("--w-reg", type=float, default=0.01,
                    help="L2 on (a, kappa) magnitudes — the W4 spec's 0.01")
    # mini-eval grid (eval defaults)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=16)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = build_args(argv)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    rng = random.Random(a.seed)
    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[w4] WARNING: cuda unavailable, falling back to cpu", flush=True)
        device = "cpu"
    amp_on = (device == "cuda") and not a.no_amp
    os.makedirs(a.out, exist_ok=True)

    from torch.utils.data import default_collate

    from eval_flagship_v4 import (_eval_cfg, _plan, build_v2_val_episodes,
                                  load_v4_from_ck, resolve_eval_frames)
    from flagship_v4_data import FlagshipV4Dataset
    from tanitad.data import parity
    from train_flagship_v4 import _to_device

    # ---- geometry FIRST, cross-checked against the run's own config.json ----
    cfg = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(
        a, cfg, label="train_v58f_unicycle_head")
    plan = _plan(cfg)
    head_cfg_path = a.head_config or str(Path(a.ckpt).parent / "config.json")
    run_cfg = None
    if Path(head_cfg_path).exists():
        try:
            run_cfg = json.loads(Path(head_cfg_path).read_text())
        except Exception as ex:
            print(f"[w4] WARNING: could not parse {head_cfg_path}: {ex}",
                  flush=True)
    frame_check = parity.assert_eval_frame_matches_run(
        run_cfg, model_frame, label="--ckpt vs W4 train frame",
        cache_frame=cache_frame)
    if not frame_check["checked"]:
        print(f"[w4] ⚠ FRAME UNVERIFIED: {frame_check['note']}", flush=True)

    # ---- frozen v5f: EXACTLY the eval harness's loader ----------------------
    print(f"[w4] loading checkpoint {a.ckpt} ...", flush=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and "head" in ck):
        raise SystemExit("[w4] --ckpt has no 'head' key — W4 retrofits the v4 "
                         "planner head; a plain trunk has no fan to retrofit.")
    world, grounding, head, base_step, hcfg, goal_head = load_v4_from_ck(
        ck, device,
        head_config_path=(head_cfg_path if Path(head_cfg_path).exists()
                          else None),
        anchors_dense_path=a.anchors_dense, frame=model_frame)
    del ck
    horizons = head.cfg.horizons
    if tuple(horizons) != tuple(range(1, len(horizons) + 1)):
        raise SystemExit(f"[w4] head horizons {horizons} are not contiguous "
                         f"1..K @10 Hz — the unicycle integration is defined "
                         f"on the dense tick only.")
    K = len(horizons)

    probes = None
    if getattr(head.cfg, "cond_imagination", False):
        pv = Path(a.probe_vocab or (Path(a.ckpt).parent / "probe_vocab.pt"))
        if not pv.exists():
            raise SystemExit(f"[w4] cond_imagination head but no {pv} — a "
                             "silent skip would score a head minus 32 inputs")
        probes = torch.load(pv, map_location=device)
        print(f"[w4] imagination probes: {tuple(probes.shape)}", flush=True)

    # ---- frozen means PROVED frozen ----------------------------------------
    # load_v4_from_ck already requires_grad_(False)d world/grounding/head/
    # goal_head; the checksum is the second lock, the emission-only optimiser
    # the third (train_unicycle_readout.py contract).
    assert not any(p.requires_grad for m in (world, grounding, head)
                   for p in m.parameters())
    md5_before = module_md5(world, head)
    print(f"[w4] trunk+head frozen · base step {base_step} · "
          f"md5 {md5_before[:12]}", flush=True)

    # ---- data ---------------------------------------------------------------
    train_eps, train_prov = build_train_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    val_eps, val_prov = build_v2_val_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    ds_train = FlagshipV4Dataset(train_eps, window=cfg.predictor.window,
                                 max_horizon=plan.max_horizon,
                                 maneuver_h=plan.maneuver_h,
                                 channels=cfg.encoder.in_channels)
    ds_val = FlagshipV4Dataset(val_eps, window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    print(f"[w4] train {len(train_eps)} eps / {len(ds_train)} windows; "
          f"val {len(val_eps)} eps / {len(ds_val)} windows", flush=True)
    sample = make_sampler(ds_train, a.eps_per_batch, rng)

    # ---- the new module -----------------------------------------------------
    tap = OffsetFeatureTap(head.decoder.offset_head)
    feat_dim = (head.decoder.offset_head.in_features if a.cond == "feature"
                else K * 2)
    emission = UnicycleEmission(feat_dim=feat_dim, k=K).to(device)
    n_par = sum(p.numel() for p in emission.parameters())
    print(f"[w4] UnicycleEmission cond={a.cond} feat_dim={feat_dim} K={K} "
          f"({n_par/1e6:.3f} M trainable; frozen everything else)", flush=True)
    opt = torch.optim.AdamW(emission.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)

    log_path = os.path.join(a.out, "train_log.jsonl")
    fh = open(log_path, "a")
    fh.write(json.dumps({
        "run": "w4-unicycle-emission", "args": vars(a),
        "base_ckpt": a.ckpt, "base_step": base_step,
        "trunk_head_md5": md5_before, "n_trainable": n_par,
        "cond_mode": a.cond, "horizons_K": K,
        "train_parity": {"n_dirs": len(a.v2_cache)},
        "_evidence_class": "MEASURED (ours; artifact = this log)"}) + "\n")
    fh.flush()

    history: list[dict] = []
    acc = {"loss": 0.0, "l_wta": 0.0, "l_speed": 0.0, "l_reg": 0.0,
           "winner_ade": 0.0, "winner_accel_mae": 0.0, "n": 0}
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = sample(a.bs)
        b = _to_device(default_collate([ds_train[i] for i in idx]), device)
        out, feat, v0, tgt = frozen_forward(world, head, tap, b, device,
                                            probes=probes, amp_on=amp_on,
                                            cond=a.cond)
        # emission + integration OUTSIDE autocast: float32 end to end, so the
        # accel gate is read off the same numerics it will be evaluated at.
        a_ctl, kappa, fan = emission(feat, v0)
        L = w4_losses(fan, a_ctl, kappa, tgt, a.w_speed, a.w_reg)
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        gnorm = torch.nn.utils.clip_grad_norm_(emission.parameters(), 5.0)
        opt.step()
        sched.step()

        bs = fan.shape[0]
        acc["loss"] += float(L["loss"].detach()) * bs
        acc["l_wta"] += float(L["l_wta"].detach()) * bs
        acc["l_speed"] += float(L["l_speed"].detach()) * bs
        acc["l_reg"] += float(L["l_reg"].detach()) * bs
        acc["winner_ade"] += float(L["winner_ade"]) * bs
        acc["winner_accel_mae"] += accel_mae(L["winner_wp"], tgt) * bs
        acc["n"] += bs

        if step % a.log_every == 0:
            rec = {"step": step,
                   "loss": round(float(L["loss"].detach()), 5),
                   "winner_ade": round(float(L["winner_ade"]), 4),
                   "gnorm": round(float(gnorm), 3),
                   "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{step}] {rec}", flush=True)

        if step % a.save_every == 0:
            n_ = max(acc["n"], 1)
            row = {"step": step,
                   **{k: round(v / n_, 5) for k, v in acc.items() if k != "n"},
                   "elapsed_s": round(time.time() - t0, 1)}
            history.append(row)
            acc = {k: 0.0 for k in acc} | {"n": 0}
            with open(os.path.join(a.out, "metrics.json"), "w") as mf:
                json.dump({"history": history, "args": vars(a),
                           "base_step": base_step,
                           "_read": "winner_ade / winner_accel_mae are "
                                    "TRAIN-batch running means over the last "
                                    "save window; the gate numbers are the "
                                    "held-out mini-eval in w4_gate.json",
                           "_evidence_class": "MEASURED (ours)"}, mf, indent=1)
            torch.save({"emission": emission.state_dict(),
                        "cond_mode": a.cond, "feat_dim": feat_dim, "k": K,
                        "step": step, "args": vars(a), "base_ckpt": a.ckpt,
                        "base_step": base_step},
                       os.path.join(a.out, "unicycle_emission.pt"))
            fh.write(json.dumps({"per500": row}) + "\n")
            fh.flush()
            print(f"[w4 @{step}] {row}", flush=True)

    # ---- frozen proof + the pre-registered gate -----------------------------
    md5_after = module_md5(world, head)
    ev = mini_eval(world, head, tap, emission, ds_val, device, probes=probes,
                   amp_on=amp_on, cond=a.cond, episodes=a.episodes,
                   stride=a.stride, batch=a.eval_batch)
    oracle_cap = round(GATE_ORACLE_FACTOR * REGISTRY_V5F_ORACLE, 6)
    gate = {
        "accel_mae_selected_lt_1p5": ev["accel_mae_selected"] < GATE_ACCEL_MAE,
        "oracle_ade_not_worse_than_plus10pct_vs_0.1975":
            ev["new_oracle"] <= oracle_cap,
        "oracle_cap_m": oracle_cap,
    }
    gate["PASS"] = (gate["accel_mae_selected_lt_1p5"]
                    and gate["oracle_ade_not_worse_than_plus10pct_vs_0.1975"])
    summary = {
        "wedge": "W4 unicycle-anchor head retrofit (V58F_FUSION.md §3)",
        "mini_eval": ev,
        "oracle_ade": ev["new_oracle"],
        "selected_ade": ev["new_selected"],
        "accel_mae_selected": ev["accel_mae_selected"],
        "reference": {"v5f_oracle_ade_registry": REGISTRY_V5F_ORACLE,
                      "orig_oracle_ade_same_grid": ev["orig_oracle"],
                      "orig_selected_ade_same_grid": ev["orig_selected"],
                      "orig_accel_mae_selected_same_grid":
                          ev["accel_mae_orig_selected"]},
        "gate": gate,
        "emission": {"cond_mode": a.cond, "feat_dim": feat_dim, "k": K,
                     "a_max": A_MAX, "kappa_max": KAPPA_MAX,
                     "n_trainable": n_par},
        "trunk_frozen_proof": {"md5_before": md5_before,
                               "md5_after": md5_after,
                               "identical": md5_before == md5_after},
        "steps": a.steps, "base_ckpt": a.ckpt, "base_step": base_step,
        "wall_s": round(time.time() - t0, 1),
        "_estimator_note": ("plain corpus-grid means over the eval-default "
                            "window grid; the DECISION-grade interval for any "
                            "registry claim is the episode-cluster bootstrap "
                            "(taniteval/ci.py) — run it before publishing"),
        "_evidence_class": "MEASURED (ours; artifact = this JSON)",
    }
    with open(os.path.join(a.out, "w4_gate.json"), "w") as gf:
        json.dump(summary, gf, indent=1)
    fh.write(json.dumps({"summary": summary}) + "\n")
    fh.close()
    tap.remove()
    print(f"\n[W4 SUMMARY] {json.dumps(summary, indent=1)}", flush=True)
    if not summary["trunk_frozen_proof"]["identical"]:
        raise SystemExit("⛔ TRUNK/HEAD CHANGED DURING TRAINING — run invalid")
    print(f"[W4 GATE] {'PASS' if gate['PASS'] else 'FAIL'} "
          f"(accel_mae_selected {ev['accel_mae_selected']:.3f} vs "
          f"{GATE_ACCEL_MAE}; oracle {ev['new_oracle']:.4f} vs cap "
          f"{oracle_cap})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
