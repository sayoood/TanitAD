"""V18 E3.4 — Stage-A PREDICTOR-ONLY post-training of the v5f trunk (the
pre-designed gain repair; registry §1.14 W7 block).

WHY (the night's convergence — MODEL_REGISTRY.md §1.14, w3_gate.json): the
v5f predictor responds to counterfactual actions with the RIGHT SIGN
laterally (99.5 %) inside a 3-dim latent subspace, but at only ~0.27 of the
physical gain (unicycle-analytic reference), and the longitudinal sign is
unreliable (74–79 %). W4b/W4c's scoring failures, W7's ceiling at every K,
and §1.12's action echo are ONE DEFECT: the trunk under-weights actions in
its rollout. Stage-A post-trains the PREDICTOR ALONE on the frozen trunk to
repair the gain. (The +encoder variant is a LATER flag/arm — deliberately
not implemented here so this arm stays single-lever and attributable.)

WHAT TRAINS: ``world.predictor`` (OperativePredictor) ONLY. Encoder, readout,
grounding, and every head stay FROZEN — enforced three ways (the
train_unicycle_readout contract): ``requires_grad`` False outside the
predictor, an optimiser built over predictor params only, and md5 checksums
of the frozen parts before/after (:func:`frozen_proof_md5`); the predictor's
own md5 must CHANGE or the run is declared a no-op and refused.

LOSSES per batch (TRAIN corpus windows only — ``--v2-cache``; val windows in
training are refused by construction: the two corpora are separate loaders
and overlapping cache dirs abort):
  (a) **L_ctrl** (``--w-ctrl`` 1.0) — THE GAIN REPAIR. Counterfactual actions
      per window via the stage_a_probes builder (left/right/brake/throttle
      kappa/accel deltas through the corpus steer encoding, stage_a_probes.
      apply_counterfactual) PLUS one random per-window (Δκ, Δa) draw, all
      clamped to the physical envelope |a| <= 4 m/s², |κ| <= 0.2 1/m
      (A_MAX/KAPPA_MAX, the W4 bounds). Roll the predictor ``--k`` = 10 under
      them (``rollout_transitions``), decode (``decode_transitions``), and L1
      to the unicycle-analytic displacement
      (``train_v58f_unicycle_head.unicycle_rollout`` via
      ``stage_a_probes.analytic_endpoints`` — IMPORTED, the W4
      discretisation). Default form ``--ctrl-form response``: the L1 is on
      the RESPONSE (decoded_cf − decoded_factual) vs (analytic_cf −
      analytic_factual) — the exact quantity the W3 gain gate measures; the
      absolute factual position is L_factual's job, so attribution between
      the two losses stays clean. ``--ctrl-form absolute`` is the literal
      pre-design form ||decode(roll(z, a_cf)) − Unicycle(a_cf, v0)||.
  (b) **L_factual** (``--w-fact`` 1.0) — the same roll under the RECORDED
      actions, L1 to the TRUE future waypoints (``gt_ego_waypoints``): keeps
      the factual rollout anchored while the gain moves.
  (c) **L_scene** (``--w-scene`` 0.3) — actions move the EGO, not the scene.
      Top-``--n-basis`` (8) UNCENTERED PCA basis of the counterfactual latent
      deltas [B*5, S], computed ON THE FLY per batch (detached); penalised:
      (i) counterfactual-roll latent change OUTSIDE that subspace vs the
      factual roll, and (ii) the factual rolled ẑ_{t+k} vs the ENCODED true
      future frame z_{t+k}, also on the complement (ego-subspace stability +
      anchor). ⚠️ ESTIMATOR STATED HONESTLY: the batch-local PCA is an
      APPROXIMATION of W3's corpus-level 3-dim action subspace (w3_gate.json
      P6) — cheap, per-batch, uncentered like the W3 P6 (deltas have a
      meaningful zero); n-basis 8 > 3 gives slack for batch noise.

MONITOR (every ``--probe-every`` = 100 steps): the W3 probe run CHEAPLY on
``--probe-windows`` = 32 FIXED val windows (the first 32 of the canonical
881 grid) — per-channel sign rate + median gain, lateral AND longitudinal,
plus the 32-window factual-roll ADE. Logged as ``train_vs_probe`` rows: the
training curve IS the gate trajectory.

END OF RUN: the FULL W3 probe pack (stage_a_probes machinery byte-imported:
``probe_batch`` + ``channel_stats`` + ``pca_subspace_stats``) on the
canonical 881 grid (episodes < 40, t % 8 == 0), run BEFORE training (the
baseline + the no-harm reference) and AFTER (the gate) in the SAME run.

⛔ PRE-REGISTERED GATE (registry §1.14, verbatim targets; written to
``<out>/stage_a_gate.json``):
  * lateral gain median ∈ [0.5, 2.0] (left AND right);
  * longitudinal sign rate >= 0.95 (brake AND throttle);
  * lateral sign rate STAYS >= 0.95 (left AND right);
  * P6 subspace dims (80 % variance) <= 32 — preserve the factorisation;
  * NO-HARM: factual-roll ADE on the val grid NOT worse than +10 % vs the
    pre-training predictor (both measured in this run, same grid, same
    decode). Longitudinal gain is REPORTED, not gated.
  BOTH OUTCOMES BOUND IN ADVANCE:
    PASS -> W7 re-run + E1.4 T1 re-run on the repaired trunk.
    FAIL -> the gain defect is not post-trainable at predictor-scale; the
            lever moves to joint trunk training (v6).

TIER STAMP: **T1-diagnostic** — the probe conditions the WM roll on
counterfactual (non-recorded) actions; it measures controllability, NEVER
driving performance (EVAL_DOCTRINE.md). No number here is quotable as a
driving result.

ESTIMATOR: point estimates over the corpus grid; per-window deltas + factual
ADEs are BANKED to ``<out>/stage_a_windows.npz`` so the decision-grade
episode-cluster bootstrap (taniteval/ci.py, 40 val episodes) can run
pod-side before any registry claim.

⚠️ POD-SIDE ONLY for the full path (GPU + v5f checkpoint + v2 corpora).
Runnable here: ``python -m py_compile`` and the CPU tests
(``stack/tests/test_stage_a_train.py``).

Usage (pod5; PYTHONPATH=/workspace/TanitAD/stack required or the trainer
dies with ModuleNotFound: tanitad):

  python3 scripts/train_stage_a.py \
      --ckpt /workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt \
      --v2-cache /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
      --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
      --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
      --v2-subframe 176x624 \
      --out /workspace/experiments/stage-a-predictor
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
# stack root too, so `import tanitad` resolves even without the pod-side
# PYTHONPATH (harmless when PYTHONPATH is set — same directory).
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from stage_a_probes import (CF_CHANNELS, CHANNEL_SIGN_AXIS,  # noqa: E402
                            DKAPPA_DEFAULT, DACCEL_DEFAULT, DT, GATE_GAIN,
                            GATE_P6_DIMS, GATE_SIGN, GATED_CHANNELS,
                            K_DEFAULT, WHEELBASE, _agg, analytic_endpoints,
                            apply_counterfactual, channel_stats,
                            kappa_of_steer, pca_subspace_stats, probe_batch,
                            steer_of_kappa)
from train_v58f_unicycle_head import (A_MAX, KAPPA_MAX,  # noqa: E402
                                      build_train_episodes, make_sampler,
                                      module_md5)

# ---- the pre-registered stage-A surface (registry §1.14, verbatim) ----------
NOHARM_FACTOR = 1.10          # factual-roll ADE post <= 1.10 * pre
LAT_CHANNELS = ("left", "right")
LON_CHANNELS = ("brake", "throttle")
TRAIN_ARMS = ("left", "right", "brake", "throttle", "random")
N_BASIS_DEFAULT = 8           # top-8 batch-local PCA (W3's subspace is 3-dim)
LR_DEFAULT = 1e-5             # small — a repair, not a retrain
STEPS_DEFAULT = 3000
PROBE_EVERY_DEFAULT = 100
PROBE_WINDOWS_DEFAULT = 32
RAND_DKAPPA_MAX_DEFAULT = 0.05   # 1/m — random-draw half-range (then clamped
RAND_DACCEL_MAX_DEFAULT = 3.0    # m/s²   to the physical envelope)
OUTCOME_PASS = "PASS -> W7 re-run + E1.4 T1 re-run on the repaired trunk"
OUTCOME_FAIL = ("FAIL -> the gain defect is not post-trainable at "
                "predictor-scale; the lever moves to joint trunk training "
                "(v6)")


# ============================================================================
# counterfactual sampling — the stage_a_probes builder + a random arm, all
# clamped to the physical envelope (pure, CPU-tested)
# ============================================================================
def sample_random_deltas(b: int, gen: torch.Generator, dkappa_max: float,
                         daccel_max: float) -> tuple[Tensor, Tensor]:
    """Per-window joint (Δκ [1/m], Δa [m/s²]) uniform draws in
    [-dkappa_max, dkappa_max] x [-daccel_max, daccel_max]. Deterministic
    under ``gen``. The ENVELOPE guarantee lives in :func:`clamp_envelope`,
    which every training counterfactual passes through."""
    if b < 1 or dkappa_max < 0 or daccel_max < 0:
        raise ValueError(f"bad sampler args b={b}, dkappa_max={dkappa_max}, "
                         f"daccel_max={daccel_max}")
    dk = (torch.rand(b, generator=gen) * 2.0 - 1.0) * dkappa_max
    da = (torch.rand(b, generator=gen) * 2.0 - 1.0) * daccel_max
    return dk, da


def apply_random_counterfactual(aw2: Tensor, fa2: Tensor, dk: Tensor,
                                da: Tensor, wheelbase: float = WHEELBASE
                                ) -> tuple[Tensor, Tensor]:
    """Per-window (Δκ, Δa) applied like ``stage_a_probes.apply_counterfactual``
    — Δκ IN KAPPA SPACE through the corpus steer encoding, only the LAST
    window action + all futures perturbed (history conditions on the
    observed past) — but with BOTH channels moved at once and per-window
    deltas ``dk``/``da`` of shape [B]."""
    if dk.shape != da.shape or dk.ndim != 1 or dk.shape[0] != aw2.shape[0]:
        raise ValueError(f"dk/da must be [B={aw2.shape[0]}], got "
                         f"{tuple(dk.shape)} / {tuple(da.shape)}")
    aw = aw2.clone()
    fa = fa2.clone()
    aw[:, -1, 0] = steer_of_kappa(
        kappa_of_steer(aw2[:, -1, 0], wheelbase) + dk, wheelbase)
    aw[:, -1, 1] = aw2[:, -1, 1] + da
    fa[:, :, 0] = steer_of_kappa(
        kappa_of_steer(fa2[:, :, 0], wheelbase) + dk[:, None], wheelbase)
    fa[:, :, 1] = fa2[:, :, 1] + da[:, None]
    return aw, fa


def clamp_envelope(aw2: Tensor, fa2: Tensor, *,
                   kappa_max: float = KAPPA_MAX, a_max: float = A_MAX,
                   wheelbase: float = WHEELBASE) -> tuple[Tensor, Tensor]:
    """Clamp the PERTURBED slots (last window action + futures) to the
    physical envelope |κ| <= kappa_max, |a| <= a_max (the W4 bounds the
    unicycle fan is feasible under). κ is clamped in KAPPA SPACE through the
    steer encoding. Earlier window actions are NOT touched — they are the
    observed history, and clamping them would change the conditioning.
    The analytic L_ctrl target is computed on the SAME clamped actions, so
    target and roll always see identical controls."""
    aw = aw2.clone()
    fa = fa2.clone()
    aw[:, -1, 0] = steer_of_kappa(
        kappa_of_steer(aw2[:, -1, 0], wheelbase).clamp(-kappa_max, kappa_max),
        wheelbase)
    aw[:, -1, 1] = aw2[:, -1, 1].clamp(-a_max, a_max)
    fa[:, :, 0] = steer_of_kappa(
        kappa_of_steer(fa2[:, :, 0], wheelbase).clamp(-kappa_max, kappa_max),
        wheelbase)
    fa[:, :, 1] = fa2[:, :, 1].clamp(-a_max, a_max)
    return aw, fa


def build_cf_actions(aw2: Tensor, fa2: Tensor, arm: str, *,
                     dkappa: float = DKAPPA_DEFAULT,
                     daccel: float = DACCEL_DEFAULT,
                     dk: Tensor | None = None, da: Tensor | None = None
                     ) -> tuple[Tensor, Tensor]:
    """One TRAINING counterfactual arm: the stage_a_probes builder for the
    four named channels, :func:`apply_random_counterfactual` for ``random``
    — then the envelope clamp for every arm. (The PROBE keeps the W3
    convention — unclamped named deltas — so the gate stays byte-comparable
    with w3_gate.json; the clamp is a TRAINING in-distribution guard.)"""
    if arm == "random":
        if dk is None or da is None:
            raise ValueError("random arm needs dk/da draws "
                             "(sample_random_deltas)")
        aw, fa = apply_random_counterfactual(aw2, fa2, dk, da)
    elif arm in GATED_CHANNELS:
        aw, fa = apply_counterfactual(aw2, fa2, arm, dkappa=dkappa,
                                      daccel=daccel)
    else:
        raise ValueError(f"unknown training arm {arm!r}; expected one of "
                         f"{TRAIN_ARMS}")
    return clamp_envelope(aw, fa)


# ============================================================================
# the batch-local action subspace (pure, CPU-tested)
# ============================================================================
def action_subspace_basis(deltas: Tensor, n_dims: int = N_BASIS_DEFAULT
                          ) -> Tensor:
    """Top-``n_dims`` UNCENTERED PCA basis [m, S] of stacked counterfactual
    latent deltas [N, S] (m = min(n_dims, N, S)); detached float32 SVD.
    Uncentered for the same stated reason as W3's P6 (stage_a_probes): a
    latent delta has a meaningful zero. ⚠️ Batch-local: an APPROXIMATION of
    the corpus-level 3-dim subspace W3 measured (module docstring)."""
    if deltas.ndim != 2:
        raise ValueError(f"deltas must be [N, S], got {tuple(deltas.shape)}")
    d = deltas.detach().float()
    if not torch.isfinite(d).all():
        raise ValueError("non-finite latent deltas — cannot build a basis")
    m = min(int(n_dims), d.shape[0], d.shape[1])
    if m < 1:
        raise ValueError(f"empty basis: n_dims={n_dims}, deltas "
                         f"{tuple(d.shape)}")
    _u, _s, vh = torch.linalg.svd(d, full_matrices=False)
    return vh[:m]                                            # [m, S]


def complement_residual(x: Tensor, basis: Tensor) -> Tensor:
    """Residual of ``x`` [..., S] OUTSIDE the row-space of ``basis`` [m, S]
    (rows orthonormal — SVD right-singular vectors):
    ``x - (x @ basisᵀ) @ basis``. A vector inside the subspace -> 0."""
    if basis.ndim != 2 or x.shape[-1] != basis.shape[-1]:
        raise ValueError(f"basis [m, S] must match x [..., S]: got "
                         f"{tuple(basis.shape)} vs {tuple(x.shape)}")
    return x - (x @ basis.T) @ basis


# ============================================================================
# the stage-A losses (predictor grads; CPU-tested on a mock predictor)
# ============================================================================
def stage_a_losses(predictor, step_readout, states: Tensor, aw2: Tensor,
                   fa2: Tensor, v0: Tensor, gt_wp: Tensor, z_true_k: Tensor,
                   k: int, *, dkappa: float = DKAPPA_DEFAULT,
                   daccel: float = DACCEL_DEFAULT,
                   rand_dk: Tensor | None = None,
                   rand_da: Tensor | None = None,
                   w_ctrl: float = 1.0, w_fact: float = 1.0,
                   w_scene: float = 0.3, n_basis: int = N_BASIS_DEFAULT,
                   ctrl_form: str = "response") -> dict:
    """One batch of the three stage-A losses (module docstring).

    ``states`` [B, W, S] (DETACHED — encoder frozen), ``aw2``/``fa2`` the
    2-channel recorded actions, ``v0`` [B], ``gt_wp`` [B, k, 2] true future
    waypoints (``gt_ego_waypoints``), ``z_true_k`` [B, S] the ENCODED true
    future frame at t+k (detached). Rolls 1 factual + ``len(TRAIN_ARMS)``
    counterfactual arms through ``rollout_transitions`` and decodes with the
    frozen ``step_readout`` (grads flow THROUGH it into the predictor; its
    params carry requires_grad False). All losses in float32."""
    from train_p8_occupancy import lift_actions3            # canary lift seam
    from tanitad.models.metric_dynamics import (decode_transitions,
                                                rollout_transitions)
    if ctrl_form not in ("response", "absolute"):
        raise ValueError(f"ctrl_form must be response|absolute, "
                         f"got {ctrl_form!r}")
    if gt_wp.shape[1] != k:
        raise ValueError(f"gt_wp horizon {gt_wp.shape[1]} != k={k}")
    # ---- factual roll (the anchor) -----------------------------------------
    aw3, fa3 = lift_actions3(aw2, fa2, v0)
    trans_f = rollout_transitions(predictor, states, aw3, fa3, k)
    wp_f, _ = decode_transitions(step_readout, trans_f, k)
    wp_f = wp_f.float()
    z_f = trans_f[k - 1][1].float()                          # ẑ_{t+k} factual
    l_fact = (wp_f - gt_wp.float()).abs().mean()
    with torch.no_grad():
        an_f = analytic_endpoints(aw2, fa2, v0, k).float()
    # ---- counterfactual arms (the gain repair) -----------------------------
    l_ctrl_arms: dict[str, Tensor] = {}
    z_deltas: list[Tensor] = []
    for arm in TRAIN_ARMS:
        aw_c, fa_c = build_cf_actions(aw2, fa2, arm, dkappa=dkappa,
                                      daccel=daccel, dk=rand_dk, da=rand_da)
        aw3c, fa3c = lift_actions3(aw_c, fa_c, v0)
        trans_c = rollout_transitions(predictor, states, aw3c, fa3c, k)
        wp_c, _ = decode_transitions(step_readout, trans_c, k)
        wp_c = wp_c.float()
        z_deltas.append(trans_c[k - 1][1].float() - z_f)
        with torch.no_grad():
            an_c = analytic_endpoints(aw_c, fa_c, v0, k).float()
        if ctrl_form == "response":
            l_ctrl_arms[arm] = ((wp_c - wp_f) - (an_c - an_f)).abs().mean()
        else:                                                # absolute
            l_ctrl_arms[arm] = (wp_c - an_c).abs().mean()
    l_ctrl = torch.stack(list(l_ctrl_arms.values())).mean()
    # ---- scene stability on the complement of the action subspace ----------
    dz = torch.cat(z_deltas, dim=0)                          # [B*5, S]
    basis = action_subspace_basis(dz, n_basis)               # detached
    l_scene_cf = complement_residual(dz, basis).pow(2).mean()
    l_scene_true = complement_residual(z_f - z_true_k.float(),
                                       basis).pow(2).mean()
    l_scene = l_scene_cf + l_scene_true
    loss = w_ctrl * l_ctrl + w_fact * l_fact + w_scene * l_scene
    return {"loss": loss, "l_ctrl": l_ctrl, "l_fact": l_fact,
            "l_scene": l_scene, "l_scene_cf": l_scene_cf,
            "l_scene_true": l_scene_true,
            "l_ctrl_arms": {a_: v.detach() for a_, v in l_ctrl_arms.items()},
            "factual_ade": (wp_f.detach() - gt_wp.float()).norm(dim=-1)
                           .mean(),
            "basis_dims": int(basis.shape[0]), "ctrl_form": ctrl_form}


# ============================================================================
# frozen-proof helpers (pure, CPU-tested)
# ============================================================================
def selective_md5(module, *, include_prefix: str | None = None,
                  exclude_prefix: str | None = None) -> str:
    """md5 over a name-filtered subset of ``module``'s parameters — the
    ``module_md5`` hash restricted to names under ``include_prefix`` and/or
    outside ``exclude_prefix`` (sorted-name order, same byte recipe)."""
    import hashlib
    h = hashlib.md5()
    for n_, p in sorted(module.named_parameters()):
        if include_prefix is not None and not n_.startswith(include_prefix):
            continue
        if exclude_prefix is not None and n_.startswith(exclude_prefix):
            continue
        h.update(n_.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def frozen_proof_md5(world, grounding) -> str:
    """One hash over EVERYTHING that must not move: the world minus its
    predictor, plus the whole grounding. Identical before/after or the run
    is invalid."""
    import hashlib
    h = hashlib.md5()
    h.update(selective_md5(world, exclude_prefix="predictor.").encode())
    h.update(module_md5(grounding).encode())
    return h.hexdigest()


def set_predictor_only_trainable(world, grounding) -> tuple[int, int]:
    """requires_grad True EXACTLY on ``world.predictor`` params; everything
    else in world + grounding False. Returns (n_trainable, n_frozen) param
    counts (elements)."""
    n_train = n_frozen = 0
    for p in grounding.parameters():
        p.requires_grad_(False)
        n_frozen += p.numel()
    for n_, p in world.named_parameters():
        t = n_.startswith("predictor.")
        p.requires_grad_(t)
        if t:
            n_train += p.numel()
        else:
            n_frozen += p.numel()
    if n_train == 0:
        raise RuntimeError("no predictor.* parameters found — wrong model?")
    return n_train, n_frozen


# ============================================================================
# the pre-registered gate (pure, every branch CPU-tested)
# ============================================================================
def stage_a_gate_dict(post: dict, pre: dict, *,
                      sign_gate: float = GATE_SIGN,
                      gain_band: tuple[float, float] = GATE_GAIN,
                      p6_dim_gate: int = GATE_P6_DIMS,
                      noharm_factor: float = NOHARM_FACTOR) -> dict:
    """Assemble the pre-registered stage-A verdicts (module docstring;
    registry §1.14 targets verbatim). ``post``/``pre`` each carry
    ``per_channel`` (channel -> channel_stats dict), ``p6``
    (pca_subspace_stats dict) and ``factual_ade`` (float). Missing stats
    yield ``pass: None`` (not computable) — never a fake verdict.
    Longitudinal GAIN is reported, NOT gated (the registry gates lon sign
    only). Both outcomes are bound in advance and written verbatim."""
    def _st(block: dict, c: str, key: str):
        return (block.get("per_channel") or {}).get(c, {}).get(key)

    checks: dict[str, dict] = {}
    passes: list = []
    for c in LAT_CHANNELS:
        gm = _st(post, c, "gain_median")
        gp = None if gm is None else bool(gain_band[0] <= gm <= gain_band[1])
        checks[f"lat_gain_{c}"] = {
            "rule": f"median lateral gain in [{gain_band[0]}, {gain_band[1]}]"
                    " vs unicycle-analytic", "value": gm, "pass": gp}
        passes.append(gp)
        sr = _st(post, c, "sign_rate")
        sp = None if sr is None else bool(sr >= sign_gate)
        checks[f"lat_sign_stays_{c}"] = {
            "rule": f"lateral sign rate STAYS >= {sign_gate}",
            "value": sr, "pre_value": _st(pre, c, "sign_rate"), "pass": sp}
        passes.append(sp)
    for c in LON_CHANNELS:
        sr = _st(post, c, "sign_rate")
        sp = None if sr is None else bool(sr >= sign_gate)
        checks[f"lon_sign_{c}"] = {
            "rule": f"longitudinal sign rate >= {sign_gate}",
            "value": sr, "pre_value": _st(pre, c, "sign_rate"), "pass": sp}
        passes.append(sp)
        checks[f"lon_gain_{c}_reported_not_gated"] = {
            "value": _st(post, c, "gain_median"),
            "pre_value": _st(pre, c, "gain_median"), "pass": "not gated"}
    p6 = post.get("p6") or {}
    p6p = (bool(p6["dims_for_var_target"] <= p6_dim_gate)
           if p6.get("computable") else None)
    checks["p6_dims"] = {
        "rule": f"dims capturing 80 % of counterfactual latent-delta "
                f"variance <= {p6_dim_gate} (preserve the factorisation)",
        "value": p6.get("dims_for_var_target"),
        "pre_value": (pre.get("p6") or {}).get("dims_for_var_target"),
        "pass": p6p}
    passes.append(p6p)
    ade_pre, ade_post = pre.get("factual_ade"), post.get("factual_ade")
    finite = (ade_pre is not None and ade_post is not None
              and bool(np.isfinite(ade_pre)) and bool(np.isfinite(ade_post)))
    nh = bool(ade_post <= noharm_factor * ade_pre) if finite else None
    checks["no_harm_factual_ade"] = {
        "rule": f"factual-roll ADE on the val grid NOT worse than "
                f"+{round((noharm_factor - 1) * 100)} % vs the pre-training "
                f"predictor (both measured in this run)",
        "ade_pre_m": ade_pre, "ade_post_m": ade_post,
        "cap_m": (None if ade_pre is None
                  else round(noharm_factor * ade_pre, 6)),
        "pass": nh}
    passes.append(nh)
    return {"checks": checks, "PASS": _agg(passes),
            "outcomes_bound_in_advance": {"PASS": OUTCOME_PASS,
                                          "FAIL": OUTCOME_FAIL}}


# ============================================================================
# the W3 probe pack, importable form (POD-SIDE — reuses probe_batch verbatim)
# ============================================================================
@torch.no_grad()
def run_probe(world, grounding, batches, k: int, *, amp_on: bool,
              dkappa: float = DKAPPA_DEFAULT, daccel: float = DACCEL_DEFAULT
              ) -> dict:
    """The W3 probe pack over an iterable of device-ready batches —
    ``stage_a_probes.probe_batch`` byte-reused (same six arms, same decode,
    UNCLAMPED named counterfactuals: the W3 convention, so numbers are
    directly comparable with w3_gate.json) — PLUS the stage-A no-harm
    instrument: per-window factual-roll ADE vs the true future waypoints
    (``gt_ego_waypoints``, the decode's own geometry). Returns per_channel /
    p6 / hold / factual_ade + banked per-window arrays."""
    from tanitad.models.metric_dynamics import gt_ego_waypoints
    d_wm = {c: [] for c in GATED_CHANNELS}
    d_an = {c: [] for c in GATED_CHANNELS}
    z_deltas = {c: [] for c in CF_CHANNELS}
    hold_norm: list[float] = []
    fade: list[np.ndarray] = []
    eids: list[np.ndarray] = []
    n = 0
    for b in batches:
        res = probe_batch(world, grounding, b, k, amp_on=amp_on,
                          dkappa=dkappa, daccel=daccel)
        aw2 = b["actions"].float().cpu()
        fa2 = b["future_actions"].float().cpu()
        v0 = b["pose_last"][:, 3].float().cpu()
        an_fact = analytic_endpoints(aw2, fa2, v0, k)
        wp_fact = res["wp"]["factual"]
        gt = gt_ego_waypoints(b["pose_last"].float().cpu(),
                              b["future_poses"].float().cpu(),
                              tuple(range(1, k + 1)))
        fade.append((wp_fact - gt).norm(dim=-1).mean(dim=-1).numpy())
        for c in GATED_CHANNELS:
            _sign, axis = CHANNEL_SIGN_AXIS[c]
            a2c, f2c = apply_counterfactual(aw2, fa2, c, dkappa=dkappa,
                                            daccel=daccel)
            an_cf = analytic_endpoints(a2c, f2c, v0, k)
            d_an[c].append((an_cf[:, k - 1, axis]
                            - an_fact[:, k - 1, axis]).numpy())
            d_wm[c].append((res["wp"][c][:, k - 1, axis]
                            - wp_fact[:, k - 1, axis]).numpy())
        hold_norm.extend((res["wp"]["hold"][:, k - 1]
                          - wp_fact[:, k - 1]).norm(dim=-1).tolist())
        for c in CF_CHANNELS:
            z_deltas[c].append(res["z_delta"][c].numpy())
        eid = b.get("episode_id")
        if torch.is_tensor(eid):
            eids.append(eid.cpu().numpy().astype(np.int64))
        n += aw2.shape[0]
    per_channel = {}
    for c in GATED_CHANNELS:
        sign, _axis = CHANNEL_SIGN_AXIS[c]
        per_channel[c] = channel_stats(np.concatenate(d_wm[c]),
                                       np.concatenate(d_an[c]), sign)
        per_channel[c]["kind"] = ("steer" if c in LAT_CHANNELS else "accel")
    p6 = pca_subspace_stats({c: np.concatenate(z_deltas[c])
                             for c in CF_CHANNELS})
    p6["k"] = k
    fade_all = np.concatenate(fade)
    return {
        "per_channel": per_channel, "p6": p6,
        "hold": {"n_grid": len(hold_norm),
                 "endpoint_delta_median_m": float(np.median(hold_norm)),
                 "endpoint_delta_p90_m": float(np.percentile(hold_norm,
                                                             90))},
        "factual_ade": float(fade_all.mean()),
        "n_windows": n,
        "bank": {"factual_ade": fade_all.astype(np.float32),
                 "episode_uid": (np.concatenate(eids) if eids
                                 else np.zeros(0, dtype=np.int64)),
                 **{f"d_wm_{c}": np.concatenate(d_wm[c]).astype(np.float32)
                    for c in GATED_CHANNELS},
                 **{f"d_an_{c}": np.concatenate(d_an[c]).astype(np.float32)
                    for c in GATED_CHANNELS}},
    }


def _probe_brief(pr: dict) -> dict:
    """The train_vs_probe log row payload: per-channel sign/gain + families."""
    ch = {c: {"sign_rate": pr["per_channel"][c].get("sign_rate"),
              "gain_median": pr["per_channel"][c].get("gain_median"),
              "n_admissible": pr["per_channel"][c].get("n_admissible")}
          for c in GATED_CHANNELS}
    lat_sr = [ch[c]["sign_rate"] for c in LAT_CHANNELS
              if ch[c]["sign_rate"] is not None]
    lon_sr = [ch[c]["sign_rate"] for c in LON_CHANNELS
              if ch[c]["sign_rate"] is not None]
    lat_gm = [ch[c]["gain_median"] for c in LAT_CHANNELS
              if ch[c]["gain_median"] is not None]
    lon_gm = [ch[c]["gain_median"] for c in LON_CHANNELS
              if ch[c]["gain_median"] is not None]
    return {"channels": ch,
            "lat_sign_min": min(lat_sr) if lat_sr else None,
            "lon_sign_min": min(lon_sr) if lon_sr else None,
            "lat_gain_median_mean": (float(np.mean(lat_gm)) if lat_gm
                                     else None),
            "lon_gain_median_mean": (float(np.mean(lon_gm)) if lon_gm
                                     else None),
            "p6_dims": pr["p6"].get("dims_for_var_target"),
            "factual_ade": pr["factual_ade"],
            "n_windows": pr["n_windows"]}


# ============================================================================
# main (POD-SIDE: GPU + v5f checkpoint + v2 train/val corpora)
# ============================================================================
def build_args(argv=None):
    ap = argparse.ArgumentParser("train_stage_a", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="v5f checkpoint (model+grounding; MODE A load — the "
                         "planner head is untouched and re-attached verbatim "
                         "in the saved repaired ckpt)")
    # corpus seams — the W4 trainer's arg surface (train + val)
    ap.add_argument("--v2-cache", required=True, nargs="+",
                    help="v2 compressed TRAIN split dir(s) — the canonical "
                         "physicalai-train-e438721ae894 build. Training "
                         "windows come ONLY from here.")
    ap.add_argument("--v2-val-cache", required=True, nargs="+",
                    help="v2 compressed VAL split dir(s) — probes/gate only, "
                         "NEVER trained on")
    ap.add_argument("--v2-lru", type=int, default=64)
    ap.add_argument("--v2-subframe", default=None, metavar="HxW")
    ap.add_argument("--require-parity", action="store_true")
    from tanitad.geometry import add_geometry_args
    add_geometry_args(ap)      # --frame-h/--frame-w/--frame-hfov/--projection
    # training
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    ap.add_argument("--bs", type=int, default=8,
                    help="batch size (6 grad-carrying k-step rolls per batch "
                         "— smaller than the W4 default on purpose)")
    ap.add_argument("--lr", type=float, default=LR_DEFAULT,
                    help="predictor lr — small; this is a repair, not a "
                         "retrain")
    ap.add_argument("--wd", type=float, default=0.0,
                    help="weight decay (default 0 — do not pull a pretrained "
                         "predictor toward zero)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--eps-per-batch", type=int, default=4,
                    help="episode-grouped sampling (the MooseFS I/O shape)")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    # stage-A knobs (defaults are the pre-registered ones)
    ap.add_argument("--k", type=int, default=K_DEFAULT,
                    help="roll length in 0.1 s ticks (10 = 1 s, the W3 "
                         "probe horizon)")
    ap.add_argument("--dkappa", type=float, default=DKAPPA_DEFAULT)
    ap.add_argument("--daccel", type=float, default=DACCEL_DEFAULT)
    ap.add_argument("--rand-dkappa-max", type=float,
                    default=RAND_DKAPPA_MAX_DEFAULT,
                    help="random-arm |Δκ| draw bound, 1/m (then envelope-"
                         "clamped)")
    ap.add_argument("--rand-daccel-max", type=float,
                    default=RAND_DACCEL_MAX_DEFAULT,
                    help="random-arm |Δa| draw bound, m/s²")
    ap.add_argument("--w-ctrl", type=float, default=1.0)
    ap.add_argument("--w-fact", type=float, default=1.0)
    ap.add_argument("--w-scene", type=float, default=0.3)
    ap.add_argument("--n-basis", type=int, default=N_BASIS_DEFAULT,
                    help="batch-local action-subspace PCA dims (W3 measured "
                         "3; 8 leaves slack)")
    ap.add_argument("--ctrl-form", choices=("response", "absolute"),
                    default="response",
                    help="'response' (default): L1 on (decoded_cf - "
                         "decoded_fact) vs (analytic_cf - analytic_fact) — "
                         "the exact W3 gain quantity; 'absolute': the "
                         "literal pre-design ||decode(roll(z,a_cf)) - "
                         "Unicycle(a_cf,v0)||")
    # monitor + gate grid (eval defaults — the 881 grid)
    ap.add_argument("--probe-every", type=int, default=PROBE_EVERY_DEFAULT)
    ap.add_argument("--probe-windows", type=int,
                    default=PROBE_WINDOWS_DEFAULT,
                    help="fixed val windows for the cheap in-training probe "
                         "(the first N of the canonical grid)")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=16)
    return ap.parse_args(argv)


def _grid_batches(ds_val, sel, batch, device):
    from torch.utils.data import default_collate

    from train_flagship_v4 import _to_device
    for b0 in range(0, len(sel), batch):
        yield _to_device(default_collate([ds_val[i]
                                          for i in sel[b0:b0 + batch]]),
                         device)


def main(argv=None) -> int:
    a = build_args(argv)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    rng = random.Random(a.seed)
    gen = torch.Generator().manual_seed(a.seed + 1)
    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[stage-a] WARNING: cuda unavailable, falling back to cpu",
              flush=True)
        device = "cpu"
    amp_on = (device == "cuda") and not a.no_amp
    os.makedirs(a.out, exist_ok=True)

    # ---- the val corpus is for probing, never training ---------------------
    overlap = set(map(os.path.abspath, a.v2_cache)) & \
        set(map(os.path.abspath, a.v2_val_cache))
    if overlap:
        raise SystemExit(f"[stage-a] ⛔ --v2-cache and --v2-val-cache share "
                         f"dirs {sorted(overlap)} — val windows in training "
                         f"are not allowed")

    from torch.utils.data import default_collate

    from eval_flagship_v4 import (_eval_cfg, _plan, build_v2_val_episodes,
                                  load_v1_from_ck, resolve_eval_frames)
    from train_flagship4b import FlagshipWindowDataset
    from train_flagship_v4 import _to_device
    from tanitad.models.metric_dynamics import gt_ego_waypoints

    # ---- geometry FIRST (the W4/eval seam) ---------------------------------
    cfg = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(a, cfg,
                                                   label="train_stage_a")
    plan = _plan(cfg)
    if plan.max_horizon < a.k:
        raise SystemExit(f"[stage-a] --k {a.k} > plan.max_horizon "
                         f"{plan.max_horizon} — future actions/frames cannot "
                         f"cover the roll")

    # ---- load MODE A; unfreeze the predictor ONLY --------------------------
    print(f"[stage-a] loading checkpoint {a.ckpt} ...", flush=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    world, grounding, base_step = load_v1_from_ck(ck, device,
                                                  frame=model_frame)
    del ck                                    # reloaded at final save (merge)
    n_train, n_frozen = set_predictor_only_trainable(world, grounding)
    world.eval()
    world.predictor.train()
    frozen_before = frozen_proof_md5(world, grounding)
    pred_before = selective_md5(world, include_prefix="predictor.")
    print(f"[stage-a] base step {base_step} · trainable predictor "
          f"{n_train/1e6:.2f} M · frozen {n_frozen/1e6:.2f} M · frozen md5 "
          f"{frozen_before[:12]} · predictor md5 {pred_before[:12]}",
          flush=True)

    # ---- data --------------------------------------------------------------
    train_eps, _train_prov = build_train_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    val_eps, _val_prov = build_v2_val_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    ds_train = FlagshipWindowDataset(train_eps, window=cfg.predictor.window,
                                     max_horizon=plan.max_horizon,
                                     maneuver_h=plan.maneuver_h,
                                     channels=cfg.encoder.in_channels)
    ds_val = FlagshipWindowDataset(val_eps, window=cfg.predictor.window,
                                   max_horizon=plan.max_horizon,
                                   maneuver_h=plan.maneuver_h,
                                   channels=cfg.encoder.in_channels)
    print(f"[stage-a] train {len(train_eps)} eps / {len(ds_train)} windows; "
          f"val {len(val_eps)} eps / {len(ds_val)} windows", flush=True)
    sample = make_sampler(ds_train, a.eps_per_batch, rng)
    grid = [i for i, (e, t) in enumerate(ds_val.index)
            if e < a.episodes and t % a.stride == 0]
    if not grid:
        raise SystemExit("[stage-a] gate grid selected 0 windows — check "
                         "--episodes/--stride against the val cache")
    mon_idx = grid[:a.probe_windows]          # FIXED: first N of the grid
    mon_cpu = [default_collate([ds_val[i]
                                for i in mon_idx[b0:b0 + a.eval_batch]])
               for b0 in range(0, len(mon_idx), a.eval_batch)]
    print(f"[stage-a] gate grid {len(grid)} windows; monitor fixed on the "
          f"first {len(mon_idx)}", flush=True)

    # ---- PRE probe: baseline + the no-harm reference (same run, same grid) -
    world.predictor.eval()
    t0 = time.time()
    probe_pre = run_probe(world, grounding,
                          _grid_batches(ds_val, grid, a.eval_batch, device),
                          a.k, amp_on=amp_on, dkappa=a.dkappa,
                          daccel=a.daccel)
    print(f"[stage-a] PRE probe ({time.time() - t0:.0f} s): "
          f"{json.dumps(_probe_brief(probe_pre))}", flush=True)
    world.predictor.train()

    # ---- optimiser over predictor params ONLY ------------------------------
    pred_params = [p for n_, p in world.named_parameters()
                   if n_.startswith("predictor.")]
    opt = torch.optim.AdamW(pred_params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)

    log_path = os.path.join(a.out, "train_log.jsonl")
    fh = open(log_path, "a")
    fh.write(json.dumps({
        "run": "stage-a-predictor-posttrain", "args": vars(a),
        "base_ckpt": a.ckpt, "base_step": base_step,
        "n_trainable": n_train, "n_frozen": n_frozen,
        "frozen_md5": frozen_before, "predictor_md5_before": pred_before,
        "probe_pre": _probe_brief(probe_pre),
        "tier": "T1-diagnostic",
        "_evidence_class": "MEASURED (ours; artifact = this log)"}) + "\n")
    fh.flush()

    history: list[dict] = []
    acc_keys = ("loss", "l_ctrl", "l_fact", "l_scene", "factual_ade")
    acc = {k_: 0.0 for k_ in acc_keys} | {"n": 0}
    recent = {k_: 0.0 for k_ in acc_keys} | {"n": 0}   # since last probe
    steps_g = tuple(range(1, a.k + 1))
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = sample(a.bs)
        b = _to_device(default_collate([ds_train[i] for i in idx]), device)
        aw2 = b["actions"].float()
        if aw2.shape[-1] != 2:
            raise SystemExit(f"[stage-a] batch actions have {aw2.shape[-1]} "
                             "channels, expected the 2-channel (steer, "
                             "accel) corpus format")
        fa2 = b["future_actions"].float()
        v0 = b["pose_last"][:, 3].float()
        gt_wp = gt_ego_waypoints(b["pose_last"].float(),
                                 b["future_poses"].float(), steps_g)
        dk, da = sample_random_deltas(aw2.shape[0], gen, a.rand_dkappa_max,
                                      a.rand_daccel_max)
        dk, da = dk.to(device), da.to(device)
        dev_type = "cuda" if device == "cuda" else "cpu"
        with torch.autocast(dev_type, dtype=torch.bfloat16,
                            enabled=amp_on and dev_type == "cuda"):
            with torch.no_grad():             # encoder frozen — no graph
                states = world.encode_window(b["frames"]).detach()
                z_true = world.encode(
                    b["future_frames"][:, a.k - 1]).detach()
            L = stage_a_losses(world.predictor, grounding.step["op"], states,
                               aw2, fa2, v0, gt_wp, z_true, a.k,
                               dkappa=a.dkappa, daccel=a.daccel,
                               rand_dk=dk, rand_da=da, w_ctrl=a.w_ctrl,
                               w_fact=a.w_fact, w_scene=a.w_scene,
                               n_basis=a.n_basis, ctrl_form=a.ctrl_form)
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        gnorm = torch.nn.utils.clip_grad_norm_(pred_params, a.clip)
        opt.step()
        sched.step()

        bs = aw2.shape[0]
        for k_ in acc_keys:
            acc[k_] += float(L[k_].detach()) * bs
            recent[k_] += float(L[k_].detach()) * bs
        acc["n"] += bs
        recent["n"] += bs

        if step % a.log_every == 0:
            rec = {"step": step,
                   "loss": round(float(L["loss"].detach()), 5),
                   "l_ctrl": round(float(L["l_ctrl"].detach()), 5),
                   "l_fact": round(float(L["l_fact"].detach()), 5),
                   "l_scene": round(float(L["l_scene"].detach()), 6),
                   "factual_ade": round(float(L["factual_ade"]), 4),
                   "gnorm": round(float(gnorm), 3),
                   "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{step}] {rec}", flush=True)

        if step % a.probe_every == 0:
            world.predictor.eval()
            pr = run_probe(world, grounding,
                           (_to_device(mb, device) for mb in mon_cpu),
                           a.k, amp_on=amp_on, dkappa=a.dkappa,
                           daccel=a.daccel)
            world.predictor.train()
            n_ = max(recent["n"], 1)
            row = {"step": step,
                   "train_recent": {k_: round(v / n_, 5)
                                    for k_, v in recent.items()
                                    if k_ != "n"},
                   "probe": _probe_brief(pr)}
            recent = {k_: 0.0 for k_ in acc_keys} | {"n": 0}
            fh.write(json.dumps({"train_vs_probe": row}) + "\n")
            fh.flush()
            print(f"[stage-a probe @{step}] {json.dumps(row['probe'])}",
                  flush=True)

        if step % a.save_every == 0:
            n_ = max(acc["n"], 1)
            row = {"step": step,
                   **{k_: round(v / n_, 5) for k_, v in acc.items()
                      if k_ != "n"},
                   "elapsed_s": round(time.time() - t0, 1)}
            history.append(row)
            acc = {k_: 0.0 for k_ in acc_keys} | {"n": 0}
            with open(os.path.join(a.out, "metrics.json"), "w") as mf:
                json.dump({"history": history, "args": vars(a),
                           "base_step": base_step,
                           "_read": "running TRAIN means per save window; "
                                    "the gate numbers are the full W3 probe "
                                    "in stage_a_gate.json",
                           "_evidence_class": "MEASURED (ours)"}, mf,
                          indent=1)
            torch.save({"predictor": {k2: v.detach().cpu() for k2, v in
                                      world.predictor.state_dict().items()},
                        "step": step, "args": vars(a), "base_ckpt": a.ckpt,
                        "base_step": base_step},
                       os.path.join(a.out, "predictor_stage_a.pt"))
            fh.write(json.dumps({"per_save": row}) + "\n")
            fh.flush()
            print(f"[stage-a @{step}] {row}", flush=True)

    # ---- frozen proof -------------------------------------------------------
    frozen_after = frozen_proof_md5(world, grounding)
    pred_after = selective_md5(world, include_prefix="predictor.")
    if pred_after == pred_before:
        raise SystemExit("⛔ PREDICTOR UNCHANGED AFTER TRAINING — the run "
                         "was a no-op (optimizer wired wrong?); refusing to "
                         "emit a gate")

    # ---- POST probe + the pre-registered gate ------------------------------
    world.predictor.eval()
    probe_post = run_probe(world, grounding,
                           _grid_batches(ds_val, grid, a.eval_batch, device),
                           a.k, amp_on=amp_on, dkappa=a.dkappa,
                           daccel=a.daccel)
    gate = stage_a_gate_dict(probe_post, probe_pre)

    # ---- bank per-window arrays (pod-side cluster bootstrap input) ---------
    np.savez_compressed(
        os.path.join(a.out, "stage_a_windows.npz"),
        **{f"pre_{k_}": v for k_, v in probe_pre["bank"].items()},
        **{f"post_{k_}": v for k_, v in probe_post["bank"].items()})

    # ---- save the REPAIRED full checkpoint (MODE A + MODE B loadable) ------
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ck["model"] = {k2: v.detach().cpu()
                   for k2, v in world.state_dict().items()}
    ck["stage_a"] = {"base_ckpt": a.ckpt, "base_step": base_step,
                     "steps": a.steps, "trained": "predictor only",
                     "args": vars(a)}
    repaired_path = os.path.join(a.out, "ckpt_stage_a.pt")
    torch.save(ck, repaired_path)
    del ck
    print(f"[stage-a] repaired checkpoint (original head/grounding + "
          f"post-trained predictor) -> {repaired_path}", flush=True)

    def _strip_bank(pr: dict) -> dict:
        return {k_: v for k_, v in pr.items() if k_ != "bank"}

    summary = {
        "item": ("V18 E3.4 stage-A predictor-only post-training — the W3 "
                 "gain repair (registry §1.14 W7 block; losses L_ctrl + "
                 "L_factual + L_scene, module docstring)"),
        "tier": "T1-diagnostic — counterfactual-action controllability of "
                "the WM roll, NEVER driving performance (EVAL_DOCTRINE.md)",
        "gate": gate,
        "probe_pre": _strip_bank(probe_pre),
        "probe_post": _strip_bank(probe_post),
        "probe_convention": ("stage_a_probes.probe_batch byte-reused — six "
                            "arms, unclamped named counterfactuals (dkappa "
                            f"{a.dkappa}, daccel {a.daccel}), the W3 pack; "
                            "directly comparable with w3_gate.json"),
        "training": {"steps": a.steps, "bs": a.bs, "lr": a.lr, "wd": a.wd,
                     "k": a.k, "ctrl_form": a.ctrl_form,
                     "weights": {"w_ctrl": a.w_ctrl, "w_fact": a.w_fact,
                                 "w_scene": a.w_scene},
                     "n_basis": a.n_basis,
                     "arms": list(TRAIN_ARMS),
                     "envelope": {"a_max_mps2": A_MAX,
                                  "kappa_max_1pm": KAPPA_MAX},
                     "random_draws": {"dkappa_max": a.rand_dkappa_max,
                                      "daccel_max": a.rand_daccel_max},
                     "n_trainable": n_train, "n_frozen": n_frozen,
                     "train_corpus_dirs": list(a.v2_cache),
                     "scene_estimator_note":
                         "batch-local top-{} uncentered PCA — an "
                         "APPROXIMATION of W3's corpus-level 3-dim action "
                         "subspace".format(a.n_basis)},
        "frozen_proof": {"frozen_md5_before": frozen_before,
                         "frozen_md5_after": frozen_after,
                         "identical": frozen_before == frozen_after,
                         "predictor_md5_before": pred_before,
                         "predictor_md5_after": pred_after,
                         "predictor_changed": pred_before != pred_after},
        "repaired_ckpt": repaired_path,
        "base_ckpt": a.ckpt, "base_step": base_step,
        "grid": {"episodes": a.episodes, "stride": a.stride,
                 "n_windows": len(grid)},
        "wall_s": round(time.time() - t0, 1),
        "_estimator_note": ("point estimates over the corpus grid; "
                            "per-window arrays banked to "
                            "stage_a_windows.npz — the DECISION-grade "
                            "interval for any registry claim is the "
                            "episode-cluster bootstrap (taniteval/ci.py) "
                            "over the 40 val episodes; run it pod-side "
                            "before publishing"),
        "_evidence_class": "MEASURED (ours; artifact = this JSON + "
                           "stage_a_windows.npz + train_log.jsonl)",
    }
    with open(os.path.join(a.out, "stage_a_gate.json"), "w") as gf:
        json.dump(summary, gf, indent=1)
    fh.write(json.dumps({"summary_gate": gate}) + "\n")
    fh.close()
    print(f"\n[STAGE-A SUMMARY] {json.dumps(summary, indent=1)}", flush=True)
    if not summary["frozen_proof"]["identical"]:
        raise SystemExit("⛔ FROZEN PARTS (encoder/readout/grounding/heads) "
                         "CHANGED DURING TRAINING — run invalid")
    verdict = ("PASS" if gate["PASS"] else
               "FAIL" if gate["PASS"] is not None else "NOT COMPUTABLE")
    outcome = (OUTCOME_PASS if gate["PASS"] else
               OUTCOME_FAIL if gate["PASS"] is not None else
               "NOT COMPUTABLE — fix the missing instrument before binding "
               "an outcome")
    print(f"[STAGE-A GATE] {verdict} — {outcome}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
