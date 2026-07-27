"""train_flagship_v4.py — the joint WM + diffusion-planner trainer (P4, §15/§16).

STATUS (2026-07-22): the CLI surface (§16), the loss ASSEMBLY (``v4_loss_step``)
and a CPU smoke are complete and tested. The full multi-day training LOOP over the
real epcaches (data pipeline, checkpointing, the canary_rollout eval, milestone
archiving) reuses ``train_flagship4b.py`` / ``train_flagship_v16.py`` machinery and
is the remaining P4 work — it is NOT launched here (this file builds and validates;
Sayed owns the go/no-go, §17). Nothing in this file starts a run.

What v4 is (§0): ONE world model carrying THREE planners, trained jointly. This
trainer wires the OPERATIVE planner (③, P1) onto the v1 trunk under:
  * the three-phase λ_plan curriculum (Phase A LP / B ramp / C joint), λ_plan a
    GRADIENT scale at the trunk→planner seam (O-20), applied inside the head;
  * the WM loss stack LIVE from step 0 (the half v1.6 deleted and lost the world
    model to — canary 0.452 → 1.1022);
  * the factorised LAT×LON×DIST selection CE and the dense-plan smoothness term;
  * the WM-integrity canary as a CONTROLLER on λ_plan, never a kill (§5.5).

The tactical instance (②, P5) and strategic planner (①, P6) are separate work
packages; their flags exist here (``--long-horizon-k``, ``--strategic``) so the
launch command is complete, but their loss terms land with P5/P6.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import refb_labels  # noqa: E402

from tanitad.data import parity  # noqa: E402
from tanitad.models.flagship_v4 import (FlagshipV4Head, V4Config,  # noqa: E402
                                        v4_config)
from tanitad.train import heldout_gate as _HG  # noqa: E402
from tanitad.train.flagship_losses import (LossWeights,  # noqa: E402
                                           build_grounding, flagship_loss,
                                           horizon_plan)
from tanitad.train.v4_curriculum import (IGNORE_INDEX,  # noqa: E402
                                         CanaryController, CurriculumPhases,
                                         factorised_ce, lambda_plan_at,
                                         plan_smoothness_loss,
                                         strategic_scalar_loss)

# The canonical parity contract — refused if anything re-selects episodes.
# ONE source of truth: tanitad/data/parity.py + the committed episode manifest.
PARITY_KEY = parity.PARITY_TRAIN_KEY
PARITY_SKIP_HASH = parity.PARITY_SKIP_HASH


# ============================================================================
# The core loss assembly — the tested unit of the trainer
# ============================================================================

@dataclass
class V4LossWeights:
    w_lat: float = 0.05
    w_lon: float = 0.05
    w_dist: float = 0.05
    w_jerk: float = 0.02
    w_curv: float = 0.01
    w_strat: float = 0.05          # aggregate weight on the strategic-scalar term


def v4_loss_step(world, grounding, head: FlagshipV4Head, batch: dict,
                 plan, cfg, step: int, phases: CurriculumPhases,
                 lw: V4LossWeights, lam_mode: str = "sched",
                 lam_mult: float = 1.0, device: str = "cpu",
                 goal_head=None) -> tuple:
    """One joint step's TOTAL loss = WM stack + planner + factorised + smoothness
    (+ the strategic goal-scalar regression when a ``goal_head`` is supplied).

    ``lam_mult`` is the canary controller's down-only multiplier on the scheduled
    λ_plan (§5.5). ``goal_head`` is the P6 strategic goal-scalar head
    (``models.strategic_goal.GoalScalarHead``); when ``None`` the step is
    byte-identical to before (the strategic term lands with P6's planner), so the
    existing smoke/tests are unaffected. Returns ``(total, log)``. Pure w.r.t. its
    inputs so it is unit-tested on the smoke config; the training LOOP that feeds
    it batches is the remaining P4 work.
    """
    # --- encode the window once; both stacks read the SAME states ---------------
    states = world.encode_window(batch["frames"])
    fut = world.encode_window(batch["future_frames"][:, plan.needed_fut])

    # --- (1) the WM stack — LIVE from step 0, Phase A included (the v1.6 lesson) -
    wm_total, wm_log, _ = flagship_loss(
        world, grounding, batch, states, fut, plan, cfg, weights=LossWeights(),
        sigreg_variant="full_relaxed", sigreg_free_dims=cfg.loss.sigreg.free_dims,
        pose_scale=10.0, fwd_step_weight=0.5, device=device)

    # --- (2) the planner, under the λ_plan gradient seam ------------------------
    lam = lam_mult * lambda_plan_at(step, phases, mode=lam_mode)
    v0 = batch["pose_last"][:, 3].float()                 # observed ego speed
    horizons = head.cfg.horizons
    traj_tgt = refb_labels.waypoint_targets(
        batch["pose_last"].float(),
        batch["future_poses"][:, :max(horizons)].float(), horizons)

    goal = _goal_inputs(head.cfg, batch, v0)
    out = head(states, v0, lambda_plan=lam, **goal)
    from tanitad.models.flagship_v15 import v15_losses
    plan_l = v15_losses(out, head.decoder.anchors, traj_tgt)

    # --- (3) factorised LAT×LON×DIST CE (masked; §6.2) --------------------------
    fac_loss = torch.zeros((), device=states.device)
    fac_log: dict = {}
    if head.cfg.factorised:
        b = states.shape[0]
        lat = batch.get("lat_target", torch.full((b,), IGNORE_INDEX))
        lon = batch.get("lon_target", torch.full((b,), IGNORE_INDEX))
        dist = batch.get("dist_target", torch.full((b,), IGNORE_INDEX))
        fac_loss, fac_log = factorised_ce(out, lat, lon, dist,
                                          lw.w_lat, lw.w_lon, lw.w_dist)

    # --- (4) plan smoothness on the DENSE emitted plan (§7) ---------------------
    sm_loss, sm_log = (plan_smoothness_loss(out["wp_seq"], lw.w_jerk, lw.w_curv)
                       if out["wp_seq"].shape[-2] >= 4
                       else (torch.zeros((), device=states.device), {}))

    # --- (5) strategic goal-scalar regression (§4.3/§7A.4; P6 head) -------------
    strat_loss = torch.zeros((), device=states.device)
    strat_log: dict = {}
    if goal_head is not None and "strat_scalars" in batch:
        # In the finished stack the head reads z_strat (P6); here it reads the
        # operative readout state so the minted labels can be proven trainable now.
        pred = goal_head(states[:, -1])
        strat_loss, strat_log = strategic_scalar_loss(
            pred, batch["strat_scalars"].to(states.device, dtype=states.dtype),
            batch["strat_scalar_mask"].to(states.device), weight=lw.w_strat)

    total = wm_total + plan_l["loss"] + fac_loss + sm_loss + strat_loss
    log = {"total": float(total.detach()), "lambda_plan": round(lam, 4),
           "wm": float(wm_total.detach()), "planner": float(plan_l["loss"].detach()),
           "plan_ade": float(plan_l["ade"].detach()),
           "oracle_ade": float(plan_l["oracle_ade"].detach()),
           # `grounding_losses` already emits this key G-PREFIXED
           # (metric_dynamics.py:389 -> `log[f"g_{lvl}_fwd_ade_m"]`), and
           # `flagship_loss` merges that dict verbatim (`**g_log`). So the key
           # actually present in `wm_log` is `g_op_fwd_ade_m`, NOT
           # `op_fwd_ade_m`. Filtering on the UNPREFIXED name matched nothing,
           # the comprehension always evaluated to `{}`, and every v4 arm's
           # train_log.jsonl was written WITHOUT the one metric that
           # `speed_benefit_recovered_frac` (the §7.5/P8 KILL secondary)
           # reduces -- so that secondary read NOT SUPPLIED and every v4 gate
           # rendered INCOMPLETE. (The row-writer below was already patched to
           # forward `g_op_fwd_ade_m`, but it was starved by THIS line, so the
           # earlier fix was silently inert: MEASURED on
           # flagship-v4.{1-10k,2-step4000} train logs, 0 occurrences each.)
           # Fix is LOG-ONLY: no loss term, no parity effect.
           #
           # ⭐ `g_op_mid_de_m` ADDED 2026-07-27 (v5 prep §1.4). The GROUNDING
           # INSTRUMENT is a PAIR and is worthless as a single number:
           #   `g_op_mid_de_m`  — metric inverse dynamics on REAL pairs
           #                      (`metric_dynamics.py:388`), i.e. how well the
           #                      encoder's own latents carry ego-motion;
           #   `g_op_fwd_ade_m` — forward consistency on the IMAGINED rollout
           #                      (`:389`), i.e. how well the predictor's
           #                      imagined transitions decode to metres.
           # The diagnostic quantity is their RATIO (real vs imagined; v1 runs
           # 2.36x at 0-1k -> 33.3x at 28-30k). With only the imagined half
           # logged, a rise is unattributable — encoder drift and predictor
           # drift are indistinguishable, which is why all three v4 logs were
           # undiagnosable. Both come from the SAME batch and the SAME forward
           # pass (one `flagship_loss` call, one `rollout_transitions` to k_max
           # shared by every level), exactly as v1's log does — v1 gets them by
           # an UNFILTERED `row.update(log)` (`train_flagship4b.py:535`), which
           # is why it never had this defect.
           # ⚠️ BOTH filters must change: this one AND the row-writer's tuple
           # below. Patching one leaves the other starved and the fix silently
           # inert — the exact failure documented above.
           **{k: v for k, v in wm_log.items()
              if k in ("g_op_fwd_ade_m", "g_op_mid_de_m")},
           # ⭐ THE FOUR SELECTION DIAGNOSTICS — the SECOND half of the M4 fix,
           # added 2026-07-27 (v5 prep §1.5).
           #
           # ⚠️ FOUND BY TEST, 2026-07-27: the M4 row-writer fix was ALREADY
           # INERT. `_training_loop`'s row tuple lists `sel_gap` / `rank_acc` /
           # `frac_sel_2x_worse_than_oracle` guarded by `if k in log` — but
           # `v15_losses` returns them in `plan_l`, which THIS dict never merged.
           # Only `ade` and `oracle_ade` were ever lifted out of `plan_l`. So the
           # guard evaluated False every step and all three were dropped exactly
           # as before, silently, with a fix in place that looked done. This is
           # the SAME double-filter failure as `g_op_fwd_ade_m` above, on the
           # very diagnostics that were computed 601x per run and thrown away —
           # the ones that would have shown held-out selection regressing from
           # ~step 11,000 while every training term improved (~29.5 GPU-h).
           # `sel_gate` / `sel_pen_span` arrive separately via `out["telemetry"]`
           # and only when the longitudinal selection term is active.
           # LOG-ONLY: detached floats, no loss term, no parity effect.
           **{k: float(plan_l[k].detach()) for k in
              ("sel_gap", "rank_acc", "frac_sel_2x_worse_than_oracle")
              if k in plan_l},
           **fac_log, **sm_log, **strat_log, **out.get("telemetry", {})}
    return total, log


def _goal_inputs(cfg: V4Config, batch: dict, v0: torch.Tensor) -> dict:
    """Assemble the head's optional goal/imagination kwargs from a batch, defaulting
    to the smoke-safe minimum. The real trainer supplies v3 goal tokens + imagined
    latents; this keeps the loss step runnable without them."""
    kw: dict = {}
    b = v0.shape[0]
    if cfg.cond_vtarget:
        kw["vt_band"] = batch.get("vt_band", torch.zeros(b, dtype=torch.long))
        kw["vt_speed"] = v0
    if cfg.cond_route:
        kw["route"] = batch.get("route", torch.zeros(b, dtype=torch.long))
        kw["route_graded"] = batch.get("route_graded", torch.zeros(b))
    return kw


# ============================================================================
# CPU smoke — build the pieces and run a few joint steps across phase A/B/C
# ============================================================================

def smoke() -> dict:
    """A self-contained joint-training smoke: WM + operative planner + factorised +
    smoothness, run across the phase-A/B/C λ_plan boundaries on toy data. Proves the
    step is finite and differentiable and that λ_plan actually moves."""
    import dataclasses
    import math

    from torch.utils.data import default_collate

    from tanitad.config import flagship4b_smoke_config
    from tanitad.data._contract import assemble_episode
    from tanitad.models.fourbrain import WorldModel
    from train_flagship4b import FlagshipWindowDataset

    def toy_episode(T: int, eid: int, size: int = 64):
        g = torch.Generator().manual_seed(100 + eid)
        frames = [torch.rand(1, size, size, generator=g) for _ in range(T)]
        rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, 8.0
        dt, yaw_rate = 0.1, (0.05 if eid % 2 else -0.05)
        accel = -1.0 if eid % 2 else 1.0
        for _ in range(T):
            rows.append([x, y, yaw, v])
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += yaw_rate * dt
            v = max(0.0, v + accel * dt)
        poses = torch.tensor(rows)
        return assemble_episode(frames, [p.numpy() for p in poses],
                                [yaw_rate] * T, 0.1, eid)

    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    world = WorldModel(cfg)
    grounding = build_grounding(world.state_dim, hidden=32)
    plan = horizon_plan(cfg, op_fwd_k=2, tac_fwd_k=3, str_fwd_k=4)

    hcfg = _smoke_head_cfg(world.state_dim, cfg.predictor.window)
    head = FlagshipV4Head(hcfg)

    eps = [toy_episode(60, i) for i in range(4)]
    ds = FlagshipWindowDataset(eps, window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    batch = default_collate([ds[i] for i in range(4)])

    phases = CurriculumPhases(phase_a=2, phase_b=6)
    opt = torch.optim.AdamW(list(world.parameters()) + list(head.parameters())
                            + list(grounding.parameters()), lr=1e-4)
    lw = V4LossWeights()
    logs = []
    for step in (0, 4, 8):                              # Phase A / B / C
        opt.zero_grad(set_to_none=True)
        total, log = v4_loss_step(world, grounding, head, batch, plan, cfg,
                                  step, phases, lw)
        total.backward()
        opt.step()
        assert torch.isfinite(total), (step, log)
        logs.append((step, log))
    # λ_plan really moves across the phases (0 -> ramp -> 1)
    assert logs[0][1]["lambda_plan"] == 0.0
    assert logs[-1][1]["lambda_plan"] == 1.0
    return {"logs": logs}


def real_smoke(train_cache: str, n_episodes: int = 3, n_windows: int = 8,
               trunk: str | None = None, seed: int = 0) -> dict:
    """PROOF: run ``v4_loss_step`` on a batch of REAL parity windows and show the
    factorised CE and the strategic-scalar loss train on real (non-IGNORE) targets.

    This is the deliverable that matters — it demonstrates that the minted v4
    labels reach the two marquee heads as real class/scalar targets (not
    IGNORE_INDEX), that both loss terms are non-zero, and that gradient flows into
    the LAT/LON/DIST heads and the goal-scalar head. Coverage per slot is reported
    (a head at 2 % coverage is nearly as dead as IGNORE). Real 256 px frames on a
    fresh trunk, CPU: keep ``n_windows`` small.
    """
    import dataclasses

    from torch.utils.data import default_collate

    from tanitad.config import flagship4b_config
    from tanitad.data.mixing import load_episode
    from tanitad.models.fourbrain import WorldModel
    from tanitad.models.strategic_goal import (GoalScalarConfig, GoalScalarHead,
                                               param_count)
    from flagship_v4_data import FlagshipV4Dataset

    torch.manual_seed(seed)
    cfg = flagship4b_config()
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    world = WorldModel(cfg)
    if trunk and str(trunk).strip().lower() != "none":   # optional warm-start (realism)
        ck = torch.load(trunk, map_location="cpu", weights_only=False)
        world.load_state_dict(ck["model"])
        print(f"[real-smoke] warm-started trunk from {trunk}", flush=True)
    else:
        # FROM-SCRATCH: the trunk stays random-initialized. This exercises the exact
        # init the v4 from-scratch fallback launches with (WM + planner co-evolve).
        print("[real-smoke] FROM-SCRATCH — trunk random-initialized (no warm-start); "
              "canary/losses start at their untrained values", flush=True)
    grounding = build_grounding(world.state_dim, hidden=32)
    plan = horizon_plan(cfg, op_fwd_k=2, tac_fwd_k=3, str_fwd_k=4)

    hcfg = v4_config()
    hcfg.state_dim = world.state_dim
    hcfg.cond_imagination = False                   # imagination is v1.5-inherited,
    hcfg.window = cfg.predictor.window              # tested elsewhere; off here so
    head = FlagshipV4Head(hcfg)                      # the proof isolates the labels
    goal_head = GoalScalarHead(GoalScalarConfig(in_dim=world.state_dim))

    # real parity windows off the epcache split dir
    files = sorted(Path(train_cache).glob("ep_*.pt"))[:max(n_episodes, 1)]
    if not files:
        raise SystemExit(f"[real-smoke] no ep_*.pt under {train_cache}")
    eps = [load_episode(str(p), mmap=True) for p in files]
    ds = FlagshipV4Dataset(eps, window=cfg.predictor.window,
                           max_horizon=plan.max_horizon, maneuver_h=plan.maneuver_h,
                           channels=cfg.encoder.in_channels)
    idx = list(range(min(n_windows, len(ds))))
    batch = default_collate([ds[i] for i in idx])
    b = len(idx)

    # per-slot coverage over THIS batch (the full-corpus figures come from build())
    def cov_long(k):
        return round(float((batch[k] != IGNORE_INDEX).float().mean()), 4)
    cov = {k: cov_long(k) for k in ("lat_target", "lon_target", "dist_target",
                                    "stop_dist_target", "route_token")}
    cov["strat_scalars"] = {n: round(float(batch["strat_scalar_mask"][:, i]
                                           .float().mean()), 4)
                            for i, n in enumerate(("ttm", "curv_3s", "curv_5s",
                                                   "tspeed_5s"))}

    lw = V4LossWeights()
    opt = torch.optim.AdamW(list(world.parameters()) + list(head.parameters())
                            + list(grounding.parameters())
                            + list(goal_head.parameters()), lr=1e-4)
    opt.zero_grad(set_to_none=True)
    total, log = v4_loss_step(world, grounding, head, batch, plan, cfg,
                              step=9000, phases=CurriculumPhases(2000, 8000),
                              lw=lw, goal_head=goal_head)
    total.backward()

    def gsum(module):
        return sum(float(p.grad.abs().sum()) for p in module.parameters()
                   if p.grad is not None)
    grads = {"lat_head": gsum(head.lat_head), "lon_head": gsum(head.lon_head),
             "dist_head": gsum(head.dist_head), "goal_head": gsum(goal_head)}

    result = {
        "batch_windows": b, "n_episodes": len(eps),
        "coverage_this_batch": cov,
        "losses": {k: log[k] for k in ("lat_ce", "lon_ce", "dist_ce",
                                       "strat_scalar_loss", "total") if k in log},
        "strat_scalar_cov_batch": log.get("strat_scalar_cov"),
        "grads_into_heads": {k: round(v, 6) for k, v in grads.items()},
        "goal_head_params": param_count(goal_head),
        "factorised_ce_trains": all(log.get(k, 0.0) > 0.0
                                    for k in ("lat_ce", "lon_ce", "dist_ce")),
        "strategic_scalar_trains": log.get("strat_scalar_loss", 0.0) > 0.0,
        "all_heads_receive_grad": all(v > 0.0 for v in grads.values()),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def _smoke_head_cfg(state_dim: int, window: int) -> V4Config:
    from tanitad.refs.refc import DecoderConfig
    cfg = v4_config()
    cfg.state_dim = state_dim
    cfg.readout_grid = 4
    cfg.d_cell = state_dim // 16
    cfg.window = window
    cfg.horizons = (1, 2, 3, 4)
    cfg.cond_imagination = False           # the imagination path is v1.5-inherited
    cfg.cond_vtarget = cfg.cond_route = False
    cfg.n_anchors = 16
    cfg.d_token = 32
    cfg.d_meas = 16
    cfg.factor_hidden = 16
    cfg.decoder = DecoderConfig(d=32, n_heads=2, layers=2, ff_mult=2,
                                aux_hidden=32, diffusion_steps=2, noise_std=0.1)
    return cfg


# ============================================================================
# The multi-day training LOOP (P4) — mirrors train_flagship_v16.main()'s
# skeleton (data pipeline, AdamW head/trunk groups, cosine LR, canary eval,
# atomic ckpt save/resume, milestone archive, metrics.json) and adds the two
# v4-specific mechanisms: v4_loss_step (the WM stack live from step 0 + the
# factorised/strategic planner terms) and the λ_plan canary CONTROLLER (§5.5).
# NOTHING here launches a run — Sayed owns the go/no-go (§17); ``main`` gates on
# ``preflight_asserts`` and only THEN calls ``train`` (never on the pod from an
# agent). The full-loop CPU proof is ``smoke_loop`` / ``--smoke-loop``.
# ============================================================================


def _cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    """Warmup-then-cosine, reused verbatim from ``train_flagship_v16.cosine_lr``
    (both the head and the trunk group follow it off their own base LR)."""
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def _param_grad_norm(params) -> float:
    """L2 norm of the gradients over a parameter subset, for logging (pre-clip). 0.0
    if none has a grad. Used to log ``gnorm_encoder`` / ``gnorm_predictor`` separately
    so the encoder and predictor are each provably UPDATING (Sayed's requirement)."""
    gs = [p.grad.detach() for p in params if p.grad is not None]
    if not gs:
        return 0.0
    return float(torch.norm(torch.stack([g.norm() for g in gs])))


def _to_device(batch: dict, device) -> dict:
    """Move every tensor value of a collated batch to ``device`` (non-tensor
    entries — e.g. ``episode_id`` after collate is a tensor anyway — pass through).
    ``v4_loss_step`` encodes ``frames``/``future_frames`` itself, so they must be
    resident before the call; ``flagship_loss`` re-``.to(device)``s the rest (a
    no-op once already there)."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


# --------------------------------------------------------------------------- #
# canary — the plan-free operative WM rollout (world-model collapse detector).  #
# Computes the SAME quantity as train_flagship_v16.canary_rollout (the ~0.452   #
# reference from eval_grounded_rollout_4b_speed.py): predictor rolled forward    #
# under TRUE actions -> grounding.step['op'] -> SE(2) -> ADE at the horizons.    #
# It is NOT byte-reused because the v4 batch differs from V16FramesDataset: v4   #
# frames are FLOAT [0,1] (EpisodeWindowDataset.to_float_frames), not uint8, so   #
# the encode goes through world.encode_window (correct for the FULLY-unfrozen v4 #
# trunk) rather than encode_window_ft's uint8/255 path; and the GT waypoints +   #
# the v0 speed action channel are derived from the batch-dict contract           #
# (pose_last/future_poses/actions) since v4 mints no precomputed traj_tgt.        #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def canary_rollout(world, grounding, ds_val, device, *,
                   horizons=(5, 10, 15, 20), k_max: int = 20,
                   episodes: int = 40, stride: int = 8, batch: int = 16,
                   amp: bool = True) -> dict:
    """Operative predictor rollout under TRUE actions -> ADE@2s (the WM-integrity
    canary). Re-encodes through the CURRENT trunk so the number reflects the world
    model as it is fine-tuned. Returns ``{"canary_ade@2s", "n"}``."""
    from tanitad.models.flagship_v15 import SPEED_SCALE
    from tanitad.models.metric_dynamics import gt_ego_waypoints, rollout_decode

    step_readout = grounding.step["op"]
    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < episodes and t % stride == 0]
    if not sel:
        return {"canary_ade@2s": float("nan"), "n": 0}
    amp_on = amp and str(device) == "cuda"
    wp_idx = torch.tensor([k - 1 for k in horizons], device=device)
    errs = []
    for b0 in range(0, len(sel), batch):
        items = [ds_val[i] for i in sel[b0:b0 + batch]]
        fr = torch.stack([x["frames"] for x in items]).to(device)          # [B,W,C,H,W]
        aw2 = torch.stack([x["actions"] for x in items]).to(device).float()  # [B,W,2]
        fa2 = torch.stack([x["future_actions"] for x in items]).to(device).float()
        fp = torch.stack([x["future_poses"] for x in items]).to(device).float()
        pl = torch.stack([x["pose_last"] for x in items]).to(device).float()  # [B,4]
        v0 = pl[:, 3]                                                        # [B]
        vch = (v0 / SPEED_SCALE)[:, None, None]
        aw = torch.cat([aw2, vch.expand(-1, aw2.shape[1], -1)], dim=-1)      # [B,W,3]
        fa = torch.cat([fa2, vch.expand(-1, fa2.shape[1], -1)], dim=-1)      # [B,H,3]
        gt = gt_ego_waypoints(pl, fp, horizons)                             # [B,len,2]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
            states = world.encode_window(fr)
            wp_full, _ = rollout_decode(world.predictor, states, aw, fa,
                                        step_readout, k_max)                # [B,k,2]
        pred = wp_full.index_select(1, wp_idx).float()
        errs.append((pred - gt).norm(dim=-1).mean(dim=1).cpu())            # [B]
    e = torch.cat(errs)
    return {"canary_ade@2s": float(e.mean()), "n": int(e.shape[0])}


# --------------------------------------------------------------------------- #
# planner eval — the head's proposal quality (oracle-in-fan, sel_gap, ADE),     #
# reusing v15_losses on val windows (the SAME diagnostic v16.evaluate reports).  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_planner(head, world, ds_val, device, *, episodes: int = 40,
                     stride: int = 8, batch: int = 16, amp: bool = True) -> dict:
    """ADE@2s + oracle-in-fan + sel_gap over the val windows, re-encoding through
    the current trunk. Mirrors ``train_flagship_v16.evaluate`` but reuses
    ``v15_losses`` (already the operative-planner loss in ``v4_loss_step``)."""
    from torch.utils.data import default_collate

    from tanitad.models.flagship_v15 import v15_losses

    head.eval()
    horizons = head.cfg.horizons
    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < episodes and t % stride == 0]
    amp_on = amp and str(device) == "cuda"
    ade, oracle, gap, miss, n = [], [], [], [], 0
    for b0 in range(0, len(sel), batch):
        b = _to_device(default_collate([ds_val[i] for i in sel[b0:b0 + batch]]),
                       device)
        v0 = b["pose_last"][:, 3].float()
        traj_tgt = refb_labels.waypoint_targets(
            b["pose_last"].float(),
            b["future_poses"][:, :max(horizons)].float(), horizons)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
            st = world.encode_window(b["frames"])
            out = head(st, v0, lambda_plan=1.0, **_goal_inputs(head.cfg, b, v0))
        lg = v15_losses(out, head.decoder.anchors, traj_tgt)
        bs = traj_tgt.shape[0]
        ade.append(float(lg["ade"]) * bs); oracle.append(float(lg["oracle_ade"]) * bs)
        gap.append(float(lg["sel_gap"]) * bs)
        fde = (out["traj"][:, -1] - traj_tgt[:, -1]).norm(dim=-1)
        miss.append(float((fde > 2.0).float().sum())); n += bs
    head.train()
    if n == 0:
        return {"n": 0}
    return {"n": n, "ade@2s": sum(ade) / n, "oracle_ade@2s": sum(oracle) / n,
            "sel_gap@2s": sum(gap) / n, "miss@2m": sum(miss) / n}


# --------------------------------------------------------------------------- #
# checkpoint — atomic save + milestone archive; resume restores EVERYTHING       #
# including the canary-controller state (so a mid-run restart is exact, §6/§7).   #
# --------------------------------------------------------------------------- #
def _save_ckpt_v4(path: Path, *, world, grounding, head, goal_head, opt, step,
                  controller: CanaryController, phases: CurriculumPhases,
                  milestones=(), heldout_gate=None) -> None:
    obj = {
        "model": world.state_dict(), "grounding": grounding.state_dict(),
        "head": head.state_dict(),
        "goal_head": goal_head.state_dict() if goal_head is not None else None,
        "opt": opt.state_dict(), "step": step,
        # the λ_plan controller state — restored bit-exact on resume so the guard
        # does not silently re-open λ_plan after a pod restart.
        "lam_mult": float(controller._mult),
        "controller": {"baseline": float(controller.baseline),
                       "_hard_streak": int(controller._hard_streak),
                       "_mult": float(controller._mult)},
        "phases": {"phase_a": phases.phase_a, "phase_b": phases.phase_b},
        # ⭐ the mid-run held-out gate's state — streak, incumbent and the PINNED
        # admitted-component set. A restart that reset the streak to 0 would let
        # a decayed run walk past two bad probes forever by dying every other
        # probe: the same unwatched-run failure with extra steps.
        "heldout_gate": (None if heldout_gate is None
                         else heldout_gate.state_dict()),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)
    # milestone archive (Sayed 2026-07-18 / §17): keep gate-step + 5k/15k/20k/30k
    # so run_gate.py can score them post-hoc. ATOMIC (a bare copy2 can leave a
    # truncated milestone that exists() then treats as done forever).
    from tanitad.train.ckpt_io import atomic_archive
    for m in milestones:
        arch = path.with_name(f"ckpt_step{m}.pt")
        if step >= m and not arch.exists():
            atomic_archive(path, arch)
            print(f"[ckpt] milestone archived: {arch.name}", flush=True)


def load_checkpoint_v4(path: Path, *, world, grounding, head, goal_head, opt,
                       controller: CanaryController, device,
                       heldout_gate=None) -> int:
    """Restore a run from ``ckpt.pt`` and return the step to RESUME at
    (``saved_step + 1``). Restores model/grounding/head/goal_head/opt AND the
    controller's ``_mult`` / ``_hard_streak`` / ``baseline`` bit-exact — and, when
    supplied, the mid-run held-out gate's streak/incumbent/pinned ranges."""
    ck = torch.load(path, map_location=device, weights_only=False)
    world.load_state_dict(ck["model"])
    grounding.load_state_dict(ck["grounding"])
    head.load_state_dict(ck["head"])
    if goal_head is not None and ck.get("goal_head") is not None:
        goal_head.load_state_dict(ck["goal_head"])
    opt.load_state_dict(ck["opt"])
    c = ck.get("controller", {})
    controller.baseline = float(c.get("baseline", controller.baseline))
    controller._hard_streak = int(c.get("_hard_streak", 0))
    controller._mult = float(c.get("_mult", ck.get("lam_mult", 1.0)))
    if heldout_gate is not None and ck.get("heldout_gate"):
        heldout_gate.load_state_dict(ck["heldout_gate"])
    return int(ck["step"]) + 1


# --------------------------------------------------------------------------- #
# parity — refuse anything that re-selects episodes off the canonical corpus     #
# --------------------------------------------------------------------------- #
def _assert_parity(train_cache: str, val_cache: str) -> dict:
    """The corpus is SACRED (CLAUDE.md §Invariants): the canonical train set is
    ``physicalai-train-e438721ae894`` (2376 eps, skip-hash ``f09e44db``).

    ⚠️ CORRECTED 2026-07-25 (Wave-1 B). This used to be a path SUBSTRING match
    plus the comment *"episode re-selection is structurally impossible"*. It is
    not: the loop consumes every ``ep_*.pt`` that is THERE, so a build the
    ``/workspace`` MooseFS quota killed halfway leaves a correctly-NAMED dir
    with fewer episodes and trained silently. Measured: a 1 200-of-2 376 cache
    passed this function (``tests/test_parity_manifest.py``, red before the
    fix). The check is now a real CONTENT check — count + ``sha256(sorted
    episode uids)`` vs the committed manifest — run BEFORE any GPU allocation.
    """
    tc = str(Path(train_cache).resolve()).replace("\\", "/")
    train_rec = parity.assert_parity_corpus(train_cache, label="--train-cache",
                                            require=True)
    # val: refuses the known-LEAKY physicalai-val-f1b378f295ae outright, and
    # count-checks physicalai-val-0c5f7dac3b11 (600). require=False so a
    # deliberate alternative held-out set warns instead of blocking.
    val_rec = parity.assert_parity_corpus(val_cache, label="--val-cache")
    return {"train_corpus_key": PARITY_KEY, "skip_hash": PARITY_SKIP_HASH,
            "train_cache": tc, "val_cache": str(Path(val_cache).resolve()),
            "episodes_verified": train_rec["episodes_loaded"],
            "episode_uid_sha256": train_rec["episode_uid_sha256"],
            "episode_reselection": "refused (count + uid-digest content check "
                                   "vs tanitad/data/parity_manifest.json)",
            "train_parity": train_rec, "val_parity": val_rec}


# --------------------------------------------------------------------------- #
# the loop — ONE body, driven by both train() (real caches) and smoke_loop()     #
# (toy episodes). This is the P4 deliverable; it never launches a pod run.        #
# --------------------------------------------------------------------------- #
def _training_loop(*, out_dir: Path, device, amp: bool, world, grounding, head,
                   goal_head, opt, plan, cfg, phases: CurriculumPhases,
                   lw: V4LossWeights, controller: CanaryController, dl, ds_val,
                   steps: int, log_every: int, eval_every: int, save_every: int,
                   warmup: int, lr_head: float, lr_trunk: float, gate_step: int,
                   lam_mode: str, canary_horizons, canary_kmax: int,
                   eval_episodes: int, batch: int, milestones,
                   accum: int = 1, canary_override=None,
                   heldout_gate=None, heldout_episodes=None) -> dict:
    """Run the joint WM + planner training loop. Auto-resumes from ``ckpt.pt`` if
    present (pod-restart safe). Returns a result dict (final step, canary trace,
    controller multiplier trace, milestone archives) for the smoke proof.

    ``accum`` micro-batches are accumulated per optimizer step (each micro loss is
    scaled by 1/accum), so the EFFECTIVE batch = ``batch × accum``. v4.2 uses
    16 × 4 = 64 to match v1's effective batch (registry §1.2:
    ``--batch-size 16 --accum 4``); v4.1 ran accum 1 = effective 16, 4× too small.

    ⭐ ``heldout_gate`` (:class:`tanitad.train.heldout_gate.HeldoutGate`) is THE
    MID-RUN HELD-OUT GATE. At its own fixed cadence it probes the DEPLOYABLE
    SURFACE on ``heldout_episodes`` under pseudo-simulation and **breaks the loop
    when the held-out primary has been separated-worse for two consecutive
    probes**. MEASURED cost of not having it on the v4 30k run: **~29.5 GPU-h,
    half the run, spent training past the best checkpoint while every training
    term improved.** It also archives ``ckpt_best.pt`` at every new incumbent, so
    the best checkpoint survives even when the run is allowed to continue.
    ``None`` leaves the loop byte-identical to before (the smoke/tests path)."""
    ckpt_p = out_dir / "ckpt.pt"
    log_f = (out_dir / "train_log.jsonl").open("a")

    step = 0
    if ckpt_p.exists():
        step = load_checkpoint_v4(ckpt_p, world=world, grounding=grounding,
                                  head=head, goal_head=goal_head, opt=opt,
                                  controller=controller, device=device,
                                  heldout_gate=heldout_gate)
        print(f"[resume] step {step} lam_mult={controller._mult}"
              + ("" if heldout_gate is None else
                 f" heldout_worse_streak={heldout_gate.worse_streak}"), flush=True)
        if heldout_gate is not None and heldout_gate.stop:
            raise SystemExit(
                f"[v4] refusing to resume: the mid-run held-out gate already "
                f"STOPPED this run at step {step - 1}. {heldout_gate.stop_reason} "
                f"The best checkpoint is ckpt_best.pt. Resuming would spend "
                f"exactly the GPU-hours this gate exists to save; if that is "
                f"deliberate, say so and clear the gate state explicitly.")

    world.train(); grounding.train(); head.train()
    if goal_head is not None:
        goal_head.train()
    trunk_params = list(world.parameters())
    # ⭐ Split the trunk so the ENCODER and PREDICTOR are provably training separately
    # (Sayed's hard requirement — gnorm_encoder>0 / gnorm_predictor>0 in the log, not
    # a merged trunk number). These are the SAME tensors as in trunk_params (subsets),
    # used only to READ per-module grad norms; the clip still acts on the whole trunk.
    encoder_params = [p for n, p in world.named_parameters() if n.startswith("encoder")]
    predictor_params = [p for n, p in world.named_parameters() if n.startswith("predictor")]
    head_group = [p for p in head.parameters()]
    head_group += list(grounding.parameters())
    if goal_head is not None:
        head_group += list(goal_head.parameters())

    # canary BASELINE on the warm (step-0) trunk — the controller's reference.
    if step == 0:
        base = canary_rollout(world, grounding, ds_val, device,
                              horizons=canary_horizons, k_max=canary_kmax,
                              episodes=eval_episodes, batch=batch, amp=amp)
        controller.baseline = base["canary_ade@2s"]
        row = {"step": 0, "canary_baseline": base, "ref_0.452": 0.452}
        print(json.dumps(row), flush=True)
        log_f.write(json.dumps(row) + "\n"); log_f.flush()

    it = iter(dl); t0 = time.time()
    canary_trace, mult_trace, eval_i = [], [], 0
    while step < steps:
        lr_h = _cosine_lr(step, steps, warmup, lr_head)
        lr_t = _cosine_lr(step, steps, warmup, lr_trunk)
        for pg in opt.param_groups:
            pg["lr"] = lr_h if pg.get("name") == "head" else lr_t
        # --- gradient accumulation: ``accum`` micro-batches -> effective batch -----
        opt.zero_grad(set_to_none=True)
        log = None
        for _micro in range(accum):
            try:
                batch_d = next(it)
            except StopIteration:
                it = iter(dl); batch_d = next(it)
            batch_d = _to_device(batch_d, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                total, log = v4_loss_step(
                    world, grounding, head, batch_d, plan, cfg, step, phases, lw,
                    lam_mode=lam_mode, lam_mult=float(controller._mult),
                    device=str(device), goal_head=goal_head)
            if not torch.isfinite(total):
                raise SystemExit(f"[v4] non-finite loss at step {step} "
                                 f"(micro {_micro}/{accum}): {log}")
            (total / accum).backward()          # mean over the accum micro-batches
        # per-module grad norms on the ACCUMULATED grad (pre-clip) — the not-frozen
        # proof: encoder AND predictor must show > 0 (they are UPDATING, not frozen).
        gn_enc = _param_grad_norm(encoder_params)
        gn_pred = _param_grad_norm(predictor_params)
        gn_h = float(torch.nn.utils.clip_grad_norm_(head_group, 1.0))
        gn_t = float(torch.nn.utils.clip_grad_norm_(trunk_params, 1.0))
        opt.step()

        if step % log_every == 0 or step == steps - 1:
            row = {"step": step, "lr_head": round(lr_h, 8),
                   "lr_trunk": round(lr_t, 8), "lam_mult": float(controller._mult),
                   "gnorm_head": round(gn_h, 3), "gnorm_trunk": round(gn_t, 3),
                   "gnorm_encoder": round(gn_enc, 3),
                   "gnorm_predictor": round(gn_pred, 3),
                   "eff_batch": batch * accum,
                   "elapsed_s": round(time.time() - t0, 1),
                   **{k: log[k] for k in ("total", "lambda_plan", "wm", "planner",
                                          "plan_ade", "oracle_ade",
                                          # g_op_fwd_ade_m is the operative-rollout
                                          # ADE the `speed_benefit_recovered_frac`
                                          # KILL secondary reduces (§7.5 / P8). It is
                                          # already computed in `log` (loss_step) but
                                          # was dropped from the written row, so a v4
                                          # arm's log was not gate-computable; the
                                          # reference arms (flagship4b) log it. Adding
                                          # it is LOG-ONLY (no loss/parity effect).
                                          "g_op_fwd_ade_m",
                                          # ⭐ THE OTHER HALF OF THE GROUNDING
                                          # INSTRUMENT (added 2026-07-27). Real
                                          # pairs (`g_op_mid_de_m`) vs imagined
                                          # pairs (`g_op_fwd_ade_m`), same batch
                                          # and same forward pass. The ratio is
                                          # the diagnostic; one number alone
                                          # cannot separate encoder drift from
                                          # predictor drift, which is why v4 was
                                          # undiagnosable (defect #3 of the last
                                          # run). v1 logs both.
                                          "g_op_mid_de_m",
                                          # ⭐ THE FOUR SELECTION DIAGNOSTICS, added
                                          # 2026-07-26 (BOOST_PROGRAM M4). They are
                                          # ALREADY COMPUTED EVERY STEP — `v15_losses`
                                          # emits sel_gap/rank_acc/
                                          # frac_sel_2x_worse_than_oracle
                                          # (flagship_v15.py:617-618) and `select()`
                                          # emits sel_gate/sel_pen_span (:462-463) —
                                          # and this row-writer DISCARDED all of them.
                                          # `train_flagship_v15.py:379,384` already
                                          # logs two of them, so v4 was strictly
                                          # behind its own sibling trainer.
                                          #
                                          # COST OF THE OMISSION, MEASURED: the 30k
                                          # run regressed on held-out SELECTION while
                                          # every training term improved (C11); the
                                          # onset was ~step 11,000 and ~29.5 GPU-h —
                                          # HALF THE RUN — went into training past the
                                          # best checkpoint. The exact diagnostics that
                                          # would have shown it were computed 601 times
                                          # and thrown away.
                                          #
                                          # LOG-ONLY: no loss term, no parity effect,
                                          # no change to any optimisation path. `if k
                                          # in log` keeps it safe when a key is absent
                                          # (e.g. a non-v15 head).
                                          "sel_gap", "rank_acc",
                                          "frac_sel_2x_worse_than_oracle",
                                          "sel_gate", "sel_pen_span")
                                       if k in log}}
            print(json.dumps(row), flush=True)
            log_f.write(json.dumps(row) + "\n"); log_f.flush()

        if step > 0 and step % eval_every == 0:
            can = canary_rollout(world, grounding, ds_val, device,
                                 horizons=canary_horizons, k_max=canary_kmax,
                                 episodes=eval_episodes, batch=batch, amp=amp)
            ev = evaluate_planner(head, world, ds_val, device,
                                  episodes=eval_episodes, batch=batch, amp=amp)
            # §5.5 CONTROLLER: feed the canary; it may only pull λ_plan DOWN.
            ctrl_val = (canary_override[min(eval_i, len(canary_override) - 1)]
                        if canary_override else can["canary_ade@2s"])
            mult, action = controller.update(ctrl_val)
            eval_i += 1
            canary_trace.append(can["canary_ade@2s"]); mult_trace.append(mult)
            row = {"step": step, "canary_ade@2s": round(can["canary_ade@2s"], 5),
                   "canary_vs_base": round(
                       can["canary_ade@2s"] - controller.baseline, 5),
                   "lam_mult": mult, "controller_action": action, "val": ev}
            print(json.dumps(row), flush=True)
            log_f.write(json.dumps(row) + "\n"); log_f.flush()

        # ---------------------------------------------------------------- #
        # ⭐ THE MID-RUN HELD-OUT GATE (v5 prep §1.6). The single most valuable
        # item on the card: it converts the measured 59-hour loss into ~20 h.
        # The primary is the MAP-FREE COMPOSITE, never `ade_0_2s` — the ADE-
        # optimal pick collides 4.7x more often than the rule-optimal one.
        # ---------------------------------------------------------------- #
        if heldout_gate is not None and heldout_gate.due(step):
            hrec = heldout_gate.probe(step, world, head, heldout_episodes,
                                      device=str(device))
            print(json.dumps({"heldout_gate": hrec}, default=str), flush=True)
            log_f.write(json.dumps({"step": step, "heldout_gate": hrec},
                                   default=str) + "\n")
            log_f.flush()
            if hrec.get("incumbent_step") == step:
                # the best-so-far checkpoint, kept SEPARATELY from ckpt.pt so the
                # run's peak survives whatever the tail does. Not having this is
                # why the last run's best weights had to be recovered by hand.
                _save_ckpt_v4(out_dir / "ckpt_best.pt", world=world,
                              grounding=grounding, head=head, goal_head=goal_head,
                              opt=opt, step=step, controller=controller,
                              phases=phases, milestones=())
            if heldout_gate.stop:
                _save_ckpt_v4(ckpt_p, world=world, grounding=grounding, head=head,
                              goal_head=goal_head, opt=opt, step=step,
                              controller=controller, phases=phases,
                              milestones=milestones, heldout_gate=heldout_gate)
                print(json.dumps({"EARLY_STOP": True, "step": step,
                                  "reason": heldout_gate.stop_reason}), flush=True)
                log_f.write(json.dumps({"step": step, "EARLY_STOP": True,
                                        "reason": heldout_gate.stop_reason})
                            + "\n")
                log_f.flush()
                break

        if step > 0 and step % save_every == 0:
            _save_ckpt_v4(ckpt_p, world=world, grounding=grounding, head=head,
                          goal_head=goal_head, opt=opt, step=step,
                          controller=controller, phases=phases,
                          milestones=milestones, heldout_gate=heldout_gate)
        step += 1

    _save_ckpt_v4(ckpt_p, world=world, grounding=grounding, head=head,
                  goal_head=goal_head, opt=opt, step=step - 1,
                  controller=controller, phases=phases, milestones=milestones)
    final_can = canary_rollout(world, grounding, ds_val, device,
                               horizons=canary_horizons, k_max=canary_kmax,
                               episodes=eval_episodes, batch=batch, amp=amp)
    final_ev = evaluate_planner(head, world, ds_val, device,
                                episodes=eval_episodes, batch=batch, amp=amp)
    archives = sorted(p.name for p in out_dir.glob("ckpt_step*.pt"))
    metrics = {"final_step": step - 1, "canary_ade@2s": final_can["canary_ade@2s"],
               "canary_baseline": controller.baseline, "val": final_ev,
               "lam_mult_final": float(controller._mult),
               "milestone_archives": archives,
               "wallclock_s": round(time.time() - t0, 1),
               # the gate's verdict is part of the run record, not a side note:
               # "why did this run end" must be answerable from metrics.json.
               "heldout_gate": (None if heldout_gate is None else {
                   "primary": _HG.PRIMARY_NAME,
                   "refused_primary": _HG.REFUSED_PRIMARY,
                   "early_stopped": bool(heldout_gate.stop),
                   "stop_reason": heldout_gate.stop_reason,
                   "worse_streak": int(heldout_gate.worse_streak),
                   "best_step": (None if heldout_gate._incumbent is None
                                 else heldout_gate._incumbent.step),
                   "best_ckpt": ("ckpt_best.pt"
                                 if (out_dir / "ckpt_best.pt").exists() else None),
                   "n_probes": len(heldout_gate.history),
                   "history": heldout_gate.history})}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                          encoding="utf-8")
    log_f.close()
    print(json.dumps({"done": True, **{k: metrics[k] for k in
                      ("final_step", "canary_ade@2s", "lam_mult_final",
                       "milestone_archives")}}), flush=True)
    return {"final_step": step - 1, "canary_trace": canary_trace,
            "mult_trace": mult_trace, "archives": archives, "metrics": metrics,
            "ckpt": str(ckpt_p),
            "early_stopped": bool(heldout_gate is not None and heldout_gate.stop),
            "heldout_history": ([] if heldout_gate is None
                                else heldout_gate.history)}


def train(a) -> dict:
    """The real multi-day run: parity-checked caches, the v1 trunk warm-started
    and fine-tuned end-to-end under the λ_plan curriculum, the operative planner +
    factorised/strategic heads, the canary controller, milestone archiving. Builds
    and runs the loop; a launch is Sayed's go (§17), executed by the orchestrator,
    NOT from an agent."""
    import dataclasses

    from tanitad.config import flagship4b_config
    from tanitad.data.mixing import load_episode
    from tanitad.models.fourbrain import WorldModel
    from tanitad.models.strategic_goal import GoalScalarConfig, GoalScalarHead
    from flagship_v4_data import FlagshipV4Dataset

    from_scratch = _is_from_scratch(a)
    for req, name in ((a.train_cache, "--train-cache"), (a.val_cache, "--val-cache")):
        if not req:
            raise SystemExit(f"[v4] real run needs {name}")
    if not from_scratch and not a.trunk:
        raise SystemExit("[v4] real run needs --trunk (the v1 warm-start ckpt) OR "
                         "--from-scratch (random-init the trunk — the v4 from-scratch "
                         "fallback that trains WM+planner jointly, the v1 regime)")
    provenance = _assert_parity(a.train_cache, a.val_cache)

    device = a.device
    amp = device == "cuda"
    torch.manual_seed(a.seed)
    out_dir = Path(a.out or "flagship-v4-run"); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- model: v1 trunk (speed_input, action_dim=3), warm-started + trainable --
    cfg = flagship4b_config()
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    cfg.train.rollout_k = a.rollout_k                     # [PM] #2: v1 verbatim (4)
    plan = horizon_plan(cfg, op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20)
    world = WorldModel(cfg).to(device)
    grounding = build_grounding(world.state_dim, device=device)
    if from_scratch:
        # ⭐ v4 FROM-SCRATCH fallback: WorldModel(cfg) + build_grounding() already
        # random-initialize every trunk/grounding tensor. Training the WM and the
        # anchored-diffusion planner JOINTLY from this random init is exactly how v1
        # was trained (canary held 0.42) — it sidesteps the warm-start coupling
        # degradation (a prediction-converged v1 WM yanked off-manifold by the new
        # planner's gradient, canary 0.452 -> 1.10+). NO warm-start is the whole point;
        # the WM has no pre-converged optimum to protect, so it co-evolves with the
        # planner to a JOINT optimum (v1 is the existence proof this does not degrade).
        trunk_step = -1
        print("[v4][from-scratch] trunk (encoder+predictor) + grounding "
              "RANDOM-INITIALIZED — no v1 warm-start; WM + planner co-evolve to a "
              "joint optimum (the v1 training regime).", flush=True)
    else:
        trunk_step = _warmstart_trunk(world, grounding, a.trunk, device)

    # ---- operative planner head (v4_config; imagination off per real_smoke) -----
    hcfg = v4_config()
    hcfg.state_dim = world.state_dim
    hcfg.cond_imagination = False
    hcfg.window = cfg.predictor.window
    hcfg.ego_null_row = a.ego_null_row                   # P5b default True (X15 off)
    head = FlagshipV4Head(hcfg).to(device)
    if a.anchors_dense:
        anc = torch.load(a.anchors_dense, weights_only=False)
        head.load_anchors((anc["anchors"] if isinstance(anc, dict) else anc).to(device))
        print(f"[v4] loaded dense anchors from {a.anchors_dense}", flush=True)
    else:
        print("[v4] WARNING: no --anchors-dense — the operative fan uses the "
              "head's DEFAULT anchor buffer (fine for a smoke, NOT for a gate run).",
              flush=True)
    goal_head = (GoalScalarHead(GoalScalarConfig(in_dim=world.state_dim)).to(device)
                 if a.strategic != "off" else None)

    # ---- AdamW: trunk @ lr_trunk, {head + goal_head + grounding} @ lr_head -------
    head_group = list(head.parameters()) + list(grounding.parameters())
    if goal_head is not None:
        head_group += list(goal_head.parameters())
    opt = torch.optim.AdamW(
        [{"params": head_group, "lr": a.lr_head, "name": "head"},
         {"params": list(world.parameters()), "lr": a.lr_trunk, "name": "trunk"}],
        weight_decay=0.01)

    # ---- data: FlagshipV4Dataset over the parity split dirs ----------------------
    train_eps = [load_episode(str(p), mmap=True)
                 for p in sorted(Path(a.train_cache).glob("ep_*.pt"))]
    val_eps = [load_episode(str(p), mmap=True)
               for p in sorted(Path(a.val_cache).glob("ep_*.pt"))]
    if not train_eps or not val_eps:
        raise SystemExit(f"[v4] no ep_*.pt under {a.train_cache} / {a.val_cache} "
                         "— do the caches point at the SPLIT dirs?")
    dk = dict(window=cfg.predictor.window, max_horizon=plan.max_horizon,
              maneuver_h=plan.maneuver_h, channels=cfg.encoder.in_channels)
    ds_train = FlagshipV4Dataset(train_eps, **dk)
    ds_val = FlagshipV4Dataset(val_eps, **dk)
    dl = DataLoader(ds_train, batch_size=a.batch, shuffle=True, drop_last=True,
                    num_workers=a.workers, persistent_workers=a.workers > 0,
                    pin_memory=device == "cuda",
                    prefetch_factor=2 if a.workers else None)
    print(f"[data] train windows={len(ds_train)} val windows={len(ds_val)} "
          f"window={cfg.predictor.window} max_h={plan.max_horizon} "
          f"needed_fut={plan.needed_fut}", flush=True)

    phases = CurriculumPhases(a.phase_a_steps, a.phase_b_steps)
    lw = V4LossWeights(w_lat=a.lat_weight, w_lon=a.lon_weight, w_dist=a.dist_weight,
                       w_jerk=a.jerk_w, w_curv=a.curv_w,
                       w_strat=a.strat_scalar_weight)
    # ⭐ v4.2: cap-and-hold controller with a FLOOR (--lam-mult-floor). Baseline set
    # from the step-0 canary inside the loop; the floor is the fix for v4.1's
    # halve-to-zero planner starvation.
    controller = CanaryController(baseline=float("inf"), mult_floor=a.lam_mult_floor)
    milestones = tuple(m for m in sorted({a.gate_step, 5000, 15000, 20000, 30000})
                       if 0 < m <= a.steps)

    # ⭐ Sayed's HARD requirement (2026-07-23): the encoder AND predictor must be
    # TRAINED JOINTLY — NO frozen part. Gate the launch on it: every trunk param
    # requires_grad, sits in the optimizer's trunk group at lr_trunk>0. gnorm_trunk>0
    # in the first log lines then proves the trunk is actually updating (not frozen).
    trunk_report = _assert_trunk_trainable(world, opt, a.lr_trunk)

    (out_dir / "config.json").write_text(json.dumps({
        "arch": "flagship-v4 (joint WM + operative planner; λ_plan curriculum)"
                + (" — FROM-SCRATCH (random-init trunk, no v1 warm-start)"
                   if from_scratch else ""),
        "parity": provenance,
        "trunk": ({"init": "from-scratch (random)", "ckpt": None, "step": -1,
                   "rationale": "v4 from-scratch fallback — WM + anchored-diffusion "
                   "planner co-evolve from random init like v1 (canary held 0.42); no "
                   "warm-start coupling degradation. v1 is the existence proof."}
                  if from_scratch else
                  {"init": "warm-start", "ckpt": a.trunk, "step": trunk_step}),
        "from_scratch": from_scratch,
        "from_scratch_schedule_note": (
            "λ_plan/floor tension differs from-scratch: there is NO pre-converged WM to "
            "protect, so the canary controller is inert BY CONSTRUCTION — its baseline "
            "is the step-0 UNTRAINED canary (high), and a co-evolving WM's canary only "
            "IMPROVES below that, so delta<0 keeps the controller at 'ok' and λ_plan "
            "follows the pure schedule to 1.0. The floor (--lam-mult-floor) is a no-op "
            "safety net in this regime. DEFAULT keeps the v4.2b schedule (phase_a/b + "
            "floor) UNCHANGED so from-scratch changes exactly ONE thing vs v4.2b (trunk "
            "init) — maximal attributability. Phase A (λ_plan=0, steps 0..phase_a) is "
            "retained on purpose: it lets the random WM establish a predictive latent "
            "before the planner gradient couples in during the Phase B ramp."
            if from_scratch else None),
        "cfg": cfg.to_json(), "head_cfg": dataclasses.asdict(hcfg),
        "horizon_plan": {"needed_fut": plan.needed_fut,
                         "max_horizon": plan.max_horizon,
                         "maneuver_h": plan.maneuver_h},
        "phases": {"phase_a": phases.phase_a, "phase_b": phases.phase_b,
                   "gate_step": a.gate_step},
        "optimizer": {"kind": "AdamW", "lr_head": a.lr_head,
                      "lr_trunk": a.lr_trunk, "wd": 0.01, "warmup": a.warmup,
                      "schedule": "cosine", "micro_batch": a.batch,
                      "accum_steps": a.accum, "effective_batch": a.batch * a.accum,
                      "effective_batch_note": "matches v1 (16*4=64); v4.1 was 16*1=16"},
        "canary_controller": {
            "kind": "cap-and-hold-floor (v4.2 fix for v4.1 halve-to-zero)",
            "mult_floor": controller.mult_floor, "ctrl_thresh": controller.ctrl_thresh,
            "alarm_thresh": controller.alarm_thresh, "ctrl_factor": controller.ctrl_factor,
            "alarm_evals": controller.alarm_evals},
        "not_frozen_proof": trunk_report,
        "loss_weights": dataclasses.asdict(lw), "lambda_plan_mode": a.lambda_plan,
        "strategic": a.strategic, "milestones": list(milestones),
        "note_on_the_fly_labels": (
            "FlagshipV4Dataset mints v3 factorised + strategic labels PER WINDOW "
            "from full-episode poses; for a 30k run precompute them once "
            "(v4_labels.build) and index — --poses-*/--labels-* are reserved for "
            "that path and are NOT read by this on-the-fly loop yet."),
        "args": vars(a),
    }, indent=2, default=str), encoding="utf-8")

    # ⭐ THE MID-RUN HELD-OUT GATE. The probed episodes are a FIXED prefix of the
    # val split — the same episodes every probe, which is what makes the paired
    # episode-cluster bootstrap valid (and what WindowAlignmentError enforces).
    hgate, hg_eps = None, None
    if getattr(a, "heldout_gate", True):
        hgate = _HG.HeldoutGate(_HG.HeldoutGateConfig(
            every=a.heldout_every, episodes=a.heldout_episodes,
            stride=a.heldout_stride, n_boot=a.heldout_nboot,
            patience=a.heldout_patience, amp=amp,
            first_probe_step=a.heldout_every))
        hg_eps = val_eps[:a.heldout_episodes]
        print(f"[heldout-gate] ON — every {a.heldout_every} steps on "
              f"{len(hg_eps)} held-out episodes; primary={_HG.PRIMARY_NAME}; "
              f"patience={a.heldout_patience}. NOT gated on ade_0_2s.", flush=True)
    else:
        print("[heldout-gate] ⚠️ OFF — this run has NO held-out early-stop "
              "signal, the cause of the last run's ~29.5 wasted GPU-h.",
              flush=True)

    return _training_loop(
        out_dir=out_dir, device=device, amp=amp, world=world, grounding=grounding,
        head=head, goal_head=goal_head, opt=opt, plan=plan, cfg=cfg, phases=phases,
        lw=lw, controller=controller, dl=dl, ds_val=ds_val, steps=a.steps,
        log_every=a.log_every, eval_every=a.eval_every, save_every=a.save_every,
        warmup=a.warmup, lr_head=a.lr_head, lr_trunk=a.lr_trunk,
        gate_step=a.gate_step, lam_mode=a.lambda_plan,
        canary_horizons=(5, 10, 15, 20), canary_kmax=20,
        eval_episodes=a.eval_episodes, batch=a.batch, accum=a.accum,
        milestones=milestones, heldout_gate=hgate, heldout_episodes=hg_eps)


def _is_from_scratch(a) -> bool:
    """True when the run RANDOM-INITIALIZES the trunk instead of warm-starting v1.

    Triggered by ``--from-scratch`` OR the sentinel ``--trunk none`` (case-
    insensitive). This is the v4 from-scratch fallback: ``WorldModel(cfg)`` and
    ``build_grounding()`` already random-initialize every tensor, so "from scratch"
    is simply NOT calling :func:`_warmstart_trunk` — the WM + anchored-diffusion
    planner then co-evolve from random init exactly the way v1 was trained."""
    return bool(getattr(a, "from_scratch", False)
                or (a.trunk or "").strip().lower() == "none")


def _warmstart_trunk(world, grounding, trunk: str, device) -> int:
    """Load the v1 trunk (``flagship4b-speedjerk-30k``) into ``world`` + the v1
    grounding heads STRICT, keeping them TRAINABLE (v4 fine-tunes the trunk end-to-
    end). Refuses the no-speed ablation control (``flagship4b-phase0-30k``, 2.918 m)
    the way ``v15_prep.load_frozen_v1`` does — its near-identical name has inverted
    the lineage before (CLAUDE.md §Source of truth)."""
    ck = torch.load(trunk, map_location="cpu", weights_only=False)
    a_dim = ck["model"]["predictor.act_emb.0.weight"].shape[1]
    if a_dim != 3:
        raise SystemExit(
            f"REFUSING trunk {trunk}: predictor action_dim={a_dim}, not 3. v4 must "
            "sit on the SPEED arm (flagship4b-speedjerk-30k), NOT the no-speed "
            "ablation control flagship4b-phase0-30k.")
    world.load_state_dict(ck["model"])                   # STRICT
    grounding.load_state_dict(ck["grounding"])           # STRICT (canary needs it)
    step = int(ck.get("step", -1))
    print(f"[v4] warm-started trunk+grounding from {trunk} step={step} "
          f"(TRAINABLE)", flush=True)
    return step


def _assert_trunk_trainable(world, opt, lr_trunk: float) -> dict:
    """⭐ Sayed's HARD requirement (2026-07-23): the world model (encoder AND
    predictor) is trained JOINTLY — NO frozen part. Gate the launch on it, loudly,
    BEFORE a GPU-day is spent.

    Verifies, and returns as a report for ``config.json``:
      * every trunk param (``world.parameters()``) has ``requires_grad=True``;
      * the encoder AND the predictor are trainable (checked by name, since those
        are the two modules the requirement names explicitly);
      * every trunk param sits in the optimizer's ``trunk`` group at ``lr_trunk``,
        and ``lr_trunk > 0`` (a param at lr 0 is frozen in all but name).
    Raises ``SystemExit`` if the trunk is frozen — the launch must not proceed.
    ``gnorm_trunk > 0`` in the first log lines is the runtime confirmation that the
    trunk is actually *updating*; this static check makes the frozen case impossible
    to reach silently."""
    trunk_ids = {id(p) for p in world.parameters()}
    n_trunk = len(trunk_ids)
    n_req = sum(1 for p in world.parameters() if p.requires_grad)

    def _mod_trainable(prefix: str) -> tuple[int, int]:
        ps = [p for n, p in world.named_parameters() if n.startswith(prefix)]
        return sum(1 for p in ps if p.requires_grad), len(ps)

    enc_req, enc_n = _mod_trainable("encoder.")
    pred_req, pred_n = _mod_trainable("predictor.")

    trunk_grp = next((g for g in opt.param_groups if g.get("name") == "trunk"), None)
    in_opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    n_trunk_in_opt = len(trunk_ids & in_opt_ids)
    grp_lr = float(trunk_grp["lr"]) if trunk_grp is not None else 0.0

    ok = (n_req == n_trunk and enc_n > 0 and pred_n > 0
          and enc_req == enc_n and pred_req == pred_n
          and trunk_grp is not None and n_trunk_in_opt == n_trunk
          and grp_lr > 0.0 and float(lr_trunk) > 0.0)

    report = {
        "not_frozen": bool(ok),
        "trunk_params_total": n_trunk,
        "trunk_params_requires_grad": n_req,
        "encoder_params_requires_grad": f"{enc_req}/{enc_n}",
        "predictor_params_requires_grad": f"{pred_req}/{pred_n}",
        "trunk_params_in_optimizer_trunk_group": n_trunk_in_opt,
        "trunk_group_lr": grp_lr, "lr_trunk_arg": float(lr_trunk),
        "trunk_tensors_frozen": n_trunk - n_req,
    }
    print("[v4][not-frozen] " + json.dumps(report), flush=True)
    if not ok:
        raise SystemExit(
            "TRUNK FROZEN — Sayed's hard requirement is that the encoder AND "
            "predictor train jointly (NO frozen part). Refusing to launch: "
            + json.dumps(report))
    print(f"[v4][not-frozen] OK — all {n_trunk} trunk tensors require grad and sit "
          f"in the AdamW 'trunk' group at lr {grp_lr:g} (>0); encoder "
          f"{enc_req}/{enc_n} + predictor {pred_req}/{pred_n} trainable. "
          f"gnorm_trunk>0 in the first log rows confirms it is UPDATING.", flush=True)
    return report


# ============================================================================
# CPU full-loop smoke — proves the LOOP + checkpoint/resume + λ_plan controller
# + milestone archive on toy episodes across phases A/B/C, in seconds.
# ============================================================================

def smoke_loop(tmp_dir: str | None = None, heldout_gate=None,
               heldout_episodes=None) -> dict:
    """The acceptance proof for P4: run the real ``_training_loop`` on toy episodes
    (tiny config, ~6 steps spanning phases A/B/C), then show:

    * every step's total loss is finite (the loop raises otherwise);
    * the canary computes on toy data;
    * the λ_plan controller is DOWN-ONLY when the canary is forced to regress
      (a soft breach halves it; three hard breaches drive it to — and HOLD it at —
      the floor, NEVER to 0 and never up: the v4.2 cap-and-hold fix, so the planner
      always keeps a real gradient);
    * a ``ckpt_step<gate>.pt`` milestone archive appears;
    * checkpoint save -> resume is state-consistent: step advances and the
      controller multiplier is restored BIT-EXACT.
    """
    import dataclasses
    import tempfile

    from tanitad.config import flagship4b_smoke_config
    from tanitad.models.fourbrain import WorldModel
    from tanitad.models.strategic_goal import GoalScalarConfig, GoalScalarHead
    from flagship_v4_data import FlagshipV4Dataset

    torch.manual_seed(0)
    out_dir = Path(tmp_dir or tempfile.mkdtemp(prefix="v4smoke_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    def build():
        cfg = flagship4b_smoke_config()
        cfg.speed_input = True
        cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
        if getattr(cfg, "tactical_pred", None) is not None:
            cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
        cfg.train.rollout_k = 2
        plan = horizon_plan(cfg, op_fwd_k=2, tac_fwd_k=3, str_fwd_k=4)
        world = WorldModel(cfg)
        grounding = build_grounding(world.state_dim, hidden=32)
        head = FlagshipV4Head(_smoke_head_cfg(world.state_dim, cfg.predictor.window))
        goal_head = GoalScalarHead(GoalScalarConfig(in_dim=world.state_dim))
        head_group = (list(head.parameters()) + list(grounding.parameters())
                      + list(goal_head.parameters()))
        opt = torch.optim.AdamW(
            [{"params": head_group, "lr": 1e-4, "name": "head"},
             {"params": list(world.parameters()), "lr": 1e-4, "name": "trunk"}],
            weight_decay=0.01)
        return cfg, plan, world, grounding, head, goal_head, opt

    # toy episodes minted the same way as smoke(); FlagshipV4Dataset mints the v3 +
    # strategic labels per window so the goal head trains on real (non-IGNORE) rows.
    eps = [_toy_episode(60, i) for i in range(4)]

    cfg, plan, world, grounding, head, goal_head, opt = build()
    ds = FlagshipV4Dataset(eps, window=cfg.predictor.window,
                           max_horizon=plan.max_horizon, maneuver_h=plan.maneuver_h,
                           channels=cfg.encoder.in_channels, min_lookahead=8)
    dl = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True, num_workers=0)
    phases = CurriculumPhases(phase_a=1, phase_b=3)          # A/B/C within 6 steps
    lw = V4LossWeights()
    controller = CanaryController(baseline=float("inf"))
    ch, ck_max = (1, 2, 3, 4), 4
    gate = 3
    milestones = (gate,)
    # force a canary regression through the controller: soft breach then 3 hard
    # breaches (baseline is set to the measured step-0 canary inside the loop).
    override = None                                          # set after baseline

    # Run the loop once (fresh). We drive the controller with an override computed
    # from the measured baseline, so the "forced regression" is relative to the
    # real step-0 canary. Peek the baseline first via a bare canary.
    world.train(); grounding.train()
    base0 = canary_rollout(world, grounding, ds, "cpu", horizons=ch, k_max=ck_max,
                           episodes=4, stride=1, batch=4, amp=False)
    b = base0["canary_ade@2s"]
    override = [b + 0.10, b + 0.40, b + 0.40, b + 0.40, b + 0.40]

    res = _training_loop(
        out_dir=out_dir, device="cpu", amp=False, world=world, grounding=grounding,
        head=head, goal_head=goal_head, opt=opt, plan=plan, cfg=cfg, phases=phases,
        lw=lw, controller=controller, dl=dl, ds_val=ds, steps=6, log_every=1,
        eval_every=1, save_every=2, warmup=1, lr_head=1e-4, lr_trunk=1e-4,
        gate_step=gate, lam_mode="sched", canary_horizons=ch, canary_kmax=ck_max,
        eval_episodes=4, batch=4, milestones=milestones, canary_override=override,
        heldout_gate=heldout_gate,
        heldout_episodes=(eps[:2] if heldout_episodes is None
                          else heldout_episodes))

    # resume proof: fresh modules + controller, restore from ckpt.pt, check state.
    cfg2, plan2, world2, grounding2, head2, goal_head2, opt2 = build()
    controller2 = CanaryController(baseline=float("inf"))
    resumed_step = load_checkpoint_v4(
        Path(res["ckpt"]), world=world2, grounding=grounding2, head=head2,
        goal_head=goal_head2, opt=opt2, controller=controller2, device="cpu")

    mults = res["mult_trace"]
    down_only = all(b2 <= a2 + 1e-12 for a2, b2 in zip(mults, mults[1:]))
    floor = controller.mult_floor
    return {
        "out_dir": str(out_dir), "final_step": res["final_step"],
        "canary_baseline": controller.baseline, "canary_trace": res["canary_trace"],
        "mult_trace": mults, "controller_down_only": down_only,
        "mult_floor": floor,
        # ⭐ v4.2: under a forced canary regression the controller HOLDS at the floor
        # (never reaches 0 — the v4.1 starvation). The planner keeps a real gradient.
        "controller_held_at_floor": bool(mults and abs(mults[-1] - floor) < 1e-9),
        "controller_never_zero": bool(mults and min(mults) > 0.0),
        "milestone_archives": res["archives"],
        "milestone_present": f"ckpt_step{gate}.pt" in res["archives"],
        "resume": {"saved_step": res["final_step"], "resumed_step": resumed_step,
                   "step_advances": resumed_step == res["final_step"] + 1,
                   "mult_saved": float(controller._mult),
                   "mult_restored": float(controller2._mult),
                   "mult_bit_exact": float(controller._mult) == float(controller2._mult)},
        # the mid-run held-out gate's observable effect on the LOOP
        "early_stopped": res.get("early_stopped", False),
        "heldout_history": res.get("heldout_history", []),
        "best_ckpt_present": (out_dir / "ckpt_best.pt").exists(),
        "train_log": str(out_dir / "train_log.jsonl"),
    }


def _toy_episode(T: int, eid: int, size: int = 64):
    """A short kinematic toy episode (shared shape with smoke())."""
    from tanitad.data._contract import assemble_episode

    g = torch.Generator().manual_seed(100 + eid)
    frames = [torch.rand(1, size, size, generator=g) for _ in range(T)]
    rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, 8.0
    dt, yaw_rate = 0.1, (0.05 if eid % 2 else -0.05)
    accel = -1.0 if eid % 2 else 1.0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.0, v + accel * dt)
    poses = torch.tensor(rows)
    return assemble_episode(frames, [p.numpy() for p in poses],
                            [yaw_rate] * T, 0.1, eid)


# ============================================================================
# CLI (§16) — the exact launch surface; used to STAGE the command, not run it
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("train_flagship_v4", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # data / trunk / io (mirrors train_flagship_v16)
    ap.add_argument("--train-cache"); ap.add_argument("--val-cache")
    ap.add_argument("--poses-train"); ap.add_argument("--poses-val")
    ap.add_argument("--labels-train"); ap.add_argument("--labels-val")
    ap.add_argument("--trunk", help="warm-start ckpt: flagship4b-speedjerk-30k (v1); "
                    "pass the sentinel 'none' (or --from-scratch) to random-init instead")
    ap.add_argument("--from-scratch", dest="from_scratch", action="store_true",
                    help="⭐ v4 FROM-SCRATCH fallback: random-initialize the WorldModel "
                    "trunk (encoder+predictor) + grounding instead of warm-starting from "
                    "v1. Trains the WM and the anchored-diffusion planner JOINTLY from "
                    "random init — the way v1 was trained (canary held 0.42) — which "
                    "sidesteps the warm-start coupling degradation (a prediction-"
                    "converged v1 WM yanked off-manifold by the new planner's gradient). "
                    "--trunk is not required and is ignored; equivalently pass --trunk none.")
    ap.add_argument("--anchors-dense", help="operative 1..20 vocabulary (build_refc_anchors)")
    ap.add_argument("--anchors-coarse", help="tactical 5..50 vocabulary")
    ap.add_argument("--probes"); ap.add_argument("--out")
    ap.add_argument("--labels", choices=("v3", "v21"), default="v3")

    # --- the v4-specific surface (§16) ---
    ap.add_argument("--lambda-plan", choices=("0", "1", "sched"), default="sched")
    ap.add_argument("--phase-a-steps", type=int, default=2000)
    ap.add_argument("--phase-b-steps", type=int, default=8000)
    ap.add_argument("--strategic", choices=("full", "head", "off"), default="full")
    ap.add_argument("--d-strat", type=int, default=128)
    ap.add_argument("--long-horizon-k", type=int, default=50,
                    help="0 => no tactical instance / no 5 s terms")
    ap.add_argument("--probe-steps", type=int, default=50)
    ap.add_argument("--probe-grad", choices=("none", "one", "all"), default="one")
    ap.add_argument("--dense-plan", dest="dense_plan", action="store_true", default=True)
    ap.add_argument("--no-dense-plan", dest="dense_plan", action="store_false")
    ap.add_argument("--lat-weight", type=float, default=0.05)
    ap.add_argument("--lon-weight", type=float, default=0.05)
    ap.add_argument("--dist-weight", type=float, default=0.05)
    ap.add_argument("--strat-goal-weight", type=float, default=0.1)
    ap.add_argument("--strat-pred-weight", type=float, default=0.5)
    ap.add_argument("--strat-scalar-weight", type=float, default=0.05)
    ap.add_argument("--jerk-w", type=float, default=0.02)
    ap.add_argument("--curv-w", type=float, default=0.01)
    # P5b: learned null row is the default; the zero-fill is X15 (ablation only)
    ap.add_argument("--ego-null-row", dest="ego_null_row", action="store_true", default=True)
    ap.add_argument("--ego-zero-fill", dest="ego_null_row", action="store_false",
                    help="X15 — the v3enc zero-fill bug; ablation ONLY, never a shipping run")
    ap.add_argument("--rollout-k", type=int, default=4,
                    help="v1 verbatim; [PM] #2 — do NOT raise before speed_benefit_recovered_frac unlocks it")
    ap.add_argument("--s2-film", action="store_true", help="0-for-4 seam family; pre-registered A/B only")

    # --- ⭐ THE MID-RUN HELD-OUT GATE (v5 prep §1.6) ---------------------------
    # Cause #1 of the last run's failure: NO held-out early-stop signal. MEASURED
    # ~29.5 GPU-h — half the run — training past the best checkpoint while every
    # training term improved. ON BY DEFAULT; turning it off takes an explicit
    # flag and trips the preflight, because the default must be the safe one.
    ap.add_argument("--heldout-gate", dest="heldout_gate", action="store_true",
                    default=True,
                    help="probe the DEPLOYABLE surface on held-out episodes under "
                         "pseudo-simulation every --heldout-every steps and STOP "
                         "when the held-out primary (the MAP-FREE COMPOSITE, not "
                         "ade_0_2s) is separated-worse for two consecutive probes")
    ap.add_argument("--no-heldout-gate", dest="heldout_gate", action="store_false",
                    help="⚠️ disables the early-stop that exists because ~29.5 "
                         "GPU-h were burned without it; trips preflight_asserts")
    ap.add_argument("--heldout-every", type=int, default=2000,
                    help="probe cadence in steps (fixed, never data-dependent)")
    ap.add_argument("--heldout-episodes", type=int, default=8,
                    help="held-out episodes per probe — the bootstrap's resampling "
                         "unit, so this is the gate's statistical power")
    ap.add_argument("--heldout-patience", type=int, default=2,
                    help="consecutive separated-worse probes that stop the run. "
                         "1 would stop on a single sample; this program's standing "
                         "rule is that an isolated read is not a refutation")
    ap.add_argument("--heldout-stride", type=int, default=8)
    ap.add_argument("--heldout-nboot", type=int, default=2000)

    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--gate-step", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=16, help="micro-batch (per forward)")
    ap.add_argument("--accum", type=int, default=4,
                    help="gradient-accumulation micro-batches per optimizer step. "
                         "v4.2 default 4 => EFFECTIVE batch = batch*accum = 16*4 = 64, "
                         "matching v1 (registry §1.2: --batch-size 16 --accum 4). "
                         "v4.1 ran accum 1 = effective 16 (4x too small, noisier grads).")
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-trunk", type=float, default=1e-4,
                    help="v4.2 default 1e-4 — BETWEEN v4's 3e-4 (degraded the WM: canary "
                         "0.42->1.3 by step 3500) and v4.1's 3e-5 (starved the planner). "
                         "Deviates from V4_DESIGN O-14's 3e-4 by Sayed's 2026-07-23 decision.")
    ap.add_argument("--lam-mult-floor", dest="lam_mult_floor", type=float, default=0.25,
                    help="⭐ v4.2 cap-and-hold FLOOR on the canary controller's λ_plan "
                         "multiplier: the planner→trunk gradient is never reduced below this, "
                         "so the planner is never starved (the v4.1 halve-to-zero bug). "
                         "A §14.4 knob; 0.25 keeps >=1/4 of the coupling while letting the "
                         "controller cut 3/4 if the WM is threatened. Must be in (0, 1].")
    # --- loop cadence (mirrors train_flagship_v16) ---
    ap.add_argument("--warmup", type=int, default=2000, help="cosine LR warmup steps")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500,
                    help="canary + planner eval AND the λ_plan controller update")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--eval-episodes", type=int, default=40,
                    help="val episodes for the in-loop eval and the canary")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="run the CPU loss-assembly smoke and exit")
    ap.add_argument("--smoke-loop", action="store_true",
                    help="run the full-loop CPU smoke (loop + checkpoint/resume + "
                         "λ_plan controller + milestone archive) and exit")
    ap.add_argument("--real-smoke", action="store_true",
                    help="run v4_loss_step on REAL parity windows (proves the "
                         "factorised + strategic-scalar heads train on real "
                         "labels) and exit; needs --train-cache")
    ap.add_argument("--n-windows", type=int, default=8, help="real-smoke batch size")
    ap.add_argument("--print-launch", action="store_true",
                    help="print the staged pod launch command and the preflight gates, then exit")
    return ap


def preflight_asserts(a) -> list[str]:
    """The §17.1 / §9 invariants a launch must satisfy — checked here so a bad
    config fails loudly BEFORE any GPU-day is spent."""
    problems = []
    phases = CurriculumPhases(a.phase_a_steps, a.phase_b_steps)
    # ⭐ CAUSE #1 of the last run's failure, made un-repeatable-by-accident.
    # ~29.5 GPU-h — half the v4 30k run — went into training past the best
    # checkpoint because there was NO held-out early-stop signal. Launching
    # without the gate is now a deliberate, visible act.
    if not getattr(a, "heldout_gate", True):
        problems.append(
            "[HELDOUT-GATE] --no-heldout-gate: this run would have NO held-out "
            "early-stop signal. MEASURED cost of that on the v4 30k run: ~29.5 "
            "GPU-h, half the run, spent training past the best checkpoint while "
            "every training term improved. Drop the flag, or record the reason.")
    if getattr(a, "heldout_gate", True):
        if a.heldout_every <= 0 or a.heldout_every > a.steps:
            problems.append(
                f"[HELDOUT-GATE] --heldout-every {a.heldout_every} never fires "
                f"within --steps {a.steps}: the gate would be present and inert, "
                f"which is worse than absent because it looks like cover.")
        if a.steps // max(a.heldout_every, 1) < a.heldout_patience + 1:
            problems.append(
                f"[HELDOUT-GATE] --steps {a.steps} / --heldout-every "
                f"{a.heldout_every} allows only {a.steps // max(a.heldout_every, 1)} "
                f"probes, but the rule needs an incumbent plus "
                f"{a.heldout_patience} consecutive challengers to ever stop.")
        if a.heldout_patience < 2:
            problems.append(
                f"[HELDOUT-GATE] --heldout-patience {a.heldout_patience} would "
                f"stop on a SINGLE separated probe. An isolated read is not a "
                f"refutation (standing rule); patience must be >= 2.")
        if a.heldout_episodes < 4:
            problems.append(
                f"[HELDOUT-GATE] --heldout-episodes {a.heldout_episodes} is the "
                f"bootstrap's resampling unit; below 4 the interval is not "
                f"informative and the gate will read 'not separated' forever "
                f"— an UNPOWERED gate, which is the silent-disable failure.")
    # ⭐ from-scratch XOR warm-start: a real --trunk together with --from-scratch is
    # ambiguous (the trunk would be built then discarded by the random init). Fail
    # loudly so the intent is unmistakable BEFORE a GPU-day.
    if _is_from_scratch(a) and a.trunk and a.trunk.strip().lower() != "none":
        problems.append(
            f"--from-scratch is set but --trunk {a.trunk!r} was also given — "
            "from-scratch RANDOM-INITIALIZES the trunk and ignores --trunk. Pass "
            "exactly one: --from-scratch (random init) XOR --trunk <v1 ckpt>.")
    if a.gate_step < phases.phase_b:
        problems.append(f"O-17 VIOLATION: gate step {a.gate_step} < phase_b "
                        f"{phases.phase_b} — the gate would fall inside the ramp.")
    if not a.ego_null_row:
        problems.append("X15 VIOLATION: --ego-zero-fill is set — the v3enc bug is "
                        "forbidden in a shipping run.")
    if a.rollout_k != 4:
        problems.append(f"[PM] #2: --rollout-k {a.rollout_k} != 4 before "
                        f"speed_benefit_recovered_frac unlocks it.")
    if not 0.0 < a.lam_mult_floor <= 1.0:
        problems.append(f"--lam-mult-floor {a.lam_mult_floor} not in (0, 1] — a floor "
                        "of 0 IS the v4.1 halve-to-zero starvation this fix removes.")
    if a.lr_trunk <= 0.0:
        problems.append(f"--lr-trunk {a.lr_trunk} <= 0 would FREEZE the trunk; v4 trains "
                        "the encoder+predictor jointly (Sayed's hard requirement).")
    if a.accum < 1:
        problems.append(f"--accum {a.accum} < 1 is invalid.")
    if a.batch * a.accum != 64:
        problems.append(f"effective batch {a.batch}*{a.accum}={a.batch * a.accum} != 64 "
                        "— v4.2 must match v1's effective batch 64 (registry §1.2). "
                        "Override intentionally only if you know why.")
    # encoder-touching lever count: λ_plan (1) + strategic (2) = 2 of 2, door CLOSED
    return problems


def _staged_command(a) -> str:
    """Reconstruct the exact pod launch command from the parsed args (§16). Printed
    by ``--print-launch`` for the orchestrator to run — the trainer NEVER launches
    it. PYTHONPATH is REQUIRED on the pod or the trainer dies with
    ``ModuleNotFound: tanitad`` (CLAUDE.md traps preflight)."""
    from_scratch = _is_from_scratch(a)
    parts = ["PYTHONPATH=/workspace/TanitAD/stack python3 scripts/train_flagship_v4.py"]
    pairs = [("--train-cache", a.train_cache), ("--val-cache", a.val_cache)]
    if not from_scratch:                     # from-scratch random-inits: NO --trunk
        pairs.append(("--trunk", a.trunk))
    pairs += [
            ("--anchors-dense", a.anchors_dense),
            ("--anchors-coarse", a.anchors_coarse), ("--out", a.out),
            ("--labels", a.labels), ("--lambda-plan", a.lambda_plan),
            ("--phase-a-steps", a.phase_a_steps), ("--phase-b-steps", a.phase_b_steps),
            ("--strategic", a.strategic), ("--long-horizon-k", a.long_horizon_k),
            ("--steps", a.steps), ("--gate-step", a.gate_step), ("--batch", a.batch),
            ("--accum", a.accum),
            ("--lr-head", a.lr_head), ("--lr-trunk", a.lr_trunk),
            ("--lam-mult-floor", a.lam_mult_floor),
            ("--warmup", a.warmup), ("--workers", a.workers),
            ("--eval-every", a.eval_every), ("--save-every", a.save_every),
            ("--eval-episodes", a.eval_episodes), ("--rollout-k", a.rollout_k),
            # ⭐ the mid-run held-out gate must appear in the STAGED command, or a
            # launch reconstructed from this string silently runs without the
            # early-stop — i.e. reproduces cause #1 by omission. Its cadence and
            # patience are part of the pre-registration, not incidental tuning.
            ("--heldout-every", a.heldout_every),
            ("--heldout-episodes", a.heldout_episodes),
            ("--heldout-patience", a.heldout_patience),
            ("--heldout-stride", a.heldout_stride),
            ("--heldout-nboot", a.heldout_nboot),
            ("--seed", a.seed), ("--device", a.device)]
    for flag, val in pairs:
        if val is not None and val != "":
            parts.append(f"{flag} {val}")
    if from_scratch:
        parts.append("--from-scratch")       # random-init the trunk (the v4 fallback)
    if not a.ego_null_row:
        parts.append("--ego-zero-fill")     # X15 ablation ONLY (preflight blocks it)
    # the gate's on/off state must be EXPLICIT in the staged string either way —
    # "the flag was absent" must never be how a run loses its early-stop.
    parts.append("--heldout-gate" if getattr(a, "heldout_gate", True)
                 else "--no-heldout-gate")
    return " ".join(parts)


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.smoke:
        out = smoke()
        print("[v4] smoke OK — joint step finite across phases A/B/C:")
        for step, log in out["logs"]:
            print(f"  step {step:>2}: lambda_plan={log['lambda_plan']} "
                  f"total={log['total']:.4f} wm={log['wm']:.4f} "
                  f"planner={log['planner']:.4f} plan_ade={log['plan_ade']:.4f}")
        return 0
    if a.smoke_loop:
        out = smoke_loop()
        print("[v4] smoke-loop OK: full loop + checkpoint/resume + lambda_plan "
              "controller + milestone archive on toy episodes:")
        print(json.dumps(out, indent=2))
        assert out["controller_down_only"], out["mult_trace"]
        assert out["controller_held_at_floor"], out["mult_trace"]
        assert out["controller_never_zero"], out["mult_trace"]
        assert out["milestone_present"], out["milestone_archives"]
        r = out["resume"]
        assert r["step_advances"] and r["mult_bit_exact"], r
        return 0
    if a.real_smoke:
        if not a.train_cache:
            raise SystemExit("[v4] --real-smoke needs --train-cache <epcache split dir>")
        # --from-scratch (or --trunk none) => no warm-start: the real-smoke then proves
        # the FROM-SCRATCH loss step (WM + planner + heads train from random init).
        real_smoke(a.train_cache, n_windows=a.n_windows,
                   trunk=(None if _is_from_scratch(a) else a.trunk), seed=a.seed)
        return 0
    if a.print_launch:
        problems = preflight_asserts(a)
        print("=== flagship v4 - STAGED launch (NOT executed; Sayed owns go/no-go) ===")
        if _is_from_scratch(a):
            print("trunk init: FROM-SCRATCH (random) -- NO v1 warm-start; WM + "
                  "anchored-diffusion planner co-evolve jointly like v1 (sidesteps the "
                  "warm-start coupling degradation; canary starts high/untrained). The "
                  "canary controller is inert by construction (baseline = untrained "
                  "canary); floor + schedule kept identical to v4.2b for attributability.")
        else:
            print(f"trunk init: warm-start from {a.trunk}")
        print(f"parity: {PARITY_KEY} / skip-hash {PARITY_SKIP_HASH} (episodes must not re-select)")
        print(f"phases: A[0,{a.phase_a_steps}) B[{a.phase_a_steps},{a.phase_b_steps}) "
              f"C[{a.phase_b_steps},{a.steps})  |  gate at {a.gate_step}")
        print(f"levers: lambda_plan + strategic = 2 of 2 encoder-touching (door CLOSED)")
        print(f"schedule fix: lr_trunk={a.lr_trunk} (v4 3e-4 / v4.1 3e-5) | cap-and-hold "
              f"lam_mult_floor={a.lam_mult_floor} (v4.1 halved to ~0)")
        print(f"same-as-v1: micro-batch {a.batch} x accum {a.accum} = EFFECTIVE {a.batch * a.accum} "
              f"(v1=64; v4.1 was 16)")
        print("staged command (run on the pod, NOT here):")
        print("  " + _staged_command(a))
        print("PREFLIGHT:", "OK" if not problems else "BLOCKED")
        for p in problems:
            print("  -", p)
        return 0 if not problems else 2
    # --- the real run (§17): gate on the preflight, THEN train. This code path is
    # correct and complete; a launch is Sayed's go, executed by the orchestrator on
    # the pod — never from an agent (RETRACTION_LOG C1: a launch is not a completion).
    problems = preflight_asserts(a)
    if problems:
        print("PREFLIGHT: BLOCKED")
        for p in problems:
            print("  -", p)
        return 2
    train(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
