"""Shared 4-brain joint-loss assembly (D-030 recovery).

This is the ONE loss body the flagship trainer (`scripts/train_flagship4b.py`)
and a future `scripts/refa_train` (4-brain variant) both call, so the flagship
and REF-A differ ONLY in the encoder (from-scratch ViT vs frozen-DINO adapter)
and the SIGReg target (full latent, position-relaxed vs predictor outputs only)
— never in the brains or their training objective.

The caller ENCODES the state window itself (frames -> ViT+readout for the
flagship, feature grids -> adapter for REF-A) and passes the encoded
``states``/``fut_states``; everything here operates on the compact STATE, so the
encoder axis is isolated. H15 imagination is intentionally NOT here — it is
encoder/token-grid specific and is added by each trainer on top (the shared list
is: JEPA + hierarchical grounding + maneuver CE + route CE + SIGReg-relaxed +
inv-dyn).

L = pred_weight   * JEPA(operative, intent-conditioned)
  + tacpred_weight * JEPA(tactical-predictor dynamics, if present)
  + roll_weight   * K-step recursive rollout (if rollout_k>1)
  + goal_weight   * JEPA(tactical GOAL latent @ 2 s)
  + wp_weight     * tactical GOAL waypoint L2 (grounded ego sub-waypoints)
  + man_weight    * maneuver CE (class-weighted)
  + route_weight  * strategic route-heading CE (valid-masked, class-weighted)
  + hierarchical metric grounding (op/tac/str: invdyn_weight, fwd_weight)
  + sigreg_weight * SIGReg  (variant-selected)
  + inv_weight    * action inverse-dynamics (A5)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from tanitad.config import StackConfig
from tanitad.models.metric_dynamics import (HierarchicalGrounding,
                                            grounding_losses, gt_ego_waypoints)
from tanitad.models.predictor import change_weighted_mse
from tanitad.models.sigreg import position_relaxed
from tanitad.train.train_worldmodel import _pred_losses, _rollout_loss


# --------------------------------------------------------------------------- #
# Horizon plan — ONE source of truth for encoded-future indices + max_horizon  #
# --------------------------------------------------------------------------- #
@dataclass
class HorizonPlan:
    level_cfg: dict[str, tuple[tuple[int, ...], int]]   # level -> (invdyn h, fwd_k)
    goal_h: int                                          # tactical goal horizon
    maneuver_h: int
    needed_fut: list[int]                                # future frame idxs to ENCODE
    idx_of: dict[int, int]
    max_horizon: int                                     # future_poses/actions span


def horizon_plan(cfg: StackConfig, op_fwd_k: int, tac_fwd_k: int,
                 str_fwd_k: int, maneuver_h: int | None = None) -> HorizonPlan:
    """Compute the encoded-future indices, per-level grounding horizons, and the
    ``max_horizon`` the dataset must supply. Used by BOTH the trainer (dataset
    sizing) and the loss (fut_states indexing) so they can never disagree."""
    assert cfg.tactical_policy is not None and cfg.strategic_policy is not None
    op_h = tuple(cfg.predictor.horizons)
    tac_h = (tuple(cfg.tactical_pred.horizons) if cfg.tactical_pred is not None
             else tuple(cfg.tactical_policy.waypoint_horizons))
    wp_h = tuple(cfg.tactical_policy.waypoint_horizons)
    goal_h = max(wp_h)
    str_h = (goal_h,)                                    # long-horizon coarse pair
    level_cfg = {"op": (op_h, op_fwd_k), "tac": (tac_h, tac_fwd_k),
                 "str": (str_h, str_fwd_k)}
    if maneuver_h is None:
        maneuver_h = goal_h
    # Future frames to ENCODE: JEPA targets (k-1, change-ref k-2), tactical-pred
    # targets, grounding metric-invdyn pairs (k-1), the goal latent (goal_h-1 AND
    # its change-weight reference goal_h-2, same as the op/tacpred JEPA targets),
    # and the K-step rollout range.
    idxs: set[int] = set()
    for k in op_h:
        idxs |= {k - 1} | ({k - 2} if k >= 2 else set())
    if cfg.tactical_pred is not None:
        for k in cfg.tactical_pred.horizons:
            idxs |= {k - 1} | ({k - 2} if k >= 2 else set())
    for hs, _ in level_cfg.values():
        idxs |= {k - 1 for k in hs}
    # The tactical GOAL latent loss is change-weighted, so it reads BOTH the goal
    # target (goal_h-1) and its previous state (goal_h-2) — encode both, mirroring
    # the op/tacpred pattern above. (Dormant at smoke where goal_h-2 is already a
    # tacpred target; at the real config goal_h=20 needs index 18 encoded.)
    idxs |= {goal_h - 1} | ({goal_h - 2} if goal_h >= 2 else set())
    if getattr(cfg.train, "rollout_k", 1) > 1:
        idxs |= set(range(cfg.train.rollout_k))
    needed = sorted(idxs)
    idx_of = {i: j for j, i in enumerate(needed)}
    max_h = max([goal_h, maneuver_h, op_fwd_k, tac_fwd_k, str_fwd_k]
                + list(op_h) + list(tac_h) + list(str_h))
    return HorizonPlan(level_cfg, goal_h, maneuver_h, needed, idx_of, max_h)


# --------------------------------------------------------------------------- #
# Grounding heads                                                              #
# --------------------------------------------------------------------------- #
def build_grounding(state_dim: int, hidden: int = 512,
                    device="cpu") -> HierarchicalGrounding:
    """The op/tac/str metric-dynamics heads (kept OUTSIDE the model, saved under
    their own ckpt key). State-dim-agnostic — same call for flagship and REF-A."""
    return HierarchicalGrounding(state_dim, hidden=hidden).to(device)


def _class_weighted_ce(logits: Tensor, target: Tensor, n_classes: int,
                       clamp: float = 10.0) -> Tensor:
    """Cross-entropy with inverse per-batch class-frequency weights (clamped) —
    the highway corpora are maneuver/route class-imbalanced (lane_keep / follow
    dominate), so the rare turn classes would otherwise be drowned."""
    counts = torch.bincount(target, minlength=n_classes).float()
    w = (target.numel() / (n_classes * counts.clamp_min(1.0))).clamp(max=clamp)
    return F.cross_entropy(logits, target, weight=w.to(logits.dtype))


# --------------------------------------------------------------------------- #
# The shared joint loss                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class LossWeights:
    pred: float = 1.0
    tacpred: float = 0.5
    roll: float = 0.5
    goal: float = 0.5           # tactical GOAL latent JEPA
    wp: float = 1.0             # tactical GOAL waypoint L2
    man: float = 0.5            # maneuver CE
    route: float = 0.5          # strategic route CE
    invdyn: float = 2.0         # hierarchical metric-invdyn (per level)
    fwd: float = 1.0            # hierarchical forward-consistency (per level)
    sigreg: float = 0.1
    inv: float = 0.5            # action inverse-dynamics (A5)


def flagship_loss(model, grounding: HierarchicalGrounding, batch: dict,
                  states: Tensor, fut_states: Tensor, plan: HorizonPlan,
                  cfg: StackConfig, *, weights: LossWeights,
                  sigreg_variant: str, sigreg_free_dims: int,
                  pose_scale: float, fwd_step_weight: float, device,
                  change_weighted: bool | None = None
                  ) -> tuple[Tensor, dict, dict]:
    """The shared 4-brain joint loss (all brains + hierarchical grounding +
    SIGReg-relaxed), computed on already-encoded states.

    ``sigreg_variant``: ``"full_relaxed"`` (flagship — SIGReg on the full latent
    AND predictions, with the first ``sigreg_free_dims`` columns exempt, §B.3) or
    ``"pred_only"`` (REF-A — SIGReg on predictor outputs only; frozen features
    need no anti-collapse). Returns ``(total, log, parts)``; ``parts`` holds the
    raw loss tensors (for gradient-reach tests)."""
    actions = batch["actions"].to(device)                 # [B, W, A]
    fut_actions = batch.get("future_actions")
    fut_actions = fut_actions.to(device) if fut_actions is not None else None
    future_poses = batch["future_poses"].to(device).float()   # [B, Hmax, 4]
    pose_last = batch["pose_last"].to(device).float()         # [B, 4]
    nav_cmd = batch["nav_cmd"].to(device)                 # [B] long (derived)
    nav_valid = batch["nav_valid"].to(device)             # [B] bool
    route_tgt = batch["route_target"].to(device)          # [B] long
    man_tgt = batch["maneuver_label"].to(device)          # [B] long
    idx_of = plan.idx_of
    cw = cfg.predictor.change_weighted if change_weighted is None else change_weighted
    z_t = states[:, -1]

    # ---- the hierarchy: strategic ctx --FiLM--> tactical intent --FiLM--> op --
    strat = model.strategic_policy(states, nav_cmd)
    tac = model.tactical_policy(states, strat["ctx"])
    preds = model.predictor(states, actions, intent=tac["intent"])

    # ---- JEPA (operative + tactical-predictor dynamics + K-step rollout) ------
    loss_pred = _pred_losses(preds, cfg.predictor.horizons, z_t, fut_states,
                             idx_of, cw)
    loss_tacpred = torch.zeros((), device=device)
    if model.tactical_pred is not None:
        tac_preds = model.tactical_pred(states, actions)
        loss_tacpred = _pred_losses(tac_preds, cfg.tactical_pred.horizons, z_t,
                                    fut_states, idx_of, cw)
    loss_roll = torch.zeros((), device=device)
    K = getattr(cfg.train, "rollout_k", 1)
    if K > 1:
        loss_roll = _rollout_loss(model, states, actions, fut_states,
                                  fut_actions, idx_of, K)

    # ---- tactical GOAL: 2 s sub-waypoints (grounded L2) + goal latent (JEPA) ---
    wp_h = cfg.tactical_policy.waypoint_horizons
    wp_pred = torch.stack([tac["waypoints"][k] for k in wp_h], dim=1)   # [B,H,2]
    wp_tgt = gt_ego_waypoints(pose_last, future_poses, wp_h)
    loss_wp = ((wp_pred - wp_tgt) / pose_scale).pow(2).mean()
    goal_tgt = fut_states[:, idx_of[plan.goal_h - 1]]
    prev_goal = fut_states[:, idx_of[plan.goal_h - 2]] if plan.goal_h >= 2 else z_t
    loss_goal = (change_weighted_mse(tac["target_latent"], goal_tgt, prev_goal)
                 if cw else (tac["target_latent"] - goal_tgt).pow(2).mean())

    # ---- maneuver CE (class-weighted) + strategic route CE (masked, weighted) -
    loss_man = _class_weighted_ce(tac["maneuver_logits"], man_tgt,
                                  cfg.tactical_policy.n_maneuvers)
    man_acc = (tac["maneuver_logits"].argmax(-1) == man_tgt).float().mean()
    n_route = cfg.strategic_policy.n_route
    if bool(nav_valid.any()):
        tv = route_tgt[nav_valid]
        loss_route = _class_weighted_ce(strat["route_logits"][nav_valid], tv,
                                        n_route)
        route_acc = (strat["route_logits"][nav_valid].argmax(-1)
                     == tv).float().mean()
    else:
        loss_route = torch.zeros((), device=device)
        route_acc = torch.zeros((), device=device)

    # ---- hierarchical metric grounding (op/tac/str) ---------------------------
    loss_ground, g_parts, g_log = grounding_losses(
        grounding, model.predictor, states, fut_states, actions, fut_actions,
        pose_last, future_poses, idx_of, plan.level_cfg, pose_scale,
        invdyn_weight=weights.invdyn, fwd_weight=weights.fwd,
        fwd_step_weight=fwd_step_weight)

    # ---- SIGReg (variant) -----------------------------------------------------
    z_pred_all = torch.cat([preds[k] for k in cfg.predictor.horizons])
    if sigreg_variant == "full_relaxed":
        z_all = torch.cat([states.reshape(-1, states.shape[-1]),
                           fut_states.reshape(-1, states.shape[-1])])
        loss_sig = (position_relaxed(model.sigreg, z_all, sigreg_free_dims)
                    + position_relaxed(model.sigreg, z_pred_all, sigreg_free_dims))
    elif sigreg_variant == "pred_only":
        loss_sig = model.sigreg(z_pred_all)
    else:
        raise ValueError(f"unknown sigreg_variant {sigreg_variant!r}")

    # ---- action inverse-dynamics (A5) -----------------------------------------
    a_hat = model.inv_dyn(states[:, -2], states[:, -1])
    loss_inv = (a_hat - actions[:, -2]).pow(2).mean()

    total = (weights.pred * loss_pred
             + weights.tacpred * loss_tacpred
             + weights.roll * loss_roll
             + weights.goal * loss_goal
             + weights.wp * loss_wp
             + weights.man * loss_man
             + weights.route * loss_route
             + loss_ground                       # weights applied inside
             + weights.sigreg * loss_sig
             + weights.inv * loss_inv)

    parts = {"pred": loss_pred, "tacpred": loss_tacpred, "roll": loss_roll,
             "goal": loss_goal, "wp": loss_wp, "man": loss_man,
             "route": loss_route, "sigreg": loss_sig, "inv": loss_inv,
             "ground": loss_ground, **g_parts}
    log = {"pred": loss_pred.item(), "tacpred": loss_tacpred.item(),
           "roll": loss_roll.item(), "goal": loss_goal.item(),
           "wp": loss_wp.item(), "man": loss_man.item(),
           "route": loss_route.item(), "man_acc": round(float(man_acc), 4),
           "route_acc": round(float(route_acc), 4),
           "sigreg": loss_sig.item(), "inv": loss_inv.item(),
           "ground": loss_ground.item(),
           "nav_valid_frac": round(float(nav_valid.float().mean()), 4),
           **g_log}
    return total, log, parts
