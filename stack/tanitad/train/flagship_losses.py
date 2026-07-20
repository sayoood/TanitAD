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
from tanitad.train.decorr import ego_decorr_loss, ego_linear_r2
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


def anchor_tactical_loss(tac: dict, anchors: Tensor, wp_tgt: Tensor,
                         pose_scale: float) -> tuple[Tensor, Tensor, Tensor]:
    """The TIME-anchored (DiffusionDrive/REF-C) tactical objective — replaces the
    unimodal wp L2 when ``v2_anchor_tactical`` is on. ONE implementation shared by
    the flagship loss and REF-A's four-brain loss (both arms hold the SAME
    :class:`~tanitad.models.fourbrain.AnchoredTacticalDecoder`).

    The GT 2 s ego waypoints ``wp_tgt`` [B, S, 2] are assigned to their NEAREST
    anchor (flattened L2 over the fixed vocabulary ``anchors`` [N, S, 2]); the
    anchor-cls CE classifies that index and the winner-takes-all L1 regresses ONLY
    the assigned anchor's refined trajectory (``tac["anchor_traj"]``). Regressing
    the assigned anchor ALONE — never pulling every anchor to GT — is what
    preserves multimodality (mirrors scripts/refc_train's 1.0/1.0 CE/L1 recipe).
    The WTA L1 is pose-scaled so the returned ``loss_wp`` stays comparable to the
    old unimodal wp term (same ``weights.wp`` knob). Returns
    ``(loss_wp := cls + wta, loss_cls, loss_wta)``."""
    b = wp_tgt.shape[0]
    a = anchors.to(wp_tgt.dtype)                              # [N, S, 2]
    dist = ((wp_tgt[:, None] - a[None]) ** 2).sum(dim=(-1, -2))   # [B, N]
    a_star = dist.argmin(dim=1)                              # [B]
    loss_cls = F.cross_entropy(tac["anchor_logits"], a_star)
    recon = tac["anchor_traj"][torch.arange(b, device=wp_tgt.device), a_star]
    loss_wta = ((recon - wp_tgt) / pose_scale).abs().mean()
    return loss_cls + loss_wta, loss_cls, loss_wta


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
    route_vis: float = 0.3      # LEVER A: nav-zeroed route-from-vision aux CE
    invdyn: float = 2.0         # hierarchical metric-invdyn (per level)
    fwd: float = 1.0            # hierarchical forward-consistency (per level)
    sigreg: float = 0.1
    inv: float = 0.5            # action inverse-dynamics (A5)
    decorr: float = 0.05        # LEVER B: encoder<->ego linear decorrelation


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

    # ---- v2 levers 1+5: ego vector to the planning brains + nav dropout -------
    # ego = [v0/pose_scale, yr0] from OBSERVED poses only (t and t-1) — the
    # tactical wp heads were speed-starved (3.38 m vs operative 0.628).
    ego = None
    ego_full = None                             # LEVER B: UNdropped ego (decorr tgt)
    keep_mask = None                            # shared ego/v0 dropout mask
    if getattr(cfg, "v2_ego_to_planners", False):
        v0n = pose_last[:, 3:4] / pose_scale
        pose_prev = batch.get("pose_prev")
        if pose_prev is not None:
            pp = pose_prev.to(device).float()
            dyaw = pose_last[:, 2] - pp[:, 2]
            yr0 = torch.atan2(torch.sin(dyaw), torch.cos(dyaw)) / 0.1
        else:                                   # old cache path: no t-1 pose
            yr0 = torch.zeros_like(pose_last[:, 2])
        ego = torch.cat([v0n, yr0.unsqueeze(1)], dim=1)          # [B, 2]
        ego_full = ego                          # capture BEFORE dropout: the TRUE
                                                # fed dynamics the encoder must not
                                                # re-encode (decorr target, LEVER B)
        p_ed = float(getattr(cfg, "v2_ego_dropout", 0.0))
        if model.training and p_ed > 0.0:      # shortcut guard (ChauffeurNet)
            keep_mask = (torch.rand(ego.shape[0], 1, device=device) >= p_ed)
            ego = ego * keep_mask.to(ego.dtype)

    # ---- v0 speed-input: the PROVEN 3rd operative action channel --------------
    # flagship-speed 0.628 m vs nospeed 2.918 m (89.4% win); ckpt forensics:
    # act_emb.0.weight (768, 3). v0 = t=0 (last-input-frame) speed ONLY,
    # constant-expanded over the window AND the future actions — never a future
    # speed (leakage-safe). SPEED_SCALE contract with
    # eval_grounded_rollout_4b_speed.py: divide by 10.0. Under v2 the SAME
    # ego-dropout keep-mask zeroes this channel jointly with the planner ego
    # vector (anti-kinematic-integrator guard; refa-dynin pattern).
    if getattr(cfg, "speed_input", False):
        v0a = pose_last[:, 3:4] / 10.0                        # [B, 1]
        if keep_mask is not None:
            v0a = v0a * keep_mask.to(v0a.dtype)
        actions = torch.cat(
            [actions, v0a.unsqueeze(1).expand(-1, actions.shape[1], -1)],
            dim=-1)
        if fut_actions is not None:
            fut_actions = torch.cat(
                [fut_actions,
                 v0a.unsqueeze(1).expand(-1, fut_actions.shape[1], -1)],
                dim=-1)
    nav_in = nav_cmd
    p_nd = float(getattr(cfg, "v2_nav_dropout", 0.0))
    if model.training and p_nd > 0.0:          # lever 5: route from VISION
        drop = torch.rand(nav_cmd.shape[0], device=device) < p_nd
        nav_in = nav_cmd.masked_fill(drop, 0)  # 0 == follow
    # ---- the hierarchy: strategic ctx --FiLM--> tactical intent --FiLM--> op --
    strat = model.strategic_policy(states, nav_in, ego=ego)
    tac = model.tactical_policy(states, strat["ctx"], ego=ego)
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
        fa_roll = fut_actions
        p_fa = float(getattr(cfg, "v2_fa_dropout", 0.0))
        if (model.training and p_fa > 0.0 and fut_actions is not None):
            # v2 lever 2: per-sample future-action WITHHOLD (zero-order-hold =
            # the fa=None semantics) so the predictor must IMAGINE the future
            # from vision instead of leaning on given controls (imagination
            # share was 8.7%; pure integrators show D~E in the ablation).
            drop = torch.rand(fut_actions.shape[0], device=device) < p_fa
            hold = actions[:, -1:].expand(-1, fut_actions.shape[1], -1)
            fa_roll = torch.where(drop[:, None, None], hold, fut_actions)
        loss_roll = _rollout_loss(model, states, actions, fut_states,
                                  fa_roll, idx_of, K)

    # ---- tactical GOAL: 2 s sub-waypoints (grounded L2) + goal latent (JEPA) ---
    wp_h = cfg.tactical_policy.waypoint_horizons
    wp_pred = torch.stack([tac["waypoints"][k] for k in wp_h], dim=1)   # [B,H,2]
    wp_tgt = gt_ego_waypoints(pose_last, future_poses, wp_h)
    # v2 lever 8: TIME-anchored multi-anchor tactical decoder — nearest-anchor CE
    # + winner-takes-all L1 (multimodal) REPLACES the unimodal wp L2. wp_pred is
    # still derived (shim off the selected traj) so the jerk penalty below and any
    # downstream tac["waypoints"] reader keep working. loss_cls/loss_wta default 0.
    loss_cls = torch.zeros((), device=device)
    loss_wta = torch.zeros((), device=device)
    if getattr(cfg, "v2_anchor_tactical", False) and "anchor_traj" in tac:
        loss_wp, loss_cls, loss_wta = anchor_tactical_loss(
            tac, model.tactical_policy.anchor_decoder.anchors, wp_tgt, pose_scale)
    else:
        loss_wp = ((wp_pred - wp_tgt) / pose_scale).pow(2).mean()
    goal_tgt = fut_states[:, idx_of[plan.goal_h - 1]]
    prev_goal = fut_states[:, idx_of[plan.goal_h - 2]] if plan.goal_h >= 2 else z_t
    loss_goal = (change_weighted_mse(tac["target_latent"], goal_tgt, prev_goal)
                 if cw else (tac["target_latent"] - goal_tgt).pow(2).mean())

    # ---- v2 lever 4: decode the trajectory FROM the imagined goal latent ------
    # (goal cos=0.885 while the linear wp heads sat at 3.38 m — exploit the
    # model's best signal instead of ignoring it).
    loss_goalwp = torch.zeros((), device=device)
    if getattr(model, "goal_traj_head", None) is not None:
        gwp = model.goal_traj_head(
            torch.cat([z_t, tac["target_latent"]], dim=-1)
        ).view(z_t.shape[0], len(wp_h), 2)
        loss_goalwp = ((gwp - wp_tgt) / pose_scale).pow(2).mean()

    # ---- v2 lever 6: jerk (3rd-diff) penalty on predicted waypoint paths ------
    # (rollout paths were jerky: tms 0.09 vs GT-like 0.5+; refbpatch pattern).
    loss_jerk = torch.zeros((), device=device)
    w_jerk = float(getattr(cfg, "v2_traj_jerk", 0.0))
    if w_jerk > 0.0 and len(wp_h) >= 4:
        paths = [wp_pred]
        if getattr(model, "goal_traj_head", None) is not None:
            paths.append(gwp)
        loss_jerk = sum(torch.diff(p / pose_scale, n=3, dim=1).pow(2).mean()
                        for p in paths) / len(paths)

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

    # ---- LEVER A: route-FROM-vision aux (v2_route_from_vision) -----------------
    # The strategic route head is a pure command-echo (H26/H25: route_skill_vs_
    # chance 0.0, follow-acc == base rate). v2_nav_dropout already zeroes nav 50%
    # of the time in the MAIN CE, but stochastically. This adds an ALWAYS-ON,
    # DETERMINISTIC aux: a SECOND strategic pass with nav FORCED to follow(0)
    # (ego still fed — proprioception is not the command), class-weighted CE vs
    # the true route on the valid mask, so route-from-vision trains EVERY step.
    # Reuses the existing route_head (no new params). 0 when off or no valid nav.
    loss_route_vis = torch.zeros((), device=device)
    if getattr(cfg, "v2_route_from_vision", False) and bool(nav_valid.any()):
        nav_follow = torch.zeros_like(nav_cmd)          # force follow(0) every row
        strat_vis = model.strategic_policy(states, nav_follow, ego=ego)
        loss_route_vis = _class_weighted_ce(
            strat_vis["route_logits"][nav_valid], route_tgt[nav_valid], n_route)

    # ---- hierarchical metric grounding (op/tac/str) ---------------------------
    # Part A loss-rebalance: cfg.v2_invdyn_gradscale (default 1.0 = no-op) softly
    # decouples the invdyn REAL-PAIR term (a) from the encoder; term (b), JEPA,
    # SIGReg untouched. getattr keeps non-v2 cfgs / REF-A byte-identical.
    loss_ground, g_parts, g_log = grounding_losses(
        grounding, model.predictor, states, fut_states, actions, fut_actions,
        pose_last, future_poses, idx_of, plan.level_cfg, pose_scale,
        invdyn_weight=weights.invdyn, fwd_weight=weights.fwd,
        fwd_step_weight=fwd_step_weight,
        invdyn_gradscale=getattr(cfg, "v2_invdyn_gradscale", 1.0))

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

    # ---- LEVER B: encoder<->ego linear decorrelation (v2_encoder_ego_decorr) --
    # Penalize z_t being LINEARLY predictive of the fed ego [v0, yr0] so the
    # trained encoder stops REDUNDANTLY re-encoding dynamics (H25: yaw R2 0.89 in-
    # latent) and frees capacity for SCENE. Grad flows to the ENCODER (via z_t);
    # ego_full carries no grad. ego_r2 is a DETACHED linear-probe proxy (lower =
    # less re-encoding) — logged only, never a training gradient. No-op (0, no log
    # key) when the lever is off OR ego is None (non-v2). tanitad.train.decorr.
    loss_decorr = torch.zeros((), device=device)
    ego_r2 = None
    if getattr(cfg, "v2_encoder_ego_decorr", False) and ego_full is not None:
        loss_decorr = ego_decorr_loss(z_t, ego_full)
        ego_r2 = ego_linear_r2(z_t, ego_full)

    total = (weights.pred * loss_pred
             + weights.tacpred * loss_tacpred
             + weights.roll * loss_roll
             + weights.goal * loss_goal
             + weights.wp * loss_wp
             + weights.man * loss_man
             + weights.route * loss_route
             + weights.route_vis * loss_route_vis  # LEVER A (0 when off)
             + loss_ground                       # weights applied inside
             + weights.sigreg * loss_sig
             + weights.inv * loss_inv
             + weights.decorr * loss_decorr      # LEVER B (0 when off)
             + 1.0 * loss_goalwp                 # v2 lever 4 (0 when off)
             + w_jerk * loss_jerk)               # v2 lever 6 (0 when off)

    parts = {"pred": loss_pred, "tacpred": loss_tacpred, "roll": loss_roll,
             "goal": loss_goal, "wp": loss_wp, "cls": loss_cls, "wta": loss_wta,
             "man": loss_man, "route": loss_route, "route_vis": loss_route_vis,
             "sigreg": loss_sig, "inv": loss_inv, "ground": loss_ground,
             "decorr": loss_decorr, "goalwp": loss_goalwp,
             "jerk": loss_jerk, **g_parts}
    log = {"goalwp": loss_goalwp.item(), "v2jerk": loss_jerk.item(),
           "pred": loss_pred.item(), "tacpred": loss_tacpred.item(),
           "roll": loss_roll.item(), "goal": loss_goal.item(),
           "wp": loss_wp.item(), "man": loss_man.item(),
           "route": loss_route.item(), "man_acc": round(float(man_acc), 4),
           "route_acc": round(float(route_acc), 4),
           "sigreg": loss_sig.item(), "inv": loss_inv.item(),
           "ground": loss_ground.item(),
           "nav_valid_frac": round(float(nav_valid.float().mean()), 4),
           **g_log}
    # v2 lever 8 anchored-tactical diagnostics (only when the decoder ran).
    if getattr(cfg, "v2_anchor_tactical", False) and "anchor_traj" in tac:
        log.update({"cls": loss_cls.item(), "wta": loss_wta.item(),
                    "n_modes": int(tac["n_modes"]),
                    "conf_norm": round(float(tac["conf_norm"]), 4),
                    "prior_norm": round(float(tac["prior_norm"]), 4)})
    # LEVER A / LEVER B diagnostics — gated so default-off adds NO log keys.
    if getattr(cfg, "v2_route_from_vision", False):
        log["route_vis"] = loss_route_vis.item()
    if ego_r2 is not None:                       # lever on AND ego present
        log["decorr"] = loss_decorr.item()
        log["ego_r2"] = round(float(ego_r2), 4)  # DETACHED vision-reliance proxy
    return total, log, parts
