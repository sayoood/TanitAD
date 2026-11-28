#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P1 — the two fixes for the Gate-1 closed-loop-aware planner fine-tune, as a
drop-in patch to gate1_finetune.py. Built + MEASURED (gate1_clean_loo.py, variants
B_catk / C_catk_dev; see GATE1_CLEAN_RUN_P0_FINDINGS.md §5).

FIX (a) — CAT-K / recovery-feasibility TARGET FILTERING.
  Method: CAT-K = "Closest-Among-Top-K" closed-loop supervised fine-tuning
  (Zhang, Karkus, et al., NVIDIA, "Closed-Loop Supervised Fine-Tuning of Tokenized
  Traffic Models", CVPR 2025) — keep closed-loop supervision anchored to the expert
  manifold so recovery targets stay FEASIBLE, instead of supervising from
  catastrophic states whose "recovery" points sharply backward. Covariate-shift
  frame: DAgger (Ross, Gordon, Bagnell, AISTATS 2011). ("RoAD" in the brief is the
  same recovery-augmentation-with-filtering principle; CAT-K is the load-bearing
  citation — no canonical "RoAD" paper was verified, so it is not attributed.)
  Here: DROP a recovery label whose GT target leaves the P1-MEASURED low-OOD
  envelope (|left| > 3.0 m over the 2 s path, or heading-correction > 12 deg) or
  points BACKWARD (final fwd <= 0). On the prototype's 675 NuRec labels this drops
  328 (49%): 147 backward, 281 beyond-lateral, 285 beyond-yaw (MEASURED).

FIX (b) — DEVIATION / STABILITY REGULARIZER (base-plan trust region).
  A behaviour-cloning / trust-region penalty lambda_dev * ||FT_traj - base_traj||_1
  keeping the fine-tuned plan close to the well-behaved BASE planner (the plan_dev
  ~0.34 anchored-diffusion family) so recovery is not bought with an aggressive
  swerve — the mechanism retracted for flagship v1 (RETRACTION_LOG 07-23 C7).
  Analogous to KL-to-reference in offline RL. Default lambda_dev = 1.0.

Envelope constants come from the P1 MEASURED envelope (lowood_flagship_ci.json /
LOWER_OOD_CLOSEDLOOP_DESIGN.md 2.2): <=1.16x baseline out to +-3 m / +-12 deg.
"""
from __future__ import annotations
import torch

ENVELOPE_LAT_M = 3.0        # P1: lateral OOD <= 1.16x baseline to 3.0 m
ENVELOPE_YAW_DEG = 12.0     # P1: yaw OOD <= 1.16x baseline to 12 deg


def catk_keep_mask(tgt: torch.Tensor,
                   envelope_lat_m: float = ENVELOPE_LAT_M,
                   envelope_yaw_deg: float = ENVELOPE_YAW_DEG):
    """tgt [N,H,2] rig frame (fwd, left) = expert recovery path 0.5..2 s ahead.
    Returns (keep_bool[N], stats). Keep only near-manifold, feasible recovery
    labels; drop catastrophic-state ones that drive the high-deviation trade."""
    fwd_end = tgt[:, -1, 0]
    dleft = tgt[:, -1, 1] - tgt[:, 0, 1]
    dfwd = (tgt[:, -1, 0] - tgt[:, 0, 0]).clamp_min(1e-3)
    yaw_corr_deg = torch.atan2(dleft.abs(), dfwd) * 180.0 / torch.pi
    backward = fwd_end <= 0.0
    too_lat = tgt[:, :, 1].abs().amax(dim=1) > envelope_lat_m
    too_yaw = yaw_corr_deg > envelope_yaw_deg
    drop = backward | too_lat | too_yaw
    keep = ~drop
    stats = {"n": int(tgt.shape[0]), "kept": int(keep.sum()), "dropped": int(drop.sum()),
             "drop_backward": int(backward.sum()), "drop_too_lat": int(too_lat.sum()),
             "drop_too_yaw": int(too_yaw.sum())}
    return keep, stats


def deviation_penalty(ft_traj: torch.Tensor, base_traj: torch.Tensor,
                      lambda_dev: float = 1.0) -> torch.Tensor:
    """L1 trust region to the base plan. ft_traj / base_traj [B,H,2]. Add to loss:
        loss = traj_L1 + cls_CE + deviation_penalty(dec['traj'], base_traj[bi])"""
    if lambda_dev <= 0:
        return ft_traj.new_zeros(())
    return lambda_dev * (ft_traj - base_traj).abs().mean()


# --- how to splice into gate1_finetune.py (comments only) ---
# 1. Before training, cache the BASE decoder's selected traj per window:
#      base_traj, _, _ = <decoder(FMAP, M, ctx, MAN, steps=2)['traj'] over all windows>
# 2. Filter labels once:  keep, stats = catk_keep_mask(TGT); tr_idx = tr_idx[keep[tr_idx]]
# 3. In the train step, after loss = loss_traj + loss_cls:
#      loss = loss + deviation_penalty(dec['traj'], base_traj[bi], lambda_dev=1.0)
# The measured reference implementation of exactly this is gate1_clean_loo.py.
