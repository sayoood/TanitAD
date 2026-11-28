#!/usr/bin/env python3
"""Phase-0 cosine pre-probe for GradCouple (v4.2b fork, brief P1).

Measures, at the ONE `states` seam that couples the planner into the shared
trunk (train_flagship_v4.v4_loss_step), the per-window cosine between
  g_wm   = d L_predict / d states     (world-model / prediction gradient)
  g_plan = d L_plan    / d states     (planner + factorised + smoothness + strategic)
on **v4.2b's checkpoint** over several batches of the CLEAN val split
(physicalai-val-0c5f7dac3b11), plus `seam_frac_removed` from the STAGED
one-sided-PCGrad `deconflict()` (grad_surgery.py). No weights are updated.

Decides which of three worlds we are in (PRE_REGISTRATION.md / DESIGN.md §7):
  * cos ~ -1, frac_removed ~ 1.0   -> objectives oppose -> surgery starves the
                                      planner -> recommend FROM-SCRATCH.
  * cos mildly-neg on a real fraction of windows, frac_removed < 0.70
                                   -> directional conflict w/ orthogonal subspace
                                      -> recommend GRADIENT-SURGERY (--coupling seam).
  * cos rarely negative (frac_conflict low), frac_removed ~ 0
                                   -> conflict NOT directional (trunk-LR
                                      re-optimisation, DESIGN §7.1) -> FROM-SCRATCH.

Faithful to v4_loss_step: same model/plan/head/goal_head construction as
train(), same loss assembly. g_plan is measured at lambda_plan=1.0 (raw planner
gradient the surgery would receive); cosine & frac_removed are scale-invariant so
this equals the geometry at any positive lam_mult. fp32 (autocast off) for a clean
angle estimate (training ran bf16; the directional verdict is precision-robust).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import statistics as st
import sys
from pathlib import Path

import torch

STACK = "/workspace/TanitAD/stack"
sys.path.insert(0, STACK)
sys.path.insert(0, STACK + "/scripts")
sys.path.insert(0, "/workspace/tmp")  # grad_surgery.py is scp'd here

from tanitad.config import flagship4b_config  # noqa: E402
from tanitad.data.mixing import load_episode  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402
from tanitad.models.strategic_goal import (GoalScalarConfig,  # noqa: E402
                                           GoalScalarHead)
from tanitad.models.flagship_v4 import FlagshipV4Head, v4_config  # noqa: E402
from tanitad.models.flagship_v15 import v15_losses  # noqa: E402
from tanitad.train.flagship_losses import (LossWeights,  # noqa: E402
                                           build_grounding, flagship_loss,
                                           horizon_plan)
from tanitad.train.v4_curriculum import (IGNORE_INDEX,  # noqa: E402
                                         factorised_ce, plan_smoothness_loss,
                                         strategic_scalar_loss)
from flagship_v4_data import FlagshipV4Dataset  # noqa: E402
import refb_labels  # noqa: E402
import train_flagship_v4 as T  # noqa: E402
from grad_surgery import deconflict  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def _stats(x):
    xs = sorted(x)
    n = len(xs)
    return {
        "mean": st.mean(x), "sd": (st.pstdev(x) if n > 1 else 0.0),
        "min": xs[0], "max": xs[-1],
        "p05": xs[int(0.05 * n)], "p10": xs[int(0.10 * n)],
        "p50": xs[int(0.50 * n)], "p90": xs[min(int(0.90 * n), n - 1)],
        "p95": xs[min(int(0.95 * n), n - 1)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-batches", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(0)

    cfgj = json.loads(Path(a.config).read_text())
    args = cfgj.get("args", {})
    d = T.V4LossWeights()
    lw = T.V4LossWeights(
        w_lat=args.get("lat_weight", d.w_lat), w_lon=args.get("lon_weight", d.w_lon),
        w_dist=args.get("dist_weight", d.w_dist), w_jerk=args.get("jerk_w", d.w_jerk),
        w_curv=args.get("curv_w", d.w_curv),
        w_strat=args.get("strat_scalar_weight", d.w_strat))
    ego_null_row = bool(args.get("ego_null_row", True))
    strategic = args.get("strategic", "full")

    # ---- model, exactly as train() ----
    cfg = flagship4b_config()
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    cfg.train.rollout_k = int(args.get("rollout_k", 4))
    plan = horizon_plan(cfg, op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20)
    world = WorldModel(cfg).to(dev)
    grounding = build_grounding(world.state_dim, device=dev)

    hcfg = v4_config()
    hcfg.state_dim = world.state_dim
    hcfg.cond_imagination = False
    hcfg.window = cfg.predictor.window
    hcfg.ego_null_row = ego_null_row
    head = FlagshipV4Head(hcfg).to(dev)
    anc = torch.load(a.anchors, weights_only=False)
    head.load_anchors((anc["anchors"] if isinstance(anc, dict) else anc).to(dev))
    goal_head = (GoalScalarHead(GoalScalarConfig(in_dim=world.state_dim)).to(dev)
                 if strategic != "off" else None)

    # ---- load v4.2b ckpt ----
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    world.load_state_dict(ck["model"])
    grounding.load_state_dict(ck["grounding"])
    head.load_state_dict(ck["head"])
    if goal_head is not None and ck.get("goal_head") is not None:
        goal_head.load_state_dict(ck["goal_head"])
    saved_step = int(ck.get("step", -1))
    world.eval(); grounding.eval(); head.eval()
    if goal_head is not None:
        goal_head.eval()

    val_eps = [load_episode(str(p), mmap=True)
               for p in sorted(Path(a.val).glob("ep_*.pt"))]
    dk = dict(window=cfg.predictor.window, max_horizon=plan.max_horizon,
              maneuver_h=plan.maneuver_h, channels=cfg.encoder.in_channels)
    ds_val = FlagshipV4Dataset(val_eps, **dk)
    gen = torch.Generator().manual_seed(0)
    dl = DataLoader(ds_val, batch_size=a.batch, shuffle=True, drop_last=True,
                    num_workers=0, generator=gen)
    horizons = head.cfg.horizons

    cos_all, frac_all, gwm_norm, gpl_norm, batch_diags = [], [], [], [], []
    n_windows, n_skipped = 0, 0

    it = iter(dl)
    for bi in range(a.n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        try:
            batch = T._to_device(batch, dev)
            with torch.no_grad():
                states0 = world.encode_window(batch["frames"])
                fut = world.encode_window(batch["future_frames"][:, plan.needed_fut])
            s = states0.detach().clone().requires_grad_(True)

            # g_wm = d L_predict / d states
            wm_total, _, _ = flagship_loss(
                world, grounding, batch, s, fut, plan, cfg, weights=LossWeights(),
                sigreg_variant="full_relaxed",
                sigreg_free_dims=cfg.loss.sigreg.free_dims,
                pose_scale=10.0, fwd_step_weight=0.5, device=dev)
            g_wm = torch.autograd.grad(wm_total, s, retain_graph=False)[0].detach()

            # g_plan = d L_plan / d states  (lambda_plan=1.0 -> raw planner grad)
            v0 = batch["pose_last"][:, 3].float()
            traj_tgt = refb_labels.waypoint_targets(
                batch["pose_last"].float(),
                batch["future_poses"][:, :max(horizons)].float(), horizons)
            goal = T._goal_inputs(head.cfg, batch, v0)
            out = head(s, v0, lambda_plan=1.0, **goal)
            plan_l = v15_losses(out, head.decoder.anchors, traj_tgt)
            L_plan = plan_l["loss"]
            if head.cfg.factorised:
                b = s.shape[0]
                lat = batch.get("lat_target", torch.full((b,), IGNORE_INDEX, device=dev))
                lon = batch.get("lon_target", torch.full((b,), IGNORE_INDEX, device=dev))
                dist = batch.get("dist_target", torch.full((b,), IGNORE_INDEX, device=dev))
                fac_loss, _ = factorised_ce(out, lat, lon, dist,
                                            lw.w_lat, lw.w_lon, lw.w_dist)
                L_plan = L_plan + fac_loss
            if out["wp_seq"].shape[-2] >= 4:
                sm_loss, _ = plan_smoothness_loss(out["wp_seq"], lw.w_jerk, lw.w_curv)
                L_plan = L_plan + sm_loss
            if goal_head is not None and "strat_scalars" in batch:
                pred = goal_head(s[:, -1])
                strat_loss, _ = strategic_scalar_loss(
                    pred, batch["strat_scalars"].to(s.device, dtype=s.dtype),
                    batch["strat_scalar_mask"].to(s.device), weight=lw.w_strat)
                L_plan = L_plan + strat_loss
            g_plan = torch.autograd.grad(L_plan, s, retain_graph=False)[0].detach()

            # staged reference diag (per-batch summary)
            _, diag = deconflict(g_plan, g_wm, per_sample=True, target_cos=0.0)
            batch_diags.append(diag)

            # pooled per-window cosine + frac_removed (mirror of deconflict math)
            B = g_plan.shape[0]
            gp = g_plan.reshape(B, -1).float()
            gr = g_wm.reshape(B, -1).float()
            dot = (gp * gr).sum(1)
            npn = gp.norm(dim=1)
            grn = gr.norm(dim=1)
            cos = dot / (npn * grn).clamp_min(1e-12)
            coeff = (dot < 0).float() * (dot / (grn ** 2).clamp_min(1e-12))
            removed = (coeff.unsqueeze(1) * gr).norm(dim=1)
            frac = removed / npn.clamp_min(1e-12)
            cos_all.extend(cos.cpu().tolist())
            frac_all.extend(frac.cpu().tolist())
            gwm_norm.extend(grn.cpu().tolist())
            gpl_norm.extend(npn.cpu().tolist())
            n_windows += B
            print(f"[probe] batch {bi} B={B} cos_mean={cos.mean():.4f} "
                  f"frac_removed_mean={frac.mean():.4f} frac_conflict="
                  f"{(cos < 0).float().mean():.3f}", file=sys.stderr, flush=True)
        except RuntimeError as e:
            n_skipped += 1
            print(f"[probe] batch {bi} SKIPPED: {e}", file=sys.stderr, flush=True)
            if dev == "cuda":
                torch.cuda.empty_cache()
            continue

    frac_conflict = sum(1 for c in cos_all if c < 0) / max(len(cos_all), 1)
    result = {
        "probe": "phase0_cosine_preprobe",
        "date": "2026-07-23",
        "ckpt": a.ckpt, "ckpt_step": saved_step,
        "val_cache": a.val, "n_windows": n_windows,
        "batch": a.batch, "n_batches_run": len(batch_diags), "n_skipped": n_skipped,
        "precision": "fp32 (autocast off); training ran bf16 — cosine is precision-robust",
        "lambda_plan_for_g_plan": 1.0,
        "eval_mode": True,
        "seam_cosine_over_windows": _stats(cos_all) if cos_all else None,
        "seam_frac_removed_over_windows": _stats(frac_all) if frac_all else None,
        "frac_windows_conflicting_cos_lt_0": frac_conflict,
        "g_wm_norm_over_windows": _stats(gwm_norm) if gwm_norm else None,
        "g_plan_norm_over_windows": _stats(gpl_norm) if gpl_norm else None,
        "staged_deconflict_batch_diag_means": {
            "seam_cos_mean": st.mean([x["seam_cos_mean"] for x in batch_diags]),
            "seam_frac_conflict": st.mean([x["seam_frac_conflict"] for x in batch_diags]),
            "seam_frac_removed_mean": st.mean([x["seam_frac_removed_mean"] for x in batch_diags]),
            "seam_cos_min": min(x["seam_cos_min"] for x in batch_diags),
        } if batch_diags else None,
        "loss_weights": dataclasses.asdict(lw),
    }
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
