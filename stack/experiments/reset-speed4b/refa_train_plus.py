"""REF-A improvement trainer (Phase-2 prep) -- ISOLATED copy of
scripts/refa_train.py that reuses the base module's helpers unchanged and adds
three TOGGLES. Nothing here touches the real refa_train.py / refa.py / any ckpt.

  FIX 1b  --aux-egomotion : ego-speed + yaw-rate aux regression heads on the
          latent  +  a per-step displacement-MAGNITUDE (scale) loss on the
          step_readout. Forces the latent to encode speed and calibrates the
          decoder magnitude (the Test-3 71-83% scale error, shown post-hoc
          UNrecoverable in refa_calib.py).
  FIX 2   --adapter temporal : motion-aware TemporalGridAdapter (refa_plus).
  FIX 4   --rollout-k 12     : longer K-step recursive rollout (existing flag).

Base losses (pred/roll/inv/sigreg + metric-dynamics grounding) are byte-identical
to refa_train.compute_losses; the toggled terms are ADDED on top.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
from torch import Tensor

import sys
sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
sys.path.insert(0, "/workspace/tmp/refa_plus")

import torch.nn.functional as F  # noqa: E402

import refa_train as base  # noqa: E402  (reuse ALL base helpers unchanged)
import refb_labels  # noqa: E402  (nav/maneuver/route pseudo-labels, shared w/ flagship)
from refa_train import (FWD_STEP_WEIGHT, FWD_WEIGHT, INV_WEIGHT,  # noqa: E402
                        INVDYN_WEIGHT, POSE_SCALE, PRED_WEIGHT, ROLL_WEIGHT,
                        SIGREG_WEIGHT, smoke_pred_config)
from refa_plus import RefAModelPlus, build_aux_heads  # noqa: E402
from tanitad.config import flagship4b_config  # noqa: E402  (the 4-brain StackConfig)
from tanitad.models.metric_dynamics import (accumulate_se2,  # noqa: E402
                                            gt_ego_waypoints, gt_step_dposes,
                                            relative_ego_pose, wrap_angle)
from tanitad.models.predictor import change_weighted_mse  # noqa: E402
from tanitad.refs.refa import refa_predictor_config  # noqa: E402
from tanitad.train.train_worldmodel import cosine_lr  # noqa: E402

DT = 0.1
SPEED_SCALE = 10.0        # normalize m/s targets like pose_scale
# 4-brain loss weights (mirror tanitad.train.flagship_losses.LossWeights defaults
# so REF-A's four brains train to the SAME objective as the flagship's).
FB_TACPRED_W, FB_GOAL_W, FB_WP_W, FB_MAN_W, FB_ROUTE_W = 0.5, 0.5, 1.0, 0.5, 0.5


def compute_losses_plus(model, batch, rollout_k, device="cpu", *,
                        metric_heads=None, mid_horizons=None,
                        invdyn_weight=INVDYN_WEIGHT, fwd_weight=FWD_WEIGHT,
                        pose_scale=POSE_SCALE, fwd_step_weight=FWD_STEP_WEIGHT,
                        aux_heads=None, aux_speed_weight=1.0,
                        aux_yaw_weight=1.0, aux_accel_weight=1.0,
                        scale_weight=0.5, jerk_weight=0.0, speed_input=False,
                        four_brain=False, fb_cfg=None, goal_h=20) -> dict:
    """refa_train.compute_losses (verbatim core) + FIX-1b aux/scale terms
    (+ the full 4-brain joint loss when ``four_brain`` — strategic route CE,
    tactical maneuver CE / goal-waypoint L2 / goal-latent JEPA, tactical-
    predictor JEPA, and the intent-conditioned operative, on the SAME objective
    as tanitad.train.flagship_losses.flagship_loss)."""
    feats = batch["feats"].to(device)
    actions = batch["actions"].to(device)
    fut_feats = batch["future_feats"].to(device)
    fut_actions = batch["future_actions"].to(device)
    if speed_input:
        v0 = batch["pose_last"].to(device).float()[:, 3:4] / SPEED_SCALE  # current ego-speed at t=0
        actions = torch.cat([actions, v0.unsqueeze(1).expand(-1, actions.shape[1], -1)], dim=-1)
        fut_actions = torch.cat([fut_actions, v0.unsqueeze(1).expand(-1, fut_actions.shape[1], -1)], dim=-1)

    states = model.encode_window(feats)
    fut_states = model.encode_window(fut_feats)
    z_t = states[:, -1]

    # ---- the hierarchy (4-brain): strategic ctx --FiLM--> tactical intent
    # --FiLM--> intent-conditioned operative. Falls back to the plain operative
    # predict() when four_brain is off (base REF-A+).
    if four_brain:
        nav_cmd = batch["nav_cmd"].to(device)
        strat = model.strategic_policy(states, nav_cmd)
        tac = model.tactical_policy(states, strat["ctx"])
        preds = model.predictor(states, actions, intent=tac["intent"])
    else:
        preds = model.predict(states, actions)
    horizons = model.pred_cfg.horizons
    loss_pred = torch.zeros((), device=states.device)
    for k in horizons:
        target = fut_states[:, k - 1]
        prev = z_t if k == 1 else fut_states[:, k - 2]
        if model.pred_cfg.change_weighted:
            loss_pred = loss_pred + change_weighted_mse(preds[k], target, prev)
        else:
            loss_pred = loss_pred + (preds[k] - target).pow(2).mean()
    loss_pred = loss_pred / len(horizons)

    loss_roll = torch.zeros((), device=states.device)
    roll_preds: list[Tensor] = []
    if rollout_k > 1:
        loss_roll, roll_preds = base._rollout(model, states, actions,
                                              fut_states, fut_actions, rollout_k)

    a_hat = model.inv_dyn(states[:, -2], states[:, -1])
    loss_inv = (a_hat - actions[:, -2]).pow(2).mean()

    z_pred_all = torch.cat([preds[k] for k in horizons] + roll_preds)
    loss_sig = model.sigreg(z_pred_all)

    loss = (PRED_WEIGHT * loss_pred + ROLL_WEIGHT * loss_roll
            + INV_WEIGHT * loss_inv + SIGREG_WEIGHT * loss_sig)
    out = {"pred": loss_pred, "roll": loss_roll, "inv": loss_inv,
           "sigreg": loss_sig, "n_sig": int(z_pred_all.shape[0]), "states": states}

    pose_last = future_poses = None
    if metric_heads is not None:
        mid, step_ro = metric_heads["metric_invdyn"], metric_heads["step_readout"]
        mh = list(mid_horizons) if mid_horizons is not None else list(horizons)
        ps = pose_scale
        pose_last = batch["pose_last"].to(device).float()
        future_poses = batch["future_poses"].to(device).float()
        loss_mid = torch.zeros((), device=states.device)
        metric_de = 0.0
        for kh in mh:
            dpose = mid(z_t, fut_states[:, kh - 1])
            tgt = relative_ego_pose(pose_last, future_poses[:, kh - 1])
            loss_mid = loss_mid \
                + ((dpose[..., :2] - tgt[..., :2]) / ps).pow(2).mean() \
                + wrap_angle(dpose[..., 2] - tgt[..., 2]).pow(2).mean()
            metric_de += float((dpose[..., :2] - tgt[..., :2]).detach()
                               .norm(dim=-1).mean())
        loss_mid = loss_mid / len(mh)
        metric_de /= len(mh)

        loss_scale = torch.zeros((), device=states.device)
        if rollout_k > 1 and roll_preds:
            prevs = [z_t] + roll_preds[:-1]
            step_dp = torch.stack(
                [step_ro(prevs[j], roll_preds[j])
                 for j in range(len(roll_preds))], dim=1)
            pred_wp = accumulate_se2(step_dp)
            gt_wp = gt_ego_waypoints(pose_last, future_poses,
                                     range(1, rollout_k + 1))
            gt_step = gt_step_dposes(pose_last, future_poses, rollout_k)
            loss_acc = ((pred_wp - gt_wp) / ps).pow(2).mean()
            loss_step = ((step_dp[..., :2] - gt_step[..., :2]) / ps).pow(2).mean() \
                + wrap_angle(step_dp[..., 2] - gt_step[..., 2]).pow(2).mean()
            loss_fwd = loss_acc + fwd_step_weight * loss_step
            fwd_ade = float((pred_wp.detach() - gt_wp).norm(dim=-1).mean())
            # FIX 1b scale loss: per-step displacement MAGNITUDE mismatch.
            sp = step_dp[..., :2].norm(dim=-1)
            sg = gt_step[..., :2].norm(dim=-1)
            loss_scale = ((sp - sg) / ps).pow(2).mean()
            # jerk/smoothness: mean-squared 3rd difference of the PREDICTED
            # longitudinal (forward) rollout path -> comfortable/smooth driving.
            loss_jerk = torch.zeros((), device=states.device)
            if pred_wp.shape[1] >= 4:
                lon = pred_wp[..., 0]
                jerk = (lon[:, 3:] - 3 * lon[:, 2:-1] + 3 * lon[:, 1:-2]
                        - lon[:, :-3])
                loss_jerk = jerk.pow(2).mean()
        else:
            loss_fwd = torch.zeros((), device=states.device)
            fwd_ade = 0.0
            loss_scale = torch.zeros((), device=states.device)
            loss_jerk = torch.zeros((), device=states.device)

        loss = loss + invdyn_weight * loss_mid + fwd_weight * loss_fwd
        out.update({"metric_invdyn": loss_mid, "fwd": loss_fwd,
                    "metric_de": round(metric_de, 4), "fwd_ade": round(fwd_ade, 4)})

        if aux_heads is not None:
            loss = loss + scale_weight * loss_scale
            out["scale"] = loss_scale
        if jerk_weight > 0:
            loss = loss + jerk_weight * loss_jerk
            out["jerk"] = loss_jerk

    # FIX 1b/accel aux supervision (needs pose targets from the metric branch).
    if aux_heads is not None and pose_last is not None:
        if "aux_speed" in aux_heads:
            v_gt = pose_last[:, 3] / SPEED_SCALE
            v_hat = aux_heads["aux_speed"](z_t).squeeze(-1)
            loss_spd = (v_hat - v_gt).pow(2).mean()
            loss = loss + aux_speed_weight * loss_spd
            out.update({"aux_speed": loss_spd,
                        "aux_speed_r2": _r2(v_hat.detach(), v_gt)})
        if "aux_yaw" in aux_heads:
            yaw_gt = wrap_angle(future_poses[:, 0, 2] - pose_last[:, 2]) / DT
            yaw_hat = aux_heads["aux_yaw"](z_t).squeeze(-1)
            loss_yaw = (yaw_hat - yaw_gt).pow(2).mean()
            loss = loss + aux_yaw_weight * loss_yaw
            out.update({"aux_yaw": loss_yaw,
                        "aux_yaw_r2": _r2(yaw_hat.detach(), yaw_gt)})
        if "aux_accel" in aux_heads:
            # GT longitudinal accel = odometry finite-diff of speed (the CLEAN,
            # realized source; action[:,1] is commanded accel, corr only 0.48).
            acc_gt = (future_poses[:, 0, 3] - pose_last[:, 3]) / DT
            acc_hat = aux_heads["aux_accel"](z_t).squeeze(-1)
            loss_acc = (acc_hat - acc_gt).pow(2).mean()
            loss = loss + aux_accel_weight * loss_acc
            out.update({"aux_accel": loss_acc,
                        "aux_accel_r2": _r2(acc_hat.detach(), acc_gt)})

    # ---- 4-brain joint loss (D-030): tactical-pred JEPA + tactical GOAL
    # (waypoint L2 + goal-latent JEPA) + maneuver CE + strategic route CE. The
    # SAME objective as flagship_losses.flagship_loss; futures here are dense
    # (fut_states[:, k-1]) so no idx_of map is needed. Needs pose targets from
    # the metric branch (always on for the 4-brain run).
    if four_brain and pose_last is not None:
        cw = model.pred_cfg.change_weighted
        # tactical-predictor dynamics (maneuver-horizon JEPA)
        loss_tacpred = torch.zeros((), device=states.device)
        if model.tactical_pred is not None:
            tp = model.tactical_pred(states, actions)
            th = fb_cfg.tactical_pred.horizons
            for k in th:
                tgt = fut_states[:, k - 1]
                prev = z_t if k == 1 else fut_states[:, k - 2]
                loss_tacpred = loss_tacpred + (change_weighted_mse(tp[k], tgt, prev)
                                               if cw else (tp[k] - tgt).pow(2).mean())
            loss_tacpred = loss_tacpred / len(th)
        # tactical GOAL: 2 s ego sub-waypoints (grounded L2) + goal latent (JEPA)
        wp_h = fb_cfg.tactical_policy.waypoint_horizons
        wp_pred = torch.stack([tac["waypoints"][k] for k in wp_h], dim=1)   # [B,H,2]
        wp_tgt = gt_ego_waypoints(pose_last, future_poses, wp_h)
        loss_wp = ((wp_pred - wp_tgt) / pose_scale).pow(2).mean()
        gtgt = fut_states[:, goal_h - 1]
        gprev = fut_states[:, goal_h - 2] if goal_h >= 2 else z_t
        loss_goal = (change_weighted_mse(tac["target_latent"], gtgt, gprev)
                     if cw else (tac["target_latent"] - gtgt).pow(2).mean())
        # maneuver CE (class-weighted) + strategic route CE (valid-masked)
        man_tgt = batch["maneuver_label"].to(device)
        loss_man = _class_weighted_ce(tac["maneuver_logits"], man_tgt,
                                      fb_cfg.tactical_policy.n_maneuvers)
        man_acc = (tac["maneuver_logits"].argmax(-1) == man_tgt).float().mean()
        nav_valid = batch["nav_valid"].to(device)
        route_tgt = batch["route_target"].to(device)
        if bool(nav_valid.any()):
            tv = route_tgt[nav_valid]
            loss_route = _class_weighted_ce(strat["route_logits"][nav_valid], tv,
                                            fb_cfg.strategic_policy.n_route)
            route_acc = (strat["route_logits"][nav_valid].argmax(-1)
                         == tv).float().mean()
        else:
            loss_route = torch.zeros((), device=states.device)
            route_acc = torch.zeros((), device=states.device)
        loss = (loss + FB_TACPRED_W * loss_tacpred + FB_GOAL_W * loss_goal
                + FB_WP_W * loss_wp + FB_MAN_W * loss_man + FB_ROUTE_W * loss_route)
        out.update({"tacpred": loss_tacpred, "goal": loss_goal, "wp": loss_wp,
                    "man": loss_man, "route": loss_route,
                    "man_acc": round(float(man_acc), 4),
                    "route_acc": round(float(route_acc), 4),
                    "nav_valid_frac": round(float(nav_valid.float().mean()), 4)})

    out["loss"] = loss
    return out


def _r2(pred, y):
    ssr = (pred - y).pow(2).sum()
    sst = (y - y.mean()).pow(2).sum().clamp_min(1e-9)
    return round(float(1 - ssr / sst), 4)


def _class_weighted_ce(logits, target, n_classes, clamp=10.0):
    """Inverse per-batch class-frequency weighted CE (verbatim from
    flagship_losses._class_weighted_ce) — the maneuver/route classes are highway-
    imbalanced (lane_keep / follow dominate) so rare turns are not drowned."""
    counts = torch.bincount(target, minlength=n_classes).float()
    w = (target.numel() / (n_classes * counts.clamp_min(1.0))).clamp(max=clamp)
    return F.cross_entropy(logits, target, weight=w.to(logits.dtype))


class FeatureWindowDataset4B(base.FeatureWindowDataset):
    """base feature windows + the 4-brain pseudo-labels (mirrors the flagship's
    FlagshipWindowDataset): the maneuver class from the window's 2 s future
    kinematics, and the strategic nav_cmd / nav_valid / route_target from the
    FULL episode poses at the window anchor (25 s route lookahead — the SAME
    refb_labels derivation the flagship uses, so REF-A and the flagship read
    identical label semantics)."""

    def __init__(self, episodes, window, max_horizon, maneuver_h):
        super().__init__(episodes, window, max_horizon)
        assert maneuver_h <= max_horizon, (maneuver_h, max_horizon)
        self.maneuver_h = maneuver_h

    def __getitem__(self, i):
        item = super().__getitem__(i)
        e_i, t = self.index[i]
        p_last = item["pose_last"]
        p1 = item["future_poses"][self.maneuver_h - 1]
        item["maneuver_label"] = refb_labels.classify_maneuver(
            p_last[2], p1[2], p_last[3], p1[3]).long()
        # nav command from the FULL episode poses at the anchor (last obs frame).
        anchor = t + self.window - 1
        nav, valid = refb_labels.nav_command(self.episodes[e_i]["poses"], anchor)
        item["nav_cmd"] = torch.tensor(nav, dtype=torch.long)
        item["nav_valid"] = torch.tensor(bool(valid))
        item["route_target"] = torch.tensor(refb_labels.route_target(nav),
                                             dtype=torch.long)
        return item


def train(args) -> dict:
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    fb_cfg = None
    goal_h = None
    if args.four_brain:
        # Full 4-brain on the frozen-DINO adapter: the SAME StackConfig the
        # flagship uses, so every shared brain is byte-for-byte the same shape;
        # the encoder (adapter vs from-scratch ViT) is the only model-axis diff.
        fb_cfg = flagship4b_config()
        if args.speed_input:
            object.__setattr__(fb_cfg.predictor, "action_dim", 3)
            if fb_cfg.tactical_pred is not None:
                object.__setattr__(fb_cfg.tactical_pred, "action_dim", 3)
        model = RefAModelPlus.from_stack_config(
            fb_cfg, n_tokens=256, adapter_kind=args.adapter).to(device)
        pred_cfg = fb_cfg.predictor
        goal_h = max(fb_cfg.tactical_policy.waypoint_horizons)
    else:
        pred_cfg = smoke_pred_config() if args.smoke else refa_predictor_config()
        if args.speed_input:
            pred_cfg = dataclasses.replace(pred_cfg, action_dim=3)
        model = RefAModelPlus(pred_cfg, adapter_kind=args.adapter).to(device)
    metric_heads = base.build_metric_heads(model.state_dim, device)
    mid_horizons = list(pred_cfg.horizons)
    which = []
    if args.aux_egomotion:
        which += ["speed", "yaw"]
    if args.aux_accel:
        which += ["accel"]
    aux_heads = build_aux_heads(model.state_dim, device, which=which) \
        if which else None
    extra = [p for h in metric_heads.values() for p in h.parameters()]
    if aux_heads:
        extra += [p for h in aux_heads.values() for p in h.parameters()]
    opt = torch.optim.AdamW(base.param_groups(model, args.lr, extra),
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    ground_kw = dict(metric_heads=metric_heads, mid_horizons=mid_horizons,
                     invdyn_weight=args.invdyn_weight, fwd_weight=args.fwd_weight,
                     pose_scale=args.pose_scale, fwd_step_weight=args.fwd_step_weight,
                     aux_heads=aux_heads, aux_speed_weight=args.aux_speed_weight,
                     aux_yaw_weight=args.aux_yaw_weight,
                     aux_accel_weight=args.aux_accel_weight,
                     scale_weight=args.scale_weight, jerk_weight=args.jerk_weight,
                     speed_input=args.speed_input,
                     four_brain=args.four_brain, fb_cfg=fb_cfg, goal_h=goal_h)
    save_heads = dict(metric_heads)
    if aux_heads:
        save_heads.update(aux_heads)

    max_h = max(max(pred_cfg.horizons), args.rollout_k)
    maneuver_h = None
    if args.four_brain:
        maneuver_h = goal_h
        tac_h = list(fb_cfg.tactical_pred.horizons) if fb_cfg.tactical_pred else []
        max_h = max([max_h, goal_h, maneuver_h] + tac_h)
    train_eps, train_dir = base.load_feature_episodes(args.data_root, "*train*",
                                                      args.episodes)
    if args.four_brain:
        ds = FeatureWindowDataset4B(train_eps, pred_cfg.window, max_h, maneuver_h)
    else:
        ds = base.FeatureWindowDataset(train_eps, pred_cfg.window, max_h)
    assert len(ds) >= args.batch, f"only {len(ds)} windows for batch {args.batch}"
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     drop_last=True)
    print(f"[refa+] {args.adapter} adapter, aux={bool(aux_heads)}, "
          f"rollout_k={args.rollout_k}: {len(train_eps)} eps / {len(ds)} windows",
          flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"arch": "REF-A+", "adapter": args.adapter, "aux_egomotion": bool(aux_heads),
         "pred_cfg": dataclasses.asdict(pred_cfg), "args": vars(args)},
        indent=2, default=str), encoding="utf-8")

    step = 0
    ckpt_path = out_dir / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        for name, h in save_heads.items():
            if name in ck:
                h.load_state_dict(ck[name])
        step = int(ck["step"]) + 1
        print(f"[resume] at step {step}", flush=True)
    else:
        t = time.perf_counter()
        model.standardizer.fit(ep["feats_fp16"] for ep in train_eps)
        print(f"[refa+] standardizer fitted ({time.perf_counter()-t:.1f}s)",
              flush=True)

    warm = {"predictor": args.warmup, "adapter": args.warmup * 10}
    data_iter = iter(dl)
    last_log: dict = {}
    while step < args.steps:
        lrs = {}
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, args.steps, warm[pg["name"]], args.lr)
            lrs[pg["name"]] = pg["lr"]
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl); batch = next(data_iter)
        opt.zero_grad(set_to_none=True)
        out = compute_losses_plus(model, batch, args.rollout_k, device, **ground_kw)
        out["loss"].backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + extra, 1.0))
        opt.step()
        if step > 0 and step % args.save_every == 0:
            base._save_ckpt(ckpt_path, model, opt, step, save_heads)
        if step % args.log_every == 0 or step == args.steps - 1:
            sc = lambda t: round(float(t.detach()), 5)
            last_log = {"step": step, "loss": sc(out["loss"]),
                        "pred": sc(out["pred"]), "roll": sc(out["roll"]),
                        "metric_de": out.get("metric_de"), "fwd_ade": out.get("fwd_ade"),
                        "adapter_std": round(model.adapter_dim_std(out["states"]), 5),
                        "gnorm": round(gnorm, 4)}
            if "scale" in out:
                last_log["scale"] = sc(out["scale"])
            if "jerk" in out:
                last_log["jerk"] = sc(out["jerk"])
            if "aux_speed" in out:
                last_log.update({"aux_speed": sc(out["aux_speed"]),
                                 "aux_yaw": sc(out["aux_yaw"]),
                                 "aux_speed_r2": out["aux_speed_r2"],
                                 "aux_yaw_r2": out["aux_yaw_r2"]})
            if "aux_accel" in out:
                last_log.update({"aux_accel": sc(out["aux_accel"]),
                                 "aux_accel_r2": out["aux_accel_r2"]})
            if "man" in out:                       # 4-brain terms
                last_log.update({"tacpred": sc(out["tacpred"]),
                                 "goal": sc(out["goal"]), "wp": sc(out["wp"]),
                                 "man": sc(out["man"]), "route": sc(out["route"]),
                                 "man_acc": out["man_acc"],
                                 "route_acc": out["route_acc"],
                                 "nav_valid_frac": out["nav_valid_frac"]})
            print(json.dumps(last_log), flush=True)
        step += 1

    base._save_ckpt(ckpt_path, model, opt, step - 1, save_heads)
    metrics = {"final": last_log, "steps": step, "device": device,
               "adapter": args.adapter, "aux_egomotion": bool(aux_heads),
               "rollout_k": args.rollout_k}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                          encoding="utf-8")
    print(json.dumps({"done": True, "steps": step, "out": str(out_dir)}), flush=True)
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--rollout-k", type=int, default=4)          # FIX 4: set 12
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--invdyn-weight", type=float, default=INVDYN_WEIGHT)
    ap.add_argument("--fwd-weight", type=float, default=FWD_WEIGHT)
    ap.add_argument("--pose-scale", type=float, default=POSE_SCALE)
    ap.add_argument("--fwd-step-weight", type=float, default=FWD_STEP_WEIGHT)
    ap.add_argument("--adapter", choices=("pool", "grid", "temporal"),
                    default="grid")                              # FIX 2: temporal
    ap.add_argument("--aux-egomotion", action="store_true")     # FIX 1b
    ap.add_argument("--aux-speed-weight", type=float, default=1.0)
    ap.add_argument("--aux-yaw-weight", type=float, default=1.0)
    ap.add_argument("--aux-accel", action="store_true")         # accel head
    ap.add_argument("--aux-accel-weight", type=float, default=1.0)
    ap.add_argument("--scale-weight", type=float, default=0.5)
    ap.add_argument("--jerk-weight", type=float, default=0.0,
                    help="smoothness: MSE of 3rd-diff of predicted longitudinal "
                         "path (modest, e.g. 0.02)")
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--speed-input", action="store_true", help="feed current ego-speed v0 as 3rd action channel")
    ap.add_argument("--four-brain", action="store_true",
                    help="build+train the FULL 4-brain (strategic route + "
                         "tactical maneuver/goal-waypoint/goal-latent + tactical-"
                         "predictor + intent-conditioned operative) on the frozen-"
                         "DINO adapter, from flagship4b_config — the SAME brains "
                         "and objective as the flagship (encoder is the only diff)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
