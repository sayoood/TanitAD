#!/usr/bin/env python3
"""GAP 2 stage 1 — dump flagship-v4's PER-CANDIDATE FAN GEOMETRY + its real score.

WHY THIS EXISTS
---------------
FIX 2 (`…/2026-07-27-confirmed-fixes/CONFIRMED_FIXES.md`) shipped a reachability
clamp that is default ON for v1.5 -- MEASURED free on REF-C-XL's emitted fan
(881 canonical val windows: 72.08 % of candidates removed, ADE-oracle surviving
100 %, paired Δ exactly 0.0000, 3.58x cheaper).  It is DELIBERATELY OFF for v4
(`V4Config.sel_reach_clamp = False`) because **v4's own fan geometry was never
dumped**: the only v4 surface in the program (`v5_v4_windows_reduced.pt`) carries
per-candidate errors but no geometry and no usable score ranking.  This script
produces the missing input.

⚠️ `base_rank` in that older surface is NOT a rank -- it is
`[as-trained pick] ++ [anchor index order]` and carries ZERO score information
(retraction class C15, hit independently by three streams).  NOTHING here reads
it.  The ranking dumped here is the head's OWN `sel_score`: the tensor
`FlagshipV15Head.select` actually argmaxes, i.e. `refined_logits` AFTER v4's
factorised LAT×LON×DIST grafts and after the VTARGET longitudinal term.

⚠️ The 256 anchors are bitwise identical to real human windows
(`furthest_point_sample` returns `pool[chosen]`).  The excess speed is the
UNBOUNDED OFFSET HEAD.  The clamp targets the refinement, never the vocabulary --
and this script dumps `decoder.anchors` beside the emitted fan so that stays
checkable.

FIDELITY, both directions (the dump is inadmissible without these):
  * the recorded `sel_idx` must equal `argmax(sel_score)` on every window -- if
    it does not, the dumped score is not the score the head ranks on;
  * the clamp is `sel_reach_clamp = False` here (the as-trained path), asserted;
  * the emitted fan's implied max mean speed is reported so it can be checked
    against the independently rescued `fan_last_along_v4.pt` surface.

🔒 PhysicalAI-AV is gated-confidential: only trajectories in the EGO frame,
scores, speeds and episode indices are written.  No clip UUID, no frame.

Usage (pod3):
  PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts \
  python v4_fan_dump.py --ckpt /workspace/v4instr/v4fs_ckpt.pt \
      --cache-dir /workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6 \
      --episodes 44 --stride 8 --out /workspace/v4instr/out
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

WP_STEPS = (5, 10, 15, 20)      # eval_flagship_v4.WP_STEPS -- the ONLY convention


def build(ckpt_path, device, head_config_path):
    """Mirrors ``eval_flagship_v4.load_v4_from_ck`` -- in particular it takes the
    head geometry from the RUN'S OWN ``config.json['head_cfg']``, never from
    ``v4_config()`` defaults.  (v4-from-scratch trained with
    ``cond_imagination = False``; the current defaults have it on, and a default
    head does not even load.)"""
    import dataclasses as dc
    import json as _json

    from tanitad.config import flagship4b_config
    from tanitad.models.flagship_v4 import FlagshipV4Head, V4Config, v4_config
    from tanitad.models.fourbrain import WorldModel
    from tanitad.refs.refc import DecoderConfig

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a_dim = int(ck["model"]["predictor.act_emb.0.weight"].shape[1])
    cfg = flagship4b_config()
    cfg.predictor = dc.replace(cfg.predictor, action_dim=a_dim)
    if cfg.tactical_pred is not None:
        cfg.tactical_pred = dc.replace(cfg.tactical_pred, action_dim=a_dim)
    cfg.encoder = dc.replace(cfg.encoder, grad_checkpoint=False)
    world = WorldModel(cfg)
    world.load_state_dict(ck["model"])                       # STRICT

    if not (head_config_path and os.path.exists(head_config_path)):
        raise SystemExit(
            "REFUSING: --head-config is required. Building the head from "
            "v4_config() defaults risks an architecture mismatch with the run "
            "(v4-from-scratch used cond_imagination=False).")
    hj = _json.loads(open(head_config_path, encoding="utf-8").read())
    hc = dict(hj.get("head_cfg", hj))
    if isinstance(hc.get("decoder"), dict):
        hc["decoder"] = DecoderConfig(**hc["decoder"])
    for tk in ("horizons", "imag_read"):
        if isinstance(hc.get(tk), list):
            hc[tk] = tuple(hc[tk])
    hcfg = V4Config(**hc)
    hcfg.state_dim = world.state_dim
    hcfg.window = cfg.predictor.window
    assert hcfg.sel_reach_clamp is False, (
        "REFUSING: V4Config.sel_reach_clamp is not False. This dump MUST be the "
        "as-trained, unclamped fan or the zero-change test is circular.")
    assert v4_config().sel_reach_clamp is False, (
        "REFUSING: the shipped V4Config default is no longer OFF -- the "
        "escalation this dump exists to settle has already been flipped.")
    head = FlagshipV4Head(hcfg)
    head.load_state_dict(ck["head"])                         # STRICT
    return (world.to(device).eval(), head.to(device).eval(), cfg, hcfg,
            int(ck.get("step", -1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--episodes", type=int, default=44)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--head-config", required=True,
                    help="the RUN's config.json (its head_cfg block)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import refb_labels
    from flagship_v4_data import FlagshipV4Dataset
    from torch.utils.data import default_collate

    from tanitad.data.mixing import load_episode
    from tanitad.train.flagship_losses import horizon_plan
    from train_flagship_v4 import _goal_inputs, _to_device

    t0 = time.time()
    world, head, cfg, hcfg, step = build(a.ckpt, a.device, a.head_config)
    print(f"[fan] step={step} state_dim={hcfg.state_dim} "
          f"n_anchors={hcfg.n_anchors} horizons={hcfg.horizons[0]}.."
          f"{hcfg.horizons[-1]} sel_reach_clamp={hcfg.sel_reach_clamp} "
          f"factorised={hcfg.factorised}", flush=True)

    # byte-identical to train_flagship_v4.train()'s call and to
    # eval_flagship_v4._plan, so the val cache is windowed exactly the
    # way the real run's own in-loop eval windowed it.
    plan = horizon_plan(cfg, op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20)
    files = sorted(f for f in os.listdir(a.cache_dir)
                   if f.startswith("ep_") and f.endswith(".pt"))[: a.episodes]
    eps = [load_episode(os.path.join(a.cache_dir, f), mmap=True) for f in files]
    ds = FlagshipV4Dataset(eps, window=cfg.predictor.window,
                           max_horizon=plan.max_horizon,
                           maneuver_h=plan.maneuver_h,
                           channels=cfg.encoder.in_channels)
    sel = [i for i, (e, t) in enumerate(ds.index)
           if e < a.episodes and t % a.stride == 0]
    print(f"[fan] {len(eps)} eps -> {len(ds)} windows, {len(sel)} selected "
          f"(stride {a.stride})", flush=True)

    horizons = head.cfg.horizons
    FAN, SCORE, SEL, V0, GT, EID, T0 = [], [], [], [], [], [], []
    with torch.no_grad():
        for b0 in range(0, len(sel), a.batch):
            idx = sel[b0:b0 + a.batch]
            b = _to_device(default_collate([ds[i] for i in idx]), a.device)
            v0 = b["pose_last"][:, 3].float()
            traj_tgt = refb_labels.waypoint_targets(
                b["pose_last"].float(),
                b["future_poses"][:, :max(horizons)].float(), horizons)
            st = world.encode_window(b["frames"])
            out = head(st, v0, lambda_plan=1.0,
                       **_goal_inputs(head.cfg, b, v0))
            FAN.append(out["anchor_traj"].float().cpu())      # [b,N,20,2]
            SCORE.append(out["sel_score"].float().cpu())      # [b,N]
            SEL.append(out["sel_idx"].cpu())                  # [b]
            V0.append(v0.cpu())
            GT.append(traj_tgt.float().cpu())                 # [b,20,2]
            for i in idx:
                e, t = ds.index[i]
                EID.append(str(eps[e].episode_id))
                T0.append(int(t))
            if b0 % (a.batch * 10) == 0:
                print(f"  {b0}/{len(sel)} ({time.time()-t0:.0f}s)", flush=True)

    fan = torch.cat(FAN)
    score = torch.cat(SCORE)
    selidx = torch.cat(SEL)
    v0 = torch.cat(V0)
    gt = torch.cat(GT)
    W, N = fan.shape[0], fan.shape[1]

    # ---- fidelity, both directions -----------------------------------------
    argmax_agree = float((score.argmax(1) == selidx).double().mean())
    v_mean = fan[:, :, -1, :].norm(dim=-1) / (horizons[-1] * 0.1)   # [W,N]
    fan_err = (fan - gt[:, None]).norm(dim=-1).mean(-1)              # [W,N]
    wp_pos = [horizons.index(k) for k in WP_STEPS]
    fan_err4 = (fan[:, :, wp_pos, :] - gt[:, None][:, :, wp_pos, :]
                ).norm(dim=-1).mean(-1)                              # [W,N]
    anchors = head.decoder.anchors.detach().float().cpu()
    anc_speed = anchors[:, -1, :].norm(dim=-1) / (horizons[-1] * 0.1)

    torch.save({"fan": fan, "score": score, "sel": selidx, "v0": v0, "gt": gt,
                "eid": EID, "t0": T0, "anchors": anchors,
                "horizons": list(horizons), "step": step},
               os.path.join(a.out, "fan_v4fs_full.pt"))
    torch.save({"v_mean": v_mean, "fan_err": fan_err, "fan_err4": fan_err4,
                "score": score, "sel": selidx, "v0": v0,
                "fan_last_xy": fan[:, :, -1, :].clone(),
                "anchor_mean_speed": anc_speed,
                "eid": EID, "t0": T0, "horizons": list(horizons),
                "step": step, "arm": "flagship-v4-fromscratch"},
               os.path.join(a.out, "fan_v4fs_reduced.pt"))

    meta = {
        "experiment": "GAP 2 stage 1 - v4 per-candidate fan geometry + score",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": "pod3 (A40)", "ckpt": a.ckpt, "ckpt_step": step,
        "arm": "flagship-v4-fromscratch",
        "corpus": {"cache_dir": a.cache_dir, "n_episodes": len(eps),
                   "PARITY": "NON-PARITY key physicalai-val-heldout-79d4e3d2d4c6",
                   "leak_check": "0.0 % poses-sha256 overlap with "
                                 "physicalai-train-e438721ae894 (MEASURED)"},
        "n_windows": W, "n_candidates": N, "stride": a.stride,
        "head": {"n_anchors": hcfg.n_anchors, "horizons": list(horizons),
                 "sel_accel_max": hcfg.sel_accel_max,
                 "sel_reach_clamp_at_dump": hcfg.sel_reach_clamp,
                 "factorised": hcfg.factorised},
        "fidelity": {
            "recorded_sel_equals_argmax_of_dumped_score": round(argmax_agree, 6),
            "PASS": bool(argmax_agree == 1.0)},
        "emitted_fan_speed_stats_mps": {
            "max": round(float(v_mean.max()), 4),
            "p99": round(float(v_mean.flatten().quantile(0.99)), 4),
            "mean": round(float(v_mean.mean()), 4),
            "gt_max": round(float(
                (gt[:, -1, :].norm(dim=-1) / (horizons[-1] * 0.1)).max()), 4)},
        "emitted_fan_speed_stats_kmh": {
            "max": round(float(v_mean.max()) * 3.6, 2),
            "p99": round(float(v_mean.flatten().quantile(0.99)) * 3.6, 2)},
        "anchor_vocabulary_speed_stats_mps": {
            "max": round(float(anc_speed.max()), 4),
            "p99": round(float(anc_speed.quantile(0.99)), 4)},
        "max_last_waypoint_along_track_m": round(
            float(fan[:, :, -1, 0].max()), 4),
        "elapsed_s": round(time.time() - t0, 1),
    }
    fp = os.path.join(a.out, "fan_v4fs_meta.json")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta, indent=1))
    print("wrote", fp)


if __name__ == "__main__":
    main()
