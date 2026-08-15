"""X0-lite (W1+W2 of V58F_FUSION): full-fan dump, feasibility census, kinematic re-rank.

PRE-REGISTERED (DIFFUSION_MPC_SYNTHESIS X0 + fusion W1/W2, gates fixed before run):
  W1 gate: kinematic-cost re-rank of the selector's top-8 closes >= 30 % of sel_gap.
  W2 output: fan feasibility census on v5f's OWN 30k fan (the REF-C-XL 72 % question).
Both from ONE GPU pass over the same 881-window grid as the banked eval (episodes/stride
defaults identical). No WM rolls (that is X0-full/W7); costs are kinematic only:
  cost(c) = mean|accel(c)| + 0.5*mean|jerk(c)| + 4*infeasible_frac(c)
  infeasible step: |a| > 4.0 m/s2 (envelope; human p99 2.78) OR |yaw_rate| > 0.33*v+0.05.
Outputs /workspace/x0_lite.json + full dump /workspace/x0_fan_dump.npz (f16, ~40 MB).
"""
import json
import os
import sys
import types
import pathlib

import numpy as np

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")

DT = 0.1


def main():
    import torch
    from torch.utils.data.dataloader import default_collate
    import eval_flagship_v4 as ev
    import refb_labels
    import goal_modes

    torch.set_grad_enabled(False)
    ns = types.SimpleNamespace(
        v2_val_cache=["/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl"],
        v2_subframe="176x624", v2_lru=64, frame_h=256, frame_w=640,
        frame_hfov=120.0, f_ref=None, projection="cylindrical", val_cache=None)
    cfg = ev._eval_cfg()
    cache_frame, train_frame = ev.resolve_eval_frames(ns, cfg)
    ck = torch.load("/workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt",
                    map_location="cpu", weights_only=False)
    world, grounding, head, step, hcfg, goal_head = ev.load_v4_from_ck(
        ck, "cuda", frame=train_frame)
    eps, _prov = ev.load_val_episodes(ns, cache_frame=cache_frame,
                                      train_frame=train_frame)
    plan = ev._plan(cfg)
    ds_val = ev.build_val_dataset_v4(eps, cfg, plan)
    _pv = pathlib.Path("/workspace/experiments/flagship-v5f-w120-30k/probe_vocab.pt")
    _probes = torch.load(_pv, map_location="cuda") if _pv.exists() else None
    from train_flagship_v4 import _imagination_inputs

    # SAME selection as the banked eval (defaults episodes=40 stride=8 -> 881 windows):
    sel = [i for i, (e, t) in enumerate(ds_val.index) if e < 40 and t % 8 == 0]
    print(f"[x0] grid {len(sel)} windows (expect 881)", flush=True)
    FANS, SCORES, GTS, V0S, SELS = [], [], [], [], []
    horizons = list(range(1, 21))
    for b0 in range(0, len(sel), 16):
        items = [ds_val[i] for i in sel[b0:b0 + 16]]
        b = default_collate(items)
        b = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in b.items()}
        v0 = b["pose_last"][:, 3].float()
        gt = refb_labels.waypoint_targets(
            b["pose_last"].float(), b["future_poses"][:, :20].float(), horizons)
        st = world.encode_window(b["frames"].float())
        extra = _imagination_inputs(world, head.cfg, b, st, probes=_probes) \
            if _probes is not None else {}
        gkw, _ = goal_modes.resolve_goal("oracle", head=head, batch=b, v0=v0,
                                         states=st, goal_head=goal_head,
                                         allow_fallback=True, oracle_channels=None)
        out = head(st, v0, lambda_plan=1.0, **extra, **gkw)
        fan = out["anchor_traj"][..., :2]
        sc = None
        for k in ("sel_score", "refined_logits", "anchor_logits"):
            if k in out:
                sc = out[k].float()
                break
        if sc is None:
            raise SystemExit(f"X0_FAIL: no score key in {list(out.keys())[:14]}")
        FANS.append(fan.float().cpu().numpy())    # FLOAT32 in census (f16 storage
                                                  # inflates second differences)
        SCORES.append(sc.cpu().numpy())
        GTS.append(gt[..., :2].cpu().numpy())
        V0S.append(v0.cpu().numpy())
        SELS.append(out["sel_idx"].cpu().numpy())
        if (b0 // 16) % 10 == 0:
            print(f"[x0] {b0}/{len(sel)}", flush=True)
    fan = np.concatenate(FANS).astype(np.float32)      # [N,256,20,2]
    sc = np.concatenate(SCORES)
    gt = np.concatenate(GTS)
    v0 = np.concatenate(V0S)
    sel_idx = np.concatenate(SELS).astype(np.int64)
    np.savez_compressed("/workspace/x0_fan_dump.npz", fan=fan.astype(np.float16),
                        scores=sc.astype(np.float16), gt=gt.astype(np.float16),
                        v0=v0.astype(np.float32), sel_idx=sel_idx)
    print(f"[x0] dump saved {fan.shape}", flush=True)

    # ---- kinematics of every candidate -----------------------------------------
    p = np.concatenate([np.zeros((*fan.shape[:2], 1, 2), np.float32), fan], axis=2)
    d = np.diff(p, axis=2)
    sp = np.linalg.norm(d, axis=-1) / DT               # [N,C,20]
    acc = np.diff(sp, axis=2) / DT                     # [N,C,19]
    jerk = np.diff(acc, axis=2) / DT
    h = np.arctan2(d[..., 1], d[..., 0])
    dh = (np.diff(h, axis=2) + np.pi) % (2 * np.pi) - np.pi
    yr = np.abs(dh) / DT                               # [N,C,19]
    vmid = np.maximum(sp[..., 1:], 0.3)
    infeas_step = (np.abs(acc) > 4.0) | (yr > 0.33 * vmid + 0.05)
    infeas_frac = infeas_step.mean(axis=2)             # [N,C]
    cand_infeasible = (infeas_frac > 0.05)
    ade = np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)   # [N,C]

    selector = sel_idx                     # the head's ACTUAL deployed pick
    N = fan.shape[0]
    rows = np.arange(N)
    sel_ade = ade[rows, selector]
    oracle_ade = ade.min(1)
    top8 = np.argsort(-sc, axis=1)[:, :8]
    top8_oracle = ade[rows[:, None], top8].min(1)
    cost = (np.abs(acc).mean(2) + 0.5 * np.abs(jerk).mean(2) + 4.0 * infeas_frac)
    cost8 = cost[rows[:, None], top8]
    pick = top8[rows, cost8.argmin(1)]
    rerank_ade = ade[rows, pick]

    # ---- W2b: 3-tap smoother probe (post-hoc, zero-train fix candidate) --------
    fs = fan.copy()
    fs[:, :, 1:-1] = 0.25 * fan[:, :, :-2] + 0.5 * fan[:, :, 1:-1] + 0.25 * fan[:, :, 2:]
    ps = np.concatenate([np.zeros((*fs.shape[:2], 1, 2), np.float32), fs], axis=2)
    dsm = np.diff(ps, axis=2)
    sps = np.linalg.norm(dsm, axis=-1) / DT
    accs = np.diff(sps, axis=2) / DT
    hs = np.arctan2(dsm[..., 1], dsm[..., 0])
    dhs = (np.diff(hs, axis=2) + np.pi) % (2 * np.pi) - np.pi
    yrs = np.abs(dhs) / DT
    vmids = np.maximum(sps[..., 1:], 0.3)
    infeas_s = (np.abs(accs) > 4.0) | (yrs > 0.33 * vmids + 0.05)
    ade_s = np.linalg.norm(fs - gt[:, None], axis=-1).mean(-1)
    cost_s = (np.abs(accs).mean(2) + 0.5 * np.abs(np.diff(accs, axis=2) / DT).mean(2)
              + 4.0 * infeas_s.mean(2))
    cost8_s = cost_s[np.arange(N)[:, None], top8]
    pick_s = top8[np.arange(N), cost8_s.argmin(1)]

    res = {
        "n_windows": int(N), "n_candidates": int(fan.shape[1]),
        "fan_dtype_in_census": "float32",
        "W2_feasibility_census": {
            "candidate_infeasible_frac (>5% bad steps)": float(cand_infeasible.mean()),
            "step_infeasible_frac": float(infeas_step.mean()),
            "selected_candidate_infeasible_frac": float(cand_infeasible[rows, selector].mean()),
            "oracle_candidate_infeasible_frac": float(cand_infeasible[rows, ade.argmin(1)].mean()),
            "fan_accel_mae_all": float(np.abs(acc).mean()),
            "_rule": "|a|>4.0 or |yr|>0.33v+0.05; candidate infeasible if >5% steps"},
        "W1_kinematic_rerank": {
            "selector_ade": float(sel_ade.mean()),
            "oracle_ade_full": float(oracle_ade.mean()),
            "oracle_ade_top8": float(top8_oracle.mean()),
            "rerank_ade": float(rerank_ade.mean()),
            "sel_gap_before": float(sel_ade.mean() - oracle_ade.mean()),
            "sel_gap_after": float(rerank_ade.mean() - oracle_ade.mean()),
            "gap_closed_frac": float((sel_ade.mean() - rerank_ade.mean()) /
                                     max(sel_ade.mean() - oracle_ade.mean(), 1e-9)),
            "_gate": "PRE-REGISTERED W1: gap_closed_frac >= 0.30",
            "_note": "kinematic cost only; top-8 pruning bounds it by oracle_ade_top8"},
        "W2b_smoother_probe": {
            "_what": "3-tap [.25 .5 .25] positional smoother, post-hoc, zero-train",
            "sel_ade_smoothed": float(ade_s[rows, selector].mean()),
            "oracle_ade_smoothed": float(ade_s.min(1).mean()),
            "sel_accel_mae_raw": float(np.abs(acc[rows, selector]).mean()),
            "sel_accel_mae_smoothed": float(np.abs(accs[rows, selector]).mean()),
            "sel_yawrate_mae_raw_deg_s": float(np.degrees(yr[rows, selector].mean())),
            "sel_yawrate_mae_smoothed_deg_s": float(np.degrees(yrs[rows, selector].mean())),
            "sel_infeas_frac_smoothed": float(infeas_s[rows, selector].mean()),
            "rerank_smoothed_ade": float(ade[rows, pick_s].mean()),
            "gap_closed_frac_smoothed_cost": float(
                (sel_ade.mean() - ade[rows, pick_s].mean()) /
                max(sel_ade.mean() - oracle_ade.mean(), 1e-9))},
    }
    json.dump(res, open("/workspace/x0_lite.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    print("X0LITE_DONE", flush=True)


if __name__ == "__main__":
    main()
