"""X10 end-to-end on REAL eval139 poses: drive the actual V3Dataset hook (OFF vs ON) over every trainer window and
measure (a) the training-target shift the hook really produces, (b) POSE-SYNC (SPEC section 8.10) against the 100 Hz
egomotion log, (c) that nothing except the declared fields moves.

    PYTHONPATH=<overlaid tree>/stack;<tree>/stack/scripts;<tree>/taniteval  python x10_dataset_e2e.py \
        --sidecar ../raw/refcv6_pose_sync_sidecar.jsonl --out ../raw/x10_dataset_e2e.json

The dataset is built from the manifest's poses/actions with STUB frames (the frames are not read by the hook); the
episode ids are the manifest's stable ids.  CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import refc_v3_train as T
import refb_labels as RL
from tanitad.data import pose_sync as PS
from tanitad.data.physicalai import signals_at

EVAL_MAN = "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt"
CAM = "C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov"
EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo"
W, MAXH, RAW_OFF = 8, 20, 2
HOR = (5, 10, 15, 20, 30, 40, 50, 60)


def q(a):
    a = np.asarray(a, np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.median(a)),
            "p95": float(np.quantile(a, 0.95)), "p99": float(np.quantile(a, 0.99)), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1, help="evaluate every k-th window (1 = all)")
    a = ap.parse_args()
    t0 = time.time()
    man = torch.load(EVAL_MAN, map_location="cpu", weights_only=False)
    eps = []
    for i in range(len(man["clip_id"])):
        P = man["poses"][i]
        A = man["actions"][i]
        eps.append(SimpleNamespace(frames=torch.zeros(P.shape[0], 9, 2, 2, dtype=torch.uint8), actions=A, poses=P,
                                   episode_id=int(man["episode_uid"][i])))
    ds_off = T.V3Dataset(eps, window=W, max_horizon=MAXH, channels=9)
    ds_on = T.V3Dataset(eps, window=W, max_horizon=MAXH, channels=9)
    for d in (ds_off, ds_on):
        d.v7_by_sid, d.v7_dt, d.ego_history = {}, 0.1, True
    census = ds_on.enable_pose_sync(a.sidecar)
    print("census", json.dumps({k: census[k] for k in ("n_clips", "n_covered", "n_uncovered_read_unshifted", "sign")}), flush=True)
    n = len(ds_off)
    assert n == 23772, n                                           # the trainer's eval window count (config.json)
    sel = list(range(0, n, a.stride))
    keys_moved = {}
    pl_off, pl_on, fe_off, fe_on, fv, nowrow, eidx = [], [], [], [], [], [], []
    hist_d, act_d, ph_off, ph_on = [], [], [], []
    for i in sel:
        x, y = ds_off[i], ds_on[i]
        assert set(x) == set(y)
        for k in x:
            if torch.is_tensor(x[k]):
                if x[k].dtype.is_floating_point:
                    if not torch.equal(x[k], y[k]):
                        keys_moved[k] = keys_moved.get(k, 0) + 1
                elif not torch.equal(x[k], y[k]):
                    keys_moved[k] = keys_moved.get(k, 0) + 1
            elif x[k] != y[k]:
                keys_moved[k] = keys_moved.get(k, 0) + 1
        pl_off.append(x["pose_last"]); pl_on.append(y["pose_last"])
        fe_off.append(x["future_poses_ext"]); fe_on.append(y["future_poses_ext"]); fv.append(x["future_valid_ext"])
        e_i, t = ds_off.index[i]
        nowrow.append(t + W - 1); eidx.append(e_i)
        hist_d.append(float((y["pose_hist"][:, :2] - x["pose_hist"][:, :2]).norm(dim=-1).max()))
        ph_off.append(x["pose_hist"]); ph_on.append(y["pose_hist"])
        act_d.append(float((y["actions"] - x["actions"]).abs().max()))
    plo, plon = torch.stack(pl_off), torch.stack(pl_on)
    feo, feon, fvv = torch.stack(fe_off), torch.stack(fe_on), torch.stack(fv)
    wo = RL.waypoint_targets(plo, feo, HOR)                       # THE TRAINER'S FUNCTION, on the hook's two outputs
    wn = RL.waypoint_targets(plon, feon, HOR)
    d = (wn - wo).norm(dim=-1).numpy()                            # [N, 8]
    valid = torch.stack([fvv[:, h - 1] for h in HOR], dim=1).numpy()
    byh = {str(h): q(d[:, k][valid[:, k]]) for k, h in enumerate(HOR)}
    # pooled 0-2 s over ticks 1..20
    allh = RL.waypoint_targets(plo, feo, tuple(range(1, 21)))
    allhn = RL.waypoint_targets(plon, feon, tuple(range(1, 21)))
    pooled02 = (allhn - allh).norm(dim=-1).numpy()
    v0_d = (plon[:, 3] - plo[:, 3]).abs().numpy()
    from tanitad.models.ego_history import ego_channels_from_poses     # what the ego-history ENCODER actually reads
    eo = ego_channels_from_poses(torch.stack(ph_off), W); en = ego_channels_from_poses(torch.stack(ph_on), W)
    ego_ch = {c: q((en[..., k] - eo[..., k]).abs().reshape(-1).numpy()) for k, c in enumerate(("speed_ms", "dspeed_per_step", "yaw_rate_rad_s"))}
    from tanitad.refs import refc_tactical as tac            # the trainer's kinematic factored labels (refc_v3_train.py:4916)
    lat_o, lon_o = tac.window_factored_labels(plo, feo[:, :20])
    lat_n, lon_n = tac.window_factored_labels(plon, feon[:, :20])
    flips = {"lat_flip_frac": float((lat_o != lat_n).float().mean()), "lon_flip_frac": float((lon_o != lon_n).float().mean()),
             "n_windows": int(lat_o.numel()), "lat_flips": int((lat_o != lat_n).sum()), "lon_flips": int((lon_o != lon_n).sum())}
    go, gn = torch.stack([ds_off[i]["goal_tac"] for i in sel[::8]]), torch.stack([ds_on[i]["goal_tac"] for i in sel[::8]])
    goal = {"goal_tac_xy_shift_m": q((gn[..., :2] - go[..., :2]).norm(dim=-1).reshape(-1).numpy()), "n_windows": int(go.shape[0])}
    res = {"n_windows_evaluated": len(sel), "census": {k: census[k] for k in ("n_clips", "n_covered", "n_uncovered_read_unshifted", "sign", "fields_shifted")},
           "keys_that_changed_at_least_once": keys_moved,
           "target_shift_by_horizon_ticks__hook_U_to_B_m": byh,
           "target_shift_pooled_0_2s_m": q(pooled02),
           "v0_shift_ms": q(v0_d),
           "kinematic_factored_label_flips": flips,
           "ego_history_encoder_input_channel_shift_abs": ego_ch,
           "goal_tac_shift": goal,
           "pose_hist_max_row_pos_shift_m": q(hist_d),
           "actions_max_abs_shift": q(act_d)}

    # ---- POSE-SYNC against the 100 Hz log (SPEC 8.10): spatial residual of pose_last vs the pose at the IMAGE instant
    resid_off, resid_on, speeds = [], [], []
    by_clip = {}
    for j, (e_i, r) in enumerate(zip(eidx, nowrow)):
        by_clip.setdefault(e_i, []).append((j, r))
    for e_i, lst in by_clip.items():
        cid = man["clip_id"][e_i]
        ts = pd.read_parquet(f"{CAM}/{cid}.timestamps.parquet")
        tf = ts[next(c for c in ts.columns if "time" in c.lower())].to_numpy(np.float64)
        tq, fi, unit = PS.resample_grid(tf)
        dl, dt, nt = PS.grid_delta(tf)
        ego = pd.read_parquet(f"{EGO}/{cid}.parquet")
        rows = np.array([r for _, r in lst]) + RAW_OFF
        t_img = tq[rows] + dl[rows] * unit                              # the instant the NOW image was captured
        _, Pt = signals_at(ego, t_img)
        Pt = Pt.astype(np.float64)
        for (j, r), pt in zip(lst, Pt):
            v = float(plo[j, 3])
            if v < 2.0:
                continue
            resid_off.append(float(np.hypot(*(plo[j, :2].numpy().astype(np.float64) - pt[:2]))) / v * 1e3)
            resid_on.append(float(np.hypot(*(plon[j, :2].numpy().astype(np.float64) - pt[:2]))) / v * 1e3)
            speeds.append(v)
    res["POSE_SYNC_ms__equivalent_timing_residual_of_pose_last_vs_image_instant_truth"] = {
        "definition": "||pose_last - pose_100Hz(t_image)|| / speed, windows with v > 2 m/s (the position residual expressed as a time)",
        "uncorrected_OFF": q(resid_off), "corrected_ON": q(resid_on), "n_windows_v_gt_2": len(resid_on)}
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1)
    print("WROTE", a.out, res["wall_s"], "s", flush=True)


if __name__ == "__main__":
    main()
