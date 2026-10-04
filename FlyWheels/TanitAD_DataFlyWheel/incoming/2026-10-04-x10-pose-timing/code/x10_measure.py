"""X10 measurement -- the pose-to-image offset and the size of the training-target shift it causes.  CPU only.

    PYTHONPATH=<clean tree>/stack;<clean tree>/stack/scripts;<clean tree>/taniteval \
      python x10_measure.py --out ../raw/x10_measure.json [--n-train 400] [--seed 20261004]

Inputs (all LOCAL, read-only; no G:, no Thor, no GPU):
  * the cache manifests (poses as the trainer reads them):
      D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt           (139 clips)
      <scratch>/d3/train_v2manifest.pt  (copy of Thor refcv6-b1-416x1024-train)   (4,369 clips)
  * the raw camera timestamps  C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov/<cid>.timestamps.parquet
  * the raw 100 Hz egomotion   C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/<cid>.parquet

Three target definitions are compared on the SAME windows (window 8, max_horizon 20, NOW row r = t + 7):
  U  uncorrected     -- exactly what refcv7 trained on: refb_labels.waypoint_targets(pose_last, future_poses_ext)
  B  window shift    -- the whole window re-sampled delta_r LATER (the proposed correction)
  A  per-row shift   -- every row re-sampled by its OWN delta_j (the literal "interpolate to each camera frame"),
                        priced here ONLY to show why it is NOT the correction
and against  Bt = B computed from the 100 Hz egomotion log (the interpolation-error control).

Every id that reaches the output is sha12 of the clip id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from tanitad.data import pose_sync as PS
from tanitad.data.physicalai import signals_at
from tanitad.data.v2_dataset import stable_episode_id
import refb_labels as RL

CAM = "C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov"
EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo"
EVAL_MAN = "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt"
# As run (md5 02e22e9d8138e2f69718bd828a36d79d), this pointed at a copy of the train-a6 _v2manifest.pt in a Claude
# session scratchpad (session-specific path, not banked). Landed copy: the same file via an env var (Master Mind, 2026-10-04).
TRAIN_MAN = os.environ.get("X10_TRAIN_V2MANIFEST", "<path to the train-a6 _v2manifest.pt>")
CLOCK = "D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl"
W, MAXH, RAW_OFF = 8, 20, 2          # refcv7 launch: window 8, max_horizon 20, n_stack 3 (config.json / V3Dataset)
HOR = (5, 10, 15, 20, 30, 40, 50, 60)   # v3.V3_HORIZONS == config.json "horizons"
H_ALL = np.arange(1, 61)


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def egof(dxy, yaw):
    """numpy twin of refb_labels.ego_frame (checked against it by control K1)."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.stack([dxy[..., 0] * c - dxy[..., 1] * s, dxy[..., 0] * s + dxy[..., 1] * c], axis=-1)


def targets(pose_now, pose_h):
    """pose_now [N,4], pose_h [N,K,4] -> ego-frame displacement [N,K,2]  (waypoint_targets' formula)."""
    return egof(pose_h[..., :2] - pose_now[:, None, :2], pose_now[:, None, 2])


def quant(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    q = np.quantile(a, [0.05, 0.5, 0.95, 0.99])
    return {"n": int(a.size), "mean": float(a.mean()), "p05": float(q[0]), "p50": float(q[1]),
            "p95": float(q[2]), "p99": float(q[3]), "max": float(a.max()), "min": float(a.min())}


def cluster_boot_mean(sums, cnts, B=1000, seed=0):
    """Episode-cluster bootstrap of a pooled mean (clips = clusters).  sums/cnts per clip."""
    sums, cnts = np.asarray(sums, np.float64), np.asarray(cnts, np.float64)
    rng = np.random.default_rng(seed)
    n = len(sums)
    idx = rng.integers(0, n, size=(B, n))
    m = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def load_clip(cid):
    ts = pd.read_parquet(f"{CAM}/{cid}.timestamps.parquet")
    tcol = next(c for c in ts.columns if "time" in c.lower())
    t_frames = ts[tcol].to_numpy(np.float64)
    ego = pd.read_parquet(f"{EGO}/{cid}.parquet")
    return t_frames, ego


def controls():
    """K1..K3 -- must hold before any real clip is scored."""
    out = {}
    # K1 numpy ego_frame == refb_labels.waypoint_targets (the trainer's function) on random float32 poses
    rng = np.random.default_rng(1)
    pl = rng.normal(size=(5, 4)).astype(np.float32)
    fp = rng.normal(size=(5, 60, 4)).astype(np.float32)
    ref = RL.waypoint_targets(torch.from_numpy(pl), torch.from_numpy(fp), HOR).numpy()
    mine = targets(pl.astype(np.float64), fp.astype(np.float64))[:, [h - 1 for h in HOR], :]
    out["K1_numpy_target_equals_trainer_waypoint_targets_maxabs"] = float(np.abs(ref - mine).max())
    # K2 the grid replica reproduces a SYNTHETIC camera at a KNOWN offset: 30 fps from t=0, grid 10 Hz-ish
    tf = np.arange(0, 20.0e6 + 1, 1e6 / 30.0).round()
    d, dt, n = PS.grid_delta(tf)
    out["K2_synth_delta0_us"] = float(d[0] * 1e6)
    out["K2_synth_delta_range_ms"] = [float(d.min() * 1e3), float(d.max() * 1e3)]
    out["K2_synth_dt_s"], out["K2_synth_n_target"] = dt, n
    da = PS.analytic_delta_s(tf[0], tf[-1], len(tf), n)
    out["K2_synth_analytic_vs_grid_maxabs_us"] = float(np.abs(d - da).max() * 1e6)
    # K3 a constant-velocity track shifted by 16.5 ms moves by v*0.0165 EXACTLY (literal expected value)
    dtq = 0.1006666
    v = 12.0
    P = np.zeros((40, 4)); P[:, 0] = v * dtq * np.arange(40); P[:, 3] = v
    Ps = PS.shift_rows(P, 0.0165 / dtq, angle_cols=(2,))
    out["K3_cv_shift_err_m"] = float(np.abs(Ps[:, 0] - (P[:, 0] + v * 0.0165))[:-1].max())
    out["K3_expected_shift_m"] = v * 0.0165
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--limit-eval", type=int, default=0)
    a = ap.parse_args()
    t_start = time.time()
    ctl = controls()
    print("controls:", json.dumps(ctl), flush=True)
    assert ctl["K1_numpy_target_equals_trainer_waypoint_targets_maxabs"] < 2e-5
    assert abs(ctl["K2_synth_delta0_us"]) < 1e-6 and ctl["K2_synth_analytic_vs_grid_maxabs_us"] < 5.0
    assert ctl["K3_cv_shift_err_m"] < 1e-9

    man_e = torch.load(EVAL_MAN, map_location="cpu", weights_only=False)
    man_t = torch.load(TRAIN_MAN, map_location="cpu", weights_only=False)
    clock = {}
    for line in open(CLOCK, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            clock[int(r["sid"])] = (float(r["grid_start_s"]), float(r["dt_s"]))

    # ---------- (1) the offset over EVERY clip of both splits (camera parquet only: cheap) ----------
    corpus = {}
    for split, man in (("eval139", man_e), ("train4369", man_t)):
        dl, miss, nmis, dts, grid_ok = [], 0, 0, [], 0
        pos_rows = 0
        for i, cid in enumerate(man["clip_id"]):
            f = f"{CAM}/{cid}.timestamps.parquet"
            if not os.path.isfile(f):
                miss += 1
                continue
            ts = pd.read_parquet(f)
            tf = ts[next(c for c in ts.columns if "time" in c.lower())].to_numpy(np.float64)
            d, dt, n = PS.grid_delta(tf)
            if n - RAW_OFF != int(man["T_out"][i]):
                nmis += 1
                continue
            grid_ok += 1
            dl.append(d[RAW_OFF:])           # provider rows only (the rows the trainer can ever read)
            dts.append(dt)
        allD = np.concatenate(dl) * 1e3
        corpus[split] = {"n_clips_in_manifest": len(man["clip_id"]), "n_missing_camera_ts": miss,
                         "n_grid_rows_mismatch": nmis, "n_clips_scored": grid_ok,
                         "n_rows": int(allD.size), "delta_ms": quant(allD),
                         "dt_s_median": float(np.median(dts)), "dt_s_min": float(np.min(dts)),
                         "dt_s_max": float(np.max(dts))}
        print(split, json.dumps(corpus[split]), flush=True)

    # ---------- (2) the target shift on eval139 + a seeded train sample ----------
    rng = np.random.default_rng(a.seed)
    jobs = []
    ev_ids = list(man_e["clip_id"])[: (a.limit_eval or None)]
    ev_idx = {c: i for i, c in enumerate(man_e["clip_id"])}
    for c in ev_ids:
        jobs.append(("eval139", c, man_e["poses"][ev_idx[c]].numpy().astype(np.float64), int(man_e["episode_uid"][ev_idx[c]])))
    tr_all = list(man_t["clip_id"])
    tr_idx = {c: i for i, c in enumerate(tr_all)}
    pick = sorted(rng.choice(len(tr_all), size=min(a.n_train, len(tr_all)), replace=False).tolist())
    for j in pick:
        c = tr_all[j]
        jobs.append(("train_sample", c, man_t["poses"][j].numpy().astype(np.float64), int(man_t["episode_uid"][j])))

    acc = {s: {k: [] for k in ("dU_B", "dxU_B", "dyU_B", "dA_B", "dA_U", "dI", "valid", "now_v", "v_shift",
                               "yaw_shift", "pos_shift", "pos_shift_true", "v_shift_true", "v_interp_err",
                               "yaw_interp_err", "pos_interp_err")} for s in ("eval139", "train_sample")}
    per_clip = {s: [] for s in acc}
    recon = {"n": 0, "maxabs_xy_m": 0.0, "maxabs_yaw_rad": 0.0, "maxabs_v_ms": 0.0, "n_fail": 0}
    clockx = {"n": 0, "grid_start_abs_diff_ms": [], "dt_abs_diff_s": []}
    anax = {"n": 0, "maxabs_us": [], "n_flip_rows": 0, "n_rows": 0, "period_us": []}
    skipped = 0
    for (split, cid, P, sid_man) in jobs:
        try:
            t_frames, ego = load_clip(cid)
        except Exception as e:                                    # noqa: BLE001
            skipped += 1
            continue
        t_query, frame_idx, unit = PS.resample_grid(t_frames)
        delta_s, dt_s, n_target = PS.grid_delta(t_frames)
        T = len(P)
        if n_target - RAW_OFF != T:
            skipped += 1
            continue
        assert stable_episode_id(cid) == sid_man, "sid mismatch with manifest episode_uid"
        # --- reconstruct the cache's poses from raw logs: proves t_query/the builder replica ---
        _, pr = signals_at(ego, t_query)
        pr = pr.astype(np.float64)
        dxy = np.abs(pr[RAW_OFF:, :2] - P[:, :2]).max()
        dya = np.abs(np.arctan2(np.sin(pr[RAW_OFF:, 2] - P[:, 2]), np.cos(pr[RAW_OFF:, 2] - P[:, 2]))).max()
        dv = np.abs(pr[RAW_OFF:, 3] - P[:, 3]).max()
        recon["n"] += 1
        recon["maxabs_xy_m"] = max(recon["maxabs_xy_m"], float(dxy))
        recon["maxabs_yaw_rad"] = max(recon["maxabs_yaw_rad"], float(dya))
        recon["maxabs_v_ms"] = max(recon["maxabs_v_ms"], float(dv))
        if dxy > 1e-3:
            recon["n_fail"] += 1
        # --- cross-checks of the offset itself ---
        te0 = np.sort(ego["timestamp"].to_numpy(np.float64))[0]
        if int(sid_man) in clock:
            g0, dtc = clock[int(sid_man)]
            clockx["n"] += 1
            clockx["grid_start_abs_diff_ms"].append(abs((t_frames[0] - te0) / 1e6 - g0) * 1e3)
            clockx["dt_abs_diff_s"].append(abs(dt_s - dtc))
        da = PS.analytic_delta_s(t_frames[0], t_frames[-1], len(t_frames), n_target)
        per_us = (t_frames[-1] - t_frames[0]) / (len(t_frames) - 1)          # ideal camera period, us
        raw_d = (delta_s - da) * 1e6
        wrapped = (raw_d + per_us / 2.0) % per_us - per_us / 2.0              # modulo ONE frame period
        anax["n"] += 1
        anax["maxabs_us"].append(float(np.abs(wrapped).max()))
        anax["n_flip_rows"] += int((np.abs(raw_d) > per_us / 2.0).sum())      # rows where ceil() and searchsorted pick adjacent frames
        anax["n_rows"] += int(raw_d.size)
        anax["period_us"].append(float(per_us))

        # --- windows exactly as V3Dataset indexes them ---
        t_max = T - W - MAXH
        now = np.arange(t_max) + W - 1                                   # provider rows
        s_now = delta_s[now + RAW_OFF] / dt_s                            # fractional rows
        s_row = delta_s[RAW_OFF:] / dt_s                                 # per provider row (for A)
        nw = len(now)
        rows_h = now[:, None] + H_ALL[None, :]                           # [nw,60]
        valid = rows_h <= (T - 1)
        # U
        Pu_now = P[now]
        Pu_h = P[np.minimum(rows_h, T - 1)]
        tU = targets(Pu_now, Pu_h)
        # B (window shift)
        uB_now = now + s_now
        uB_h = rows_h + s_now[:, None]
        PB_now = PS.sample_rows(P, uB_now, (2,))
        PB_h = PS.sample_rows(P, uB_h.reshape(-1), (2,)).reshape(nw, 60, 4)
        tB = targets(PB_now, PB_h)
        # A (per-row shift)
        Pa = PS.shift_rows_per_row(P, s_row, (2,))
        tA = targets(Pa[now], Pa[np.minimum(rows_h, T - 1)])
        # Bt: B straight from the 100 Hz log
        tq_now = t_query[now + RAW_OFF] + delta_s[now + RAW_OFF] * unit
        tq_h = t_query[np.minimum(rows_h + RAW_OFF, n_target - 1)] + (delta_s[now + RAW_OFF] * unit)[:, None]
        _, Pt_now = signals_at(ego, tq_now)
        _, Pt_h = signals_at(ego, tq_h.reshape(-1))
        Pt_now, Pt_h = Pt_now.astype(np.float64), Pt_h.astype(np.float64).reshape(nw, 60, 4)
        tBt = targets(Pt_now, Pt_h)
        # also the uncorrected-time truth for the pose offset itself: pose at t_query vs at t_image
        _, Pq_all = signals_at(ego, t_query[RAW_OFF:])
        _, Pi_all = signals_at(ego, t_query[RAW_OFF:] + delta_s[RAW_OFF:] * unit)
        Pq_all, Pi_all = Pq_all.astype(np.float64), Pi_all.astype(np.float64)

        a_ = acc[split]
        vm = valid
        a_["dU_B"].append(np.linalg.norm(tB - tU, axis=-1)[vm])
        a_["dxU_B"].append((tB - tU)[..., 0][vm])
        a_["dyU_B"].append((tB - tU)[..., 1][vm])
        a_["dA_B"].append(np.linalg.norm(tA - tB, axis=-1)[vm])
        a_["dA_U"].append(np.linalg.norm(tA - tU, axis=-1)[vm])
        a_["dI"].append(np.linalg.norm(tB - tBt, axis=-1)[vm])
        a_["valid"].append(np.broadcast_to(H_ALL[None, :], vm.shape)[vm])      # horizon label per entry
        a_["v_shift"].append(np.abs(PB_now[:, 3] - Pu_now[:, 3]))
        a_["yaw_shift"].append(np.abs(np.arctan2(np.sin(PB_now[:, 2] - Pu_now[:, 2]), np.cos(PB_now[:, 2] - Pu_now[:, 2]))))
        a_["pos_shift"].append(np.hypot(*(PB_now[:, :2] - Pu_now[:, :2]).T))
        a_["now_v"].append(Pu_now[:, 3])
        a_["pos_shift_true"].append(np.hypot(*(Pi_all[:, :2] - Pq_all[:, :2]).T))
        a_["v_shift_true"].append(np.abs(Pi_all[:, 3] - Pq_all[:, 3]))
        a_["v_interp_err"].append(np.abs(PB_now[:, 3] - Pt_now[:, 3]))
        a_["yaw_interp_err"].append(np.abs(np.arctan2(np.sin(PB_now[:, 2] - Pt_now[:, 2]), np.cos(PB_now[:, 2] - Pt_now[:, 2]))))
        a_["pos_interp_err"].append(np.hypot(*(PB_now[:, :2] - Pt_now[:, :2]).T))
        # per-clip per-horizon sums for the cluster bootstrap
        nrm = np.linalg.norm(tB - tU, axis=-1)
        per_clip[split].append({"sha12": sha12(cid), "n_windows": int(nw),
                                "sum_by_h": [(float(np.where(vm[:, h - 1], nrm[:, h - 1], 0).sum())) for h in HOR],
                                "cnt_by_h": [int(vm[:, h - 1].sum()) for h in HOR],
                                "sum_0_2s": float(np.where(vm[:, :20], nrm[:, :20], 0).sum()),
                                "cnt_0_2s": int(vm[:, :20].sum()),
                                "sum_0_6s": float(np.where(vm, nrm, 0).sum()), "cnt_0_6s": int(vm.sum())})
        if (len(per_clip["eval139"]) + len(per_clip["train_sample"])) % 100 == 0:
            print(f"  {len(per_clip['eval139'])} eval / {len(per_clip['train_sample'])} train clips  "
                  f"{time.time() - t_start:.0f}s", flush=True)

    # ---------- (3) summarise ----------
    res = {"controls": ctl, "corpus_delta": corpus, "skipped_clips": skipped,
           "reconstruction_of_cached_poses_from_raw_logs": recon,
           "clock_sidecar_crosscheck": {"n": clockx["n"],
                                        "grid_start_abs_diff_ms": quant(clockx["grid_start_abs_diff_ms"]),
                                        "dt_abs_diff_s": quant(clockx["dt_abs_diff_s"])},
           "analytic_delta_crosscheck": {"what": "delta from (t_first, t_last, n_frames, n_target) with an IDEAL constant-period camera vs delta from the real per-frame timestamps; difference taken modulo one frame period (a flip of ceil() vs searchsorted at a boundary is a whole-period jump, not a disagreement)",
                                      "per_clip_maxabs_wrapped_us": quant(anax["maxabs_us"]), "n_flip_rows": anax["n_flip_rows"],
                                      "n_rows": anax["n_rows"], "ideal_period_us": quant(anax["period_us"])},
           "window_geometry": {"window": W, "max_horizon": MAXH, "raw_offset": RAW_OFF,
                               "horizons_ticks": list(HOR), "tick_s_nominal": 0.1,
                               "tick_s_median": corpus["train4369"]["dt_s_median"]},
           "splits": {}}
    for s, a_ in acc.items():
        if not a_["dU_B"]:
            continue
        cat = {k: np.concatenate(v) for k, v in a_.items()}
        hlab = cat["valid"]
        out = {"n_clips": len(per_clip[s]), "n_windows": int(sum(c["n_windows"] for c in per_clip[s]))}
        # per-window offset metrics
        out["now_pose_shift_m__B_minus_U"] = quant(cat["pos_shift"])
        out["now_pose_shift_m__truth_100Hz"] = quant(cat["pos_shift_true"])
        out["now_speed_shift_ms__B_minus_U"] = quant(cat["v_shift"])
        out["now_speed_shift_ms__truth_100Hz"] = quant(cat["v_shift_true"])
        out["now_yaw_shift_rad__B_minus_U"] = quant(cat["yaw_shift"])
        out["interp_error_at_now"] = {"pos_m": quant(cat["pos_interp_err"]), "speed_ms": quant(cat["v_interp_err"]),
                                      "yaw_rad": quant(cat["yaw_interp_err"])}
        out["now_speed_ms"] = quant(cat["now_v"])
        # per-horizon target shift
        byh = {}
        for k, h in enumerate(HOR):
            m = hlab == h
            sums = [c["sum_by_h"][k] for c in per_clip[s]]
            cnts = [c["cnt_by_h"][k] for c in per_clip[s]]
            byh[str(h)] = {"t_s_nominal": h * 0.1,
                           "U_to_B_norm_m": quant(cat["dU_B"][m]),
                           "U_to_B_norm_mean_ci95_cluster": cluster_boot_mean(sums, cnts),
                           "U_to_B_longitudinal_signed_m": {"mean": float(cat["dxU_B"][m].mean()),
                                                            "abs_mean": float(np.abs(cat["dxU_B"][m]).mean())},
                           "U_to_B_lateral_abs_mean_m": float(np.abs(cat["dyU_B"][m]).mean()),
                           "A_to_B_norm_m": quant(cat["dA_B"][m]), "A_to_U_norm_m": quant(cat["dA_U"][m]),
                           "interp_error_B_vs_100Hz_truth_m": quant(cat["dI"][m]),
                           "frac_U_to_B_gt_0p05m": float((cat["dU_B"][m] > 0.05).mean()),
                           "frac_U_to_B_gt_0p10m": float((cat["dU_B"][m] > 0.10).mean())}
        out["by_horizon_ticks"] = byh
        m2 = hlab <= 20
        out["pooled_0_2s"] = {"U_to_B_norm_m": quant(cat["dU_B"][m2]),
                              "U_to_B_norm_mean_ci95_cluster": cluster_boot_mean([c["sum_0_2s"] for c in per_clip[s]],
                                                                                 [c["cnt_0_2s"] for c in per_clip[s]]),
                              "A_to_B_norm_m": quant(cat["dA_B"][m2]), "A_to_U_norm_m": quant(cat["dA_U"][m2]),
                              "interp_error_m": quant(cat["dI"][m2])}
        out["pooled_0_6s"] = {"U_to_B_norm_m": quant(cat["dU_B"]),
                              "U_to_B_norm_mean_ci95_cluster": cluster_boot_mean([c["sum_0_6s"] for c in per_clip[s]],
                                                                                 [c["cnt_0_6s"] for c in per_clip[s]]),
                              "A_to_B_norm_m": quant(cat["dA_B"]), "A_to_U_norm_m": quant(cat["dA_U"]),
                              "interp_error_m": quant(cat["dI"])}
        res["splits"][s] = out
    res["per_clip_rows"] = {s: per_clip[s] for s in per_clip}
    res["wall_s"] = round(time.time() - t_start, 1)
    res["train_sample"] = {"seed": a.seed, "n": a.n_train, "rule": "numpy default_rng(seed).choice(4369, n, replace=False), sorted"}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("WROTE", a.out, f"{res['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
