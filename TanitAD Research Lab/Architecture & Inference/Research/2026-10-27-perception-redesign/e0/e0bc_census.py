"""e0bc_census.py -- E0b (range census) and E0c (per-frame egomotion pitch / roll spread), REFCV9_DESIGN.md s4.1.  CPU only; written to run on Thor (the train data live there).

E0c  per train clip: pitch / roll of the rig from the egomotion quaternion (numpy, intrinsic ZYX -> [yaw, pitch, roll]; tested against scipy), deviation = angle - clip mean (degrees);
     p50 / p95 of |deviation| pooled over all egomotion samples of all clips, plus the distribution of per-clip p95.  Decision (design): R10 pitch head IN iff pooled pitch p95 > 0.25 deg.
     NOTE: the quaternion is the rig pose in the local frame, so a slope change along a clip also moves pitch; the quantity read is exactly the one the design names.
E0b  (path) per train clip: windows t0 = every 0.4 s from the clip start while t0 + 6 s is inside the clip; the GT 6 s path = the egomotion x, y at t in (t0, t0 + 6 s] in the ego frame at t0.
     Reported: share of windows whose path END x exceeds 60 / 80 / 100 m, and whose path max |y| exceeds 16 / 24 m.
E0b  (lead)  from the v9 train label release (`--labels`, rows are per-frame ticks): share of rows with lead_valid, and of those the share whose lead_gap_m exceeds 60 / 80 / 100 m
     (also over ALL rows).  lead_gap_m is a longitudinal gap, and a lead is in-lane by construction, so the lateral |y| > 16 / 24 m clauses are read on the GT path only.

    python -X utf8 e0bc_census.py --clip-list <ids.txt> --clip-index <clip_index.parquet> --ego-dir <labels/egomotion> --labels <v9_labels_train.npz> --out E0BC_RESULT.json [--max-clips N]
Clip ids are read in process; the output carries counts and distributions only.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np

PITCH_P95_THRESH_DEG = 0.25


def euler_zyx_deg(q):
    """q [N, 4] (qx, qy, qz, qw) -> [N, 3] (yaw, pitch, roll) degrees."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return np.degrees(np.stack([yaw, pitch, roll], 1))                 # numpy only (no scipy on the Thor train venv); the test checks it against scipy


def hp_dev(a, win=300):
    """deviation of a 100 Hz angle series from its centred moving average (win samples = 3 s): removes slope AND the frame-tilt x yaw coupling that a clip-mean deviation keeps."""
    a = np.asarray(a, np.float64)
    pad = np.pad(a, (win // 2, win - win // 2), mode="edge")
    c = np.concatenate([[0.0], np.cumsum(pad)])
    return a - (c[win:win + len(a)] - c[:len(a)]) / win


def pitch_roll_dev_hp(q):
    e = euler_zyx_deg(q)
    return hp_dev(e[:, 1]), hp_dev(e[:, 2])


def pitch_roll_dev(q):
    e = euler_zyx_deg(q)
    return e[:, 1] - e[:, 1].mean(), e[:, 2] - e[:, 2].mean()


def window_paths(t_us, x, y, yaw_deg, step_s=0.4, horizon_s=6.0):
    """-> list of (x_end, max_abs_y) in the ego frame at each window start t0."""
    t = (np.asarray(t_us, np.float64) - t_us[0]) * 1e-6
    out = []
    t0 = 0.0
    while t0 + horizon_s <= t[-1] + 1e-9:
        i0 = int(np.searchsorted(t, t0))
        m = (t > t[i0]) & (t <= t[i0] + horizon_s + 1e-9)
        if m.sum() >= 2:
            th = np.deg2rad(yaw_deg[i0])
            dx, dy = x[m] - x[i0], y[m] - y[i0]
            px, py = np.cos(th) * dx + np.sin(th) * dy, -np.sin(th) * dx + np.cos(th) * dy
            out.append((float(px[-1]), float(np.abs(py).max())))
        t0 += step_s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-list", required=True)
    ap.add_argument("--clip-index", required=True)
    ap.add_argument("--ego-dir", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-clips", type=int, default=0)
    a = ap.parse_args()
    import pandas as pd
    ids = [ln.strip() for ln in open(a.clip_list) if ln.strip()]
    if a.max_clips:
        ids = list(np.random.default_rng(0).permutation(ids)[: a.max_clips])
    ci = pd.read_parquet(a.clip_index)
    chunk_of = ci["chunk"]
    zips = {}
    pdev, rdev, pdev_hp, rdev_hp, clip_p95_p, clip_p95_r, paths = [], [], [], [], [], [], []
    n_ok = n_miss = 0
    for cid in ids:
        try:
            ch = int(chunk_of.loc[cid])
            zf = zips.get(ch) or zips.setdefault(ch, zipfile.ZipFile(Path(a.ego_dir) / f"egomotion.chunk_{ch:04d}.zip"))
            df = pd.read_parquet(io.BytesIO(zf.read(f"{cid}.egomotion.parquet")))
        except Exception:
            n_miss += 1
            continue
        n_ok += 1
        q = df[["qx", "qy", "qz", "qw"]].to_numpy()
        dp, dr = pitch_roll_dev(q)
        hp_p, hp_r = pitch_roll_dev_hp(q)
        pdev_hp.append(np.abs(hp_p))
        rdev_hp.append(np.abs(hp_r))
        pdev.append(np.abs(dp))
        rdev.append(np.abs(dr))
        clip_p95_p.append(float(np.quantile(np.abs(dp), 0.95)))
        clip_p95_r.append(float(np.quantile(np.abs(dr), 0.95)))
        paths += window_paths(df["timestamp"].to_numpy(), df["x"].to_numpy(), df["y"].to_numpy(), euler_zyx_deg(q)[:, 0])
    if n_ok == 0:
        raise SystemExit("no clip could be read: refusing to write an empty census")
    P, Rr = np.concatenate(pdev), np.concatenate(rdev)
    Ph, Rh = np.concatenate(pdev_hp), np.concatenate(rdev_hp)
    pa = np.array(paths)
    out = {"tool": "e0bc_census.py", "n_clips_requested": len(ids), "n_clips_read": n_ok, "n_clips_missing": n_miss,
           "E0c": {"n_samples": int(P.size), "pitch_abs_dev_deg": {"p50": float(np.quantile(P, 0.5)), "p95": float(np.quantile(P, 0.95)), "p99": float(np.quantile(P, 0.99))},
                   "roll_abs_dev_deg": {"p50": float(np.quantile(Rr, 0.5)), "p95": float(np.quantile(Rr, 0.95)), "p99": float(np.quantile(Rr, 0.99))},
                   "per_clip_p95_pitch_deg": {str(q_): float(np.quantile(clip_p95_p, q_)) for q_ in (0.1, 0.5, 0.9)},
                   "per_clip_p95_roll_deg": {str(q_): float(np.quantile(clip_p95_r, q_)) for q_ in (0.1, 0.5, 0.9)},
                   "R10_in_pitch_p95_gt_0p25": bool(np.quantile(P, 0.95) > PITCH_P95_THRESH_DEG),
                   "dynamic_hp3s": {"pitch_abs_dev_deg": {"p50": float(np.quantile(Ph, 0.5)), "p95": float(np.quantile(Ph, 0.95)), "p99": float(np.quantile(Ph, 0.99))},
                                    "roll_abs_dev_deg": {"p50": float(np.quantile(Rh, 0.5)), "p95": float(np.quantile(Rh, 0.95)), "p99": float(np.quantile(Rh, 0.99))},
                                    "R10_in_pitch_p95_gt_0p25": bool(np.quantile(Ph, 0.95) > PITCH_P95_THRESH_DEG)}},
           "E0b_path": {"n_windows": int(len(pa)), "end_x_percentiles_m": {str(q_): float(np.quantile(pa[:, 0], q_)) for q_ in (0.5, 0.9, 0.99)},
                        "share_end_x_gt_m": {str(m): float((pa[:, 0] > m).mean()) for m in (60, 80, 100)},
                        "share_max_abs_y_gt_m": {str(m): float((pa[:, 1] > m).mean()) for m in (16, 24)}}}
    if a.labels:
        z = np.load(a.labels, allow_pickle=False)
        valid = z["row__lead_valid"].astype(bool)
        gap = z["row__lead_gap_m"].astype(np.float64)
        out["E0b_lead"] = {"n_rows": int(valid.size), "share_rows_with_lead": float(valid.mean()),
                           "share_gap_gt_m_of_valid_leads": {str(m): float((gap[valid] > m).mean()) for m in (60, 80, 100)},
                           "share_gap_gt_m_of_all_rows": {str(m): float((valid & (gap > m)).mean()) for m in (60, 80, 100)},
                           "gap_percentiles_m_valid": {str(q_): float(np.quantile(gap[valid], q_)) for q_ in (0.5, 0.9, 0.99)}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in out if k.startswith("E0")})[:1500])


if __name__ == "__main__":
    main()
