#!/usr/bin/env python3
"""END-TO-END: obstacle.offline -> the eval window grid -> LONGITUDINAL distance-keeping.

Three things happen here, in order, and each is a check on the one before:

1. REGISTRATION PROOF. For clips whose CAMERA timestamps parquet is on this disk, the episode's
   pose grid is rebuilt exactly as `physicalai.build_episode` does (no video decode — poses come
   from `signals_at`, the pixels are irrelevant to this). `lead_source.register_poses_to_time`
   then recovers the clip time of every pose index FROM THE POSES ALONE, and the result is
   compared against the true `t_query`. If that agrees, the same function run on the eval host's
   cached episodes is exact there too — which is what makes val40 scoring a one-command job.

2. THE JOIN, on the real eval window grid (`window_last_indices`, the one that reproduces the
   canonical 881).

3. THE METRIC. GT vs the hold-v0 CV floor, per speed band, paired episode-cluster bootstrap.

PARITY: read-only. Nothing here writes to _epcache, re-selects a clip, or touches
r0_selection / phase0_selection.
PRIVACY: gated corpus — every artifact carries `clip_<sha256[:8]>`, never the UUID.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from tanitad.data.physicalai import TARGET_HZ, signals_at   # noqa: E402
from taniteval.lead_metrics import (distance_keeping,        # noqa: E402
                                    distance_keeping_by_speed,
                                    paired_distance_keeping)
from taniteval.lead_source import (LEAD, NO_LABEL, NO_LEAD,  # noqa: E402
                                   RegistrationError, lead_block,
                                   register_poses_to_time, window_last_indices)
from lead_state_gate import VEHICLE_CLASSES, quaternion_yaw  # noqa: E402  (stack/scripts)

ROOT = Path(r"C:/Users/Admin/tanitad-data/physicalai")
N_STACK = 3
WP_REL_S = np.array([0.5, 1.0, 1.5, 2.0])     # rollout.WP_STEPS at 10 Hz — the sparse eval view
DENSE_REL_S = np.arange(1, 21) * 0.1          # the dense 10 Hz path


def anon(cid: str) -> str:
    return "clip_" + hashlib.sha256(str(cid).encode()).hexdigest()[:8]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
def episode_grid(ts_parquet: Path, ego: pd.DataFrame):
    """Reproduce `physicalai.build_episode`'s pose grid WITHOUT decoding the video.

    build_episode: t_query = linspace(t_frames[0], t_frames[-1], n_target); poses = poses[k:n]
    with k = n_stack - 1. Returns (t_query_of_pose_index [T], poses [T, 4]).
    """
    ts = pd.read_parquet(ts_parquet)
    tcol = next(c for c in ts.columns if "time" in c.lower())
    t_frames = ts[tcol].to_numpy(np.float64)
    span = t_frames[-1] - t_frames[0]
    unit = 1.0
    for cand in (1e9, 1e6, 1e3):
        if span / cand > 1.0:
            unit = cand
            break
    n_target = max(int(span / unit * TARGET_HZ), N_STACK + 1)
    t_query = np.linspace(t_frames[0], t_frames[-1], n_target)
    _actions, poses = signals_at(ego, t_query)
    k = N_STACK - 1
    return t_query[k:] / unit, poses[k:]          # seconds, [T,4]


def ego_series(ego: pd.DataFrame) -> dict:
    t = ego["timestamp"].to_numpy(np.float64) / 1e6
    o = np.argsort(t)
    g = lambda c: ego[c].to_numpy(np.float64)[o]   # noqa: E731
    return {"t": t[o], "x": g("x"), "y": g("y"),
            "yaw": np.unwrap(quaternion_yaw(g("qx"), g("qy"), g("qz"), g("qw"))),
            "v": np.hypot(g("vx"), g("vy"))}


def obs_dict(df: pd.DataFrame) -> dict:
    return {"t": df["timestamp_us"].to_numpy(np.float64) / 1e6,
            "track": df["track_id"].astype(str).to_numpy(object),
            "center_x": df["center_x"].to_numpy(np.float64),
            "center_y": df["center_y"].to_numpy(np.float64),
            "size_x": df["size_x"].to_numpy(np.float64),
            "is_vehicle": df["label_class"].astype(str).isin(VEHICLE_CLASSES).to_numpy()}


def gt_and_cv(poses: np.ndarray, last: np.ndarray, t_s: np.ndarray, rel_s: np.ndarray):
    """GT and hold-v0 CV paths in each window's origin frame, on the SAME grid as `rel_s`.

    GT is the ego's own future pose expressed in the window frame — the same construction
    `rollout.gt_ego_waypoints` uses, rebuilt here from the pose array directly.
    """
    steps = np.round(rel_s / 0.1).astype(int)
    x, y, yaw, v = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
    w, k = last.size, steps.size
    gt = np.full((w, k, 2), np.nan)
    cv = np.full((w, k, 2), np.nan)
    for i, l in enumerate(last):
        j = l + steps
        ok = j < poses.shape[0]
        c, s = np.cos(yaw[l]), np.sin(yaw[l])
        dx, dy = x[np.clip(j, 0, poses.shape[0] - 1)] - x[l], \
            y[np.clip(j, 0, poses.shape[0] - 1)] - y[l]
        gt[i, ok, 0] = (dx * c + dy * s)[ok]
        gt[i, ok, 1] = (-dx * s + dy * c)[ok]
        cv[i, :, 0] = v[l] * rel_s
        cv[i, :, 1] = 0.0
    return gt, cv


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-clips", type=int, default=10_000)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    sel = pd.read_parquet(ROOT / "r0" / "r0_selection.parquet")
    sel["clip_id"] = sel["clip_id"].astype(str)
    chunk_of = dict(zip(sel["clip_id"], sel["chunk"].astype(int)))
    cam = {p.name.split(".")[0]: p for p in (ROOT / "r0" / "camera_front_wide").rglob("*.mp4")}
    obs_chunks = {int(p.name.split("_")[-1].split(".")[0])
                  for p in (ROOT / "labels" / "obstacle.offline").glob("*.zip")}

    todo = sorted(c for c in cam if c in chunk_of)
    log(f"{len(todo)} R0 clips with local camera; {len(obs_chunks)} obstacle chunks on disk")

    reg_res, ab = [], []
    W_head, W_ttc, W_tg = {}, {}, {}
    rows = {"gt": [], "cv": []}
    eids, speeds, states, banded = [], [], [], []
    n_clip = n_win = 0
    per_clip = []
    by_chunk: dict[int, list[str]] = {}
    for c in todo:
        by_chunk.setdefault(chunk_of[c], []).append(c)

    for ch in sorted(by_chunk):
        ezp = ROOT / "labels" / "egomotion" / f"egomotion.chunk_{ch:04d}.zip"
        if not ezp.exists():
            continue
        ozp = ROOT / "labels" / "obstacle.offline" / f"obstacle.offline.chunk_{ch:04d}.zip"
        has_obs = ozp.exists()
        oz = zipfile.ZipFile(ozp) if has_obs else None
        with zipfile.ZipFile(ezp) as ez:
            emap = {n.split("/")[-1].split(".")[0]: n for n in ez.namelist()
                    if n.endswith(".parquet")}
            omap = ({n.split("/")[-1].split(".")[0]: n for n in oz.namelist()
                     if n.endswith(".parquet")} if has_obs else {})
            for cid in by_chunk[ch]:
                if n_clip >= a.max_clips:
                    break
                if cid not in emap:
                    continue
                tsp = cam[cid].with_name(cam[cid].name.replace(".mp4", ".timestamps.parquet"))
                if not tsp.exists():
                    continue
                ego_df = pd.read_parquet(io.BytesIO(ez.read(emap[cid])))
                try:
                    t_true, poses = episode_grid(tsp, ego_df)
                except Exception as e:                        # noqa: BLE001
                    log(f"  {anon(cid)}: episode grid failed {e!r}")
                    continue
                p = ego_series(ego_df)
                try:
                    reg = register_poses_to_time(poses[:, :2], p["t"], p["x"], p["y"])
                except RegistrationError as e:
                    log(f"  {anon(cid)}: REGISTRATION REFUSED {e}")
                    continue
                err = np.abs(reg["t_s"] - t_true)
                reg_res.append({"clip": anon(cid), "T": int(poses.shape[0]),
                                "median_residual_m": reg["residual_m"]["median"],
                                "max_residual_m": reg["residual_m"]["max"],
                                "t_err_median_s": float(np.median(err)),
                                "t_err_max_s": float(err.max()),
                                "b_fit_s": reg["b"], "a_fit_s": reg["a"],
                                "a_true_s": float(t_true[0]),
                                "ego_t0_s": float(p["t"][0])})
                ab.append((reg["a"], reg["b"], float(t_true[0]), float(p["t"][0])))
                n_clip += 1

                last = window_last_indices(int(poses.shape[0]))
                if last.size == 0:
                    continue
                t0s = t_true[last]                    # TRUE grid, not the fit — this is the proof
                od = obs_dict(pd.read_parquet(io.BytesIO(oz.read(omap[cid])))) \
                    if (has_obs and cid in omap) else None
                blk = lead_block(t0s, WP_REL_S, od, p)
                gt, cv = gt_and_cv(poses, last, t_true, WP_REL_S)
                keep = np.isfinite(gt).all(axis=(1, 2))
                if not keep.any():
                    continue
                rows["gt"].append(gt[keep])
                rows["cv"].append(cv[keep])
                for key, arr in (("leads", blk["leads"]), ):
                    W_head.setdefault(key, []).append(arr[keep])
                W_ttc.setdefault("lead_lens", []).append(blk["lead_lens"][keep])
                W_tg.setdefault("speeds", []).append(blk["speeds"][keep])
                states.append(blk["state"][keep])
                eids.extend([anon(cid)] * int(keep.sum()))
                n_win += int(keep.sum())
                per_clip.append({"clip": anon(cid), "chunk": ch, "n_windows": int(keep.sum()),
                                 "has_obstacle_feature": bool(od is not None),
                                 **{k: int(v) for k, v in blk["counts"].items()}})
        if oz is not None:
            oz.close()
        log(f"chunk {ch:04d}: {n_clip} clips / {n_win} windows cumulative")

    if not rows["gt"]:
        raise SystemExit("no windows built")

    GT = np.concatenate(rows["gt"])
    CV = np.concatenate(rows["cv"])
    LEADS = np.concatenate(W_head["leads"])
    LENS = np.concatenate(W_ttc["lead_lens"])
    SPD = np.concatenate(W_tg["speeds"])
    ST = np.concatenate(states)
    EID = np.array(eids, dtype=object)
    dt = float(WP_REL_S[1] - WP_REL_S[0])

    dk_gt = distance_keeping(GT, LEADS, LENS, SPD, dt)
    dk_cv = distance_keeping(CV, LEADS, LENS, SPD, dt)
    log(f"GT  n={dk_gt['n']}/{dk_gt['n_windows']}  headway {dk_gt.get('mean_headway_min_m')} m")
    log(f"CV  n={dk_cv['n']}/{dk_cv['n_windows']}  headway {dk_cv.get('mean_headway_min_m')} m")

    out = {
        "_what": ("LONGITUDINAL distance-keeping on the REAL eval window grid, from "
                  "obstacle.offline. GT vs the hold-v0 CV floor."),
        "_evidence_class": "MEASURED (ours; this run)",
        "_parity": ("read-only over labels/*.zip + r0/r0_selection.parquet + the camera "
                    "timestamps parquets. No episode re-selected, no cache written."),
        "corpus": {
            "n_clips_registered": n_clip,
            "n_windows": int(GT.shape[0]),
            "n_clips_with_obstacle_feature": int(sum(
                1 for r in per_clip if r["has_obstacle_feature"])),
            "window_states": {k: int((ST == k).sum()) for k in (LEAD, NO_LEAD, NO_LABEL)},
            "lead_rate_over_labelled_windows": round(
                float((ST == LEAD).sum()) / max(int((ST != NO_LABEL).sum()), 1), 4),
            "waypoint_grid_s": WP_REL_S.tolist(), "dt_s": dt,
        },
        "registration_proof": {
            "_what": ("register_poses_to_time recovers the clip time of every pose index FROM "
                      "THE POSES ALONE. Compared here against the true t_query the episode was "
                      "built on. This is what makes the same call exact on the eval host, where "
                      "the camera timestamps parquet does not exist."),
            "n_clips": len(reg_res),
            "pose_match_residual_m": {
                "median_of_medians": round(float(np.median(
                    [r["median_residual_m"] for r in reg_res])), 6),
                "worst_max": round(float(np.max([r["max_residual_m"] for r in reg_res])), 6)},
            "recovered_time_error_s": {
                "median_of_medians": round(float(np.median(
                    [r["t_err_median_s"] for r in reg_res])), 6),
                "p95_of_max": round(float(np.percentile(
                    [r["t_err_max_s"] for r in reg_res], 95)), 6),
                "worst_max": round(float(np.max([r["t_err_max_s"] for r in reg_res])), 6)},
            "grid_dt_s_fit": {"median": round(float(np.median([r["b_fit_s"] for r in reg_res])), 6),
                              "min": round(float(np.min([r["b_fit_s"] for r in reg_res])), 6),
                              "max": round(float(np.max([r["b_fit_s"] for r in reg_res])), 6)},
        },
        "GT": {k: v for k, v in dk_gt.items() if not isinstance(v, np.ndarray)},
        "CV": {k: v for k, v in dk_cv.items() if not isinstance(v, np.ndarray)},
        "GT_by_speed": distance_keeping_by_speed(dk_gt, SPD, EID, states=ST, n_boot=a.n_boot),
        "CV_by_speed": distance_keeping_by_speed(dk_cv, SPD, EID, states=ST, n_boot=a.n_boot),
        "paired_GT_minus_CV": paired_distance_keeping(dk_gt, dk_cv, EID, names=("GT", "CV"),
                                                      n_boot=a.n_boot),
    }
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    Path(a.out).with_name("per_clip.json").write_text(json.dumps(per_clip, indent=0))
    Path(a.out).with_name("registration_per_clip.json").write_text(
        json.dumps(reg_res, indent=0))
    log(f"wrote {a.out}")
    print(json.dumps({k: out[k] for k in ("corpus", "registration_proof")}, indent=1))


if __name__ == "__main__":
    main()
