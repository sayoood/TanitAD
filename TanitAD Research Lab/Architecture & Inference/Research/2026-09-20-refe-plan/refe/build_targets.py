"""Stage 2 -- turn teacher rollouts into REFe training tuples.

One tuple per simulation step:
    front-camera JPEG path
    ego kinematics at t
    goal points at t                      <- the augmentation input the teacher was conditioned on
    teacher trajectory t+1..t+20 in the ego frame at t   <- the WTA L1 target
    route rank                            <- which augmented route produced this rollout

WHAT THE TARGET IS, and why. Their loss is "winner-takes-all L1 to the teacher rollout", so the
target is the trajectory the teacher ACTUALLY DROVE closed-loop, not its per-step plan. Their head
is 20 steps @ 5 Hz = 4.0 s, so the target is 20 SAMPLES AT STRIDE 2 over the 10 Hz history.

RATES, because getting these wrong halves the horizon silently (see STRIDE below):
  simulation history  10 Hz  (MEASURED: dt 0.1000 s, 149 steps over 14.80 s)
  policy queried       5 Hz  (preflight interval_s=0.2; config target_sample_rate_hz 5.0)
  CAM_F0              10 Hz  (MEASURED: median 100 ms)

CAMERA PAIRING. The DBs carry image.filename_jpg + timestamp; the JPEGs come from the sensor blobs.
Camera and history are both 10 Hz, so the nearest-in-time frame is taken and the residual is small
(MEASURED mean 24 ms, max 38 ms). A step whose nearest frame is further than --max-dt-ms is DROPPED
rather than silently mispaired.

SCOPE. The scorer's six-PDM-component target is NOT produced here. It needs THEIR six released
reward calculators fed a `ScenarioData` with the candidate appended to the ego track -- see
`score_proposals.py`, whose self-test is green. No rollout per proposal is required (an earlier note
here said otherwise and was wrong). The trajectory target is the main supervision and comes first.

Usage:
  python build_targets.py --run C:/dzo/m-nr-n --rank 0 --out D:/.../refe_targets [--images <dir>]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

HORIZON = 20          # 20 SAMPLES, their output head
# MEASURED 2026-09-20, and this bit me: the simulation HISTORY is logged at 10 Hz (dt = 0.1000 s,
# 149 steps over 14.80 s), while their head is 20 steps @ 5 Hz = 4.0 s. Taking 20 consecutive steps
# therefore spans 2.0 s, not 4.0 s -- a silent HALVING of the horizon that still produces a
# plausible-looking target bank. The policy itself is queried every 0.2 s (preflight interval_s=0.2,
# config target_sample_rate_hz 5.0), so STRIDE 2 is both the fix and the policy's own cadence.
# Caught by the bank's own check that displacement/horizon must equal the measured mean speed: it
# read 3.30 m/s against a measured 6.34 m/s, i.e. exactly 2x off.
STRIDE = 2            # 20 samples x stride 2 x 0.1 s = 4.0 s
SPAN = HORIZON * STRIDE
DEFAULT_MAX_DT_MS = 120.0    # camera is 10 Hz => a correct pairing is <=50 ms; 120 is generous


def ego_frame(px, py, pyaw, xs, ys):
    """World points -> the ego frame at (px, py, pyaw)."""
    c, s = math.cos(-pyaw), math.sin(-pyaw)
    dx, dy = xs - px, ys - py
    return np.stack([dx * c - dy * s, dx * s + dy * c], axis=-1)


# ⭐ PI DECISION 2026-09-20: FOUR CAMERAS, as the paper. The model was moved first and this
# builder was still single-camera, which would have trained a 4-camera model on 25 % of its input.
CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")


def camera_index(db_path: str, channels=CAMERAS) -> dict[str, list[tuple[int, str]]]:
    """{channel: [(timestamp_us, filename_jpg)]} ascending, for each requested camera.

    ⚠️ The cameras are NOT synchronised to a common clock. Each is indexed on its own timestamps
    and paired to the ego step independently, so a per-camera residual is reported rather than one
    camera's timing being assumed for the rig.
    """
    c = sqlite3.connect(db_path)
    out: dict[str, list[tuple[int, str]]] = {}
    for ch in channels:
        row = c.execute("SELECT token FROM camera WHERE channel=?", (ch,)).fetchone()
        if row is None:
            out[ch] = []
            continue
        out[ch] = c.execute(
            "SELECT timestamp, filename_jpg FROM image WHERE camera_token=? ORDER BY timestamp",
            (row[0],)).fetchall()
    c.close()
    return out


_IMAGE_INDEX: dict[str, str] = {}
_INDEX_BY_CAMERA: dict[str, int] = {}


def index_images(images_root: str | None) -> int:
    """basename -> full path, built once. Also reads container zips so both layouts work.

    ⛔ TWO FOUR-CAMERA DEFECTS FIXED HERE 2026-09-21.

    1. The container branch matched `_CAM_F0.zip` ONLY, so the L0/R0/B0 containers this same
       package fetches were invisible. Every 4-camera tuple would have failed its image lookup and
       been counted as `missing_image_file` -- a builder that drops 100 % of its rows and reports
       it as a data problem rather than a code one.
    2. The index is keyed by BASENAME, and that is only safe because nuPlan names each frame by a
       per-image hash (`c4d1ab5e4d935dba.jpg`). ⚠️ It is now VERIFIED rather than assumed: a key
       that resolves to two different files RAISES. Silently serving CAM_F0's pixels for a CAM_B0
       slot is precisely the failure the camera extension is most exposed to, and it would leave
       no trace in any loss curve.
    """
    _IMAGE_INDEX.clear()
    _INDEX_BY_CAMERA.clear()
    if not images_root or not os.path.isdir(images_root):
        return 0
    collisions: list[tuple[str, str, str]] = []

    def put(key: str, val: str, cam: str) -> None:
        old = _IMAGE_INDEX.get(key)
        if old is not None and old != val:
            collisions.append((key, old, val))
        _IMAGE_INDEX[key] = val
        _INDEX_BY_CAMERA[cam] = _INDEX_BY_CAMERA.get(cam, 0) + 1

    def cam_of(path: str) -> str:
        for c in CAMERAS:
            if c in path:
                return c
        return "?"

    for r, _, fs in os.walk(images_root):
        for f in fs:
            if f.endswith(".jpg"):
                p = os.path.join(r, f)
                put(f, p, cam_of(p))
            elif f.endswith(".zip") and "_CAM_" in f:
                import zipfile
                zp = os.path.join(r, f)
                cam = cam_of(f)
                try:
                    for n in zipfile.ZipFile(zp).namelist():
                        put(os.path.basename(n), f"{zp}::{n}", cam)
                except Exception:
                    pass
    if collisions:
        k, a, b = collisions[0]
        raise RuntimeError(
            f"image basename collision ({len(collisions)} of them): {k!r} resolves to BOTH {a!r} "
            f"and {b!r}. The index cannot tell the cameras apart and would feed the wrong view.")
    return len(_IMAGE_INDEX)


def build_one(log_path: str, db_dir: str, images_root: str | None, rank: int,
              max_dt_ms: float) -> tuple[list[dict], dict]:
    from nuplan.planning.simulation.simulation_log import SimulationLog
    log = SimulationLog.load_data(file_path=Path(log_path))
    sc = log.scenario
    smps = log.simulation_history.data
    n = len(smps)

    P = np.array([[s.ego_state.rear_axle.x, s.ego_state.rear_axle.y,
                   s.ego_state.rear_axle.heading] for s in smps])
    ts = np.array([int(s.ego_state.time_point.time_us) for s in smps])

    cams = camera_index(os.path.join(db_dir, f"{sc.log_name}.db"))
    cam_ts = {ch: (np.array([t for t, _ in v], dtype=np.int64) if v else np.zeros(0, np.int64))
              for ch, v in cams.items()}
    missing = [ch for ch in CAMERAS if cam_ts[ch].size == 0]
    if missing:
        # ⛔ a rig with a missing camera cannot produce a 4-camera tuple; say which, do not pad
        return [], {"steps": n, "kept": 0, "dropped_no_camera": n, "dropped_short_horizon": 0,
                    "missing_image_file": 0, "camera_rows": 0, "missing_channels": missing}

    rows, dropped_nocam, dropped_short, missing_file = [], 0, 0, 0
    for i in range(n):
        if i + SPAN >= n:
            dropped_short += 1
            continue
        # ⛔ THIS USED TO READ `if cam_ts.size == 0`, A LEFTOVER FROM THE SINGLE-CAMERA VERSION
        # WHERE `cam_ts` WAS AN ndarray. Under the 4-camera extension it is a {channel: ndarray}
        # dict, so the line raised `AttributeError: 'dict' object has no attribute 'size'` on the
        # FIRST step of the FIRST log -- i.e. the builder that feeds training could not emit a
        # single tuple, and nothing noticed because no instrument ran it after the extension.
        # The emptiness case it was reaching for is already handled by the `missing` check above,
        # which returns before this loop and NAMES the absent channels.
        # pair EVERY camera independently and keep the WORST residual as the tuple's dt
        rels, dts, bad = [], [], False
        for ch in CAMERAS:
            jj = int(np.argmin(np.abs(cam_ts[ch] - ts[i])))
            d = abs(int(cam_ts[ch][jj]) - int(ts[i])) / 1000.0
            if d > max_dt_ms:
                bad = True
                break
            rels.append(cams[ch][jj][1]); dts.append(d)
        if bad:
            dropped_nocam += 1
            continue
        dt_ms = max(dts)
        if images_root is not None:
            # Index ONCE per call, not once per tuple. The original globbed the whole tree for every
            # step -- O(steps x tree) over 26,060 files, which turned a seconds-long build into a
            # many-minute one. Same family as any per-item filesystem walk: the cost is invisible at
            # 8 tuples and dominates at a thousand.
            imgs = [_IMAGE_INDEX.get(os.path.basename(r)) for r in rels]
            if any(x is None for x in imgs):
                missing_file += 1
                continue
            img = imgs
        else:
            img = rels

        sel = slice(i + STRIDE, i + SPAN + 1, STRIDE)          # 20 samples, 4.0 s
        fut = ego_frame(P[i, 0], P[i, 1], P[i, 2], P[sel, 0], P[sel, 1])
        yaw = ((P[sel, 2] - P[i, 2] + math.pi) % (2 * math.pi)) - math.pi
        assert fut.shape[0] == HORIZON, f"horizon {fut.shape[0]} != {HORIZON}"
        traj = np.concatenate([fut, yaw[:, None]], axis=-1)

        st = smps[i].ego_state
        v = st.dynamic_car_state.rear_axle_velocity_2d
        a = st.dynamic_car_state.rear_axle_acceleration_2d
        # ⛔ SEVEN REAL KINEMATIC SCALARS -- no pad, no constants. This vector was briefly 8-D with a
        # hardcoded trailing 0.0, then 9-D with the ego's length and width appended "per Table A2".
        # Table A2 is the **teacher's** input schema, so that was a scope error; and on nuPlan the
        # footprint is a constant Pacifica, so those two slots carried no information while
        # invalidating every banked row. See the retraction in `model.py`'s `ego_dim`.
        # ⚠️ The width is asserted against the model config by `diag_consumer_conformance.py`, so a
        # future change here cannot silently diverge from `REFeConfig.ego_dim` again.
        ego = [float(v.x), float(v.y), float(a.x), float(a.y),
               float(st.dynamic_car_state.angular_velocity),
               float(st.tire_steering_angle), float(math.hypot(v.x, v.y))]

        g = getattr(smps[i].trajectory, "goal_points", None)
        goal = np.asarray(g, dtype=float).reshape(-1).tolist() if g is not None else [0.0] * 4

        rows.append({"image": img, "cameras": list(CAMERAS), "ego": ego, "goal": goal, "traj": traj.tolist(),
                     "rank": rank, "step": i, "dt_ms": dt_ms,
                     "scenario_type": sc.scenario_type, "token": sc.scenario_name,
                     "log_name": sc.log_name})

    # `camera_rows` was `int(cam_ts.size)` -- the same dict-vs-ndarray leftover as above. Report it
    # PER CHANNEL as well as in total, because "the rig has 16,080 rows" hides a camera that is
    # short by a thousand frames and would silently drop tuples through the dt filter.
    stats = {"steps": n, "kept": len(rows), "dropped_no_camera": dropped_nocam,
             "dropped_short_horizon": dropped_short, "missing_image_file": missing_file,
             "camera_rows": int(sum(v.size for v in cam_ts.values())),
             "camera_rows_by_channel": {ch: int(cam_ts[ch].size) for ch in CAMERAS}}
    return rows, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. C:/dzo/m-nr-n")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--db-dir", default="D:/Projects/TanitAD/data/nuplan/dblinks/driverl_val14")
    ap.add_argument("--images", default=None, help="root of fetched CAM_F0 jpgs; omit to record relative paths")
    ap.add_argument("--max-dt-ms", type=float, default=DEFAULT_MAX_DT_MS)
    a = ap.parse_args(argv)

    logs = sorted(glob.glob(f"{a.run}/**/*.msgpack.xz", recursive=True))
    print(f"  {len(logs)} simulation logs under {a.run}")
    n_idx = index_images(a.images)
    if a.images:
        print(f"  image index: {n_idx:,} frames under {a.images}")
    os.makedirs(a.out, exist_ok=True)
    allrows, allstats = [], []
    for lp in logs:
        rows, st = build_one(lp, a.db_dir, a.images, a.rank, a.max_dt_ms)
        allrows += rows
        allstats.append(st)
        print(f"    {st['scenario_type'] if 'scenario_type' in st else Path(lp).stem[:16]:44s}"
              if False else
              f"    {rows[0]['scenario_type'][:42] if rows else Path(lp).parts[-4][:42]:42s} "
              f"steps {st['steps']:3d}  kept {st['kept']:3d}  "
              f"no-cam {st['dropped_no_camera']:3d}  short {st['dropped_short_horizon']:3d}  "
              f"missing-file {st['missing_image_file']:3d}")
    out = os.path.join(a.out, f"targets_rank{a.rank}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in allrows:
            f.write(json.dumps(r) + "\n")
    tot = sum(s["steps"] for s in allstats)
    kept = sum(s["kept"] for s in allstats)
    print(f"\n  wrote {kept:,} tuples to {out}")
    print(f"  coverage: {kept:,} / {tot:,} steps ({100*kept/max(tot,1):.1f} %)")
    print(f"  horizon {HORIZON} samples x stride {STRIDE} x 0.1 s = {SPAN*0.1:.1f} s; the last "
          f"{SPAN} raw steps of each scenario cannot form a target and are dropped by design")
    json.dump({"run": a.run, "rank": a.rank, "per_log": allstats, "kept": kept, "steps": tot},
              open(os.path.join(a.out, f"targets_rank{a.rank}_stats.json"), "w"), indent=1)
    print("BUILD_TARGETS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
