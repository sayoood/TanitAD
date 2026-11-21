#!/usr/bin/env python3
"""P2 - build the LiDAR BEV ground truth for one clip, and PROVE it is registered.

⛔ A POSITIVE ASSERTION ALONE IS NOT A CHECK. "The boxes overlap the returns" passes
happily on a mirrored or shifted geometry if the scene is dense enough. This script
therefore scores the real boxes AND three deliberately wrong box sets that MUST collapse:

  * `mirror_y`  -- y -> -y. This is exactly `R-2026-09-08-wpa-mirror`, the defect that
                   inverted the programme's reading of whether the trunk localises
                   agents. If it does not collapse, our azimuth sense is unfalsified.
  * `shift_y5`  -- boxes moved 5 m to the LEFT (10 cells).
  * `shift_x10` -- boxes moved 10 m forward (20 cells).

⭐ And a NO-INFORMATION control: `marginal`, the corpus-frequency of an occupied cell.
A registration score that does not beat its own base rate has measured nothing.

Usage:
  python p2_build_bev_gt.py --clip <uuid> --frames 60,120,180 --out-dir raw/
"""
from __future__ import annotations

import argparse
import glob
import json
import lzma
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lidar_bev import (  # noqa: E402
    CartesianBEVSpec, PolarBEVSpec, ZBand, DRACO_ATTR_INTENSITY,
    DRACO_ATTR_POINT_TS_US, artifact_meta, deskew_rigid, lidar_to_rig,
    rasterise_cartesian, rasterise_polar,
)

CAM_DIR = r"C:\Users\Admin\tanitad-data\physicalai-b1\r0\camera_front_wide_120fov"
EXT_GLOB = r"C:\Users\Admin\tanitad-data\physicalai-b1\calibration\sensor_extrinsics\*.parquet"
EGO_DIR = r"C:\Users\Admin\tanitad-data\physicalai\labels\egomotion_alpamayo"
LIDAR_CACHE = r"C:\Users\Admin\tanitad-caches\lidar-bev-20260911"
JOIN = (Path(__file__).resolve().parents[4] / "Benchmarks & Evals" / "Research"
        / "2026-09-06-b1-agent-join" / "raw" / "b1eval_agents.jsonl.xz")


def load_extrinsics(clip_id: str, sensor: str) -> dict:
    import pyarrow.parquet as pq
    for fp in sorted(glob.glob(EXT_GLOB)):
        t = pq.read_table(fp).to_pydict()
        for i, (cid, sn) in enumerate(zip(t["clip_id"], t["sensor_name"])):
            if cid == clip_id and sn == sensor:
                return {k: float(t[k][i]) for k in ("x", "y", "z", "qx", "qy", "qz", "qw")}
    raise SystemExit(f"no extrinsics for {sensor} (searched {EXT_GLOB})")


def load_ego(clip_id: str):
    import pyarrow.parquet as pq
    p = Path(EGO_DIR) / f"{clip_id}.parquet"
    if not p.exists():
        return None
    return pq.read_table(p).to_pydict()


def ego_state_at(ego: dict, t_us: int) -> tuple[float, float]:
    """(speed m/s, yaw-rate rad/s) at t_us. Returns (0,0) when egomotion is absent."""
    if ego is None:
        return 0.0, 0.0
    ts = np.asarray(ego["timestamp"], dtype=np.float64)
    # egomotion timestamps are µs on the same clock
    i = int(np.clip(np.searchsorted(ts, float(t_us)), 1, len(ts) - 1))
    vx = np.asarray(ego.get("vx", np.zeros_like(ts)), dtype=np.float64)
    vy = np.asarray(ego.get("vy", np.zeros_like(ts)), dtype=np.float64)
    v = float(math.hypot(vx[i], vy[i]))
    # yaw rate from the quaternion sequence
    try:
        qz = np.asarray(ego["qz"], dtype=np.float64)
        qw = np.asarray(ego["qw"], dtype=np.float64)
        yaw = 2.0 * np.arctan2(qz, qw)
        dt = (ts[i] - ts[i - 1]) * 1e-6
        dyaw = math.atan2(math.sin(yaw[i] - yaw[i - 1]), math.cos(yaw[i] - yaw[i - 1]))
        r = dyaw / dt if dt > 0 else 0.0
    except Exception:
        r = 0.0
    return v, r


def box_cells(agents, spec: CartesianBEVSpec, dy: float = 0.0, dx: float = 0.0,
              mirror: bool = False) -> np.ndarray:
    """Boolean [n_x, n_y] of cells whose CENTRE lies inside an oriented footprint.

    Exactly `bev_raster`'s predicate: closed inequality on the rotated half-extents.
    """
    nx, ny = spec.n_x, spec.n_y
    gx = (np.arange(nx) + 0.5) * spec.cell_m
    gy = (np.arange(ny) + 0.5) * spec.cell_m - spec.y_half_m
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    out = np.zeros((nx, ny), dtype=bool)
    for a in agents:
        cx, cy = float(a["cx"]) + dx, float(a["cy"])
        yaw = float(a["yaw"])
        if mirror:
            cy, yaw = -cy, -yaw
        cy = cy + dy
        l, w = float(a["l"]), float(a["w"])
        c, s = math.cos(-yaw), math.sin(-yaw)
        ex, ey = GX - cx, GY - cy
        u = ex * c - ey * s
        v = ex * s + ey * c
        out |= (np.abs(u) <= l / 2.0) & (np.abs(v) <= w / 2.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", default="60,180,300,420",
                    help="camera frame indices (30 Hz) to build GT for")
    ap.add_argument("--out-dir", default="raw")
    ap.add_argument("--no-deskew", action="store_true")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import DracoPy

    clip = args.clip
    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    cart, polar, zb = CartesianBEVSpec(), PolarBEVSpec(), ZBand()

    lidar_pq = Path(LIDAR_CACHE) / f"{clip}.lidar_top_360fov.parquet"
    if not lidar_pq.exists():
        raise SystemExit(f"missing {lidar_pq} - run p1_lidar_probe.py --save-parquet-dir first")
    pf = pq.ParquetFile(lidar_pq)
    spins = pq.read_table(lidar_pq, columns=["spin_index", "spin_start_timestamp",
                                             "spin_end_timestamp"]).to_pydict()
    s_start = np.asarray(spins["spin_start_timestamp"], dtype=np.int64)
    s_end = np.asarray(spins["spin_end_timestamp"], dtype=np.int64)

    cam_ts = pq.read_table(Path(CAM_DIR) / f"{clip}.timestamps.parquet").to_pydict()
    cam_t = np.asarray(cam_ts["timestamp"], dtype=np.int64)

    ext = load_extrinsics(clip, "lidar_top_360fov")
    cam_ext = load_extrinsics(clip, "camera_front_wide_120fov")
    ego = load_ego(clip)

    # ---- agent boxes, keyed on TIME, never on a frame index -------------------
    # ⛔ THE INDEX-SPACE TRAP, and it bit this script on its first run. The join's
    # `frame` is the RAW v2ep index on the ~10 Hz EPISODE grid; the camera runs at
    # 30 Hz. Looking boxes up by camera-frame index silently returned agents ~3x too
    # far back in time and every registration arm read `n_box_cells_observed = 0`.
    # `t_s` is unambiguous and both sides carry it, so the lookup is by TIMESTAMP.
    join_t: list[float] = []
    join_agents: list[list] = []
    with lzma.open(JOIN, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("clip_id") == clip:
                join_t.append(float(r["t_s"]))
                join_agents.append(r.get("agents", []))
    join_t_arr = np.asarray(join_t, dtype=np.float64)

    report: dict = {
        "schema": "tanitad.lidar_bev_gt_report/1",
        "clip_sha12": __import__("hashlib").sha256(clip.encode()).hexdigest()[:12],
        "lidar_parquet_bytes": lidar_pq.stat().st_size,
        "n_spins": int(pf.metadata.num_rows),
        "n_camera_frames": int(len(cam_t)),
        "camera_rate_hz": round(1e6 / float(np.median(np.diff(cam_t))), 4),
        "lidar_rate_hz": round(1e6 / float(np.median(np.diff(s_start))), 4),
        "camera_t_span_s": [float(cam_t[0] / 1e6), float(cam_t[-1] / 1e6)],
        "lidar_t_span_s": [float(s_start[0] / 1e6), float(s_end[-1] / 1e6)],
        "lidar_extrinsics_rig": ext,
        "camera_front_wide_extrinsics_rig": cam_ext,
        "deskew": not args.no_deskew,
        "frames": [],
    }

    # ---- alignment, over EVERY camera frame, not just the sampled ones -------
    idx = np.searchsorted(s_start, cam_t) - 1
    idx = np.clip(idx, 0, len(s_start) - 1)
    # nearest spin CENTRE, which is the honest reference instant for a rolling scan
    s_mid = (s_start + s_end) // 2
    near = np.abs(s_mid[:, None] - cam_t[None, :]).argmin(axis=0)
    dt_us = np.abs(s_mid[near] - cam_t)
    report["align_dt_ms"] = {
        "median": float(np.median(dt_us) / 1e3),
        "p95": float(np.percentile(dt_us, 95) / 1e3),
        "max": float(dt_us.max() / 1e3),
        "note": "|camera frame timestamp - nearest LiDAR spin MIDPOINT|, all camera frames. "
                "A 10 Hz scan against 30 Hz video bounds this at half a spin = 50 ms.",
    }

    channels: dict[str, list] = {}
    for fi in [int(x) for x in args.frames.split(",") if x.strip()]:
        if fi >= len(cam_t):
            continue
        t_ref = int(cam_t[fi])
        si = int(np.abs(s_mid - t_ref).argmin())
        blob = pf.read_row_group(si, columns=["draco_encoded_pointcloud"]).column(0)[0].as_py()
        t0 = time.time()
        pc = DracoPy.decode(blob)
        dec_s = time.time() - t0
        pts = np.asarray(pc.points)
        # ⛔ CONTENT assertion: a decode into a pre-allocated array can leave zeros
        if pts.shape[0] == 0 or float(np.abs(pts).max()) == 0.0:
            raise SystemExit(f"frame {fi}: decoded cloud is EMPTY/ALL-ZERO - refusing to write")
        attrs = {a["unique_id"]: a["data"] for a in pc.attributes}
        inten = attrs[DRACO_ATTR_INTENSITY][:, 0].astype(np.uint8)
        pt_ts = attrs[DRACO_ATTR_POINT_TS_US][:, 0].astype(np.int64)

        rig = lidar_to_rig(pts, ext)
        v_ms, yaw_rate = ego_state_at(ego, t_ref)
        skew_m = v_ms * float(pt_ts.max() - pt_ts.min()) * 1e-6
        if not args.no_deskew:
            rig = deskew_rigid(rig, pt_ts, t_ref, v_ms, yaw_rate)

        C = rasterise_cartesian(rig, inten, cart, zb)
        P = rasterise_polar(rig, polar, zb)

        # ---- ground-plane re-derivation, as an INDEPENDENT control -----------
        r2 = np.hypot(rig[:, 0], rig[:, 1])
        nearm = (r2 > 3) & (r2 < 20)
        h, e = np.histogram(rig[nearm, 2], bins=np.arange(-2, 2.001, 0.05))
        ground_z = float(e[int(np.argmax(h))] + 0.025)

        rec = {
            "camera_frame": fi,
            "camera_t_us": t_ref,
            "spin_index": si,
            "align_dt_ms": float(abs(s_mid[si] - t_ref) / 1e3),
            "n_points": int(pts.shape[0]),
            "draco_decode_s": round(dec_s, 3),
            "ego_speed_ms": round(v_ms, 3),
            "ego_yaw_rate_rps": round(yaw_rate, 4),
            "intra_spin_skew_m": round(skew_m, 3),
            "measured_ground_peak_z_m": round(ground_z, 3),
            "cart_occ_mean": float(C["occ"].mean()),
            "cart_occ_cells": int(C["occ"].sum()),
            "cart_observed_frac": float(C["observed"].mean()),
            "cart_zmax_max_m": float(C["z_max"].max()),
            "cart_pts_in_grid": int(C["n_pts"].sum()),
            "polar_occ_mean": float(P["occ"].mean()),
            "polar_observed_frac": float(P["observed"].mean()),
        }

        # ---- REGISTRATION: do our own 3D boxes land on the returns? ----------
        ag = None
        if join_t_arr.size:
            ji = int(np.abs(join_t_arr - t_ref * 1e-6).argmin())
            rec["join_row"] = ji
            rec["join_dt_ms"] = round(abs(join_t_arr[ji] - t_ref * 1e-6) * 1e3, 2)
            ag = join_agents[ji]
        if ag is not None:
            occ = C["occ"] > 0
            occz = C["occ_zband_only"] > 0
            obs = C["observed"] > 0
            base = float(occ[obs].mean())         # no-information control
            base_z = float(occz[obs].mean())
            in_grid = [a for a in ag
                       if 0.0 <= float(a["cx"]) < cart.x_max_m
                       and abs(float(a["cy"])) < cart.y_half_m]
            arms = {}
            for name, kw in (("real", {}), ("mirror_y", {"mirror": True}),
                             ("shift_y5", {"dy": 5.0}), ("shift_x10", {"dx": 10.0})):
                bc = box_cells(ag, cart, **kw) & obs
                n = int(bc.sum())
                arms[name] = {
                    "n_box_cells_observed": n,
                    "hit_rate": float(occ[bc].mean()) if n else None,
                    "lift_over_marginal": (float(occ[bc].mean()) / base) if (n and base > 0) else None,
                    "hit_rate_zband_rule": float(occz[bc].mean()) if n else None,
                    "lift_zband_rule": (float(occz[bc].mean()) / base_z) if (n and base_z > 0) else None,
                }
            rec["registration"] = {
                "n_agents": len(ag),
                "n_agents_in_grid": len(in_grid),
                "marginal_occ_rate_observed": base,
                "marginal_occ_rate_observed_zband_rule": base_z,
                "arms": arms,
            }
        report["frames"].append(rec)
        print(json.dumps(rec, indent=1), flush=True)

        for k, v in C.items():
            channels.setdefault(f"cart_{k}", []).append(v)
        for k, v in P.items():
            channels.setdefault(f"polar_{k}", []).append(v)

    meta = artifact_meta(cart, polar, zb,
                         clip_sha12=report["clip_sha12"],
                         camera_frames=[r["camera_frame"] for r in report["frames"]],
                         camera_t_us=[r["camera_t_us"] for r in report["frames"]],
                         dt_s="camera frames are 30 Hz; LiDAR spins are 10 Hz; each GT "
                              "frame is the spin whose MIDPOINT is nearest that camera "
                              "frame's timestamp",
                         deskew_applied=not args.no_deskew)
    npz = outd / f"bev_gt_{report['clip_sha12']}.npz"
    np.savez_compressed(npz, meta_json=np.array(json.dumps(meta)),
                        **{k: np.stack(v) for k, v in channels.items()})
    report["artifact"] = str(npz)
    report["artifact_bytes"] = npz.stat().st_size
    report["meta"] = meta
    (outd / f"bev_gt_report_{report['clip_sha12']}.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    print(f"[p2] wrote {npz} ({npz.stat().st_size} B) and the report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
