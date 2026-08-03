#!/usr/bin/env python3
"""Decide — by measurement, not by name — how the NuRec `dynamic_rigids` actors are posed.

The layer stores 115,824 gaussians, 35 tracks, plus
`tracks_calib.tracks_delta_{q,t}` [1833,4]/[1833,3] and
`time_embed.timestamps_us_ranges` [35,2]. The word "delta" *suggests* a calibration
refinement on top of the dataset's cuboid tracks, but a name is not evidence. Three
things are measured here and each has a falsifiable answer:

  1. **dtype/units of the ranges** — read as int64 and check they land inside the
     scene's own [t_lo, t_hi]. (Read as float64 they were 1e-313 garbage.)
  2. **Are the 1833 rows the tracks' FULL poses?** If per-track pose counts derived
     from the time ranges at some sampling dt sum to exactly 1833, the array is a
     concatenated per-track pose table and no external mapping is needed.
  3. **Are the gaussian positions track-LOCAL or world?** Per-cuboid centroid + extent
     settles it: a local cloud is centred near the origin with vehicle-scale extent; a
     world cloud sits tens of metres out.

Run: python probe_actors.py --scene-dir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--layer", default="dynamic_rigids")
    args = ap.parse_args()
    from nurec_loader import NuRecScene, RigTrajectories, read_volume_nurec

    sd = Path(args.scene_dir).expanduser()
    sc = NuRecScene(read_volume_nurec(sd / "volume.nurec"))
    rig = RigTrajectories(sd / "rig_trajectories.json")
    L = args.layer
    p = f".gaussians_nodes.{L}."
    ex = sc.sd[".gaussians_nodes.background.time_embed._extra_state"]
    t_lo, t_hi = int(ex["timestamps_us_min"]), int(ex["timestamps_us_max"])
    out = {"t_lo": t_lo, "t_hi": t_hi, "span_us": t_hi - t_lo}

    # -- 1. time ranges ---------------------------------------------------------
    rr = np.frombuffer(sc.sd[p + "time_embed.timestamps_us_ranges"], np.int64).reshape(-1, 2)
    out["n_tracks_ranges"] = int(rr.shape[0])
    out["ranges_inside_scene"] = bool((rr[:, 0] >= t_lo - 10 ** 7).all()
                                      and (rr[:, 1] <= t_hi + 10 ** 7).all())
    out["range_first5"] = rr[:5].tolist()
    out["range_durations_us"] = (rr[:, 1] - rr[:, 0]).tolist()

    # -- 2. do the per-track pose counts explain 1833? --------------------------
    dq = np.frombuffer(sc.sd[p + "tracks_calib.tracks_delta_q"], np.float16).reshape(-1, 4)
    dt_ = np.frombuffer(sc.sd[p + "tracks_calib.tracks_delta_t"], np.float16).reshape(-1, 3)
    out["n_delta_rows"] = int(dq.shape[0])
    dur = (rr[:, 1] - rr[:, 0]).astype(np.float64)
    for name, step in (("30Hz", 1e6 / 30), ("10Hz", 1e5), ("camera_33ms", 33333.0),
                       ("2Hz", 5e5)):
        n = np.round(dur / step).astype(int) + 1
        out[f"sum_counts_{name}"] = int(n.sum())
    out["delta_t_abs_mean"] = float(np.abs(dt_.astype(np.float32)).mean())
    out["delta_t_abs_max"] = float(np.abs(dt_.astype(np.float32)).max())
    out["delta_q_w_mean"] = float(dq.astype(np.float32)[:, 0].mean())
    out["delta_q_row0"] = dq[0].astype(float).tolist()
    out["delta_t_row0"] = dt_[0].astype(float).tolist()
    out["verdict_deltas_are_poses"] = bool(out["delta_t_abs_max"] > 5.0)

    # -- 3. per-cuboid gaussian geometry ----------------------------------------
    g = sc.gaussians(L, time_basis=np.eye(sc.fourier_dim(L))[0].astype(np.float32))
    cid = np.frombuffer(sc.sd[p + "gaussian_cuboid_ids"], np.int32).copy()
    raw = sc.raw(L)
    q = raw["rotations"]
    qn = np.linalg.norm(q, axis=1)
    scales = np.exp(raw["log_scales"])
    opac = 1.0 / (1.0 + np.exp(-raw["density_logits"].astype(np.float64)))
    shx = np.concatenate([raw["albedo"][:, :1, :], raw["specular"]], axis=1)
    keep = (np.isfinite(raw["positions"]).all(1) & np.isfinite(q).all(1)
            & np.isfinite(scales).all(1) & np.isfinite(opac)
            & np.isfinite(shx).all(axis=(1, 2)) & (qn > 1e-6))
    cid_k = cid[keep]
    out["n_gauss_kept"] = int(keep.sum())
    out["cid_aligned"] = bool(cid_k.shape[0] == g.means.shape[0])
    per = {}
    for c in np.unique(cid_k)[:40]:
        m = g.means[cid_k == c]
        per[int(c)] = {"n": int(m.shape[0]),
                       "centroid": [round(float(x), 2) for x in m.mean(0)],
                       "extent": [round(float(x), 2) for x in (m.max(0) - m.min(0))],
                       "radius": round(float(np.linalg.norm(m - m.mean(0), axis=1).max()), 2)}
    out["per_cuboid"] = per
    cen = np.array([v["centroid"] for v in per.values()])
    out["centroid_norm_mean"] = round(float(np.linalg.norm(cen, axis=1).mean()), 2)
    out["extent_mean"] = [round(float(x), 2) for x in
                          np.array([v["extent"] for v in per.values()]).mean(0)]
    out["verdict_positions_local"] = bool(out["centroid_norm_mean"] < 5.0)

    # -- context: where is the ego, in the same frame? --------------------------
    T0 = rig.T_rig_world(rig.camera_names()[0] if False else "camera_front_wide_120fov", 0)
    out["ego_world_xyz_f0"] = [round(float(x), 2) for x in T0[:3, 3]]
    out["world_to_nre_translation"] = [round(float(x), 2) for x in rig.world_to_nre[:3, 3]]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
