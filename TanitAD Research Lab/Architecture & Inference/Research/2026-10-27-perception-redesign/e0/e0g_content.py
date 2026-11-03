"""e0g_content.py -- the E0g CONTENT ASSERTION on real data (REFCV9_DESIGN s4.1 E0g: "non-zero foreground share per frame,
printed mean").  Reads TRAIN-split `obstacle.offline` cuboids + the clip's own front-wide extrinsics / f-theta intrinsics,
builds `e0g_depth_targets` for every sampled frame, prints and writes mean / p5 / p95 of the foreground share, the depth
histogram and the controls.  Counts / sha12 / distributions only -- no clip id is written.

    W=$(cygpath -w <tree>); export PYTHONPATH="$W\\stack;$W\\stack\\scripts"; CUDA_VISIBLE_DEVICES=-1 \
    python -X utf8 e0g_content.py --root C:/Users/Admin/tanitad-data/physicalai --n-clips 30 --out raw/E0G_RESULT.json

Selection rule (frozen here, no episode re-selection -- parity is sacred): clips of `obstacle_offline_b1train` whose
`clip_index.split == "train"`, ordered by sha256(clip_id) hex ascending, first `--n-clips`; frames at
`--times` seconds (clip-relative; the cuboid of each track nearest in time within `DEFAULT_TOL_S`).  ALL cuboids are
ray-cast (every class, every distance), exactly as `precompute_vis1_sidecar` uses them as occluders.
Assertion B: every frame with ZERO foreground (that has cuboids) has no cuboid point inside the field within 100 m.
Assertion A: every frame that has >= 1 cuboid whose centre is in front, inside the observed frame and within 95 m
has fg pixels > 0 (a cuboid that far in range cannot vanish -- a nearer one would be the one hit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0g_depth_targets as E  # noqa: E402

DEPTH_EDGES = (0, 5, 10, 20, 30, 40, 60, 80, 100)


def q(a, ps=(5, 50, 95)):
    a = np.asarray(a, dtype=np.float64)
    return {"mean": float(a.mean()), **{f"p{p}": float(np.percentile(a, p)) for p in ps},
            "min": float(a.min()), "max": float(a.max())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n-clips", type=int, default=30)
    ap.add_argument("--times", default="3,6,9,12,15")
    ap.add_argument("--n-vis1", type=int, default=20, help="frames also run through vis1_zbuffer.zbuffer (id agreement)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import pandas as pd
    import torch
    from build_rig_extrinsics_table import _front_wide_rows
    from tanitad.data import vis1_zbuffer as VZ
    from tanitad.data.agent_cuboid_gt import cuboids_at_time, read_clip_cuboids
    from tanitad.data.rig_projection import project_cam_to_frame

    root = Path(a.root)
    ci = pd.read_parquet(root / "clip_index.parquet")
    obs_dir = root / "labels" / "obstacle_offline_b1train"
    stems = sorted(p.stem for p in obs_dir.glob("*.parquet"))
    train = [s for s in stems if s in ci.index and str(ci.loc[s, "split"]) == "train"]
    order = sorted(train, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:a.n_clips]
    times = [float(x) for x in a.times.split(",")]
    print(f"[e0g] b1train parquets {len(stems)}, split=train {len(train)}, using {len(order)} clips x {len(times)} times")

    rec = {"share": [], "cell8": [], "cell16": [], "boxes_scene": [], "boxes_visible": [], "expected_visible": [],
           "viol_A": 0, "viol_B": 0, "zero_fg_frames": 0, "skipped_invalid": 0, "skipped_inside": 0, "oor_px": 0, "unobs_px": 0, "fg_on_unobs_px": 0,
           "obs_mask_f_theta": 0, "obs_mask_bounds_only": 0, "t_frame_s": []}
    depth_hist = np.zeros(len(DEPTH_EDGES) - 1, np.int64)
    tau_hist = np.zeros(8, np.int64)                     # fg PIXELS by LID-bin octile (10 bins each)
    vis1 = []
    n_frames = 0
    ext_cache: dict = {}
    for ci_i, cid in enumerate(order):
        ch = int(ci.loc[cid, "chunk"])
        if ch not in ext_cache:
            ext_cache[ch] = _front_wide_rows(pd.read_parquet(root / "calibration" / "sensor_extrinsics" /
                                                              f"sensor_extrinsics.chunk_{ch:04d}.parquet"))
        rows = ext_cache[ch]
        rows = rows[rows["clip_id"].astype(str) == cid]
        if not len(rows):
            continue
        r = rows.iloc[0]
        entry = dict(qx=float(r.qx), qy=float(r.qy), qz=float(r.qz), qw=float(r.qw), x=float(r.x), y=float(r.y),
                     z=float(r.z), chunk=ch)
        R, t, obs, why, _ = VZ.clip_camera(cid, entry, [str(root / "calibration")])
        rec["obs_mask_f_theta" if why == VZ.MASK_SOURCE_FTHETA else "obs_mask_bounds_only"] += 1
        cam = E.Camera(R, t)
        tab = read_clip_cuboids(obs_dir / f"{cid}.parquet")
        for ts in times:
            c8 = cuboids_at_time(tab, ts)                       # (cx, cy, yaw, l, w, occ, cz, h)
            boxes = (np.c_[c8[:, 0], c8[:, 1], c8[:, 6], c8[:, 3], c8[:, 4], c8[:, 7], c8[:, 2]]
                     if c8.shape[0] else np.zeros((0, 7)))
            t0 = time.time()
            dt = E.build_depth_targets(boxes, cam, observed=obs)
            rec["t_frame_s"].append(time.time() - t0)
            st = E.frame_stats(dt)
            n_frames += 1
            rec["share"].append(st["fg_pixel_share"])
            rec["cell8"].append(st["fg_cell_share_s8"])
            rec["cell16"].append(st["fg_cell_share_s16"])
            rec["boxes_scene"].append(int(boxes.shape[0]))
            rec["boxes_visible"].append(st["boxes_visible"])
            rec["zero_fg_frames"] += int(st["fg_pixel_share"] == 0.0)
            rec["skipped_invalid"] += dt.meta["skipped"]["invalid"]
            rec["skipped_inside"] += dt.meta["skipped"]["camera_inside"]
            rec["oor_px"] += int((dt.pixel_state == E.PIX_OOR).sum())
            rec["unobs_px"] += int((dt.pixel_state == E.PIX_UNOBS).sum())
            rec["fg_on_unobs_px"] += int(((dt.pixel_state == E.PIX_UNOBS) & (dt.depth < E.D_MAX_M)).sum())   # cuboid pixels the sensor never saw
            m = dt.pixel_state == E.PIX_FG
            depth_hist += np.histogram(dt.depth[m], bins=DEPTH_EDGES)[0]
            tau_hist += np.histogram(E.lid_tau(dt.depth[m]), bins=np.arange(0, 81, 10))[0]
            # assertion A
            n_exp = 0
            if boxes.shape[0]:
                pc = (boxes[:, :3] - t) @ R
                col, row, valid = project_cam_to_frame(torch.from_numpy(pc), cam.frame)
                rng = np.linalg.norm(pc, axis=1)
                ci_, ri_ = np.clip(np.rint(col.numpy()).astype(int), 0, E.FRAME_W - 1), np.clip(np.rint(row.numpy()).astype(int), 0, E.FRAME_H - 1)
                n_exp = int((valid.numpy() & (rng <= 95.0) & obs[ri_, ci_]).sum())
            rec["expected_visible"].append(n_exp)
            if n_exp > 0 and st["fg_pixel_share"] == 0.0:
                rec["viol_A"] += 1
            if st["fg_pixel_share"] == 0.0 and boxes.shape[0]:
                # assertion B: a frame with NO foreground must have no cuboid point (8 corners + centre) in the field within 100 m
                pts = np.concatenate([np.vstack([E._corners(*b[:6], b[6]), b[None, :3]]) for b in boxes])
                pcb = (pts - t) @ R
                _, _, vb = project_cam_to_frame(torch.from_numpy(pcb), cam.frame)
                near = np.linalg.norm(pcb, axis=1) <= 100.0
                rec["viol_B"] += int((vb.numpy() & near).any())
            if len(vis1) < a.n_vis1 and boxes.shape[0]:
                _, _, _, _, idb = VZ.zbuffer(boxes, R, t, obs)
                ref = idb[VZ.PAD_V:VZ.PAD_V + E.FRAME_H, VZ.PAD_U:VZ.PAD_U + E.FRAME_W]
                both = (ref >= 0) | (dt.box_id >= 0)
                vis1.append({"agree_all": float((ref == dt.box_id).mean()),
                             "agree_obj": (float(((ref == dt.box_id) & both).sum() / both.sum()) if both.sum() else None),
                             "n_obj_px": int(both.sum())})
        if (ci_i + 1) % 10 == 0:
            print(f"[e0g] {ci_i + 1}/{len(order)} clips, {n_frames} frames, mean fg share {np.mean(rec['share']):.4f}", flush=True)

    sh = np.asarray(rec["share"])
    has_obj = np.asarray(rec["boxes_scene"]) > 0
    out = {
        "what": "E0g content assertion: box-derived foreground depth targets on real TRAIN-split frames",
        "evidence_class": "MEASURED (ours; this file)",
        "selection_rule": "obstacle_offline_b1train clips with clip_index.split == 'train', sha256(clip_id) ascending, first n_clips; "
                          "frames at the listed clip-relative times; cuboids nearest-in-time per track within DEFAULT_TOL_S",
        "n_clips": len(order), "n_frames": int(n_frames), "times_s": times,
        "frame": {"H": E.FRAME_H, "W": E.FRAME_W, "f_ref": float(E.Camera(np.eye(3), np.zeros(3)).f_ref), "projection": "cylindrical"},
        "params": {"depth_kind": "range", "n_bins": E.N_BINS, "d_min": E.D_MIN_M, "d_max": E.D_MAX_M,
                   "alpha_fg": E.ALPHA_FG, "alpha_bg": E.ALPHA_BG, "fg_min_frac": E.FG_MIN_FRAC, "strides": list(E.STRIDES)},
        "fg_pixel_share": q(sh), "fg_cell_share_s8": q(rec["cell8"]), "fg_cell_share_s16": q(rec["cell16"]),
        "fg_pixel_share_frames_with_cuboids": q(sh[has_obj]) if has_obj.any() else None,
        "frames_with_zero_fg": int(rec["zero_fg_frames"]), "frames_with_zero_cuboids": int((~has_obj).sum()),
        "assertion_A_violations": int(rec["viol_A"]),
        "assertion_B_violations": int(rec["viol_B"]),
        "assertion_B_note": "B: every zero-foreground frame that has cuboids has none of their 8 corners + centre inside the field within 100 m",
        "assertion_A_frames_with_expected_visible_box": int((np.asarray(rec["expected_visible"]) > 0).sum()),
        "cuboids_per_frame": q(rec["boxes_scene"]), "cuboids_with_fg_pixels_per_frame": q(rec["boxes_visible"]),
        "depth_hist_fg_pixels": {"edges_m": list(DEPTH_EDGES), "counts": depth_hist.tolist(),
                                 "share": (depth_hist / max(depth_hist.sum(), 1)).round(5).tolist()},
        "lid_octile_hist_fg_pixels": {"bins_per_octile": 10, "counts": tau_hist.tolist()},
        "skipped": {"invalid": rec["skipped_invalid"], "camera_inside": rec["skipped_inside"]},
        "out_of_range_pixels_total": rec["oor_px"], "unobserved_pixels_total": rec["unobs_px"],
        "cuboid_pixels_on_unobserved_total": rec["fg_on_unobs_px"],
        "cuboid_pixels_on_unobserved_share_of_all_cuboid_pixels": float(rec["fg_on_unobs_px"] / max(rec["fg_on_unobs_px"] + int(depth_hist.sum()), 1)),
        "observed_mask": {"f_theta": rec["obs_mask_f_theta"], "image_bounds_only": rec["obs_mask_bounds_only"]},
        "vis1_zbuffer_id_agreement": {"n_frames": len(vis1),
                                      "n_frames_with_object_pixels": int(sum(v["agree_obj"] is not None for v in vis1)),
                                      "min_agree_all": float(min(v["agree_all"] for v in vis1)) if vis1 else None,
                                      "min_agree_over_object_pixels": min((v["agree_obj"] for v in vis1 if v["agree_obj"] is not None), default=None),
                                      "mean_agree_over_object_pixels": (float(np.mean([v["agree_obj"] for v in vis1 if v["agree_obj"] is not None]))
                                                                        if any(v["agree_obj"] is not None for v in vis1) else None),
                                      "total_object_pixels_compared": int(sum(v["n_obj_px"] for v in vis1))},
        "seconds_per_frame": q(rec["t_frame_s"]),
        "no_ids_written": True,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_clips", "n_frames", "fg_pixel_share", "fg_cell_share_s8", "fg_cell_share_s16",
                                          "frames_with_zero_fg", "assertion_A_violations", "assertion_B_violations", "depth_hist_fg_pixels",
                                          "vis1_zbuffer_id_agreement", "seconds_per_frame")}, indent=1))
    return 0 if (out["assertion_A_violations"] == 0 and out["assertion_B_violations"] == 0 and out["n_frames"] >= 50) else 1


if __name__ == "__main__":
    raise SystemExit(main())
