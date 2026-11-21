"""Map-signal audit, task 1/4 input: what the BEV lift can RESOLVE, from the project's own geometry.

ANALYTIC-EXACT (no model, no learning): every number comes from `tanitad.models.bev_lift.
build_lift_geometry` / `project_rig_points` applied to the REAL per-clip extrinsics of the eval kit
(`D:/refcv6_eval_kit/data/refcv6_train_eval139_extrinsics.json`, the table refcv6 trained with) on
the 416 x 1024 cylindrical frame refcv6 read (`config.json` refcv6_perception.frame: f_ref
488.92398517830253). Clips = the 137 eval clips that have a SAM3 GT file in the eval kit.

What it measures, per x band of the rig grid (0-20 / 20-40 / 40-60 m):
  * how much of the grid the lift can sample at all (`map_valid` = any height valid), with the
    bottom-43-row observed mask refcv6's LiftGeometryBank carried (D-REFCV6-EQUALIZE-DROPPED: the
    LIFT got 43, the trunk got 0) and without it;
  * how much of it is sampled ON THE ROAD PLANE (height 0) -- the only height at which a road
    marking is in the sampled pixel;
  * image pixels per metre, lateral (d col / d y) and longitudinal (d row / d x), at height 0;
  * how many DISTINCT feature rows / columns of the stride-16 (refcv6), stride-8 (NEW-2) and
    stride-4 maps the road-plane samples of a band fall in -- the band's effective feature
    resolution.

Output: raw/lift_resolution.json. CPU, a few seconds per stride. Clip ids never printed (sha12).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "raw" / "lift_resolution.json"
KIT = Path("D:/refcv6_eval_kit/data")

import tanitad  # noqa: E402
assert "msa_tree_" in tanitad.__file__, f"tanitad from {tanitad.__file__}, not a clean msa tree"
from tanitad.data.bev_raster import BEVGrid, GRID_DEFAULT  # noqa: E402
from tanitad.models.bev_lift import HEIGHTS_M, build_lift_geometry  # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width  # noqa: E402


def sha12(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:12]


def main() -> None:
    t0 = time.time()
    frame = frame_for_width(1024, 416)
    assert abs(frame.f_ref - 488.92398517830253) < 1e-9, frame
    extr = json.load(open(KIT / "refcv6_train_eval139_extrinsics.json", encoding="utf-8"))
    by12 = {sha12(k): v for k, v in extr.items()}
    gt12 = sorted(Path(p).name.split(".")[0]
                  for p in glob.glob(str(KIT / "sam3_gt_eval_thor137" / "*.sam3mapgt.npz")))
    clips = [s for s in gt12 if s in by12]
    assert len(clips) == len(gt12) == 137, (len(clips), len(gt12))
    H, W = frame.height, frame.width
    eq_rows = 43
    obs = torch.ones(H, W, dtype=torch.bool)
    obs[-eq_rows:, :] = False
    bands = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]
    res: dict = {"frame": {"H": H, "W": W, "f_ref": frame.f_ref, "projection": frame.projection},
                 "heights_m": list(HEIGHTS_M), "n_clips": len(clips),
                 "clips_sha12_first5": clips[:5], "eq_rows_lift": eq_rows,
                 "bands_m": bands, "per_stride": {}}
    # --- control: the analytic image resolution of the cylindrical frame (1 px = 1/f_ref rad) ---
    res["mrad_per_px"] = 1000.0 / frame.f_ref
    for stride in (16, 8, 4):
        acc = {b: {"valid_any_eq": [], "valid_any_noeq": [], "z0_valid_eq": [], "z0_valid_noeq": [],
                   "lat_px_per_m": [], "lon_px_per_m": [], "n_feat_rows": [], "n_feat_cols": [],
                   "n_img_rows": [], "valid_any_but_z0_masked_by_eq": []}
               for b in range(3)}
        near_eq_x = []                       # rig x where the road plane leaves the masked strip
        cam_h = []
        at_x = {xm: {"lat": [], "lon": [], "row": []} for xm in (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 59.0)}
        for s in clips:
            e = by12[s]
            g_eq = build_lift_geometry(e, frame=frame, stride=stride, observed=obs)
            g_no = build_lift_geometry(e, frame=frame, stride=stride, observed=None)
            cam_h.append(float(e["z"]))
            X = (torch.arange(GRID_DEFAULT.shape[0], dtype=torch.float64) + 0.5) * GRID_DEFAULT.cell_m
            Y = -GRID_DEFAULT.y_half_m + (torch.arange(GRID_DEFAULT.shape[1], dtype=torch.float64)
                                          + 0.5) * GRID_DEFAULT.cell_m
            row0 = g_no.row[0]                                  # [X, Y] z = 0 (road plane)
            col0 = g_no.col[0]
            v0_no = g_no.valid[0]
            v0_eq = g_eq.valid[0]
            # finite-difference Jacobians on the 0.5 m grid, z = 0
            drow_dx = torch.zeros_like(row0)
            drow_dx[:-1] = (row0[1:] - row0[:-1]) / GRID_DEFAULT.cell_m
            drow_dx[-1] = drow_dx[-2]
            dcol_dy = torch.zeros_like(col0)
            dcol_dy[:, :-1] = (col0[:, 1:] - col0[:, :-1]) / GRID_DEFAULT.cell_m
            dcol_dy[:, -1] = dcol_dy[:, -2]
            centre = (Y.abs() <= 4.0)[None, :].expand_as(row0)     # the ego's own lanes
            for bi, (lo, hi) in enumerate(bands):
                m = ((X >= lo) & (X < hi))[:, None].expand_as(row0)
                acc[bi]["valid_any_eq"].append(float(g_eq.valid.any(0)[m].float().mean()))
                acc[bi]["valid_any_noeq"].append(float(g_no.valid.any(0)[m].float().mean()))
                acc[bi]["z0_valid_eq"].append(float(v0_eq[m].float().mean()))
                acc[bi]["z0_valid_noeq"].append(float(v0_no[m].float().mean()))
                acc[bi]["valid_any_but_z0_masked_by_eq"].append(
                    float((g_eq.valid.any(0) & v0_no & ~v0_eq)[m].float().mean()))
                mc = m & centre & v0_no
                if bool(mc.any()):
                    acc[bi]["lat_px_per_m"].append(float(dcol_dy[mc].abs().median()))
                    acc[bi]["lon_px_per_m"].append(float(drow_dx[mc].abs().median()))
                    acc[bi]["n_feat_rows"].append(int(torch.unique(torch.floor(row0[mc] / stride)).numel()))
                    acc[bi]["n_img_rows"].append(int(torch.unique(torch.floor(row0[mc])).numel()))
                    acc[bi]["n_feat_cols"].append(int(torch.unique(torch.floor(col0[mc] / stride)).numel()))
            # per-range Jacobians on the two centre columns (y = -0.25 / +0.25 m)
            for xm, d in at_x.items():
                i = int(xm / GRID_DEFAULT.cell_m)
                for j in (GRID_DEFAULT.shape[1] // 2 - 1, GRID_DEFAULT.shape[1] // 2):
                    if bool(v0_no[i, j]):
                        d["lat"].append(float(dcol_dy[i, j].abs()))
                        d["lon"].append(float(drow_dx[i, j].abs()))
                        d["row"].append(float(row0[i, j]))
            # rig x (centre column) at which the road-plane sample enters the observed area
            jc = GRID_DEFAULT.shape[1] // 2
            ok = (v0_no[:, jc] & ~v0_eq[:, jc])
            near_eq_x.append(float(X[ok].max()) if bool(ok.any()) else float("nan"))
        rows = []
        for bi, (lo, hi) in enumerate(bands):
            a = acc[bi]
            q = lambda v: (None if not v else [round(float(np.percentile(v, p)), 4) for p in (10, 50, 90)])
            lat = np.median(a["lat_px_per_m"]) if a["lat_px_per_m"] else float("nan")
            lon = np.median(a["lon_px_per_m"]) if a["lon_px_per_m"] else float("nan")
            rows.append({
                "band_m": [lo, hi],
                "map_valid_frac_with_eq43_p10_50_90": q(a["valid_any_eq"]),
                "map_valid_frac_no_eq_p10_50_90": q(a["valid_any_noeq"]),
                "road_plane_z0_valid_frac_with_eq43_p10_50_90": q(a["z0_valid_eq"]),
                "road_plane_z0_valid_frac_no_eq_p10_50_90": q(a["z0_valid_noeq"]),
                "frac_cells_valid_only_via_heights_gt0_because_eq43_masks_z0_p10_50_90":
                    q(a["valid_any_but_z0_masked_by_eq"]),
                "lateral_img_px_per_m_median": round(float(lat), 3),
                "longitudinal_img_px_per_m_median": round(float(lon), 3),
                "lateral_m_per_img_px": round(1.0 / lat, 4) if lat == lat and lat > 0 else None,
                "longitudinal_m_per_img_px": round(1.0 / lon, 4) if lon == lon and lon > 0 else None,
                f"lateral_m_per_stride{stride}_feature": round(stride / lat, 4) if lat > 0 else None,
                f"longitudinal_m_per_stride{stride}_feature": round(stride / lon, 4) if lon > 0 else None,
                "distinct_feature_rows_in_band_central_p10_50_90": q(a["n_feat_rows"]),
                "distinct_feature_cols_in_band_central_p10_50_90": q(a["n_feat_cols"]),
                "distinct_image_rows_in_band_central_p10_50_90": q(a["n_img_rows"]),
            })
        per_x = []
        for xm, d in at_x.items():
            if not d["lat"]:
                per_x.append({"rig_x_m": xm, "n": 0})
                continue
            lat, lon = float(np.median(d["lat"])), float(np.median(d["lon"]))
            per_x.append({"rig_x_m": xm, "n": len(d["lat"]),
                          "img_row_median": round(float(np.median(d["row"])), 2),
                          "lat_px_per_m": round(lat, 3), "lon_px_per_m": round(lon, 3),
                          "lat_m_per_px": round(1 / lat, 4), "lon_m_per_px": round(1 / lon, 4),
                          f"lat_m_per_s{stride}_feature": round(stride / lat, 3),
                          f"lon_m_per_s{stride}_feature": round(stride / lon, 3),
                          "px_width_of_0.10m_line": round(0.10 * lat, 2),
                          "px_width_of_0.15m_line": round(0.15 * lat, 2),
                          "px_height_of_0.40m_stop_line": round(0.40 * lon, 2),
                          "px_height_of_3m_crosswalk_depth": round(3.0 * lon, 2)})
        res["per_stride"][str(stride)] = {
            "per_rig_x_centre_columns": per_x,
            "feat_hw": [H // stride, W // stride], "bands": rows,
            "rig_x_where_road_plane_leaves_eq43_strip_centre_col_p10_50_90":
                [round(float(np.nanpercentile(near_eq_x, p)), 3) for p in (10, 50, 90)],
        }
        print(f"[liftres] stride {stride}: done ({time.time() - t0:.1f} s)", flush=True)
    res["camera_mount_z_m_p10_50_90"] = [round(float(np.percentile(cam_h, p)), 4) for p in (10, 50, 90)]
    # --- controls that must read known values ---------------------------------------------------
    # (1) lateral px/m at range r on the boresight must be ~ f_ref / r (cylindrical: col = f*atan2)
    # (2) the stride only changes the feature counts, never the image px/m
    s16 = res["per_stride"]["16"]["bands"]
    s8 = res["per_stride"]["8"]["bands"]
    res["controls"] = {
        "img_px_per_m_identical_across_strides": all(
            s16[i]["lateral_img_px_per_m_median"] == s8[i]["lateral_img_px_per_m_median"]
            and s16[i]["longitudinal_img_px_per_m_median"] == s8[i]["longitudinal_img_px_per_m_median"]
            for i in range(3)),
    }
    res["elapsed_s"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
