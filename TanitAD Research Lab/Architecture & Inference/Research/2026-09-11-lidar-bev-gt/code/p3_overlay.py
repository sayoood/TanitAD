#!/usr/bin/env python3
"""P3 - VISUAL validation: LiDAR BEV occupancy vs our own `obstacle.offline` boxes.

⭐ The PI's standing preference: **camera projection AND metric BEV together, with a
text overlay**. If the 3D boxes do not sit on the LiDAR returns, the geometry is wrong
and no amount of training fixes it.

Left panel  - the CANONICAL 256x640 CYLINDRICAL frame the model actually sees, with
              LiDAR returns projected in (coloured by range) and the `obstacle.offline`
              cuboids drawn as wireframes.
Right panel - the metric BEV in the rig frame: LiDAR occupancy, the shadow/unobserved
              region, and the same cuboids as footprints.

⛔ The camera path goes through `tanitad.data.calib.cylindrical_rectify` and
`tanitad.data.rig_projection`, never a re-derived formula: the corpus is CYLINDRICAL,
where the column is LINEAR IN AZIMUTH (HFOV 120.000°), and the pinhole formula gives
92.641° on the same data and looks entirely plausible.

⛔ `cylindrical_rectify` REFUSES a corpus-median intrinsic, because the front-wide has
TWO rigs (cy~543 / cy~755) and a ray fan centred on the wrong one is ~215 px off. This
script therefore loads the PER-CLIP principal point and lets that refusal stand.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import lzma
import math
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lidar_bev import (  # noqa: E402
    CartesianBEVSpec, ZBand, DRACO_ATTR_INTENSITY, DRACO_ATTR_POINT_TS_US,
    deskew_rigid, lidar_to_rig, rasterise_cartesian,
)
from p2_build_bev_gt import (  # noqa: E402
    CAM_DIR, EGO_DIR, LIDAR_CACHE, JOIN, box_cells, ego_state_at, load_ego,
    load_extrinsics,
)

INTRINSICS_CSV = (r"C:\Users\Admin\tanitad-data\physicalai-b1\calibration"
                  r"\physicalai_front_wide_intrinsics.csv")
F_REF_CYL = 305.5775
FRAME_H, FRAME_W = 256, 640


def decode_frame(mp4: Path, frame_index: int) -> np.ndarray:
    """[H,W,3] uint8 RGB of one frame, by index."""
    import av
    with av.open(str(mp4)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        for i, f in enumerate(c.decode(st)):
            if i == frame_index:
                return f.to_ndarray(format="rgb24")
    raise SystemExit(f"frame {frame_index} not found in {mp4}")


def per_clip_intrinsics(clip_id: str):
    import csv
    from tanitad.data.calib import FThetaIntrinsics
    with open(INTRINSICS_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["clip_id"] == clip_id:
                return FThetaIntrinsics(
                    poly=tuple(float(row[f"fw_poly_{i}"]) for i in range(5)),
                    cx=float(row["cx"]), cy=float(row["cy"]),
                    width=int(row["width"]), height=int(row["height"]),
                    per_clip=True)
    raise SystemExit(f"no per-clip intrinsics for this clip in {INTRINSICS_CSV}")


def box_corners_rig(a: dict, h: float = 1.6) -> np.ndarray:
    """8 corners [8,3] of a cuboid footprint extruded from the road plane.

    ⚠️ The B1 join schema drops `z` and `size_z` (it carries cx, cy, yaw, l, w only),
    so the vertical extent here is a DECLARED DRAWING CHOICE, not a measurement, and it
    is stated in the panel. The footprint - the half that the BEV target uses - is
    exact.
    """
    cx, cy, yaw, l, w = (float(a["cx"]), float(a["cy"]), float(a["yaw"]),
                         float(a["l"]), float(a["w"]))
    c, s = math.cos(yaw), math.sin(yaw)
    base = np.array([[+l / 2, +w / 2], [+l / 2, -w / 2],
                     [-l / 2, -w / 2], [-l / 2, +w / 2]])
    rot = np.stack([base[:, 0] * c - base[:, 1] * s,
                    base[:, 0] * s + base[:, 1] * c], axis=1)
    xy = rot + np.array([cx, cy])
    lo = np.concatenate([xy, np.zeros((4, 1))], axis=1)
    hi = np.concatenate([xy, np.full((4, 1), h)], axis=1)
    return np.concatenate([lo, hi], axis=0)


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", default="180,300,420")
    ap.add_argument("--out-dir", default="media")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly
    import torch
    import pyarrow.parquet as pq
    import DracoPy
    import tanitad
    from tanitad.data.calib import CanonicalFrame, cylindrical_rectify
    from tanitad.data.rig_projection import RigCamera

    print(f"[p3] tanitad imported from {tanitad.__file__}")

    clip = args.clip
    sha12 = hashlib.sha256(clip.encode()).hexdigest()[:12]
    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    cart, zb = CartesianBEVSpec(), ZBand()

    frame = CanonicalFrame(height=FRAME_H, width=FRAME_W, f_ref=F_REF_CYL,
                           projection="cylindrical")
    intr = per_clip_intrinsics(clip)
    cam_ext = load_extrinsics(clip, "camera_front_wide_120fov")
    lid_ext = load_extrinsics(clip, "lidar_top_360fov")

    class _E:
        def __init__(self, d):
            self.__dict__.update(d)

        def rotation_cam_to_vehicle(self):
            from tanitad.data.physicalai import FrontWideExtrinsics
            return FrontWideExtrinsics(**{k: self.__dict__[k]
                                          for k in ("qx", "qy", "qz", "qw",
                                                    "x", "y", "z")}
                                       ).rotation_cam_to_vehicle()

    rigcam = RigCamera.from_extrinsics(_E(cam_ext), frame)

    lidar_pq = Path(LIDAR_CACHE) / f"{clip}.lidar_top_360fov.parquet"
    pf = pq.ParquetFile(lidar_pq)
    sp = pq.read_table(lidar_pq, columns=["spin_start_timestamp",
                                          "spin_end_timestamp"]).to_pydict()
    s_mid = (np.asarray(sp["spin_start_timestamp"], dtype=np.int64)
             + np.asarray(sp["spin_end_timestamp"], dtype=np.int64)) // 2
    cam_t = np.asarray(pq.read_table(Path(CAM_DIR) / f"{clip}.timestamps.parquet")
                       .to_pydict()["timestamp"], dtype=np.int64)
    ego = load_ego(clip)

    join_t, join_ag = [], []
    with lzma.open(JOIN, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("clip_id") == clip:
                join_t.append(float(r["t_s"]))
                join_ag.append(r.get("agents", []))
    join_t = np.asarray(join_t)

    mp4 = Path(CAM_DIR) / f"{clip}.mp4"
    written = []
    for fi in [int(x) for x in args.frames.split(",") if x.strip()]:
        t_ref = int(cam_t[fi])
        si = int(np.abs(s_mid - t_ref).argmin())
        blob = pf.read_row_group(si, columns=["draco_encoded_pointcloud"]).column(0)[0].as_py()
        pc = DracoPy.decode(blob)
        pts = np.asarray(pc.points)
        if pts.shape[0] == 0 or float(np.abs(pts).max()) == 0.0:
            raise SystemExit(f"frame {fi}: EMPTY cloud - refusing to draw")
        attrs = {a["unique_id"]: a["data"] for a in pc.attributes}
        inten = attrs[DRACO_ATTR_INTENSITY][:, 0].astype(np.uint8)
        pt_ts = attrs[DRACO_ATTR_POINT_TS_US][:, 0].astype(np.int64)
        v_ms, yaw_rate = ego_state_at(ego, t_ref)
        rig = deskew_rigid(lidar_to_rig(pts, lid_ext), pt_ts, t_ref, v_ms, yaw_rate)
        C = rasterise_cartesian(rig, inten, cart, zb)

        ji = int(np.abs(join_t - t_ref * 1e-6).argmin())
        agents = join_ag[ji]

        img = decode_frame(mp4, fi)
        vid = torch.from_numpy(img).permute(2, 0, 1)[None]
        can = cylindrical_rectify(vid, intr, frame)[0].permute(1, 2, 0).numpy()
        obs_frac = float(cylindrical_rectify.last_observed_frac)

        # ---- project LiDAR into the canonical frame ----------------------
        keep = (rig[:, 0] > 0.5) & (np.hypot(rig[:, 0], rig[:, 1]) < 70) & (rig[:, 2] < 4.0)
        p = torch.as_tensor(rig[keep], dtype=torch.float64)
        col, row, valid = rigcam.project(p)
        col, row, valid = col.numpy(), row.numpy(), valid.numpy().astype(bool)
        inside = valid & (col >= 0) & (col < FRAME_W) & (row >= 0) & (row < FRAME_H)
        rr = np.hypot(rig[keep][:, 0], rig[keep][:, 1])

        fig = plt.figure(figsize=(17.5, 7.2), dpi=110)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1, 0.16],
                              hspace=0.18, wspace=0.10)
        axc = fig.add_subplot(gs[0, 0])
        axb = fig.add_subplot(gs[0, 1])
        axt = fig.add_subplot(gs[1, :]); axt.axis("off")

        axc.imshow(can)
        sc = axc.scatter(col[inside], row[inside], c=rr[inside], s=0.7, cmap="turbo",
                         vmin=0, vmax=60, alpha=0.55, linewidths=0)
        n_box_drawn = 0
        for a in agents:
            cn = box_corners_rig(a)
            pc3 = torch.as_tensor(cn, dtype=torch.float64)
            cc, rrw, vv = rigcam.project(pc3)
            cc, rrw, vv = cc.numpy(), rrw.numpy(), vv.numpy().astype(bool)
            if not vv.all():
                continue
            if cc.min() < -FRAME_W or cc.max() > 2 * FRAME_W:
                continue
            drew = False
            for i, j in EDGES:
                if abs(cc[i] - cc[j]) > FRAME_W * 0.5:
                    continue
                axc.plot([cc[i], cc[j]], [rrw[i], rrw[j]], "-", lw=1.3,
                         color="#ff2d95", alpha=0.95)
                drew = True
            n_box_drawn += drew
        axc.set_xlim(0, FRAME_W); axc.set_ylim(FRAME_H, 0)
        axc.set_title(f"canonical 256x640 CYLINDRICAL (f_ref {F_REF_CYL}, HFOV 120.000 deg)"
                      f"  ·  LiDAR returns coloured by range  ·  obstacle.offline boxes (magenta)",
                      fontsize=8.5)
        axc.set_xticks([]); axc.set_yticks([])
        cb = fig.colorbar(sc, ax=axc, fraction=0.020, pad=0.005, location="bottom",
                          orientation="horizontal")
        cb.set_label("LiDAR return range [m]", fontsize=7.5); cb.ax.tick_params(labelsize=7)

        # ---- BEV panel ----------------------------------------------------
        occ = C["occ"]; obsv = C["observed"] > 0
        disp = np.zeros((cart.n_x, cart.n_y, 3), dtype=np.float32)
        disp[...] = np.array([0.10, 0.10, 0.13])          # unobserved / shadow
        disp[obsv] = np.array([0.16, 0.30, 0.22])          # observed free
        disp[occ > 0] = np.array([1.00, 0.85, 0.10])       # LiDAR occupancy
        # ⛔ AXIS SENSE IS ASSERTED, NOT EYEBALLED. `disp` is [n_x, n_y] with row i =
        # x in [0.5i, 0.5(i+1)) and col j = y in [-16 + 0.5j, ...), i.e. col 0 is the
        # RIGHT. Displaying it needs transpose to [n_y, n_x] and `origin="lower"`, and
        # NOTHING ELSE: an extra `[::-1]` mirrors the panel while every number on it
        # stays correct. That is exactly `R-2026-09-08-wpa-mirror` in a plotting
        # costume, and the first render of this figure had it.
        assert disp.shape[:2] == (cart.n_x, cart.n_y)
        axb.imshow(np.transpose(disp, (1, 0, 2)), origin="lower",
                   extent=[0, cart.x_max_m, -cart.y_half_m, cart.y_half_m],
                   aspect="equal", interpolation="nearest")
        for a in agents:
            cn = box_corners_rig(a)[:4, :2]
            if cn[:, 0].max() < -5 or cn[:, 0].min() > cart.x_max_m + 5:
                continue
            axb.add_patch(MplPoly(np.stack([cn[:, 0], cn[:, 1]], axis=1),
                                  closed=True, fill=False, ec="#ff2d95", lw=1.6))
        axb.plot([0], [0], marker="^", ms=9, color="#00e5ff")
        # ⭐ the axis-sense control, drawn ON the panel so a reader can falsify it:
        # a marker at a KNOWN rig coordinate, labelled. If "+10 m LEFT" is not in the
        # upper half of this panel, the panel is mirrored.
        axb.plot([8.0], [10.0], marker="*", ms=11, color="#ffffff", mec="#000000", mew=0.6)
        axb.annotate("control: rig (x=8, y=+10) = 10 m LEFT", (8.0, 10.0),
                     textcoords="offset points", xytext=(7, -3), fontsize=6.5,
                     color="#ffffff")
        axb.set_xlim(0, cart.x_max_m); axb.set_ylim(-cart.y_half_m, cart.y_half_m)
        axb.set_xlabel("x forward [m]", fontsize=8)
        axb.set_ylabel("y LEFT [m]", fontsize=8)
        axb.tick_params(labelsize=7)
        axb.set_title("metric BEV, RIG frame, 0.5 m cells  ·  yellow = LiDAR occupancy  ·  "
                      "dark = shadow/unobserved", fontsize=8.5)
        axb.grid(alpha=0.15, lw=0.4)

        bc = box_cells(agents, cart) & obsv
        hit = float((occ > 0)[bc].mean()) if bc.sum() else float("nan")
        bcm = box_cells(agents, cart, mirror=True) & obsv
        hit_m = float((occ > 0)[bcm].mean()) if bcm.sum() else float("nan")
        base = float((occ > 0)[obsv].mean())

        txt = (
            f"clip sha12 {sha12}   camera frame {fi} @ t = {t_ref/1e6:+.4f} s   |   "
            f"LiDAR spin {si} (10.0006 Hz), |dt| = {abs(s_mid[si]-t_ref)/1e3:.1f} ms   "
            f"join row {ji}, |dt| = {abs(join_t[ji]-t_ref*1e-6)*1e3:.1f} ms\n"
            f"points {pts.shape[0]:,} (Draco, LiDAR sensor frame) -> rig via sensor_extrinsics "
            f"z={lid_ext['z']:.4f} m   |   ego v = {v_ms:.2f} m/s, yaw rate {yaw_rate:+.3f} rad/s, "
            f"intra-spin skew {v_ms*float(pt_ts.max()-pt_ts.min())*1e-6:.2f} m (DESKEWED per-point)\n"
            f"BEV: occupancy {occ.mean()*100:.2f} % of grid, observed {obsv.mean()*100:.1f} %   |   "
            f"agents {len(agents)}, box cells observed {int(bc.sum())}   |   "
            f"HIT RATE real {hit:.3f}  vs  mirror_y {hit_m:.3f}  vs  marginal {base:.3f}\n"
            f"units m; extent x[0,{cart.x_max_m:g}] y[{-cart.y_half_m:g},{cart.y_half_m:g}]; "
            f"cell {cart.cell_m} m; frame RIG (+x fwd, +y LEFT, +z up, origin rear axle on the road "
            f"plane); z-band obstacle [{zb.obstacle_min_m},{zb.obstacle_max_m}] m + spread "
            f">= {zb.spread_min_m} m   |   camera observed frac {obs_frac:.3f}; box height 1.6 m is a "
            f"DRAWING choice (the join carries no size_z)   |   LABEL ONLY - inference is VISION-ONLY"
        )
        axt.text(0.005, 0.96, txt, va="top", ha="left", fontsize=7.4, family="monospace",
                 transform=axt.transAxes)

        out = outd / f"lidar_bev_overlay_{sha12}_f{fi:04d}.png"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(str(out))
        print(f"[p3] {out}  hit_real={hit:.3f} hit_mirror={hit_m:.3f} marginal={base:.3f} "
              f"boxes_drawn_in_image={n_box_drawn}")

    print(json.dumps({"written": written}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
