"""⛔⛔ SUPERSEDED 2026-09-13 -- DO NOT REUSE THIS PACKING OR THIS RENDERER.

The views built here (focal 313.7 / 664.2 / 1732.3 px @896) are OUTSIDE the fixed camera geometry
Qwen-Drive's weights were trained on (721.0 px, 63.7 deg). MEASURED on the upstream demo: changing ONLY
the focal drops detection 54/57 -> 0/57. The renderer decoded occupancy/map from a guessed schema.
Replacement: TanitAD Research Lab/Data Engineering/Research/2026-09-13-qwen-drive-usage-review/
(code/build_frames_v2.py + the upstream scripts/visualize_perception.py). Kept for provenance only.
"""
"""Render one Qwen-Drive perception frame to the PI's standing visual standard.

⭐ THE STANDARD (Sayed, standing): camera projection AND a metric BEV inset
TOGETHER, with a text overlay of the quantities. This renderer adds the thing
that makes the sample judgeable by eye: OUR OWN `obstacle.offline` ground-truth
cuboids are drawn in the SAME frames, in a different colour, so the teacher can
be read against ground truth without trusting a summary number.

⛔ PERCEPTION ONLY. No trajectory, plan or action of Qwen-Drive is read, drawn or
distilled anywhere in this package -- it is last-place closed-loop and its
planner weights were never downloaded.

WHY NOT `qwen_drive_perception.visualize.render_frame`: it requires GROUND-TRUTH
occupancy and a GROUND-TRUTH map raster (`frame.gt["occ"]`, `frame.gt["map"]`).
PhysicalAI-AV publishes neither -- its card says verbatim that open maps data is
not included, and its only 3D label feature is `obstacle.offline` (dynamic agents
only). Those two panels would be empty by construction, so this renderer shows
the PREDICTED occupancy and map with no GT counterpart and says so on the figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

DET_CLASS_NAMES = ("vehicle", "czone_sign", "bicycle", "generic_object",
                   "pedestrian", "traffic_cone", "barrier")
DET_BOX_COLORS = {
    "vehicle": "#ff8a00", "czone_sign": "#9b51e0", "bicycle": "#20bf6b",
    "generic_object": "#8d6e63", "pedestrian": "#eb3b5a",
    "traffic_cone": "#ffd60a", "barrier": "#7f8c8d",
}
GT_COLOR = "#00e5ff"          # our obstacle.offline cuboids -- deliberately cold
BEV_RADIUS = 50.0
# Ring cells of a 3x3 grid keyed by the canonical azimuth each view covers.
RING_SLOTS = {"CAM_F0": (0, 1), "CAM_L0": (0, 0), "CAM_R0": (0, 2),
              "CAM_L2": (2, 0), "CAM_B0": (2, 1), "CAM_R2": (2, 2)}


def build_lidar2img(K, R_cam_to_lidar, t_cam_in_lidar):
    """Qwen-Drive's own composition (`qwen_drive_perception.geometry`), so the
    drawn boxes sit exactly where the model's projection put them."""
    lidar2cam_r = np.linalg.inv(np.asarray(R_cam_to_lidar, np.float64))
    lidar2cam_t = np.asarray(t_cam_in_lidar, np.float64) @ lidar2cam_r.T
    rt = np.eye(4)
    rt[:3, :3] = lidar2cam_r.T
    rt[3, :3] = -lidar2cam_t
    viewpad = np.eye(4)
    viewpad[:3, :3] = np.asarray(K, np.float64)
    return viewpad @ rt.T


def box_corners(boxes: np.ndarray) -> np.ndarray:
    """(N,9) [x,y,z,w,l,h,yaw,...] with z at the BOTTOM -> (N,8,3) corners.

    mmdet3d LiDARInstance3DBoxes order: w is the extent along heading (local +x),
    l the lateral extent (local +y); bottom face CCW from (+x,+y) then the top.
    """
    boxes = np.asarray(boxes, np.float64)
    if not len(boxes):
        return np.zeros((0, 8, 3))
    centers = boxes[:, None, :3]
    w, l, h = boxes[:, 3:4], boxes[:, 4:5], boxes[:, 5:6]
    yaw = boxes[:, 6]
    cos, sin = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    local = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], np.float64) * \
        np.concatenate([w / 2, l / 2], axis=-1)[:, None, :]
    lx = local[..., 0] * cos - local[..., 1] * sin
    ly = local[..., 0] * sin + local[..., 1] * cos
    xy = np.stack([lx, ly], axis=-1)
    zb = np.zeros((len(boxes), 4))
    zt = np.repeat(h, 4, axis=1)
    corners = np.concatenate([
        np.concatenate([xy, zb[..., None]], axis=-1),
        np.concatenate([xy, zt[..., None]], axis=-1)], axis=1)
    return centers + corners


def project(corners, lidar2img, width, height):
    """(N,8,3) -> (N,8,2) pixels + validity (all corners in front, any on screen)."""
    if not len(corners):
        return np.zeros((0, 8, 2)), np.zeros((0,), bool)
    ch = np.concatenate([corners, np.ones(corners.shape[:2] + (1,))], axis=-1)
    p = np.einsum("ij,nkj->nki", np.asarray(lidar2img, np.float64), ch)
    d = p[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = p[..., :2] / d[..., None]
    ok = (d > 1e-3).all(axis=1)
    on = ((uv[..., 0] >= 0) & (uv[..., 0] < width) &
          (uv[..., 1] >= 0) & (uv[..., 1] < height)).any(axis=1)
    return uv, ok & on


def draw_cube(ax, uv, color, lw):
    for i in range(4):
        for a, b in ((i, (i + 1) % 4), (i + 4, (i + 1) % 4 + 4), (i, i + 4)):
            ax.add_line(Line2D([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]],
                               color=color, linewidth=lw, alpha=0.95))


def draw_bev(ax, pred, pred_lab, gt):
    ax.set_xlim(-BEV_RADIUS, BEV_RADIUS)
    ax.set_ylim(-BEV_RADIUS, BEV_RADIUS)
    ax.set_aspect("equal")
    ax.set_facecolor("#0d1117")
    for r in (10, 20, 30, 40, 50):
        ax.add_patch(patches.Circle((0, 0), r, fill=False, edgecolor="#2c3440",
                                    linewidth=0.7, zorder=1))
        ax.text(0.7, r - 1.6, f"{r}m", color="#54606e", fontsize=6, zorder=2)

    def blocks(boxes, color_of, lw, z):
        c = box_corners(boxes)
        for i in range(len(boxes)):
            foot = c[i][:4, :2]
            xy = np.stack([-foot[:, 1], foot[:, 0]], axis=1)   # screen: right=-y, up=+x
            col = color_of(i)
            ax.add_patch(patches.Polygon(xy, closed=True, fill=False,
                                         edgecolor=col, linewidth=lw, zorder=z))
            front = c[i][:2, :2].mean(axis=0)
            ctr = foot.mean(axis=0)
            ax.plot([-ctr[1], -front[1]], [ctr[0], front[0]], color=col,
                    linewidth=lw, zorder=z)

    if len(gt):
        blocks(gt, lambda i: GT_COLOR, 1.9, 4)
    if len(pred):
        blocks(pred, lambda i: DET_BOX_COLORS[DET_CLASS_NAMES[int(pred_lab[i])]], 1.3, 5)
    ax.add_patch(patches.FancyArrow(0, 0, 0, 5.0, width=0.9, head_width=2.6,
                                    head_length=3.0, fc="#f0f6fc", ec="none", zorder=6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("metric BEV (ego frame, +x fwd / +y left)  -  cold = OUR GT, warm = Qwen-Drive",
                 fontsize=8, color="#c9d1d9", pad=4)


DET_RANGE_M = 51.2          # `perception/config.json: det_pc_range` half-extent


def match_stats(pred, pred_lab, scores, gt, gt_lab, radius=2.0):
    """Greedy centre-distance matching, for an HONEST first read only.

    ⛔ NOT a benchmark. No tuning is done on this sample and no rate from it is
    quotable: n is tiny and the matcher is a 2 m centre-distance rule, not the
    programme's estimator. Reported as raw counts.

    ⛔ SCOPE, and it is load-bearing: our `obstacle.offline` cuboids run out to
    MEASURED 199 m, while the model's `det_pc_range` is +/-51.2 m. Counting a
    cuboid at 150 m as a "miss" would score the teacher on a region it is
    architecturally incapable of predicting -- the same scope error as reading
    `df` on a pod. GT is therefore restricted to the model's OWN declared range
    before anything is counted, and the discarded count is reported alongside.
    """
    out = {"matched": 0, "missed": 0, "false_pos": 0, "range_err": [],
           "class_ok": 0, "class_wrong": 0, "gt_out_of_range": 0}
    if len(gt):
        r = np.linalg.norm(gt[:, :2], axis=1)
        keep = r <= DET_RANGE_M
        out["gt_out_of_range"] = int((~keep).sum())
        gt, gt_lab = gt[keep], gt_lab[keep]
    if not len(gt) and not len(pred):
        return out
    used = set()
    order = np.argsort(-scores) if len(scores) else []
    for pi in order:
        if not len(gt):
            out["false_pos"] += 1
            continue
        d = np.linalg.norm(gt[:, :2] - pred[pi, :2], axis=1)
        for gi in np.argsort(d):
            if gi in used:
                continue
            if d[gi] <= radius:
                used.add(int(gi))
                out["matched"] += 1
                rp = float(np.linalg.norm(pred[pi, :2]))
                rg = float(np.linalg.norm(gt[gi, :2]))
                out["range_err"].append(rp - rg)
                if int(pred_lab[pi]) == int(gt_lab[gi]):
                    out["class_ok"] += 1
                else:
                    out["class_wrong"] += 1
            else:
                out["false_pos"] += 1
            break
        else:
            out["false_pos"] += 1
    out["missed"] = len(gt) - len(used)
    return out


def render(frame_dir: Path, pred_path: Path, out_png: Path, thr: float = 0.30) -> dict:
    meta = json.loads((frame_dir / "meta.json").read_text(encoding="utf-8"))
    fj = json.loads((frame_dir / "frame.json").read_text(encoding="utf-8"))
    calib = np.load(frame_dir / "calib.npz")
    gtz = np.load(frame_dir / "gt.npz", allow_pickle=True)
    gt, gt_lab = gtz["boxes"], gtz["labels"]
    res = np.load(pred_path)
    keep = res["scores"] >= thr
    pred, pred_lab, scores = res["boxes"][keep], res["labels"][keep], res["scores"][keep]

    cams = fj["cam_order"]
    fig = plt.figure(figsize=(19.5, 13.2), dpi=125)
    fig.patch.set_facecolor("#161b22")
    gs = fig.add_gridspec(3, 3, left=0.008, right=0.992, top=0.955, bottom=0.175,
                          wspace=0.03, hspace=0.10,
                          width_ratios=[1.22, 1.0, 1.22], height_ratios=[1, 1.28, 1])

    for i, sid in enumerate(cams):
        r, c = RING_SLOTS[sid]
        ax = fig.add_subplot(gs[r, c])
        img = np.asarray(Image.open(frame_dir / "images" / f"{sid}.jpg"))
        ax.imshow(img)
        l2i = build_lidar2img(calib["cam_intrinsic"][i],
                              calib["sensor2lidar_rotation"][i],
                              calib["sensor2lidar_translation"][i])
        uvg, okg = project(box_corners(gt), l2i, img.shape[1], img.shape[0])
        for k in np.where(okg)[0]:
            draw_cube(ax, uvg[k], GT_COLOR, 1.5)
        uvp, okp = project(box_corners(pred), l2i, img.shape[1], img.shape[0])
        for k in np.where(okp)[0]:
            draw_cube(ax, uvp[k], DET_BOX_COLORS[DET_CLASS_NAMES[int(pred_lab[k])]], 1.1)
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        cm = meta["cameras"][sid]
        ax.set_title(f"{meta['view_tag'][sid]}  {meta['feature_of'][sid]}\n"
                     f"rectified {cm['hfov_deg']:.0f}x{cm['vfov_deg']:.0f} deg  "
                     f"f={cm['f_px']:.0f}px  observed {cm['observed_frac']*100:.1f}%  "
                     f"dt={cm['dt_to_ref_ms']:+.0f}ms",
                     fontsize=7.2, color="#c9d1d9", pad=3)
        for sp in ax.spines.values():
            sp.set_color("#30363d")

    draw_bev(fig.add_subplot(gs[1, 1]), pred, pred_lab, gt)

    # ---- occupancy + map (PREDICTION ONLY -- PhysicalAI publishes no GT for either)
    axo = fig.add_subplot(gs[1, 0])
    # Top-down view of the occupancy volume: the highest-priority NON-empty class
    # in each pillar. Cast first -- `occ` is uint8 and the -1 sentinel underflows.
    occ = res["occ"].astype(np.int16)
    top = np.where(occ < 9, occ, -1).max(axis=2) if occ.ndim == 3 else occ
    axo.imshow(np.flipud(top.T), cmap="turbo", vmin=-1, vmax=9, interpolation="nearest")
    axo.set_title("occupancy PREDICTION (no GT exists in PhysicalAI)",
                  fontsize=7.5, color="#c9d1d9", pad=3)
    axo.set_xticks([]); axo.set_yticks([])

    axm = fig.add_subplot(gs[1, 2])
    axm.imshow(np.flipud(res["map"].T), cmap="viridis", interpolation="nearest")
    axm.set_title("BEV map PREDICTION (PhysicalAI ships NO map data at all)",
                  fontsize=7.5, color="#c9d1d9", pad=3)
    axm.set_xticks([]); axm.set_yticks([])

    st = match_stats(pred, pred_lab, scores, gt, gt_lab)
    rng = np.asarray(st["range_err"]) if st["range_err"] else np.zeros(0)
    cls_counts = {}
    for li in pred_lab:
        n = DET_CLASS_NAMES[int(li)]
        cls_counts[n] = cls_counts.get(n, 0) + 1
    gt_counts = {}
    for li in gt_lab:
        n = DET_CLASS_NAMES[int(li)]
        gt_counts[n] = gt_counts.get(n, 0) + 1

    lines = [
        f"TOKEN {meta['token']}   clip {meta['clip_id']}   tier {meta.get('tier','?')}"
        f"   t_ref {meta['t_ref_us']/1e6:.2f}s   score threshold {thr:.2f}",
        f"Qwen-Drive-1.0-4B PERCEPTION HEAD ONLY (Apache-2.0).  "
        f"PLANNER WEIGHTS NOT DOWNLOADED, NOT RUN, NOT DISTILLED -- it is "
        f"2.25x worse than Alpamayo-R1 closed-loop; teacher for boxes/occupancy/map only.",
        f"predicted {len(pred)} boxes >= {thr:.2f}   " +
        "  ".join(f"{k}:{v}" for k, v in sorted(cls_counts.items())),
        f"OUR GT     {len(gt)} cuboids (obstacle.offline, rig frame), of which "
        f"{len(gt)-st['gt_out_of_range']} inside the model's +/-51.2 m det range   " +
        "  ".join(f"{k}:{v}" for k, v in sorted(gt_counts.items())),
        f"IN-RANGE counts on THIS frame: matched {st['matched']}  missed {st['missed']}  "
        f"false-pos {st['false_pos']}  class-agree {st['class_ok']}/{st['matched']}"
        + (f"   range err mean {rng.mean():+.2f} m, |err| median {np.median(np.abs(rng)):.2f} m"
           if len(rng) else "   range err n/a")
        + f"   ({st['gt_out_of_range']} GT beyond 51.2 m EXCLUDED -- outside the head's range)",
        "n is tiny and the matcher is a 2 m centre-distance rule -- these are COUNTS, not rates, "
        "and nothing here was tuned on this sample.",
    ]
    fig.text(0.008, 0.155, "\n".join(lines), fontsize=9.0, color="#e6edf3",
             va="top", ha="left", family="monospace", linespacing=1.55)
    present = sorted({DET_CLASS_NAMES[int(i)] for i in pred_lab})
    handles = [Line2D([], [], color=GT_COLOR, lw=2.6,
                      label="OUR obstacle.offline GT (rig frame)")]
    handles += [Line2D([], [], color=DET_BOX_COLORS[n], lw=2.0,
                       label=f"Qwen-Drive {n}") for n in present]
    fig.legend(handles=handles, loc="lower right", ncol=max(1, len(handles) // 2),
               facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3",
               fontsize=8.5, bbox_to_anchor=(0.992, 0.012))
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    st["range_err"] = [float(x) for x in st["range_err"]]
    return {"token": meta["token"], "clip": meta["clip_id"], "tier": meta.get("tier"),
            "n_pred": int(len(pred)), "n_gt": int(len(gt)), **st}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--score-threshold", type=float, default=0.30)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(p for p in args.frames.iterdir() if p.is_dir()):
        pred = args.predictions / f"{d.name}.npz"
        if not pred.exists():
            print(f"skip {d.name}: no prediction")
            continue
        r = render(d, pred, args.out / f"{d.name}.png", args.score_threshold)
        rows.append(r)
        print(f"{r['token']}: pred {r['n_pred']} gt {r['n_gt']} "
              f"matched {r['matched']} missed {r['missed']} fp {r['false_pos']}")
    (args.out / "SCORECARD.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
