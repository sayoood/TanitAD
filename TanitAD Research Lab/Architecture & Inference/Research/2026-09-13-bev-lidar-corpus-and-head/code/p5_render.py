#!/usr/bin/env python3
"""P5 - predicted vs LiDAR BEV on TEST frames, with the camera image the head's trunk saw.

Viz standard (PI): camera projection AND metric BEV together, with a text overlay.
  left   the CURRENT frame of the D-015 stack (raw v2ep frame i = j + 2), 256x640 cylindrical,
         with LiDAR-occupied cell centres (green) and PREDICTED-occupied cell centres
         (magenta) projected at z = 0.5 m through the per-clip rig camera
  middle LiDAR BEV ground truth, polar48, 4-state (loader encoding)
  right  predicted occupancy probability, polar48, occluded/out-of-field cells hatched

⛔ Frames are chosen by RULE, never by score: the test clips in sha12 order, raw frame 100
   (the clip midpoint); a frame with no label falls back to the nearest valid one.
⛔ ORIENTATION IS ASSERTED ON THE PANEL: forward is UP, LEFT is LEFT, and a labelled control
   marker sits at rig (x = 8 m, y = +10 m) = 10 m LEFT. If that marker is not on the left
   half, the panel is mirrored (the 09-11 package's first BEV render was).
⛔ Clip ids never leave this process; the figure carries sha12 only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bev_gt_loader as L  # noqa: E402
import p4_bev_head as H  # noqa: E402
import p2_build_corpus as B  # noqa: E402

N_RNG, N_AZ, CELL_R, CELL_DEG, HALF = 48, 40, 1.25, 3.0, 60.0


def rig_to_plot(x, y):
    """Plot coordinates: horizontal = -y (LEFT is LEFT), vertical = x (forward is UP)."""
    return -np.asarray(y), np.asarray(x)


def cell_centre_rig(r_bin, col):
    r = (r_bin + 0.5) * CELL_R
    az = math.radians(HALF - (col + 0.5) * CELL_DEG)       # col 0 = +60 deg = LEFT
    return r * math.cos(az), r * math.sin(az)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="main")
    ap.add_argument("--n-frames", type=int, default=2)
    ap.add_argument("--raw-frame", type=int, default=100)
    ap.add_argument("--runs", default=str(H.WORK / "runs"))
    ap.add_argument("--panel", default=str(HERE.parent / "raw" / "p4_panel.json"))
    ap.add_argument("--out-dir", default=str(HERE.parent / "media"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
    from matplotlib.collections import PatchCollection
    import torch
    import torchvision.io as tvio
    from sklearn.metrics import average_precision_score
    from tanitad.data.calib import CanonicalFrame
    from tanitad.data.physicalai import FrontWideExtrinsics
    from tanitad.data.rig_projection import RigCamera

    # orientation self-check of the plot transform (literal)
    px, py = rig_to_plot(8.0, 10.0)
    assert float(px) == -10.0 and float(py) == 8.0, "plot transform is not LEFT-is-left"

    D = H.load_panel_data()
    rows = np.load(Path(args.runs) / args.arm / "rows.npz")
    te = rows["test"]
    probs = np.load(Path(args.runs) / args.arm / "test_probs.npy").astype(np.float32)
    panel = json.loads(Path(args.panel).read_text(encoding="utf-8"))
    tau = panel["arms"][args.arm]["tau_from_val"]

    # sha12 -> clip id, locally only
    join = (HERE.parents[3] / "Benchmarks & Evals" / "Research" / "2026-09-06-b1-agent-join"
            / "raw" / "b1eval_agents.jsonl.xz")
    id_of = {}
    with lzma.open(join, "rt", encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)["clip_id"]
            id_of.setdefault(hashlib.sha256(c.encode()).hexdigest()[:12], c)

    test_clips = sorted({D["sha"][c] for c in D["clip_of_row"][te]})
    written = []
    for s12 in test_clips[: args.n_frames]:
        ci = D["sha"].index(s12)
        crow = te[D["clip_of_row"][te] == ci]
        fr = D["raw_frame"][crow]
        k = int(np.argmin(np.abs(fr - args.raw_frame)))
        row = int(crow[k])
        i_raw = int(fr[k])
        p = probs[np.nonzero(te == row)[0][0]]
        occ, mB = D["occ"][row], D["mB"][row]
        cid = id_of[s12]
        man = {json.loads(l)["clip_sha12"]: json.loads(l)
               for l in (H.GT_DIR / "manifest.jsonl").read_text(encoding="utf-8").splitlines()}
        clip = L.load_clip(H.GT_DIR / man[s12]["artifact"], grid="polar48", rule="B")
        state = clip.state()[i_raw]
        t_img = int(clip.t_img_us[i_raw])

        v2 = torch.load(Path(B.V2EP_DIR) / f"{cid}.v2ep.pt", map_location="cpu", weights_only=False)
        lens = v2["jpeg_len"].to(torch.int64)
        offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens, 0)])
        img = tvio.decode_png(v2["jpeg_buf"][int(offs[i_raw]):int(offs[i_raw + 1])],
                              mode=tvio.ImageReadMode.RGB).permute(1, 2, 0).numpy()
        frame = CanonicalFrame.from_dict(v2["frame"])
        cam = RigCamera.from_extrinsics(FrontWideExtrinsics(**B.extrinsics(cid, "camera_front_wide_120fov")), frame)

        y = occ[mB]
        sc = p[mB]
        f_ap = float(average_precision_score(y, sc)) if y.any() and (~y).any() else float("nan")
        pred = (p >= tau)
        tp = int((pred & occ & mB).sum()); fp = int((pred & ~occ & mB).sum()); fn = int((~pred & occ & mB).sum())
        f_iou = tp / max(tp + fp + fn, 1)

        fig = plt.figure(figsize=(20.0, 6.2), dpi=110)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.75, 1.0, 1.0], wspace=0.16,
                              left=0.01, right=0.97, top=0.93, bottom=0.30)
        axc = fig.add_subplot(gs[0, 0]); axg = fig.add_subplot(gs[0, 1]); axp = fig.add_subplot(gs[0, 2])

        axc.imshow(img)
        for (grid, color, lab) in ((occ & mB, "#39ff14", "LiDAR occupied (scored)"),
                                   (pred & clip.in_field, "#ff2d95", f"predicted p>={tau:.2f}")):
            rr, cc = np.nonzero(grid)
            if rr.size == 0:
                continue
            pts = np.array([[*cell_centre_rig(a, b), 0.5] for a, b in zip(rr, cc)])
            col, row_px, val = cam.project(torch.as_tensor(pts, dtype=torch.float64))
            val = val.numpy().astype(bool)
            axc.scatter(col.numpy()[val], row_px.numpy()[val], s=10 if color == "#39ff14" else 4,
                        facecolors="none" if color == "#39ff14" else color, edgecolors=color,
                        linewidths=0.8, label=lab, alpha=0.9)
        axc.set_xlim(0, frame.width); axc.set_ylim(frame.height, 0)
        axc.set_xticks([]); axc.set_yticks([])
        axc.legend(loc="lower left", fontsize=7, framealpha=0.6)
        axc.set_title("CURRENT frame of the 3-frame stack (raw v2ep frame i = j+2), 256x640 CYLINDRICAL, "
                      "cell centres projected at z = 0.5 m", fontsize=8.5)

        colors = {L.OUT_OF_FIELD: "#3a3a3a", L.OBSERVED_EMPTY: "#1f5f3a",
                  L.OCCUPIED: "#ffd21a", L.OCCLUDED: "#0b0b12"}
        for ax, title, mode in ((axg, "LiDAR BEV GT (label only), polar48 1.25 m x 3 deg, rule B", "gt"),
                                (axp, f"PREDICTED p(occupied), arm `{args.arm}`, frozen trunk", "pred")):
            patches, fcs = [], []
            cmap = plt.get_cmap("magma")
            for a in range(N_RNG):
                for b in range(N_AZ):
                    az_hi = HALF - b * CELL_DEG
                    az_lo = az_hi - CELL_DEG
                    w = Wedge((0, 0), (a + 1) * CELL_R, az_lo + 90.0, az_hi + 90.0, width=CELL_R)
                    patches.append(w)
                    if mode == "gt":
                        fcs.append(colors[int(state[a, b])])
                    else:
                        if not mB[a, b]:
                            fcs.append("#2a2a2a" if state[a, b] == L.OUT_OF_FIELD else "#101018")
                        else:
                            fcs.append(cmap(float(p[a, b])))
            ax.add_collection(PatchCollection(patches, facecolor=fcs, edgecolor="none"))
            for rr in (15, 30, 45, 60):
                ax.add_patch(Wedge((0, 0), rr, 30, 150, width=0.001, fill=False, ec="#888888", lw=0.4))
            cx, cy = rig_to_plot(8.0, 10.0)
            ax.plot([cx], [cy], marker="*", ms=12, color="white", mec="black", mew=0.7)
            ax.annotate("control: rig (x=8, y=+10) = 10 m LEFT", (float(cx), float(cy)),
                        textcoords="offset points", xytext=(6, 4), fontsize=6.5, color="white")
            ax.plot([0], [0], marker="^", ms=9, color="#00e5ff")
            ax.set_xlim(-55, 55); ax.set_ylim(-2, 62); ax.set_aspect("equal")
            ax.set_facecolor("#000000")
            ax.set_xlabel("<- LEFT   lateral [m]   RIGHT ->", fontsize=8)
            ax.set_ylabel("forward x [m]", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_title(title, fontsize=8.5)
        if True:
            sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, 1))
            cb = fig.colorbar(sm, ax=axp, fraction=0.03, pad=0.01)
            cb.set_label("p(occupied)", fontsize=7); cb.ax.tick_params(labelsize=6)

        arm_row = panel["arms"][args.arm]
        txt = (f"clip sha12 {s12}   raw v2ep frame {i_raw} (stacked row j={i_raw - 2})   t_img = {t_img / 1e6:+.3f} s   "
               f"TEST split (clip-disjoint; frame chosen by RULE: raw frame nearest {args.raw_frame}, not by score)\n"
               f"THIS FRAME: AP {f_ap:.3f}  IoU@tau {f_iou:.3f}  (tau {tau:.2f} from VAL)  prevalence {float(y.mean()):.3f}  "
               f"scored cells {int(mB.sum())}/1920   |   PANEL arm `{args.arm}`: test AP {arm_row['ap_all_B']:.4f} "
               f"[{arm_row['ap_all_B_ci95'][0]:.4f}, {arm_row['ap_all_B_ci95'][1]:.4f}]  IoU 0-30 m {arm_row['iou_0_30m_B']:.4f}\n"
               f"GT state: yellow OCCUPIED, green OBSERVED-EMPTY, black OCCLUDED (beyond first hit, no return), grey OUT-OF-FIELD "
               f"(cam_vis<128). Scored = valid & in-field & observed (rule B). Frame RIG: +x fwd, +y LEFT; origin rear axle; "
               f"z-band [0.30, 3.00] m + spread >= 0.30 m\n"
               f"INPUT: refcv5-v2 trunk (frozen, 90,458,632 params) tokens [704x8x20] of the 9-channel stack -> transformer head "
               f"(2,284,993 params). LiDAR is a LABEL ONLY - inference is VISION-ONLY.")
        fig.text(0.012, 0.215, txt, va="top", ha="left", fontsize=7.6, family="monospace")
        out = Path(args.out_dir) / f"bevhead_{args.arm}_{s12}_f{i_raw:03d}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(out.name)
        print(f"[p5] {out.name} frame AP {f_ap:.3f} IoU {f_iou:.3f}", flush=True)
    print(json.dumps({"written": written}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
