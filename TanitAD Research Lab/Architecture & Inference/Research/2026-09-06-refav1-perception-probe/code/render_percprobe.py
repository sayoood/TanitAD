"""P2 — VISUALISE what the frozen trunk decodes about the environment.

The PI asked for this directly: "... and visualize its results?"

STANDING VIZ STANDARD (Sayed): camera projection + metric BEV inset TOGETHER,
plus a text overlay of the numbers. Both panes draw the DECODED lead against
the GROUND-TRUTH track, and the RAW-PIXEL FLOOR's decode beside it -- because
the floor is the control that makes the trunk's number mean anything.

⛔ TRAPS HONOURED
  * projection is CYLINDRICAL (f_ref 305.577, column LINEAR IN AZIMUTH, 120 deg).
    The pinhole formula gives 92.6 deg and looks entirely plausible. We call
    tanitad.data.rig_projection, which keys the formula on frame.projection and
    is the ONE implementation -- we never re-derive it here.
  * the v2ep buffer is named `jpeg_buf` while its `codec` field says `png`.
    We READ THE CODEC FIELD.
  * camera height is a PER-CLIP quantity (554 distinct values over 2,400 clips);
    RigCamera.from_extrinsics per clip is the only admissible constructor.
  * ⛔ ASSERT ON CONTENT, never on exit code: the last render was stretched
    3.9 % on every frame while every exit code read 0. We decode the written
    mp4 back and assert frame count, geometry and non-zero luminance.

Only SCORED-SPLIT clips are drawn: the head never saw them.
"""
import argparse
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/nvidia/refav1_lon/code/stack")

BEV_W, BEV_H = 420, 512          # BEV inset pixels
BEV_XMAX, BEV_YHALF = 40.0, 10.0  # metres: forward range, lateral half-width
UP = 2                            # camera upscale
LANE_HALF = 1.75                  # the lead corridor used by the probe

C_GT = (80, 230, 80)         # ground truth      (BGR)
C_FIELD = (230, 80, 230)     # refav1 trunk decode
C_PIX = (220, 200, 60)       # raw-pixel floor
C_TXT = (245, 245, 245)


SENSOR = "camera_front_wide_120fov"


def load_extrinsics(clip, chunk, extr_dir):
    """Per-CLIP front-wide mount pose. ⛔ Never a constant: 554 distinct camera
    heights over the 2,400 parity clips, and the mount PITCH is what puts the
    horizon on the right row."""
    import pandas as pd
    from tanitad.data.physicalai import FrontWideExtrinsics
    p = os.path.join(extr_dir, "sensor_extrinsics.chunk_%04d.parquet" % chunk)
    df = pd.read_parquet(p).reset_index()
    r = df[(df.clip_id == clip) & (df.sensor_name == SENSOR)]
    if len(r) != 1:
        raise RuntimeError("expected exactly 1 %s row for %s in chunk %d, got %d"
                           % (SENSOR, clip, chunk, len(r)))
    r = r.iloc[0]
    return FrontWideExtrinsics(qx=float(r.qx), qy=float(r.qy), qz=float(r.qz),
                               qw=float(r.qw), x=float(r.x), y=float(r.y),
                               z=float(r.z))


def bev_xy_to_px(cx, cy):
    """rig (+x fwd, +y LEFT) -> BEV pixels (ego at bottom centre, x up)."""
    px = int(round(BEV_W / 2.0 - cy / BEV_YHALF * (BEV_W / 2.0)))
    py = int(round(BEV_H - 1 - cx / BEV_XMAX * (BEV_H - 1)))
    return px, py


def draw_bev(cv2, gt, pf, pp):
    img = np.full((BEV_H, BEV_W, 3), 22, np.uint8)
    for m in range(10, int(BEV_XMAX) + 1, 10):
        _, y = bev_xy_to_px(m, 0.0)
        cv2.line(img, (0, y), (BEV_W, y), (55, 55, 55), 1)
        cv2.putText(img, "%dm" % m, (6, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (120, 120, 120), 1, cv2.LINE_AA)
    for s in (-1, 1):
        x0, y0 = bev_xy_to_px(0.0, s * LANE_HALF)
        x1, y1 = bev_xy_to_px(BEV_XMAX, s * LANE_HALF)
        cv2.line(img, (x0, y0), (x1, y1), (70, 70, 70), 1, cv2.LINE_AA)
    ex, ey = bev_xy_to_px(0.0, 0.0)
    cv2.rectangle(img, (ex - 9, ey - 18), (ex + 9, ey), (200, 200, 200), 1)
    for val, col, r in ((gt, C_GT, 8), (pp, C_PIX, 5), (pf, C_FIELD, 7)):
        if val is None:
            continue
        cx, cy = val
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        px, py = bev_xy_to_px(float(cx), float(cy))
        if 0 <= px < BEV_W and 0 <= py < BEV_H:
            if col is C_GT:
                cv2.circle(img, (px, py), r, col, 2, cv2.LINE_AA)
            else:
                cv2.drawMarker(img, (px, py), col, cv2.MARKER_CROSS, r * 2, 2)
    cv2.putText(img, "BEV (metric, ego frame)", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return img


def main():
    import cv2
    import torch
    from tanitad.data.calib import CanonicalFrame
    from tanitad.data.rig_projection import RigCamera

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="/home/nvidia/percprobe/raw/decode_pred.npz")
    ap.add_argument("--out", default="/home/nvidia/percprobe/raw/percprobe_lead.mp4")
    ap.add_argument("--n-clips", type=int, default=4)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--ep-dirs", default="/home/nvidia/data/physicalai-b1-w120-256x640cyl,"
                                         "/home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl")
    ap.add_argument("--r0", default="/home/nvidia/data/physicalai-b1/r0/r0_selection.parquet")
    ap.add_argument("--extr", default="/home/nvidia/data/physicalai-b1/calibration/sensor_extrinsics")
    args = ap.parse_args()

    z = np.load(args.pred, allow_pickle=True)
    clip = z["clip"].astype(str)
    fidx = z["frame_idx"]
    in_sc = z["in_score_split"]
    lab = z["labelled_lead"]
    gt_g, gt_l = z["gap_true"], z["lat_true"]
    pf_g, pf_l = z["gap_pred_field"], z["lat_pred_field"]
    pp_g, pp_l = z["gap_pred_pix"], z["lat_pred_pix"]

    # pick the scored clips with the most labelled lead frames
    cand = {}
    for c in np.unique(clip[in_sc]):
        m = (clip == c) & in_sc & lab
        cand[c] = int(m.sum())
    picks = [c for c, n in sorted(cand.items(), key=lambda kv: -kv[1])][:args.n_clips]
    print("clips drawn:", [(c[:8], cand[c]) for c in picks], flush=True)

    ep_index = {}
    import glob
    for d in args.ep_dirs.split(","):
        for p in glob.glob(d + "/*.v2ep.pt"):
            ep_index.setdefault(os.path.basename(p)[:-8], p)

    CW, CH = 640 * UP, 256 * UP
    W, H = CW + BEV_W, max(CH, BEV_H) + 74
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter refused to open %s" % args.out)

    n_written, lum_sum, n_proj_ok = 0, 0.0, 0
    from PIL import Image
    import pandas as pd
    sel = pd.read_parquet(args.r0)
    chunk_of = dict(zip(sel.clip_id, sel.chunk))

    for c in picks:
        ep = torch.load(ep_index[c], map_location="cpu", weights_only=False)
        codec = ep.get("codec")
        if codec != "png":
            raise RuntimeError("codec is %r -- refusing (buffer NAME is a lie "
                               "by design; we read the FIELD)" % codec)
        fr = ep["frame"]
        frame = CanonicalFrame(height=int(fr["height"]), width=int(fr["width"]),
                               f_ref=float(fr["f_ref"]),
                               projection=str(fr["projection"]))
        # per-CLIP extrinsics (554 distinct heights over 2,400 clips: never a constant)
        extr = load_extrinsics(c, int(chunk_of[c]), args.extr)
        cam = RigCamera.from_extrinsics(extr, frame)
        print("  %s cam_height=%.4f m  pitch=%+.3f deg  hfov=%.1f deg"
              % (c[:8], extr.z, np.degrees(extr.optical_axis_pitch_rad()),
                 np.degrees(2.0 * (frame.width / 2.0) / frame.f_ref)), flush=True)

        buf = ep["jpeg_buf"].numpy().tobytes()
        lens = ep["jpeg_len"].tolist()
        offs = np.concatenate([[0], np.cumsum(lens)])

        rows = np.where((clip == c) & in_sc)[0]
        for r in rows:
            i = int(fidx[r])
            raw = buf[offs[i]:offs[i] + lens[i]]
            im = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))[:, :, ::-1]
            im = cv2.resize(im, (CW, CH), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((H, W, 3), 18, np.uint8)
            canvas[:CH, :CW] = im

            def proj(cx, cy):
                if not (np.isfinite(cx) and np.isfinite(cy)):
                    return None
                p = torch.tensor([[float(cx), float(cy), 0.0]], dtype=torch.float64)
                col, row_, valid = cam.project(p)
                if not bool(valid[0]):
                    return None
                return int(round(float(col[0]) * UP)), int(round(float(row_[0]) * UP))

            has_gt = bool(lab[r])
            gtxy = (gt_g[r], gt_l[r]) if has_gt else None
            fxy = (pf_g[r], pf_l[r])
            pxy = (pp_g[r], pp_l[r])

            for val, col_, mk in ((pxy, C_PIX, 14), (fxy, C_FIELD, 20),
                                  (gtxy, C_GT, 24)):
                if val is None:
                    continue
                q = proj(val[0], val[1])
                if q is None:
                    continue
                n_proj_ok += 1
                if col_ is C_GT:
                    cv2.circle(canvas, q, 13, col_, 3, cv2.LINE_AA)
                else:
                    cv2.drawMarker(canvas, q, col_, cv2.MARKER_TILTED_CROSS, mk, 3)

            bev = draw_bev(cv2, gtxy, fxy, pxy)
            canvas[:BEV_H, CW:CW + BEV_W] = bev

            y0 = max(CH, BEV_H)
            f = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, "%s  frame %d   [SCORED SPLIT - head never saw "
                                "this clip]" % (c[:8], i), (12, y0 + 20), f, 0.52,
                        C_TXT, 1, cv2.LINE_AA)
            if has_gt:
                t = ("GT lead  gap %5.1f m  lat %+5.2f m   |   "
                     "refav1 TRUNK  gap %5.1f m (err %+5.1f)   |   "
                     "pixel FLOOR  gap %5.1f m (err %+5.1f)"
                     % (gt_g[r], gt_l[r], pf_g[r], pf_g[r] - gt_g[r],
                        pp_g[r], pp_g[r] - gt_g[r]))
            else:
                t = ("no labelled lead in corridor (|cy| <= %.2f m, cx <= 30 m)"
                     "   |   refav1 TRUNK gap %5.1f m   |   pixel FLOOR %5.1f m"
                     % (LANE_HALF, pf_g[r], pp_g[r]))
            cv2.putText(canvas, t, (12, y0 + 44), f, 0.46, C_TXT, 1, cv2.LINE_AA)
            cv2.putText(canvas, "GT", (12, y0 + 66), f, 0.46, C_GT, 2, cv2.LINE_AA)
            cv2.putText(canvas, "refav1 frozen trunk", (54, y0 + 66), f, 0.46,
                        C_FIELD, 2, cv2.LINE_AA)
            cv2.putText(canvas, "raw-pixel floor", (268, y0 + 66), f, 0.46,
                        C_PIX, 2, cv2.LINE_AA)

            vw.write(canvas)
            n_written += 1
            lum_sum += float(canvas[:CH, :CW].mean())

    vw.release()

    # ---- CONTENT ASSERTIONS: decode the output BACK ----------------------
    cap = cv2.VideoCapture(args.out)
    got, lum2, k = 0, 0.0, 0
    while True:
        ok, fr_ = cap.read()
        if not ok:
            break
        got += 1
        if k % 20 == 0:
            lum2 += float(fr_.mean())
        k += 1
    gw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    res = {"path": os.path.abspath(args.out),
           "bytes": os.path.getsize(args.out),
           "frames_written": n_written, "frames_decoded_back": got,
           "geometry_written": [W, H], "geometry_decoded": [gw, gh],
           "mean_luminance_src": lum_sum / max(n_written, 1),
           "mean_luminance_decoded": lum2 / max(1, (k + 19) // 20),
           "projected_markers_in_frame": n_proj_ok,
           "clips": [str(c) for c in picks]}
    res["assert_frames_match"] = (got == n_written)
    res["assert_geometry_match"] = ([gw, gh] == [W, H])
    res["assert_nonzero_luminance"] = (res["mean_luminance_decoded"] > 1.0)
    res["assert_projection_nonvacuous"] = (n_proj_ok > 0)
    res["ALL_ASSERTIONS_PASS"] = all(
        res[k2] for k2 in ("assert_frames_match", "assert_geometry_match",
                           "assert_nonzero_luminance",
                           "assert_projection_nonvacuous"))
    print(json.dumps(res, indent=1))
    with open(args.out + ".assert.json", "w") as fh:
        json.dump(res, fh, indent=1)
    if not res["ALL_ASSERTIONS_PASS"]:
        raise SystemExit("CONTENT ASSERTIONS FAILED -- the exit code is not the check")


if __name__ == "__main__":
    main()
