"""Render the STANDARD IDM-from-video validation video for one held-out clip.

Left panel  : front-camera projection of GT (green) + IDM-reconstructed (orange)
              2 s ego trajectory, faithful flat-ground pinhole (f_eff 265.83,
              cx=cy=128, cam_h 1.5) = the taniteval standard for physicalai/comma,
              with a metric BEV inset + text HUD (decoded tactical maneuver, GT
              maneuver, strategic route, per-frame ADE, IDM vs GT speed).
Right panel : FULL-ROUTE BEV — the whole clip dead-reckoned from the IDM's
              per-frame (speed,yaw_rate) (orange) vs GT global path (green),
              with the live position dot and running endpoint error.

CPU-only (PIL + ffmpeg). Reads the npz written by idmval_run.py + the val-cache
frames. Nothing about the trajectory is re-computed here — it renders the saved
IDM outputs.
"""
from __future__ import annotations
import argparse, math, subprocess, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/root/v4eval/stack")

# ---- STANDARD constants (taniteval cam_overlay / flagship_overlay) ----
F_EFF, CX, CY, CAM_H, UP = 265.83, 128.0, 128.0, 1.5, 2
S = 256 * UP                                    # 512
COL_GT = (110, 235, 131)                        # green  (GT)
COL_PRED = (255, 122, 61)                       # orange (IDM reconstruction)
HUD_BG, HUD_FG, HUD_DIM = (10, 14, 19), (233, 237, 243), (150, 160, 175)
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
DT = 0.1
HORIZ_S = [0.5, 1.0, 1.5, 2.0]

try:
    from tanitad.refs.refb import MANEUVER_CLASSES, ROUTE_CLASSES
except Exception:
    MANEUVER_CLASSES = ["keep_lane", "accelerate", "decelerate", "stop",
                        "turn_left", "turn_right", "lane_left", "lane_right"]
    ROUTE_CLASSES = ["route_straight", "route_left", "route_right", "route_follow"]


def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


F_HUD, F_SUB, F_TINY = font(15), font(12), font(11)


def pretty_man(m):
    return MANEUVER_CLASSES[m].replace("_", " ") if 0 <= m < len(MANEUVER_CLASSES) else "n/a"


def pretty_route(r):
    return ROUTE_CLASSES[r].replace("route_", "") if 0 <= r < len(ROUTE_CLASSES) else "n/a"


def ego_future_path(poses, t, k=20):
    x0, y0, yaw = float(poses[t, 0]), float(poses[t, 1]), float(poses[t, 2])
    c, s = math.cos(yaw), math.sin(yaw)
    fut = poses[t+1:t+1+k, :2] - np.array([x0, y0])
    return np.stack([c*fut[:, 0] + s*fut[:, 1], -s*fut[:, 0] + c*fut[:, 1]], 1)


def project(pts):
    out = []
    for p in pts:
        X, Y = float(p[0]), float(p[1])
        if X < 1.2:
            continue
        out.append(((CX - F_EFF*Y/X)*UP, (CY + F_EFF*CAM_H/X)*UP))
    return out


def draw_bev_inset(im, gt, pred, xmax, ymax):
    d = ImageDraw.Draw(im, "RGBA")
    bw, bh = 152, 196
    x0, y0 = S - bw - 8, 92
    x1, y1 = x0 + bw, y0 + bh
    d.rectangle([x0, y0, x1, y1], fill=(8, 11, 15, 205), outline=(60, 70, 82), width=1)
    pad = 12
    cx = (x0 + x1)//2
    by = y1 - pad
    top = y0 + pad + 8

    def m2px(X, Y):
        return (cx - (Y/ymax)*((bw/2)-pad), by - (max(X, 0.)/xmax)*(by-top))
    step = 10 if xmax > 25 else 5
    r = step
    while r <= xmax + 0.1:
        _, py = m2px(r, 0)
        d.line([(x0+5, py), (x1-5, py)], fill=(38, 46, 56, 255))
        d.text((x0+6, py-11), f"{r}m", fill=(96, 106, 120), font=F_TINY)
        r += step
    d.line([(cx, top), (cx, by)], fill=(38, 46, 56, 255))
    g = [m2px(*p) for p in gt]
    if len(g) >= 2:
        d.line(g, fill=COL_GT, width=3)
    p = [m2px(*q) for q in pred]
    if len(p) >= 2:
        d.line(p, fill=COL_PRED, width=2)
    for q in pred:
        x, y = m2px(*q)
        d.ellipse([x-3, y-3, x+3, y+3], outline=COL_PRED, width=2)
    d.polygon([(cx-4, by), (cx+4, by), (cx, by-8)], fill=(232, 236, 242))
    d.text((x0+6, y0+3), "BEV top-down (m)", fill=HUD_DIM, font=F_TINY)


def draw_route_panel(W, H, route, cur, pred_g, ext, title):
    """Whole-clip route BEV (metres, initial-heading up). GT driven route green;
    the IDM's 2 s-ahead prediction AT THIS FRAME (video->head, re-expressed in the
    clip frame) orange from the live position. Spatial context, no drift."""
    im = Image.new("RGB", (W, H), (7, 10, 14))
    d = ImageDraw.Draw(im)
    xmn, xmx, ymn, ymx = ext
    pad = 30
    sc = min((W - 2*pad)/max(ymx - ymn, 1e-3), (H - 2*pad - 26)/max(xmx - xmn, 1e-3))

    def mp(p):
        return (W/2 - sc*(p[1] - (ymn+ymx)/2), (H-16) - pad - sc*(p[0] - xmn))
    # distance rings every 20 m along forward
    d.line([mp(route[0]), mp(route[-1])], fill=(7, 10, 14))     # noop keep aa
    d.line([mp(p) for p in route], fill=COL_GT, width=3)
    sxp, syp = mp(route[0])
    d.ellipse([sxp-4, syp-4, sxp+4, syp+4], fill=(235, 235, 235))   # start
    if len(pred_g) >= 2:
        d.line([mp(p) for p in pred_g], fill=COL_PRED, width=3)
    for p in pred_g[1:]:
        x, y = mp(p)
        d.ellipse([x-3, y-3, x+3, y+3], outline=COL_PRED, width=2)
    cx, cy = mp(cur)
    d.ellipse([cx-5, cy-5, cx+5, cy+5], fill=COL_GT, outline=(15, 15, 15))
    d.rectangle([0, 0, W, 20], fill=HUD_BG)
    d.text((6, 4), title, fill=HUD_FG, font=F_SUB)
    d.text((6, H-14), "GT route green · IDM 2 s-ahead orange · start ○",
           fill=HUD_DIM, font=F_TINY)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=9)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--out", default="/root/idmval/results/idm_recon_ep00009.mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=200)
    args = ap.parse_args()
    npz = args.npz or f"/root/idmval/results/recon_ep{args.idx:05d}.npz"
    z = np.load(npz)
    centers = z["centers"]; tp = z["traj_pred"]; tg = z["traj_gt"]
    sp = z["scal_pred"]; sg = z["scal_gt"]; adepw = z["ade_per_window"]
    poses = z["poses"]; man_gt = z["maneuvers"]
    dec_man = z["dec_man"]; dec_route = z["dec_route"]
    dr_pred = z["dr_pred"]; dr_gt = z["dr_gt"]; epid = int(z["episode_id"])
    clip_ade = float(adepw.mean())

    ep = torch.load(f"{VAL}/ep_{args.idx:05d}.pt", weights_only=False)
    frames = ep["frames_u8"]                      # [T,9,256,256], current = ch 6:9

    # clip display frame: initial GT heading up, anchored at first center
    yaw0 = float(poses[int(centers[0]), 2])
    st = poses[int(centers[0]), :2].copy()
    c, s = math.cos(-yaw0), math.sin(-yaw0)

    def disp(P):
        q = np.atleast_2d(P) - st
        return np.stack([c*q[:, 0] - s*q[:, 1], s*q[:, 0] + c*q[:, 1]], 1)
    route_disp = disp(poses[centers.astype(int), :2])         # GT driven route
    # per-frame IDM 2 s-ahead waypoints ego -> global -> display, from live pos
    pred_disp = []
    for n in range(len(centers)):
        t = int(centers[n])
        yt = float(poses[t, 2]); xy = poses[t, :2]
        cc, ss = math.cos(yt), math.sin(yt)
        gwp = np.stack([xy[0] + cc*tp[n, :, 0] - ss*tp[n, :, 1],
                        xy[1] + ss*tp[n, :, 0] + cc*tp[n, :, 1]], 1)
        pred_disp.append(np.concatenate([disp(xy), disp(gwp)], 0))
    allr = np.concatenate([route_disp] + pred_disp, 0)
    ext = (float(allr[:, 0].min()) - 3, float(allr[:, 0].max()) + 3,
           float(allr[:, 1].min()) - 3, float(allr[:, 1].max()) + 3)

    # fixed BEV extent for the 2 s inset
    xmax = max(12.0, float(np.abs(tg[:, :, 0]).max()), float(np.abs(tp[:, :, 0]).max()))
    xmax = min(60.0, 5*(int(xmax//5)+1))
    ymax = max(4.0, 1.2*max(float(np.abs(tg[:, :, 1]).max()), float(np.abs(tp[:, :, 1]).max())))

    RW = 300
    frames_dir = Path("/root/idmval/results/_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    picks = list(range(min(len(centers), args.max_frames)))
    for n in picks:
        t = int(centers[n])
        rgb = frames[t, -3:].permute(1, 2, 0).numpy()     # current frame (ch 6:9)
        im = Image.fromarray(rgb).resize((S, S), Image.LANCZOS).convert("RGB")
        d = ImageDraw.Draw(im)
        d.line([(0, CY*UP), (S, CY*UP)], fill=(70, 66, 40), width=1)   # horizon
        gt_dense = ego_future_path(poses, t, 20)
        pred_wp = np.concatenate([np.zeros((1, 2)), tp[n]], 0)          # ego->wp
        g = project(gt_dense)
        if len(g) >= 2:
            d.line(g, fill=COL_GT, width=6)
        for x, y in project(gt_dense[[4, 9, 14, 19]]):
            d.ellipse([x-3, y-3, x+3, y+3], fill=COL_GT)
        p = project(pred_wp)
        if len(p) >= 2:
            d.line(p, fill=COL_PRED, width=3)
        for x, y in project(tp[n]):
            d.ellipse([x-6, y-6, x+6, y+6], outline=COL_PRED, width=3)
        draw_bev_inset(im, gt_dense, tp[n], xmax, ymax)
        # HUD
        dm = pretty_man(int(dec_man[n])); gm = pretty_man(int(man_gt[t]))
        rt = pretty_route(int(dec_route[n]))
        d.rectangle([0, 0, S, 24], fill=HUD_BG)
        d.text((8, 5), f"idm_head_v1 · PAI-val ep_{args.idx:05d} (id {epid}) HELD-OUT "
               f"· GT green / IDM-from-video orange", fill=HUD_DIM, font=F_TINY)
        d.rectangle([0, 24, S, 92], fill=HUD_BG)
        d.text((8, 28), f"IDM maneuver: {dm}   (GT: {gm})", fill=HUD_FG, font=F_HUD)
        d.text((8, 50), f"f{t:03d}  ADE(2s) {adepw[n]:.2f} m  clip-mean {clip_ade:.2f} m  "
               f"IDM v {sp[n,0]:.1f} (GT {sg[n,0]:.1f}) m/s", fill=HUD_DIM, font=F_SUB)
        d.text((8, 70), f"route: {rt} · flat pinhole f{F_EFF:.0f} h{int(CY)} · "
               f"flagship-v1 FROZEN · non-causal IDM", fill=HUD_DIM, font=F_SUB)

        route = draw_route_panel(RW, S, route_disp, route_disp[n], pred_disp[n],
                                 ext, "WHOLE-CLIP ROUTE (m) · IDM 2 s-ahead")
        canv = Image.new("RGB", (S + RW, S), (0, 0, 0))
        canv.paste(im, (0, 0)); canv.paste(route, (S, 0))
        canv.save(frames_dir / f"f{n:04d}.png")

    out = Path(args.out)
    subprocess.run(["ffmpeg", "-y", "-r", str(args.fps), "-i",
                    str(frames_dir / "f%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "21", "-movflags",
                    "+faststart", str(out)], check=True, capture_output=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()
    frames_dir.rmdir()
    print(f"WROTE {out} ({out.stat().st_size/1e6:.2f} MB, {len(picks)} frames)")
    print("IDMVAL_RENDER_DONE", flush=True)


if __name__ == "__main__":
    main()
