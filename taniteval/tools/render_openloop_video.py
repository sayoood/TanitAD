#!/usr/bin/env python3
"""LONG open-loop videos: camera + GT/pred trajectory overlays + a large metric BEV.

⛔ **THE LABEL THIS FILE EXISTS TO BURN INTO EVERY FRAME.** The rollout shown here
is ``rollout_decode`` under the **expert's TRUE FUTURE ACTIONS** — the same surface
``taniteval.rollout.collect`` scores, whose own PC2 record says
``actions_source="expert_future"`` and ``pc2_pass=False`` by construction. It is a
**world-model fidelity** decode of a known control sequence, **NOT autonomous
driving and NOT a hierarchy result**. A video of an orange line tracking a green
line is the single most over-claimable artefact this programme produces, so the
frame says what it is, permanently, in the banner.

⚠️ **OPEN LOOP** additionally means the ego follows the logged trajectory: every
frame is the rig's real image at the real pose. The model never moves the car,
so there is no control drift and no divergence — what is left is prediction.

Three panels, the programme's standard (`taniteval.corpus_overlay`):

1. **CAMERA** — GT (green, wide) and prediction (orange, narrow) projected through
   the flat-ground pinhole (cx = cy = 128, f_eff 266, cam_h 1.5), correct by build
   for a principal-point-centred 256 crop.
2. **LARGE METRIC BEV** — the same two paths in metres, top-down, ego at
   bottom-centre. **Calibration-independent**: if the projection were wrong the
   camera panel would lie and this panel would not, which is why it is big here
   rather than a 152 px inset.
3. **HUD** — decoded tactical manoeuvre + strategic route, per-frame and rolling
   ADE, speed, and a scrolling ADE trace so a spike is visible as it happens.

Episodes are concatenated into ONE long file with a banner per episode. Selection
is ``spread`` by default — evenly spaced across the corpus — because ``first`` is
an arbitrary slice and ``best`` is cherry-picking; whichever is used is written
into the banner and the sidecar JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

W_CAM = 512
W_BEV = 420
H_BAN = 46                      # top banner
H_HUD = 96                      # bottom HUD
PAD = 10


def _p(*a):
    print(*a, flush=True)


def _ffmpeg() -> str:
    """⛔ Resolved BEFORE the render, never after. A missing encoder discovered
    at the encode step throws away every frame that was just rendered — on a
    30-episode run that is several thousand frames and ~10 minutes of GPU.

    PATH first; then the static binary `imageio-ffmpeg` ships, which is the one
    that is actually installable on a pod where `apt-get` has no package lists."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("no ffmpeg on PATH and imageio-ffmpeg is not installed "
                 "(`pip install imageio-ffmpeg` ships a static binary). "
                 "Refusing to render frames that could not be encoded.")


def draw_bev_large(size, gt, pred, xmax, ymax, past, cols, fonts):
    """Top-down metric BEV — ego at bottom-centre, forward up, left is left.

    ⭐ Calibration-INDEPENDENT. The camera panel depends on a projection model;
    this one depends only on the numbers the metric is computed from, so the two
    panels disagreeing is a signal (bad calib), never noise."""
    COL_GT, COL_PRED, HUD_DIM = cols["gt"], cols["pred"], cols["dim"]
    F_TINY, F_SUB = fonts["tiny"], fonts["sub"]
    w, h = size
    im = Image.new("RGB", (w, h), (8, 11, 15))
    d = ImageDraw.Draw(im, "RGBA")
    pad = 34
    cx, by, top = w // 2, h - pad, pad + 14

    def m2px(X, Y):
        return (cx - (Y / ymax) * ((w / 2) - pad),
                by - (max(X, 0.0) / xmax) * (by - top))

    step = 10 if xmax > 25 else 5
    r = step
    while r <= xmax + 0.1:
        _, py = m2px(r, 0)
        d.line([(12, py), (w - 12, py)], fill=(34, 42, 52))
        d.text((14, py - 12), f"{r} m", fill=(96, 106, 120), font=F_TINY)
        r += step
    for lat in (-ymax / 2, ymax / 2):
        px, _ = m2px(0, lat)
        d.line([(px, top), (px, by)], fill=(28, 34, 42))
    d.line([(cx, top), (cx, by)], fill=(44, 54, 66))

    if past is not None and len(past) >= 2:          # where the ego came from
        pp = [m2px(float(a), float(b)) for a, b in past]
        d.line(pp, fill=(70, 82, 96), width=2)
    g = [m2px(float(p[0]), float(p[1])) for p in gt]
    if len(g) >= 2:
        d.line(g, fill=COL_GT, width=6)
    q = [m2px(float(p[0]), float(p[1])) for p in pred]
    if len(q) >= 2:
        d.line(q, fill=COL_PRED, width=3)
    for i in (4, 9, 14, 19):                          # 0.5 / 1.0 / 1.5 / 2.0 s
        if i < len(gt):
            x, y = m2px(float(gt[i][0]), float(gt[i][1]))
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=COL_GT)
        if i < len(pred):
            x, y = m2px(float(pred[i][0]), float(pred[i][1]))
            d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=COL_PRED, width=3)
    d.polygon([(cx - 6, by), (cx + 6, by), (cx, by - 12)], fill=(232, 236, 242))
    d.text((12, 8), "BEV top-down · metres · calibration-independent",
           fill=HUD_DIM, font=F_TINY)
    d.text((w - 96, h - 20), "◯ = 0.5 s marks", fill=(96, 106, 120), font=F_TINY)
    return im


def draw_trace(size, hist, cols, fonts, ymax=3.0):
    """Scrolling per-frame ADE. A clip-mean hides the spike that produced it."""
    w, h = size
    im = Image.new("RGB", (w, h), (10, 14, 19))
    d = ImageDraw.Draw(im)
    if not hist:
        return im
    n = len(hist)
    xs = np.linspace(0, w - 1, n)
    ys = [h - 3 - min(v / ymax, 1.0) * (h - 8) for v in hist]
    for lvl in (0.5, 1.0, 2.0):
        y = h - 3 - min(lvl / ymax, 1.0) * (h - 8)
        d.line([(0, y), (w, y)], fill=(34, 42, 52))
        d.text((2, y - 11), f"{lvl:g}", fill=(80, 90, 104), font=fonts["tiny"])
    d.line(list(zip(xs, ys)), fill=cols["pred"], width=2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-config", default=None)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True, help="output .mp4 FILE")
    ap.add_argument("--episodes", type=int, default=30,
                    help="how many episodes to concatenate into ONE long video")
    ap.add_argument("--select", default="spread",
                    choices=["spread", "first"],
                    help="⛔ recorded in the banner and the sidecar. 'spread' is "
                         "evenly spaced across the corpus (not cherry-picked); "
                         "'first' is an arbitrary slice. For worst/best clips "
                         "use --episode-list with indices derived from a banked "
                         "eval, so the ranking's provenance is explicit.")
    ap.add_argument("--episode-list", default=None,
                    help="comma-separated episode indices, used verbatim. "
                         "--select is then reported as 'explicit:<label>'.")
    ap.add_argument("--select-label", default="explicit",
                    help="what --episode-list means, e.g. 'worst-30-by-ADE'. "
                         "⛔ burned into the banner: a hand-picked reel must "
                         "never be quotable as a representative one.")
    ap.add_argument("--corpus-label", default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--speed-input", action="store_true")
    ap.add_argument("--max-frames-per-ep", type=int, default=0,
                    help="0 = all")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if os.path.isdir(a.out):
        sys.exit(f"--out must be a FILE, got a directory: {a.out}")
    ffmpeg = _ffmpeg()          # resolved BEFORE the render, not after it

    from taniteval import corpus_overlay as co, loaders
    from taniteval.cam_overlay import ego_future_path
    from taniteval.data import load_frames
    from taniteval.flagship_overlay import (COL_GT, COL_PRED, HUD_BG, HUD_DIM,
                                            HUD_FG, K, _font)

    cols = {"gt": COL_GT, "pred": COL_PRED, "dim": HUD_DIM, "fg": HUD_FG}
    fonts = {"big": _font(20), "hud": _font(15), "sub": _font(12),
             "tiny": _font(10)}

    entry = {"arch": "flagship-worldmodel-v2" if a.run_config
             else "flagship-worldmodel",
             "ckpt": a.ckpt, "run_config": a.run_config,
             "speed_input": bool(a.speed_input)}
    h = loaders.load(entry, device=a.device)
    model, sr = h["model"].eval(), h["step_readout"]
    _p(f"[model] {a.arm} step={h.get('step')} loaded STRICT")

    files = sorted(Path(a.corpus).glob("ep_*.pt"))
    if not files:
        sys.exit(f"no ep_*.pt under {a.corpus}")
    n_avail = len(files)
    if a.episode_list:
        idx = [int(x) for x in a.episode_list.split(",") if x.strip() != ""]
        bad = [i for i in idx if not 0 <= i < n_avail]
        if bad:
            sys.exit(f"--episode-list has out-of-range indices {bad[:5]} for a "
                     f"{n_avail}-episode corpus")
        select = f"explicit:{a.select_label}"
    elif a.select == "first":
        idx = list(range(min(a.episodes, n_avail)))
        select = "first"
    else:
        idx = list(np.linspace(0, n_avail - 1,
                               min(a.episodes, n_avail)).astype(int))
        select = "spread"
    _p(f"[corpus] {n_avail} episodes; rendering {len(idx)} ({select})")

    proj = co.FlatProjector(128.0)
    clabel = a.corpus_label or Path(a.corpus).name
    frames_dir = Path(a.out).with_suffix("")
    frames_dir = frames_dir.parent / f"_frames_{frames_dir.name}"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    W = PAD + W_CAM + PAD + W_BEV + PAD
    H = H_BAN + W_CAM + H_HUD
    n_out, per_ep, t0 = 0, [], time.time()

    for rank, ei in enumerate(idx):
        ep = load_frames([files[ei]])[0]
        poses, actions = ep.poses.float(), ep.actions.float()
        preds = co.episode_rollouts(model, sr, ep.feats, poses, actions,
                                    "frames", a.speed_input, False, a.device)
        ts = sorted(preds)
        if a.max_frames_per_ep:
            ts = ts[:a.max_frames_per_ep]
        if not ts:
            _p(f"  [skip] ep{ei:05d}: too few frames")
            continue
        ades = [preds[t]["ade"] for t in ts]
        mean_ade = float(np.mean(ades))
        xmax, ymax = co.clip_extent(preds, poses)
        hist = []
        for t in ts:
            dct = preds[t]
            gt = ego_future_path(poses, t, K)
            wp = dct["wp"]
            hist.append(dct["ade"])
            canvas = Image.new("RGB", (W, H), (6, 9, 12))
            d = ImageDraw.Draw(canvas)

            # --- camera panel ---------------------------------------------- #
            # RawEp exposes the frame stack as `.feats` (data.RawEp:224);
            # `-3:` is the CURRENT RGB of the 3-frame stack.
            rgb = torch.as_tensor(ep.feats[t, -3:]).permute(1, 2, 0).numpy()
            cam = Image.fromarray(rgb).resize((W_CAM, W_CAM), Image.LANCZOS)
            cd = ImageDraw.Draw(cam)
            g = proj(gt)
            if len(g) >= 2:
                cd.line(g, fill=COL_GT, width=7)
            q = proj(wp)
            if len(q) >= 2:
                cd.line(q, fill=COL_PRED, width=3)
            for x, y in proj(gt[[4, 9, 14, 19]]):
                cd.ellipse([x - 3, y - 3, x + 3, y + 3], fill=COL_GT)
            for x, y in proj(wp[[4, 9, 14, 19]]):
                cd.ellipse([x - 6, y - 6, x + 6, y + 6], outline=COL_PRED, width=3)
            canvas.paste(cam, (PAD, H_BAN))

            # --- large BEV -------------------------------------------------- #
            past = poses[max(0, t - 30):t + 1, :2].numpy()
            if len(past) >= 2:                    # into the ego frame at t
                c, s = np.cos(float(poses[t, 2])), np.sin(float(poses[t, 2]))
                dxy = past - past[-1]
                past = np.stack([dxy[:, 0] * c + dxy[:, 1] * s,
                                 -dxy[:, 0] * s + dxy[:, 1] * c], 1)
            bev = draw_bev_large((W_BEV, W_CAM - 70), gt, wp, xmax, ymax,
                                 past, cols, fonts)
            canvas.paste(bev, (PAD + W_CAM + PAD, H_BAN))
            canvas.paste(draw_trace((W_BEV, 60), hist[-240:], cols, fonts),
                         (PAD + W_CAM + PAD, H_BAN + W_CAM - 60))
            d.text((PAD + W_CAM + PAD + 4, H_BAN + W_CAM - 78),
                   "per-frame ADE (last 24 s)", fill=HUD_DIM, font=fonts["tiny"])

            # --- banner: the claim guard ------------------------------------ #
            d.rectangle([0, 0, W, H_BAN], fill=HUD_BG)
            d.text((PAD, 5), f"{a.arm}  ·  {clabel}  ·  ep {ei:05d} "
                             f"({rank+1}/{len(idx)}, {select})",
                   fill=HUD_FG, font=fonts["big"])
            d.text((PAD, 28),
                   "OPEN LOOP — ego follows the LOG. Rollout decodes the "
                   "EXPERT'S TRUE FUTURE ACTIONS: world-model FIDELITY, "
                   "NOT autonomous driving, NOT a hierarchy result.",
                   fill=(235, 180, 90), font=fonts["sub"])

            # --- HUD --------------------------------------------------------- #
            y = H_BAN + W_CAM
            d.rectangle([0, y, W, H], fill=HUD_BG)
            man = co.pretty_man(dct["man"])
            route = co.pretty_route(dct["route"])
            d.text((PAD, y + 6), f"tactical: {man}     strategic: route {route}",
                   fill=HUD_FG, font=fonts["hud"])
            d.text((PAD, y + 30),
                   f"frame {t:03d}   ADE {dct['ade']:.2f} m   "
                   f"clip-mean {mean_ade:.2f} m   v0 {dct['v0']:.1f} m/s",
                   fill=HUD_DIM, font=fonts["sub"])
            d.text((PAD, y + 50),
                   "GT = green (wide) · prediction = orange (narrow) · "
                   "markers at 0.5/1.0/1.5/2.0 s · horizon 2 s @ 10 Hz",
                   fill=HUD_DIM, font=fonts["sub"])
            d.text((PAD, y + 70),
                   "camera: flat-ground pinhole cx=cy=128 f_eff 266 cam_h 1.5 "
                   "(correct by build for a principal-point-centred 256 crop); "
                   "BEV is calibration-independent",
                   fill=(110, 120, 134), font=fonts["tiny"])
            d.line([(PAD, y), (W - PAD, y)], fill=(40, 48, 58))
            canvas.save(frames_dir / f"f{n_out:06d}.png")
            n_out += 1
        per_ep.append({"episode": int(ei), "frames": len(ts),
                       "mean_ade_m": round(mean_ade, 4),
                       "max_ade_m": round(float(np.max(ades)), 4)})
        _p(f"  ep{ei:05d} ({rank+1}/{len(idx)}) {len(ts)} frames  "
           f"ADE {mean_ade:.3f}  [{time.time()-t0:.0f}s, {n_out} frames total]")

    if not n_out:
        sys.exit("no frames rendered")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-r", str(a.fps),
                    "-i", str(frames_dir / "f%06d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
                    "-movflags", "+faststart", a.out],
                   check=True, capture_output=True)
    shutil.rmtree(frames_dir)
    dur = n_out / a.fps
    side = {
        "arm": a.arm, "ckpt": a.ckpt, "corpus": a.corpus,
        "corpus_label": clabel, "select": select,
        "episodes_rendered": len(per_ep), "episodes_available": n_avail,
        "frames": n_out, "fps": a.fps, "duration_s": round(dur, 1),
        "resolution": f"{W}x{H}",
        "mean_ade_over_rendered_m": round(
            float(np.mean([e["mean_ade_m"] for e in per_ep])), 4),
        "per_episode": per_ep,
        "⛔_what_this_is": (
            "OPEN LOOP: the ego follows the logged trajectory, so there is no "
            "control drift. The rollout decodes the EXPERT'S TRUE FUTURE "
            "ACTIONS (rollout.collect's PC2 record: actions_source="
            "'expert_future', pc2_pass False by construction) — this is "
            "world-model FIDELITY, not autonomous driving and not a hierarchy "
            "result. The banner says so on every frame."),
        "⛔_not_a_selection_claim": (
            f"episodes were chosen by '{select}'. 'spread' is evenly spaced "
            f"across the corpus and is the only non-cherry-picked option here; "
            f"the choice is burned into the banner so a clip cannot be quoted "
            f"without it."),
    }
    with open(a.out + ".json", "w") as f:
        json.dump(side, f, indent=2)
    _p(f"\n[video] {a.out}  {n_out} frames  {dur:.0f}s  {W}x{H}")
    _p(f"[video] mean ADE over rendered episodes "
       f"{side['mean_ade_over_rendered_m']:.4f} m")
    _p(f"[video] sidecar {a.out}.json")


if __name__ == "__main__":
    main()
