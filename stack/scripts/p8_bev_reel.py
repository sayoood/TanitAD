"""P8 / I1c DELIVERABLE REEL — what the world model believes the world will look like.

Three panes per frame, over the prediction horizon:

    camera (the input)  |  decode(ẑ_{t+k}) — the WM's BELIEF  |  GT raster at t+k

``ẑ_{t+k}`` is the PREDICTED latent (``rollout_transitions`` under the observed
actions), decoded by the frozen P8 occupancy readout — so the middle pane is a
picture of the predictor's own future scene, not a re-encoding of it. The right
pane is the ``obstacle.offline`` join rasterised at the same step. Both panes use
the run's own operating point τ* when the gate JSON carries one (attempt-2's
threshold sweep), so what is drawn is exactly what was scored.

Every heavy import is lazy: this module imports on CPU without torch-CUDA, so the
frame compositor is unit-testable off-pod.

Pod usage (after a p8 run):
  OMP_NUM_THREADS=6 PYTHONPATH=/workspace/TanitAD_head/stack \\
  python3 scripts/p8_bev_reel.py \\
      --p8-run /workspace/experiments/p8-occupancy-c \\
      --ckpt   /workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt \\
      --v2-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \\
      --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \\
      --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \\
      --v2-subframe 176x624 \\
      --raster-source join-file --join-file /workspace/data/p8_join/combined140.jsonl \\
      --out /workspace/experiments/p8_reel --n-windows 12
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

PANE_H = 320                     # BEV pane render height (px)
CAM_H = 320                      # camera pane height (px)
PAD = 14
BG = (17, 21, 28)
FG = (226, 232, 240)
MUTED = (128, 141, 158)
BELIEF = (56, 189, 248)          # cyan — the WM's belief
TRUTH = (250, 204, 21)           # amber — ground truth
AGREE = (74, 222, 128)           # green — overlap


def _font(size: int):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def raster_to_rgb(occ: np.ndarray, colour, grid_lines: bool = True
                  ) -> np.ndarray:
    """Occupancy [nx, ny] in {0,1} → an ego-frame BEV image (+x forward = up)."""
    occ = np.asarray(occ)
    nx, ny = occ.shape
    img = np.zeros((nx, ny, 3), dtype=np.uint8)
    img[:] = np.array(BG, dtype=np.uint8)
    if grid_lines:
        img[::20, :] = np.array((32, 39, 50), dtype=np.uint8)
        img[:, ::16] = np.array((32, 39, 50), dtype=np.uint8)
    m = occ > 0.5
    img[m] = np.array(colour, dtype=np.uint8)
    # ego marker at the origin row (x=0 is the first row of the grid)
    img[0:3, ny // 2 - 1:ny // 2 + 2] = np.array((248, 250, 252), dtype=np.uint8)
    return np.flipud(img)                      # forward up


def overlay_rgb(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Belief vs truth in one pane: belief-only, truth-only, and agreement."""
    pred = np.asarray(pred) > 0.5
    gt = np.asarray(gt) > 0.5
    nx, ny = pred.shape
    img = np.zeros((nx, ny, 3), dtype=np.uint8)
    img[:] = np.array(BG, dtype=np.uint8)
    img[::20, :] = np.array((32, 39, 50), dtype=np.uint8)
    img[:, ::16] = np.array((32, 39, 50), dtype=np.uint8)
    img[pred & ~gt] = np.array(BELIEF, dtype=np.uint8)
    img[gt & ~pred] = np.array(TRUTH, dtype=np.uint8)
    img[pred & gt] = np.array(AGREE, dtype=np.uint8)
    img[0:3, ny // 2 - 1:ny // 2 + 2] = np.array((248, 250, 252), dtype=np.uint8)
    return np.flipud(img)


def compose_frame(cam: np.ndarray, pred_occ: np.ndarray, gt_occ: np.ndarray,
                  caption: str, subcaption: str = "", width: int = 1280
                  ) -> np.ndarray:
    """One reel frame: camera | belief | truth(+overlay). Pure function."""
    from PIL import Image, ImageDraw
    cam = np.asarray(cam)
    if cam.dtype != np.uint8:
        cam = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    if cam.ndim == 2:
        cam = np.stack([cam] * 3, -1)
    if cam.shape[-1] > 3:
        cam = cam[..., -3:]

    pane_w = (width - 4 * PAD) // 3
    canvas = Image.new("RGB", (width, CAM_H + 2 * PAD + 42), BG)
    d = ImageDraw.Draw(canvas)
    f_lab = _font(15)
    f_cap = _font(13)

    def _place(arr, x, label, colour=FG):
        im = Image.fromarray(arr).resize((pane_w, PANE_H), Image.NEAREST)
        canvas.paste(im, (x, PAD + 22))
        d.text((x, PAD + 2), label, font=f_lab, fill=colour)

    _place(cam, PAD, "camera — input", FG)
    _place(raster_to_rgb(pred_occ, BELIEF), 2 * PAD + pane_w,
           "decode(ẑ) — the WM's belief", BELIEF)
    _place(overlay_rgb(pred_occ, gt_occ), 3 * PAD + 2 * pane_w,
           "belief ∩ truth", AGREE)
    y = PAD + 22 + PANE_H + 6
    d.text((PAD, y), caption, font=f_cap, fill=FG)
    if subcaption:
        d.text((PAD, y + 17), subcaption, font=f_cap, fill=MUTED)
    return np.array(canvas)


def _tau_star(p8_run: str) -> float:
    """The run's own operating point, so the reel shows what was scored."""
    for name in ("p8_gate.json", "gate.json"):
        p = os.path.join(p8_run, name)
        if os.path.exists(p):
            try:
                g = json.load(open(p))
                for blk in (g, g.get("mini_eval", {})):
                    if isinstance(blk, dict) and "tau_star" in blk:
                        return float(blk["tau_star"])
            except Exception:
                pass
    return 0.5


def main(argv=None) -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    from train_p8_occupancy import (BEVOccupancyHead, batch_rasters,
                                    build_args, build_raster_source,
                                    p8_latents)
    from tanitad.data.bev_raster import GRID_DEFAULT

    # P8's own arg surface + the reel's extras (parsed off the same parser so
    # the geometry/corpus seams cannot drift from the run being visualised).
    import argparse
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--p8-run", required=True)
    extra.add_argument("--n-windows", type=int, default=12)
    extra.add_argument("--reel-ks", default="5,10,15,20")
    extra.add_argument("--fps", type=int, default=3)
    known, rest = extra.parse_known_args(argv)
    a = build_args(rest + ["--out", known.p8_run + "_reel_tmp"])

    device = a.device if torch.cuda.is_available() else "cpu"
    amp_on = (device == "cuda") and not a.no_amp
    ks = tuple(int(x) for x in known.reel_ks.split(",") if x)
    os.makedirs(a.out, exist_ok=True)
    out_dir = os.path.join(known.p8_run, "reel")
    os.makedirs(out_dir, exist_ok=True)

    from torch.utils.data import default_collate

    from eval_flagship_v4 import (_eval_cfg, _plan, build_v2_val_episodes,
                                  load_v1_from_ck, resolve_eval_frames)
    from train_flagship4b import FlagshipWindowDataset
    from train_flagship_v4 import _to_device

    cfg = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(a, cfg, label="p8_bev_reel")
    plan = _plan(cfg)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    world, _g, base_step = load_v1_from_ck(ck, device, frame=model_frame)
    del ck
    val_eps, _ = build_v2_val_episodes(a, cache_frame=cache_frame,
                                       train_frame=model_frame)
    ds_val = FlagshipWindowDataset(val_eps, window=cfg.predictor.window,
                                   max_horizon=plan.max_horizon,
                                   maneuver_h=plan.maneuver_h,
                                   channels=cfg.encoder.in_channels)
    source = build_raster_source(a, val_eps)

    head = BEVOccupancyHead(world.state_dim, grid=GRID_DEFAULT,
                            ch0=a.ch0, ch1=a.ch1, enforce_band=True).to(device)
    hp = os.path.join(known.p8_run, "head.pt")
    if not os.path.exists(hp):
        cands = [f for f in os.listdir(known.p8_run) if f.endswith(".pt")]
        if not cands:
            raise SystemExit(f"[reel] no head checkpoint in {known.p8_run}")
        hp = os.path.join(known.p8_run, sorted(cands)[-1])
    sd = torch.load(hp, map_location=device, weights_only=False)
    head.load_state_dict(sd["head"] if isinstance(sd, dict) and "head" in sd
                         else sd)
    head.eval()
    tau = _tau_star(known.p8_run)
    print(f"[reel] head {os.path.basename(hp)} · tau* {tau} · trunk step "
          f"{base_step}", flush=True)

    # windows on the eval grid that the join actually labels at every k
    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < a.episodes and t % a.stride == 0]
    frames_out = []
    used = 0
    with torch.no_grad():
        for i in sel:
            if used >= known.n_windows:
                break
            b = _to_device(default_collate([ds_val[i]]), device)
            r0, keep0, _ = batch_rasters(ds_val, [i], source, 0, GRID_DEFAULT)
            if r0 is None:
                continue
            _zt, _ze, z_hat = p8_latents(world, b, ks, amp_on=amp_on,
                                         want_pred=True, want_enc_k=False)
            cam = b["frames"][0, -1].detach().float().cpu().numpy()
            if cam.ndim == 3 and cam.shape[0] in (1, 3, 9, 12):
                cam = np.transpose(cam, (1, 2, 0))
            ok = True
            per_k = []
            for k in ks:
                rk, keep, _m = batch_rasters(ds_val, [i], source, k,
                                             GRID_DEFAULT)
                if rk is None:
                    ok = False
                    break
                logits = head(z_hat[k][keep])
                p = torch.sigmoid(logits)[0].detach().float().cpu().numpy()
                per_k.append((k, (p > tau).astype(np.float32),
                              rk[0].numpy()))
            if not ok:
                continue
            for k, pred, gt in per_k:
                inter = float(((pred > .5) & (gt > .5)).sum())
                union = float(((pred > .5) | (gt > .5)).sum())
                iou = inter / union if union > 0 else float("nan")
                frames_out.append(compose_frame(
                    cam, pred, gt,
                    caption=f"window {used + 1}/{known.n_windows}   "
                            f"horizon k = {k}  ({k / 10:.1f} s ahead)   "
                            f"IoU {iou:.3f}",
                    subcaption="cyan = predicted-latent belief · amber = GT "
                               f"agents · green = agreement · τ* = {tau}"))
            used += 1
            print(f"[reel] window {used} rendered ({len(per_k)} horizons)",
                  flush=True)

    if not frames_out:
        raise SystemExit("[reel] no labelled windows on the grid — nothing to render")
    # Video when an encoder exists; otherwise a PIL contact sheet — the pods do
    # not all carry imageio/cv2/ffmpeg, and a missing codec must not cost the
    # deliverable (MEASURED 2026-08-12: pod4's venv has PIL only).
    from PIL import Image
    still = os.path.join(out_dir, "p8_belief_still.png")
    Image.fromarray(frames_out[len(frames_out) // 2]).save(still)
    sheet = os.path.join(out_dir, "p8_belief_sheet.png")
    cols = 1
    fh, fw = frames_out[0].shape[:2]
    rows = min(len(frames_out), 12)
    canvas = Image.new("RGB", (fw * cols, fh * rows), (17, 21, 28))
    for i, f in enumerate(frames_out[:rows]):
        canvas.paste(Image.fromarray(f), (0, i * fh))
    canvas.save(sheet)
    made = [still, sheet]
    try:
        import imageio.v2 as imageio
        mp4 = os.path.join(out_dir, "p8_belief_reel.mp4")
        w = imageio.get_writer(mp4, fps=known.fps, macro_block_size=1)
        for f in frames_out:
            w.append_data(f)
        w.close()
        made.append(mp4)
    except Exception as e:
        print(f"[reel] no video encoder ({type(e).__name__}) — stills only",
              flush=True)
        for i, f in enumerate(frames_out):
            Image.fromarray(f).save(os.path.join(out_dir, f"frame_{i:03d}.png"))
        made.append(f"{len(frames_out)} PNG frames")
    print(f"[reel] wrote {len(frames_out)} frames -> {made}", flush=True)
    print("P8REEL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
