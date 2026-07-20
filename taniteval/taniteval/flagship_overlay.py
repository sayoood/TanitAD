"""TanitEval — flagship-30k front-camera trajectory overlay videos.

Flagship variant of cam_overlay.py (which is REF-B-specific in its prediction
path: waypoint head, k in {5,10,15,20} only).  Here the predicted trajectory is
the GROUNDED OPERATIVE ROLLOUT — encode_window -> predictor rollout under TRUE
actions -> step_readout (grounding.step['op']) -> SE(2) accumulate — the exact
path the leaderboard ADE uses (taniteval/rollout.py), run at stride 1 with the
FULL k=20 (2 s @ 10 Hz) trajectory kept per frame.

Camera model, projection and GT-path helpers are reused from cam_overlay
verbatim (per-clip principal-point crop => cx=cy=128 for every clip/rig).

Usage (on the eval pod):
  PYTHONPATH=/root/taniteval:/root/TanitAD/stack python -m taniteval.flagship_overlay
  ... --only 3            # single clip (validation pass)
  ... --stills-only 3     # 3 sanity PNGs for a clip, no video
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import loaders                                      # noqa: E402
from taniteval.cam_overlay import UP, ego_future_path, project     # noqa: E402
from taniteval.registry import MODELS                              # noqa: E402
from tanitad.data.mixing import load_episode                       # noqa: E402
from tanitad.models.metric_dynamics import rollout_decode          # noqa: E402

VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
OUT = Path("/root/taniteval/results/videos")
K = 20                                   # 2 s @ 10 Hz — leaderboard horizon
WINDOW = 8
SPEED_SCALE = 10.0                       # v0 channel scale (matches trainers)
WP_IDX = torch.tensor([4, 9, 14, 19])    # ADE steps 5/10/15/20 (0-based)
S = 256 * UP                             # rendered frame size (512)

COL_GT = (110, 235, 131)                 # green
COL_PRED = (255, 122, 61)                # orange
HUD_BG = (10, 14, 19)
HUD_FG = (233, 237, 243)
HUD_DIM = (150, 160, 175)

# (ep_idx, scenario_tag) — picked from windows_flagship-30k.pt strata:
#   ep03 sharp turn (net-heading up to 68 deg), ep38 gentle curve ~9.5 m/s,
#   ep17 straight cruise 18 m/s (ep04/23 m/s was near-black night — swapped),
#   ep27 braking 16 -> 3 m/s,
#   ep31 high-speed 36 m/s (weak stratum, ep ADE 0.92 ~ 2x mean),
#   ep28 high-speed curve ~20 m/s (ep ADE 0.67, worst window 1.32).
CLIPS = [
    (3, "sharpturn"),
    (38, "gentleturn"),
    (17, "straightcruise"),
    (27, "braking"),
    (31, "highspeed-wrong"),
    (28, "highspeed-curve"),
]


def _font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


F_HUD = _font(15)
F_SUB = _font(13)


@torch.no_grad()
def flagship_episode_rollouts(model, step_readout, ep, device, batch=16):
    """Stride-1 grounded k=20 rollouts for every frame.

    Returns dict t -> (wp_full [20,2] ego, window ADE m, v0 m/s); t = window
    end (= displayed frame). Ports rollout.collect verbatim (speed_input on)."""
    fr, T = ep.frames, ep.frames.shape[0]
    poses, actions = ep.poses.float(), ep.actions.float()
    starts = list(range(0, T - WINDOW - K))
    out = {}
    for i in range(0, len(starts), batch):
        ch = starts[i:i + batch]
        last = torch.tensor([s + WINDOW - 1 for s in ch])
        fw = torch.stack([torch.as_tensor(fr[s:s + WINDOW])
                          for s in ch]).to(device).float().div_(255.0)
        aw = torch.stack([actions[s:s + WINDOW] for s in ch]).to(device)
        fa = torch.stack([actions[s + WINDOW:s + WINDOW + K]
                          for s in ch]).to(device)
        v0 = (poses[last, 3:4] / SPEED_SCALE).to(device)   # 3rd action channel
        aw = torch.cat([aw, v0.unsqueeze(1).expand(-1, aw.shape[1], -1)],
                       dim=-1)
        fa = torch.cat([fa, v0.unsqueeze(1).expand(-1, fa.shape[1], -1)],
                       dim=-1)
        states = model.encode_window(fw)
        wp_full, _ = rollout_decode(model.predictor, states, aw, fa,
                                    step_readout, K)          # [b, 20, 2]
        wp_full = wp_full.cpu().float()
        for j, s in enumerate(ch):
            t = s + WINDOW - 1
            gt = ego_future_path(poses, t, K)                  # [20, 2]
            ade = float(torch.linalg.norm(
                wp_full[j][WP_IDX] - gt[WP_IDX], dim=-1).mean())
            out[t] = (wp_full[j], ade, float(poses[t, 3]))
    return out


def draw_frame(rgb_hwc, gt_path, pred_path, top, bottom):
    im = Image.fromarray(rgb_hwc).resize((S, S), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(im)
    # GT wide underneath, pred narrow on top -> both visible when aligned
    g = project(gt_path)
    if len(g) >= 2:
        d.line(g, fill=COL_GT, width=7)
    p = project(pred_path)
    if len(p) >= 2:
        d.line(p, fill=COL_PRED, width=3)
    for x, y in project(gt_path[WP_IDX]):       # GT horizon dots
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=COL_GT)
    for x, y in project(pred_path[WP_IDX]):     # pred rings at the 4 horizons
        d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=COL_PRED, width=3)
    d.rectangle([0, 0, S, 24], fill=HUD_BG)
    d.text((8, 4), top, fill=HUD_DIM, font=F_SUB)
    d.rectangle([0, S - 26, S, S], fill=HUD_BG)
    d.text((8, S - 22), bottom, fill=HUD_FG, font=F_HUD)
    return im


def render_episode(model, step_readout, ep, clip_name, label, device, fps=10,
                   max_frames=180, stills_only=False):
    preds = flagship_episode_rollouts(model, step_readout, ep, device)
    ts = sorted(preds)[:max_frames]
    ades = [preds[t][1] for t in ts]
    mean_ade = sum(ades) / len(ades)
    frames_dir = OUT / f"_frames_{clip_name}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    top = "flagship-30k grounded rollout · GT green · pred orange · 2 s horizon"
    picks = ts if not stills_only else [ts[len(ts) // 4], ts[len(ts) // 2],
                                        ts[3 * len(ts) // 4]]
    for n, t in enumerate(picks):
        wp, ade, v0 = preds[t]
        rgb = ep.frames[t, -3:].permute(1, 2, 0).numpy()
        gt = ego_future_path(ep.poses.float(), t, K)
        bottom = f"{label}  f{t:03d}  ADE {ade:.2f} m  v0 {v0:.1f} m/s"
        im = draw_frame(rgb, gt, wp, top, bottom)
        if stills_only:
            im.save(OUT / f"still_{clip_name}_f{t:03d}.png")
        else:
            im.save(frames_dir / f"f{n:04d}.png")
    if stills_only:
        return None, len(picks), mean_ade
    mp4 = OUT / f"{clip_name}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-r", str(fps), "-i", str(frames_dir / "f%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
         "-movflags", "+faststart", str(mp4)],
        check=True, capture_output=True)
    shutil.rmtree(frames_dir)
    return str(mp4), len(ts), mean_ade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None,
                    help="render just this episode index")
    ap.add_argument("--stills-only", type=int, default=None,
                    help="3 sanity PNGs for this episode index, no video")
    args = ap.parse_args()
    device = "cuda"
    entry = [m for m in MODELS if m["key"] == "flagship-30k"][0]
    L = loaders.load(entry, device)
    model, sr = L["model"], L["step_readout"]
    assert sr is not None, "flagship-30k step_readout missing"
    print(f"[load] flagship-30k step={L['step']}", flush=True)
    files = sorted(Path(VAL).glob("ep_*.pt"))
    OUT.mkdir(parents=True, exist_ok=True)
    want = args.only if args.only is not None else args.stills_only
    clips = [c for c in CLIPS if want is None or c[0] == want]
    for idx, tag in clips:
        ep = load_episode(str(files[idx]), mmap=True)
        name = f"flagship30k_overlay_{tag}_ep{idx:02d}"
        mp4, nfr, mean_ade = render_episode(
            model, sr, ep, name, f"ep{idx:02d} {tag}", device,
            stills_only=args.stills_only is not None)
        print(f"[video] {name}: {mp4} frames={nfr} clip-meanADE={mean_ade:.3f}",
              flush=True)
    print("OVERLAY_DONE", flush=True)


if __name__ == "__main__":
    main()
