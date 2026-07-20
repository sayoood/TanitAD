"""TanitEval — direct-trajectory-head overlay videos (REF-B / REF-C).

THE SAME THREE-PANEL STANDARD as ``taniteval.corpus_overlay`` — camera
projection + metric BEV inset + tactical/strategic/ADE text HUD — rendered by
IMPORTING corpus_overlay's own drawing primitives (``draw_frame``, ``draw_bev``
via draw_frame, ``FlatProjector``, ``clip_extent``, ``pretty_man``,
``pretty_route``), so the visual contract is literally identical rather than
merely similar.

What differs is the PREDICTION path. corpus_overlay renders the grounded
operative rollout (encode_window -> rollout_decode under true actions ->
step_readout -> SE(2) accumulate) and asserts a step_readout exists. REF-B (a
hierarchical planner: tactical waypoint heads) and REF-C (a DiffusionDrive-style
anchored-diffusion decoder) OWN their trajectory surface and have NO grounded
operative rollout — ``loaders.load`` returns step_readout=None for both — so
that path cannot render them. This module supplies the direct-head equivalent,
calling each arm exactly the way its scoring collector does
(``taniteval.refb_eval.collect`` / ``taniteval.refc_eval.collect``) so the ADE
burned into the HUD is the SAME quantity the leaderboard row reports.

TRAJECTORY SURFACE (honest note). Both arms emit 4 TIME waypoints at the shared
WP_STEPS 5/10/15/20 (= 0.5/1/1.5/2 s, ego frame of the last window pose) — not
a dense 20-step path like the world-model arms. For rendering, those 4
waypoints are placed at their TRUE slots of a 20-point path (indices 4/9/14/19
= the standard's WP_IDX, so every marker sits on a real model output) and the
intermediate points are straight-line interpolation between consecutive
waypoints. The interpolation is a DRAWING DEVICE ONLY: no curvature is invented
between waypoints, the ring markers are the model's actual predictions, and
every scored quantity (per-frame ADE, clip-mean ADE) is computed from the 4
waypoints alone.

Usage (eval pod):
  PYTHONPATH=/root/taniteval:/root/TanitAD/stack \
    python -m taniteval.direct_overlay --models refb-v2-30k,refc-xl-live \
      --clips 31:highspeed-straight,3:sharpturn,11:failure-worstwindow
  ... --corpus physicalai        # default (the canonical val)
  ... --thumbs                   # one mid-frame PNG per clip, no video
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import loaders                                      # noqa: E402
from taniteval.cam_overlay import ego_future_path                  # noqa: E402
from taniteval.corpus_overlay import (HORIZON, FlatProjector,      # noqa: E402
                                      clip_extent, draw_frame,
                                      pretty_man, pretty_route)
from taniteval.flagship_overlay import K, WINDOW, WP_IDX, _font    # noqa: E402
from taniteval.registry import CORPORA, MODELS                     # noqa: E402

OUT = Path("/root/taniteval/results/videos")
S_PX = 512                                # rendered frame width (256 * UP)
F_HUD, F_SUB = _font(15), _font(12)
DT = 0.1                                  # 10 Hz
WP_STEPS = (5, 10, 15, 20)                # shared eval horizons (0.5..2 s)
KNOTS = (0,) + WP_STEPS                   # + the ego origin at t=0

# Regime clips on the canonical val, chosen from the per-episode strata of
# results/windows_refb-v2-30k.pt (v_mean / max net-heading / mean+worst ADE).
DEFAULT_CLIPS = ("31:highspeed-straight,3:sharpturn,"
                 "11:failure-worstwindow,28:highspeed-curve,17:straightcruise")


def _fit(s: str, font, limit=S_PX - 16) -> str:
    """Ellipsize a HUD string that would overflow the frame. The HUD is drawn
    at x=8 with no clipping, so an over-long line silently loses its TAIL —
    which is exactly where the model/step/ADE labels live. Guard it."""
    from PIL import Image, ImageDraw
    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    if d.textlength(s, font=font) <= limit:
        return s
    while s and d.textlength(s + "…", font=font) > limit:
        s = s[:-1]
    return s + "…"


def densify(wp4: torch.Tensor) -> torch.Tensor:
    """4 time waypoints [4,2] -> a 20-point path [20,2] whose indices 4/9/14/19
    ARE those waypoints (the standard's WP_IDX) and whose in-between points are
    straight-line interpolation from the ego origin. Drawing device only."""
    knots = torch.cat([torch.zeros(1, 2, dtype=wp4.dtype), wp4], dim=0)  # [5,2]
    out = torch.empty(K, 2, dtype=wp4.dtype)
    for j in range(K):                                  # path index j = step j+1
        step = j + 1
        for a in range(len(KNOTS) - 1):
            s0, s1 = KNOTS[a], KNOTS[a + 1]
            if s0 < step <= s1:
                w = (step - s0) / (s1 - s0)
                out[j] = knots[a] * (1.0 - w) + knots[a + 1] * w
                break
    return out


@torch.no_grad()
def episode_direct(model, ep, arch, device, window, speed_input, yaw_input,
                   mode="diffusion", batch=8):
    """Stride-1 direct-head prediction + decoded intent for every frame.

    Calls the arm exactly as its scoring collector does. Returns
    t -> {wp[20,2] (densified for drawing), wp4[4,2] (the real outputs),
    ade, v0, man, route}; t = window end (the pose the ego frame is anchored
    to), matching corpus_overlay.episode_rollouts."""
    import refb_labels as rl                    # scripts/ on sys.path

    frames, poses = ep.frames, ep.poses.float()
    T = min(frames.shape[0], poses.shape[0])
    starts = list(range(0, T - window - K))
    out = {}
    for i in range(0, len(starts), batch):
        ch = starts[i:i + batch]
        last = torch.tensor([s + window - 1 for s in ch])
        fw = torch.stack([torch.as_tensor(frames[s:s + window])
                          for s in ch]).to(device).float().div_(255.0)
        v0 = poses[last, 3].to(device) if speed_input else None
        kw = {}
        if arch == "refb" and yaw_input:
            # arch-v2 (B2) yr0: BACKWARD-diff yaw-rate at t0, RAW rad/s —
            # identical to refb_eval.collect / refb_train.compute_losses.
            kw["yr0"] = (rl.wrap_to_pi(poses[last, 2] - poses[last - 1, 2])
                         / DT).to(device)
        elif arch == "refc":
            kw["steps"] = (model.cfg.decoder.diffusion_steps
                           if mode == "diffusion" else 0)
        o = model(fw, nav_cmd=None, v0=v0, **kw)          # follow-command eval
        wp = torch.stack([o["waypoints"][k] for k in WP_STEPS],
                         dim=1).cpu().float()             # [b, 4, 2]
        man = o["maneuver_logits"].argmax(-1).cpu().tolist()
        route = o["route_logits"].argmax(-1).cpu().tolist()
        for j, s in enumerate(ch):
            t = s + window - 1
            gt = ego_future_path(poses, t, K)
            ade = float(torch.linalg.norm(wp[j] - gt[WP_IDX], dim=-1).mean())
            out[t] = dict(wp=densify(wp[j]), wp4=wp[j], ade=ade,
                          v0=float(poses[t, 3]), man=man[j], route=route[j])
    return out


def render_episode(model, ep, arch, name, model_key, step, corpus, kind, proj,
                   device, fps, window, speed_input, yaw_input, mode,
                   max_frames=200, thumbs=False, surface=""):
    poses = ep.poses.float()
    preds = episode_direct(model, ep, arch, device, window, speed_input,
                           yaw_input, mode)
    ts = sorted(preds)[:max_frames]
    if not ts:
        print(f"[skip] {name}: too few frames (T={ep.frames.shape[0]})")
        return None, 0, 0.0
    ades = [preds[t]["ade"] for t in ts]
    mean_ade = sum(ades) / len(ades)
    xmax, ymax = clip_extent(preds, poses)
    cam = _fit(surface, F_SUB)
    top = _fit(f"{model_key} · step {step} · {corpus} · "
               f"GT green / pred orange · 2 s", F_SUB)
    frames_dir = OUT / f"_frames_{name}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    picks = ts if not thumbs else [ts[len(ts) // 2]]
    for n, t in enumerate(picks):
        d = preds[t]
        man, route = pretty_man(d["man"]), pretty_route(d["route"])
        gt = ego_future_path(poses, t, K)
        rgb = ep.frames[t, -3:].permute(1, 2, 0).numpy()
        l1 = _fit(f"tactical: {man}    strategic: route {route}", F_HUD)
        l2 = _fit(f"f{t:03d}   ADE {d['ade']:.2f} m   v0 {d['v0']:.1f} m/s   "
                  f"clip-mean {mean_ade:.2f} m", F_SUB)
        im = draw_frame(rgb, gt, d["wp"], proj, top, l1, l2, cam, xmax, ymax)
        if thumbs:
            pth = OUT / f"thumb_{name}_f{t:03d}.png"
            im.save(pth)
            print(f"[thumb] {pth}  ADE {d['ade']:.2f} man={man} route={route} "
                  f"clipADE {mean_ade:.2f}", flush=True)
            shutil.rmtree(frames_dir)
            return str(pth), 1, mean_ade
        im.save(frames_dir / f"f{n:04d}.png")
    mp4 = OUT / f"{name}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-r", str(fps), "-i", str(frames_dir / "f%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
         "-movflags", "+faststart", str(mp4)],
        check=True, capture_output=True)
    shutil.rmtree(frames_dir)
    return str(mp4), len(ts), mean_ade


def main():
    ap = argparse.ArgumentParser("direct_overlay")
    ap.add_argument("--corpus", default="physicalai",
                    choices=[c["key"] for c in CORPORA])
    ap.add_argument("--models", default="refb-v2-30k,refc-xl-live")
    ap.add_argument("--clips", default=DEFAULT_CLIPS)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--horizon", type=float, default=None)
    ap.add_argument("--thumbs", action="store_true")
    ap.add_argument("--max-frames", type=int, default=200)
    args = ap.parse_args()
    device = "cuda"

    corp = [c for c in CORPORA if c["key"] == args.corpus][0]
    files = sorted(Path(corp["root"]).glob("ep_*.pt"))
    assert files, f"no episodes under {corp['root']}"
    clips = [(int(a.split(":")[0]), a.split(":")[1])
             for a in args.clips.split(",") if a.strip()]
    cy = args.horizon if args.horizon is not None else HORIZON[args.corpus]
    proj = FlatProjector(cy)
    kind = "in-dist" if args.corpus == "physicalai" else "OOD"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] corpus={args.corpus} fps={args.fps} clips={clips}", flush=True)

    from tanitad.data.mixing import load_episode
    done = []
    for mk in [m.strip() for m in args.models.split(",") if m.strip()]:
        entry = [m for m in MODELS if m["key"] == mk][0]
        arch = entry["arch"]
        assert arch in ("refb", "refc"), (
            f"{mk} is arch={arch}: a grounded-rollout arm — render it with "
            "taniteval.corpus_overlay (THE standard for step_readout arms); "
            "this module is the direct-trajectory-head branch only")
        L = loaders.load(entry, device)
        assert L["step_readout"] is None, f"{mk} unexpectedly has a step_readout"
        model, step = L["model"], L["step"]
        speed_input = bool(entry.get("speed_input"))
        yaw_input = bool(entry.get("yaw_input"))
        window = int(getattr(model.cfg, "window", WINDOW)) \
            if arch == "refc" else WINDOW
        surface = ("pred = 4 wp @ 0.5/1/1.5/2 s · "
                   + ("REF-B tactical heads" if arch == "refb"
                      else f"REF-C anchored-diffusion "
                           f"({model.cfg.anchors.n_anchors} anchors)"))
        print(f"[load] {mk} step={step} arch={arch} window={window} "
              f"speed_input={speed_input} yaw_input={yaw_input}", flush=True)

        for idx, tag in clips:
            ep = load_episode(str(files[idx]), mmap=True)
            name = f"{mk}_step{step}_{args.corpus}_ep{idx:02d}_{tag}"
            mp4, nfr, mean_ade = render_episode(
                model, ep, arch, name, mk, step, args.corpus, kind, proj,
                device, args.fps, window, speed_input, yaw_input,
                entry.get("mode", "diffusion"), args.max_frames,
                thumbs=args.thumbs, surface=surface)
            print(f"[video] {name}: {mp4} frames={nfr} "
                  f"clip-meanADE={mean_ade:.3f}", flush=True)
            done.append((name, mp4, nfr, mean_ade))
        del model, L
        torch.cuda.empty_cache()

    print("DIRECT_OVERLAY_DONE", flush=True)
    for name, mp4, nfr, ade in done:
        print(f"  {name}: frames={nfr} ADE={ade:.3f} -> {mp4}")


if __name__ == "__main__":
    main()
