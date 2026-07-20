"""TanitEval — cross-corpus GT-vs-pred trajectory overlays (THE STANDARD).

Renders the grounded operative rollout (encode_window -> rollout_decode under
true actions -> step_readout -> SE(2) accumulate, k=20 / 2 s) for ANY registered
arm on ANY corpus (physicalai / comma2k19 / cosmos), each an ep_*.pt dir in the
canonical epcache contract (frames_u8 [T,9,256,256], actions [T,2],
poses [T,4]=x,y,yaw,v).

THE STANDARD (default for ALL future viz), every frame carries THREE panels:
  1. CAMERA PROJECTION  — GT (green) + pred (orange) paths projected into the
     front image via the flat-ground pinhole (cx=128, f_eff 266, cam_h 1.5).
  2. METRIC BEV INSET   — the same GT/pred in metres, top-down, ego at bottom-
     centre (calibration-INDEPENDENT honest reference).
  3. TEXT HUD           — the model's decoded TACTICAL maneuver (tactical_policy
     maneuver_logits argmax) + STRATEGIC route/goal (strategic_policy
     route_logits argmax, follow/deploy command) + per-frame ADE + v0.

CAMERA CALIBRATION (2026-07-19 per-clip fix; NO MORE GLOBAL CONSTANTS)
comma2k19 / physicalai: canonical 256 crop centred on the principal point =>
flat-ground pinhole with cx=cy=128, f_eff 266 (unchanged, correct by build).
cosmos: Cosmos-Drive-Dreams generations INHERIT the source clip's rig geometry
(PhysicalAI-AV front-wide, fx~944 / cy~755 native — all 46 val eps are rig B),
NOT the nominal 120-deg pinhole the loader assumed. The as-built cache is
therefore ~1.70x zoomed (true f_eff ~452, not 266) with the horizon at row
~217-227, not 128 (and not the old 180 eyeball hack — that was fitted with the
wrong focal). Projection now uses the EXACT per-clip chain from the source
dataset's own pinhole_intrinsic + camera pose (cam->vehicle extrinsic):
  native 1920x1080 -> generation 1280x704 (x*2/3, y=(y-12)*2/3)
  -> cache 256 (the as-run crop c=356 @ top=174/left=462, *256/356)
loaded from results/cosmos_calib.json (built by cosmos_calib_build.py; per-ep
gate stills in results/videos/_calib_exact/). An ep without a verified calib
entry gets NO camera-pane trajectory overlay (label only; the BEV pane carries
the GT-vs-pred comparison). Chunk-1 eps carry a 'GT-seg caveat' tag: their
cached poses come from the clip's chunk-0 pose segment (val-cache rebuild
flagged upstream); calibration itself is unaffected.
NOTE the honest geometry consequence of f_eff 452 at cam height ~1.32 m: ground
closer than ~18 m projects BELOW the 256 frame, so overlays live in the far
field — the BEV inset is the near-field reference.

Both arms carry trained strategic_policy + tactical_policy brains (flagship &
REF-A 4-brain), so the maneuver/route HUD is populated for both; if an arm has
no policy brains the HUD shows n/a and the geometry panels still render.

Usage (eval pod):
  PYTHONPATH=/root/taniteval:/root/TanitAD/stack \
    python -m taniteval.corpus_overlay --corpus comma
  ... --corpus cosmos                       # per-clip exact calib auto
  ... --models flagship-30k,refa-dynin-30k  # default: BOTH arms
  ... --corpus comma --clips 35:straightcruise,18:curve
  ... --corpus cosmos --thumbs              # one mid-frame PNG per clip, no video
  ... --corpus comma --horizon 130          # override flat-model horizon row
                                            # (comma/physicalai only; cosmos is
                                            # per-clip exact and ignores it)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

import torch                                                        # noqa: E402

from taniteval import data, loaders                                # noqa: E402
from taniteval.cam_overlay import (CAM_H, F_EFF, UP,               # noqa: E402
                                   ego_future_path)
from taniteval.flagship_overlay import (                          # noqa: E402
    K, WINDOW, WP_IDX, S, COL_GT, COL_PRED, HUD_BG, HUD_FG, HUD_DIM, _font)
from taniteval.registry import MODELS, CORPORA                    # noqa: E402
from taniteval.rollout import append_ego                          # noqa: E402
from tanitad.data.mixing import load_episode                      # noqa: E402
from tanitad.models.metric_dynamics import rollout_decode         # noqa: E402
from tanitad.refs.refb import MANEUVER_CLASSES, ROUTE_CLASSES     # noqa: E402

OUT = Path("/root/taniteval/results/videos")
F_HUD = _font(15)
F_SUB = _font(12)
F_TINY = _font(10)

# Flat-ground pinhole horizon row for the corpora whose 256 crop is centred on
# the principal point (correct by build). cosmos does NOT take a global row: it
# uses the per-clip exact calib below (see module docstring).
HORIZON = {"physicalai": 128.0, "comma": 128.0}
COSMOS_CALIB = Path("/root/taniteval/results/cosmos_calib.json")

DEFAULT_MODELS = "flagship-30k,refa-dynin-30k"   # v1 vs frozen-encoder REF-A

# Default clip picks per corpus (ep_idx:tag). comma = daylight diverse maneuvers;
# cosmos = the two visually-verified pinhole fits (sunny straight + foggy curve).
DEFAULT_CLIPS = {
    "comma": "35:straightcruise,18:curve,59:highspeed",
    "cosmos": "17:sunnyhighway,18:foggycurve",
    "physicalai": "3:sharpturn,17:straightcruise,28:highspeed-curve",
}
FPS = {"comma": 10, "cosmos": 5, "physicalai": 10}   # cosmos slowed (~11 frames)


class FlatProjector:
    """Flat-ground pinhole with horizon row cy (comma / physicalai — their 256
    crop is principal-point-centred so cx=cy=128, f_eff 266 hold by build)."""

    label = "flat cy"

    def __init__(self, cy, cx=128.0, f=F_EFF, cam_h=CAM_H):
        self.cy, self.cx, self.f, self.cam_h = cy, cx, f, cam_h
        self.horizon_row = cy

    def __call__(self, pts, up=UP):
        out = []
        for p in pts:
            X, Y = float(p[0]), float(p[1])
            if X < 1.2:
                continue
            out.append(((self.cx - self.f * Y / X) * up,
                        (self.cy + self.f * self.cam_h / X) * up))
        return out


class ExactProjector:
    """Cosmos per-clip EXACT chain (see module docstring): ego-frame ground
    point -> camera frame (true cam->vehicle extrinsic: fwd offset, height,
    pitch/roll) -> native pinhole (per-clip fx/fy/cx/cy) -> generation frame
    -> the as-run cache crop. Built from a cosmos_calib.json entry."""

    label = "exact per-clip"

    def __init__(self, e):
        self.R = np.array(e["R"])
        self.t = np.array(e["t"])
        self.fx, self.fy, self.cx, self.cy = e["fx"], e["fy"], e["cx"], e["cy"]
        self.vt, self.sx = e["vcrop_top"], e["sx"]
        self.c, self.top, self.left = e["crop_c"], e["crop_top"], e["crop_left"]
        self.horizon_row = e["horizon_row_256"]
        self.f_eff = e["f_eff_256"]
        self.chunk = e["chunk"]

    def __call__(self, pts, up=UP):
        out = []
        for p in np.asarray(pts, dtype=np.float64):
            pv = np.array([p[0], p[1], 0.0])          # ego frame, ground plane
            pc = self.R.T @ (pv - self.t)
            if pc[2] < 1.0:                           # behind / at the lens
                continue
            u = self.fx * pc[0] / pc[2] + self.cx
            v = self.fy * pc[1] / pc[2] + self.cy
            ug = u * self.sx
            vg = (v - self.vt) * self.sx
            out.append(((ug - self.left) * 256.0 / self.c * up,
                        (vg - self.top) * 256.0 / self.c * up))
        return out


def cosmos_projectors(files):
    """ep_*.pt path -> ExactProjector | None (None = gate-disabled, no camera
    overlay). Keyed by the ep's index in the sorted generation-mp4 order (how
    build_cosmos.py numbered the eps)."""
    table = json.loads(COSMOS_CALIB.read_text())
    out = {}
    for f in files:
        idx = int(f.stem.split("_")[1])
        e = table.get(str(idx))
        out[idx] = ExactProjector(e) if (e and e.get("calib")) else None
    return out


def pretty_man(m):
    return MANEUVER_CLASSES[m].replace("_", " ") if m is not None else "n/a"


def pretty_route(r):
    return ROUTE_CLASSES[r].replace("route_", "") if r is not None else "n/a"


def draw_bev(im, gt, pred, xmax, ymax):
    """Top-down metric BEV inset (metres) — ego at bottom-centre, forward = up,
    left = panel-left. Calibration-independent honest GT-vs-pred reference."""
    d = ImageDraw.Draw(im, "RGBA")
    bw, bh = 152, 196
    x0, y0 = S - bw - 8, 94          # below the top text HUD (rows 0-88)
    x1, y1 = x0 + bw, y0 + bh
    d.rectangle([x0, y0, x1, y1], fill=(8, 11, 15, 205),
                outline=(60, 70, 82), width=1)
    pad = 12
    cx = (x0 + x1) // 2
    by = y1 - pad
    top = y0 + pad + 8

    def m2px(X, Y):
        px = cx - (Y / ymax) * ((bw / 2) - pad)
        py = by - (max(X, 0.0) / xmax) * (by - top)
        return px, py

    step = 10 if xmax > 25 else 5
    r = step
    while r <= xmax + 0.1:
        _, py = m2px(r, 0)
        d.line([(x0 + 5, py), (x1 - 5, py)], fill=(38, 46, 56, 255))
        d.text((x0 + 6, py - 11), f"{r}m", fill=(96, 106, 120), font=F_TINY)
        r += step
    d.line([(cx, top), (cx, by)], fill=(38, 46, 56, 255))
    gpts = [m2px(float(p[0]), float(p[1])) for p in gt]
    if len(gpts) >= 2:
        d.line(gpts, fill=COL_GT, width=3)
    ppts = [m2px(float(p[0]), float(p[1])) for p in pred]
    if len(ppts) >= 2:
        d.line(ppts, fill=COL_PRED, width=2)
    for p in pred[WP_IDX]:
        x, y = m2px(float(p[0]), float(p[1]))
        d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=COL_PRED, width=2)
    d.polygon([(cx - 4, by), (cx + 4, by), (cx, by - 8)], fill=(232, 236, 242))
    d.text((x0 + 6, y0 + 3), "BEV top-down (m)", fill=HUD_DIM, font=F_TINY)


def draw_frame(rgb_hwc, gt_path, pred_path, proj, top, l1, l2, l3, xmax, ymax):
    """Camera projection (per-clip projector; None = gate-disabled) + metric
    BEV inset + 3-line text HUD."""
    im = Image.fromarray(rgb_hwc).resize((S, S), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(im)
    if proj is not None:
        hr = getattr(proj, "horizon_row", None)
        if isinstance(proj, ExactProjector) and hr is not None:
            d.line([(0, hr * UP), (S, hr * UP)],       # verified horizon
                   fill=(170, 150, 60), width=1)
        g = proj(gt_path)                             # GT wide underneath
        if len(g) >= 2:
            d.line(g, fill=COL_GT, width=7)
        p = proj(pred_path)                           # pred narrow on top
        if len(p) >= 2:
            d.line(p, fill=COL_PRED, width=3)
        for x, y in proj(gt_path[WP_IDX]):
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=COL_GT)
        for x, y in proj(pred_path[WP_IDX]):
            d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=COL_PRED, width=3)
    else:
        d.text((8, S // 2 - 8), "camera overlay disabled — calibration "
               "unverified (see BEV)", fill=(235, 180, 90), font=F_SUB)
    draw_bev(im, gt_path, pred_path, xmax, ymax)  # metric BEV inset (calib-free)
    # All text at the TOP (sky rows): with the true cosmos calib the visible
    # ground band is the BOTTOM ~33 rows — a bottom bar would cover the paths.
    d.rectangle([0, 0, S, 24], fill=HUD_BG)
    d.text((8, 5), top, fill=HUD_DIM, font=F_SUB)
    d.rectangle([0, 24, S, 88], fill=HUD_BG)
    d.text((8, 28), l1, fill=HUD_FG, font=F_HUD)       # decoded intent
    d.text((8, 48), l2, fill=HUD_DIM, font=F_SUB)      # metrics
    d.text((8, 68), l3, fill=HUD_DIM, font=F_SUB)      # camera calibration
    return im


def clip_extent(preds, poses):
    """Clip-level BEV extent (fixed across frames for temporal stability)."""
    xm, ym = 12.0, 3.0
    for t, dct in preds.items():
        gt = ego_future_path(poses, t, K)
        for arr in (dct["wp"], gt):
            xm = max(xm, float(arr[:, 0].max()))
            ym = max(ym, float(arr[:, 1].abs().max()))
    xm = min(80.0, 5 * (int(xm // 5) + 1))
    ym = max(4.0, 1.2 * ym)
    return xm, ym


@torch.no_grad()
def episode_rollouts(model, step_readout, enc_input, poses, actions, feed,
                     speed_input, dyn_input, device, batch=16):
    """Stride-1 grounded k=20 rollout + decoded intent for every frame.

    enc_input is raw frames [T,9,256,256] uint8 (flagship/refb) OR frozen
    features [T,256,d] fp16 (REF-A). Appends the canonical ego action-channels
    ([v0] speed_input, +[yr0] dyn_input) so the fed action_dim matches the ckpt.
    Returns dict t -> {wp[20,2], ade, v0, man, route}; t = window end."""
    is_feats = feed != "frames"
    T = min(enc_input.shape[0], poses.shape[0], actions.shape[0])
    starts = list(range(0, T - WINDOW - K))
    has_policy = (getattr(model, "strategic_policy", None) is not None and
                  getattr(model, "tactical_policy", None) is not None)
    out = {}
    for i in range(0, len(starts), batch):
        ch = starts[i:i + batch]
        last = torch.tensor([s + WINDOW - 1 for s in ch])
        fw = torch.stack([torch.as_tensor(enc_input[s:s + WINDOW])
                          for s in ch]).to(device).float()
        if not is_feats:
            fw = fw.div_(255.0)
        aw = torch.stack([actions[s:s + WINDOW] for s in ch]).to(device)
        fa = torch.stack([actions[s + WINDOW:s + WINDOW + K]
                          for s in ch]).to(device)
        aw, fa = append_ego(aw, fa, poses, last, speed_input, False,
                            dyn_input, device)
        states = model.encode_window(fw)                        # [b, W, S]
        wp_full, _ = rollout_decode(model.predictor, states, aw, fa,
                                    step_readout, K)             # [b, 20, 2]
        wp_full = wp_full.cpu().float()
        man_ids = route_ids = None
        if has_policy:
            follow = torch.zeros(len(ch), dtype=torch.long, device=device)
            sf = model.strategic_policy(states, follow)          # deploy command
            route_ids = sf["route_logits"].argmax(-1).cpu().tolist()
            tacf = model.tactical_policy(states, sf["ctx"])
            man_ids = tacf["maneuver_logits"].argmax(-1).cpu().tolist()
        for j, s in enumerate(ch):
            t = s + WINDOW - 1
            gt = ego_future_path(poses, t, K)
            ade = float(torch.linalg.norm(
                wp_full[j][WP_IDX] - gt[WP_IDX], dim=-1).mean())
            out[t] = dict(wp=wp_full[j], ade=ade, v0=float(poses[t, 3]),
                          man=man_ids[j] if man_ids is not None else None,
                          route=route_ids[j] if route_ids is not None else None)
    return out


def render_episode(model, sr, ep, enc_input, feed, speed_input, dyn_input,
                   name, model_key, corpus, kind, proj, device, fps,
                   max_frames=200, thumbs=False):
    poses, actions = ep.poses.float(), ep.actions.float()
    preds = episode_rollouts(model, sr, enc_input, poses, actions, feed,
                             speed_input, dyn_input, device)
    ts = sorted(preds)[:max_frames]
    if not ts:
        print(f"[skip] {name}: too few frames (T={ep.frames.shape[0]})")
        return None, 0, 0.0
    ades = [preds[t]["ade"] for t in ts]
    mean_ade = sum(ades) / len(ades)
    xmax, ymax = clip_extent(preds, poses)
    if proj is None:
        cam = "cam: overlay OFF — calib unverified (see BEV)"
    elif isinstance(proj, ExactProjector):
        cam = (f"cam: exact per-clip · f_eff {proj.f_eff:.0f} · "
               f"horizon {proj.horizon_row:.0f}")
        if proj.chunk == 1:
            cam += " · GT-seg caveat"
    else:
        cam = f"cam: flat pinhole · f_eff {F_EFF:.0f} · horizon {int(proj.horizon_row)}"
    top = f"{model_key} · {corpus} ({kind}) · GT green · pred orange · 2 s"
    frames_dir = OUT / f"_frames_{name}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    picks = ts if not thumbs else [ts[len(ts) // 2]]
    for n, t in enumerate(picks):
        dct = preds[t]
        wp, ade, v0 = dct["wp"], dct["ade"], dct["v0"]
        man, route = pretty_man(dct["man"]), pretty_route(dct["route"])
        gt = ego_future_path(poses, t, K)
        rgb = ep.frames[t, -3:].permute(1, 2, 0).numpy()
        l1 = f"tactical: {man}    strategic: route {route}"
        l2 = (f"f{t:03d}   ADE {ade:.2f} m   v0 {v0:.1f} m/s   "
              f"clip-mean {mean_ade:.2f} m")
        im = draw_frame(rgb, gt, wp, proj, top, l1, l2, cam, xmax, ymax)
        if thumbs:
            pth = OUT / f"thumb_{name}_f{t:03d}.png"
            im.save(pth)
            print(f"[thumb] {pth}  ADE {ade:.2f} man={man} route={route} "
                  f"clipADE {mean_ade:.2f}")
            shutil.rmtree(frames_dir)
            return None, 1, mean_ade
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    choices=[c["key"] for c in CORPORA])
    ap.add_argument("--models", default=DEFAULT_MODELS,
                    help="comma-separated model keys (default: flagship + REF-A)")
    ap.add_argument("--clips", default=None, help="idx:tag,idx:tag (override)")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--horizon", type=float, default=None,
                    help="override flat-model horizon row cy (comma/physicalai "
                         "only; cosmos uses per-clip exact calib)")
    ap.add_argument("--thumbs", action="store_true", help="1 mid PNG/clip, no vid")
    ap.add_argument("--max-frames", type=int, default=200)
    args = ap.parse_args()
    device = "cuda"

    corp = [c for c in CORPORA if c["key"] == args.corpus][0]
    files = sorted(Path(corp["root"]).glob("ep_*.pt"))
    assert files, f"no episodes under {corp['root']}"
    spec = args.clips or DEFAULT_CLIPS[args.corpus]
    clips = [(int(a.split(":")[0]), a.split(":")[1]) for a in spec.split(",")]
    fps = args.fps or FPS.get(args.corpus, 10)
    if args.corpus == "cosmos":
        if args.horizon is not None:
            print("[cfg] --horizon ignored for cosmos (per-clip exact calib)")
        projs = cosmos_projectors([files[idx] for idx, _ in clips])
        proj_of = lambda f: projs[int(f.stem.split("_")[1])]   # noqa: E731
    else:
        cy = args.horizon if args.horizon is not None else HORIZON[args.corpus]
        flat = FlatProjector(cy)
        proj_of = lambda f: flat                               # noqa: E731
    kind = "in-dist" if args.corpus == "physicalai" else "OOD"
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] corpus={args.corpus} fps={fps} models={model_keys} "
          f"clips={clips}", flush=True)

    done = []
    for mk in model_keys:
        entry = [m for m in MODELS if m["key"] == mk][0]
        L = loaders.load(entry, device)
        model, sr = L["model"], L["step_readout"]
        assert sr is not None, f"{mk} step_readout missing"
        feed = L["feed"]
        speed_input = bool(entry.get("speed_input"))
        dyn_input = bool(entry.get("dyn_input"))
        print(f"[load] {mk} step={L['step']} feed={feed} "
              f"speed_input={speed_input} dyn_input={dyn_input}", flush=True)

        feat_by_idx = {}
        if feed != "frames":                       # REF-A: frozen features
            clip_files = [files[idx] for idx, _ in clips]
            feateps = data.load_features(clip_files, feed, device)
            feat_by_idx = {idx: feateps[i].feats
                           for i, (idx, _) in enumerate(clips)}

        for idx, tag in clips:
            ep = load_episode(str(files[idx]), mmap=True)
            enc_input = feat_by_idx[idx] if feed != "frames" else ep.frames
            name = f"{mk}_{args.corpus}_{tag}_overlay"
            mp4, nfr, mean_ade = render_episode(
                model, sr, ep, enc_input, feed, speed_input, dyn_input,
                name, mk, args.corpus, kind, proj_of(files[idx]), device, fps,
                args.max_frames, thumbs=args.thumbs)
            print(f"[video] {name}: {mp4} frames={nfr} "
                  f"clip-meanADE={mean_ade:.3f}", flush=True)
            done.append((name, mp4, nfr, mean_ade))

    print("CORPUS_OVERLAY_DONE", flush=True)
    for name, mp4, nfr, ade in done:
        print(f"  {name}: frames={nfr} ADE={ade:.3f} -> {mp4}")


if __name__ == "__main__":
    main()
