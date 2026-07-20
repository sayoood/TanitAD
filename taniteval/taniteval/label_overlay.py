"""TanitEval — LABEL-VALIDATION overlay videos (old vs new vs VLM vs model).

Sayed asked to see the route-label fix, not read about it: "validate it by
creating numerous videos with the strategic and tactical labels as overlays".
This renders exactly that. Per frame the HUD carries FOUR readings of the same
window, stacked so a wrong label is visible at a glance:

    OLD  kinematic v2   (scripts/refb_labels.route_from_future)
    NEW  kinematic v2.1 (route_from_future_v21 — adaptive horizon, never
         defaults to straight, uses net heading)
    VLM  Cosmos-Reason2-8B PASS A — future FRAMES only, no numeric future track,
         so it is independent evidence rather than an echo of our kinematics
    PRED the model's own decoded tactical/route argmax (optional; --model)

VISUAL CONTRACT. Same standard as ``corpus_overlay`` / ``direct_overlay``:
camera projection + metric BEV inset + text HUD, drawn with corpus_overlay's own
``FlatProjector`` / ``draw_bev`` primitives so the picture is literally the same
surface. The ONE departure is HUD HEIGHT: the standard's 3-line band cannot hold
four label sources plus the VLM's free-text evidence, so this module draws a
taller band and the paths are pushed down accordingly. Everything else — colours,
projection, BEV metric grid, GT green / pred orange — is imported, not restated.

Rows DISAGREE in colour: a label source that differs from the NEW kinematic
reading is drawn amber, and a source that says `straight` while the future turns
by more than FALSE_TURN_DEG is drawn red. Those are the two failure modes the
audit counts, made visible frame by frame.

Usage (pod3):
  PYTHONPATH=/root/taniteval:/workspace/TanitAD/stack \
    python -m taniteval.label_overlay \
      --val /workspace/pai_epcache/physicalai-val-f1b378f295ae \
      --vlm /workspace/vlm_passA --out /root/taniteval/results/videos \
      --clips 69:widedrift-479m,3:sharpturn,17:straightcruise
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

for p in ("/root/taniteval", "/workspace/TanitAD/stack",
          "/workspace/TanitAD/stack/scripts", "/root/TanitAD/stack",
          "/root/TanitAD/stack/scripts"):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from PIL import Image, ImageDraw                                  # noqa: E402

import refb_labels as R                                           # noqa: E402
from taniteval.cam_overlay import ego_future_path                 # noqa: E402
from taniteval.corpus_overlay import (COL_GT, COL_PRED, F_HUD,    # noqa: E402
                                      F_SUB, F_TINY, HUD_BG,
                                      HUD_DIM, HUD_FG, S, UP,
                                      FlatProjector, draw_bev,
                                      pretty_man)
from taniteval.flagship_overlay import K, WP_IDX                  # noqa: E402

FALSE_TURN_DEG = 30.0
DT = 0.1
COL_BAD = (232, 96, 88)          # says straight through a real turn
COL_WARN = (232, 176, 80)        # disagrees with the new kinematic reading
COL_OK = (120, 210, 150)
HUD_H = 176                      # taller than the 3-line standard (88): 6 rows
ROUTE_NAME = {R.ROUTE_LEFT: "LEFT", R.ROUTE_STRAIGHT: "straight",
              R.ROUTE_RIGHT: "RIGHT", R.ROUTE_UNKNOWN: "UNKNOWN"}
VLM_TO_KIN = {"left": R.ROUTE_LEFT, "straight": R.ROUTE_STRAIGHT,
              "right": R.ROUTE_RIGHT, "unknown": R.ROUTE_UNKNOWN,
              "u_turn": R.ROUTE_UNKNOWN}


def _fit(s, font, limit=S - 16):
    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    if d.textlength(s, font=font) <= limit:
        return s
    while s and d.textlength(s + "…", font=font) > limit:
        s = s[:-1]
    return s + "…"


def load_vlm(vlm_dir: str, ep_name: str) -> dict:
    """anchor t -> pass-A record. Pass B is ignored here on purpose: it saw our
    numeric future track, so putting it on screen next to the kinematic rows
    would show agreement that means nothing."""
    out = {}
    for f in sorted(glob.glob(os.path.join(vlm_dir, f"{ep_name}_t*.json"))):
        try:
            rec = json.load(open(f))
        except Exception:
            continue
        a = rec.get("pass_A")
        if a:
            out[int(rec["t"])] = a
    return out


def nearest_vlm(vlm: dict, t: int, max_gap: int = 40):
    """The VLM was asked every `stride` frames; a frame between anchors shows
    the nearest anchor AND how far away it is — never silently attributed."""
    if not vlm:
        return None, None
    best = min(vlm, key=lambda a: abs(a - t))
    return (vlm[best], best) if abs(best - t) <= max_gap else (None, None)


def draw_labels_frame(rgb, gt, pred, proj, rows, xmax, ymax):
    """corpus_overlay's picture with a taller HUD. `rows` is a list of
    (text, colour, font)."""
    im = Image.fromarray(rgb).resize((S, S), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(im)
    if proj is not None:
        g = proj(gt)
        if len(g) >= 2:
            d.line(g, fill=COL_GT, width=7)
        for x, y in proj(gt[WP_IDX]):
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=COL_GT)
        if pred is not None:
            p = proj(pred)
            if len(p) >= 2:
                d.line(p, fill=COL_PRED, width=3)
            for x, y in proj(pred[WP_IDX]):
                d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=COL_PRED,
                          width=3)
    draw_bev(im, gt, pred if pred is not None else gt, xmax, ymax)
    d.rectangle([0, 0, S, HUD_H], fill=HUD_BG)
    y = 5
    for text, col, font in rows:
        d.text((8, y), _fit(text, font), fill=col, font=font)
        y += 20 if font is F_HUD else 17
    return im


@torch.no_grad()
def model_preds(model, ep, device, window, batch=8):
    """REF-C direct anchored-decoder predictions, called exactly as
    taniteval.refc_eval.collect does, so the ADE on screen is the leaderboard
    quantity."""
    frames = ep["frames_u8"]
    poses = ep["poses"].float()
    T = min(frames.shape[0], poses.shape[0])
    starts = list(range(0, max(T - window - K, 0)))
    out = {}
    for i in range(0, len(starts), batch):
        ch = starts[i:i + batch]
        last = torch.tensor([s + window - 1 for s in ch])
        fw = torch.stack([frames[s:s + window] for s in ch]) \
            .to(device).float().div_(255.0)
        v0 = poses[last, 3].to(device)
        o = model(fw, nav_cmd=None, v0=v0,
                  steps=model.cfg.decoder.diffusion_steps)
        wp = torch.stack([o["waypoints"][k] for k in (5, 10, 15, 20)],
                         dim=1).cpu().float()
        man = o["maneuver_logits"].argmax(-1).cpu().tolist()
        rt = o["route_logits"].argmax(-1).cpu().tolist()
        for j, s in enumerate(ch):
            t = s + window - 1
            gt = ego_future_path(poses, t, K)
            ade = float(torch.linalg.norm(wp[j] - gt[WP_IDX], dim=-1).mean())
            knots = torch.cat([torch.zeros(1, 2), wp[j]], 0)
            dense = torch.empty(K, 2)
            steps = (0, 5, 10, 15, 20)
            for q in range(K):
                st = q + 1
                for a in range(4):
                    if steps[a] < st <= steps[a + 1]:
                        w = (st - steps[a]) / (steps[a + 1] - steps[a])
                        dense[q] = knots[a] * (1 - w) + knots[a + 1] * w
                        break
            out[t] = dict(wp=dense, ade=ade, man=man[j], route=rt[j])
    return out


def render(ep_path, tag, idx, vlm_dir, out_dir, proj, fps, model, device,
           window, max_frames, stride_render):
    ep = torch.load(ep_path, map_location="cpu", weights_only=False)
    poses = ep["poses"].float()
    frames = ep["frames_u8"]
    T = min(frames.shape[0], poses.shape[0])
    ep_name = Path(ep_path).stem
    vlm = load_vlm(vlm_dir, ep_name) if vlm_dir else {}
    preds = model_preds(model, ep, device, window) if model is not None else {}

    # Render only frames that have the FULL 2 s GT path: ego_future_path
    # truncates near the clip end and the WP_IDX markers (step 20) would index
    # past it. This still reaches deep into the clip tail — the late-clip
    # windows v2 could not label are exactly what we are here to show — it just
    # stops 2 s short of the final frame.
    ts = list(range(0, min(T - 1 - K, max_frames), stride_render))
    if not ts:
        print(f"[skip] {Path(ep_path).stem}: T={T} too short", flush=True)
        return None
    xmax, ymax = 12.0, 3.0
    cache = {}
    for t in ts:
        gt = ego_future_path(poses, t, K)
        r2 = R.route_from_future(poses, t)
        r21 = R.route_from_future_v21(poses, t)
        cache[t] = (gt, r2, r21)
        xmax = max(xmax, float(gt[:, 0].max()))
        ymax = max(ymax, float(gt[:, 1].abs().max()))
        if t in preds:
            xmax = max(xmax, float(preds[t]["wp"][:, 0].max()))
            ymax = max(ymax, float(preds[t]["wp"][:, 1].abs().max()))

    name = f"labels_{ep_name}_{tag}"
    fdir = Path(out_dir) / f"_frames_{name}"
    fdir.mkdir(parents=True, exist_ok=True)
    n_false_old = n_false_new = n_masked_old = 0
    for n, t in enumerate(ts):
        gt, r2, r21 = cache[t]
        deg = math.degrees(r21["net_dyaw"])
        real_turn = abs(deg) >= FALSE_TURN_DEG
        man = R.window_maneuver_labels_v2(
            poses[t:t + 1], poses[t + 1:t + 1 + R.LABEL_HORIZON].unsqueeze(0)
        )[0].item() if t + R.LABEL_HORIZON < T else None

        old_bad = real_turn and r2["route"] == R.ROUTE_STRAIGHT
        n_false_old += int(old_bad and r2["valid"])
        n_masked_old += int(not r2["valid"])
        new_bad = real_turn and r21["route"] == R.ROUTE_STRAIGHT and r21["valid"]
        n_false_new += int(new_bad)

        va, anchor = nearest_vlm(vlm, t)
        vroute = VLM_TO_KIN.get((va or {}).get("ROUTE", ""), None)
        v_dis = vroute is not None and r21["valid"] and vroute != r21["route"]

        p = preds.get(t)
        rows = [
            (f"{ep_name} · {tag} · f{t:03d}/{T}  ·  GT green / pred orange · "
             f"2 s  ·  ROUTE LABEL VALIDATION", HUD_DIM, F_SUB),
            (f"TACTICAL (kin v2): {pretty_man(man)}"
             + (f"    MODEL: {pretty_man(p['man'])}  ADE {p['ade']:.2f} m"
                if p else ""), HUD_FG, F_HUD),
            (f"OLD  route v2  : {ROUTE_NAME[r2['route']]:9s}"
             f" [{'valid' if r2['valid'] else 'MASKED->emitted straight'}]",
             COL_BAD if old_bad else (COL_WARN if not r2["valid"] else COL_OK),
             F_SUB),
            (f"NEW  route v2.1: {ROUTE_NAME[r21['route']]:9s}"
             f" [{'valid' if r21['valid'] else 'masked'}·{r21['reason']}]"
             f"  net {deg:+.0f}°  arc {r21['arc_m']:.0f} m  h {r21['h_steps']}",
             COL_BAD if new_bad else COL_OK, F_SUB),
            (f"VLM  Reason2-8B (pass A, future FRAMES only): "
             + (f"{va['ROUTE']}  conf {va.get('route_confidence')}"
                f"  @f{anchor}" if va else "no anchor within 4 s"),
             COL_WARN if v_dis else (COL_OK if va else HUD_DIM), F_SUB),
            (f"VLM evidence: {(va or {}).get('route_evidence', '-')}",
             HUD_DIM, F_TINY),
            (f"model route argmax: "
             + (["left", "straight", "right"][p["route"]] if p else "n/a")
             + f"   |   red = says straight through a >={FALSE_TURN_DEG:.0f}° "
               f"turn   amber = disagrees with v2.1", HUD_DIM, F_TINY),
        ]
        im = draw_labels_frame(frames[t, -3:].permute(1, 2, 0).numpy(), gt,
                               p["wp"] if p else None, proj, rows, xmax, ymax)
        im.save(fdir / f"f{n:04d}.png")

    mp4 = Path(out_dir) / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-r", str(fps), "-i",
                    str(fdir / "f%04d.png"), "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-crf", "21", "-movflags", "+faststart",
                    str(mp4)], check=True, capture_output=True)
    shutil.rmtree(fdir)
    stat = {"clip": name, "mp4": str(mp4), "frames": len(ts),
            "old_false_straight": n_false_old, "old_masked": n_masked_old,
            "new_false_straight": n_false_new, "vlm_anchors": len(vlm)}
    print(f"[video] {name}: frames={len(ts)} old_false_straight={n_false_old} "
          f"old_masked={n_masked_old} new_false_straight={n_false_new} "
          f"-> {mp4}", flush=True)
    return stat


def main():
    ap = argparse.ArgumentParser("label_overlay")
    ap.add_argument("--val", default="/workspace/pai_epcache/"
                                     "physicalai-val-f1b378f295ae")
    ap.add_argument("--vlm", default="/workspace/vlm_passA")
    ap.add_argument("--out", default="/root/taniteval/results/videos")
    ap.add_argument("--clips", required=True,
                    help="idx:tag,idx:tag — episode index in the sorted val dir")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--horizon", type=float, default=128.0)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--stride-render", type=int, default=1)
    ap.add_argument("--model", default=None,
                    help="REF-C run dir (ckpt.pt + config.json); omit to render "
                         "labels only")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.val, "ep_*.pt")))
    assert files, f"no episodes under {args.val}"
    os.makedirs(args.out, exist_ok=True)
    proj = FlatProjector(args.horizon)

    model, window = None, 8
    if args.model:
        from tanitad.refs.refc import RefCModel, refc_xl_config
        cfg = refc_xl_config()
        cj = Path(args.model) / "config.json"
        if cj.exists():
            from taniteval.loaders import _apply_overrides
            _apply_overrides(cfg, json.loads(cj.read_text()).get("cfg", {}))
        ck = torch.load(Path(args.model) / "ckpt.pt", map_location="cpu",
                        weights_only=False)
        model = RefCModel(cfg)
        model.load_state_dict(ck["model"])
        model = model.to(args.device).eval()
        window = int(getattr(cfg, "window", 8))
        print(f"[model] REF-C from {args.model} step={ck.get('step')} "
              f"window={window}", flush=True)

    stats = []
    for spec in args.clips.split(","):
        if not spec.strip():
            continue
        idx, tag = spec.split(":")
        s = render(files[int(idx)], tag, int(idx), args.vlm, args.out,
                   proj, args.fps, model, args.device, window,
                   args.max_frames, args.stride_render)
        if s:
            stats.append(s)
    with open(Path(args.out) / "label_overlay_manifest.json", "w") as f:
        json.dump(stats, f, indent=1)
    print("LABEL_OVERLAY_DONE", flush=True)
    for s in stats:
        print(f"  {s['clip']}: {s['frames']} frames  old_false={s['old_false_straight']}"
              f" old_masked={s['old_masked']} new_false={s['new_false_straight']}")


if __name__ == "__main__":
    main()
