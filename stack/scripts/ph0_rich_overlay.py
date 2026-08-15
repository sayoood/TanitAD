"""The FULL PH0 pipeline in one frame: engine A numbers, engine B symbols,
engine C pixels — rendered together so the whole extraction is inspectable.

PI 2026-08-12: *"generate the rendered video of the vlm/sam3 pipeline with all
rich information"* — and it gates production, so what is drawn has to be the
thing that would be shipped, not a flattering subset.

LAYOUT
  ┌──────────────────────────────┬──────────────┐
  │ camera 2x                    │ TEXT PANEL   │
  │  · SAM3 masks (translucent)  │  A: geometry │
  │  · SAM3 boxes + concept/score│  B: symbols  │
  │  · VLM B3 sign boxes (dashed)│  C: counts   │
  ├──────────────────────────────┤  situations  │
  │ BEV: integrated ego path     │  ego state   │
  └──────────────────────────────┴──────────────┘

⭐ THE COLOUR RULE, AND IT IS THE POINT OF THE FIGURE. Colour encodes WHICH
ENGINE produced a mark, never how confident it looks:
  · SAM3 detections     — solid, one hue per concept
  · VLM B3 sign boxes   — DASHED WHITE, always, on top
so a viewer can see at a glance where the two engines agree and where only one
of them fired. Merging them into one "detection" colour would hide exactly the
disagreement the cross-engine check exists to surface.

⚠️ WHAT THIS FIGURE MUST NOT IMPLY. A drawn box is not a validated box. The
VLM↔SAM3 sign agreement currently reads 0/8, but the cross-check snaps the VLM's
grounded frame to a strided frame up to ~3.5 s away, so that number is a
property of MY alignment, not of the VLM. The panel therefore prints the match
as `n/n (ALIGNMENT UNVERIFIED)` rather than as a score.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PANEL_W = 470
SCALE = 2
BEV_H = 210

# one hue per concept — stable across clips so the eye can track a class
CONCEPT_COL = {
    "car": (86, 156, 255), "truck": (255, 149, 61), "bus": (255, 196, 61),
    "pedestrian": (255, 92, 138), "cyclist": (170, 120, 255),
    "traffic light": (94, 224, 156), "traffic sign": (255, 233, 120),
}
VLM_COL = (255, 255, 255)
INK = (232, 236, 244)
INK_DIM = (150, 158, 172)
BG = (18, 20, 26)


def _font(sz: int):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def _dashed_rect(d, box, colour, width=2, dash=5):
    x0, y0, x1, y1 = box
    for x in range(int(x0), int(x1), dash * 2):
        d.line([x, y0, min(x + dash, x1), y0], fill=colour, width=width)
        d.line([x, y1, min(x + dash, x1), y1], fill=colour, width=width)
    for y in range(int(y0), int(y1), dash * 2):
        d.line([x0, y, x0, min(y + dash, y1)], fill=colour, width=width)
        d.line([x1, y, x1, min(y + dash, y1)], fill=colour, width=width)


def draw_camera(frame, sam_dets, vlm_boxes, *, scale=SCALE):
    """Camera pane: SAM3 masks + boxes, then the VLM's sign boxes ON TOP."""
    import numpy as np
    from PIL import Image, ImageDraw
    base = np.asarray(frame).astype(np.float32)
    over = base.copy()
    for det in sam_dets:
        col = np.array(CONCEPT_COL.get(det.get("concept"), (200, 200, 200)),
                       np.float32)
        for r, a, b in det.get("rle_rows", []) or []:
            if 0 <= r < over.shape[0]:
                over[r, a:b] = 0.45 * col + 0.55 * over[r, a:b]
    img = Image.fromarray(np.clip(over, 0, 255).astype(np.uint8)).convert("RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    f = _font(11)
    for det in sam_dets:
        b = det.get("box_xyxy")
        if not b:
            continue
        col = CONCEPT_COL.get(det.get("concept"), (200, 200, 200))
        x0, y0, x1, y1 = [v * scale for v in b]
        d.rectangle([x0, y0, x1, y1], outline=col, width=2)
        lab = f"{det['concept']} {det.get('score', 0):.2f}"
        ty = max(0, y0 - 12)
        d.rectangle([x0, ty, x0 + 7 * len(lab), ty + 12], fill=(0, 0, 0))
        d.text((x0 + 2, ty), lab, fill=col, font=f)
    # VLM boxes LAST so they are never hidden by a SAM3 mark
    for vb in vlm_boxes:
        x0, y0, x1, y1 = [v * scale for v in vb["box_xyxy"]]
        _dashed_rect(d, (x0, y0, x1, y1), VLM_COL, width=2)
        lab = f"VLM:{vb.get('label', 'sign')}"
        if vb.get("text"):
            lab += f' "{vb["text"]}"'
        d.text((x0 + 2, min(img.height - 12, y1 + 2)), lab, fill=VLM_COL,
               font=f)
    return img


def draw_bev(size, engine_a, frame_i, n_frames):
    """Integrated ego path in the t0 frame, with the current time marked."""
    from PIL import Image, ImageDraw
    W, H = size
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f = _font(11)
    d.text((8, 6), "BEV · integrated ego path (engine A)", fill=INK_DIM, font=f)
    poly = (engine_a or {}).get("polyline_xy") or []
    if not poly:
        d.text((8, H // 2), "no ego poses for this clip", fill=INK_DIM, font=f)
        return img
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    sx = (W - 60) / max(1e-6, x1 - x0)
    sy = (H - 46) / max(1e-6, y1 - y0)
    s = min(sx, sy)

    def to_px(p):
        # forward (x) runs UP the image; lateral (y) runs right-to-left
        return (W / 2 - (p[1] - (y0 + y1) / 2) * s,
                H - 16 - (p[0] - x0) * s)

    pts = [to_px(p) for p in poly]
    d.line(pts, fill=(86, 156, 255), width=2)
    t0i = int((engine_a or {}).get("t0_idx", 0))
    if 0 <= t0i < len(pts):
        px, py = pts[t0i]
        d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 233, 120))
        d.text((px + 7, py - 6), "t0", fill=(255, 233, 120), font=f)
    # where we are in the clip
    if n_frames > 1:
        k = int(len(pts) * frame_i / max(1, n_frames - 1))
        k = max(0, min(k, len(pts) - 1))
        px, py = pts[k]
        d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 92, 138))
    return img


def _wrap(d, text, font, maxw):
    out, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if d.textlength(t, font=font) <= maxw:
            line = t
        else:
            if line:
                out.append(line)
            line = w
    if line:
        out.append(line)
    return out


def draw_panel(size, rec, sam_clip, engine_a, frame_i):
    """Everything the three engines extracted, as text."""
    from PIL import Image, ImageDraw
    W, H = size
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    fh, fb, fs = _font(13), _font(12), _font(11)
    y = 8

    def hdr(t, col=(120, 200, 255)):
        nonlocal y
        d.text((10, y), t, fill=col, font=fh)
        y += 17

    def row(t, col=INK, font=None):
        nonlocal y
        for ln in _wrap(d, t, font or fb, W - 24):
            if y > H - 14:
                return
            d.text((14, y), ln, fill=col, font=font or fb)
            y += 14

    sc = rec.get("scene") or {}
    hdr("ENGINE B — VLM (Qwen3.5-9B, grammar-constrained)")
    row(f"scene   {sc.get('illumination','?')} · {sc.get('weather','?')} · "
        f"{sc.get('road_type','?')} / {sc.get('domain','?')}")
    row(f"lanes   visible {sc.get('lanes_visible','?')} · ego lane "
        f"{sc.get('lane_ego','?')} · conf {sc.get('conf','?')}")
    signs = (rec.get("signs") or {}).get("signs") or []
    row(f"signs   n={len(signs)}", INK_DIM)
    for i, s in enumerate(signs[:4]):
        txt = f'  [{i}] {s.get("kind")}'
        if s.get("text"):
            txt += f' "{s["text"]}"'
        if s.get("state") and s["state"] != "none":
            txt += f' ({s["state"]})'
        txt += "  ego" if s.get("applies_to_ego") else ""
        row(txt, (255, 233, 120), fs)
    sym = rec.get("symbols") or {}
    row(f"GOAL    {sym.get('goal_kind','—')}"
        + (f"  ← sign {sym['goal_evidence_sign']}"
           if sym.get("goal_evidence_sign") is not None else ""),
        (150, 255, 190))
    acts = ", ".join(f"{a.get('verb')}"
                     + (f"/{a['direction']}" if a.get("direction") not in
                        (None, "none") else "")
                     for a in (sym.get("actions") or [])) or "—"
    row(f"actions {acts}", (150, 255, 190))
    y += 5

    hdr("ENGINE A — geometry (integrated ego path)", (255, 190, 120))
    r = (engine_a or {}).get("route", {})
    spd = (engine_a or {}).get("speed_profile", {})
    row(f"route   {r.get('token','—')} (valid {r.get('token_valid','—')}) · "
        f"dist {r.get('dist_m','—')} m")
    row(f"dyaw    {r.get('maneuver_dyaw_rad','—')} rad · arc "
        f"{r.get('arc_m','—')} m")
    row(f"speed   v@t0 {spd.get('v_t0_ms','—')} · min {spd.get('v_min_future_ms','—')}"
        f" · max {spd.get('v_max_future_ms','—')} m/s")
    row(f"net dv  {spd.get('net_dv_ms','—')} m/s · stops {spd.get('stops','—')}")
    es = rec.get("ego_state") or {}
    if es:
        row(f"ego@t0  {es.get('v_now_kmh','—')} km/h · {es.get('motion','—')} · "
            f"{es.get('turning','—')}", INK)
    sit = (engine_a or {}).get("situations") or {}
    if sit and "null_reason" not in sit:
        on = [k for k in ("lane_change", "intersection", "roundabout")
              if sit.get(k)]
        row(f"situation {', '.join(on) if on else 'none detected'}",
            (255, 190, 120))
    y += 5

    hdr("ENGINE C — SAM3 (independent detection)", (150, 255, 190))
    hits = (sam_clip or {}).get("per_concept_hits", {})
    live = {k: v for k, v in hits.items() if v}
    if live:
        for k, v in sorted(live.items(), key=lambda kv: -kv[1]):
            col = CONCEPT_COL.get(k, INK)
            row(f"  {k:<14s} {v}", col, fs)
    else:
        row("  no detections on the sampled frames", INK_DIM, fs)
    xc = (sam_clip or {}).get("vlm_cross_check") or []
    n_m = sum(1 for c in xc if c.get("matched"))
    row(f"VLM↔SAM3 sign match {n_m}/{len(xc)}", INK_DIM)
    row("⚠ ALIGNMENT UNVERIFIED — the cross-check snaps the VLM frame to a "
        "strided frame up to ~3.5 s away, so this is not yet evidence about "
        "the VLM's boxes.", (255, 149, 61), fs)

    d.text((10, H - 15), f"frame {frame_i}   white dashed = VLM · "
           f"solid = SAM3", fill=INK_DIM, font=fs)
    return img


def clip_layout(rec, sam_clip, frames):
    """Everything render_clip needs BEFORE any encoder is touched: the canvas
    geometry, the sorted SAM3 frame keys, and the VLM boxes in pixels.

    ⛔ SPLIT OUT ON PURPOSE. The first pod run died here on
    ``sorted(..., default=None)`` while a smoke test of the three drawing
    helpers passed — the risky part is the ASSEMBLY, not the drawing, and the
    assembly was previously reachable only through a video encoder that is not
    installed everywhere. Now it is a pure function with a unit test."""
    fh, fw = frames[0].shape[0], frames[0].shape[1]
    cam_w, cam_h = fw * SCALE, fh * SCALE
    sam_frames = (sam_clip or {}).get("frames") or {}
    keys = sorted(int(k) for k in sam_frames)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ph0_v2 import norm_to_px
    signs = (rec.get("signs") or {}).get("signs") or []
    vlm_boxes = []
    for i, g in enumerate(rec.get("grounding") or []):
        if not g or not g.get("visible") or not g.get("bbox"):
            continue
        vlm_boxes.append({
            "box_xyxy": norm_to_px(g["bbox"], fw, fh),
            "frame_idx": int(g.get("frame_idx", 0)),
            "label": signs[i].get("kind", "sign") if i < len(signs) else "sign",
            "text": signs[i].get("text", "") if i < len(signs) else ""})
    return {"cam_w": cam_w, "cam_h": cam_h, "W": cam_w + PANEL_W,
            "H": cam_h + BEV_H, "sam_frames": sam_frames, "keys": keys,
            "vlm_boxes": vlm_boxes}


def dets_for_frame(lay, fi, max_gap=4):
    """The nearest sampled SAM3 frame, or [] — the gap is BOUNDED and stated
    rather than silently interpolated across seconds of driving."""
    if not lay["keys"]:
        return []
    nk = min(lay["keys"], key=lambda k: abs(k - fi))
    if abs(nk - fi) > max_gap:
        return []
    return [d for d in lay["sam_frames"][str(nk)].get("det", [])
            if "score" in d]


def compose_frame(rec, sam_clip, frames, engine_a, lay, fi):
    """One fully annotated canvas. No encoder involved."""
    from PIL import Image
    cam = draw_camera(frames[fi], dets_for_frame(lay, fi),
                      [v for v in lay["vlm_boxes"]
                       if abs(v["frame_idx"] - fi) <= 1])
    bev = draw_bev((lay["cam_w"], BEV_H), engine_a, fi, len(frames))
    pan = draw_panel((PANEL_W, lay["H"]), rec, sam_clip, engine_a, fi)
    canvas = Image.new("RGB", (lay["W"], lay["H"]), BG)
    canvas.paste(cam, (0, 0))
    canvas.paste(bev, (0, lay["cam_h"]))
    canvas.paste(pan, (lay["cam_w"], 0))
    return canvas


def render_clip(rec, sam_clip, frames, engine_a, out_path, fps=4):
    """One clip -> one mp4 with every frame fully annotated."""
    import imageio.v2 as imageio
    fh, fw = frames[0].shape[0], frames[0].shape[1]
    lay = clip_layout(rec, sam_clip, frames)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                quality=8, macro_block_size=1)
    import numpy as np
    try:
        for fi in range(len(frames)):
            writer.append_data(np.asarray(
                compose_frame(rec, sam_clip, frames, engine_a, lay, fi)))
    finally:
        writer.close()
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("ph0_rich_overlay")
    ap.add_argument("--v2-json", required=True)
    ap.add_argument("--sam3-json", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--ego-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--fps", type=int, default=4)
    a = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ph0_pilot import (POSE_HZ, engine_a_summary, load_ego_poses,
                           sample_clip_frames)

    os.makedirs(a.out, exist_ok=True)
    v2 = json.load(open(a.v2_json))
    s3 = json.load(open(a.sam3_json))
    sam_by_clip = {c.get("clip_id"): c for c in s3.get("clips", [])}

    made = []
    for rec in v2.get("clips", [])[:a.n]:
        cid = rec.get("clip_id")
        if not cid or rec.get("fatal"):
            continue
        vp = os.path.join(a.video_root, f"{cid}.mp4")
        if not os.path.exists(vp):
            print(f"[rich] {str(cid)[:8]} NO VIDEO", flush=True)
            continue
        frames, _t, _n = sample_clip_frames(vp, t0_s=8.0)
        ea = None
        if a.ego_root:
            try:
                poses = load_ego_poses(cid, a.ego_root)
                if poses is not None:
                    ea = engine_a_summary(poses, int(round(8.0 * POSE_HZ)))
            except Exception as e:
                print(f"[rich] engine A {str(cid)[:8]}: "
                      f"{type(e).__name__}: {e}", flush=True)
        out = os.path.join(a.out, f"{str(cid)[:8]}_rich.mp4")
        render_clip(rec, sam_by_clip.get(cid), frames, ea, out, fps=a.fps)
        made.append(out)
        nd = sum(f.get("n_det", 0) for f in
                 (sam_by_clip.get(cid, {}).get("frames") or {}).values())
        print(f"[rich] {str(cid)[:8]} -> {os.path.basename(out)} "
              f"({len(frames)} frames, {nd} sam3 det)", flush=True)

    print(f"[rich] {len(made)} clips rendered", flush=True)
    print("RICH_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
