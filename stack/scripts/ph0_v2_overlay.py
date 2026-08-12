"""PH0 v2 result visualisation — camera overlay │ BEV │ text, as MP4 + stills.

Renders what the pipeline actually extracted, so a reader can check the claims
against the pixels instead of against a pass rate:

  LEFT   camera frame with grounded sign boxes (B3), each labelled with its
         kind + verbatim OCR text (B2). Box colour encodes VALIDATION STATUS,
         not identity — a box that failed `validate_v2` is drawn dashed red so
         a bad grounding can never look like a good one.
  MIDDLE bird's-eye view from Engine A's integrated ego path: the realised
         polyline, the t0 marker, and the ±1.5 m band. ⚠️ The band is a
         HAND-DEFINED assumption, not a perceived lane — PhysicalAI-AV ships no
         map data — so it is drawn dashed and labelled, same rule as the LF0
         figure.
  RIGHT  the symbols: B1 scene, B4 goal + actions, and Engine A's MEASURED
         numbers beside them. The split is the point — the VLM chose the
         symbols, the algorithm supplied every metre and second.

CPU only; heavy imports are lazy so the module imports for tests without PIL.
"""
from __future__ import annotations

import argparse
import json
import os

# status colours (reserved — never reused for identity)
C_OK = (42, 120, 214)        # a grounded, validated box
C_BAD = (214, 69, 45)        # failed validation
C_PATH = (42, 120, 214)
C_BAND = (150, 150, 150)
C_INK = (26, 26, 26)
C_MUTE = (110, 110, 110)
C_BG = (252, 252, 251)
PANEL_W = 430
BEV_W = 300
BEV_FWD_M = 60.0
BEV_HALF_M = 16.0


def _font(sz: int):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def _dashed_rect(d, box, colour, width=2, dash=6):
    x0, y0, x1, y1 = box
    for x in range(int(x0), int(x1), dash * 2):
        d.line([x, y0, min(x + dash, x1), y0], fill=colour, width=width)
        d.line([x, y1, min(x + dash, x1), y1], fill=colour, width=width)
    for y in range(int(y0), int(y1), dash * 2):
        d.line([x0, y, x0, min(y + dash, y1)], fill=colour, width=width)
        d.line([x1, y, x1, min(y + dash, y1)], fill=colour, width=width)


def draw_boxes(img, boxes: list[dict]):
    """boxes: [{bbox, label, ok}] in the frame's own pixel coordinates."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    f = _font(13)
    for b in boxes:
        x0, y0, x1, y1 = [int(v) for v in b["bbox"]]
        col = C_OK if b.get("ok") else C_BAD
        if b.get("ok"):
            d.rectangle([x0, y0, x1, y1], outline=col, width=2)
        else:
            _dashed_rect(d, (x0, y0, x1, y1), col, 2)
        lab = b.get("label", "")
        if lab:
            tw = d.textlength(lab, font=f)
            ly = max(0, y0 - 17)
            d.rectangle([x0, ly, x0 + tw + 8, ly + 16], fill=col)
            d.text((x0 + 4, ly + 2), lab, fill=(255, 255, 255), font=f)
    return img


def draw_bev(size, engine_a: dict | None):
    """Engine A's realised path, ego at bottom centre, forward up."""
    from PIL import Image, ImageDraw
    w, h = size
    img = Image.new("RGB", (w, h), C_BG)
    d = ImageDraw.Draw(img)
    f = _font(11)

    def xy(fwd_m, lat_m):
        # +lat is LEFT, so screen x decreases with lat (matches bev_raster)
        return (w / 2 - lat_m / BEV_HALF_M * (w / 2),
                h - fwd_m / BEV_FWD_M * h)

    # the ASSUMED corridor band — dashed, never a solid lane
    for s in (-1.5, 1.5):
        x = xy(0, s)[0]
        for y in range(0, h, 10):
            d.line([x, y, x, min(y + 5, h)], fill=C_BAND, width=1)
    d.text((4, 4), "BEV — assumed ±1.5 m band (no map data)", fill=C_MUTE,
           font=f)

    poly = (engine_a or {}).get("polyline_xy") or []
    pts = [xy(p[0], p[1]) for p in poly
           if -BEV_HALF_M <= p[1] <= BEV_HALF_M and 0 <= p[0] <= BEV_FWD_M]
    if len(pts) > 1:
        d.line(pts, fill=C_PATH, width=3)
    for m in (10, 20, 30, 40, 50):
        y = xy(m, 0)[1]
        d.line([0, y, 6, y], fill=C_MUTE, width=1)
        d.text((8, y - 6), f"{m}m", fill=C_MUTE, font=f)
    ex, ey = xy(0, 0)
    d.polygon([(ex - 5, ey), (ex + 5, ey), (ex, ey - 10)], fill=C_INK)
    if not pts:
        d.text((4, h // 2), "no Engine A polyline", fill=C_MUTE, font=f)
    return img


def _wrap(d, text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if d.textlength(t, font=font) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def draw_panel(size, rec: dict, engine_a: dict | None, frame_i: int):
    """Symbols on the left of the divide, MEASURED numbers on the right."""
    from PIL import Image, ImageDraw
    w, h = size
    img = Image.new("RGB", (w, h), C_BG)
    d = ImageDraw.Draw(img)
    fb, f, fs = _font(15), _font(13), _font(11)
    y = 8

    def line(txt, font=f, col=C_INK, dy=17):
        nonlocal y
        for ln in _wrap(d, txt, font, w - 16):
            d.text((8, y), ln, fill=col, font=font)
            y += dy

    cid = str(rec.get("clip_id", ""))[:8]
    ok = rec.get("_all_valid")
    line(f"clip {cid}   frame {frame_i}", fb)
    d.rectangle([8, y, 8 + 130, y + 15],
                fill=C_OK if ok else C_BAD)
    d.text((12, y + 1), "ALL CALLS VALID" if ok else "HAS VIOLATIONS",
           fill=(255, 255, 255), font=fs)
    y += 24

    sc = rec.get("scene") or {}
    line("SCENE  (VLM symbols)", fb, C_INK)
    if sc:
        line(f"{sc.get('illumination','?')} · {sc.get('weather','?')} · "
             f"{sc.get('road_type','?')} · {sc.get('domain','?')}", f, C_MUTE)
        line(f"lanes {sc.get('lanes_visible','?')} · ego lane "
             f"{sc.get('lane_ego','?')} · conf {sc.get('conf','?')}", f, C_MUTE)
    else:
        line("unavailable", f, C_BAD)
    y += 6

    sg = (rec.get("signs") or {}).get("signs") or []
    line(f"SIGNS  n={len(sg)}", fb)
    if not sg:
        line("none legible (abstained)", f, C_MUTE)
    for i, s in enumerate(sg[:6]):
        txt = s.get("text") or "—"
        line(f"[{i}] {s.get('kind','?')} \"{txt}\" "
             f"{'ego' if s.get('applies_to_ego') else 'other'}", fs, C_MUTE, 15)
    y += 6

    sy = rec.get("symbols") or {}
    line("GOAL + ACTIONS  (VLM symbols)", fb)
    if sy:
        ev = sy.get("goal_evidence_sign")
        line(f"goal: {sy.get('goal_kind','?')}"
             + (f"  (sign {ev})" if ev is not None else ""), f, C_INK)
        for act in sy.get("actions", []):
            dirn = act.get("direction")
            line(f"  · {act.get('verb','?')}"
                 + (f" {dirn}" if dirn and dirn != "none" else ""), f, C_MUTE)
        line(f"conf {sy.get('conf','?')}", fs, C_MUTE, 15)
    else:
        line("unavailable", f, C_BAD)
    y += 8

    d.line([8, y, w - 8, y], fill=(210, 210, 210), width=1)
    y += 8
    line("ENGINE A  (MEASURED, not generated)", fb)
    ea = engine_a or {}
    r, sp = ea.get("route", {}), ea.get("speed_profile", {})
    if r or sp:
        line(f"route {r.get('token','?')}  arc {r.get('arc_m','?')} m", f,
             C_MUTE)
        line(f"dyaw {r.get('maneuver_dyaw_rad','?')} rad · "
             f"kappa {ea.get('peak_kappa_per_m','?')}", fs, C_MUTE, 15)
        line(f"v {sp.get('v_min_future_ms','?')}–"
             f"{sp.get('v_max_future_ms','?')} m/s · "
             f"dv {sp.get('net_dv_ms','?')}", fs, C_MUTE, 15)
        line(f"stops {sp.get('stops','?')}", fs, C_MUTE, 15)
    else:
        line("unavailable", f, C_BAD)

    viol = [v for c in rec.get("_calls", []) for v in (c.get("violations") or [])]
    if viol:
        y += 6
        line("VIOLATIONS", fb, C_BAD)
        for v in viol[:4]:
            line("· " + v, fs, C_BAD, 15)
    return img


def render_clip(rec: dict, frames: list, engine_a: dict | None,
                out_dir: str, *, fps: int = 2):
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    cid = str(rec.get("clip_id", "clip"))[:8]

    # group grounded boxes by the frame they were reported in
    by_frame: dict[int, list] = {}
    signs = (rec.get("signs") or {}).get("signs") or []
    # ⛔ B3 returns NORMALIZED 0–1000 (Qwen-VL's convention). Drawing those as
    # pixels puts every box off-frame — which is exactly how the coordinate-space
    # bug looked before it was understood. Prefer the converted grounding_px,
    # and convert here if an older artifact lacks it.
    px_boxes = rec.get("grounding_px")
    fw, fh = (rec.get("_frame_wh") or [frames[0].shape[1], frames[0].shape[0]])
    for i, g in enumerate(rec.get("grounding") or []):
        if not g or not g.get("visible"):
            continue
        if px_boxes and i < len(px_boxes) and px_boxes[i]:
            g = dict(g, bbox=px_boxes[i])
        elif g.get("bbox"):
            from ph0_v2 import norm_to_px
            g = dict(g, bbox=norm_to_px(g["bbox"], fw, fh))
        s = signs[i] if i < len(signs) else {}
        bad = [v for c in rec.get("_calls", [])
               if c["call"] == f"B3_ground_{i}" for v in (c.get("violations") or [])]
        lab = f"{s.get('kind','sign')}"
        if s.get("text"):
            lab += f' "{s["text"]}"'
        by_frame.setdefault(int(g.get("frame_idx", 0)), []).append(
            {"bbox": g.get("bbox", [0, 0, 0, 0]), "label": lab, "ok": not bad})

    h = max(frames[0].shape[0], 300)
    panels = []
    for fi, fr in enumerate(frames):
        cam = Image.fromarray(fr).convert("RGB")
        draw_boxes(cam, by_frame.get(fi, []))
        bev = draw_bev((BEV_W, h), engine_a)
        pan = draw_panel((PANEL_W, h), rec, engine_a, fi)
        W = cam.width + BEV_W + PANEL_W
        canvas = Image.new("RGB", (W, h), C_BG)
        canvas.paste(cam, (0, 0))
        canvas.paste(bev, (cam.width, 0))
        canvas.paste(pan, (cam.width + BEV_W, 0))
        panels.append(canvas)

    still = os.path.join(out_dir, f"{cid}_still.png")
    panels[len(panels) // 2].save(still)
    outs = [still]
    try:
        import imageio.v2 as imageio
        mp4 = os.path.join(out_dir, f"{cid}.mp4")
        w = imageio.get_writer(mp4, fps=fps, macro_block_size=1)
        import numpy as np
        for p in panels:
            w.append_data(np.asarray(p))
        w.close()
        outs.append(mp4)
    except Exception as e:
        print(f"[v2viz] no video encoder ({type(e).__name__}) — stills only",
              flush=True)
    return outs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("ph0_v2_overlay")
    ap.add_argument("--v2-json", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--ego-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args(argv)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ph0_pilot import (POSE_HZ, engine_a_summary, load_ego_poses,
                           sample_clip_frames)

    d = json.load(open(a.v2_json))
    os.makedirs(a.out, exist_ok=True)
    made = []
    for rec in d.get("clips", [])[:a.n]:
        cid = rec.get("clip_id")
        if not cid or rec.get("fatal"):
            continue
        try:
            frames, _t, _np_ = sample_clip_frames(
                os.path.join(a.video_root, f"{cid}.mp4"), t0_s=8.0)
            ea = None
            if a.ego_root:
                poses = load_ego_poses(cid, a.ego_root)
                if poses is not None:
                    ea = engine_a_summary(poses, int(round(8.0 * POSE_HZ)))
            made += render_clip(rec, frames, ea, a.out)
            print(f"[v2viz] {str(cid)[:8]} rendered", flush=True)
        except Exception as e:
            print(f"[v2viz] {str(cid)[:8]} FAILED {type(e).__name__}: {e}",
                  flush=True)
    print(f"[v2viz] {len(made)} artifacts in {a.out}", flush=True)
    print("V2VIZ_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
