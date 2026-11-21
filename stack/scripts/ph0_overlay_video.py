"""PH0 deliverable video renderer — camera overlay + BEV pane + text panel.

Renders, for each pilot clip labeled by `ph0_pilot.py`, an mp4 with three
synchronized panes (compositing conventions follow the ops-bundle
`render_v5f_bev.py`: PIL canvas, camera pane on top, BEV below, dark theme,
ffmpeg assembly; this script adds a right-hand text panel):

  (a) CAMERA: the sampled clip frames with ALL image-grounded overlays drawn —
      sign / light / agent bboxes with labels + OCR text, SAM mask contours
      where present, DISPUTED items dashed red;
  (b) BEV (ego frame at t0, forward = up): Engine A integrated path polyline,
      route/lane-change/speed events, ego marker, range rings, agent
      positions where derivable (numeric bev_xy on a claim, e.g. from the
      Alpamayo grounding join);
  (c) TEXT PANEL: rolling display of scenario/domain fields, the strategic
      goal + actions with their constraint slots, Alpamayo meta-actions, and
      the fusion verdicts (pass green / disputed red).

Pod-side: reads ph0_pilot's per-clip JSONs + the clip videos, re-samples
frames with the exact `_provenance.video_sampling` parameters (so bbox
frame indices line up by construction), writes frames + mp4 under /tmp
(container disk, off MooseFS during writes — render_v5f_bev convention),
then copies the mp4 to --out.

CPU-testable: `compose_frame` is a pure function (numpy + PIL only); video IO
and ffmpeg are reached only from `main`. No GPU anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ph0_pilot import rle_decode, sample_clip_frames  # noqa: E402

# ---- layout (render_v5f_bev conventions: camera top, BEV below; text right) --
CAM_W = 896                       # camera pane width (448 px frames x2)
BEV_H = 380
BEV_XR = 60.0                     # metres ahead shown
BEV_S = (BEV_H - 40) / BEV_XR     # px per metre
TEXT_W = 480
BG = (12, 14, 18)
GRID = (45, 50, 60)
FG = (200, 205, 215)
DIM = (110, 118, 130)
C_PASS = (118, 255, 40)
C_DISPUTED = (255, 64, 64)
C_SIGN = (255, 220, 120)
C_AGENT = (0, 220, 255)
C_PATH = (238, 51, 119)
C_SAM = (170, 120, 255)


def _fonts():
    from PIL import ImageFont
    try:
        f = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        fs = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:                                       # noqa: BLE001
        f = fs = ImageFont.load_default()
    return f, fs


def _dashed_line(dr, xy, color, width=2, dash=6):
    """Dashed polyline (PIL has none built in)."""
    for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
        seg = np.hypot(x1 - x0, y1 - y0)
        n = max(1, int(seg // dash))
        for k in range(0, n, 2):
            t0, t1 = k / n, min((k + 1) / n, 1.0)
            dr.line([x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0,
                     x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1],
                    fill=color, width=width)


def _dashed_rect(dr, box, color, width=2, dash=6):
    x0, y0, x1, y1 = box
    _dashed_line(dr, [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
                 color, width, dash)


def _mask_contour(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a binary mask (mask XOR 4-neighbour erosion) — pure
    numpy, no cv2 needed for the CPU test path."""
    m = np.asarray(mask, dtype=bool)
    er = m.copy()
    er[1:, :] &= m[:-1, :]
    er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]
    er[:, :-1] &= m[:, 1:]
    ys, xs = np.nonzero(m & ~er)
    return np.stack([xs, ys], axis=1) if xs.size else np.zeros((0, 2), int)


def _is_disputed(item: dict) -> bool:
    return bool(item.get("disputed") or item.get("retracted")
                or not item.get("grounded", True))


def _claim_color(kind: str, item: dict):
    if _is_disputed(item):
        return C_DISPUTED
    return C_SIGN if kind == "sign" else C_AGENT


def _sam_by_frame(rec: dict) -> dict:
    out: dict = {}
    sam = rec.get("sam") or {}
    for inst in sam.get("instances") or []:
        out.setdefault(int(inst["frame_idx"]), []).append(inst)
    return out


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def _fmt_constraints(cons: dict | None) -> str:
    if not cons:
        return ""
    parts = [f"{k}={cons[k]:g}" if isinstance(cons[k], (int, float))
             else f"{k}={cons[k]}"
             for k in ("within_m", "by_time_s", "at_arc_m", "hold_for_s")
             if cons.get(k) is not None]
    return (" [" + ", ".join(parts) + "]") if parts else ""


def compose_frame(frame_rgb: np.ndarray, rec: dict, frame_idx: int,
                  n_past: int, fps: float = 2.0) -> np.ndarray:
    """Compose one output frame — PURE function of (sampled camera frame,
    per-clip record, frame index). Returns an (H, W, 3) uint8 array.

    frame_idx is the index in the SAMPLED (2 fps) sequence — the same index
    space every bbox `frame_idx` in the record uses (image-coordinate rule).
    """
    from PIL import Image, ImageDraw
    font, font_s = _fonts()

    fh, fw = frame_rgb.shape[:2]
    sc = CAM_W / fw
    cam_h = int(round(fh * sc))
    H_TOT = max(cam_h + BEV_H, 640)
    W_TOT = CAM_W + TEXT_W

    canvas = Image.new("RGB", (W_TOT, H_TOT), BG)
    cam = Image.fromarray(np.ascontiguousarray(frame_rgb)).resize(
        (CAM_W, cam_h), Image.LANCZOS)
    canvas.paste(cam, (0, 0))
    dr = ImageDraw.Draw(canvas, "RGBA")

    # ---- (a) camera overlays ------------------------------------------------
    def draw_claim(kind, idx, item):
        if not isinstance(item, dict) or item.get("bbox") is None:
            return
        try:
            x0, y0, x1, y1 = [float(v) * sc for v in item["bbox"]]
        except (TypeError, ValueError):
            return
        if int(item.get("frame_idx", -1)) != frame_idx:
            return
        col = _claim_color(kind, item)
        if _is_disputed(item):
            _dashed_rect(dr, (x0, y0, x1, y1), col, width=2)
        else:
            dr.rectangle([x0, y0, x1, y1], outline=col, width=2)
        label = item.get("class") or item.get("kind") or kind
        state = item.get("state")
        ocr = str(item.get("text_ocr") or "").strip()
        txt = str(label) + (f":{state}" if state else "")
        if ocr:
            txt += f' "{ocr}"'
        if _is_disputed(item):
            txt += " [DISPUTED]"
        ty = max(0, y0 - 15)
        dr.rectangle([x0, ty, x0 + 7 * len(txt) + 4, ty + 14],
                     fill=(0, 0, 0, 180))
        dr.text((x0 + 2, ty), txt, fill=col, font=font_s)

    for i, s in enumerate(rec.get("signs") or []):
        draw_claim("sign", i, s)
    for i, ag in enumerate((rec.get("scenario") or {}).get("agents") or []):
        draw_claim("agent", i, ag)
    for inst in _sam_by_frame(rec).get(frame_idx, []):
        try:
            pts = _mask_contour(rle_decode(inst["rle"]))
        except (KeyError, ValueError, TypeError):
            continue
        for x, y in pts[::2]:
            dr.point((float(x) * sc, float(y) * sc), fill=C_SAM)

    t_rel = (frame_idx - n_past) / max(fps, 1e-6)
    dr.rectangle([0, 0, CAM_W, 24], fill=(0, 0, 0, 180))
    dr.text((8, 4),
            f"PH0 {rec.get('clip_id', '?')} · frame {frame_idx} · "
            f"t0{t_rel:+.1f}s ({'PAST' if t_rel < 0 else 'FUTURE'})"
            " · dashed red = disputed",
            fill=(255, 255, 255), font=font_s)

    # ---- (b) BEV pane -------------------------------------------------------
    y0b = cam_h
    dr.rectangle([0, y0b, CAM_W, H_TOT], fill=BG)
    cx = CAM_W // 2
    ey = y0b + BEV_H - 20

    def bev_px(x_fwd, y_left):
        return cx - y_left * BEV_S, ey - x_fwd * BEV_S

    for rr in range(10, int(BEV_XR) + 1, 10):
        py = ey - rr * BEV_S
        dr.line([cx - 300, py, cx + 300, py], fill=GRID, width=1)
        dr.text((cx + 306, py - 7), f"{rr} m", fill=DIM, font=font_s)
    dr.line([cx, y0b + 10, cx, ey + 6], fill=GRID, width=1)

    ea = rec.get("engine_a")
    if ea:
        poly = ea.get("polyline_xy") or []
        if len(poly) >= 2:
            xy = [bev_px(p[0], p[1]) for p in poly]
            dr.line(xy, fill=C_PATH, width=3)
        for ev in ea.get("lane_change_events", []):
            arc = ev.get("arc_from_t0_m")
            if arc is not None and 0 <= arc <= BEV_XR:
                px, py = bev_px(arc, 0.0)
                dr.ellipse([px - 5, py - 5, px + 5, py + 5], outline=C_AGENT,
                           width=2)
                dr.text((px + 8, py - 7), ev["token"], fill=C_AGENT,
                        font=font_s)
        for ev in ea.get("speed_events", []):
            arc = ev.get("arc_from_t0_m")
            if arc is not None and 0 <= arc <= BEV_XR:
                px, py = bev_px(arc, 3.0)
                dr.ellipse([px - 4, py - 4, px + 4, py + 4], fill=C_SIGN)
                dr.text((px + 7, py - 7), ev["token"], fill=C_SIGN,
                        font=font_s)
        route = ea.get("route", {})
        dr.text((8, y0b + 6),
                f"BEV · Engine A path (pink) · route {route.get('token')} "
                f"({route.get('dist_band', '')}) · "
                f"v0 {ea.get('speed_profile', {}).get('v_t0_ms', 0) * 3.6:.0f}"
                " km/h", fill=FG, font=font_s)
    else:
        dr.text((8, y0b + 6), "BEV · engine_a: null (no ego data)",
                fill=C_DISPUTED, font=font_s)

    # agents with derivable BEV positions (numeric bev_xy on the claim)
    for ag in (rec.get("scenario") or {}).get("agents") or []:
        pos = ag.get("bev_xy") if isinstance(ag, dict) else None
        if (isinstance(pos, (list, tuple)) and len(pos) == 2
                and all(isinstance(v, (int, float)) for v in pos)):
            px, py = bev_px(float(pos[0]), float(pos[1]))
            col = _claim_color("agent", ag)
            dr.rectangle([px - 4, py - 4, px + 4, py + 4], outline=col,
                         width=2)
            dr.text((px + 7, py - 7), str(ag.get("class", "?")), fill=col,
                    font=font_s)
    dr.polygon([(cx, ey - 10), (cx - 6, ey + 4), (cx + 6, ey + 4)],
               fill=(255, 255, 255))

    # ---- (c) text panel -----------------------------------------------------
    tx = CAM_W + 12
    dr.rectangle([CAM_W, 0, W_TOT, H_TOT], fill=(20, 23, 29))
    dr.line([CAM_W, 0, CAM_W, H_TOT], fill=GRID, width=2)
    y = 8
    line_h, wrapw = 16, 52

    def put(text, color=FG, fnt=None, indent=0):
        nonlocal y
        for ln in _wrap(text, wrapw - indent // 6):
            if y > H_TOT - line_h:
                return
            dr.text((tx + indent, y), ln, fill=color, font=fnt or font_s)
            y += line_h

    sc_ = rec.get("scenario") or {}
    road = sc_.get("road") or {}
    dom = rec.get("domain") or {}
    put("SCENARIO", C_SIGN, font)
    put(f"{sc_.get('daynight', '?')} · {sc_.get('illumination', '?')} · "
        f"{sc_.get('weather', '?')}")
    put(f"road {road.get('type', '?')} · lanes {road.get('lanes_visible', '?')}"
        f" · ego lane {road.get('lane_ego', '?')}")
    put(f"domain {dom.get('class', '?')} "
        f"(conf {dom.get('confidence', 0):.2f})" if dom else "domain ?")
    put(f"ego: {sc_.get('ego_behaviour', '?')}", DIM)
    y += 6

    strat = rec.get("strategic") or {}
    goal = strat.get("goal") or {}
    gv = goal.get("fusion", "pass")
    put("STRATEGIC GOAL", C_SIGN, font)
    gtxt = str(goal.get("kind", "?"))
    if goal.get("target_text"):
        gtxt += f' -> "{goal["target_text"]}"'
    gtxt += f" · src {goal.get('source', '?')}"
    put(gtxt + (" [DISPUTED]" if gv == "disputed" else ""),
        C_DISPUTED if gv == "disputed" else C_PASS)
    for a in strat.get("actions") or []:
        verdict = a.get("geometric_consistency", "?")
        col = C_PASS if verdict == "pass" else C_DISPUTED
        d = f" {a['direction']}" if a.get("direction") else ""
        put(f"· {a.get('verb', '?')}{d}{_fmt_constraints(a.get('constraints'))}"
            f" — {verdict.upper()}", col, indent=4)
        if verdict == "disputed":
            for r in a.get("fusion_reasons") or []:
                put(f"  {r}", DIM, indent=10)
    y += 6

    alp = rec.get("alpamayo")
    put("ALPAMAYO", C_SIGN, font)
    if alp:
        for ma in (alp.get("meta_actions") or [])[:4]:
            put(f"· {ma}", C_AGENT, indent=4)
        put(f"{alp.get('n_rows', 0)} record rows · tasks: "
            + ", ".join(sorted((alp.get('tasks') or {}).keys())), DIM)
    else:
        put("no records for this clip", DIM)
    y += 6

    fus = rec.get("fusion") or {}
    put("FUSION", C_SIGN, font)
    put(f"actions: {fus.get('actions_pass', 0)} pass · "
        f"{fus.get('actions_disputed', 0)} disputed · engine A "
        + ("present" if fus.get("engine_a_available") else "ABSENT"),
        C_PASS if not fus.get("actions_disputed") else C_DISPUTED)
    for it in (fus.get("ungrounded_disputed") or [])[:5]:
        put(f"· ungrounded {it['kind']}[{it['index']}]", C_DISPUTED, indent=4)

    return np.asarray(canvas, dtype=np.uint8)


# =============================================================================
# assembly
# =============================================================================

def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def render_clip(rec: dict, out_mp4: Path, fps_out: int = 4,
                frames_dir: str = "/tmp/ph0_overlay_frames") -> int:
    """Render one labeled clip to mp4. Re-samples the clip video with the
    provenance sampling parameters so record frame indices align exactly."""
    vs = (rec.get("_provenance") or {}).get("video_sampling") or {}
    video_path = vs.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise FileNotFoundError(f"clip video not found: {video_path!r}")
    frames, _times, n_past = sample_clip_frames(
        video_path, t0_s=vs.get("t0_s", 8.0), fps=vs.get("fps", 2.0),
        px=vs.get("px", 448))
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)
    from PIL import Image
    for i, fr in enumerate(frames):
        out = compose_frame(fr, rec, i, n_past, fps=vs.get("fps", 2.0))
        # hold each sampled (2 fps) frame for fps_out/fps repeats -> real-time
        reps = max(1, int(round(fps_out / max(vs.get("fps", 2.0), 1e-6))))
        for r in range(reps):
            Image.fromarray(out).save(
                f"{frames_dir}/f{i * reps + r:06d}.png")
    n = len(frames)
    tmp_mp4 = "/tmp/ph0_overlay_tmp.mp4"
    subprocess.run([_ffmpeg_exe(), "-y", "-framerate", str(fps_out), "-i",
                    f"{frames_dir}/f%06d.png", "-c:v", "libx264", "-preset",
                    "medium", "-crf", "21", "-threads", "4", "-pix_fmt",
                    "yuv420p", tmp_mp4], check=True)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tmp_mp4, out_mp4)
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot-dir", required=True,
                    help="ph0_pilot.py --out directory (per-clip JSONs)")
    ap.add_argument("--out", required=True, help="output dir for mp4s")
    ap.add_argument("--fps-out", type=int, default=4,
                    help="output video framerate (sampled frames are held)")
    ap.add_argument("--concat", action="store_true",
                    help="also concatenate all clips into ph0_overlay_all.mp4")
    args = ap.parse_args(argv)

    pilot = Path(args.pilot_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_jsons = sorted(p for p in pilot.glob("*.json")
                        if p.name != "pilot_summary.json")
    made = []
    for cj in clip_jsons:
        rec = json.loads(cj.read_text())
        out_mp4 = out_dir / (cj.stem + ".mp4")
        try:
            n = render_clip(rec, out_mp4, fps_out=args.fps_out)
            made.append(out_mp4)
            print(f"[ph0-overlay] {cj.name}: {n} frames -> {out_mp4}",
                  flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"[ph0-overlay] {cj.name}: FAILED {type(e).__name__}: {e}",
                  flush=True)
    if args.concat and made:
        lst = Path("/tmp/ph0_concat.txt")
        lst.write_text("".join(f"file '{m.resolve()}'\n" for m in made))
        subprocess.run([_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0",
                        "-i", str(lst), "-c", "copy",
                        str(out_dir / "ph0_overlay_all.mp4")], check=True)
    print(f"[ph0-overlay] {len(made)}/{len(clip_jsons)} clips rendered",
          flush=True)
    print("PH0_OVERLAY_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
