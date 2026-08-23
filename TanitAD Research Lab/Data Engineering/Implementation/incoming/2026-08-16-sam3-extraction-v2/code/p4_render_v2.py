"""STEP 4 — render what v2 added, so the PI can SEE it.

Draws, on the SAME BYTES the model saw (re-bridged from the w120 shard, never
the pre-bridged mp4 — C79: a different encode moves ~7 % of detections):

  * agent detections   — mask fill + CONTOUR outline + the ORIENTED extent
  * scene detections   — contour outlines, one hue per scene class
  * the ego-lane derivation — its near-field band, its clustered boundaries,
    and the verdict or the REASON there is none
  * a header strip with the per-channel counts and the liveness control

⛔ ONLY THE FRAMES THE ENGINE ACTUALLY RAN. The record holds detections on the
strided frames; drawing them onto the frames in between — by holding the
nearest one — is the SNAPPING confound this engine already retracted once
(a 0/8 "agreement" that was a property of the snapping, not of the data). So
the video is one second per RUN frame and says which frame it is, and the
contact sheet is the artifact that actually reads well.

⚠️ HOST-AGNOSTIC ON PURPOSE. It needs NO GPU — only banked records, the video
bridge and PIL — so it runs on the dev box as happily as on the VM, and it had
to: free-Colab reclaimed the T4 the moment the 115-clip run finished, and a
renderer that could only run beside the model would have been dead with it.

Writes <out>/v2_<clip8>.mp4 and <out>/v2_<clip8>_sheet.png.
"""
import json
import os
import sys

_COLAB = os.path.isdir("/content/repo")
if _COLAB:
    os.chdir("/content")
    _ROOTS = ("/content/repo/colab", "/content/repo/stack",
              "/content/repo/stack/scripts")
    _OUT, _WORK = "/content/out", "/content/p4work"
else:                                   # dev box: the repo is right here
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PKG = os.path.dirname(_HERE)
    _REPO = os.path.abspath(os.path.join(_PKG, "..", "..", "..", "..", ".."))
    _ROOTS = (os.path.join(_REPO, "colab"), os.path.join(_REPO, "stack"),
              os.path.join(_REPO, "stack", "scripts"))
    _OUT = os.path.join(_PKG, "video")
    _WORK = os.path.join(os.environ.get("TEMP", "."), "tanitad-p4work")
    try:
        import truststore
        truststore.inject_into_ssl()    # the dev box sits behind a TLS proxy
    except Exception:
        pass
for p in _ROOTS:
    if p not in sys.path:
        sys.path.insert(0, p)
from pathlib import Path                                         # noqa: E402
import s2_lab_lib as L                                           # noqa: E402
import ph0_sam3                                                  # noqa: E402

SCALE = 3
OUT = Path(_OUT)
OUT.mkdir(parents=True, exist_ok=True)
WORK = Path(_WORK)
WORK.mkdir(parents=True, exist_ok=True)

AGENT_COL = {"car": (86, 156, 255), "truck": (255, 149, 61),
             "bus": (255, 196, 61), "pedestrian": (255, 92, 138),
             "cyclist": (170, 120, 255), "traffic light": (94, 224, 156),
             "traffic sign": (255, 233, 120)}
SCENE_COL = {"lane marking": (255, 255, 255), "road marking": (120, 255, 235),
             "road curb": (255, 128, 0), "guardrail": (200, 120, 255)}
LANE_COL = (0, 255, 128)
INK = (232, 236, 244)


def font(sz):
    """⚠️ The Windows roots are not padding. `ImageFont.load_default()` is a
    bitmap font with no `·`, and this overlay's header is full of them — on the
    dev box every separator would render as a tofu box (the same trap
    `ph0_rich_overlay._font` documents)."""
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              r"C:\Windows\Fonts\consola.ttf",
              r"C:\Windows\Fonts\DejaVuSansMono.ttf",
              r"C:\Windows\Fonts\segoeui.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def poly(det, s):
    xy = det.get("contour_xy") or []
    return [(xy[i] * s, xy[i + 1] * s) for i in range(0, len(xy) - 1, 2)]


def obb_corners(det, s):
    import math
    o = det.get("obb_cxcylwa")
    if not o:
        return None
    cx, cy, lg, sh, deg = o
    t = math.radians(deg)
    ux, uy = math.cos(t), math.sin(t)
    vx, vy = -uy, ux
    hl, hs = lg / 2.0, sh / 2.0
    return [((cx + a * hl * ux + b * hs * vx) * s,
             (cy + a * hl * uy + b * hs * vy) * s)
            for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def render_frame(frame, rec, fi):
    import numpy as np
    from PIL import Image, ImageDraw
    fr = rec["frames"][str(fi)]
    dets, scene = fr.get("det") or [], fr.get("scene") or []
    base = np.asarray(frame).astype(np.float32)
    for d in dets:
        col = np.array(AGENT_COL.get(d.get("concept"), (200, 200, 200)),
                       np.float32)
        for r, a, b in d.get("rle_rows") or []:
            if 0 <= r < base.shape[0]:
                base[r, a:b] = 0.40 * col + 0.60 * base[r, a:b]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGB")
    img = img.resize((img.width * SCALE, img.height * SCALE), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    f, fs = font(11), font(9)

    for s in scene:                                   # scene: OUTLINES only
        col = SCENE_COL.get(s.get("concept"), (180, 180, 180))
        pts = poly(s, SCALE)
        if len(pts) >= 3:
            d.line(pts + [pts[0]], fill=col, width=1)

    for det in dets:
        col = AGENT_COL.get(det.get("concept"), (200, 200, 200))
        pts = poly(det, SCALE)
        if len(pts) >= 3:
            d.line(pts + [pts[0]], fill=col, width=2)          # ⭐ CONTOUR
        c = obb_corners(det, SCALE)
        if c:                                                  # ⭐ ORIENTED
            d.line(c + [c[0]], fill=(255, 255, 255), width=1)
        b = det.get("box_xyxy")
        if b:                                                  # box, faint
            d.rectangle([v * SCALE for v in b], outline=col, width=1)
            lab = f"{det['concept']} {det.get('score', 0):.2f}"
            ty = max(0, b[1] * SCALE - 11)
            d.rectangle([b[0] * SCALE, ty, b[0] * SCALE + 6 * len(lab),
                         ty + 11], fill=(0, 0, 0))
            d.text((b[0] * SCALE + 1, ty), lab, fill=col, font=fs)

    lane = ((rec.get("ego_lane") or {}).get("frames") or {}).get(str(fi)) or {}
    if lane:
        yb = int(lane.get("near_frac", 0.75) * img.height)
        d.line([0, yb, img.width, yb], fill=(90, 90, 90), width=1)
        d.line([lane.get("ego_u", 0) * SCALE, yb,
                lane.get("ego_u", 0) * SCALE, img.height],
               fill=(120, 120, 120), width=1)
        for bnd in lane.get("boundaries") or []:
            x = bnd["x"] * SCALE
            d.line([x, yb, x, img.height], fill=LANE_COL, width=2)
            d.text((x + 2, img.height - 22), f"n{bnd['n']}", fill=LANE_COL,
                   font=fs)
        msg = (f"ego lane idx {lane['lane_idx_est']} · width "
               f"{lane['lane_width_px']} px"
               if lane.get("lane_idx_est") is not None
               else f"ego lane: {lane.get('reason', 'n/a')}")
        d.text((4, img.height - 12), "DERIVED · " + msg, fill=LANE_COL, font=fs)

    hdr = (f"frame {fi} · agents {fr.get('n_det', 0)} · scene "
           f"{fr.get('n_scene', 0)} · contours "
           f"{sum(1 for x in dets + scene if x.get('contour_xy'))}")
    d.rectangle([0, 0, img.width, 13], fill=(0, 0, 0))
    d.text((4, 1), hdr, fill=INK, font=f)
    return img


def main():
    api = L.hf_api()
    far = L.list_far(api, L.DS_LABELS, L.BACKFILL_V2_PREFIX)
    stems = [rf for rf in far if rf.endswith(".json") and "/_runs/" not in rf]
    assert stems, "no v2 records banked yet"
    best = []
    for rf in stems:
        rec = json.load(open(L.hf_download(L.DS_LABELS, rf)))
        best.append((int(rec.get("n_scene_det_total") or 0)
                     + int(rec.get("n_det_total") or 0), rec["clip_id"], rec))
    best.sort(reverse=True)
    picks = [b[1] for b in best[:2]]
    print(f"[p4] {len(stems)} banked · rendering richest: "
          + ", ".join(f"{c[:8]}({n})" for n, c, _ in best[:2]))

    # ⚠️ Re-bridging is the expensive AND fragile half (shard pull + a torch
    # load per clip; MEASURED segfaulting under MSYS on a repeat run), and it
    # is pure waste when the bytes are already on disk. Skip it when every
    # picked clip already has its bridged mp4 — the frames are identical by
    # construction, which is the C79 requirement.
    have = all((WORK / "videos" / f"{c}.mp4").exists() for c in picks)
    if have:
        print("[p4] bridged mp4s already present — reusing the SAME bytes")
    else:
        loc = L.w120_locations(api)
        REC_PQ = str(WORK / "records.parquet")
        if not Path(REC_PQ).exists():
            import shutil
            shutil.copyfile(L.hf_download(L.DS_ALP, "records.parquet"), REC_PQ)
        L.bridge_batch(picks, loc, REC_PQ, WORK)

    import imageio.v2 as imageio
    import numpy as np
    import ph0_pilot
    from PIL import Image
    for _n, cid, rec in best[:2]:
        frames = ph0_pilot.sample_clip_frames(
            str(WORK / "videos" / f"{cid}.mp4"), t0_s=8.0)[0]
        fis = sorted(int(k) for k in rec["frames"])
        imgs = [render_frame(frames[fi], rec, fi) for fi in fis]
        mp4 = OUT / f"v2_{cid[:8]}.mp4"
        # ⚠️ libx264 REFUSES an odd height and imageio reports it only as a
        # wall of ffmpeg stderr while still creating a 0-byte file — MEASURED
        # here at 1344x537 (179 px frame x SCALE 3). `macro_block_size=1` does
        # NOT cover this. Pad to even and check the size afterwards, because a
        # 0-byte mp4 beside a healthy PNG is exactly the kind of "it ran" that
        # this package keeps finding.
        def _even(im):
            W2, H2 = im.width + im.width % 2, im.height + im.height % 2
            if (W2, H2) == (im.width, im.height):
                return np.asarray(im)
            pad = Image.new("RGB", (W2, H2), (10, 12, 16))
            pad.paste(im, (0, 0))
            return np.asarray(pad)

        w = imageio.get_writer(str(mp4), fps=1, macro_block_size=1)
        for im in imgs:
            w.append_data(_even(im))
        w.close()
        if mp4.stat().st_size == 0:
            raise SystemExit(f"{mp4} is ZERO BYTES — the encoder refused it")
        cols = 2
        rows = (len(imgs) + cols - 1) // cols
        W, H = imgs[0].width, imgs[0].height
        sheet = Image.new("RGB", (cols * W, rows * H), (10, 12, 16))
        for i, im in enumerate(imgs):
            sheet.paste(im, ((i % cols) * W, (i // cols) * H))
        png = OUT / f"v2_{cid[:8]}_sheet.png"
        sheet.save(png)
        print(f"[p4] {cid[:8]} {len(imgs)} run frames -> {mp4.name} "
              f"({mp4.stat().st_size} B) + {png.name} "
              f"({png.stat().st_size} B)", flush=True)
    print("P4_DONE")


main()
