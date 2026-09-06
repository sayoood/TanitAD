#!/usr/bin/env python3
"""verify_reel_geometry.py -- DECODE the delivered mp4 back to pixels and prove
its GEOMETRY matches the frames that were drawn.

⛔ WHY THIS EXISTS, AND WHY `verify_mp4.py` IS NOT ENOUGH.
MEASURED 2026-09-05 on the refav1 arms reel: the intro card was 1122 px tall and
the clip frames 1080 px, so ffmpeg's image demuxer adopted the FIRST frame's size
and rescaled every subsequent frame into it -- the entire reel shipped with a
3.9 % VERTICAL STRETCH. `ffmpeg` exited 0. `ffprobe` reported a valid stream.
`verify_mp4.py`'s full decode succeeded and its frame count was exactly right.
Every single exit code and count was correct, and the delivered artefact was
geometrically wrong. Container metadata and frame counts cannot see this class of
defect at all, because nothing about it is malformed -- it is the WRONG PICTURE,
correctly encoded.

The only probe that can see it compares DECODED PIXELS against the SOURCE PIXELS.

WHAT IT DOES
============
  (1) samples `--n` frame indices spread across the reel;
  (2) decodes exactly those frames out of the mp4 with ffmpeg;
  (3) loads the corresponding source PNGs (left by the renderer's --keep-frames);
  (4) asserts the DIMENSIONS match, then reports the Pearson correlation of the
      two grayscale images.

⭐ AND IT CARRIES ITS OWN NEGATIVE CONTROL, WHICH IS WHAT MAKES A PASS MEAN
ANYTHING. `--self-test` re-runs the comparison against a deliberately
`--stretch`-distorted copy of each source frame. A probe that has never been
shown to fail is not evidence; MEASURED on the refav1 reel, the true frames read
r >= 0.999572 and the 3.9 %-stretched copies read r = 0.1438 -- four orders of
magnitude of headroom between PASS and the defect it exists to catch.

⚠️ Correlation, not equality: lossy H.264 will never reproduce a PNG bit-exactly,
so the threshold is a high correlation (default 0.99), and a geometric transform
destroys correlation far faster than compression noise does.

USAGE
=====
    python verify_reel_geometry.py <reel.mp4> [--frames <dir>] [--n 7]
                                   [--min-corr 0.99] [--self-test] [--json out]

Exit status: 0 = PASS, 1 = FAIL, 3 = INCONCLUSIVE (could not read enough pairs).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _p(*a):
    print(*a, flush=True)


def n_frames(path: str) -> int:
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def decode_one(path: str, idx: int, out_png: str) -> bool:
    """Decode frame number `idx` (0-based) out of `path`."""
    r = subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-i", path,
         "-vf", f"select=eq(n\\,{idx})", "-vsync", "0", "-frames:v", "1",
         out_png], capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(out_png) and os.path.getsize(out_png) > 0


def corr(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size == 0:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    da, db = float((a * a).sum()) ** 0.5, float((b * b).sum()) ** 0.5
    if da == 0 or db == 0:
        return float("nan")
    return float((a * b).sum() / (da * db))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4")
    ap.add_argument("--frames", default=None,
                    help="dir of source PNGs (default: <mp4 stem>_frames)")
    ap.add_argument("--pattern", default="f_*.png")
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--min-corr", type=float, default=0.99)
    ap.add_argument("--self-test", action="store_true",
                    help="also score a deliberately stretched copy; it MUST fail")
    ap.add_argument("--stretch", type=float, default=1.039,
                    help="the self-test's vertical stretch factor (the MEASURED defect)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if not (FFMPEG and FFPROBE):
        _p("[geom] INCONCLUSIVE: ffmpeg/ffprobe not on PATH")
        return 3
    try:
        from PIL import Image
    except Exception as e:                                    # pragma: no cover
        _p(f"[geom] INCONCLUSIVE: PIL unavailable ({e})")
        return 3

    frames_dir = a.frames or (os.path.splitext(a.mp4)[0] + "_frames")
    src = sorted(glob.glob(os.path.join(frames_dir, a.pattern)))
    nf = n_frames(a.mp4)
    _p(f"[geom] mp4={a.mp4}")
    _p(f"[geom] decoded frame count = {nf}   source PNGs in {frames_dir} = {len(src)}")
    if nf <= 0 or not src:
        _p("[geom] INCONCLUSIVE: no decodable frames or no source PNGs -- this is "
           "a statement about the PROBE, not about the reel.")
        return 3

    #: ⛔ A COUNT MISMATCH IS ITSELF A GEOMETRY-CLASS DEFECT. If the encoder
    #: dropped or duplicated frames the index alignment below is meaningless, so
    #: say so rather than silently comparing frame i to frame j.
    aligned = (nf == len(src))
    if not aligned:
        _p(f"[geom] ⚠ COUNT MISMATCH decoded {nf} vs source {len(src)} -- index "
           f"alignment is NOT guaranteed; comparing on the common prefix.")
    n_common = min(nf, len(src))
    if a.n >= n_common:
        idxs = list(range(n_common))
    else:
        step = n_common / float(a.n)
        idxs = sorted({int(i * step) for i in range(a.n)})

    tmp = tempfile.mkdtemp(prefix="geomprobe_")
    rows, fails, inconc = [], 0, 0
    st_rows = []
    try:
        for i in idxs:
            out = os.path.join(tmp, f"dec_{i:06d}.png")
            if not decode_one(a.mp4, i, out):
                _p(f"[geom]  frame {i:6d}  INCONCLUSIVE (decode failed)")
                inconc += 1
                continue
            im_d = Image.open(out).convert("L")
            im_s = Image.open(src[i]).convert("L")
            same_dim = (im_d.size == im_s.size)
            r = corr(im_d, im_s) if same_dim else float("nan")
            ok = same_dim and (r == r) and r >= a.min_corr
            rows.append(dict(idx=i, src=os.path.basename(src[i]),
                             dec_size=list(im_d.size), src_size=list(im_s.size),
                             same_dim=bool(same_dim), corr=None if r != r else round(r, 6),
                             pass_=bool(ok)))
            _p(f"[geom]  frame {i:6d}  dec {im_d.size}  src {im_s.size}  "
               f"dim {'OK ' if same_dim else 'MISMATCH'}  r={r:.6f}  "
               f"{'PASS' if ok else 'FAIL'}   {os.path.basename(src[i])}")
            if not ok:
                fails += 1

            if a.self_test:
                w, h = im_s.size
                sh = max(1, int(round(h * a.stretch)))
                #: ⛔ THE REPRODUCTION HAS TO DISPLACE CONTENT, NOT JUST BLUR IT.
                #: MEASURED 2026-09-06: a first version resized to `sh` and back
                #: to `h`, which is a LOW-PASS FILTER -- the rows stay where they
                #: were, so a 3.9 % "stretch" still correlated at r = 0.9931-0.9958
                #: and the probe reported ITSELF BLIND (correctly, and that refusal
                #: is why this was caught rather than shipped).
                #: The real defect rescales an h-tall frame INTO a taller canvas,
                #: so every row lands at the wrong y. Reproduce that: stretch to
                #: `sh` and KEEP THE ORIGINAL CANVAS by cropping, which displaces
                #: content progressively down the frame.
                bad = im_s.resize((w, sh)).crop((0, 0, w, h))
                rb = corr(im_d, bad)
                st_rows.append(dict(idx=i, corr=round(rb, 6)))
                _p(f"[geom]    self-test stretched x{a.stretch}: r={rb:.6f} "
                   f"{'(correctly BELOW threshold)' if rb < a.min_corr else '⛔ ABOVE THRESHOLD -- THE PROBE IS BLIND'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_ok = sum(1 for r in rows if r["pass_"])
    _p("")
    _p(f"[geom] compared {len(rows)} frame pairs: PASS {n_ok}  FAIL {fails}  "
       f"INCONCLUSIVE {inconc}")
    verdict = "PASS"
    if not rows:
        verdict = "INCONCLUSIVE"
    elif fails or not aligned:
        verdict = "FAIL"
    if a.self_test:
        blind = [r for r in st_rows if r["corr"] >= a.min_corr]
        if not st_rows:
            _p("[geom] ⛔ self-test produced no rows -- the control did not run")
            verdict = "INCONCLUSIVE"
        elif blind:
            _p(f"[geom] ⛔ SELF-TEST FAILED on {len(blind)} frames: the stretched "
               f"control scored ABOVE the threshold, so a PASS here means nothing.")
            verdict = "INCONCLUSIVE"
        else:
            worst = max(r["corr"] for r in st_rows)
            best_true = min(r["corr"] for r in rows if r["corr"] is not None) if rows else float("nan")
            _p(f"[geom] ⭐ self-test PASSED: stretched control max r={worst:.6f} "
               f"< threshold {a.min_corr}; true frames min r={best_true:.6f}. "
               f"The probe is PROVEN ABLE TO FAIL.")
    _p(f"[geom] VERDICT = {verdict}")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(mp4=a.mp4, frames_dir=frames_dir, n_decoded=nf,
                           n_source=len(src), aligned=aligned,
                           min_corr=a.min_corr, rows=rows,
                           self_test_stretch=a.stretch if a.self_test else None,
                           self_test=st_rows, verdict=verdict), fh, indent=1)
        _p(f"[geom] json -> {a.json}")
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3}[verdict]


if __name__ == "__main__":
    sys.exit(main())
