#!/usr/bin/env python3
"""verify_mp4.py — verify a rendered reel by DECODING IT BACK.

⛔ WHY THIS EXISTS. `ffmpeg` exiting 0 is not evidence that the file it wrote is
the file you meant to write, and neither is its size: a truncated mp4 keeps a
header claiming the full length, so `ffprobe`'s `duration` and `nb_frames` can
both read correctly while the decoder yields a fraction of the frames. That is
the same family as every trap in `CLAUDE.md` where a probe reports the wrong
scope and is read as an answer — and it is worse for a video, because the
artefact is delivered to a human who will not re-derive it.

So this runs TWO probes that differ in PATH-BINDING, because one is not evidence:

  (A) the CONTAINER metadata — `ffprobe -show_streams -show_format`: codec,
      pixel format, width/height, `avg_frame_rate`, `duration`, file size;
  (B) a REAL DECODE of every packet — `ffmpeg -i … -f null -`, which fails on a
      truncated or corrupt stream, plus `ffprobe -count_frames`, which counts
      the frames the decoder ACTUALLY produced.

and prints `n / fps` beside the container's own duration so the two must agree.

⭐ IT ALSO GATES ON SIZE, BECAUSE SIZE IS A DELIVERY GATE, NOT A PREFERENCE.
MEASURED 2026-09-03: the 30 000-step five-panel reel (38.74 MiB) was REFUSED by
the PI's delivery channel and reached only the desktop app. Anything at or over
**30 MiB** does not arrive, so every reel ships as a pair — the full-quality
render and a re-encoded copy comfortably under the limit — and this tool says
plainly which side of the line each file is on.

⚠️ Build the small copy by RE-ENCODING FROM THE ORIGINAL FRAMES (`--keep-frames`
leaves them beside the mp4), never by transcoding the finished mp4: a transcode
stacks a second generation of loss on the first and buys nothing.

USAGE
=====
    python taniteval/tools/verify_mp4.py <file.mp4> [<file.mp4> ...]

Exit status is non-zero if any file failed to decode.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
if not (FFMPEG and FFPROBE):
    raise SystemExit("ffmpeg/ffprobe not on PATH")

MIB = 1024 * 1024
#: the delivery channel's hard ceiling — MEASURED by a refusal, not assumed.
LIMIT_MIB = 30.0


def probe(path: str) -> dict:
    """(A) container metadata."""
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-show_format",
         "-of", "json", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"ffprobe failed on {path}: {r.stderr[:500]}")
    return json.loads(r.stdout)


def decode_count(path: str) -> tuple[int, bool, str]:
    """(B) a real decode. Returns (frames_decoded, decode_was_clean, stderr)."""
    r = subprocess.run(
        [FFMPEG, "-v", "error", "-i", path, "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = (r.returncode == 0 and not r.stderr.strip())
    r2 = subprocess.run(
        [FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    return int(r2.stdout.strip() or -1), ok, r.stderr.strip()[:400]


def main(argv: list[str] | None = None) -> int:
    paths = list(argv if argv is not None else sys.argv[1:])
    if not paths:
        raise SystemExit("usage: verify_mp4.py <file.mp4> [<file.mp4> ...]")
    rc = 0
    for path in paths:
        size = os.path.getsize(path)
        meta = probe(path)
        v = next(s for s in meta["streams"] if s["codec_type"] == "video")
        n, ok, err = decode_count(path)
        dur = float(meta["format"]["duration"])
        num, den = (int(x) for x in v["avg_frame_rate"].split("/"))
        fps = num / den if den else 0.0
        under = size / MIB < LIMIT_MIB
        print("=" * 78)
        print(f"FILE        {path}")
        print(f"size        {size:,} B = {size / MIB:.2f} MiB   "
              f"[{'UNDER' if under else 'OVER'} the {LIMIT_MIB:.0f} MiB "
              f"delivery limit]")
        print(f"codec       {v['codec_name']} / {v.get('profile')} "
              f"pix={v.get('pix_fmt')}")
        print(f"dimensions  {v['width']}x{v['height']}   (container metadata)")
        print(f"avg_fps     {fps:.3f}      duration {dur:.2f} s")
        print(f"DECODED     {n} frames   decode_clean={ok}"
              + (f"   stderr={err!r}" if err else ""))
        print(f"consistency n/fps = {n / fps if fps else float('nan'):.2f} s "
              f"vs container {dur:.2f} s")
        if not ok or n <= 0:
            print("⛔ DECODE FAILED — the file is not deliverable")
            rc = 1
    print("=" * 78)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
