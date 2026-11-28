"""STEP 2 — the per-concept score distribution over every banked detection.

⚠️ **A COUNT IS NOT A RELIABILITY MEASUREMENT, AND A DISTRIBUTION IS NOT EITHER.**
This step is descriptive only. It says where the mass sits relative to the
vendor's `confidence_threshold=0.5`; it says NOTHING about whether a detection is
correct. Precision comes from the labelled check (step 4) and from nowhere else —
inferring correctness from the detector's own score is circular, which is the
exact failure this whole study exists to catch.

What it answers:
  · per concept: n, min/median/max, and the FULL histogram in 0.02-wide bins;
  · the mass in [0.50, 0.55) — a class that clusters against the boundary is
    fragile in a way its total count does not show (C79: a re-encode of the same
    clip moved `traffic light` 0 -> 2);
  · box geometry per concept (area in px², fraction of frame), because a class
    whose detections are all a handful of pixels is a different failure mode
    from one whose boxes are plausible.

Reads the local record cache written by `r1_pull_records.py`. No network, no GPU.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics
import sys

CACHE = (r"C:\Users\Admin\AppData\Local\Temp\claude"
         r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
         r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\sam3rel\records")


def load(cache: str):
    """Only records the FIXED engine produced. A stale C77 record carries error
    strings and no `liveness` block; mixing it in would put its zeros into a
    distribution as if they were measurements."""
    recs = {}
    for fn in sorted(os.listdir(cache)):
        if not fn.endswith(".json"):
            continue
        r = json.load(open(os.path.join(cache, fn), encoding="utf-8"))
        if r.get("liveness") is None:
            continue                        # still-stale C77 record
        recs[r["clip_id"]] = r
    return recs


def quant(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = q * (len(s) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return round(s[lo] + (s[hi] - s[lo]) * (i - lo), 4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("r2_score_dist")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--bin", type=float, default=0.02)
    a = ap.parse_args(argv)

    recs = load(a.cache)
    per = collections.defaultdict(list)         # concept -> [score]
    geom = collections.defaultdict(list)        # concept -> [frac of frame]
    boxpx = collections.defaultdict(list)       # concept -> [box area px]
    perclip = collections.defaultdict(set)      # concept -> {clip}
    exact = collections.Counter()               # concept -> n at exactly .5
    liveness = collections.defaultdict(list)    # control concept -> n_det
    for cid, r in recs.items():
        fw, fh = (r.get("frame_wh") or [0, 0])
        for f in (r.get("frames") or {}).values():
            for d in f.get("det", []):
                if "score" not in d:
                    continue
                c, s = d["concept"], float(d["score"])
                per[c].append(s)
                perclip[c].add(cid)
                if abs(s - 0.5) < 1e-9:
                    exact[c] += 1
                b = d.get("box_xyxy")
                if b and fw and fh:
                    w, h = max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])
                    boxpx[c].append(w * h)
                    geom[c].append((w * h) / float(fw * fh))
        for k, v in ((r.get("liveness") or {}).get("n_det") or {}).items():
            liveness[k].append(int(v))

    out = {"n_clips_complete": len(recs),
           "n_detections": sum(len(v) for v in per.values()),
           "bin_width": a.bin,
           "frame_wh_modal": collections.Counter(
               tuple(r.get("frame_wh") or []) for r in recs.values()
           ).most_common(1)[0][0],
           "concepts": {}}
    for c, xs in sorted(per.items(), key=lambda kv: -len(kv[1])):
        hist = collections.Counter(
            round(math.floor(s / a.bin) * a.bin, 4) for s in xs)
        g = geom.get(c) or []
        out["concepts"][c] = {
            "n": len(xs),
            "n_clips_present": len(perclip[c]),
            "min": round(min(xs), 4), "max": round(max(xs), 4),
            "mean": round(statistics.fmean(xs), 4),
            "median": quant(xs, 0.5),
            "p10": quant(xs, 0.10), "p25": quant(xs, 0.25),
            "p75": quant(xs, 0.75), "p90": quant(xs, 0.90),
            "n_exactly_0.5": exact[c],
            "frac_below_0.55": round(sum(1 for s in xs if s < 0.55) / len(xs), 4),
            "frac_below_0.60": round(sum(1 for s in xs if s < 0.60) / len(xs), 4),
            "frac_below_0.70": round(sum(1 for s in xs if s < 0.70) / len(xs), 4),
            "frac_above_0.90": round(sum(1 for s in xs if s >= 0.90) / len(xs), 4),
            "n_surviving": {f"{t:.2f}": sum(1 for s in xs if s >= t)
                            for t in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                                      0.80, 0.85, 0.90)},
            "box_frac_of_frame": {
                "median": quant(g, 0.5), "p10": quant(g, 0.10),
                "p90": quant(g, 0.90)} if g else None,
            "box_area_px_median": quant(boxpx.get(c) or [], 0.5),
            "hist": {f"{k:.2f}": v for k, v in sorted(hist.items())}}
    out["liveness_control"] = {
        k: {"n_clips": len(v), "n_zero": sum(1 for x in v if x == 0),
            "median": quant(v, 0.5), "max": max(v) if v else None}
        for k, v in sorted(liveness.items())}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"[dist] {out['n_clips_complete']} clips · "
          f"{out['n_detections']} detections", flush=True)
    for c, d in out["concepts"].items():
        print(f"  {c:<14} n={d['n']:>5} clips={d['n_clips_present']:>3} "
              f"min={d['min']:.3f} med={d['median']:.3f} "
              f"<0.55={d['frac_below_0.55']*100:5.1f}% "
              f">=0.9={d['frac_above_0.90']*100:5.1f}% "
              f"@0.5exact={d['n_exactly_0.5']}", flush=True)
    print("DIST_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
