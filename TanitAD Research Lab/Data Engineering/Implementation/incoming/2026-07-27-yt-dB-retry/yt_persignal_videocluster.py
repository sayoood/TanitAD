#!/usr/bin/env python3
"""Per-signal IDM read on the FULL n=343 YouTube corpus, with the resampling unit
CORRECTED from clip to VIDEO.

WHY THIS EXISTS
---------------
The 2026-07-26 D-B run banked 343 clip-latents, but they come from only **55 distinct
videos** (43 channels). Clips cut from the same continuous upload share a camera, a
mount, a focal length, a driver and a road — they are NOT independent draws. Measured
on the harvest pointers, the Kish effective sample size is **n_eff = 35.3**, i.e. a
**design effect of 9.72x**; a clip-level bootstrap therefore understates the standard
error by ~sqrt(9.72) = 3.1x.

`yt_persignal.py` (the shipped instrument, 2026-07-26) resamples CLIPS. That is the
same class of error as using `overlapping_holdout_se` instead of the episode-cluster
bootstrap: the correlated unit is not the one being resampled.

WHAT THIS DOES
--------------
It does NOT reimplement the statistic. It imports `yt_persignal` and calls its
`summarise_channel` / `spread_ratio_ci` **verbatim** — those functions resample
whatever list of per-unit arrays they are handed. Passing one array per VIDEO (instead
of one per clip) turns the identical code into a video-cluster bootstrap. So the two
arms below differ ONLY in the grouping, never in the estimator implementation.

Both arms are reported:
  * `unit_clip`  — comparable to the 2026-07-26 n=200 numbers
  * `unit_video` — the decision-grade interval

NEVER `overlapping_holdout_se`. Reported PER CORPUS, never pooled.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
sys.path.insert(0, "/workspace/tmp/yt_scaleup/scripts")

import idm_head as ih                       # noqa: E402
import yt_persignal as YP                   # noqa: E402  (the shipped instrument)


def group_by_video(rows):
    """rows: list of (video_id, 1-D np array of that clip's windows) -> list of
    one concatenated array PER VIDEO (the corrected cluster)."""
    g = collections.OrderedDict()
    for vid, arr in rows:
        g.setdefault(vid, []).append(arr)
    return [np.concatenate(v) for v in g.values()], list(g.keys())


def retag(summary, unit):
    """Rewrite the estimator string so a copied number can never lose its unit."""
    if isinstance(summary, dict):
        if "estimator" in summary:
            summary["estimator"] = (
                f"{unit}-cluster bootstrap, n_boot={YP.N_BOOT}; resampling unit = "
                f"{unit}. NOT overlapping_holdout_se.")
        for v in summary.values():
            if isinstance(v, dict):
                retag(v, unit)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="/workspace/tmp/yt_scaleup/latents")
    ap.add_argument("--ref-latents",
                    default="/workspace/tmp/branchb_eval/lat_flagshipv1")
    ap.add_argument("--ref-tags", default="va_")
    ap.add_argument("--ref-max", type=int, default=40)
    ap.add_argument("--pointers", default="/workspace/tmp/yt_scaleup/w0/pointers.jsonl,"
                                          "/workspace/tmp/yt_scaleup/w1/pointers.jsonl")
    ap.add_argument("--exclude-videos", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out",
                    default="/workspace/tmp/yt_scaleup/results/persignal_n343_videocluster.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"[vc] device={device} SCALAR_NAMES={ih.SCALAR_NAMES}", flush=True)
    labeler = YP.PL.build_labeler(device, seed=args.seed)

    # clip file -> video_id, from the harvest pointers (authoritative)
    ptr = {}
    for p in args.pointers.split(","):
        p = p.strip()
        if not p or not os.path.exists(p):
            continue
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                ptr[os.path.basename(r.get("clip_path", ""))] = r["video_id"]

    excl = {v.strip() for v in args.exclude_videos.split(",") if v.strip()}

    yt = YP.load_latent_dir(args.latents)
    print(f"[vc] {len(yt)} YouTube clip-latents", flush=True)

    rows = {c: [] for c in ih.SCALAR_NAMES}     # (video_id, arr) per clip
    n_novid = 0
    for fname, vid, z in yt:
        if not vid:
            vid = ptr.get(fname)
        if not vid:
            vid = f"__unknown_{fname}"
            n_novid += 1
        if vid in excl:
            continue
        s = YP.run_head(labeler, z, device)
        if s is None:
            continue
        for i, c in enumerate(ih.SCALAR_NAMES):
            rows[c].append((vid, s[:, i]))

    vids = sorted({v for v, _ in rows[ih.SCALAR_NAMES[0]]})
    counts = collections.Counter(v for v, _ in rows[ih.SCALAR_NAMES[0]])
    n_clips = len(rows[ih.SCALAR_NAMES[0]])
    neff = (n_clips ** 2) / sum(c * c for c in counts.values())
    print(f"[vc] {n_clips} clips from {len(vids)} videos; "
          f"Kish n_eff={neff:.1f} design_effect={n_clips/neff:.2f}x", flush=True)

    # ---- PhysicalAI reference (SEPARATE corpus, never pooled) ----
    refdir = Path(args.ref_latents)
    ref_rows = {c: [] for c in ih.SCALAR_NAMES}
    rfiles = sorted(refdir.glob(f"{args.ref_tags}*.pt"))[:args.ref_max] if refdir.is_dir() else []
    for p in rfiles:
        d = torch.load(p, map_location="cpu", weights_only=False)
        s = YP.run_head(labeler, d["z"].float(), device)
        if s is None:
            continue
        for i, c in enumerate(ih.SCALAR_NAMES):
            ref_rows[c].append((p.stem, s[:, i]))
    print(f"[vc] {len(rfiles)} PhysicalAI reference clips", flush=True)

    out = {
        "experiment": "yt_dB_retry_persignal_n343_video_vs_clip_cluster",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "labeler": "v1 frozen encoder + IDMHead{parity rigA[:60]+rigB[:60]+comma[:40]}",
        "scalar_names": list(ih.SCALAR_NAMES),
        "n_youtube_clips": n_clips,
        "n_youtube_videos": len(vids),
        "clips_per_video": dict(counts),
        "kish_n_eff_videos": round(neff, 2),
        "design_effect_clip_vs_video": round(n_clips / neff, 3),
        "n_physicalai_ref_clips": len(rfiles),
        "no_ground_truth_warning": (
            "YouTube has NO ground truth. NOTHING here is an accuracy number. These are "
            "DISTRIBUTIONS and PHYSICAL-PLAUSIBILITY rates only."),
        "estimator_note": (
            "Two arms, identical code (yt_persignal.summarise_channel / "
            "spread_ratio_ci), differing ONLY in the resampling unit. "
            "unit_video is the decision-grade interval. NEVER overlapping_holdout_se."),
        "excluded_video_ids": sorted(excl),
    }

    for unit in ("clip", "video"):
        yt_arm, ref_arm, ratio_arm = {}, {}, {}
        for c in ih.SCALAR_NAMES:
            if unit == "clip":
                yv = [a for _, a in rows[c]]
                rv = [a for _, a in ref_rows[c]]
            else:
                yv, _ = group_by_video(rows[c])
                rv = [a for _, a in ref_rows[c]]   # ref clips are already episode-level
            yt_arm[c] = retag(YP.summarise_channel(c, yv, seed=args.seed), unit)
            if rv:
                ref_arm[c] = retag(YP.summarise_channel(c, rv, seed=args.seed), "clip")
                r = YP.spread_ratio_ci(yv, rv, seed=args.seed)
                if r:
                    r["estimator"] = (
                        f"independent bootstrap; YouTube unit = {unit}, PhysicalAI unit "
                        f"= clip; n_boot={YP.N_BOOT}; statistic = "
                        f"(p95-p05)_youtube / (p95-p05)_physicalai")
                    ratio_arm[c] = r
        out[f"unit_{unit}"] = {
            "youtube_out_of_corpus": yt_arm,
            "physicalai_reference_SEPARATE_never_pooled": ref_arm,
            "spread_ratio_youtube_over_physicalai": ratio_arm,
        }
        print(f"[vc] unit={unit} done ({time.time()-t0:.0f}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[vc] WROTE {args.out} ({time.time()-t0:.0f}s)")

    # compact console summary
    for c in ih.SCALAR_NAMES:
        rc = out["unit_clip"]["spread_ratio_youtube_over_physicalai"].get(c)
        rv = out["unit_video"]["spread_ratio_youtube_over_physicalai"].get(c)
        if rc and rv:
            print(f"  {c:11s} ratio clip {rc['spread_ratio_yt_over_physicalai']:.3f} "
                  f"{rc['ci95']} sep={rc['excludes_1']}   |   "
                  f"video {rv['spread_ratio_yt_over_physicalai']:.3f} "
                  f"{rv['ci95']} sep={rv['excludes_1']}")


if __name__ == "__main__":
    main()
