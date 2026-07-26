"""D-B retry — PER-SIGNAL IDM read on out-of-corpus YouTube video.

Sayed asked for speed / yaw-rate / steering / acceleration SEPARATELY, each with its
own interval. The shipped YouTube pipeline cannot answer that: `pseudo_label.py`
persists only `speed_*` summaries + `yaw_rate_abs_mean_CAVEAT`, and
`run_youtube_pilot_downstream.py` emits only {speed_r2, yaw_r2, ade_2s}. `steer` and
`long_accel` are dropped before anything is written to disk.

This script re-runs the SAME labeler (same recipe, same seed) over the persisted
YouTube latents and emits ALL FOUR channels.

⚠️ WHAT THIS CAN AND CANNOT BE
YouTube has NO ground truth. There is therefore NO R², NO MAE, NO accuracy of any
kind here, and none is reported. What IS measurable without GT:
  * the DISTRIBUTION of each channel on out-of-corpus video,
  * the PHYSICAL-PLAUSIBILITY rate of each channel (a label outside physical limits
    is wrong regardless of what the truth is — this is a one-sided accuracy bound),
  * the DISTRIBUTION SHIFT vs the in-corpus (PhysicalAI) reference, reported
    PER CORPUS and never pooled.
Accuracy per signal is only measurable where GT exists (PhysicalAI val / comma2k19)
and is reported from those runs, not invented here.

ESTIMATOR (named, as required): **clip-cluster bootstrap**, n_boot=2000, seed=0 —
resample the harvested CLIPS with replacement and recompute the statistic over the
pooled windows of the resampled clips. The clip is the correlated unit here exactly
as the episode is in `taniteval.ci.episode_cluster_bootstrap`; windows within a clip
are strongly dependent, so a per-window interval would be far too narrow.
`overlapping_holdout_se` is NOT used — it biases both the interval AND the point
estimate (CLAUDE.md, up to x4.15 with sign flips).

PHYSICAL LIMITS are the IDM-v2 winsorisation limits (…/2026-07-26-idm-v2/):
yaw 1.5 rad/s, accel 12 m/s^2, speed 0-60 m/s, steer 1.0.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
sys.path.insert(0, "/workspace/tmp/yt_scaleup/scripts")
import idm_head as ih                                                # noqa: E402

# the labeler recipe, imported verbatim from the shipped pilot script so this cannot
# silently diverge from what produced the pseudo-labels
import pseudo_label as PL                                            # noqa: E402

# IDM-v2 winsorisation limits = the physical-plausibility envelope
LIMITS = {"speed": (0.0, 60.0), "yaw_rate": (-1.5, 1.5),
          "steer": (-1.0, 1.0), "long_accel": (-12.0, 12.0)}
UNITS = {"speed": "m/s", "yaw_rate": "rad/s", "steer": "rad (normalised)",
         "long_accel": "m/s^2"}
N_BOOT = 2000


def clip_cluster_bootstrap(per_clip_values, stat, n_boot=N_BOOT, seed=0):
    """95% CI of `stat` over CLIPS resampled with replacement.

    `per_clip_values` : list of 1-D np arrays, one per clip (its windows).
    `stat`            : callable(pooled_1d_array) -> float
    Returns (point, lo, hi, n_clips, n_windows).
    """
    rng = np.random.default_rng(seed)
    n = len(per_clip_values)
    if n == 0:
        return None, None, None, 0, 0
    pooled = np.concatenate(per_clip_values)
    point = float(stat(pooled))
    if n == 1:
        return point, None, None, 1, int(pooled.size)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samp = np.concatenate([per_clip_values[i] for i in idx])
        draws[b] = stat(samp)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), n, int(pooled.size)


def summarise_channel(name, per_clip_values, seed=0):
    lo_lim, hi_lim = LIMITS[name]

    def frac_impossible(a):
        return float(((a < lo_lim) | (a > hi_lim)).mean())

    out = {"channel": name, "units": UNITS[name],
           "physical_limits": [lo_lim, hi_lim],
           "estimator": (f"clip-cluster bootstrap, n_boot={N_BOOT}, seed={seed}; "
                         "resampling unit = harvested clip (NOT window, NOT "
                         "overlapping_holdout_se)")}
    for label, fn in (("mean", np.mean), ("median", np.median),
                      ("p05", lambda a: np.percentile(a, 5)),
                      ("p95", lambda a: np.percentile(a, 95)),
                      ("std", np.std),
                      ("frac_outside_physical_limits", frac_impossible)):
        pt, lo, hi, nc, nw = clip_cluster_bootstrap(per_clip_values, fn, seed=seed)
        out[label] = None if pt is None else round(pt, 5)
        out[f"{label}_ci95"] = ([round(lo, 5), round(hi, 5)]
                                if lo is not None else None)
        out["n_clips"], out["n_windows"] = nc, nw
    pooled = np.concatenate(per_clip_values) if per_clip_values else np.zeros(0)
    if pooled.size:
        out["min"] = round(float(pooled.min()), 5)
        out["max"] = round(float(pooled.max()), 5)
    return out


def load_latent_dir(latdir, pattern="yt_*.pt"):
    files = sorted(glob.glob(str(Path(latdir) / pattern)))
    out = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        z = d["z"] if isinstance(d, dict) and "z" in d else d
        out.append((os.path.basename(f), d.get("video_id") if isinstance(d, dict) else None,
                    z.float()))
    return out


def run_head(labeler, z, device, k=4, stride=2):
    zw, _s, _t = ih.build_windows(z, torch.zeros(z.shape[0], 4),
                                  torch.zeros(z.shape[0], 2), k=k, stride=stride)
    if zw.shape[0] == 0:
        return None
    with torch.no_grad():
        o = labeler(zw.to(device))
    return o["scalars"].cpu().numpy()          # [N, 4] in SCALAR_NAMES order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="/workspace/tmp/yt_scaleup/latents")
    ap.add_argument("--ref-latents", default="/workspace/tmp/branchb_eval/lat_flagshipv1",
                    help="in-corpus PhysicalAI reference latents (reported SEPARATELY)")
    ap.add_argument("--ref-tags", default="va_",
                    help="prefix of reference latent tags to use as the in-corpus arm")
    ap.add_argument("--ref-max", type=int, default=40)
    ap.add_argument("--exclude-videos", default="",
                    help="comma-separated video_ids to EXCLUDE (corpus contamination)")
    ap.add_argument("--out", default="/workspace/tmp/yt_scaleup/results/persignal.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"[persignal] device={device} SCALAR_NAMES={ih.SCALAR_NAMES}", flush=True)
    labeler = PL.build_labeler(device, seed=args.seed)

    excl = {v.strip() for v in args.exclude_videos.split(",") if v.strip()}

    # ---------------- YouTube (out-of-corpus, NO ground truth) ----------------
    yt = load_latent_dir(args.latents)
    print(f"[persignal] {len(yt)} YouTube clip-latents", flush=True)
    chans_all = {c: [] for c in ih.SCALAR_NAMES}
    chans_clean = {c: [] for c in ih.SCALAR_NAMES}
    per_clip_rows, n_excluded = [], 0
    for fname, vid, z in yt:
        s = run_head(labeler, z, device)
        if s is None:
            continue
        row = {"file": fname, "video_id": vid, "n_windows": int(s.shape[0])}
        for i, c in enumerate(ih.SCALAR_NAMES):
            chans_all[c].append(s[:, i])
            row[f"{c}_mean"] = round(float(s[:, i].mean()), 4)
        contaminated = vid in excl
        row["excluded_as_contaminated"] = contaminated
        if contaminated:
            n_excluded += 1
        else:
            for i, c in enumerate(ih.SCALAR_NAMES):
                chans_clean[c].append(s[:, i])
        per_clip_rows.append(row)

    yt_all = {c: summarise_channel(c, v, seed=args.seed) for c, v in chans_all.items()}
    yt_clean = {c: summarise_channel(c, v, seed=args.seed)
                for c, v in chans_clean.items()} if n_excluded else None

    # ---------------- PhysicalAI in-corpus reference (SEPARATE, never pooled) ----
    ref = {}
    refdir = Path(args.ref_latents)
    if refdir.is_dir():
        rfiles = sorted(refdir.glob(f"{args.ref_tags}*.pt"))[:args.ref_max]
        rch = {c: [] for c in ih.SCALAR_NAMES}
        for p in rfiles:
            d = torch.load(p, map_location="cpu", weights_only=False)
            s = run_head(labeler, d["z"].float(), device)
            if s is None:
                continue
            for i, c in enumerate(ih.SCALAR_NAMES):
                rch[c].append(s[:, i])
        if rfiles:
            ref = {c: summarise_channel(c, v, seed=args.seed) for c, v in rch.items()}
            ref["_n_ref_clips"] = len(rfiles)
            ref["_source"] = str(refdir) + f" tags={args.ref_tags}*"
    else:
        ref = {"_skipped": f"reference latent dir not found: {refdir}"}

    out = {
        "experiment": "db_retry_persignal_idm_on_youtube",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "labeler": "v1 frozen encoder + IDMHead{parity rigA[:60]+rigB[:60]+comma[:40]}",
        "scalar_names": list(ih.SCALAR_NAMES),
        "estimator": (f"clip-cluster bootstrap, n_boot={N_BOOT}, seed={args.seed}; "
                      "resampling unit = clip. NOT overlapping_holdout_se."),
        "no_ground_truth_warning": (
            "YouTube has NO ground truth. NOTHING here is an accuracy number. These are "
            "DISTRIBUTIONS and PHYSICAL-PLAUSIBILITY rates. Per-signal ACCURACY is only "
            "measurable on GT-bearing corpora (PhysicalAI val / comma2k19)."),
        "known_defects_carried": [
            "physically impossible yaw labels are still present in the TRAIN label set "
            "(IDM-v2: 9/4195 val windows move pooled yaw R2 0.105 -> 0.497)",
            "comma2k19 heading derivation is arctan2 of ENU velocity, undefined at "
            "standstill — the labeler's yaw channel inherits this",
            "long_accel is still in SCALAR_NAMES despite IDM-v2's pre-committed "
            "recommendation to remove it (a perfect kinematic estimator caps at "
            "R2 0.188 against the CAN label)",
        ],
        "n_youtube_clips": len(yt),
        "n_excluded_contaminated": n_excluded,
        "excluded_video_ids": sorted(excl),
        "youtube_all_clips": yt_all,
        "youtube_contamination_excluded": yt_clean,
        "physicalai_reference_SEPARATE_never_pooled": ref,
        "per_clip": per_clip_rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[persignal] WROTE {args.out}")
    for c in ih.SCALAR_NAMES:
        a = yt_all[c]
        print(f"  {c:11s} mean {a['mean']} CI {a['mean_ci95']}  "
              f"outside-limits {a['frac_outside_physical_limits']} "
              f"CI {a['frac_outside_physical_limits_ci95']}")


if __name__ == "__main__":
    main()
