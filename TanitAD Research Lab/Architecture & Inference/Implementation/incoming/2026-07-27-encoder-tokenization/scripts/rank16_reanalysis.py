"""R1 — the PAIRED contrasts the rank-16 dose-response was never given.

WHY THIS EXISTS
---------------
The program's most load-bearing encoder claim is a "monotone dose-response" that
"vision enters the downstream predictor best at rank ~16":

    ego 3.659x -> +k16 3.685x -> +k64 3.000x -> +k256 2.116x -> +k2048 1.59x

It is quoted in at least four documents and it is currently steering the v5
encoder architecture (a Perceiver-style fixed-16-latent resampler was proposed
on the strength of it). Every one of those numbers is an UNPAIRED point estimate
of `AP / base_rate`. The source artifact
(`2026-07-26-situation-semantics/artifacts/t1_probe.json`) contains a paired
bootstrap of every arm **against chance**, and NOT ONE paired contrast between
two arms of the ladder. So the ladder's shape has never had an interval on it.

The raw held-out scores are banked (`artifacts/t1_heldout_scores.npz`, 50,119
frames x 26 arm-scores + `clip_cluster`), so the missing test costs seconds.

WHAT THIS SCRIPT MEASURES
-------------------------
  R1-A  the CONCATENATED ladder, paired:  (ego_win + img_pcaK) - ego_win  for
        K in {16, 64, 256}, and the adjacent rungs against each other. Answers
        "does vision ADD anything at its best rank, and is the degradation real?"

  R1-B  the IMAGE-ONLY ladder, paired: img_pca{16,64,256} vs img_t (raw 2048).
        THIS IS THE DISCRIMINATOR. If 16 dims were the true information content
        of the visual state, the image-ALONE arm must also peak at 16 and fall.
        If the image-alone arm is FLAT across rank while the concatenated arm
        falls, the "rank-16 optimum" is a property of the READER (one shared
        ridge lambda over a concatenated [ego | image] block at n=198 train
        clip-clusters), not of the predictor's information need.

VALIDATION IN BOTH DIRECTIONS (both able to fail, per the operating standard)
  C-FID  every arm's AP and AP_over_base recomputed from the npz must reproduce
         the published t1_probe.json values to 1e-6. If the npz and the JSON
         disagree, nothing here is quotable.
  C-NEG  `img_t_SHUFFLED` (column-shuffled features, a deliberately destroyed
         input) must NOT separate from chance. If it separates, the harness leaks.

ESTIMATOR: episode-cluster bootstrap over the 322 held-out CLIP CLUSTERS,
B=2000, seed=0, using the program's own `taniteval.ci._draws` and the SAME
`average_precision` as the source stream (copied verbatim below so this file is
self-contained and character-identical). The paired form shares the resampled
clusters between the two arms of every contrast, so per-clip difficulty cancels.
`overlapping_holdout_se` is NOT used anywhere.

Run:
  python rank16_reanalysis.py --out ../artifacts/rank16_reanalysis.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(
    HERE, "..", "..", "2026-07-26-situation-semantics", "artifacts"))
REPO = os.path.abspath(os.path.join(HERE, *([".."] * 6)))   # scripts -> repo root


# --- verbatim from 2026-07-26-situation-semantics/scripts/t1_probe.py --------
def average_precision(y, s):
    """Step-interpolated AP, character-identical to `h2c_stats.average_precision`."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    yt = y[o]
    tp = np.cumsum(yt)
    fp = np.cumsum(1.0 - yt)
    P = tp / np.maximum(tp + fp, 1e-12)
    R = tp / yt.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], R])) * P))
# ---------------------------------------------------------------------------


def _pct(d):
    """2.5/97.5 percentile bounds; refuses on too few usable draws (not NaN-silent)."""
    d = d[np.isfinite(d)]
    if d.size <= 50:
        return float("nan"), float("nan"), int(d.size)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi), int(d.size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "artifacts",
                                                  "rank16_reanalysis.json"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--reader", default="ridge", choices=["ridge", "logistic", "both"])
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(REPO, "taniteval"))
    from taniteval.ci import _draws, episode_index          # noqa: E402

    z = np.load(os.path.join(SRC, "t1_heldout_scores.npz"), allow_pickle=True)
    pub = json.load(open(os.path.join(SRC, "t1_probe.json")))

    y = np.asarray(z["y_NOT_T_seen"], float)
    eid = np.asarray(z["clip_cluster"])
    base = float(y.mean())

    uniq, idx_by_ep = episode_index(eid)
    draws = list(_draws(uniq, idx_by_ep, args.boot, SEED))
    ok = np.array([bool(y[s_].sum() > 0) for s_ in draws])

    def ap_draws(s):
        out = np.empty(len(draws))
        for i, sel in enumerate(draws):
            out[i] = average_precision(y[sel], s[sel]) if ok[i] else np.nan
        return out

    readers = ["ridge", "logistic"] if args.reader == "both" else [args.reader]
    res = {
        "what": "PAIRED contrasts for the rank-16 dose-response ladder",
        "source_scores": "2026-07-26-situation-semantics/artifacts/t1_heldout_scores.npz",
        "estimator": ("episode-cluster bootstrap over held-out CLIP CLUSTERS "
                      "(taniteval.ci._draws), B=%d, seed=%d; paired = same "
                      "resampled clusters for both arms of a contrast" % (args.boot, SEED)),
        "NOT_used": "overlapping_holdout_se",
        "n_frames": int(y.size), "n_positives": int(y.sum()),
        "base_rate": base, "n_clip_clusters": int(len(uniq)),
        "readers": readers, "contrasts": {}, "arms": {},
    }

    # ---------------------------------------------------------------- C-FID
    fid = {"tol": 1e-6, "checked": [], "all_match": True}
    for arm, a in pub["arms"].items():
        for rd in ("ridge", "logistic"):
            key = f"{arm}_{rd}"
            if key not in z.files or rd not in a:
                continue
            got = average_precision(y, np.asarray(z[key], float))
            want = float(a[rd]["AP"]["point"])
            ok_ = abs(got - want) < 1e-6
            fid["all_match"] &= bool(ok_)
            fid["checked"].append({"arm": key, "recomputed": round(got, 8),
                                   "published": want, "match": bool(ok_)})
    res["C_FID_reproduce_published_AP"] = fid

    # ------------------------------------------------------- per-arm draws
    cache = {}
    for rd in readers:
        for arm in sorted({k.rsplit("_", 1)[0] for k in z.files
                           if k.endswith("_" + rd)}):
            key = f"{arm}_{rd}"
            s = np.asarray(z[key], float)
            d = ap_draws(s)
            cache[key] = d
            pa = average_precision(y, s)
            lo, hi, n = _pct(d)
            res["arms"][key] = {"AP": round(pa, 6), "AP_lo": round(lo, 6),
                                "AP_hi": round(hi, 6), "AP_over_base": round(pa / base, 4),
                                "n_draws_used": n}

    # ---------------------------------------------------------------- C-NEG
    rng = np.random.default_rng(SEED)
    rand_score = rng.random(len(y))
    ch = ap_draws(rand_score)
    neg = {}
    for rd in readers:
        k = f"img_t_SHUFFLED_{rd}"
        if k not in cache:
            continue
        d = cache[k] - ch
        lo, hi, n = _pct(d)
        neg[k] = {"delta_vs_random_ranker": round(float(np.nanmean(d)), 6),
                  "lo": round(lo, 6), "hi": round(hi, 6),
                  "separated": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                  "n_draws_used": n}
    res["C_NEG_shuffled_must_not_separate"] = neg

    # ------------------------------------------------------------ contrasts
    PAIRS = [
        # R1-A the CONCATENATED ladder — does vision ADD, and is the fall real?
        ("R1-A", "ego_win+img_pca16", "ego_win",
         "THE LOAD-BEARING ONE: does rank-16 vision add anything to ego?"),
        ("R1-A", "ego_win+img_pca64", "ego_win", "does rank-64 vision add to ego?"),
        ("R1-A", "ego_win+img_pca256", "ego_win", "does rank-256 vision add to ego?"),
        ("R1-A", "ego_win+img_pca16", "ego_win+img_pca64",
         "is the k16->k64 'degradation' separated?"),
        ("R1-A", "ego_win+img_pca64", "ego_win+img_pca256",
         "is the k64->k256 degradation separated?"),
        ("R1-A", "ego_win+img_pca16", "ego_win+img_pca256",
         "is the k16->k256 degradation separated?"),
        ("R1-A", "ego_win", "ego_t", "is the ego window itself worth its 16 dims?"),
        # R1-B the IMAGE-ONLY ladder — THE DISCRIMINATOR
        ("R1-B", "img_pca16", "img_t",
         "DISCRIMINATOR: image-alone rank16 vs raw2048 - flat or peaked?"),
        ("R1-B", "img_pca64", "img_t", "image-alone rank64 vs raw2048"),
        ("R1-B", "img_pca256", "img_t", "image-alone rank256 vs raw2048"),
        ("R1-B", "img_pca16", "img_pca64", "image-alone k16 vs k64"),
        ("R1-B", "img_pca16", "img_pca256", "image-alone k16 vs k256"),
        ("R1-B", "img_t", "img_t_SHUFFLED",
         "does the raw 2048-d image state carry signal at all?"),
    ]
    for rd in readers:
        for tag, a_name, b_name, why in PAIRS:
            ka, kb = f"{a_name}_{rd}", f"{b_name}_{rd}"
            if ka not in cache or kb not in cache:
                continue
            d = cache[ka] - cache[kb]
            lo, hi, n = _pct(d)
            pa = res["arms"][ka]["AP"]
            pb = res["arms"][kb]["AP"]
            res["contrasts"][f"[{tag}] {a_name} - {b_name} ({rd})"] = {
                "question": why,
                "AP_a": pa, "AP_b": pb,
                "ratio_a": res["arms"][ka]["AP_over_base"],
                "ratio_b": res["arms"][kb]["AP_over_base"],
                "delta_AP_point": round(pa - pb, 6),
                "delta_boot_median": round(float(np.nanmedian(d)), 6),
                "lo": round(lo, 6), "hi": round(hi, 6),
                "separated": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                "direction": ("a>b" if pa > pb else "b>a"),
                "n_draws_used": n,
            }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    # ------------------------------------------------------------- console
    print(f"C-FID reproduce published AP : all_match={fid['all_match']} "
          f"({len(fid['checked'])} arms)")
    for k, v in neg.items():
        print(f"C-NEG {k:28s} separated={v['separated']} "
              f"[{v['lo']:+.5f}, {v['hi']:+.5f}]  (must be False)")
    print()
    print(f"{'contrast':62s} {'dAP':>9s} {'95% CI':>22s}  sep")
    for k, v in res["contrasts"].items():
        print(f"{k:62s} {v['delta_AP_point']:+9.5f} "
              f"[{v['lo']:+8.5f},{v['hi']:+8.5f}]  {'YES' if v['separated'] else 'no'}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
