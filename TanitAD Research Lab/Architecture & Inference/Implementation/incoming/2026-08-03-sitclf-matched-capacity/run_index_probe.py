"""B4 addendum - IS THE PERMUTED-FEATURE NULL ACTUALLY A CHANCE FLOOR?

WHY THIS EXISTS
---------------
`run_sitclf_opt.py` and `run_matched_capacity.py` both build their NEG_FEATURE control the same way:
row *t* of clip *i* receives the features of row *t* of clip *j*. That destroys the frame-to-label
correspondence but **PRESERVES THE WITHIN-CLIP FRAME INDEX**. The label is not uniform in *t* -
`anticipation_target` masks the last `lead_s` frames and manoeuvre onsets are not uniformly
distributed inside a clip - so a head could in principle score above the base rate from POSITION
alone, and the "null" would not be a chance floor.

Three arms, all VISION-ONLY-or-less, on the same rows as the ladder:

  IDX_ONLY      a head fed nothing but the normalised frame index (t/T, t, T-t) - the DIRECT
                measurement of how much AP the index alone carries. This is the decisive number:
                whatever it scores is an upper bound on the index's contribution to any arm.
  NULL_CLIP     the ladder's own null, re-scored here for a like-for-like comparison.
  NULL_ROLL     clip-permuted AND time-rolled by a random per-clip offset, which destroys the index
                alignment as well - the STRICTER empirical floor.

Reported per situation with AP-lift + the episode-cluster bootstrap and the paired contrast
NULL_CLIP - NULL_ROLL. Per RETRACTION_LOG ("report the empirical null, not the nominal one") the
larger of the two nulls is what any separation claim must clear.

usage:
  python run_index_probe.py --substrate <npz> --out results_index_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from tanitad.eval.ap_ci import (ap_episode_cluster_bootstrap,        # noqa: E402
                                ap_lift, average_precision,
                                paired_ap_episode_cluster_bootstrap)
from tanitad.eval.sitclf import (causal_window, clip_runs,           # noqa: E402
                                 cluster_folds, head_param_count,
                                 ridge_scores, width_for_param_budget)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_matched_capacity import (WIN, fit_pca, fold_splits,         # noqa: E402
                                  project, train_tf_fold)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def index_features(cc, st, en):
    """(t/T, t/100, (T-1-t)/100) per row - the clip position and nothing else."""
    X = np.zeros((len(cc), 3), np.float32)
    for a, b in zip(st, en):
        T = b - a
        t = np.arange(T, dtype=np.float32)
        X[a:b, 0] = t / max(T - 1, 1)
        X[a:b, 1] = t / 100.0
        X[a:b, 2] = (T - 1 - t) / 100.0
    return X


def permutation(st, en, seed, roll: bool):
    """Row map for the null. `roll=False` reproduces the ladder's NEG_FEATURE exactly."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(st))
    n_rows = int(en[-1])
    out = np.arange(n_rows)
    for i, (s0, e0) in enumerate(zip(st, en)):
        j = perm[i]
        nj = en[j] - st[j]
        ni = e0 - s0
        if roll:
            d = int(rng.integers(0, nj))
            out[s0:e0] = st[j] + ((np.arange(ni) + d) % nj)
        else:
            n = min(ni, nj)
            out[s0:s0 + n] = st[j] + np.arange(n)
            if ni > n:
                out[s0 + n:e0] = st[j] + n - 1
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--substrate", required=True)
    ap_.add_argument("--out", default="results_index_probe.json")
    ap_.add_argument("--n-boot", type=int, default=2000)
    ap_.add_argument("--device", default=None)
    a = ap_.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    z = np.load(a.substrate)
    F = z["F"]
    Yi = z["Y"].astype(np.int64)
    Y = z["Y"].astype(np.float32)
    cc = z["clip_cluster"]
    sits = [str(s) for s in z["situations"]]
    st, en = clip_runs(cc)
    folds = cluster_folds(cc, 2, seed=0)
    _, hist_ok = causal_window(np.zeros((len(cc), 1), np.float32), st, en, WIN)
    V = z["V"].astype(bool) & hist_ok[:, None]
    Vf = V.astype(np.float32)
    splits = [fold_splits(cc, folds, f, seed=0) for f in (0, 1)]
    d128 = width_for_param_budget(417_028, 16, WIN, 3)
    log(f"substrate {F.shape[0]:,} rows, {len(st)} clips, device={device}")

    scores = {}

    # ---- IDX_ONLY: the frame index and nothing else -------------------------
    Xi, _ = causal_window(index_features(cc, st, en), st, en, WIN)
    for tag, fit_fn in (("IDX_ONLY_ridge", "ridge"), ("IDX_ONLY_tf_d128", "tf")):
        out = np.zeros_like(Y, dtype=np.float32)
        for f in (0, 1):
            fit, sel, te = splits[f]
            if fit_fn == "ridge":
                out[te] = ridge_scores(Xi[fit], Y[fit], Vf[fit], Xi[te], lam=1.0)
            else:
                s_te, _ep, _c = train_tf_fold(Xi, Y, Vf, fit, sel, te, 3, d128, device,
                                              seed=0, tag=f"{tag} ")
                out[te] = s_te
        scores[tag] = out
        log(f"  {tag} done")

    # ---- the two nulls on the real features ---------------------------------
    for tag, roll in (("NULL_CLIP", False), ("NULL_ROLL", True)):
        rows = permutation(st, en, seed=0, roll=roll)
        Fs = F[rows]
        for kind in ("ridge_pca16_w8", "tf_pca16_d128"):
            name = f"{tag}__{kind}"
            out = np.zeros_like(Y, dtype=np.float32)
            for f in (0, 1):
                fit, sel, te = splits[f]
                mu, W = fit_pca(Fs, np.flatnonzero(fit), 16, seed=0, device=device)
                X, _ = causal_window(project(Fs, mu, W), st, en, WIN)
                if kind.startswith("ridge"):
                    out[te] = ridge_scores(X[fit], Y[fit], Vf[fit], X[te], lam=10000.0)
                else:
                    s_te, _ep, _c = train_tf_fold(X, Y, Vf, fit, sel, te, 16, d128, device,
                                                  seed=0, tag=f"{name} ")
                    out[te] = s_te
                del X
            scores[name] = out
            log(f"  {name} done")
        del Fs

    np.savez_compressed(Path(a.out).with_suffix(".scores.npz"),
                        clip_cluster=cc, y=z["Y"], valid=V.astype(np.uint8),
                        situations=np.array(sits), **scores)

    res = {"_what": "is the clip-permuted NEG_FEATURE null a chance floor?",
           "substrate": str(a.substrate),
           "arms": {"IDX_ONLY_ridge": "ridge on (t/T, t, T-1-t) over an 8-frame window",
                    "IDX_ONLY_tf_d128": f"CausalSitHead d={d128} on the same 3 index channels "
                                        f"({head_param_count(3, WIN, d128, 3):,} params)",
                    "NULL_CLIP": "the ladder's own null - features permuted across clips, "
                                 "frame index PRESERVED",
                    "NULL_ROLL": "permuted across clips AND time-rolled by a random per-clip "
                                 "offset - index alignment destroyed"},
           "n_boot": a.n_boot, "per_situation": {}}
    for i, s in enumerate(sits):
        m = V[:, i]
        yv = Yi[m, i]
        eid = cc[m]
        row = {"n_scorable": int(m.sum()), "n_pos": int(yv.sum()),
               "base_rate": round(float(yv.mean()), 6), "arms": {}, "paired": {}}
        for n, sc in scores.items():
            v = sc[m, i].astype(np.float64)
            r_ = ap_episode_cluster_bootstrap(yv, v, eid, n_boot=a.n_boot, lift=True)
            r_["ap"] = round(average_precision(yv, v), 5)
            row["arms"][n] = r_
            log(f"  {s} {n:>26}: AP={r_['ap']:.5f} lift={r_['point']:.3f} "
                f"[{r_['lo']:.3f},{r_['hi']:.3f}]")
        for kind in ("ridge_pca16_w8", "tf_pca16_d128"):
            row["paired"][f"NULL_CLIP - NULL_ROLL ({kind})"] = (
                paired_ap_episode_cluster_bootstrap(
                    yv, scores[f"NULL_CLIP__{kind}"][m, i].astype(np.float64),
                    scores[f"NULL_ROLL__{kind}"][m, i].astype(np.float64), eid,
                    n_boot=a.n_boot, lift=True))
        res["per_situation"][s] = row
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
