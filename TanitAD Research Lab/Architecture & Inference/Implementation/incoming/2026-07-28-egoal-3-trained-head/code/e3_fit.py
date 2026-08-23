#!/usr/bin/env python3
"""E-GOAL-3 S2 -- TRAIN THE GOAL HEAD. Real per-window predictions, not a
resampled residual.

Two training configurations, both reported:

  T-OOF    5 EPISODE-DISJOINT folds over the 600 val episodes
           (`eg_common.clip_folds`, seed 0, IMPORTED). Out-of-fold predictions
           on all 13,198 windows. This is the brief's literal ask.
  T-TRAIN  fitted on the PARITY TRAIN CORPUS `physicalai-train-e438721ae894`
           (2376 episodes, dense grid), predicting all 13,198 val windows with
           ZERO val episodes in training -- verified by pose-content fingerprint
           (`e3_leak.py`), not by filename. This is the DEPLOYABLE
           configuration, and it is what E-GOAL-1 §6.3 asked for.

Arms (PRE_REGISTRATION §3.1). Every fitted arm uses `eg_fit.fit_predict`
IMPORTED, with E-GOAL-1's hyper-parameters unchanged, so extra columns can only
help an arm through held-out generalisation:

  H_ego          all 10 columns                       ⭐ THE TREATMENT
  H_nohist       history dropped (6 columns)
  H_noise_hist   ⛔ the 4 history columns REPLACED by Gaussian noise matched to
                 each column's own mean and SD -- capacity fixed, information
                 removed. THE DELIBERATELY FAILING INPUT the C31 contrast needs.
  H_v0           `v` only -- the parent head's ego content
  N_SHUF         ⛔ all 10 columns PERMUTED ACROSS EPISODES -- the second
                 deliberately failing input, re-run at the ACTUAL n
  CV_head        no fit: yhat = v * 2.0 s
  P_ORACLE_TRUE  no fit: yhat = y_long  (positive control / bound, never a
                 capability -- retraction class BOUND-QUOTED-AS-CAPABILITY)

Run (dev box; the features were extracted on pod2 from the PARITY caches and
carried here -- no parity-dependent step runs on this host):
    OMP_NUM_THREADS=6 python e3_fit.py --val <val.npz> --train <train.npz>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STREAM = HERE.parent
ARCH = STREAM.parent
EG1 = ARCH / "2026-07-27-egoal-1-lead-vehicle"
sys.path.insert(0, str(EG1 / "code"))

from eg_common import SEED, clip_folds, r4  # noqa: E402
from eg_fit import fit_predict  # noqa: E402

COLS = ["v", "ax", "ay", "curv", "abs_curv", "yawrate",
        "dv_0p5", "dv_1p0", "v_lag_0p5", "v_lag_1p0"]
HIST = ["dv_0p5", "dv_1p0", "v_lag_0p5", "v_lag_1p0"]
NOHIST_IDX = [i for i, c in enumerate(COLS) if c not in HIST]
HIST_IDX = [COLS.index(c) for c in HIST]
HORIZON_S = 2.0

#: ⭐ AMENDMENT (PRE_REGISTRATION §10, A1) -- the SPEED-DIFFERENCING LADDER.
#: Added AFTER the primary placement returned `H_ego` vs `H_nohist` = only ~2 %
#: of the recovery, against E-GOAL-2's "64 % of the recovery is speed history".
#: `H_nohist` still contains `ax = (v[L] - v[L-1])/DT`, i.e. 0.1 s of speed
#: history -- so "no history" meant "no 0.5-1.0 s history", NOT "no history".
#: These two arms separate INSTANTANEOUS STATE from SPEED DIFFERENCING OF ANY
#: LENGTH, which is the question E-GOAL-2's claim actually turns on.
INST = ["v", "ay", "curv", "abs_curv", "yawrate"]   # no speed difference at all
V_AX = ["v", "ax"]                                  # v + 0.1 s of speed history
INST_IDX = [COLS.index(c) for c in INST]
V_AX_IDX = [COLS.index(c) for c in V_AX]

#: names that would be future-derived. Nothing here may reach an arm.
ORACLE_NAMES = frozenset({"y_long", "y_lat", "gt", "a_gt", "head_deg",
                          "v_target", "vt_valid", "vt_lookahead", "v_fut_2s"})


def assert_no_oracle(names):
    bad = sorted(set(map(str, names)) & ORACLE_NAMES)
    if bad:
        raise RuntimeError(f"ORACLE FIELD IN ARM INPUT: {bad}")


def noisy_copy(X, rng, stats=None):
    """The 4 history columns replaced by Gaussian noise matched to each column's
    own mean and SD. SAME SHAPE, SAME COLUMN POSITIONS, NO INFORMATION."""
    out = X.copy()
    st = {}
    for j in HIST_IDX:
        mu, sd = ((float(np.mean(X[:, j])), float(np.std(X[:, j])))
                  if stats is None else stats[COLS[j]])
        out[:, j] = rng.normal(mu, sd, len(X))
        st[COLS[j]] = (mu, sd)
    return out, st


def shuffled_across_episodes(X, epi, rng):
    """Permute the whole feature block ACROSS episodes (eg_fit.build_arms style):
    every window keeps a real feature vector, but from a DIFFERENT episode."""
    uniq = np.unique(epi)
    idx_by = {u: np.flatnonzero(epi == u) for u in uniq}
    order = rng.permutation(len(uniq))
    out = np.empty_like(X)
    for i, u in enumerate(uniq):
        src, dst = idx_by[uniq[order[i]]], idx_by[u]
        out[dst] = X[np.resize(src, len(dst))]
    return out


def metrics(pred, y):
    e = pred - y
    return {"along_rms_m": r4(float(np.sqrt(np.mean(e ** 2)))),
            "along_mae_m": r4(float(np.mean(np.abs(e)))),
            "speed_err_ms_rms": r4(float(np.sqrt(np.mean(e ** 2))) / HORIZON_S),
            "speed_err_ms_mae": r4(float(np.mean(np.abs(e))) / HORIZON_S),
            "bias_m": r4(float(e.mean())),
            "rms_over_mae": r4(float(np.sqrt(np.mean(e ** 2))
                                     / max(np.mean(np.abs(e)), 1e-12))),
            "gaussian_ref_rms_over_mae": 1.2533,
            "shrinkage_alpha": r4(float(np.polyfit(y, pred, 1)[0])),
            "p95_abs_m": r4(float(np.percentile(np.abs(e), 95))),
            "median_abs_m": r4(float(np.median(np.abs(e)))),
            "frac_under_0p5m": r4(float((np.abs(e) < 0.5).mean()))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--train", default="")
    ap.add_argument("--train-stride", type=int, default=1,
                    help="sub-sample the dense train grid (1 = keep all)")
    ap.add_argument("--out", default=str(STREAM / "raw" / "e3_head.json"))
    ap.add_argument("--preds", default=str(STREAM / "raw" / "e3_head_preds.npz"))
    a = ap.parse_args(argv)
    t0 = time.time()

    zv = np.load(a.val, allow_pickle=True)
    Xv, Yv, epi = zv["X"], zv["Y"], zv["epi"]
    assert list(map(str, zv["cols"])) == COLS, "column order changed"
    assert_no_oracle(COLS)
    y = Yv[:, 0]                                   # ⭐ the 2 s ALONG-track goal
    W = len(y)
    print(f"[fit] val: {W} windows / {len(np.unique(epi))} episodes", flush=True)

    rng = np.random.default_rng(20260728)
    Xv_noise, noise_stats_val = noisy_copy(Xv, rng)
    Xv_shuf = shuffled_across_episodes(Xv, epi, np.random.default_rng(777))

    preds: dict[str, np.ndarray] = {}
    res = {"_stream": "2026-07-28-egoal-3-trained-head", "_stage": "S2 head fit",
           "_cols": COLS, "_hist_cols": HIST,
           "_target": "y_long = gt[:, -1, 0], the 2 s ALONG-track displacement",
           "_regressor": ("sklearn HistGradientBoostingRegressor, "
                          "hyper-parameters IMPORTED from eg_fit.fit_predict "
                          "(max_iter=400, lr=0.06, leaves=31, min_leaf=40, "
                          "l2=1.0, no early stopping, seed 0) -- IDENTICAL for "
                          "every arm"),
           "_n_val_windows": int(W),
           "_n_val_episodes": int(len(np.unique(epi))),
           "_noise_stats_val": {k: [r4(v[0]), r4(v[1])]
                                for k, v in noise_stats_val.items()},
           "T_OOF": {}, "T_TRAIN": {}}

    # ------------------------------------------------------------------ T-OOF
    arms_oof = {"H_ego": Xv, "H_nohist": Xv[:, NOHIST_IDX],
                "H_noise_hist": Xv_noise, "H_v0": Xv[:, [0]],
                "H_inst": Xv[:, INST_IDX], "H_v0_ax": Xv[:, V_AX_IDX],
                "N_SHUF": Xv_shuf}
    oof = {k: np.full(W, np.nan) for k in arms_oof}
    epis = epi.astype(str)
    for f, (tr, te) in enumerate(clip_folds(epis, k=5, seed=SEED)):
        for k, X in arms_oof.items():
            oof[k][te] = fit_predict(X[tr], y[tr], X[te], "gbm")
        print(f"  T-OOF fold {f+1}/5 done ({time.time()-t0:.0f}s)", flush=True)
    for k, p in oof.items():
        assert np.isfinite(p).all(), f"{k}: OOF prediction missing"
        preds[f"T_OOF|{k}"] = p
        res["T_OOF"][k] = metrics(p, y)
    # the two closed-form arms are identical in both configurations
    cv_pred = Xv[:, 0] * HORIZON_S
    preds["T_OOF|CV_head"] = cv_pred
    res["T_OOF"]["CV_head"] = metrics(cv_pred, y)
    preds["T_OOF|P_ORACLE_TRUE"] = y.copy()
    res["T_OOF"]["P_ORACLE_TRUE"] = metrics(y, y)

    # ---------------------------------------------------------------- T-TRAIN
    if a.train:
        zt = np.load(a.train, allow_pickle=True)
        Xt, Yt, epit = zt["X"], zt["Y"], zt["epi"]
        assert list(map(str, zt["cols"])) == COLS, "train column order changed"
        if a.train_stride > 1:
            m = np.arange(len(Xt)) % a.train_stride == 0
            Xt, Yt, epit = Xt[m], Yt[m], epit[m]
        yt = Yt[:, 0]
        good = np.isfinite(yt) & np.isfinite(Xt).all(1)
        Xt, yt, epit = Xt[good], yt[good], epit[good]
        print(f"[fit] train: {len(yt)} windows / {len(np.unique(epit))} "
              f"episodes (stride {a.train_stride})", flush=True)
        res["_n_train_windows"] = int(len(yt))
        res["_n_train_episodes"] = int(len(np.unique(epit)))

        rng2 = np.random.default_rng(20260728)
        Xt_noise, noise_stats_tr = noisy_copy(Xt, rng2)
        #: the val matrix gets noise drawn from the TRAIN column statistics, so
        #: train and test see the same (information-free) distribution
        Xv_noise_tr, _ = noisy_copy(Xv, np.random.default_rng(20260729),
                                    stats=noise_stats_tr)
        Xt_shuf = shuffled_across_episodes(Xt, epit, np.random.default_rng(778))
        res["_noise_stats_train"] = {k: [r4(v[0]), r4(v[1])]
                                     for k, v in noise_stats_tr.items()}

        pairs = {"H_ego": (Xt, Xv),
                 "H_nohist": (Xt[:, NOHIST_IDX], Xv[:, NOHIST_IDX]),
                 "H_noise_hist": (Xt_noise, Xv_noise_tr),
                 "H_v0": (Xt[:, [0]], Xv[:, [0]]),
                 "H_inst": (Xt[:, INST_IDX], Xv[:, INST_IDX]),
                 "H_v0_ax": (Xt[:, V_AX_IDX], Xv[:, V_AX_IDX]),
                 "N_SHUF": (Xt_shuf, Xv_shuf)}
        for k, (A, B) in pairs.items():
            p = fit_predict(A, yt, B, "gbm")
            preds[f"T_TRAIN|{k}"] = p
            res["T_TRAIN"][k] = metrics(p, y)
            print(f"  T-TRAIN {k:14s} rms={res['T_TRAIN'][k]['along_rms_m']:.4f} "
                  f"mae={res['T_TRAIN'][k]['along_mae_m']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        preds["T_TRAIN|CV_head"] = cv_pred
        res["T_TRAIN"]["CV_head"] = res["T_OOF"]["CV_head"]
        preds["T_TRAIN|P_ORACLE_TRUE"] = y.copy()
        res["T_TRAIN"]["P_ORACLE_TRUE"] = res["T_OOF"]["P_ORACLE_TRUE"]

    res["_reference_EGOAL_1_dev_corpus"] = {
        "E1_ego_along_rms_m": 0.9305, "E1_ego_along_mae_m": 0.4925,
        "E1_nohist_along_rms_m": 1.0733, "E1_noise_hist_along_rms_m": 1.0737,
        "E0_v0_along_rms_m": 1.5246, "CV_along_rms_m": 1.3827,
        "_class": "INHERITED from …/2026-07-28-egoal-2-power/raw/e2_arms.json "
                  "and eg_fit_gbm.json; a DIFFERENT window set (99,935 dev-corpus "
                  "windows on a 10 Hz grid), quoted for orientation only"}
    res["_wall_s"] = round(time.time() - t0, 1)

    Path(a.preds).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.preds, y=y, epi=epi, **preds)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"T_OOF": {k: {"rms": v["along_rms_m"],
                                    "mae": v["along_mae_m"]}
                                for k, v in res["T_OOF"].items()},
                      "T_TRAIN": {k: {"rms": v["along_rms_m"],
                                      "mae": v["along_mae_m"]}
                                  for k, v in res["T_TRAIN"].items()}}, indent=1),
          flush=True)
    print(f"[fit] -> {a.out} + {a.preds} ({res['_wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
