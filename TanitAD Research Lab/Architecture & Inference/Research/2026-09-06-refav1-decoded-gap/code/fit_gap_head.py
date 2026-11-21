#!/usr/bin/env python3
"""D-REFAV1-DK-DECODED P2 -- fit the VISION-ONLY lead-gap head, cross-fitted over
the 141-clip v7.2 EVAL corpus, and collapse it to a single linear functional the
planner can evaluate in-loop.

⭐ WHAT COMES OUT. For each of the 5 clip-disjoint folds, one weight tensor
``V [160, 1024]`` and one bias ``b`` such that

    gap_hat = (pool(_last_state) * V).sum() + b

is EXACTLY the two-stage-PCA + ridge pipeline's prediction. The equality is
CHECKED NUMERICALLY on every fold before the bundle is written -- always, with no
flag to switch it off -- and a relative error above 1e-4 is a REFUSAL: a head the
planner evaluates differently from the head that was scored is TWO heads, and the
difference would be invisible in every downstream number.

⛔ EVERY HYPER-PARAMETER IS FIT INSIDE THE TRAINING FOLDS. Stage-A channel PCA,
stage-B row PCA, the ridge lambda (clip-grouped inner CV) and the fit-mean all
see the training folds only. The scored fold is scored, never tuned on.
*(Selecting lambda on the scored split once produced +0.0000 with a zero-width CI
and hid a real +0.020 -- the 2026-08-22 failure this discipline exists for.)*

⛔ CROSS-FITTING IS A MEASUREMENT DEVICE, NOT THE DEPLOYMENT STORY. A deployed
head would be fit on the TRAIN corpus and applied to EVAL. That corpus's fp8
cache is not on this box, so the head is fit inside the eval corpus and
cross-fitting is what makes it admissible: no clip's own labels ever enter its
own prediction, and every one of the 141 clips still receives one.

⛔ CONTROLS, and they are not decoration:
  * ``constant``  -- must read the no-information value EXACTLY +0.000000;
  * ``pix``       -- the RAW-PIXEL FLOOR, an IDENTICAL pipeline with only the
                     source swapped. A learned representation that does not beat
                     raw input added NOTHING;
  * ``shuffle_within_clip`` -- structure surviving a within-clip shuffle is clip
                     identity, not perception (it demolished M84's agent-count cell);
  * ``d`` and ``n`` printed for every arm -- ``n << d`` is underpowered BY
    CONSTRUCTION, not a negative. Both arms are reduced to the SAME ``d`` so the
    floor comparison is not confounded by dimensionality.

⚠️ MEMORY. Everything streams in row CHUNKS: a whole-array float32 cast would be
14237 x 160 x 1024 x 4 B = 9.3 GB on a box with ~14 GB free -- the dense-windowed
-tensor trap that once paged a job to disk for 2.5 h while `nvidia-smi` showed an
idle GPU.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

CHAN_K = 32                     # stage-A channel PCA
FINAL_K = 128                   # stage-B final PCA = d of EVERY arm (matched)
LAMBDAS = np.logspace(-3, 6, 19)
N_FOLDS = 5
HEAD_VERSION = "dkgap-linear-v1"
CHUNK = 512


# --------------------------------------------------------------------------- #
# estimator core -- the same statistics M84 used                                #
# --------------------------------------------------------------------------- #
def r2_skill(y, pred, fit_mean):
    """1 - SSE_model / SSE_fitmean. The constant arm reads EXACTLY 0."""
    sse_c = float(np.sum((y - fit_mean) ** 2))
    if sse_c == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y - pred) ** 2)) / sse_c


def ridge_fit(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d, dtype=np.float64), X.T @ y)


def pick_lambda(X, y, groups, n_folds=5, seed=0):
    """Clip-grouped inner CV INSIDE the training folds. Never sees the scored fold."""
    uq = np.unique(groups)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uq))
    fo = {g: (perm[i] % n_folds) for i, g in enumerate(uq)}
    fid = np.array([fo[g] for g in groups])
    best, best_sse = None, np.inf
    for lam in LAMBDAS:
        sse = 0.0
        for f in range(n_folds):
            tr, va = fid != f, fid == f
            if va.sum() == 0 or tr.sum() <= X.shape[1]:
                continue
            m = y[tr].mean()
            w = ridge_fit(X[tr], y[tr] - m, lam)
            sse += float(np.sum((y[va] - (X[va] @ w + m)) ** 2))
        if sse < best_sse:
            best_sse, best = sse, lam
    return best if best is not None else LAMBDAS[len(LAMBDAS) // 2]


def boot_ci(y, pred, fm, cid, n_boot=2000, seed=0):
    """Episode-CLUSTER bootstrap: resample CLIPS with replacement.
    ⚠️ It answers 'would another draw of EPISODES say this?' and nothing else --
    not 'would another training run', and not 'would another inference run'."""
    uq = np.unique(cid)
    idx = {c: np.where(cid == c)[0] for c in uq}
    rng = np.random.RandomState(seed)
    v = np.empty(n_boot)
    for b in range(n_boot):
        ii = np.concatenate([idx[uq[p]] for p in rng.randint(0, len(uq), len(uq))])
        v[b] = r2_skill(y[ii], pred[ii], fm)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_boot(y, pa, pb, fm, cid, n_boot=2000, seed=0):
    """PAIRED cluster bootstrap on R2(a) - R2(b): the SAME resampled clips score
    BOTH arms in every draw. Two overlapping marginal CIs are NOT a null."""
    uq = np.unique(cid)
    idx = {c: np.where(cid == c)[0] for c in uq}
    rng = np.random.RandomState(seed)
    v = np.empty(n_boot)
    for b in range(n_boot):
        ii = np.concatenate([idx[uq[p]] for p in rng.randint(0, len(uq), len(uq))])
        v[b] = r2_skill(y[ii], pa[ii], fm) - r2_skill(y[ii], pb[ii], fm)
    d = r2_skill(y, pa, fm) - r2_skill(y, pb, fm)
    return d, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def rand_pca(F, k, seed, oversample=32, n_iter=2):
    """Randomized PCA basis of the row space, fit on the rows handed in."""
    rng = np.random.RandomState(seed)
    d = F.shape[1]
    k = int(min(k, F.shape[0] - 1, d))
    Om = rng.normal(size=(d, k + oversample)).astype(np.float32)
    Y = F @ Om
    for _ in range(n_iter):
        Y, _ = np.linalg.qr(F @ (F.T @ Y))
    Q, _ = np.linalg.qr(Y)
    _, _, Vt = np.linalg.svd(Q.T @ F, full_matrices=False)
    return Vt[:k].T.astype(np.float32), k


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm", required=True, help="consolidate_bank.py output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--gap-cap", type=float, default=30.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    bmeta = json.load(open(os.path.join(a.mm, "_bank_meta.json")))
    if not bmeta["freeze"]["held"]:
        raise SystemExit("the bank's freeze proof did not hold -- refusing to fit")
    if not bmeta.get("alignment_proven"):
        raise SystemExit("the bank's label-free alignment proof did not hold")
    FIELD = np.load(os.path.join(a.mm, "field.npy"), mmap_mode="r")
    PIX = np.load(os.path.join(a.mm, "pix.npy"), mmap_mode="r")
    M = np.load(os.path.join(a.mm, "meta.npz"), allow_pickle=True)
    cid = M["clip_id"].astype(str)
    frame = M["frame"].astype(np.int64)
    gap = M["gap"].astype(np.float64)
    state = M["state"].astype(str)
    speed = M["speed"].astype(np.float64)
    clips = sorted(set(cid.tolist()))
    N, POOL, C = FIELD.shape
    is_lead = state == "LEAD"
    is_nolead = state == "NO_LEAD"
    print("rows %d | LEAD %d | NO_LEAD %d | clips %d (with LEAD %d) | pool %d d %d"
          % (N, int(is_lead.sum()), int(is_nolead.sum()), len(clips),
             len(set(cid[is_lead])), POOL, C), flush=True)

    # ---- fold assignment: LABEL-FREE (sha256 of clip_id), seeded ------------
    fold_of = {c: int(hashlib.sha256(("%s|%d" % (c, a.seed)).encode())
                      .hexdigest()[:8], 16) % N_FOLDS for c in clips}
    row_fold = np.array([fold_of[c] for c in cid])
    print("fold sizes (clips):",
          [sum(1 for c in clips if fold_of[c] == f) for f in range(N_FOLDS)],
          flush=True)

    # ---- the PIXEL FLOOR's raw design matrix: fold-independent, built once --
    XPIX = np.empty((N, int(np.prod(PIX.shape[1:]))), dtype=np.float32)
    for s in range(0, N, CHUNK):
        XPIX[s:s + CHUNK] = (np.asarray(PIX[s:s + CHUNK], dtype=np.float32)
                             .reshape(min(CHUNK, N - s), -1) / 255.0)
    print("pixel design matrix %s (%.2f GB)"
          % (XPIX.shape, XPIX.nbytes / 1e9), flush=True)

    stage_a_cache = {}

    def field_design(f):
        """Stage-A channel PCA fit on fold-f TRAINING tokens, then the projected
        design matrix. Cached per fold, since three targets share it."""
        if f in stage_a_cache:
            return stage_a_cache[f]
        rng = np.random.RandomState(a.seed * 100 + f)
        tr_rows = np.flatnonzero(row_fold != f)
        take = rng.choice(tr_rows, min(1200, len(tr_rows)), replace=False)
        toks = []
        for r in np.array_split(np.sort(take), max(1, len(take) // 64)):
            toks.append(np.asarray(FIELD[r], dtype=np.float32).reshape(-1, C))
        T = np.concatenate(toks, 0)                 # ~192k tokens x 1024
        mu_a = T.mean(0)
        T -= mu_a                                   # ⚠️ in place: `T - mu_a`
        Cv = (T.T @ T) / max(len(T) - 1, 1)         # twice would double the peak
        w_, V_ = np.linalg.eigh(Cv.astype(np.float64))
        P_a = V_[:, np.argsort(w_)[::-1][:CHAN_K]].astype(np.float32)
        del T, Cv, toks
        X = np.empty((N, POOL * CHAN_K), dtype=np.float32)
        for s in range(0, N, CHUNK):
            blk = np.asarray(FIELD[s:s + CHUNK], dtype=np.float32) - mu_a
            X[s:s + CHUNK] = (blk @ P_a).reshape(blk.shape[0], -1)
        stage_a_cache.clear()          # one fold in memory at a time
        stage_a_cache[f] = (mu_a, P_a, X)
        return stage_a_cache[f]

    collapse_err = []

    def cross_fit(arm, mask, y, want_head):
        oof = np.full(N, np.nan)
        heads, lams = [], []
        for f in range(N_FOLDS):
            tr = mask & (row_fold != f)
            te = mask & (row_fold == f)
            if tr.sum() < 50 or te.sum() < 1:
                continue
            if arm == "pix":
                mu_a, P_a, X = None, None, XPIX
            else:
                mu_a, P_a, X = field_design(f)
            mu_b = X[tr].mean(0)
            P_b, _ = rand_pca(X[tr] - mu_b, FINAL_K, a.seed * 100 + f)
            Z = np.empty((N, P_b.shape[1]), dtype=np.float32)
            for s in range(0, N, CHUNK):
                Z[s:s + CHUNK] = (X[s:s + CHUNK] - mu_b) @ P_b
            sd = Z[tr].std(0) + 1e-6
            Z /= sd
            lam = pick_lambda(Z[tr], y[tr], cid[tr], seed=a.seed)
            fm = float(y[tr].mean())
            w = ridge_fit(Z[tr].astype(np.float64), y[tr] - fm, lam)
            oof[te] = Z[te].astype(np.float64) @ w + fm
            lams.append(float(lam))
            if want_head and arm == "field":
                # pred = ((x - mu_a) @ P_a).flat @ u - mu_b@u + fm, with
                # u = P_b @ (w/sd) and V[c] = P_a @ u_c   =>   one dot product.
                u = (P_b.astype(np.float64) @ (w / sd))
                Vv = u.reshape(POOL, CHAN_K) @ P_a.T.astype(np.float64)
                bias = fm - float(mu_b @ u) - float(Vv.sum(0) @ mu_a)
                te_i = np.flatnonzero(te)
                chk = te_i[:: max(1, len(te_i) // 200)][:200]
                got = np.array([float((np.asarray(FIELD[g], dtype=np.float64)
                                       * Vv).sum() + bias) for g in chk])
                dmax = float(np.abs(got - oof[chk]).max())
                rel = dmax / max(1e-9, float(np.abs(oof[chk]).max()))
                collapse_err.append({"fold": int(f), "max_abs": dmax, "rel": rel,
                                     "n_checked": int(len(chk))})
                if rel > 1e-4:
                    raise SystemExit(
                        "[fit_gap_head] the collapsed linear head does NOT "
                        "reproduce the pipeline (fold %d: max|delta| %.3e rel "
                        "%.3e over %d rows). Refusing to ship two heads under "
                        "one name." % (f, dmax, rel, len(chk)))
                heads.append({"fold": f, "V": Vv.astype(np.float32),
                              "bias": float(bias), "lam": float(lam),
                              "fit_mean": fm,
                              "collapse_max_abs_err": dmax,
                              "collapse_rel_err": rel,
                              "n_fit_rows": int(tr.sum()),
                              "n_fit_clips": int(len(set(cid[tr])))})
            del Z
        return oof, heads, lams

    out = {"bank_meta": {k: bmeta[k] for k in
                         ("n_clips", "n_rows", "n_lead_rows", "pool", "grid",
                          "win", "ckpt_sha256", "ckpt_step", "block",
                          "speed_check_max_mps", "speed_misjoin_ctrl_mean_mps",
                          "alignment_proven", "freeze")},
           "n_folds": N_FOLDS, "seed": a.seed, "chan_k": CHAN_K,
           "final_k": FINAL_K, "head_version": HEAD_VERSION,
           "arms_note": ("the DINOv3 external reference is deliberately NOT re-run "
                         "here: it is in none of this spec's bars, M84 already "
                         "published it on the TRAIN corpus, and its 4.7 GB beside "
                         "field's would exceed free RAM. The required control is "
                         "the RAW-PIXEL FLOOR, which is present."),
           "cells": {}}

    targets = [("gap_all_lead", is_lead, gap),
               ("gap_le%dm" % int(a.gap_cap),
                is_lead & (gap <= a.gap_cap), gap),
               ("present", is_lead | is_nolead, is_lead.astype(np.float64))]

    bundle_heads = {}
    for tag, mask, y in targets:
        print("\n=== %s : n=%d rows / %d clips ==="
              % (tag, int(mask.sum()), len(set(cid[mask]))), flush=True)
        preds = {}
        for arm in ("field", "pix"):
            oof, heads, lams = cross_fit(arm, mask, y,
                                         want_head=(arm == "field"))
            preds[arm] = oof
            if arm == "field" and heads:
                bundle_heads[tag] = heads
            print("  %-5s done  lambdas %s  [%.0f s]"
                  % (arm, lams, time.time() - t0), flush=True)
        rows = mask & np.isfinite(preds["field"]) & np.isfinite(preds["pix"])
        ys, cs = y[rows], cid[rows]
        fm = float(np.mean([np.mean(y[mask & (row_fold != f)])
                            for f in range(N_FOLDS)]))
        co = {"n_score": int(rows.sum()),
              "n_clusters": int(len(np.unique(cs))), "d": FINAL_K,
              "fit_mean": fm, "arms": {}}
        co["arms"]["constant"] = {"r2": r2_skill(ys, np.full(rows.sum(), fm), fm),
                                  "d": 0, "note": "must read EXACTLY +0.000000"}
        rng = np.random.RandomState(a.seed + 7)
        shuf = preds["field"][rows].copy()
        for c in np.unique(cs):
            m = cs == c
            shuf[m] = shuf[m][rng.permutation(int(m.sum()))]
        co["arms"]["shuffle_within_clip"] = {"r2": r2_skill(ys, shuf, fm),
                                             "d": FINAL_K}
        for arm in ("field", "pix"):
            p = preds[arm][rows]
            lo, hi = boot_ci(ys, p, fm, cs, n_boot=a.n_boot, seed=a.seed)
            yc, pc = ys.astype(np.float64).copy(), p.astype(np.float64).copy()
            for c in np.unique(cs):
                m = cs == c
                yc[m] -= ys[m].mean()
                pc[m] -= p[m].mean()
            co["arms"][arm] = {"r2": r2_skill(ys, p, fm), "ci": [lo, hi],
                               "d": FINAL_K,
                               "r2_within_clip": r2_skill(yc, pc, 0.0),
                               "mae": float(np.mean(np.abs(ys - p))),
                               "bias": float(np.mean(p - ys))}
            print("  %-5s R2=%+.4f CI[%+.4f,%+.4f] wc=%+.4f MAE=%.3f bias=%+.3f"
                  % (arm, co["arms"][arm]["r2"], lo, hi,
                     co["arms"][arm]["r2_within_clip"], co["arms"][arm]["mae"],
                     co["arms"][arm]["bias"]), flush=True)
        d, lo, hi = paired_boot(ys, preds["field"][rows], preds["pix"][rows],
                                fm, cs, n_boot=a.n_boot, seed=a.seed)
        co["paired"] = {"field-pix": {"delta": d, "ci": [lo, hi],
                                      "excludes_zero": bool(lo > 0 or hi < 0)}}
        print("  PAIRED field-pix %+0.4f [%+0.4f,%+0.4f] %s"
              % (d, lo, hi, "EXCLUDES 0" if (lo > 0 or hi < 0) else "spans 0"),
              flush=True)
        print("  constant %+0.6f (EXACT 0 required) | shuffle_within_clip %+0.4f"
              % (co["arms"]["constant"]["r2"],
                 co["arms"]["shuffle_within_clip"]["r2"]), flush=True)
        out["cells"][tag] = co
        np.savez_compressed(
            os.path.join(a.out, "oof_%s.npz" % tag),
            clip_id=cid[rows], frame=frame[rows], y=ys, speed=speed[rows],
            pred_field=preds["field"][rows], pred_pix=preds["pix"][rows],
            fold=row_fold[rows], fit_mean=fm)

    # ---- the deployable bundle ---------------------------------------------
    import torch
    bundle = {"version": HEAD_VERSION, "kind": "linear_pooled_laststate",
              "pool": list(bmeta["pool"]), "grid": list(bmeta["grid"]),
              "win": int(bmeta["win"]), "d_state": int(bmeta["d_state"]),
              "ckpt_sha256": bmeta["ckpt_sha256"],
              "ckpt_step": int(bmeta["ckpt_step"]),
              "fit_corpus": "v7.2-eval-141",
              "fit_scheme": "%dfold-crossfit-clip-disjoint-seed%d" % (N_FOLDS, a.seed),
              "fold_of_clip": fold_of, "targets": {}}
    for tag, heads in bundle_heads.items():
        bundle["targets"][tag] = {"by_fold": {
            int(h["fold"]): {"V": torch.tensor(h["V"]), "bias": float(h["bias"]),
                             "lam": h["lam"], "fit_mean": h["fit_mean"],
                             "n_fit_rows": h["n_fit_rows"],
                             "n_fit_clips": h["n_fit_clips"]} for h in heads}}
    bp = os.path.join(a.out, "gap_head_bundle.pt")
    torch.save(bundle, bp)
    sha = hashlib.sha256(open(bp, "rb").read()).hexdigest()
    out["bundle"] = {"path": bp, "sha256": sha, "n_bytes": os.path.getsize(bp),
                     "targets": sorted(bundle["targets"])}
    out["collapse_check"] = collapse_err
    json.dump(out, open(os.path.join(a.out, "decode_panel.json"), "w"), indent=1)
    print("\ncollapse check: max rel err over %d folds = %.3e"
          % (len(collapse_err),
             max([c["rel"] for c in collapse_err], default=float("nan"))))
    print("bundle -> %s  sha256 %s  (%.1f MB)  [%.0f s]"
          % (bp, sha[:16], os.path.getsize(bp) / 1e6, time.time() - t0), flush=True)


if __name__ == "__main__":
    sys.exit(main())
