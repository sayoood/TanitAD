#!/usr/bin/env python3
"""S2 (CEILING) + S3 (REALISED PICK) -- state-conditioned anchor sets.

Builds CoverNet-style anchor sets on the canonical 881-window / 40-episode val
deployment and measures, at MATCHED per-window proposal count:

  CEILING   oracle_in_fan  = mean_w min_a ADE(anchor_a, GT_w)
  REALISED  nearest-anchor-to-a-reference, over a family of references spanning
            the accuracy range, so the quantisation tax can be read off AT v1's
            operating point (0.4271) -- the number the CONFIRM bar needs.

Everything is fitted OUT-OF-FOLD on 5 episode-disjoint folds. In-sample is also
reported: it is the MAXIMUM-POSSIBLE-OVERFIT upper bound, so if the in-sample
conditioned arm fails a bar, thin fitting data cannot be the excuse.

Negative controls NC1 (shuffled v0) and NC2 (Gaussian noise) prove the ceiling
rule CAN return a failing value.

Usage:  python fanc_anchors.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from fanc_common import (OUT, SEED, ade, ade_set, assign_bucket, bucket_edges,
                         ci_paired, ci_single, eid_str, episode_folds,
                         fit_anchors, load_refc_fan, pick_nearest_to, r4)

N_GRID = (16, 32, 64, 128, 256)
B_GRID = (1, 2, 4, 8, 16)
K_FOLDS = 5
ALPHAS = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0, 1.3)


# --------------------------------------------------------------------------- #
def eval_anchor_sets(gt, v0, refs, N, B, tr, te, cond_var=None, seed=SEED):
    """Fit on `tr`, evaluate on `te`. Returns per-test-window dict of vectors.

    `cond_var` is the variable the buckets are built on (defaults to v0; the
    negative controls pass a shuffled or random vector instead). `B == 1` is the
    UNCONDITIONED fixed anchor set.
    """
    cv_ = v0 if cond_var is None else cond_var
    n_te = int(te.sum())
    ceil = np.empty(n_te)
    realised = {k: np.empty(n_te) for k in refs}
    prior = np.empty(n_te)
    n_per_bucket, n_anch_actual = [], []

    if B == 1:
        anchors, wts = fit_anchors(gt[tr], N, seed)
        groups = [(np.ones(n_te, bool), anchors, wts)]
        n_per_bucket.append(int(tr.sum()))
        n_anch_actual.append(len(anchors))
    else:
        edges = bucket_edges(cv_[tr], B)
        b_tr = assign_bucket(cv_[tr], edges)
        b_te = assign_bucket(cv_[te], edges)
        groups = []
        for b in range(B):
            m_tr = b_tr == b
            m_te = b_te == b
            n_per_bucket.append(int(m_tr.sum()))
            if m_tr.sum() == 0:                     # empty bucket -> global set
                anchors, wts = fit_anchors(gt[tr], N, seed)
            else:
                anchors, wts = fit_anchors(gt[tr][m_tr], N, seed)
            n_anch_actual.append(len(anchors))
            if m_te.any():
                groups.append((m_te, anchors, wts))

    gt_te = gt[te]
    for m_te, anchors, wts in groups:
        e = ade_set(anchors, gt_te[m_te])           # [w, n]
        ceil[m_te] = e.min(1)
        prior[m_te] = e[:, int(np.argmax(wts))]
        for k, R in refs.items():
            idx = pick_nearest_to(R[te][m_te], anchors)
            realised[k][m_te] = e[np.arange(len(idx)), idx]

    return {"ceiling": ceil, "prior": prior, "realised": realised,
            "n_per_bucket": n_per_bucket, "n_anchors_actual": n_anch_actual}


def run_grid(gt, v0, refs, eid, folds, cond_var=None, tag="v0"):
    """Full N x B grid, out-of-fold and in-sample, as per-window OOF vectors."""
    W = len(gt)
    out = {}
    for N in N_GRID:
        for B in B_GRID:
            key = f"N{N}_B{B}"
            acc = {"ceiling": np.empty(W), "prior": np.empty(W),
                   "realised": {k: np.empty(W) for k in refs}}
            acc_is = {"ceiling": np.empty(W), "prior": np.empty(W),
                      "realised": {k: np.empty(W) for k in refs}}
            npb, naa = [], []
            for tr, te in folds:
                r = eval_anchor_sets(gt, v0, refs, N, B, tr, te, cond_var)
                acc["ceiling"][te] = r["ceiling"]
                acc["prior"][te] = r["prior"]
                for k in refs:
                    acc["realised"][k][te] = r["realised"][k]
                npb += r["n_per_bucket"]
                naa += r["n_anchors_actual"]
                # IN-SAMPLE: fit on the test fold itself (max-overfit bound)
                ri = eval_anchor_sets(gt, v0, refs, N, B, te, te, cond_var)
                acc_is["ceiling"][te] = ri["ceiling"]
                acc_is["prior"][te] = ri["prior"]
                for k in refs:
                    acc_is["realised"][k][te] = ri["realised"][k]
            out[key] = {"oof": acc, "insample": acc_is,
                        "n_train_per_bucket_min": int(np.min(npb)),
                        "n_train_per_bucket_median": float(np.median(npb)),
                        "n_anchors_actual_min": int(np.min(naa)),
                        "n_anchors_requested": N,
                        "anchors_starved": bool(np.min(naa) < N)}
    return out


def main() -> None:
    t0 = time.time()
    d = load_refc_fan("xl")
    gt = d["gt"].numpy().astype(np.float64)
    v0 = d["v0"].numpy().astype(np.float64)
    cv = d["cv"].numpy().astype(np.float64)
    sel = d["sel"].numpy()
    fan = d["fan"].numpy().astype(np.float64)
    refc_pick = fan[np.arange(len(sel)), sel]
    eid = eid_str(d)
    W = len(gt)

    # ---- reference family: real error STRUCTURE, swept in magnitude --------- #
    # R(alpha) = GT + alpha * (CV - GT). alpha=0 -> GT (0 m), alpha=1 -> CV.
    # This keeps the longitudinal-dominated error structure of a real planner
    # rather than isotropic Gaussian noise, and it is validated against TWO real
    # references (CV, and REF-C-XL's own selected trajectory).
    refs = {f"alpha{a:g}": gt + a * (cv - gt) for a in ALPHAS}
    refs["REAL_cv"] = cv
    refs["REAL_refcxl_pick"] = refc_pick
    ref_own_ade = {k: float(ade(R, gt).mean()) for k, R in refs.items()}

    folds = list(episode_folds(eid, K_FOLDS))
    print(f"folds: {[int(te.sum()) for _, te in folds]} windows, "
          f"{[len(set(eid[te])) for _, te in folds]} episodes")

    res = {"_stream": "2026-07-27-fan-conditioning",
           "_estimator": "paired_episode_cluster_bootstrap B=2000, unit=episode",
           "_n_windows": W, "_n_episodes": int(len(set(eid))),
           "_k_folds": K_FOLDS,
           "_reference_own_ade": {k: r4(v) for k, v in ref_own_ade.items()},
           "_bars": {"CONFIRM": 0.4907, "STRONG": 0.4271,
                     "v1_own_ade": 0.4271,
                     "refc_xl_as_trained": 0.4714,
                     "refc_xl_oracle_in_fan": 0.1640,
                     "v4_fan_C2": 0.5645, "v4_fan_oracle": 0.2505,
                     "v4_fan_measured_quantisation_tax": 0.1374}}

    # ------------------------------------------------------- main v0 grid ---- #
    grid = run_grid(gt, v0, refs, eid, folds, cond_var=None, tag="v0")
    print(f"main grid done {time.time()-t0:.1f}s")

    # ------------------------------------------------- negative controls ----- #
    rng = np.random.default_rng(SEED)
    v0_shuf = rng.permutation(v0)
    noise = rng.normal(size=W)
    grid_nc1 = run_grid(gt, v0_shuf, refs, eid, folds, cond_var=v0_shuf, tag="shuf")
    grid_nc2 = run_grid(gt, noise, refs, eid, folds, cond_var=noise, tag="noise")
    print(f"negative controls done {time.time()-t0:.1f}s")

    # ------------------------------------------------------- adjudication ---- #
    def summarise(g, name):
        o = {}
        for key, v in g.items():
            N = int(key.split("_")[0][1:])
            B = int(key.split("_")[1][1:])
            base_key = f"N{N}_B1"
            store_key = f"N{min(N*B, 4096)}_B1"
            e = {"N": N, "B": B,
                 "n_train_per_bucket_min": v["n_train_per_bucket_min"],
                 "anchors_starved": v["anchors_starved"],
                 "n_anchors_actual_min": v["n_anchors_actual_min"]}
            for split in ("oof", "insample"):
                a = v[split]
                e[f"{split}_ceiling"] = ci_single(a["ceiling"], eid)
                e[f"{split}_prior"] = ci_single(a["prior"], eid)
                e[f"{split}_realised"] = {
                    k: ci_single(a["realised"][k], eid) for k in a["realised"]}
                if B > 1:
                    b = g[base_key][split]
                    e[f"{split}_dCeiling_vs_fixedN"] = ci_paired(
                        a["ceiling"], b["ceiling"], eid)
                    e[f"{split}_dRealised_vs_fixedN"] = {
                        k: ci_paired(a["realised"][k], b["realised"][k], eid)
                        for k in a["realised"]}
                    if store_key in g:      # equal-TOTAL-STORAGE control
                        c = g[store_key][split]
                        e[f"{split}_dCeiling_vs_fixedNB"] = ci_paired(
                            a["ceiling"], c["ceiling"], eid)
            o[key] = e
        return o

    res["S2_S3_v0_conditioned"] = summarise(grid, "v0")
    res["NC1_shuffled_v0"] = summarise(grid_nc1, "shuf")
    res["NC2_gaussian_noise"] = summarise(grid_nc2, "noise")

    # ---- fidelity: nearest-to-GT must reproduce the ceiling exactly --------- #
    fid = {}
    for N in (64, 256):
        a = grid[f"N{N}_B1"]["oof"]
        fid[f"N{N}_B1_oracle_ref_equals_ceiling"] = {
            "max_abs_diff": r4(np.abs(a["realised"]["alpha0"] - a["ceiling"]).max()),
            "ok": bool(np.allclose(a["realised"]["alpha0"], a["ceiling"]))}
    res["_fidelity"] = fid
    print("fidelity (nearest-to-GT == ceiling):", fid)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "fanc_anchors_perwindow.npz",
                        **{f"{k}_{s}_ceiling": grid[k][s]["ceiling"]
                           for k in grid for s in ("oof", "insample")},
                        **{f"{k}_oof_real_{r}": grid[k]["oof"]["realised"][r]
                           for k in grid for r in refs},
                        eid=eid, v0=v0)
    (OUT / "fanc_anchors.json").write_text(json.dumps(res, indent=2))
    print("wrote", OUT / "fanc_anchors.json", f"  total {time.time()-t0:.1f}s")

    # ---- console headline -------------------------------------------------- #
    print("\nCEILING (oof) by N x B  [B=1 is the unconditioned fixed set]")
    print("   N |" + "".join(f"  B={b:<3d}" for b in B_GRID))
    for N in N_GRID:
        row = f"{N:4d} |"
        for B in B_GRID:
            row += f" {res['S2_S3_v0_conditioned'][f'N{N}_B{B}']['oof_ceiling']['mean']:6.4f}"
        print(row)
    print("\nREALISED, reference = REAL_refcxl_pick (own ADE "
          f"{ref_own_ade['REAL_refcxl_pick']:.4f}) (oof)")
    print("   N |" + "".join(f"  B={b:<3d}" for b in B_GRID))
    for N in N_GRID:
        row = f"{N:4d} |"
        for B in B_GRID:
            row += (f" {res['S2_S3_v0_conditioned'][f'N{N}_B{B}']['oof_realised']['REAL_refcxl_pick']['mean']:6.4f}")
        print(row)


if __name__ == "__main__":
    main()
