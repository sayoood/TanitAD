"""⚠️ POST-HOC DIAGNOSTIC, NOT PRE-REGISTERED — is `C-ORACLE-PS` a control that CANNOT FAIL?

WHY THIS EXISTS
---------------
`PRE_REGISTRATION.md` Sec 4 registers `C-ORACLE-PS` — the per-situation cell chosen on the TEST
fold — as the upper bound on any per-situation gain, and Sec 5 uses "the oracle did NOT separate"
to license the strongest null (`NO_PER_SITUATION_GAIN_EXISTS`).

That direction is sound. The OTHER direction is not, and I noticed it only after seeing the number:
**picking the maximum of 25 noisy estimates and then testing that maximum against the baseline ON
THE SAME ROWS is upward-biased by construction.** A firing oracle is therefore weak evidence that a
gain exists — it is partly guaranteed. That is precisely the failure the pre-registration warns
about in the E-SEL stream's `C-shuffled` leg: a control that cannot fail is not a control.

⛔ SO THIS CHANGES NO VERDICT. Every pre-registered predicate stands exactly as
`results_horizon_ps.json` recorded it. This measures how much of the oracle's margin is winner's
curse, so the report can say what the oracle's separation is worth. It can only make the reading
MORE conservative — it cannot manufacture a win.

THE DIAGNOSTIC
--------------
Run the IDENTICAL oracle procedure on the clip-permuted (NEG_FEAT) substrate, where there is no
clip-specific signal for a per-situation configuration to exploit. Whatever margin the oracle earns
THERE is pure selection bias. The comparison is exact because both use the same grid, the same
folds, the same draws and the same code — this module imports the study's own functions rather than
re-implementing them.

usage:
  python oracle_winners_curse.py --substrate <npz> --out oracle_winners_curse.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_per_situation_horizon as S                                  # noqa: E402
from tanitad.eval.sitclf import causal_window, clip_runs, cluster_folds  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate",
                    default=r"C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.npz")
    ap.add_argument("--results", default="results_horizon_ps.json")
    ap.add_argument("--out", default="oracle_winners_curse.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    z = np.load(a.substrate)
    F = z["F"]
    cc = z["clip_cluster"]
    st, en = clip_runs(cc)
    N = F.shape[0]
    folds = cluster_folds(cc, 2, seed=0)
    per_clip = S.events_per_clip(N)

    Y_at, V_at = {}, {}
    for L in S.LEADS:
        Y_at[L], V_at[L] = S.targets_at(per_clip, L)
    _, hist32 = causal_window(np.zeros((N, 1), np.float32), st, en, S.WIN_MAX)
    EVAL = V_at[S.H_MAX_S] & hist32[:, None]
    Y_EVAL = Y_at[S.DEPLOY_LEAD_S]

    splits = []
    for f in (0, 1):
        te = folds == f
        tr = ~te
        tr_cl = np.unique(cc[tr])
        perm = np.random.default_rng(0 + f).permutation(tr_cl)
        sel_cl = set(int(x) for x in perm[:max(1, int(round(S.SEL_FRAC * len(tr_cl))))])
        is_sel = np.array([int(x) in sel_cl for x in cc])
        splits.append((tr & ~is_sel, tr & is_sel, te))

    perm_clip = np.random.default_rng(0).permutation(len(st))
    shuf_rows = np.arange(N)
    for i, (s0, e0) in enumerate(zip(st, en)):
        j = perm_clip[i]
        n = min(e0 - s0, en[j] - st[j])
        shuf_rows[s0:s0 + n] = st[j] + np.arange(n)
        if e0 - s0 > n:
            shuf_rows[s0 + n:e0] = st[j] + n - 1

    # ---- the NULL grid, refit by the study's own code path ---------------------------
    src = F[shuf_rows]
    XW = {}
    for f in (0, 1):
        mu, Wp = S.fit_pca(src, np.flatnonzero(splits[f][0]), S.RANK, device=device)
        P = S.project(src, mu, Wp)
        for W in S.WINS:
            XW[(f, W)] = causal_window(P, st, en, W)[0]
    del src
    log(f"null PCA + windows built ({time.time()-t0:.0f}s)")

    CELL = {}
    for W in S.WINS:
        for L in S.LEADS:
            out = np.zeros((N, 3), np.float32)
            for f in (0, 1):
                X = XW[(f, W)]
                fit, sel, te = splits[f]
                fitr = fit & hist32
                vL = V_at[L] & hist32[:, None]
                Sc = S.ridge_multi(X[fitr], Y_at[L][fitr].astype(np.float32),
                                   vL[fitr].astype(np.float32), X, S.LAMBDAS)
                for i in range(3):
                    msel = sel & vL[:, i]
                    best = (-np.inf, S.LAMBDAS[0])
                    for lam in S.LAMBDAS:
                        if msel.sum() < 50 or Y_at[L][msel, i].sum() < 5:
                            continue
                        v = S.ap_lift_safe(Y_at[L][msel, i].astype(np.float64),
                                           Sc[lam][msel, i].astype(np.float64))
                        if np.isfinite(v) and v > best[0]:
                            best = (v, lam)
                    out[te, i] = Sc[best[1]][te, i]
            CELL[(W, L)] = out
        log(f"  null grid W={W} done ({time.time()-t0:.0f}s)")
    del XW

    # ---- the SAME oracle procedure, on the null substrate -----------------------------
    NULL_FROZEN = CELL[S.FROZEN_CELL]
    NULL_ORACLE = np.zeros((N, 3), np.float32)
    picked = {}
    for f in (0, 1):
        te = splits[f][2]
        for i in range(3):
            m = te & EVAL[:, i]
            best = (-np.inf, *S.FROZEN_CELL)
            if m.sum() >= 50 and Y_EVAL[m, i].sum() >= 5:
                yv = Y_EVAL[m, i].astype(np.float64)
                for W in S.WINS:
                    for L in S.LEADS:
                        v = S.ap_lift_safe(yv, CELL[(W, L)][m, i].astype(np.float64))
                        if np.isfinite(v) and v > best[0]:
                            best = (v, W, L)
            NULL_ORACLE[te, i] = CELL[(best[1], best[2])][te, i]
            picked[f"fold{f}.{S.SITS[i]}"] = {"win": best[1], "lead_s": best[2]}

    uniq_cl, picks = S.cluster_picks(cc)
    cl_index = {int(u): k for k, u in enumerate(uniq_cl.astype(np.int64))}
    R = json.loads(Path(a.results).read_text(encoding="utf-8"))
    out = {"_what": "POST-HOC: how much of C-ORACLE-PS's margin is winner's curse?",
           "_not_prereg": ("added after seeing the oracle fire. It changes NO verdict — every "
                           "pre-registered predicate stands as results_horizon_ps.json recorded "
                           "it. It can only make the reading more conservative."),
           "_method": ("the IDENTICAL test-fold argmax over the same 25 cells, run on the "
                       "clip-permuted (NEG_FEAT) substrate where no clip-specific per-situation "
                       "gain can exist. Whatever margin it earns there is pure selection bias."),
           "null_oracle_cells": picked, "per_situation": {}}

    for i, s in enumerate(S.SITS):
        m = EVAL[:, i]
        rows = np.flatnonzero(m)
        yv = Y_EVAL[rows, i].astype(np.float64)
        rcl = np.array([cl_index[int(x)] for x in cc[rows]])
        order = np.argsort(rcl, kind="mergesort")
        rows_by_cl = [order[np.searchsorted(rcl[order], k, "left"):
                            np.searchsorted(rcl[order], k, "right")]
                      for k in range(len(uniq_cl))]
        A = NULL_ORACLE[rows, i].astype(np.float64)
        B = NULL_FROZEN[rows, i].astype(np.float64)
        pa, pb = S.ap_lift_safe(yv, A), S.ap_lift_safe(yv, B)
        d = np.empty(len(picks))
        for k, pick in enumerate(picks):
            sel = np.concatenate([rows_by_cl[p] for p in pick])
            ys = yv[sel]
            b = ys.mean()
            if b <= 0:
                d[k] = np.nan
                continue
            from tanitad.eval.ap_ci import average_precision            # noqa: PLC0415
            d[k] = (average_precision(ys, A[sel]) - average_precision(ys, B[sel])) / b
        pk = S.pack_paired(pa - pb, d)
        real = R["per_situation"][s]["QA_TRAIN_HORIZON"]["arms"]["C_ORACLE_PS"][
            "paired_vs_frozen"]
        share = (abs(pk["delta"]) / abs(real["delta"])) if real["delta"] else float("nan")
        out["per_situation"][s] = {
            "null_oracle_lift": round(pa, 5), "null_frozen_lift": round(pb, 5),
            "NULL_ORACLE_vs_NULL_FROZEN": pk,
            "REAL_ORACLE_vs_FROZEN": {k: real[k] for k in
                                      ("delta", "lo", "hi", "separated")},
            "winners_curse_share_of_real_margin": (round(share, 4)
                                                   if np.isfinite(share) else None),
            "READING": ("the real oracle's margin is NOT distinguishable from selection bias"
                        if pk["separated"] and pk["delta"] > 0
                        and pk["delta"] >= 0.5 * real["delta"]
                        else "selection bias alone does NOT reproduce the real oracle's margin"
                        if not (pk["separated"] and pk["delta"] > 0)
                        else "selection bias reproduces PART of the real oracle's margin")}
        log(f"  {s:>13}: NULL oracle vs NULL frozen {pk['delta']:+.4f} "
            f"[{pk['lo']:+.4f},{pk['hi']:+.4f}]{' SEP' if pk['separated'] else ''}  |  "
            f"REAL oracle margin {real['delta']:+.4f}  |  curse share "
            f"{out['per_situation'][s]['winners_curse_share_of_real_margin']}")
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
