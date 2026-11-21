"""IS THE SHARED sitclf RE-TUNE A DEPLOYABLE CHANGE, AND WHAT DOES IT COST?

Pre-registration: ./PRE_REGISTRATION.md — hash-pinned and staged BEFORE any number here was read.

THE GAP THIS CLOSES
-------------------
The parent stream (../2026-08-03-sitclf-horizon/) banked `C-GLOBAL` at +0.3333 event recall on
`lane_change`. ⛔ `C-GLOBAL` IS NOT A SETTING — it is a per-fold SELECTION RULE, and its own
`arm_configurations` show it chose (W32, L1.0) in fold 0 and (W1, L1.0) in fold 1: the two OPPOSITE
extremes of the window grid. A deployed head has ONE (W, lead_s). This script asks whether a
SINGLE FIXED cell survives, with multiplicity control, fold-consistency, and a chance floor.

⛔ VISION-ONLY at inference (PI ruling 2026-08-03): every arm reads frozen v1 camera latents only.

usage:
  python run_event_retune.py --out results_event_retune.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "2026-08-03-sitclf-horizon"
REPO = HERE.parents[5]
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

import run_per_situation_horizon as P                                   # noqa: E402
from tanitad.eval.sitclf import (causal_window, clip_runs,              # noqa: E402
                                 cluster_folds)
from tanitad.eval.sitclf_deploy import (event_anticipation_report,      # noqa: E402
                                        precision_recall_at_budget)

SITS = P.SITS
WINS, LEADS = P.WINS, P.LEADS
FROZEN_CELL = P.FROZEN_CELL
POWERED = ("lane_change", "intersection")        # C_POW >= 40; roundabout has 37 -> no verdict
N_CHANCE = 200
ALPHA = 0.05
HB = int(round(P.H_MAX_S * P.HZ))


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, "replace").decode(enc, "replace"), flush=True)


def cell_name(W, L):
    return f"CELL_w{W}_L{L}"


def ridge_params(W, rank=P.RANK, n_out=3):
    """flat dim = rank*W, plus the unpenalised intercept, per situation output."""
    return (rank * W + 1) * n_out


# --------------------------------------------------------------------------- #
# the event yardstick, in the PARENT's exact bootstrap form                    #
# (fixed alarm set = the deployment operating point; clusters resampled)       #
# --------------------------------------------------------------------------- #
def qb_stat(col, rows, rcl, row_of, og, nxt):
    k = max(1, int(round(P.TOP_FRAC * len(rows))))
    al = np.zeros(len(rows), bool)
    al[np.argsort(-col, kind="mergesort")[:k]] = True
    warned = np.zeros(len(og), bool)
    lead = np.full(len(og), np.nan)
    for j, g in enumerate(og):
        idx = row_of[max(0, g - HB):g]
        idx = idx[idx >= 0]
        hit = idx[al[idx]]
        if hit.size:
            warned[j] = True
            lead[j] = (g - rows[hit[0]]) / P.HZ
    return {"alarm": al, "warned": warned, "lead": lead,
            "useful5": nxt[al] <= P.H_MAX_S * P.HZ,
            "useful3": nxt[al] <= P.DEPLOY_LEAD_S * P.HZ,
            "acl": rcl[al]}


def qb_boot(d, ocl, picks, n_cl):
    rec = np.bincount(ocl, weights=d["warned"].astype(float), minlength=n_cl)
    tot = np.bincount(ocl, minlength=n_cl).astype(float)
    p5 = np.bincount(d["acl"], weights=d["useful5"].astype(float), minlength=n_cl)
    p3 = np.bincount(d["acl"], weights=d["useful3"].astype(float), minlength=n_cl)
    na = np.bincount(d["acl"], minlength=n_cl).astype(float)
    B = len(picks)
    R, P5, P3 = np.empty(B), np.empty(B), np.empty(B)
    for k, pick in enumerate(picks):
        cnt = np.bincount(pick, minlength=n_cl).astype(float)
        T = cnt @ tot
        R[k] = (cnt @ rec) / T if T > 0 else np.nan
        A = cnt @ na
        P5[k] = (cnt @ p5) / A if A > 0 else np.nan
        P3[k] = (cnt @ p3) / A if A > 0 else np.nan
    return R, P5, P3


def pack_paired(point, d):
    return P.pack_paired(point, d)


def holm(pvals: dict[str, float], alpha=ALPHA) -> dict[str, dict]:
    """Holm-Bonferroni step-down over a family of one-sided bootstrap p-values."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    out, rejected_so_far = {}, True
    for r, k in enumerate(order):
        thr = alpha / (m - r)
        rej = rejected_so_far and (pvals[k] <= thr)
        rejected_so_far = rej
        out[k] = {"p_one_sided": pvals[k], "holm_threshold": thr, "holm_reject": bool(rej),
                  "rank": r + 1, "family_size": m}
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate",
                    default=r"C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.npz")
    ap.add_argument("--banked", default=str(PARENT / "results_horizon_ps.scores.npz"))
    ap.add_argument("--out", default="results_event_retune.json")
    ap.add_argument("--scores-out", default="grid_null.npz")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # ---------------- substrate + the parent's exact fold/split construction -------------
    z = np.load(a.substrate)
    F = z["F"]
    cc = z["clip_cluster"]
    st, en = clip_runs(cc)
    N = F.shape[0]
    folds = cluster_folds(cc, 2, seed=0)
    log(f"substrate {N:,} x {F.shape[1]}  {len(st)} clips  device={device}")

    per_clip = P.events_per_clip(N)
    Y_at, V_at = {}, {}
    for L in LEADS:
        Y_at[L], V_at[L] = P.targets_at(per_clip, L)
    _, hist32 = causal_window(np.zeros((N, 1), np.float32), st, en, P.WIN_MAX)
    EVAL = V_at[P.H_MAX_S] & hist32[:, None]
    Y_EVAL = Y_at[P.DEPLOY_LEAD_S]

    splits = []
    for f in (0, 1):
        te = folds == f
        tr = ~te
        tr_cl = np.unique(cc[tr])
        perm = np.random.default_rng(0 + f).permutation(tr_cl)
        sel_cl = set(int(x) for x in perm[:max(1, int(round(P.SEL_FRAC * len(tr_cl))))])
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

    # ---------------- rebuild BOTH grids (real + clip-permuted null) --------------------
    CELL = {}
    for shuf in (False, True):
        src = F[shuf_rows] if shuf else F
        XW = {}
        for f in (0, 1):
            mu, Wp = P.fit_pca(src, np.flatnonzero(splits[f][0]), P.RANK, device=device)
            Pr = P.project(src, mu, Wp)
            for W in WINS:
                XW[(f, W)] = causal_window(Pr, st, en, W)[0]
        del src
        for W in WINS:
            for L in LEADS:
                out = np.zeros((N, 3), np.float32)
                for f in (0, 1):
                    X = XW[(f, W)]
                    fit, sel, te = splits[f]
                    fitr = fit & hist32
                    vL = V_at[L] & hist32[:, None]
                    S = P.ridge_multi(X[fitr], Y_at[L][fitr].astype(np.float32),
                                      vL[fitr].astype(np.float32), X, P.LAMBDAS)
                    for i in range(3):
                        msel = sel & vL[:, i]
                        best = (-np.inf, P.LAMBDAS[0])
                        for lam in P.LAMBDAS:
                            if msel.sum() < 50 or Y_at[L][msel, i].sum() < 5:
                                continue
                            v = P.ap_lift_safe(Y_at[L][msel, i].astype(np.float64),
                                               S[lam][msel, i].astype(np.float64))
                            if np.isfinite(v) and v > best[0]:
                                best = (v, lam)
                        out[te, i] = S[best[1]][te, i]
                CELL[(shuf, W, L)] = out
            log(f"  grid W={W:>2} shuf={shuf} done ({time.time()-t0:.0f}s)")
        del XW

    # ---------------- C-FID-GRID -------------------------------------------------------
    # ⚠️ MEASURED: a re-run of the parent's own pipeline does NOT reproduce its grid
    # bit-exactly — `torch.svd_lowrank(..., niter=4)` on CUDA is not run-to-run
    # deterministic. So the REAL arms use the parent's BANKED columns verbatim (they are
    # the published ones by definition) and C-FID-GRID becomes a MEASUREMENT of the drift
    # plus its only decision-relevant consequence: does any ALARM SET move?
    Z = np.load(a.banked)
    fid, alarm_mm = {}, {}
    for W in WINS:
        for L in LEADS:
            mine, bank = CELL[(False, W, L)], Z[cell_name(W, L)]
            fid[cell_name(W, L)] = float(np.abs(mine - bank).max())
            mm = 0
            for i in range(3):
                r = np.flatnonzero(EVAL[:, i])
                k = max(1, int(round(P.TOP_FRAC * len(r))))
                am = np.zeros(len(r), bool)
                am[np.argsort(-mine[r, i].astype(np.float64), kind="mergesort")[:k]] = True
                ab = np.zeros(len(r), bool)
                ab[np.argsort(-bank[r, i].astype(np.float64), kind="mergesort")[:k]] = True
                mm += int((am != ab).sum())
            alarm_mm[cell_name(W, L)] = mm
    fid_max = max(fid.values())
    fid_eval = float(np.abs(EVAL.astype(np.uint8) - Z["eval_rows"]).max())
    fid_y = float(np.abs(Y_EVAL.astype(np.uint8) - Z["y_lead3"]).max())
    if fid_eval > 0 or fid_y > 0:
        raise SystemExit(f"C-FID-GRID FAILED on labels: eval {fid_eval} y {fid_y}")
    log(f"C-FID-GRID: eval_rows and y_lead3 BIT-IDENTICAL; grid max|diff| {fid_max:.3e} "
        f"(NOT bit-exact — CUDA svd_lowrank); alarm-set rows moved: "
        f"{sum(alarm_mm.values())} over 25 cells x 3 situations")
    # the real arms are the PARENT'S BANKED columns, so nothing downstream depends on the drift
    for W in WINS:
        for L in LEADS:
            CELL[(False, W, L)] = Z[cell_name(W, L)]

    onsets, onset_sit, onset_cl = Z["onsets"], Z["onset_sit"], Z["onset_cluster"]

    # per-row distance to the next onset of that situation, within the clip
    NEXT = np.full((N, 3), np.inf, np.float64)
    for ci, (s0, e0) in enumerate(zip(st, en)):
        for i in range(3):
            g = onsets[(onset_sit == i) & (onsets >= s0) & (onsets < e0)]
            if g.size == 0:
                continue
            t = np.arange(s0, e0)
            k = np.searchsorted(g, t, side="right")
            has = k < g.size
            NEXT[t[has], i] = g[k[has]] - t[has]

    uniq_cl, picks = P.cluster_picks(cc)
    cl_index = {int(u): k for k, u in enumerate(uniq_cl.astype(np.int64))}
    n_cl = len(uniq_cl)
    log(f"{len(picks)} cluster draws over {n_cl} clusters prepared ({time.time()-t0:.0f}s)")

    res = {"_what": "is the SHARED sitclf re-tune a DEPLOYABLE change, and what does it cost?",
           "_prereg": "./PRE_REGISTRATION.md (hash-pinned before any number here was read)",
           "_prereg_sha1": Path("PRE_REGISTRATION.sha1").read_text().strip()
                           if Path("PRE_REGISTRATION.sha1").exists() else "MISSING",
           "_vision_only": "every arm reads frozen v1 camera latents only (PI ruling 2026-08-03)",
           "substrate": a.substrate, "banked_parent_scores": a.banked,
           "C_FID_GRID": {
               "_what": ("a re-run of the parent's own pipeline vs its banked columns. ⚠️ NOT "
                         "bit-exact: torch.svd_lowrank(niter=4) on CUDA is not run-to-run "
                         "deterministic. The REAL arms below therefore use the parent's BANKED "
                         "columns verbatim; this block measures the drift and its only "
                         "decision-relevant consequence, whether any ALARM SET moves."),
               "max_abs_diff_over_25_cells": fid_max, "per_cell": fid,
               "alarm_set_rows_moved_per_cell": alarm_mm,
               "alarm_set_rows_moved_total": int(sum(alarm_mm.values())),
               "eval_rows_max_abs_diff": fid_eval, "y_lead3_max_abs_diff": fid_y,
               "labels_bit_identical": True,
               "real_arms_use": "the parent's banked CELL_* columns, verbatim"},
           "powered_situations": list(POWERED),
           "roundabout_note": ("UNDERPOWERED_C_POW: 37 positive clusters against a bar of 40. "
                               "Reported with its n; NO VERDICT is issued for it and the bar was "
                               "not lowered."),
           "n_boot": P.N_BOOT, "top_frac": P.TOP_FRAC, "h_max_s": P.H_MAX_S,
           "per_situation": {}}

    # ---------------- per-situation event analysis --------------------------------------
    SEL_CRIT = {}                        # (shuf, W, L, f) -> mean event recall on SEL rows
    for i, s in enumerate(SITS):
        t1 = time.time()
        m = EVAL[:, i]
        rows = np.flatnonzero(m)
        rcl = np.array([cl_index[int(x)] for x in cc[rows]])
        row_of = np.full(N, -1, np.int64)
        row_of[rows] = np.arange(len(rows))
        nxt = NEXT[rows, i]
        on_m = onset_sit == i
        og = onsets[on_m]
        ocl = np.array([cl_index[int(x)] for x in onset_cl[on_m]])
        ofold = folds[og]

        cols = {}
        for W in WINS:
            for L in LEADS:
                cols[cell_name(W, L)] = CELL[(False, W, L)][rows, i].astype(np.float64)
                cols["NULL_" + cell_name(W, L)] = CELL[(True, W, L)][rows, i].astype(np.float64)
        FRZ, NFRZ = cell_name(*FROZEN_CELL), "NULL_" + cell_name(*FROZEN_CELL)

        stats = {n: qb_stat(c, rows, rcl, row_of, og, nxt) for n, c in cols.items()}
        boots = {n: qb_boot(stats[n], ocl, picks, n_cl) for n in cols}

        # ---- C-BUDGET -----------------------------------------------------------------
        na = {n: int(stats[n]["alarm"].sum()) for n in cols}
        budget_ok = len(set(na.values())) == 1

        # ---- C-CHANCE: the uniform-random floor at the identical budget ---------------
        rng = np.random.default_rng(12345 + i)
        ch_rec, ch_p5, ch_p3 = [], [], []
        for _ in range(N_CHANCE):
            d = qb_stat(rng.random(len(rows)), rows, rcl, row_of, og, nxt)
            ch_rec.append(float(d["warned"].mean()))
            ch_p5.append(float(d["useful5"].mean()))
            ch_p3.append(float(d["useful3"].mean()))
        ch = {"n_draws": N_CHANCE,
              "event_recall": {"mean": round(float(np.mean(ch_rec)), 5),
                               "p2.5": round(float(np.percentile(ch_rec, 2.5)), 5),
                               "p97.5": round(float(np.percentile(ch_rec, 97.5)), 5),
                               "min": round(float(np.min(ch_rec)), 5)},
              "alarm_precision_5s": {"mean": round(float(np.mean(ch_p5)), 5),
                                     "p2.5": round(float(np.percentile(ch_p5, 2.5)), 5),
                                     "p97.5": round(float(np.percentile(ch_p5, 97.5)), 5)},
              "alarm_precision_3s": {"mean": round(float(np.mean(ch_p3)), 5),
                                     "p2.5": round(float(np.percentile(ch_p3, 2.5)), 5),
                                     "p97.5": round(float(np.percentile(ch_p3, 97.5)), 5)}}

        # ---- per-arm rows -------------------------------------------------------------
        yv = Y_EVAL[rows, i].astype(np.float64)
        arms_out, pvals = {}, {}
        for n in cols:
            d, (R, P5, P3) = stats[n], boots[n]
            base = NFRZ if n.startswith("NULL_") else FRZ
            Rb, P5b, P3b = boots[base]
            e = {"n_alarm": na[n], "n_onsets": int(len(og)),
                 "n_onsets_warned": int(d["warned"].sum()),
                 "event_recall": P.pack_single(float(d["warned"].mean()), R),
                 "alarm_precision_5s": P.pack_single(float(d["useful5"].mean()), P5),
                 "alarm_precision_3s": P.pack_single(float(d["useful3"].mean()), P3),
                 "median_lead_s": (round(float(np.median(d["lead"][d["warned"]])), 3)
                                   if d["warned"].any() else None),
                 "vs_chance_floor": {
                     "event_recall_minus_chance_mean":
                         round(float(d["warned"].mean() - np.mean(ch_rec)), 5),
                     "event_recall_below_chance_p2.5":
                         bool(d["warned"].mean() < np.percentile(ch_rec, 2.5)),
                     "alarm_prec5_minus_chance_mean":
                         round(float(d["useful5"].mean() - np.mean(ch_p5)), 5),
                     "alarm_prec5_above_chance_p97.5":
                         bool(d["useful5"].mean() > np.percentile(ch_p5, 97.5))}}
            if n != base:
                e["paired_vs_frozen"] = {
                    "event_recall": pack_paired(
                        float(d["warned"].mean() - stats[base]["warned"].mean()), R - Rb),
                    "alarm_precision_5s": pack_paired(
                        float(d["useful5"].mean() - stats[base]["useful5"].mean()), P5 - P5b),
                    "alarm_precision_3s": pack_paired(
                        float(d["useful3"].mean() - stats[base]["useful3"].mean()), P3 - P3b)}
                dd = (R - Rb)[np.isfinite(R - Rb)]
                pvals[n] = float((1 + (dd <= 0).sum()) / (1 + dd.size))
            # C-FOLD: the same delta, restricted to each fold's onsets, global alarm set
            if not n.startswith("NULL_"):
                fold_d = {}
                for f in (0, 1):
                    sf = ofold == f
                    fold_d[f"fold{f}"] = {
                        "n_onsets": int(sf.sum()),
                        "event_recall": round(float(d["warned"][sf].mean()), 5),
                        "delta_vs_frozen": round(
                            float(d["warned"][sf].mean() - stats[FRZ]["warned"][sf].mean()), 5)}
                e["C_FOLD"] = fold_d
            # frame-level operating point at the DEPLOYED 3 s label, both denominators
            e["op_5pct_at_deploy_label_3s"] = precision_recall_at_budget(
                yv, cols[n], np.ones(len(rows), bool))
            e["ridge_params"] = ridge_params(int(n.split("_w")[1].split("_")[0]))
            arms_out[n] = e

        # ---- multiplicity over the 25 REAL cells (frozen excluded: it is the reference) -
        fam = {n: pvals[n] for n in pvals if not n.startswith("NULL_") and n != FRZ}
        holm_out = holm(fam)

        # ---- C_ORACLE_POOLED: max over the 25 cells of the POOLED delta ---------------
        def oracle(shuf):
            pre = "NULL_" if shuf else ""
            b = pre + cell_name(*FROZEN_CELL)
            cand = {pre + cell_name(W, L): (stats[pre + cell_name(W, L)]["warned"].mean()
                                            - stats[b]["warned"].mean())
                    for W in WINS for L in LEADS}
            k = max(cand, key=cand.get)
            return {"cell": k, "delta_event_recall": round(float(cand[k]), 5)}
        orc, orc_null = oracle(False), oracle(True)

        # ---- SEL-split event criterion, for RETUNE_SEL --------------------------------
        for shuf in (False, True):
            pre = "NULL_" if shuf else ""
            for f in (0, 1):
                selm = splits[f][1]
                srows = np.flatnonzero(m & selm)
                if len(srows) < 100:
                    continue
                srow_of = np.full(N, -1, np.int64)
                srow_of[srows] = np.arange(len(srows))
                sog = og[selm[og]]
                for W in WINS:
                    for L in LEADS:
                        c = CELL[(shuf, W, L)][srows, i].astype(np.float64)
                        k = max(1, int(round(P.TOP_FRAC * len(srows))))
                        al = np.zeros(len(srows), bool)
                        al[np.argsort(-c, kind="mergesort")[:k]] = True
                        w = 0
                        for g in sog:
                            idx = srow_of[max(0, g - HB):g]
                            idx = idx[idx >= 0]
                            if idx.size and al[idx].any():
                                w += 1
                        SEL_CRIT[(shuf, W, L, f, i)] = (w / len(sog)) if len(sog) else np.nan
                        SEL_CRIT[("n", f, i)] = int(len(sog))

        res["per_situation"][s] = {
            "n_scorable_rows": int(len(rows)), "n_pos_lead3": int(yv.sum()),
            "n_onsets": int(len(og)), "n_onset_clusters": int(len(np.unique(ocl))),
            "C_POW_pass": bool(len(np.unique(ocl)) >= P.C_POW_BAR),
            "C_BUDGET_all_arms_same_n_alarm": budget_ok, "n_alarm": na[FRZ],
            "C_CHANCE": ch, "arms": arms_out, "HOLM_25_cells": holm_out,
            "C_ORACLE_POOLED": orc, "C_ORACLE_POOLED_NULL": orc_null,
            "_stats_cache": None}
        log(f"  {s}: {len(og)} onsets, chance floor recall "
            f"{ch['event_recall']['mean']:.4f}, FROZEN {stats[FRZ]['warned'].mean():.4f}, "
            f"oracle {orc['cell']} {orc['delta_event_recall']:+.4f}, "
            f"null-oracle {orc_null['delta_event_recall']:+.4f} ({time.time()-t1:.0f}s)")
        # keep the per-situation stats for the RETUNE_SEL assembly pass
        res["per_situation"][s]["_keep"] = True
        globals().setdefault("_CACHE", {})[s] = (rows, rcl, row_of, og, ocl, nxt, stats, boots)

    # ---------------- RETUNE_SEL: one SHARED cell per fold, chosen on the SEL split -----
    CACHE = globals()["_CACHE"]
    retune = {}
    for shuf, tag in ((False, "RETUNE_SEL"), (True, "NULL_RETUNE_SEL")):
        for variant, sit_set in (("powered2", POWERED), ("all3", SITS)):
            pick = {}
            for f in (0, 1):
                best = (-np.inf, *FROZEN_CELL)
                for W in WINS:
                    for L in LEADS:
                        v = [SEL_CRIT.get((shuf, W, L, f, SITS.index(s)), np.nan)
                             for s in sit_set]
                        v = [x for x in v if np.isfinite(x)]
                        if v and float(np.mean(v)) > best[0]:
                            best = (float(np.mean(v)), W, L)
                pick[f] = (best[1], best[2])
            key = f"{tag}::{variant}"
            retune[key] = {"fold0_cell": f"W{pick[0][0]}_L{pick[0][1]}",
                           "fold1_cell": f"W{pick[1][0]}_L{pick[1][1]}",
                           "single_cell_in_both_folds": bool(pick[0] == pick[1]),
                           "per_situation": {}}
            for i, s in enumerate(SITS):
                rows, rcl, row_of, og, ocl, nxt, stats, boots = CACHE[s]
                pre = "NULL_" if shuf else ""
                col = np.zeros(len(rows))
                te_rows = folds[rows]
                for f in (0, 1):
                    src = CELL[(shuf, *pick[f])][rows, i].astype(np.float64)
                    col[te_rows == f] = src[te_rows == f]
                d = qb_stat(col, rows, rcl, row_of, og, nxt)
                R, P5, P3 = qb_boot(d, ocl, picks, len(uniq_cl))
                b = pre + cell_name(*FROZEN_CELL)
                Rb, P5b, P3b = boots[b]
                retune[key]["per_situation"][s] = {
                    "n_alarm": int(d["alarm"].sum()), "n_onsets": int(len(og)),
                    "n_onsets_warned": int(d["warned"].sum()),
                    "event_recall": P.pack_single(float(d["warned"].mean()), R),
                    "alarm_precision_5s": P.pack_single(float(d["useful5"].mean()), P5),
                    "paired_vs_frozen": {
                        "event_recall": pack_paired(
                            float(d["warned"].mean() - stats[b]["warned"].mean()), R - Rb),
                        "alarm_precision_5s": pack_paired(
                            float(d["useful5"].mean() - stats[b]["useful5"].mean()), P5 - P5b)}}
            log(f"  {key}: fold0={retune[key]['fold0_cell']} fold1={retune[key]['fold1_cell']} "
                f"lane_change dR="
                f"{retune[key]['per_situation']['lane_change']['paired_vs_frozen']['event_recall']['delta']:+.4f}")
    res["RETUNE_SEL"] = retune
    res["SEL_split_onsets"] = {f"{SITS[i]}.fold{f}": SEL_CRIT.get(("n", f, i))
                               for i in range(3) for f in (0, 1)}

    # ---------------- C-FID-QB2: the PROMOTED stack function on the same columns --------
    fid2, bad2 = {}, []
    for i, s in enumerate(SITS):
        for W, L in ((8, 3.0), (8, 1.0), (32, 1.0), (1, 1.0)):
            got = event_anticipation_report(
                CELL[(False, W, L)][:, i].astype(np.float64), EVAL[:, i], onsets[onset_sit == i],
                cc, top_frac=P.TOP_FRAC, h_max_s=P.H_MAX_S, deploy_lead_s=P.DEPLOY_LEAD_S)
            want = res["per_situation"][s]["arms"][cell_name(W, L)]
            dr = abs(got["event_recall"] - want["event_recall"]["point"])
            dp = abs(got["alarm_precision_h_max"] - want["alarm_precision_5s"]["point"])
            fid2[f"{s}::{cell_name(W, L)}"] = {"d_event_recall": round(dr, 9),
                                               "d_alarm_prec5": round(dp, 9),
                                               "n_alarm_match": got["n_alarm"] == want["n_alarm"],
                                               "n_onsets_match": got["n_onsets"] == want["n_onsets"]}
            if dr > 1e-4 or dp > 1e-4 or got["n_alarm"] != want["n_alarm"]:
                bad2.append((s, W, L, dr, dp))
    res["C_FID_QB_promoted_stack_fn"] = {
        "_what": ("every headline re-derived by tanitad.eval.sitclf_deploy.event_anticipation_report"
                  " — a second implementation, not this script checking itself"),
        "cells": fid2, "MISMATCHES": bad2,
        "PASS": not bad2}
    log(f"C-FID-QB (promoted stack fn): {len(fid2)} cells, {len(bad2)} mismatches")

    for s in res["per_situation"]:
        res["per_situation"][s].pop("_stats_cache", None)
        res["per_situation"][s].pop("_keep", None)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(a.scores_out,
                        **{"NULL_" + cell_name(W, L): CELL[(True, W, L)]
                           for W in WINS for L in LEADS})
    log(f"wrote {a.out} and {a.scores_out} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
