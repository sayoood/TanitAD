"""cueprobe.py -- PREREG_V8F_PROOF_ADD5 (CUEPROBE), frozen card F:/TanitAD-scratch/v8proof/PREREG_V8F_PROOF_ADD5.md.
Zero training, CPU only. Tier T0 for the probe reads (frozen tokens + ego f8, closed-form kernel ridge, cross-fitted by episode), T1-kin on logged frames for the shield reads.
A diagnostic of what the input interface carries; not a deployable arm, never driving performance. STRATEGIC: PAUSED (PI item 25).
    python cueprobe.py build [--max-clips N] [--threads 4]    now-frame inputs of the 2,399 windows -> cue/cache/ (Gram matrices + E + meta)
    python cueprobe.py run   [--smoke] [--nboot 2000]          reads 1-4, controls k1-k4 -> RESULT_CUEPROBE.json, VERDICT_CUEPROBE.md (exit 2 if a control fails)
Definitions with file:line pointers and the numbered deviations R1.. : CUE_DEFS.md (same folder).
"""
import argparse
import gc
import glob
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
V8 = "F:/TanitAD-scratch/v8proof"
RES_DIR, RULE_DIR = V8 + "/res", V8 + "/rule"
CACHE = HERE + "/cache"
for _p in (RULE_DIR, RES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import resprobe as RP  # noqa: E402   (numpy-only at import; torch is imported lazily inside its P2 probe, which this card does not use)
import stop_shield as SH  # noqa: E402  (pure numpy)

ART = "E:/Projects/TanitAD-artifacts"
FR_CACHE = f"{ART}/sfresh_inputs/cache/eval/clips"
TRACKS = f"{ART}/sfresh_inputs/inputs/native_tracks.pt"
RAW = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
T1 = "F:/TanitAD-scratch/stopbal_read/t1"
BANKED = RULE_DIR + "/RESULT_RULESHIELD.json"
LATE_STEPS = 10                                   # k4 deliberate regression: tokens 2.0 s later (cache step = 0.2 s)
SETS = ("E", "V40", "V160", "V40+E", "V160+E")
FOLD_SEEDS = (0, 1, 2, 3, 4)
PLAN_SEEDS = (0, 1)
DT, A = 0.2, SH.A_BRAKE
Q_SHIELD = A * DT
TOL = 1e-6
FPR_BOUND, CONF_LEVEL = 0.02, 0.9
COV_BAND = (0.84, 0.95)
STOP_GI, CI_STOP_X, CI_STOP_Y = 6, 16, 17
LON_FAM = ("ade_m", "fde_m", "speed_mae_mps", "along_mae_m", "accel_mae_mps2", "dk_hw_abs_err_m", "dk_tg_abs_err_s")
LAT_FAM = ("cross_mae_m", "heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm")
ARMS = ("PF-PD", "OF-PD", "PF-OD", "OF-OD")
BAND_KEYS = {"operative_0_2s": (0, 10), "tactical_2_6s": (10, 30), "full_0_6s": (0, 30)}


def set_threads(n=4):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(n)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


_RS = []


def rs():
    """ruleshield_read.py imported as a module (heavy: torch, tanitad from rule/tree); used only for load(), the join, arm_arrays, band_rows and the CI function."""
    if not _RS:
        import ruleshield_read as m
        _RS.append(m)
    return _RS[0]


# ===================================================================================== pure layer (unit-tested, numpy only)
def tau_for_fpr(neg_scores, fpr=FPR_BOUND):
    """Smallest threshold tau such that the share of ``neg_scores`` with score >= tau is <= fpr (card: false-positive rate <= 0.02 on the training non-stop windows)."""
    s = np.sort(np.asarray(neg_scores, np.float64))
    n = len(s)
    if n == 0:
        return float("inf")
    kmax = int(math.floor(fpr * n + 1e-12))
    cand = np.unique(s)
    cnt = n - np.searchsorted(s, cand, side="left")
    ok = cand[cnt <= kmax]
    return float(ok.min()) if len(ok) else float(np.nextafter(s[-1], np.inf))


def conformal_q(resid, level=CONF_LEVEL):
    """Split-conformal margin: the ceil((n+1)*level)-th smallest residual (d_hat - d_true), so d_hat - q <= d_true with probability >= level under exchangeability."""
    r = np.sort(np.asarray(resid, np.float64))
    n = len(r)
    if n == 0:
        return float("nan")
    k = min(n, int(math.ceil((n + 1) * level - 1e-12)))
    return float(r[k - 1])


def assert_rows_not_after(rows_used, now_rows):
    """k4 leak guard: every token row used for a window is at or before the now-frame of that window."""
    rows_used, now_rows = np.asarray(rows_used), np.asarray(now_rows)
    if rows_used.shape != now_rows.shape:
        raise AssertionError(f"leak guard: shape {rows_used.shape} vs {now_rows.shape}")
    bad = rows_used > now_rows
    if bad.any():
        raise AssertionError(f"LEAK: {int(bad.sum())} token rows lie after the now-frame (max excess {int((rows_used - now_rows).max())} steps)")
    return True


def cue_d0(plan, xy):
    if not np.all(np.isfinite(xy)):
        return None
    return SH.rest_arc(plan, xy)[0]


def apply_d0(plan, d0s, a=A, dt=DT, fallback=True):
    """plan [N,30,2], d0s list (None or non-finite = inactive: the plan row is left untouched) -> shielded plans, per-window records (the logic of ruleshield_read.apply_variant)."""
    out, rec = plan.copy(), []
    for i, d0 in enumerate(d0s):
        if d0 is None or not np.isfinite(d0):
            rec.append(None)
            continue
        r = SH.shield_plan(plan[i], float(d0), a=a, dt=dt, fallback=fallback)
        out[i] = r["plan"]
        rec.append(r)
    return out, rec


def overrun_of(plan_i, d_ref, dt=DT):
    v = SH.seg_speeds(plan_i, dt)
    s_end = float(v.sum() * dt)
    return max(0.0, s_end - d_ref), max(0.0, s_end + float(SH.brake_dist(max(v[-1] - Q_SHIELD, 0.0), Q_SHIELD, dt)) - d_ref)


def make_idx_draws(eidx, B, seed=0):
    """B episode-cluster resamples as index arrays into the window vector (one draw shared by every paired comparison)."""
    u = np.unique(eidx)
    lists = [np.flatnonzero(eidx == e) for e in u]
    rng = np.random.default_rng(seed)
    return [np.concatenate([lists[k] for k in rng.integers(0, len(u), len(u))]) for _ in range(B)]


def boot_median_gain(draws, ae_ref, ae_set):
    """gain = median|err|(ref) - median|err|(set); positive = the set is better. Paired episode-cluster bootstrap over the same draws."""
    pt = float(np.median(ae_ref) - np.median(ae_set))
    g = np.array([np.median(ae_ref[ix]) - np.median(ae_set[ix]) for ix in draws])
    lo, hi = np.percentile(g, [2.5, 97.5])
    return {"gain": pt, "lo": float(lo), "hi": float(hi), "separated_better": bool(lo > 0), "n_windows": int(len(ae_ref))}


def auc_from_boot(boot):
    return RP.ci(boot)


# ===================================================================================== features and probes (RESPROBE code path)
def fold_features(Ks, E, folds):
    """Per outer fold: {set: [N, D] float64}. Kernel PCA (rank 256, fitted on the training rows only, RP.kpca_fit) per grid; +E = the standardised f8 scaled to the mean score variance (RP.e_block)."""
    N = len(E)
    out = []
    for f in range(RP.N_FOLDS):
        tr = np.flatnonzero(folds != f)
        d = {}
        mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-9
        d["E"] = (E - mu) / sd
        for g, K in Ks.items():
            proj, _ = RP.kpca_fit(K, tr, rank=min(RP.RANK, max(8, len(tr) // 3)))
            S = proj(np.arange(N))
            sv = float(S[tr].var(0).mean())
            d[g] = S
            d[g + "+E"] = np.concatenate([S, RP.e_block(E, tr, sv)], 1)
        out.append(d)
    return out


def flag_cv(FF, folds, y, valid, epi, sets):
    """Out-of-fold stop-window scores (kernel ridge, alpha inside the training folds) for the labelled windows."""
    N = len(y)
    oof = {s: np.full(N, np.nan) for s in sets}
    for f in range(RP.N_FOLDS):
        tr = np.flatnonzero((folds != f) & valid)
        te = (folds == f) & valid
        for s in sets:
            F = FF[f][s]
            oof[s][te] = RP.fit_p1(F[tr], y[tr], epi[tr])(F[te])
    return oof


def dist_cv(FF, folds, d, stop, epi, sets):
    """Out-of-fold rest-distance predictions for the stop windows (probe fitted on the stop windows of the training folds) + the constant (training-fold median) floor."""
    N = len(d)
    oof = {s: np.full(N, np.nan) for s in sets}
    const = np.full(N, np.nan)
    for f in range(RP.N_FOLDS):
        tr = np.flatnonzero((folds != f) & stop)
        te = (folds == f) & stop
        if len(tr) < 8 or not te.any():
            continue
        const[te] = float(np.median(d[tr]))
        for s in sets:
            F = FF[f][s]
            oof[s][te] = RP.fit_p1(F[tr], d[tr], epi[tr])(F[te])
    return oof, const


def pipeline(fs, Ks, E, eid_str, epi, valid, stop, y, d_by_seed):
    """One fold seed: probes for all five sets (read 1, read 2) and, per plan seed, the selection / margin / threshold of the shield arms.
    Selection (set with the lowest inner-fold DIST median |error|), q (inner cross-fitted residual quantile) and tau are computed from the TRAINING folds of each outer fold only."""
    N = len(y)
    folds = RP.make_folds(eid_str, seed=fs)
    FF = fold_features(Ks, E, folds)
    res = {"folds": folds, "flag_oof": {s: np.full(N, np.nan) for s in SETS}, "plan": {}}
    for ps in d_by_seed:
        res["plan"][ps] = {"dist_oof": {s: np.full(N, np.nan) for s in SETS}, "const": np.full(N, np.nan), "dhat": np.full(N, np.nan), "q": np.full(N, np.nan),
                           "tau": np.full(N, np.nan), "tau_cf": np.full(N, np.nan), "flag": np.full(N, np.nan), "chosen": [None] * RP.N_FOLDS, "inner_med": [None] * RP.N_FOLDS,
                           "tau_in_train_fpr": [None] * RP.N_FOLDS}
    for f in range(RP.N_FOLDS):
        te, tr = folds == f, folds != f
        tr_lab = np.flatnonzero(tr & valid)
        sc = {}
        for s in SETS:
            F = FF[f][s]
            sc[s] = RP.fit_p1(F[tr_lab], y[tr_lab], epi[tr_lab])(F)
            res["flag_oof"][s][te & valid] = sc[s][te & valid]
        for ps, d in d_by_seed.items():
            P = res["plan"][ps]
            m_tr = np.flatnonzero(tr & stop)
            if len(m_tr) < 8 or len(np.unique(epi[m_tr])) < 8:       # degenerate (smoke): no distance probe, shield arms inactive on this fold
                P["chosen"][f] = "E"
                continue
            if (te & stop).any():
                P["const"][te & stop] = float(np.median(d[m_tr]))
            dall = {}
            for s in SETS:
                dall[s] = RP.fit_p1(FF[f][s][m_tr], d[m_tr], epi[m_tr])(FF[f][s])
                P["dist_oof"][s][te] = dall[s][te]
            ifo = RP.make_folds(eid_str[m_tr], seed=fs + 101)
            inner, med = {}, {}
            for s in SETS:
                io = np.full(len(m_tr), np.nan)
                for j in range(RP.N_FOLDS):
                    a, b = ifo != j, ifo == j
                    if a.sum() >= 5 and b.any() and len(np.unique(epi[m_tr[a]])) >= 5:
                        io[b] = RP.fit_p1(FF[f][s][m_tr[a]], d[m_tr[a]], epi[m_tr[a]])(FF[f][s][m_tr[b]])
                inner[s] = io
                med[s] = float(np.nanmedian(np.abs(io - d[m_tr])))
            ch = min(SETS, key=lambda s: (np.inf if not np.isfinite(med[s]) else med[s]))
            P["chosen"][f], P["inner_med"][f] = ch, med
            res_c = inner[ch] - d[m_tr]
            q = conformal_q(res_c[np.isfinite(res_c)])
            neg_tr = tr & valid & ~stop
            tau_in = tau_for_fpr(sc[ch][neg_tr])
            lab_ep = eid_str[tr_lab]
            ifl = RP.make_folds(lab_ep, seed=fs + 202)
            cf = np.full(len(tr_lab), np.nan)
            for j in range(RP.N_FOLDS):
                a, b = ifl != j, ifl == j
                if a.sum() >= 5 and b.any() and len(np.unique(epi[tr_lab[a]])) >= 5:
                    cf[b] = RP.fit_p1(FF[f][ch][tr_lab[a]], y[tr_lab[a]], epi[tr_lab[a]])(FF[f][ch][tr_lab[b]])
            cfn = cf[~stop[tr_lab]]
            tau_cf = tau_for_fpr(cfn[np.isfinite(cfn)])
            P["dhat"][te], P["q"][te], P["tau"][te], P["tau_cf"][te], P["flag"][te] = dall[ch][te], q, tau_in, tau_cf, sc[ch][te]
            P["tau_in_train_fpr"][f] = float((sc[ch][neg_tr] >= tau_in).mean())
    return res


# ===================================================================================== reads
def read_flag(Fo, y, valid, epi, C):
    yy = np.where(valid, y, np.nan)
    auc = {s: RP.auc_pooled(yy, Fo[s]) for s in SETS}
    out = {"auroc": auc, "n_labelled": int(valid.sum()), "n_stop": int(((y == 1) & valid).sum())}
    if C is not None:
        bt = {s: RP.boot_auc(epi, yy, Fo[s], C) for s in SETS}
        out["auroc_ci95"] = {s: RP.ci(bt[s]) for s in SETS}
        out["vs_E"] = {s: RP.diff_row(bt[s], bt["E"], auc[s], auc["E"]) for s in SETS if s != "E"}
    return out


def read_dist(P, d, stop, epi, draws):
    ds, ep = d[stop], epi[stop]
    out = {"n_stop": int(stop.sum()), "sets": {}}
    const = P["const"][stop]
    ae_c = np.abs(const - ds)
    out["const_floor"] = {"median_abs_err_m": float(np.median(ae_c)), "r2": RP.r2_pooled(ds, const)}
    for s in SETS:
        p = P["dist_oof"][s][stop]
        ae = np.abs(p - ds)
        out["sets"][s] = {"median_abs_err_m": float(np.median(ae)), "mean_abs_err_m": float(ae.mean()), "r2": RP.r2_pooled(ds, p)}
    best = min(SETS, key=lambda s: out["sets"][s]["median_abs_err_m"])
    out["best_set"] = best
    if draws is not None:
        ae_b = np.abs(P["dist_oof"][best][stop] - ds)
        ae_e = np.abs(P["dist_oof"]["E"][stop] - ds)
        out["best_vs_const"] = boot_median_gain(draws, ae_c, ae_b)
        out["best_vs_E"] = boot_median_gain(draws, ae_e, ae_b) if best != "E" else {"gain": 0.0, "lo": 0.0, "hi": 0.0, "separated_better": False, "note": "best set is E itself"}
        vis = min(("V40", "V160", "V40+E", "V160+E"), key=lambda s: out["sets"][s]["median_abs_err_m"])
        out["best_vision_set"] = vis
        ae_v = np.abs(P["dist_oof"][vis][stop] - ds)
        out["bestvision_vs_const"] = boot_median_gain(draws, ae_c, ae_v)
        out["bestvision_vs_E"] = boot_median_gain(draws, ae_e, ae_v)
        m = out["sets"][best]["median_abs_err_m"]
        better = out["best_vs_const"]["separated_better"] and out["best_vs_E"]["separated_better"]
        out["label"] = "CUE-USABLE" if (m <= 5.0 and better) else ("CUE-COARSE" if (5.0 < m <= 8.0 and better) else "CUE-ABSENT")
    return out


def coverage(P, d, stop):
    m = stop & np.isfinite(P["dhat"])
    if not m.any():
        return {"share": None, "n": 0, "label": "NOT-EVALUATED (no held-out stop window with a distance probe)"}
    d0 = P["dhat"][m] - P["q"][m]
    k = int((d0 <= d[m] + 1e-12).sum())
    sh = k / int(m.sum())
    lab = "COVERAGE-HOLDS" if COV_BAND[0] <= sh <= COV_BAND[1] else ("COVERAGE-FAILS (below band: d_0 overshoots d_true too often)" if sh < COV_BAND[0] else "COVERAGE-FAILS (above band: margin too conservative)")
    return {"share": sh, "n": int(m.sum()), "n_covered": k, "band": list(COV_BAND), "label": lab,
            "median_margin_m": float(np.median(P["q"][m])), "chosen_sets": [c for c in P["chosen"]]}


# ===================================================================================== shield arms (read 3)
def arm_d0(arm, plan, stop, flag, tau, dhat, q, d_true):
    """d_0 per window (None = inactive) for one arm of the card."""
    N = len(plan)
    act = (flag >= tau) & np.isfinite(flag) & np.isfinite(tau)
    d0 = [None] * N
    for i in range(N):
        est = dhat[i] - q[i] if np.isfinite(dhat[i]) and np.isfinite(q[i]) else None
        if arm == "PF-PD":
            d0[i] = est if act[i] else None
        elif arm == "OF-PD":
            d0[i] = est if stop[i] else None
        elif arm == "PF-OD":
            d0[i] = (d_true[i] if stop[i] else est) if act[i] else None
        elif arm == "OF-OD":
            d0[i] = d_true[i] if stop[i] else None
        else:
            raise ValueError(arm)
    return d0, act


def light_deltas(plan, P, g, stop):
    BL = rs().BL
    a_s = BL.band_rows(P, g, 10, 30)["ade_m"][stop].mean() - BL.band_rows(plan, g, 10, 30)["ade_m"][stop].mean()
    a_n = BL.band_rows(P, g, 0, 30)["ade_m"][~stop].mean() - BL.band_rows(plan, g, 0, 30)["ade_m"][~stop].mean()
    return {"stop_2_6s_ade_delta": float(a_s), "nonstop_0_6s_ade_delta": float(a_n)}


def arm_arrays_full(P, X, J):
    R = rs()
    a = R.SG.arm_arrays(P, X["g"], J)
    a["full_0_6s"] = R.BL.band_rows(P, X["g"], 0, 30)
    return a


def cells_for(arrs, null, sets, eid, nb):
    CI = rs().CI
    cells = {}
    for band in BAND_KEYS:
        for met in sorted(null[band]):
            if met not in LON_FAM + LAT_FAM:
                continue
            for sn, mk in sets.items():
                a, b = np.asarray(arrs[band][met], np.float64), np.asarray(null[band][met], np.float64)
                m = mk & np.isfinite(a) & np.isfinite(b)
                key = f"{band}|{met}|{sn}"
                if m.sum() < 2:
                    cells[key] = {"status": "NOT-APPLICABLE", "n_windows": int(m.sum())}
                    continue
                r = CI.paired_episode_cluster_bootstrap(a[m], b[m], eid[m], n_boot=nb, seed=0)
                cells[key] = {"status": "OK", "family": "LONGITUDINAL" if met in LON_FAM else "LATERAL", "null_mean": float(b[m].mean()), "arm_mean": float(a[m].mean()),
                              "delta": float(a[m].mean() - b[m].mean()), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]),
                              "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(eid[m])))}
    return cells


def shield_read(X, J, pl, d_true, valid, nb, k3_out):
    """Full read for one plan seed at fold seed 0: four arms, four metric families, paired bootstrap against NULL. pl = pipeline plan block."""
    R = rs()
    plan, stop, eid = X["plan"], X["stop"], X["eid"]
    N = len(plan)
    nonstop = ~stop
    sets = {"stop": stop, "nonstop": nonstop, "all": np.ones(N, bool)}
    null = arm_arrays_full(plan, X, J)
    sidx = np.flatnonzero(stop)
    res = {"n_windows": int(N), "n_stop": int(stop.sum()), "n_nonstop": int(nonstop.sum()), "arms": {}}
    # k3: no active window -> bit-identical to NULL
    P0, rec0 = apply_d0(plan, [None] * N)
    k3_out["no_active_window_bit_identical"] = bool(np.array_equal(P0, plan) and all(r is None for r in rec0))
    Pinf, _ = apply_d0(plan, arm_d0("PF-PD", plan, stop, pl["flag"], np.full(N, np.inf), pl["dhat"], pl["q"], d_true)[0])
    k3_out["tau_inf_bit_identical"] = bool(np.array_equal(Pinf, plan))
    for arm in ARMS:
        d0, act = arm_d0(arm, plan, stop, pl["flag"], pl["tau"], pl["dhat"], pl["q"], d_true)
        P, rec = apply_d0(plan, d0)
        arrs = arm_arrays_full(P, X, J)
        cells = cells_for(arrs, null, sets, eid, nb)
        a = np.array([overrun_of(P[i], d_true[i])[0] for i in sidx])
        b = np.array([overrun_of(plan[i], d_true[i])[0] for i in sidx])
        r = R.CI.paired_episode_cluster_bootstrap(a, b, eid[sidx], n_boot=nb, seed=0)
        cells["OVERRUN_vs_logged_rest_m|stop"] = {"status": "OK", "family": "LONGITUDINAL", "null_mean": float(b.mean()), "arm_mean": float(a.mean()), "delta": float(a.mean() - b.mean()),
                                                  "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]), "n_windows": int(len(sidx))}
        tp, fp = int((act & stop & valid).sum()), int((act & ~stop & valid).sum())
        fn, tn = int((~act & stop & valid).sum()), int((~act & ~stop & valid).sum())
        ov = np.array([overrun_of(P[i], d_true[i]) for i in sidx])
        ovn = np.array([overrun_of(plan[i], d_true[i]) for i in sidx])
        cued = [i for i in range(N) if rec[i] is not None]
        infeas_stop = [i for i in sidx if rec[i] is not None and not rec[i]["feasible"]]
        res["arms"][arm] = {"cells": cells,
                            "tactical": {"active": int(act.sum()), "TP": tp, "FP": fp, "FN": fn, "TN": tn, "hit_rate": tp / max(1, tp + fn), "fp_rate": fp / max(1, fp + tn),
                                         "plan_lon_label_changed_share": float(np.mean(arrs["_lon"] != null["_lon"])), "plan_lat_label_changed_share": float(np.mean(arrs["_lat"] != null["_lat"]))},
                            "longitudinal_overrun_stop": {"share_overrun": float((ov[:, 0] > TOL).mean()), "mean_m": float(ov[:, 0].mean()), "mean_tail_m": float(ov[:, 1].mean()),
                                                          "null_share_overrun": float((ovn[:, 0] > TOL).mean()), "null_mean_m": float(ovn[:, 0].mean()), "n_stop_infeasible_at_start": len(infeas_stop)},
                            "n_cued": len(cued)}
        if arm == "OF-OD":
            res["_P_OFOD"] = None
    return res


def k2_check(shield, seed, full):
    """Control k2: OF-OD through this pipeline equals the banked RULESHIELD ORACLE numbers."""
    bk = json.load(open(BANKED, encoding="utf-8"))["seeds"][str(seed)]
    c = shield["arms"]["OF-OD"]["cells"]["tactical_2_6s|ade_m|stop"]
    b = bk["cells"]["ORACLE|tactical_2_6s|ade_m|stop_131"]
    ov, bo = shield["arms"]["OF-OD"]["longitudinal_overrun_stop"], bk["overrun_vs_logged_rest_stop_windows"]["ORACLE"]
    out = {"evaluated": bool(full), "mine_delta": c["delta"], "banked_delta": b["delta"], "abs_diff": abs(c["delta"] - b["delta"]), "mine_overrun_share": ov["share_overrun"], "banked_overrun_share": bo["share_overrun"],
           "mine_overrun_mean_m": ov["mean_m"], "banked_overrun_mean_m": bo["mean_m"], "n_stop_infeasible_at_start": ov["n_stop_infeasible_at_start"]}
    out["pass"] = bool((not full) or (out["abs_diff"] <= 1e-9 and abs(ov["share_overrun"] - bo["share_overrun"]) <= 1e-9 and abs(ov["mean_m"] - bo["mean_m"]) <= 1e-9))
    return out


def shield_outcome(sh, kk):
    """Arm PF-PD outcome for one plan seed."""
    g = sh["arms"]["PF-PD"]["cells"]["tactical_2_6s|ade_m|stop"]
    n = sh["arms"]["PF-PD"]["cells"]["full_0_6s|ade_m|nonstop"]
    orc = sh["arms"]["OF-OD"]["cells"]["tactical_2_6s|ade_m|stop"]["delta"]
    gain_ok = bool(g["delta"] < 0 and g["hi"] < 0)
    size = abs(g["delta"]) / abs(orc) if orc else float("nan")
    size_ok = bool(g["delta"] < 0 and size >= 0.30)
    non_ok = bool(n["hi"] <= 0.05)
    lab = "CUE-PROBE-COSTS" if not non_ok else ("CUE-PROBE-GAINS" if (gain_ok and size_ok) else "CUE-PROBE-NO-GAIN")
    return {"label": lab, "stop_2_6s_ade": g, "nonstop_0_6s_ade": n, "oracle_gain_m": orc, "size_vs_oracle": size, "bars": {"ade_negative_ci_excl_0": gain_ok, "size_ge_0.30_of_oracle": size_ok, "nonstop_hi_le_0.05": non_ok}}


# ===================================================================================== build
def cmd_build(a):
    set_threads(a.threads)
    import torch
    torch.set_num_threads(a.threads)
    sys.path[:0] = [p for p in os.environ.get("PYTHONPATH", "").split(";") if p]
    import tanitad
    from tanitad.data import v8f_mini_data as D
    print("[build] tanitad", os.path.abspath(tanitad.__file__), flush=True)
    os.makedirs(CACHE, exist_ok=True)
    fs = sorted(glob.glob(f"{T1}/dump_BAL-NULL_s0_m10_causal_heads/ep*.npz"))
    assert len(fs) == 141, len(fs)
    zs = [np.load(f) for f in fs]
    ci = np.concatenate([z["clip_index"] for z in zs]).astype(np.int64)
    ws = np.concatenate([z["ws"] for z in zs]).astype(np.int64)
    v0d = np.concatenate([z["v0"] for z in zs]).astype(np.float64)
    assert len(ws) == 2399
    clip_of = np.load(RAW + "/join_eval141.npz", allow_pickle=False)["clip"].astype(str)
    ucl = np.unique(ci)
    use_cl = ucl if a.max_clips <= 0 else ucl[:a.max_clips]
    win = np.flatnonzero(np.isin(ci, use_cl))
    n = len(win)
    tr = D.load_native_tracks(TRACKS)
    fk = D._fk()
    paths = {k: f"{CACHE}/_tmp_{k}.npy" for k in ("X40", "X160", "X40L", "X160L")}
    mm = {"X40": np.lib.format.open_memmap(paths["X40"], mode="w+", dtype=np.float16, shape=(n, 40 * 1024)),
          "X160": np.lib.format.open_memmap(paths["X160"], mode="w+", dtype=np.float16, shape=(n, 160 * 1024)),
          "X40L": np.lib.format.open_memmap(paths["X40L"], mode="w+", dtype=np.float16, shape=(n, 40 * 1024)),
          "X160L": np.lib.format.open_memmap(paths["X160L"], mode="w+", dtype=np.float16, shape=(n, 160 * 1024))}
    kin = np.full((n, 3), np.nan)
    rows_real, rows_late = np.zeros(n, np.int64), np.zeros(n, np.int64)
    t0 = time.time()
    for k, c in enumerate(use_cl):
        pos = np.flatnonzero(ci[win] == c)                 # positions in the selected set
        now = ws[win][pos]
        name = clip_of[int(c)]
        o = torch.load(f"{FR_CACHE}/{name}.pt", map_location="cpu", weights_only=True, mmap=True)
        T = o["pool4x10"].shape[0]
        assert o["pool4x10"].shape[1:] == (40, 1024) and o["pool8x20"].shape[1:] == (160, 1024), (name, o["pool4x10"].shape, o["pool8x20"].shape)
        rows = now.copy()                                  # the rows actually used for the card inputs: the now-frame of the window, nothing later
        assert_rows_not_after(rows, now)                   # k4 leak guard, at the moment of indexing
        late = now + LATE_STEPS
        assert late.max() < T, (name, late.max(), T)       # a window always has 6 s (30 steps) of future, so this holds
        for key, grid, rr in (("X40", "pool4x10", rows), ("X160", "pool8x20", rows), ("X40L", "pool4x10", late), ("X160L", "pool8x20", late)):
            mm[key][pos] = o[grid][torch.as_tensor(rr)].to(torch.float32).reshape(len(rr), -1).numpy().astype(np.float16)
        rows_real[pos], rows_late[pos] = rows, late
        k3 = D.yaw_kin_from_track(tr[name])[0]
        kin[pos] = k3[now]
        del o
        if k % 20 == 0:
            print(f"[build] {k}/{len(use_cl)} clips {time.time() - t0:.0f}s", flush=True)
    for m in mm.values():
        m.flush()
    del mm, m
    gc.collect()
    E = fk.kin_features(kin).astype(np.float64)
    assert np.isfinite(E).all(), "non-finite E at a now-frame"
    np.savez(f"{CACHE}/frames.npz", win=win, ci=ci[win], ws=ws[win], clip=clip_of[ci[win]], E=E, kin=kin, rows_real=rows_real, rows_late=rows_late, v0_dump=v0d[win],
             kin_v0_maxdiff=float(np.max(np.abs(kin[:, 0] - v0d[win]))))
    for tag, key in (("K40", "X40"), ("K160", "X160"), ("K40L", "X40L"), ("K160L", "X160L")):
        X = np.load(paths[key], mmap_mode="r")
        K = np.zeros((n, n), np.float64)
        for s0 in range(0, X.shape[1], 8192):
            Xb = np.asarray(X[:, s0:s0 + 8192], np.float32)
            K += Xb @ Xb.T
        np.save(f"{CACHE}/{tag}.npy", K.astype(np.float32))
        X._mmap.close()
        del X, Xb
        gc.collect()
        print(f"[build] {tag} {K.shape} {time.time() - t0:.0f}s", flush=True)
    sizes = []
    gc.collect()
    for p in paths.values():
        sizes.append((p, os.path.getsize(p)))
        for attempt in range(5):
            try:
                os.remove(p)
                break
            except PermissionError:
                gc.collect()
                time.sleep(2)
        else:
            print("[build] WARNING could not delete temp", p, flush=True)
    json.dump({"deleted_temp": sizes, "n_windows": int(n), "n_clips": int(len(use_cl))}, open(f"{CACHE}/build_deleted_temp.json", "w", encoding="utf-8"))
    print("[build] done; deleted temp", sizes, flush=True)


# ===================================================================================== run
def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def cmd_run(a):
    set_threads(a.threads)
    R = rs()
    t0 = time.time()
    Z = np.load(f"{CACHE}/frames.npz")
    win = Z["win"]
    full = len(win) == 2399
    Ks = {"V40": np.load(f"{CACHE}/K40.npy"), "V160": np.load(f"{CACHE}/K160.npy")}
    Kl = {"V40": np.load(f"{CACHE}/K40L.npy"), "V160": np.load(f"{CACHE}/K160L.npy")}
    E = Z["E"]
    XS = {}
    for s in PLAN_SEEDS:
        x = R.load(s)
        for k in ("plan", "g", "v0", "eid", "ci", "ws", "stop", "has", "or_xy"):
            x[k] = x[k][win]
        XS[s] = x
    X0 = XS[0]
    assert np.array_equal(X0["ci"], Z["ci"]) and np.array_equal(X0["ws"], Z["ws"]), "cache windows differ from the dump windows"
    assert np.array_equal(XS[1]["stop"], X0["stop"]) and np.array_equal(XS[1]["has"], X0["has"]) and np.array_equal(XS[1]["ws"], X0["ws"]), "seeds disagree on windows/labels"
    assert_rows_not_after(Z["rows_real"], Z["ws"])
    eid_str = X0["eid"].astype(str)
    uep = sorted(set(eid_str))
    epi = np.array([uep.index(e) for e in eid_str])
    N = len(win)
    stop, has = X0["stop"].astype(bool), X0["has"].astype(bool)
    valid = has.copy()
    y = stop.astype(np.float64)
    d_by_seed = {}
    for s in PLAN_SEEDS:
        d = np.full(N, np.nan)
        for i in np.flatnonzero(stop):
            v = cue_d0(XS[s]["plan"][i], XS[s]["or_xy"][i])
            d[i] = np.nan if v is None else v
        assert np.isfinite(d[stop]).all(), "non-finite d_true on a stop window"
        d_by_seed[s] = d
    nb = a.nboot if not a.smoke else min(a.nboot, 200)
    fseeds = FOLD_SEEDS if not a.smoke else (0,)
    C = RP.boot_counts(len(uep), B=nb)
    draws = make_idx_draws(epi[stop], nb)
    meta = {"card": "PREREG_V8F_PROOF_ADD5.md", "n_windows": int(N), "n_stop": int(stop.sum()), "n_labelled": int(valid.sum()), "n_stop_without_label": int((stop & ~valid).sum()), "n_episodes": len(uep),
            "full_set": bool(full), "smoke": bool(a.smoke), "n_boot": nb, "fold_seeds": list(fseeds), "tiers": "T0 probe reads; T1-kin shield reads", "STRATEGIC": "PAUSED (PI item 25)",
            "kin_v0_maxdiff_vs_dump_mps": float(Z["kin_v0_maxdiff"]), "tanitad_file": os.path.abspath(R.tanitad.__file__)}
    res = {"meta": meta, "fold_seed": {}, "controls": {}}
    ctl = res["controls"]
    pipes = {}
    for fs in fseeds:
        pipes[fs] = pipeline(fs, Ks, E, eid_str, epi, valid, stop, y, d_by_seed)
        print(f"[run] pipeline fold seed {fs} {time.time() - t0:.0f}s", flush=True)
        blk = {"flag": read_flag(pipes[fs]["flag_oof"], y, valid, epi, C if fs == 0 else None), "dist": {}, "coverage": {}}
        for s in PLAN_SEEDS:
            P = pipes[fs]["plan"][s]
            blk["dist"][s] = read_dist(P, d_by_seed[s], stop, epi, draws if fs == 0 else None)
            blk["coverage"][s] = coverage(P, d_by_seed[s], stop)
            blk["dist"][s]["selected_sets_by_fold"] = P["chosen"]
            blk["dist"][s]["tau_train_fpr_by_fold"] = P["tau_in_train_fpr"]
            if fs != 0:                                                       # spread beside the interval: point estimates of the PF-PD deltas
                d0, act = arm_d0("PF-PD", X0["plan"], stop, P["flag"], P["tau"], P["dhat"], P["q"], d_by_seed[s])
                Pp, _ = apply_d0(XS[s]["plan"], d0)
                blk.setdefault("pfpd_points", {})[s] = light_deltas(XS[s]["plan"], Pp, XS[s]["g"], stop)
        res["fold_seed"][fs] = blk
    p0 = pipes[0]
    # ------- read 3 (fold seed 0, both plan seeds)
    res["shield"], res["outcome_shield"], res["k2"] = {}, {}, {}
    k3 = {}
    for s in PLAN_SEEDS:
        X = XS[s]
        J = R.SG.join({"v0": X["v0"], "ws": X["ws"], "clip_index": X["ci"]})
        k3s = {}
        sh = shield_read(X, J, p0["plan"][s], d_by_seed[s], valid, nb, k3s)
        sh.pop("_P_OFOD", None)
        res["shield"][s], k3[s] = sh, k3s
        res["k2"][s] = k2_check(sh, s, full)
        res["outcome_shield"][s] = shield_outcome(sh, res["k2"][s])
        # secondary, not a card outcome: the tactical rates at the inner cross-fitted threshold
        P = p0["plan"][s]
        act_cf = (P["flag"] >= P["tau_cf"]) & valid
        res["shield"][s]["secondary_tau_cf"] = {"hit_rate": float((act_cf & stop).sum() / max(1, (stop & valid).sum())), "fp_rate": float((act_cf & ~stop).sum() / max(1, (~stop & valid).sum())), "active": int(act_cf.sum())}
        print(f"[run] shield plan seed {s} {time.time() - t0:.0f}s", flush=True)
    outs = [res["outcome_shield"][s]["label"] for s in PLAN_SEEDS]
    res["outcome_shield_combined"] = ("CUE-PROBE-GAINS" if all(o == "CUE-PROBE-GAINS" for o in outs) else ("CUE-PROBE-COSTS" if all(o == "CUE-PROBE-COSTS" for o in outs) else "CUE-PROBE-NO-GAIN"))
    res["outcome_shield_per_seed"] = {str(s): res["outcome_shield"][s]["label"] for s in PLAN_SEEDS}
    # ------- outcomes of reads 1, 2, 4
    fl = res["fold_seed"][0]["flag"]
    if "vs_E" in fl:
        d160, d40 = fl["vs_E"]["V160+E"], fl["vs_E"]["V40+E"]
        res["outcome_flag"] = {"label": "VISION-ADDS" if (d160["diff"] >= 0.02 and d160["separated_pos"]) else "VISION-FLAT", "V160+E_minus_E": d160,
                               "V40+E_minus_E": d40, "V40+E_label": "VISION-ADDS" if (d40["diff"] >= 0.02 and d40["separated_pos"]) else "VISION-FLAT"}
    rank = {"CUE-USABLE": 0, "CUE-COARSE": 1, "CUE-ABSENT": 2}
    dl = {s: res["fold_seed"][0]["dist"][s]["label"] for s in PLAN_SEEDS}
    res["outcome_dist_per_seed"] = {str(s): dl[s] for s in PLAN_SEEDS}
    res["outcome_dist_combined"] = max(dl.values(), key=lambda x: rank[x])
    res["outcome_coverage"] = {str(s): res["fold_seed"][0]["coverage"][s]["label"] for s in PLAN_SEEDS}
    # ------- controls
    rng = np.random.default_rng(12345)
    FF0 = fold_features(Ks, E, p0["folds"])
    vi = np.flatnonzero(valid)
    ysh = y.copy()
    ysh[vi] = y[rng.permutation(vi)]
    ok_sh = flag_cv(FF0, p0["folds"], ysh, valid, epi, SETS)
    k1f = {s: RP.auc_pooled(np.where(valid, ysh, np.nan), ok_sh[s]) for s in SETS}
    si = np.flatnonzero(stop)
    dsh = d_by_seed[0].copy()
    dsh[si] = d_by_seed[0][rng.permutation(si)]
    sdo, sconst = dist_cv(FF0, p0["folds"], dsh, stop, epi, SETS)
    k1d = {}
    ae_c = np.abs(sconst[stop] - dsh[stop])
    for s in SETS:
        ae = np.abs(sdo[s][stop] - dsh[stop])
        k1d[s] = {"median_abs_err_m": float(np.median(ae)), "const_floor_m": float(np.median(ae_c)), "better_than_floor": boot_median_gain(draws, ae_c, ae)["separated_better"]}
    ctl["k1"] = {"auroc_shuffled": k1f, "dist_shuffled": k1d, "pass": bool(all(abs(v - 0.5) <= 0.05 for v in k1f.values()) and not any(v["better_than_floor"] for v in k1d.values()))}
    ctl["k2"] = {str(s): res["k2"][s] for s in PLAN_SEEDS}
    ctl["k2"]["pass"] = bool(all(res["k2"][s]["pass"] for s in PLAN_SEEDS))
    ctl["k3"] = {str(s): k3[s] for s in PLAN_SEEDS}
    ctl["k3"]["pass"] = bool(all(v["no_active_window_bit_identical"] and v["tau_inf_bit_identical"] for v in k3.values()))
    guard_fires = False
    try:
        assert_rows_not_after(Z["rows_late"], Z["ws"])
    except AssertionError as e:
        guard_fires = True
    FFl = fold_features(Kl, E, p0["folds"])
    lset = ("V40", "V160", "V40+E", "V160+E")
    olate = flag_cv(FFl, p0["folds"], y, valid, epi, lset)
    yy = np.where(valid, y, np.nan)
    real_auc = {s: RP.auc_pooled(yy, p0["flag_oof"][s]) for s in lset}
    late_auc = {s: RP.auc_pooled(yy, olate[s]) for s in lset}
    ctl["k4"] = {"guard_passes_now_frame_rows": True, "guard_fires_on_late_rows": guard_fires, "auroc_real": real_auc, "auroc_late_2s": late_auc,
                 "late_raises_auroc": {s: bool(late_auc[s] > real_auc[s]) for s in lset}, "pass": bool(guard_fires and late_auc["V160+E"] > real_auc["V160+E"] and late_auc["V40+E"] > real_auc["V40+E"])}
    res["controls_pass"] = bool(ctl["k1"]["pass"] and ctl["k2"]["pass"] and ctl["k3"]["pass"] and ctl["k4"]["pass"])
    meta["runtime_s"] = time.time() - t0
    json.dump(to_py(res), open(HERE + "/RESULT_CUEPROBE.json", "w", encoding="utf-8"), indent=1)
    open(HERE + "/VERDICT_CUEPROBE.md", "w", encoding="utf-8").write(verdict_md(res))
    print("WROTE RESULT_CUEPROBE.json VERDICT_CUEPROBE.md controls_pass", res["controls_pass"], f"{time.time() - t0:.0f}s", flush=True)
    if not res["controls_pass"]:
        sys.exit(2)


def pf(b):
    return "PASS" if b else "FAIL"


def verdict_md(res):
    m, ctl = res["meta"], res["controls"]
    L = ["# VERDICT_CUEPROBE (PREREG_V8F_PROOF_ADD5, frozen 2026-11-13)", "",
         f"Evidence class MEASURED (ours, zero training, CPU). Tier T0 for the probe reads (frozen tokens, closed-form kernel ridge, 5 folds by episode), T1-kin on logged frames for the shield reads. n windows {m['n_windows']} ({m['n_stop']} stop windows, {m['n_labelled']} labelled), {m['n_episodes']} episodes; paired episode-cluster bootstrap B {m['n_boot']}; fold seeds {m['fold_seeds']} (outcomes read at fold seed 0, the others shown as spread). A diagnostic of the input interface, never driving performance. STRATEGIC family: PAUSED (PI item 25), not a pass, not omitted.",
         f"Full window set: {m['full_set']}; smoke: {m['smoke']}.", ""]
    L += ["## Controls", f"* all controls pass: **{res['controls_pass']}**" + ("" if res["controls_pass"] else "  (a control failed: the reads below are not valid)"),
          f"* k1 shuffled labels {pf(ctl['k1']['pass'])}: AUROC {json.dumps({k: round(v, 3) for k, v in ctl['k1']['auroc_shuffled'].items()})}; DIST better than floor with CI: {json.dumps({k: v['better_than_floor'] for k, v in ctl['k1']['dist_shuffled'].items()})}",
          f"* k2 OF-OD vs banked ORACLE {pf(ctl['k2']['pass'])}: " + "; ".join(f"seed {s}: delta {ctl['k2'][str(s)]['mine_delta']:.10f} vs {ctl['k2'][str(s)]['banked_delta']:.10f} (diff {ctl['k2'][str(s)]['abs_diff']:.1e}), overrun share {ctl['k2'][str(s)]['mine_overrun_share']:.4f} vs {ctl['k2'][str(s)]['banked_overrun_share']:.4f}, infeasible-at-start stop windows {ctl['k2'][str(s)]['n_stop_infeasible_at_start']}, evaluated {ctl['k2'][str(s)]['evaluated']}" for s in PLAN_SEEDS),
          f"* k3 no active window bit-identical to NULL {pf(ctl['k3']['pass'])}",
          f"* k4 leak guard {pf(ctl['k4']['pass'])}: guard fires on tokens 2.0 s later {ctl['k4']['guard_fires_on_late_rows']}; AUROC real {json.dumps({k: round(v, 3) for k, v in ctl['k4']['auroc_real'].items()})} vs late {json.dumps({k: round(v, 3) for k, v in ctl['k4']['auroc_late_2s'].items()})}", ""]
    fl = res["fold_seed"][0]["flag"]
    L += ["## Read 1 FLAG (T0, AUROC over labelled windows, fold seed 0)", "", "| set | AUROC | 95 % CI | minus E | CI | separated |", "|---|---|---|---|---|---|"]
    for s in SETS:
        v = fl.get("vs_E", {}).get(s)
        ci = fl.get("auroc_ci95", {}).get(s)
        L.append(f"| {s} | {fl['auroc'][s]:.4f} | {ci} | " + (f"{v['diff']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['separated_pos']} |" if v else "- | - | - |"))
    if "outcome_flag" in res:
        o = res["outcome_flag"]
        L += ["", f"Outcome: **{o['label']}** (V160+E minus E {o['V160+E_minus_E']['diff']:+.4f}, CI {o['V160+E_minus_E']['ci95']}); V40+E minus E: **{o['V40+E_label']}** ({o['V40+E_minus_E']['diff']:+.4f}, CI {o['V40+E_minus_E']['ci95']}). Bar: diff >= 0.02 and CI excludes 0: {pf(o['label'] == 'VISION-ADDS')}."]
    L += ["", "AUROC spread over fold seeds (V160+E minus E): " + ", ".join(f"fs{fs} {res['fold_seed'][fs]['flag']['auroc']['V160+E'] - res['fold_seed'][fs]['flag']['auroc']['E']:+.4f}" for fs in res["fold_seed"]), ""]
    for s in PLAN_SEEDS:
        dd = res["fold_seed"][0]["dist"][s]
        L += [f"## Read 2 DIST (T0, plan seed {s}, n stop {dd['n_stop']}, fold seed 0)", "", "| set | median abs err m | mean abs err m | R2 |", "|---|---|---|---|",
              f"| constant (training-fold median) | {dd['const_floor']['median_abs_err_m']:.3f} | - | {dd['const_floor']['r2']:.3f} |"]
        for k in SETS:
            v = dd["sets"][k]
            L.append(f"| {k} | {v['median_abs_err_m']:.3f} | {v['mean_abs_err_m']:.3f} | {v['r2']:.3f} |")
        bc, be = dd.get("best_vs_const"), dd.get("best_vs_E")
        if bc:
            L += ["", f"Best set {dd['best_set']}: median {dd['sets'][dd['best_set']]['median_abs_err_m']:.3f} m (bar <= 5.0 usable, <= 8.0 coarse); gain over constant floor {bc['gain']:+.3f} m CI [{bc['lo']:+.3f}, {bc['hi']:+.3f}] better: {bc['separated_better']}; over E {be['gain']:+.3f} m CI [{be['lo']:+.3f}, {be['hi']:+.3f}] better: {be['separated_better']}.",
                  f"Best vision set {dd['best_vision_set']}: vs const {dd['bestvision_vs_const']['gain']:+.3f} [{dd['bestvision_vs_const']['lo']:+.3f}, {dd['bestvision_vs_const']['hi']:+.3f}]; vs E {dd['bestvision_vs_E']['gain']:+.3f} [{dd['bestvision_vs_E']['lo']:+.3f}, {dd['bestvision_vs_E']['hi']:+.3f}].",
                  f"Outcome: **{dd['label']}**. Median error of the best set over fold seeds: " + ", ".join(f"fs{fs} {res['fold_seed'][fs]['dist'][s]['sets'][res['fold_seed'][fs]['dist'][s]['best_set']]['median_abs_err_m']:.2f}" for fs in res["fold_seed"]), ""]
    L += [f"DIST outcome (needs both plan seeds, the weaker label is reported): **{res['outcome_dist_combined']}**; per seed {res['outcome_dist_per_seed']}", ""]
    L += ["## Read 3 SHIELD with the probe cue (T1-kin, fold seed 0; a = 3.0 m/s2, dt 0.2 s; paired episode-cluster bootstrap against NULL)", ""]
    for s in PLAN_SEEDS:
        o, sh = res["outcome_shield"][s], res["shield"][s]
        c, n = o["stop_2_6s_ade"], o["nonstop_0_6s_ade"]
        L += [f"### Plan seed {s}: **{o['label']}**", "",
              "| bar | measured | result |", "|---|---|---|",
              f"| stop-window 2-6 s ADE change negative, CI excludes 0 | {c['delta']:+.4f} m [{c['lo']:+.4f}, {c['hi']:+.4f}] (n {c['n_windows']}) | {pf(o['bars']['ade_negative_ci_excl_0'])} |",
              f"| size >= 0.30 of the ORACLE gain at this seed | {o['size_vs_oracle']:.3f} (ORACLE {o['oracle_gain_m']:+.4f} m) | {pf(o['bars']['size_ge_0.30_of_oracle'])} |",
              f"| non-stop 0-6 s ADE change, upper CI end <= +0.05 m | {n['delta']:+.4f} m [{n['lo']:+.4f}, {n['hi']:+.4f}] (n {n['n_windows']}) | {pf(o['bars']['nonstop_hi_le_0.05'])} |", "",
              "| arm | stop 2-6 s ADE | non-stop 0-6 s ADE | all 0-6 s ADE | overrun share (stop, NULL) | overrun mean m | hit rate | FP rate | active | TP/FP/FN/TN | lon label changed |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            a = sh["arms"][arm]
            cs = lambda k: (lambda x: f"{x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]{'*' if x['separated'] else ''}")(a["cells"][k])
            t, ov = a["tactical"], a["longitudinal_overrun_stop"]
            L.append(f"| {arm} | {cs('tactical_2_6s|ade_m|stop')} | {cs('full_0_6s|ade_m|nonstop')} | {cs('full_0_6s|ade_m|all')} | {ov['share_overrun']:.3f} ({ov['null_share_overrun']:.3f}) | {ov['mean_m']:.3f} | {t['hit_rate']:.3f} | {t['fp_rate']:.4f} | {t['active']} | {t['TP']}/{t['FP']}/{t['FN']}/{t['TN']} | {t['plan_lon_label_changed_share']:.3f} |")
        L += ["", f"Secondary (not a card outcome): tactical rates at the inner cross-fitted threshold: {sh['secondary_tau_cf']}.", "",
              "All cells (family; variant minus NULL; * = CI excludes 0):", "", "| arm | cell | delta [lo, hi] | n |", "|---|---|---|---|"]
        for arm in ARMS:
            for k, c2 in sorted(sh["arms"][arm]["cells"].items()):
                if c2.get("status") == "OK":
                    L.append(f"| {arm} | {c2['family']} {k} | {c2['delta']:+.4f} [{c2['lo']:+.4f}, {c2['hi']:+.4f}]{'*' if c2['separated'] else ''} | {c2['n_windows']} |")
        L.append("")
        pts = [(fs, res["fold_seed"][fs].get("pfpd_points", {}).get(s)) for fs in res["fold_seed"] if fs != 0]
        if pts:
            L.append(f"PF-PD point deltas at the other fold seeds (spread, no interval): " + "; ".join(f"fs{fs} stop {p['stop_2_6s_ade_delta']:+.3f} / non-stop {p['nonstop_0_6s_ade_delta']:+.3f}" for fs, p in pts if p) + "\n")
    L += [f"SHIELD outcome (PF-PD, needs both seeds): **{res['outcome_shield_combined']}**; per seed {res['outcome_shield_per_seed']}", ""]
    L += ["## Read 4 Coverage (held-out stop windows with d_0 <= d_true, nominal 0.9, band 0.84 to 0.95 for n = 131)", ""]
    for s in PLAN_SEEDS:
        cv = res["fold_seed"][0]["coverage"][s]
        L.append(f"* plan seed {s}: share {cv['share']} ({cv.get('n_covered')}/{cv['n']}), median margin q {cv.get('median_margin_m')}: **{cv.get('label')}**; other fold seeds: " + ", ".join(f"fs{fs} {res['fold_seed'][fs]['coverage'][s]['share']}" for fs in res["fold_seed"] if fs != 0))
    L += ["", "## Outcomes in the card's words", "",
          f"* FLAG: {res.get('outcome_flag', {}).get('label')}  (V40+E: {res.get('outcome_flag', {}).get('V40+E_label')})",
          f"* DIST: {res['outcome_dist_combined']}", f"* SHIELD (PF-PD): {res['outcome_shield_combined']}", f"* COVERAGE: {res['outcome_coverage']}",
          "", "Reading choices and deviations R1.. are in CUE_DEFS.md (same folder). A separated interval answers 'another draw of episodes' only; the closed-form probes have no training seed, the fold assignment is the one free choice and its spread is shown above.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nboot", type=int, default=2000)
    args = ap.parse_args()
    {"build": cmd_build, "run": cmd_run}[args.cmd](args)
