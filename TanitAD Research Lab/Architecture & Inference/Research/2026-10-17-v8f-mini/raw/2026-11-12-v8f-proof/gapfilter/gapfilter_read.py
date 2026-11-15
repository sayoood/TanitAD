r"""gapfilter_read.py -- the reader of the frozen card PREREG_V8F_PROOF_ADD28 (GAPFILTER). Definitions: GAPFILTER_DEFS.md.
The closing speed of the lead from a causal filter (E1 slope, E2 alpha-beta) over the head's own per-row gap reads. Zero training, existing files, CPU only.
Tier T0: a diagnostic on logged frames, never driving performance. STRATEGIC: PAUSED (PI item 25); LONGITUDINAL, LATERAL, TACTICAL: not measured by this read.

  python gapfilter_read.py run --arms LS_s0,... --in-dir DIR --states-dir DIR --out-dir DIR [--B 2000]

The metric, label, window, bootstrap, record-check and R3 / DELTA logic is imported from leadsup_read_n1.py (the reader of card ADD24 with note N1).
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

ROOT = "F:/TanitAD-scratch/v8proof"
READ_DIR = ROOT + "/spatial/leadsup/read"
if READ_DIR not in sys.path:
    sys.path.insert(0, READ_DIR)
import leadsup_read_n1 as N  # noqa: E402

H = N.H
O = N.O
RP = N.RP
DT = O.DT
S0F = O.S0F
TIER = O.TIER
FAMILIES = O.FAMILIES
KS = (3, 5, 8, 10, 15)
ALPHAS = (0.2, 0.35, 0.5, 0.7)
BETAS = (0.02, 0.05, 0.1, 0.2)
MIN_PRESENT = 3
MAX_MISS = 5
JUMP = None                                   # None = the card (E1 and E2 as written). A float = the post-hoc same-lead variant (reports only, names nothing)
BAR_F2 = 0.8
BAR_PERM = 0.02
FLOOR = 0.089
SIGMAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
F8_KS = (5, 10, 15)
F8_DRAWS = 20
ARMS = N.ARMS
TYPES = N.TYPES
N1_RESULT = READ_DIR + "/out_n1/RESULT_LEADSUP_T0_N1.json"
OUT_JSON = "RESULT_GAPFILTER.json"
OUT_MD = "VERDICT_GAPFILTER.md"
STAMP = ("Tier T0 (a diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). "
         "LONGITUDINAL, LATERAL, TACTICAL: not measured by this read. Evidence class MEASURED (ours, T0).")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================================ estimators (pure)
def present_mask(gap, plogit):
    """A row is present when its gap and presence logit are finite and the presence probability is at least 0.5 (logit >= 0)."""
    gap, plogit = np.asarray(gap, np.float64), np.asarray(plogit, np.float64)
    return np.isfinite(gap) & np.isfinite(plogit) & (plogit >= 0.0)


def e1_series(gap, present, K):
    """E1 for every row of one dense series: least-squares slope (m/s) of the gap over the present rows of the K slots ending at the row; NaN below 3 present rows."""
    gap = np.asarray(gap, np.float64)
    g = np.concatenate([np.full(K - 1, np.nan), np.where(np.asarray(present, bool), gap, np.nan)])
    sw = sliding_window_view(g, K)
    m = np.isfinite(sw)
    if JUMP is not None:                      # post-hoc variant only (off in the card run): the trailing run of the same lead, as the label path
        alive = np.ones(len(sw), bool)
        last = np.full(len(sw), np.nan)
        keep = np.zeros(sw.shape, bool)
        for j in range(K - 1, -1, -1):
            v = sw[:, j]
            fin = np.isfinite(v)
            with np.errstate(invalid="ignore"):
                brk = fin & alive & np.isfinite(last) & (np.abs(v - last) > JUMP)
            ok = fin & alive & ~brk
            keep[:, j] = ok
            last = np.where(ok, v, last)
            alive = alive & ~brk
        m = keep
    c = O.ls_slope(np.where(m, sw, 0.0), m)             # (a window slot outside the kept run is excluded from the fit)
    c = np.asarray(c, np.float64).copy()
    c[m.sum(1) < MIN_PRESENT] = np.nan
    return c


def e2_series(gap, present, alpha, beta):
    """E2 for every row of one dense series: alpha-beta filter, restart at the start of the series and after more than MAX_MISS missing rows in a row; the estimate (the rate)
    is NaN until MIN_PRESENT present rows since the restart."""
    n = len(gap)
    out = np.full(n, np.nan)
    gl = np.asarray(gap, np.float64).tolist()
    pl = np.asarray(present, bool).tolist()
    k = beta / DT
    alive = False
    cnt = 0
    miss = 0
    x = 0.0
    r = 0.0
    prev_m = None
    for i in range(n):
        if alive:
            x += r * DT
        if pl[i]:
            m = gl[i]
            if JUMP is not None and alive and prev_m is not None and abs(m - prev_m) > JUMP:
                alive = False                  # post-hoc variant only: a step above the bound is another lead, restart
            prev_m = m
            if not alive:
                x = m
                r = 0.0
                alive = True
                cnt = 1
            else:
                res = m - x
                x += alpha * res
                r += k * res
                cnt += 1
            miss = 0
        elif alive:
            miss += 1
            if miss > MAX_MISS:
                alive = False
                cnt = 0
                miss = 0
        if alive and cnt >= MIN_PRESENT:
            out[i] = r
    return out


def cand_key(cand):
    return "E1_K%d" % cand[1] if cand[0] == "E1" else "E2_a%g_b%g" % cand[1]


def all_cands():
    return [("E1", K) for K in KS] + [("E2", (a, b)) for a in ALPHAS for b in BETAS]


def estimate_series(cand, gap, present):
    return e1_series(gap, present, cand[1]) if cand[0] == "E1" else e2_series(gap, present, cand[1][0], cand[1][1])


def estimate_rows(cand, gap_rows, present_rows, starts, lens):
    """Per-clip application of a candidate to the row arrays of a layout (rows of a clip are contiguous, t = 0 .. len-1)."""
    out = np.full(len(gap_rows), np.nan)
    for s, ln in zip(starts, lens):
        out[s:s + ln] = estimate_series(cand, gap_rows[s:s + ln], present_rows[s:s + ln])
    return out


def make_layout(clip_idx, t, n_clips):
    """starts / lens of the contiguous clips; raises when the rows are not sorted by clip with t = 0 .. len-1."""
    clip_idx = np.asarray(clip_idx, np.int64)
    t = np.asarray(t, np.int64)
    lens = np.bincount(clip_idx, minlength=n_clips)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    ok = bool((np.diff(clip_idx) >= 0).all())
    if ok:
        ok = bool(np.array_equal(t, np.concatenate([np.arange(ln) for ln in lens])))
    if not ok:
        raise SystemExit("REFUSED: rows are not contiguous per clip with t = 0, 1, 2, ...")
    return starts.astype(np.int64), lens.astype(np.int64)


# ============================================================================================ calibration, selection (train rows only)
def fit_cal(e, y):
    m = np.isfinite(e) & np.isfinite(y)
    if m.sum() < 3:
        return (float("nan"), float("nan"))
    A = np.c_[e[m], np.ones(int(m.sum()))]
    a, b = np.linalg.lstsq(A, y[m], rcond=None)[0]
    return (float(a), float(b))


def apply_F(e, h, cal):
    """F = calibrated estimate where it exists, else the closing-speed output of the head."""
    return np.where(np.isfinite(e), cal[0] * e + cal[1], h)


def cv_predict(e, y, h, fold, rows):
    """Clip-grouped cross-fitted F on the rows; returns (prediction array, list of fold MSE)."""
    pred = np.full(len(e), np.nan)
    mses = []
    for f in sorted(set(fold[rows].tolist())):
        te = rows & (fold == f)
        tr = rows & (fold != f)
        cal = fit_cal(np.where(tr, e, np.nan), np.where(tr, y, np.nan))
        pred[te] = apply_F(e[te], h[te], cal)
        mm = te & np.isfinite(pred) & np.isfinite(y)
        if mm.any():
            mses.append(float(np.mean((y[mm] - pred[mm]) ** 2)))
    return pred, mses


def select_from(cand_est, y, h, fold, rows):
    """Selection: the lowest mean fold MSE among E1 (five K) and among E2 (sixteen gains), then the lower of the two; calibration of every candidate on all rows."""
    scores = {}
    for c, e in cand_est.items():
        _, mses = cv_predict(e, y, h, fold, rows)
        scores[cand_key(c)] = float(np.mean(mses)) if mses else float("inf")
    e1 = [c for c in cand_est if c[0] == "E1"]
    e2 = [c for c in cand_est if c[0] == "E2"]
    best1 = min(e1, key=lambda c: scores[cand_key(c)]) if e1 else None
    best2 = min(e2, key=lambda c: scores[cand_key(c)]) if e2 else None
    pool = [c for c in (best1, best2) if c is not None]
    chosen = min(pool, key=lambda c: scores[cand_key(c)])
    cal = {cand_key(c): fit_cal(np.where(rows, e, np.nan), np.where(rows, y, np.nan)) for c, e in cand_est.items()}
    return {"chosen": chosen, "chosen_key": cand_key(chosen), "best_E1": best1, "best_E2": best2, "scores": scores, "cal": cal, "n_rows": int(rows.sum())}


def train_candidates(train):
    pres = present_mask(train["gap"], train["plog"])
    return {c: estimate_rows(c, train["gap"], pres, train["starts"], train["lens"]) for c in all_cands()}


def fit_F(train, eval_labels=None):
    """The whole choice of F. Receives the train probe rows only; `eval_labels` is accepted so that control f5 can pass noise and show that it is never read."""
    cand_est = train_candidates(train)
    sel = select_from(cand_est, train["y"], train["clos"], train["fold"], train["rows"])
    return sel, cand_est


def sel_signature(sel):
    return {"chosen": sel["chosen_key"], "scores": sel["scores"], "cal": {k: list(v) for k, v in sel["cal"].items()}}


def control_f5(fit_fn, train, labels_real, labels_noise):
    s1, _ = fit_fn(train, labels_real)
    s2, _ = fit_fn(train, labels_noise)
    a, b = sel_signature(s1), sel_signature(s2)
    return {"pass": bool(a == b), "chosen_real": a["chosen"], "chosen_noise_labels": b["chosen"], "calibration_chosen_real": a["cal"][a["chosen"]], "calibration_chosen_noise_labels": b["cal"][b["chosen"]],
            "all_scores_and_calibrations_identical": bool(a == b)}


def window_F(sel, gap_pref, plog_pref, h_last):
    """F of one window from its causal prefix (rows 0 .. S0 of the clip) and the closing output of the head at S0."""
    e = estimate_series(sel["chosen"], gap_pref, present_mask(gap_pref, plog_pref))[-1]
    cal = sel["cal"][sel["chosen_key"]]
    return float(cal[0] * e + cal[1]) if np.isfinite(e) else float(h_last), bool(np.isfinite(e))


# ============================================================================================ small pure helpers for the reads
def r2_point(y, p, m):
    m = m & np.isfinite(y) & np.isfinite(p)
    return float(RP.r2_pooled(y[m], p[m])) if m.sum() > 2 else float("nan")


def paired_r2(y, pa, pb, valid, ep, n, C):
    m = valid & np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb) & (ep >= 0)
    Sa = RP.ep_stats_r2(ep[m], y[m], pa[m], n)
    Sb = RP.ep_stats_r2(ep[m], y[m], pb[m], n)
    d = RP.diff_row(RP.boot_r2(Sa, C), RP.boot_r2(Sb, C), RP.r2_pooled(y[m], pa[m]), RP.r2_pooled(y[m], pb[m]))
    d["n"] = int(m.sum())
    return d


def pearson_stats(x, y):
    return np.array([len(x), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()], np.float64)


def corr_from(S):
    n, sx, sy, sxx, syy, sxy = [S[..., i] for i in range(6)]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (n * sxy - sx * sy) / np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))


def f7_stats(head_gap, label_gap, head_present, clip_idx, n_clips, C, only_present=False):
    """f7: error of the head gap on the evaluation rows with a gap label: RMSE, RMSE of its change between consecutive rows, lag-1 autocorrelation; episode-cluster intervals."""
    err = head_gap - label_gap
    ok = np.isfinite(err)
    if only_present:
        ok = ok & head_present
    ci = np.asarray(clip_idx)
    same = np.r_[False, ci[1:] == ci[:-1]]
    pair = ok & np.r_[False, ok[:-1]] & same
    prev = np.r_[np.nan, err[:-1]]
    d = err - prev
    Sn = np.zeros(n_clips)
    Ss = np.zeros(n_clips)
    np.add.at(Sn, ci[ok], 1.0)
    np.add.at(Ss, ci[ok], err[ok] ** 2)
    Dn = np.zeros(n_clips)
    Ds = np.zeros(n_clips)
    np.add.at(Dn, ci[pair], 1.0)
    np.add.at(Ds, ci[pair], d[pair] ** 2)
    P = np.zeros((n_clips, 6))
    for c in np.unique(ci[pair]):
        mm = pair & (ci == c)
        P[c] = pearson_stats(prev[mm], err[mm])
    out = {"n_rows": int(ok.sum()), "n_pairs": int(pair.sum())}
    out["rmse_m"] = float(np.sqrt(Ss.sum() / max(Sn.sum(), 1)))
    out["rmse_m_ci95"] = RP.ci(np.sqrt((C @ Ss) / np.maximum(C @ Sn, 1)))
    out["rmse_change_m"] = float(np.sqrt(Ds.sum() / max(Dn.sum(), 1)))
    out["rmse_change_m_ci95"] = RP.ci(np.sqrt((C @ Ds) / np.maximum(C @ Dn, 1)))
    out["lag1_autocorr"] = float(corr_from(P.sum(0)))
    out["lag1_autocorr_ci95"] = RP.ci(corr_from(C @ P))
    return out


# ============================================================================================ names (pure)
def _sep(row, key="ci95"):
    return bool(row is not None and row.get(key) is not None and row["ci95"][0] > 0)


def name_outcomes_gf(arms_res, gate_ok, floor=FLOOR):
    """Card section 5, per arm type, both seeds needed. arms_res[arm] = dict(read, M1F{r2, r2_ci95}, paired{diff, ci95}, delta_f{skill, ci95}, delta_perm_seed0)."""
    if not gate_ok:
        return {"named": False, "reason": "a control failed; no outcome is named"}
    per = {}
    for ty in TYPES:
        rs = [arms_res.get("%s_s%d" % (ty, s)) for s in (0, 1)]
        if any(r is None or not r.get("read") for r in rs):
            per[ty] = {"closing": None, "slot": None, "reason": "a seed of this arm type is missing or was not read; no outcome is named"}
            continue
        m1f = all(r["M1F"]["r2"] > 0 and r["M1F"]["r2_ci95"][0] > 0 for r in rs)
        pd = all(r["paired"]["diff"] > 0 and r["paired"]["ci95"][0] > 0 for r in rs)
        mean_d = float(np.mean([r["paired"]["diff"] for r in rs]))
        big = bool(mean_d > floor)
        closing = "FILTER-READS-CLOSING" if (m1f and pd and big) else "FILTER-FLAT"
        dl = all(r["delta_f"]["skill"] > 0 and r["delta_f"]["ci95"][0] > 0 for r in rs)
        pm = all(r["delta_perm_seed0"] <= BAR_PERM for r in rs)
        slot = "FILTER-SLOT-IMAGINES" if (dl and pm) else "FILTER-SLOT-FLAT"
        per[ty] = {"closing": closing, "slot": slot, "m1f_above0_both_seeds": bool(m1f), "paired_above0_both_seeds": bool(pd), "mean_paired_diff": mean_d, "mean_paired_diff_above_floor": big,
                   "delta_f_above0_both_seeds": bool(dl), "delta_perm_at_most_bar_both_seeds": bool(pm), "floor": floor}
    return {"named": True, "per_type": per}


# ============================================================================================ data loading
def load_layout(path):
    z = np.load(path, allow_pickle=False)
    return {"clips": np.asarray(z["clips"]).astype(str), "clip_idx": np.asarray(z["clip_idx"]).astype(np.int64), "t": np.asarray(z["t"]).astype(np.int64)}


def load_lead(path):
    z = np.load(path, allow_pickle=False)
    return {k: np.asarray(z[k], np.float64) for k in ("presence_logit", "gap_m", "closing_ms")}


def bank_label_rows(names, t):
    B = np.load(O.BANKP)
    pos = {(e, int(g)): i for i, (e, g) in enumerate(zip(B["beid"].astype(str), B["bg"].astype(np.int64)))}
    lg = B["lead_gap"].astype(np.float64)
    ii = np.array([pos.get((str(e), int(g)), -1) for e, g in zip(names, t)])
    return np.where(ii >= 0, lg[np.where(ii >= 0, ii, 0)], np.nan)


def window_rows(lay, clip_names, eid, g0):
    """Row index of the S0 row of every window (-1 where the clip or the row is absent)."""
    cmap = {c: i for i, c in enumerate(clip_names)}
    out = np.full(len(eid), -1, np.int64)
    for j, (e, g) in enumerate(zip(eid, g0)):
        c = cmap.get(str(e), -1)
        if c >= 0 and 0 <= int(g) < lay["lens"][c]:
            out[j] = lay["starts"][c] + int(g)
    return out, np.array([cmap.get(str(e), -1) for e in eid])


# ============================================================================================ controls on windows
def control_f3(sel, cand_list, gap, plog, clos, lay, wrow, wclip, g0, sc, seed=7):
    """Causal: replace every row after S0 of the window's clip by random values; no estimate at S0 changes (exactly 0)."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    nan_mismatch = 0
    n = 0
    for j in np.flatnonzero(sc & (wrow >= 0)):
        c = wclip[j]
        s, ln = lay["starts"][c], lay["lens"][c]
        r0 = int(g0[j])
        g, p, h = gap[s:s + ln].copy(), plog[s:s + ln].copy(), clos[s:s + ln].copy()
        g2, p2, h2 = g.copy(), p.copy(), h.copy()
        k = ln - (r0 + 1)
        if k > 0:
            g2[r0 + 1:] = rng.uniform(0, 60, k)
            p2[r0 + 1:] = rng.normal(0, 5, k)
            h2[r0 + 1:] = rng.normal(0, 3, k)
        for cd in cand_list:
            a = estimate_series(cd, g, present_mask(g, p))[r0]
            b = estimate_series(cd, g2, present_mask(g2, p2))[r0]
            if np.isfinite(a) != np.isfinite(b):
                nan_mismatch += 1
            elif np.isfinite(a):
                worst = max(worst, abs(float(a) - float(b)))
        fa = window_F(sel, g[:r0 + 1], p[:r0 + 1], h[r0])[0]
        fb = window_F(sel, g2[:r0 + 1], p2[:r0 + 1], h2[r0])[0]
        worst = max(worst, abs(fa - fb))
        n += 1
    return {"pass": bool(worst == 0.0 and nan_mismatch == 0), "max_abs_change": float(worst), "nan_pattern_mismatches": int(nan_mismatch), "n_windows": int(n), "n_candidates": len(cand_list)}


def permuted_F(sel, gap, plog, clos, lay, wrow, wclip, g0, sc, seed):
    """F of the scored windows when their prefix sequences (gap, presence logit, closing output) are permuted among the scored windows."""
    idx = np.flatnonzero(sc & (wrow >= 0))
    perm = np.random.default_rng(seed).permutation(len(idx))
    out = np.full(len(g0), np.nan)
    for j, d in zip(idx, idx[perm]):
        c = wclip[d]
        s = lay["starts"][c]
        r0 = int(g0[d])
        out[j] = window_F(sel, gap[s:s + r0 + 1], plog[s:s + r0 + 1], clos[s + r0])[0]
    return out, bool(np.array_equal(idx, idx[perm]))


def f2_f8(ctx, lab_rows, lay, wrow, B_noise_seeds=F8_DRAWS):
    """f2 (E1 K=5 on the label gap) and f8 (label gap plus white noise through E1 at K = 5, 10, 15), closing-speed R2 on the valid windows."""
    valid = ctx["valid"]
    y = ctx["c_lab"]
    pres = np.isfinite(lab_rows)

    def r2_for(g, K):
        e = estimate_rows(("E1", K), g, pres, lay["starts"], lay["lens"])
        p = np.where(wrow >= 0, e[np.where(wrow >= 0, wrow, 0)], np.nan)
        return r2_point(y, p, valid), p

    r2_5, p5 = r2_for(lab_rows, 5)
    f2 = {"pass": bool(r2_5 >= BAR_F2), "r2_E1_K5_label_gap": r2_5, "bar": BAR_F2, "n": int((valid & np.isfinite(p5)).sum()), "n_valid": int(valid.sum())}
    f8 = {}
    for sg in SIGMAS:
        for K in F8_KS:
            vals = []
            for d in range(F8_DRAWS if sg > 0 else 1):
                noise = np.random.default_rng(d).normal(0.0, sg, len(lab_rows)) if sg > 0 else 0.0
                vals.append(r2_for(lab_rows + noise, K)[0])
            f8["sigma%g|K%d" % (sg, K)] = {"sigma_m": sg, "K": K, "r2_mean": float(np.mean(vals)), "r2_min": float(np.min(vals)), "r2_max": float(np.max(vals)), "n_draws": len(vals), "n": int(valid.sum())}
    return f2, f8


# ============================================================================================ one arm
def process_arm(arm, ctx, TR, EV, files, B, n1_ref, C_train, C_ev, ctrl, noise_labels):
    """TR: train probe inputs (layout, labels, folds); EV: evaluation layout, label rows, windows. Returns the result dict of the arm."""
    log(arm, "load")
    tl = load_lead(files["train_lead"])
    el = load_lead(files["eval_lead"])
    train = dict(TR, gap=tl["gap_m"], plog=tl["presence_logit"], clos=tl["closing_ms"])
    t0 = time.time()
    sel, cand_tr = fit_F(train, None)
    log(arm, "selection done %.0f s: %s" % (time.time() - t0, sel["chosen_key"]))
    f5 = control_f5(lambda tr, lab: fit_F(tr, lab), train, {"c_lab": ctx["c_lab"]}, noise_labels)
    ctrl["f5_no_eval_data_in_choice|%s" % arm] = f5
    lay = EV["lay"]
    eg, ep_, eh = el["gap_m"], el["presence_logit"], el["closing_ms"]
    epres = present_mask(eg, ep_)
    cand_ev = {c: estimate_rows(c, eg, epres, lay["starts"], lay["lens"]) for c in all_cands()}
    wrow, wclip = EV["wrow"], EV["wclip"]
    safe = np.where(wrow >= 0, wrow, 0)

    def at_w(a):
        return np.where(wrow >= 0, a[safe], np.nan)
    c_h = at_w(eh)
    g_h = at_w(eg)
    key = sel["chosen_key"]
    e_w = at_w(cand_ev[sel["chosen"]])
    cF = apply_F(e_w, c_h, sel["cal"][key])
    y, valid, sc = ctx["c_lab"], ctx["valid"], ctx["sc"]
    ep, n, C = ctx["ep10"], ctx["n10"], ctx["C10"]
    R = {"read": True, "files": files, "selected": {"estimator": key, "kind": sel["chosen"][0], "params": (sel["chosen"][1] if sel["chosen"][0] == "E1" else list(sel["chosen"][1])),
                                                     "calibration_a_b": list(sel["cal"][key]), "best_E1": cand_key(sel["best_E1"]), "best_E2": cand_key(sel["best_E2"]),
                                                     "cv_mse_best_E1": sel["scores"][cand_key(sel["best_E1"])], "cv_mse_best_E2": sel["scores"][cand_key(sel["best_E2"])], "n_train_rows": sel["n_rows"]}}
    R["selection_scores"] = sel["scores"]
    R["M1"] = N.fid_one(y, c_h, valid, ep, n, C)
    R["M1F"] = N.fid_one(y, cF, valid, ep, n, C)
    R["paired"] = paired_r2(y, cF, c_h, valid, ep, n, C)
    R["counts"] = {"windows_valid": int(valid.sum()), "windows_scored": int(sc.sum()), "F_estimate_missing_valid": int((valid & ~np.isfinite(e_w)).sum()),
                   "F_estimate_missing_scored": int((sc & ~np.isfinite(e_w)).sum()), "F_and_fallback_missing_valid": int((valid & ~np.isfinite(cF)).sum()),
                   "S0_row_absent": int((wrow < 0).sum()), "S0_row_not_present_by_head": int((sc & ~(at_w(epres.astype(float)) > 0)).sum())}
    # DELTA-F and the head's own DELTA through the same call
    PVc, PVn = N.n1_inputs(g_h, cF, ctx["V"], ctx["y"], sc, ctx["fold"])
    R["delta_f"] = N.delta_n1_row(ctx["y"], PVc, PVn, sc, ep, n, C)
    R["delta_f_coefs_R3c"] = PVc["coefs"]
    PVh, PVhn = N.n1_inputs(g_h, c_h, ctx["V"], ctx["y"], sc, ctx["fold"])
    R["delta_head_own"] = N.delta_n1_row(ctx["y"], PVh, PVhn, sc, ep, n, C)
    idx_sc = np.flatnonzero(sc)
    dperm = []
    for sd in range(5):
        cp = cF.copy()
        cp[idx_sc] = cF[idx_sc[np.random.default_rng(sd).permutation(len(idx_sc))]]
        a, b = N.n1_inputs(g_h, cp, ctx["V"], ctx["y"], sc, ctx["fold"])
        dperm.append(N.delta_n1_point(ctx["y"], a, b, sc))
    R["delta_perm_seed0"] = dperm[0]
    R["delta_perm_all_seeds_0_to_4"] = dperm
    # alignment x1 against the finished read of card ADD24 note N1
    ref = n1_ref["arms"][arm]
    R["alignment_x1"] = {"M1_this_path": R["M1"]["r2"], "M1_n1_result": ref["M1"]["r2"], "delta_head_this_path": R["delta_head_own"]["skill"], "delta_head_n1_result": ref["M3"]["delta_n1"]["skill"]}
    ctrl["x1_alignment_m1_delta|%s" % arm] = {"pass": bool(abs(R["M1"]["r2"] - ref["M1"]["r2"]) <= 5e-4 and abs(R["delta_head_own"]["skill"] - ref["M3"]["delta_n1"]["skill"]) <= 5e-4),
                                              "m1_this_path": R["M1"]["r2"], "m1_n1_result": ref["M1"]["r2"], "delta_this_path": R["delta_head_own"]["skill"], "delta_n1_result": ref["M3"]["delta_n1"]["skill"], "tol": 5e-4}
    # f3 causal, f4 permuted
    ctrl["f3_causal|%s" % arm] = control_f3(sel, all_cands(), eg, ep_, eh, lay, wrow, wclip, ctx["g0"], sc)
    f4 = []
    ident = None
    for sd in range(5):
        pf, ident = permuted_F(sel, eg, ep_, eh, lay, wrow, wclip, ctx["g0"], sc, sd)
        f4.append(r2_point(y, pf, valid))
    idpf = permuted_F_identity(sel, eg, ep_, eh, lay, wrow, wclip, ctx["g0"], sc)
    ctrl["f4_permuted|%s" % arm] = {"pass": bool(f4[0] <= BAR_PERM), "m1f_permuted_seed0": f4[0], "all_seeds_0_to_4": f4, "bar": BAR_PERM, "n": int(valid.sum()),
                                    "identity_permutation_reproduces_F_max_abs_diff": float(np.nanmax(np.abs(idpf[valid & np.isfinite(cF)] - cF[valid & np.isfinite(cF)])))}
    # by K and by gains (calibration of each candidate on the train probe rows)
    tbl = {}
    for c in all_cands():
        k2 = cand_key(c)
        pc = apply_F(at_w(cand_ev[c]), c_h, sel["cal"][k2])
        row = N.fid_one(y, pc, valid, ep, n, C)
        tbl[k2] = {"r2": row["r2"], "r2_ci95": row["r2_ci95"], "n": row["n"], "cv_mse_train": sel["scores"][k2], "calibration_a_b": list(sel["cal"][k2]), "estimate_missing_valid": int((valid & ~np.isfinite(at_w(cand_ev[c]))).sum())}
    R["by_candidate"] = tbl
    # E3 (names nothing): blend of E1, E2 and the head closing output, fitted on the train probe rows
    R["E3"] = read_e3(sel, cand_tr, cand_ev, train, c_h, at_w, ctx)
    # cross-fitted variant inside the evaluation bank (names nothing)
    R["crossfit_eval_bank"] = read_crossfit(cand_ev, eh, EV, c_h, at_w, ctx, wclip)
    # train probe clips (names nothing)
    e_tr = cand_tr[sel["chosen"]]
    pin = apply_F(e_tr, train["clos"], sel["cal"][key])
    pcv, mses = cv_predict(e_tr, train["y"], train["clos"], train["fold"], train["rows"])
    ci = TR["clip_idx"]
    R["train"] = {"M1F_in_sample": N.fid_one(train["y"], pin, train["rows"], ci, TR["n_clips"], C_train), "M1F_cross_fitted_5fold": N.fid_one(train["y"], pcv, train["rows"], ci, TR["n_clips"], C_train),
                  "M1_head_closing": N.fid_one(train["y"], train["clos"], train["rows"], ci, TR["n_clips"], C_train), "rows_with_closing_label": int(train["rows"].sum()),
                  "estimate_missing_rows": int((train["rows"] & ~np.isfinite(e_tr)).sum())}
    # f7 (names nothing)
    R["f7"] = {"all_rows": f7_stats(eg, EV["lab_rows"], epres, lay["clip_idx"], len(lay["starts"]), EV["C_clips"]),
               "rows_with_head_presence_at_least_half": f7_stats(eg, EV["lab_rows"], epres, lay["clip_idx"], len(lay["starts"]), EV["C_clips"], only_present=True)}
    return R


def permuted_F_identity(sel, gap, plog, clos, lay, wrow, wclip, g0, sc):
    out = np.full(len(g0), np.nan)
    for j in np.flatnonzero(sc & (wrow >= 0)):
        c = wclip[j]
        s = lay["starts"][c]
        r0 = int(g0[j])
        out[j] = window_F(sel, gap[s:s + r0 + 1], plog[s:s + r0 + 1], clos[s + r0])[0]
    return out


def read_e3(sel, cand_tr, cand_ev, train, c_h, at_w, ctx):
    k1, k2 = sel["best_E1"], sel["best_E2"]
    X = np.c_[cand_tr[k1], cand_tr[k2], train["clos"]]
    m = train["rows"] & np.isfinite(X).all(1) & np.isfinite(train["y"])
    A = np.c_[X[m], np.ones(int(m.sum()))]
    w = np.linalg.lstsq(A, train["y"][m], rcond=None)[0]
    Xe = np.c_[at_w(cand_ev[k1]), at_w(cand_ev[k2]), c_h]
    fin = np.isfinite(Xe).all(1)
    p = np.where(fin, np.c_[Xe, np.ones(len(Xe))] @ w, c_h)
    row = N.fid_one(ctx["c_lab"], p, ctx["valid"], ctx["ep10"], ctx["n10"], ctx["C10"])
    return {"weights_E1_E2_head_intercept": [float(v) for v in w], "E1": cand_key(k1), "E2": cand_key(k2), "r2": row["r2"], "r2_ci95": row["r2_ci95"], "n": row["n"],
            "windows_with_all_three": int((ctx["valid"] & fin).sum()), "names_nothing": True}


def read_crossfit(cand_ev, eh, EV, c_h, at_w, ctx, wclip):
    """Every choice (selection and calibration) fitted by episode inside the evaluation bank, applied to the windows of the held-out episodes."""
    fold = EV["fold_rows"]
    y, rows = EV["c_rows"], EV["c_ok_rows"]
    wf = np.where(wclip >= 0, EV["fold_clip"][np.where(wclip >= 0, wclip, 0)], -1)
    p = np.full(len(wf), np.nan)
    chosen = {}
    for f in range(5):
        tr = rows & (fold != f)
        sel = select_from(cand_ev, y, eh, fold, tr)
        chosen[f] = sel["chosen_key"]
        e_w = at_w(cand_ev[sel["chosen"]])
        m = wf == f
        p[m] = apply_F(e_w[m], c_h[m], sel["cal"][sel["chosen_key"]])
    row = N.fid_one(ctx["c_lab"], p, ctx["valid"], ctx["ep10"], ctx["n10"], ctx["C10"])
    return {"r2": row["r2"], "r2_ci95": row["r2_ci95"], "n": row["n"], "chosen_per_fold": chosen, "names_nothing": True}


# ============================================================================================ run
def run(arms, in_dir, states_dir, out_dir, B, pack_dir):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    P = N.load_pack(pack_dir + "/eval_pack.npz")
    PT = N.load_pack(pack_dir + "/train_pack.npz")
    ctx = N.window_ctx(P, B)
    n1_ref = json.load(open(N1_RESULT, encoding="utf-8"))
    res = {"tier": TIER, "families": FAMILIES, "card": "PREREG_V8F_PROOF_ADD28", "B": B, "arms": {}, "controls": {}, "stamp": STAMP,
           "counts": {"windows": int(len(ctx["eid"])), "scored": int(ctx["sc"].sum()), "valid_with_closing_label": int(ctx["valid"].sum()), "episodes_scored": int(ctx["n10"])}}
    ctrl = {}
    ctrl["f1_label_path"] = N.control_k1(ctx, P["meta"]["k1_reference_r2_skill"])
    # train probe inputs (arm independent)
    names = np.asarray(PT["clips"]).astype(str)[PT["clip_idx"]]
    FS = H.build_fit_sets(names, PT["t"].astype(np.int64), PT["gap"], PT["v"])
    tlay = make_layout(PT["clip_idx"], PT["t"], len(PT["clips"]))
    TR = {"starts": tlay[0], "lens": tlay[1], "y": np.asarray(FS["c"], np.float64), "rows": np.asarray(FS["c_ok"], bool) & np.isfinite(FS["c"]), "fold": RP.make_folds(names),
          "clip_idx": PT["clip_idx"].astype(np.int64), "n_clips": len(PT["clips"])}
    C_train = RP.boot_counts(TR["n_clips"], B, O.SEED)
    # evaluation layout (arm independent; checked per arm)
    f0 = N.arm_files(in_dir, arms[0], states_dir)
    el = load_layout(f0["eval_states"])
    elay = make_layout(el["clip_idx"], el["t"], len(el["clips"]))
    lay = {"starts": elay[0], "lens": elay[1], "clip_idx": el["clip_idx"], "t": el["t"]}
    enames = el["clips"][el["clip_idx"]]
    lab_rows = bank_label_rows(enames, el["t"])
    FSe = H.build_fit_sets(enames, el["t"], lab_rows, np.zeros(len(lab_rows)))
    wrow, wclip = window_rows(lay, el["clips"], ctx["eid"], ctx["g0"])
    fold_clip = RP.make_folds(el["clips"])
    EV = {"lay": lay, "lab_rows": lab_rows, "wrow": wrow, "wclip": wclip, "C_clips": RP.boot_counts(len(el["clips"]), B, O.SEED), "fold_clip": fold_clip, "fold_rows": fold_clip[el["clip_idx"]],
          "c_rows": np.asarray(FSe["c"], np.float64), "c_ok_rows": np.asarray(FSe["c_ok"], bool) & np.isfinite(FSe["c"])}
    f2, f8 = f2_f8(ctx, lab_rows, lay, wrow)
    ctrl["f2_sign_units_alignment"] = f2
    res["f8_precision_a_rate_needs"] = f8
    rng = np.random.default_rng(5)
    noise_labels = {"c_lab": rng.normal(size=len(ctx["c_lab"])), "G15": rng.normal(size=ctx["G15"].shape)}
    outs = {}
    for arm in arms:
        files = N.arm_files(in_dir, arm, states_dir)
        recs, ck = N.check_records(files, False, True)
        lay_ok = False
        if recs["pass"]:
            e2 = load_layout(files["eval_states"])
            t2 = load_layout(files["train_states"])
            lay_ok = bool(np.array_equal(e2["clip_idx"], el["clip_idx"]) and np.array_equal(e2["t"], el["t"]) and list(e2["clips"]) == list(el["clips"])
                          and np.array_equal(t2["clip_idx"], PT["clip_idx"]) and np.array_equal(t2["t"], PT["t"]) and list(t2["clips"]) == list(map(str, PT["clips"])))
        ctrl["f6_records|%s" % arm] = {"pass": bool(recs["pass"] and lay_ok), "records": recs, "layouts_equal_to_pack_and_to_each_other": lay_ok}
        if not (recs["pass"] and lay_ok):
            res["arms"][arm] = {"read": False, "files": files, "ckpt_sha256": ck}
            continue
        res["arms"][arm] = process_arm(arm, ctx, TR, EV, files, B, n1_ref, C_train, None, ctrl, noise_labels)
        res["arms"][arm]["ckpt_sha256"] = ck
    for arm in arms:
        pass
    res["controls"] = ctrl
    flat = {k: bool(v["pass"]) for k, v in ctrl.items()}
    res["controls_failed"] = sorted(k for k, v in flat.items() if not v)
    all_read = all(res["arms"].get(a, {}).get("read") for a in arms)
    gate_ok = bool(not res["controls_failed"] and all_read)
    res["controls_pass"] = gate_ok
    floor_measured = None
    if "LSD_s0" in res["arms"] and "LSD_s1" in res["arms"] and res["arms"]["LSD_s0"].get("read") and res["arms"]["LSD_s1"].get("read"):
        floor_measured = abs(res["arms"]["LSD_s0"]["M1"]["r2"] - res["arms"]["LSD_s1"]["M1"]["r2"])
    res["floor_card"] = FLOOR
    res["floor_measured_abs_diff_m1_lsd"] = floor_measured
    res["outcomes"] = name_outcomes_gf(res["arms"], gate_ok)
    if JUMP is not None:
        res["variant"] = "POST-HOC same-lead variant (step above %.1f m restarts or truncates the run); written after f2 of the card run failed; names nothing" % JUMP
        res["outcomes"] = {"named": False, "reason": res["variant"]}
    elif not gate_ok:
        res["outcomes"]["reason"] = "controls failed: %s%s" % (", ".join(res["controls_failed"]) or "none", "" if all_read else "; an arm was not read")
    res["elapsed_s"] = round(time.time() - t0, 1)
    O.jdump(res, os.path.join(out_dir, OUT_JSON))
    write_verdict(os.path.join(out_dir, OUT_MD), res, arms)
    log("controls", "PASS" if gate_ok else "FAILED %s" % res["controls_failed"], "elapsed %.0f s" % res["elapsed_s"])
    return 0 if gate_ok else 2


# ============================================================================================ verdict
def _c(v):
    return "[%+.3f, %+.3f]" % (v[0], v[1])


def write_verdict(path, res, arms):
    L = []
    L.append("# VERDICT_GAPFILTER: card PREREG_V8F_PROOF_ADD28 (the closing speed of the lead from a causal filter over the head's own gap reads)")
    L.append("")
    L.append(STAMP)
    L.append("")
    if res.get("variant"):
        L.append("**%s. Not the card's estimator; every number below is information for a later card.**" % res["variant"])
        L.append("")
    L.append("Counts: %d windows, %d scored, %d scored with a closing-speed label, %d scored episodes; episode-cluster bootstrap B %d, rng 0." % (
        res["counts"]["windows"], res["counts"]["scored"], res["counts"]["valid_with_closing_label"], res["counts"]["episodes_scored"], res["B"]))
    L.append("")
    L.append("## Controls (first)")
    L.append("")
    for k, v in res["controls"].items():
        extra = ""
        if k.startswith("f1"):
            extra = "R2 skill this path %.7f, banked 0.7499068 (n=%d)" % (v["r2_skill_this_path"], v["n"])
        elif k.startswith("f2"):
            extra = "E1 K=5 on the label gap: R2 %+.3f (bar at least %.1f), n=%d of %d" % (v["r2_E1_K5_label_gap"], v["bar"], v["n"], v["n_valid"])
        elif k.startswith("f3"):
            extra = "max abs change %.1f, NaN pattern mismatches %d, %d windows x %d candidates" % (v["max_abs_change"], v["nan_pattern_mismatches"], v["n_windows"], v["n_candidates"])
        elif k.startswith("f4"):
            extra = "M1F with permuted sequences %+.3f (bar at most +%.2f), seeds 0 to 4: %s; identity permutation reproduces F, max abs diff %.2g" % (
                v["m1f_permuted_seed0"], v["bar"], ", ".join("%+.3f" % x for x in v["all_seeds_0_to_4"]), v["identity_permutation_reproduces_F_max_abs_diff"])
        elif k.startswith("f5"):
            extra = "selection with real labels %s, with noise labels %s; identical %s" % (v["chosen_real"], v["chosen_noise_labels"], v["all_scores_and_calibrations_identical"])
        elif k.startswith("f6"):
            extra = "records match, one checkpoint per arm, layouts equal: %s" % v["layouts_equal_to_pack_and_to_each_other"]
        elif k.startswith("x1"):
            extra = "M1 %+.4f vs N1 result %+.4f; DELTA of the head %+.4f vs %+.4f" % (v["m1_this_path"], v["m1_n1_result"], v["delta_this_path"], v["delta_n1_result"])
        L.append("- %s: **%s** %s" % (k, "PASS" if v["pass"] else "FAIL", extra))
    L.append("")
    L.append("Controls failed: %s" % (", ".join(res["controls_failed"]) if res["controls_failed"] else "none"))
    L.append("")
    L.append("## Names (card section 5)")
    L.append("")
    out = res["outcomes"]
    if not out.get("named"):
        L.append("No outcome is named. %s" % out.get("reason", ""))
    else:
        for ty, d in out["per_type"].items():
            if d.get("closing") is None:
                L.append("- %s: %s" % (ty, d["reason"]))
            else:
                L.append("- %s: **%s** (M1F above 0 with the interval above 0 at both seeds: %s; paired difference above 0 with the interval above 0 at both seeds: %s; mean paired difference %+.3f against the floor %.3f: %s)" % (
                    ty, d["closing"], d["m1f_above0_both_seeds"], d["paired_above0_both_seeds"], d["mean_paired_diff"], d["floor"], d["mean_paired_diff_above_floor"]))
                L.append("  - **%s** (DELTA-F above 0 with the interval above 0 at both seeds: %s; permuted control at most +%.2f at both seeds: %s)" % (
                    d["slot"], d["delta_f_above0_both_seeds"], BAR_PERM, d["delta_perm_at_most_bar_both_seeds"]))
    L.append("")
    L.append("Floor: card value %.3f; measured |M1(LSD_s0) - M1(LSD_s1)| = %s." % (res["floor_card"], "%.4f" % res["floor_measured_abs_diff_m1_lsd"] if res["floor_measured_abs_diff_m1_lsd"] is not None else "n/a"))
    L.append("")
    L.append("## Per arm (413 scored windows with a closing-speed label for M1, M1F, paired; 429 scored windows for DELTA)")
    L.append("")
    L.append("| arm | selected F | a, b | M1 head closing R2 | M1F R2 | M1F minus M1 | DELTA-F | DELTA-F permuted (seed 0) | F estimate missing (of 413) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        r = res["arms"].get(a, {})
        if not r.get("read"):
            L.append("| %s | not read | | | | | | | |" % a)
            continue
        s = r["selected"]
        L.append("| %s | %s | %+.3f, %+.3f | %+.3f %s n=%d | %+.3f %s n=%d | %+.3f %s n=%d | %+.3f %s n=%d | %+.3f | %d |" % (
            a, s["estimator"], s["calibration_a_b"][0], s["calibration_a_b"][1], r["M1"]["r2"], _c(r["M1"]["r2_ci95"]), r["M1"]["n"], r["M1F"]["r2"], _c(r["M1F"]["r2_ci95"]), r["M1F"]["n"],
            r["paired"]["diff"], _c(r["paired"]["ci95"]), r["paired"]["n"], r["delta_f"]["skill"], _c(r["delta_f"]["ci95"]), r["delta_f"]["n"], r["delta_perm_seed0"], r["counts"]["F_estimate_missing_valid"]))
    L.append("")
    for a in arms:
        r = res["arms"].get(a, {})
        if not r.get("read"):
            continue
        s = r["selected"]
        L.append("### %s" % a)
        L.append("")
        L.append("- Selection on the train probe clips (%d closing rows, five clip-grouped folds): best E1 %s (mean fold MSE %.4f), best E2 %s (%.4f); chosen %s." % (
            s["n_train_rows"], s["best_E1"], s["cv_mse_best_E1"], s["best_E2"], s["cv_mse_best_E2"], s["estimator"]))
        L.append("- Windows where F is missing and the head closing output is used: %d of %d valid, %d of %d scored; fallback also missing: %d." % (
            r["counts"]["F_estimate_missing_valid"], r["counts"]["windows_valid"], r["counts"]["F_estimate_missing_scored"], r["counts"]["windows_scored"], r["counts"]["F_and_fallback_missing_valid"]))
        L.append("- Reported beside, names nothing: DELTA of the head's own closing output (same call) %+.3f %s; DELTA-F permuted seeds 0 to 4: %s." % (
            r["delta_head_own"]["skill"], _c(r["delta_head_own"]["ci95"]), ", ".join("%+.3f" % x for x in r["delta_perm_all_seeds_0_to_4"])))
        t = r["train"]
        L.append("- M1F on the train probe clips (names nothing; %d closing rows, %d clips): in sample %+.3f %s; cross-fitted five-fold %+.3f %s; head closing output %+.3f %s." % (
            t["rows_with_closing_label"], 1200, t["M1F_in_sample"]["r2"], _c(t["M1F_in_sample"]["r2_ci95"]), t["M1F_cross_fitted_5fold"]["r2"], _c(t["M1F_cross_fitted_5fold"]["r2_ci95"]),
            t["M1_head_closing"]["r2"], _c(t["M1_head_closing"]["r2_ci95"])))
        e3, cf = r["E3"], r["crossfit_eval_bank"]
        L.append("- E3 blend (names nothing): R2 %+.3f %s n=%d. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 %+.3f %s n=%d, chosen per fold %s." % (
            e3["r2"], _c(e3["r2_ci95"]), e3["n"], cf["r2"], _c(cf["r2_ci95"]), cf["n"], cf["chosen_per_fold"]))
        f7 = r["f7"]["all_rows"]
        f7p = r["f7"]["rows_with_head_presence_at_least_half"]
        L.append("- f7 (names nothing), head gap error on evaluation rows with a gap label (n=%d rows, %d consecutive pairs): RMSE %.2f m %s; RMSE of the change between consecutive rows %.2f m %s; lag-1 autocorrelation %+.3f %s. Rows with head presence at least 0.5 (n=%d): RMSE %.2f m, change %.2f m, lag-1 %+.3f." % (
            f7["n_rows"], f7["n_pairs"], f7["rmse_m"], _c(f7["rmse_m_ci95"]), f7["rmse_change_m"], _c(f7["rmse_change_m_ci95"]), f7["lag1_autocorr"], _c(f7["lag1_autocorr_ci95"]),
            f7p["n_rows"], f7p["rmse_m"], f7p["rmse_change_m"], f7p["lag1_autocorr"]))
        L.append("")
        L.append("M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):")
        L.append("")
        L.append("| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |")
        L.append("|---|---|---|---|---|---|")
        for k, v in r["by_candidate"].items():
            L.append("| %s | %+.3f | %s | %d | %d | %.4f |" % (k, v["r2"], _c(v["r2_ci95"]), v["n"], v["estimate_missing_valid"], v["cv_mse_train"]))
        L.append("")
    L.append("## f8: the precision a rate needs (label gap plus white noise through E1; closing-speed R2 on the 413 windows, mean over %d noise draws, range in brackets)" % F8_DRAWS)
    L.append("")
    L.append("| noise sd (m) | K=5 | K=10 | K=15 |")
    L.append("|---|---|---|---|")
    for sg in SIGMAS:
        cells = []
        for K in F8_KS:
            v = res["f8_precision_a_rate_needs"]["sigma%g|K%d" % (sg, K)]
            cells.append("%+.3f [%+.3f, %+.3f]" % (v["r2_mean"], v["r2_min"], v["r2_max"]))
        L.append("| %g | %s |" % (sg, " | ".join(cells)))
    L.append("")
    L.append("## Limits")
    L.append("")
    L.append("Two seeds per arm type; the bootstrap answers the question about episodes, not about training runs; the floor is from the two detached arms. The head saw the train probe clips during training, so its gap reads are cleaner there than on the evaluation episodes. 413 scored windows of 48 episodes; one corpus. A rate from a longer baseline lags a changing closing speed; the label is the closing speed at S0.")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--in-dir", default=READ_DIR + "/in")
    r.add_argument("--states-dir", default="C:/Users/Admin/v8proof_c/s1states")
    r.add_argument("--out-dir", default=ROOT + "/gapfilter/out")
    r.add_argument("--pack-dir", default=READ_DIR)
    r.add_argument("--B", type=int, default=2000)
    r.add_argument("--variant-jump-m", type=float, default=None, help="post-hoc same-lead variant; omit for the card")
    a = ap.parse_args(argv)
    global JUMP
    JUMP = a.variant_jump_m
    return run(a.arms.split(","), a.in_dir, a.states_dir, a.out_dir, a.B, a.pack_dir)


if __name__ == "__main__":
    sys.exit(main())
