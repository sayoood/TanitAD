"""RESCURVE: PREREG_V8F_PROOF_ADD7 (frozen 2026-11-13 01:38). Tier T0: a frozen-feature probe on logged frames, a perception diagnostic,
NOT driving performance. CPU only, <= 4 threads. Zero training. Input: the RESPROBE cache (res/cache, read-only).
    python rescurve.py run [--threads 4] [--smoke N]
Order: controls c1 (both grids, before any read; exit 2 if it fails) and c2, then read A (rank), read B (data), read C (selection, with c3).
Definitions and numbered deviations: CURVE_DEFS.md. Reuses resprobe.py (kpca_fit, ridge_solve, inner_split, make_folds, e_block, bootstrap helpers).
"""
import sys
sys.dont_write_bytecode = True
import argparse, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "F:/TanitAD-scratch/v8proof/res"
sys.path.insert(0, RES)
import resprobe as R  # noqa: E402  (read-only reuse)

CACHE = f"{RES}/cache"
SEED = R.SEED
N_FOLDS = R.N_FOLDS
TARGETS = R.TARGETS
RANKS = {"r16": 16, "r64": 64, "r256": 256, "r1024": 1024, "full": "full"}
RANK_ORDER = ["r16", "r64", "r256", "r1024", "full"]
FRACS = (0.125, 0.25, 0.5)
BUDGETS = (0.25, 0.5)
GRIDS = ("V40", "V160")
CFG = {"B_BOOT": 2000, "N_DRAWS_B": 5, "N_DRAWS_C": 20}
BAR = 0.03
LEV_LAMBDA_REL = 1e-2
LAT = "tier T0, a perception diagnostic on logged frames, never driving performance"


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- pure numpy pieces (unit-tested)
def _centre(K):
    cm = K.mean(0); gm = cm.mean()
    return K - cm[:, None] - cm[None, :] + gm, cm, gm


def _centre_x(Kx, cm, gm):
    return Kx - Kx.mean(1)[:, None] - cm[None, :] + gm


def _solve_pd(Kc, reg, b):
    """(Kc + reg I)^-1 b, Cholesky with an LU fallback."""
    A = Kc.copy()
    A[np.diag_indices_from(A)] += reg
    try:
        from scipy.linalg import cho_factor, cho_solve
        c = cho_factor(A, lower=True, overwrite_a=True, check_finite=False)
        return cho_solve(c, b, check_finite=False)
    except Exception:
        A = Kc.copy(); A[np.diag_indices_from(A)] += reg
        return np.linalg.solve(A, b)


def krr_fit(Kt, y, ep_t, d=None, alphas=R.ALPHAS):
    """Full-kernel ridge with intercept, the dual form of R.fit_p1 with no rank cut. Kt: uncentred train kernel [n, n]; y [n]; ep_t [n].
    Ridge strength a * trace(Kc) / d (d = number of features; default n), alpha by the same inner episode split as R.fit_p1.
    Returns (pred(Kx[m, n]) -> [m], best_inner_val_mse, inner_val_var)."""
    n = len(y)
    d = n if d is None else d

    def fit_on(idx, a_list):
        Kc, cm, gm = _centre(Kt[np.ix_(idx, idx)] if len(idx) < n else Kt)
        ym = y[idx].mean(); yc = y[idx] - ym
        sc = np.trace(Kc) / d
        outs = [(a, _solve_pd(Kc, a * sc, yc)) for a in a_list]
        return outs, cm, gm, ym

    if len(set(map(str, ep_t))) < 2:
        outs, cm, gm, ym = fit_on(np.arange(n), [1e-2])
        dual = outs[0][1]
        return (lambda Kx: _centre_x(Kx, cm, gm) @ dual + ym), float("nan"), float("nan")
    iv = R.inner_split(ep_t)
    itr, iva = np.flatnonzero(~iv), np.flatnonzero(iv)
    outs, cm, gm, ym = fit_on(itr, list(alphas))
    Kxv = _centre_x(Kt[np.ix_(iva, itr)], cm, gm)
    mses = [float(((y[iva] - (Kxv @ du + ym)) ** 2).mean()) for _, du in outs]
    k = int(np.argmin(mses))
    del outs
    full, cm, gm, ym = fit_on(np.arange(n), [alphas[k]])
    dual = full[0][1]
    return (lambda Kx: _centre_x(Kx, cm, gm) @ dual + ym), mses[k], float(y[iva].var())


def fit_p1x(Ztr, ytr, ep_tr):
    """R.fit_p1 with the best inner-validation MSE and the validation variance returned as well (same arithmetic, same alpha choice)."""
    if len(set(map(str, ep_tr))) < 2:
        a, w, b = R.ridge_solve(Ztr, ytr, [1e-2])[0]
        return (lambda Z: Z @ w + b), float("nan"), float("nan")
    iv = R.inner_split(ep_tr)
    sols = R.ridge_solve(Ztr[~iv], ytr[~iv], R.ALPHAS)
    mses = [float(((ytr[iv] - (Ztr[iv] @ s[1] + s[2])) ** 2).mean()) for s in sols]
    k = int(np.argmin(mses))
    a, w, b = R.ridge_solve(Ztr, ytr, [sols[k][0]])[0]
    return (lambda Z: Z @ w + b), mses[k], float(ytr[iv].var())


def nested_subsets(train_eps, fracs, seed):
    """Nested episode subsets of the training episodes: one permutation, every fraction is a prefix (so smaller sets lie inside larger ones)."""
    train_eps = np.asarray(sorted(train_eps))
    order = train_eps[np.random.default_rng(seed).permutation(len(train_eps))]
    return {f: np.sort(order[:max(1, int(round(f * len(order))))]) for f in fracs}


def select_episodes(rule, Z, ep_tr, budget, seed=0):
    """Label-free choice of ``budget`` WHOLE episodes. Z [n_tr, d] features of the training frames (no labels enter), ep_tr [n_tr] episode ids.
    random: seeded permutation. kcentre: greedy farthest-first on episode-mean features (start: the episode nearest the mean of the episode means).
    leverage: episodes ranked by the SUM over their frames of the ridge leverage z (Z^T Z + lam I)^-1 z, lam = LEV_LAMBDA_REL * trace(Z^T Z)/d."""
    ue = np.asarray(sorted(set(ep_tr.tolist())))
    k = int(min(budget, len(ue)))
    if rule == "random":
        return np.sort(np.random.default_rng(seed).permutation(ue)[:k])
    pos = np.searchsorted(ue, ep_tr)
    if rule == "kcentre":
        cnt = np.bincount(pos, minlength=len(ue)).astype(np.float64)
        M = np.zeros((len(ue), Z.shape[1]))
        for j in range(Z.shape[1]):
            M[:, j] = np.bincount(pos, weights=Z[:, j], minlength=len(ue)) / cnt
        first = int(np.argmin(((M - M.mean(0)) ** 2).sum(1)))
        chosen = [first]
        mind = ((M - M[first]) ** 2).sum(1)
        for _ in range(k - 1):
            j = int(np.argmax(mind)); chosen.append(j)
            mind = np.minimum(mind, ((M - M[j]) ** 2).sum(1))
        return np.sort(ue[np.array(chosen)])
    if rule == "leverage":
        Zc = Z - Z.mean(0)
        A = Zc.T @ Zc
        lam = LEV_LAMBDA_REL * np.trace(A) / A.shape[0]
        lev = ((Zc @ np.linalg.inv(A + lam * np.eye(A.shape[0]))) * Zc).sum(1)
        tot = np.bincount(pos, weights=lev, minlength=len(ue))
        return np.sort(ue[np.argsort(-tot, kind="stable")[:k]])
    raise ValueError(rule)


def metric_of(kind, y, p):
    return R.auc_pooled(y, p) if kind == "auc" else R.r2_pooled(y, p)


def boot_metric(kind, epi, y, p, Cb, n_ep):
    """(bootstrap vector over episode-cluster draws Cb, point value); AUROC in chunks to bound memory."""
    if kind == "auc":
        v = np.concatenate([R.boot_auc(epi, y, p, Cb[i:i + 250]) for i in range(0, len(Cb), 250)])
        return v, R.auc_pooled(y, p)
    return R.boot_r2(R.ep_stats_r2(epi, y, p, n_ep), Cb), R.r2_pooled(y, p)


def last_doubling_gain(kind, epi, y, p_full, p_halves, Cb, n_ep):
    """Gain of the last doubling (1/2 -> 1), paired: for each draw the full-fraction predictions minus that draw's half-fraction predictions
    on the SAME held-out frames and the SAME episode-cluster resamples; averaged over the draws. Returns dict(diff, ci95)."""
    bf, pf = boot_metric(kind, epi, y, p_full, Cb, n_ep)
    bh, ph = zip(*[boot_metric(kind, epi, y, p, Cb, n_ep) for p in p_halves])
    d = bf - np.mean(bh, axis=0)
    return {"diff": float(pf - np.mean(ph)), "ci95": R.ci(d)}


# ----------------------------------------------------------------------------- fold context
def build_scores(Kg, tr, te, rank):
    r = int(min(rank, len(tr) - 30))
    proj, _ = R.kpca_fit(Kg, tr, rank=r)
    return proj(tr), proj(te)


class Ctx:
    """One fit set: training frames ``tr`` (global indices), held-out ``te``, kernel PCA scores of both, E-block scale ``sv``."""

    def __init__(s, Kg, E, tr, te, Str, Ste, sv):
        s.Kg, s.E, s.tr, s.te, s.Str, s.Ste, s.sv = Kg, E, tr, te, Str, Ste, sv
        s._k = {}

    def _kern(s, with_e):
        if with_e not in s._k:
            Ktr = np.asarray(s.Kg[np.ix_(s.tr, s.tr)], np.float64)
            Kte = np.asarray(s.Kg[np.ix_(s.te, s.tr)], np.float64)
            if with_e:
                Eb = R.e_block(s.E, s.tr, s.sv)
                Ktr += Eb[s.tr] @ Eb[s.tr].T
                Kte += Eb[s.te] @ Eb[s.tr].T
            s._k = {with_e: (Ktr, Kte)}
        return s._k[with_e]

    def predict(s, y, valid, epi, rank, with_e):
        """Held-out predictions [len(te)] (NaN where the target is invalid), inner-val MSE, inner-val variance."""
        tr, te = s.tr, s.te
        mtr, mte = valid[tr], valid[te]
        out = np.full(len(te), np.nan)
        if mtr.sum() < 5 or not mte.any():
            if mtr.any():
                out[mte] = y[tr][mtr].mean()
            return out, float("nan"), float("nan")
        ytr, eptr = y[tr][mtr], epi[tr][mtr]
        if rank == "full":
            Ktr, Kte = s._kern(with_e)
            itr, ite = np.flatnonzero(mtr), np.flatnonzero(mte)
            Kt = Ktr if len(itr) == len(tr) else Ktr[np.ix_(itr, itr)]
            pred, vm, vv = krr_fit(Kt, ytr, eptr, d=len(itr))
            out[ite] = pred(Kte[np.ix_(ite, itr)])
            return out, vm, vv
        Ftr, Fte = s.Str[:, :rank], s.Ste[:, :rank]
        if with_e:
            Eb = R.e_block(s.E, tr, s.sv)
            Ftr, Fte = np.concatenate([Ftr, Eb[tr]], 1), np.concatenate([Fte, Eb[te]], 1)
        pred, vm, vv = fit_p1x(Ftr[mtr], ytr, eptr)
        out[mte] = pred(Fte[mte])
        return out, vm, vv


def e_only(E, y, valid, epi, tr, te):
    mtr, mte = valid[tr], valid[te]
    out = np.full(len(te), np.nan)
    if mtr.sum() < 5 or not mte.any():
        if mtr.any():
            out[mte] = y[tr][mtr].mean()
        return out
    mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-9
    pred, _, _ = fit_p1x(((E - mu) / sd)[tr][mtr], y[tr][mtr], epi[tr][mtr])
    out[mte] = pred(((E - mu) / sd)[te][mte])
    return out


# ----------------------------------------------------------------------------- the run
class State:
    pass


def load_data(smoke):
    Z = np.load(f"{CACHE}/frames.npz")
    keep = np.isfinite(Z["E"]).all(1)
    idx_keep = np.flatnonzero(keep)
    beid = Z["beid"].astype(str)[keep]
    if smoke:
        rng = np.random.default_rng(3)
        ue = sorted(set(beid)); tot = 0; chosen = set()
        for e in rng.permutation(ue):
            n_e = int((beid == e).sum())
            if tot + n_e <= smoke:
                chosen.add(e); tot += n_e
        sub = np.flatnonzero(np.isin(beid, sorted(chosen)))
        idx_keep, beid = idx_keep[sub], beid[sub]
        E = Z["E"][keep][sub]
        Y = {t: Z[v][keep][sub].astype(np.float64) for t, (v, _) in TARGETS.items()}
    else:
        E = Z["E"][keep]
        Y = {t: Z[v][keep].astype(np.float64) for t, (v, _) in TARGETS.items()}
    uep = sorted(set(beid)); n_ep = len(uep)
    epi = np.searchsorted(np.array(uep), beid)
    return idx_keep, beid, E, Y, epi, n_ep


def load_K(g, idx_keep):
    name = {"V40": "K40", "V160": "K160"}[g]
    v = np.load(f"{CACHE}/{name}.npy", mmap_mode="r")
    return np.asarray(v[np.ix_(idx_keep, idx_keep)])


def cmd_run(a):
    R.set_threads(a.threads)
    outdir = os.path.join(HERE, "smoke") if a.smoke else HERE
    os.makedirs(outdir, exist_ok=True)
    if a.smoke:
        CFG.update(B_BOOT=300, N_DRAWS_B=2, N_DRAWS_C=4)
    t0 = time.time()
    S = State()
    S.idx_keep, S.beid, S.E, S.Y, S.epi, S.n_ep = load_data(a.smoke)
    N = len(S.beid)
    S.folds = R.make_folds(S.beid)
    S.Cb = R.boot_counts(S.n_ep, CFG["B_BOOT"])
    S.valid = {t: np.isfinite(S.Y[t]) for t in TARGETS}
    S.kind = {t: TARGETS[t][1] for t in TARGETS}
    S.tr = [np.flatnonzero(S.folds != f) for f in range(N_FOLDS)]
    S.te = [np.flatnonzero(S.folds == f) for f in range(N_FOLDS)]
    S.OOF, S.INFO, S.SC = {}, {}, {}
    res = {"meta": {"n_frames": int(N), "n_episodes": int(S.n_ep), "folds": N_FOLDS, "B": CFG["B_BOOT"], "tier": LAT, "smoke": int(a.smoke),
                    "STRATEGIC": "PAUSED (PI decision queue item 25, 2026-09-19): not applicable, not a pass, not omitted",
                    "estimator": "pooled out-of-fold R2 (AUROC for LEADP); episode-cluster bootstrap over the held-out episodes, paired draws; closed-form probes, no training seed",
                    **{f"n_{t}": int(S.valid[t].sum()) for t in TARGETS}, **{f"n_ep_{t}": int(len(set(S.epi[S.valid[t]]))) for t in TARGETS}},
           "controls": {}, "A": {}, "B": {}, "C": {}}

    def checkpoint(tag):
        res["meta"]["elapsed_s"] = round(time.time() - t0, 1)
        json.dump(res, open(os.path.join(outdir, "RESULT_RESCURVE.partial.json"), "w", encoding="utf-8"), indent=1, default=float)
        np.savez_compressed(os.path.join(outdir, "OOF_RESCURVE.partial.npz"), **{k: v.astype(np.float32) for k, v in S.OOF.items()})
        log(f"[run] checkpoint {tag} {time.time() - t0:.0f}s")

    # ---------------- phase 0: c1 (and c2) on rank 256, both grids, before any read
    banked = json.load(open(f"{RES}/RESULT_RESPROBE.json", encoding="utf-8"))["metric"]
    c1, c2 = {}, {}
    rng = np.random.default_rng(SEED + 99)
    ysh = {}
    for t in TARGETS:
        y = S.Y[t].copy()
        for f in range(N_FOLDS):
            te = np.flatnonzero((S.folds == f) & S.valid[t]); y[te] = S.Y[t][rng.permutation(te)]
        ysh[t] = y

    def oof_of(ctxs, y, t, rank, with_e, key=None):
        oof = np.full(N, np.nan)
        for f in range(N_FOLDS):
            out, vm, vv = ctxs[f].predict(y, S.valid[t], S.epi, rank, with_e)
            oof[S.te[f]] = out
            if key is not None:
                S.INFO.setdefault(key, []).append(vm / vv if vv and np.isfinite(vv) and vv > 0 else float("nan"))
        return oof

    def c2_row(t, oofk):
        if S.kind[t] == "auc":
            pf = [R.auc_pooled(np.where(S.folds == f, ysh[t], np.nan), oofk) for f in range(N_FOLDS)]
            v = float(np.nanmean(pf)); return {"value": v, "bar": "within 0.03 of 0.5", "pass": bool(abs(v - 0.5) <= 0.03), "kind": "auroc_perfold_mean"}
        v = R.r2_pooled(ysh[t], oofk)
        return {"value": v, "bar": "<= 0.02", "pass": bool(v <= 0.02), "kind": "r2_pooled"}

    for t in TARGETS:                                    # E floor (grid independent), P1
        oof = np.full(N, np.nan); oofk = np.full(N, np.nan)
        for f in range(N_FOLDS):
            oof[S.te[f]] = e_only(S.E, S.Y[t], S.valid[t], S.epi, S.tr[f], S.te[f])
            oofk[S.te[f]] = e_only(S.E, ysh[t], S.valid[t], S.epi, S.tr[f], S.te[f])
        S.OOF[f"E|{t}"] = oof
        c1[f"{t}|E|P1"] = {"value": metric_of(S.kind[t], S.Y[t], oof), "banked": banked[f"{t}|E|P1"]}
        c2[f"{t}|E"] = c2_row(t, oofk)
    for g in GRIDS:
        Kg = load_K(g, S.idx_keep)
        S.SC[g] = []
        for f in range(N_FOLDS):
            Str, Ste = build_scores(Kg, S.tr[f], S.te[f], 256)
            S.SC[g].append((Str, Ste, float(Str.var(0).mean())))
            log(f"[c1] kpca {g} fold {f} {time.time() - t0:.0f}s")
        ctxs = [Ctx(Kg, S.E, S.tr[f], S.te[f], *S.SC[g][f]) for f in range(N_FOLDS)]
        for t in TARGETS:
            for setn, we in ((g, False), (g + "+E", True)):
                oof = oof_of(ctxs, S.Y[t], t, 256, we, key=f"info|{g}|{t}|r256" if we else None)
                S.OOF[f"A|{g}|{t}|r256|{'V+E' if we else 'V'}"] = oof
                c1[f"{t}|{setn}|P1"] = {"value": metric_of(S.kind[t], S.Y[t], oof), "banked": banked[f"{t}|{setn}|P1"]}
                c2[f"{t}|{setn}"] = c2_row(t, oof_of(ctxs, ysh[t], t, 256, we))
        del Kg, ctxs
    for k, v in c1.items():
        v["abs_diff"] = abs(v["value"] - v["banked"]); v["pass"] = bool(v["abs_diff"] <= 1e-6)
    res["controls"]["c1"] = {"bar": "banked RESPROBE P1 values reproduced to 1e-6 at fraction 1, rank 256", "rows": c1,
                             "pass": all(v["pass"] for v in c1.values()), "max_abs_diff": max(v["abs_diff"] for v in c1.values()),
                             "checked": not a.smoke, "note": "banked values exist for the full set only; the subsample smoke cannot check c1" if a.smoke else ""}
    res["controls"]["c2"] = {"rows": c2, "pass": all(v["pass"] for v in c2.values()), "scope": "rank 256, sets E, V40, V160, V40+E, V160+E (full-kernel shuffle added in read A)"}
    log(f"[c1] pass {res['controls']['c1']['pass']} max diff {res['controls']['c1']['max_abs_diff']:.2e}; c2 pass {res['controls']['c2']['pass']}")
    checkpoint("c1c2")
    if not a.smoke and not res["controls"]["c1"]["pass"]:
        res["outcome"] = {"STOPPED": "c1 failed; no read was made"}
        write_outputs(res, outdir, S, stopped=True)
        sys.exit(2)

    read_A(S, res, t0, checkpoint)
    read_B(S, res, t0, checkpoint)
    read_C(S, res, t0, checkpoint)
    res["meta"]["runtime_s"] = round(time.time() - t0, 1)
    write_outputs(res, outdir, S, stopped=False)
    allc = res["controls"]
    ok = allc["c1"]["pass"] or a.smoke
    ok = ok and allc["c2"]["pass"] and allc["c2_full"]["pass"] and allc["c3"]["pass"]
    log("[run] done; controls ok" if ok else "[run] done; A CONTROL MISSED ITS BAR (see VERDICT)")
    sys.exit(0 if (ok or a.smoke) else 2)


def read_A(S, res, t0, checkpoint):
    N = len(S.beid)
    for g in GRIDS:
        Kg = load_K(g, S.idx_keep)
        for f in range(N_FOLDS):
            Str, Ste, sv = S.SC[g][f]
            c256 = Ctx(Kg, S.E, S.tr[f], S.te[f], Str, Ste, sv)
            c1024 = Ctx(Kg, S.E, S.tr[f], S.te[f], *build_scores(Kg, S.tr[f], S.te[f], 1024)[:2], sv)
            log(f"[A] {g} fold {f} kpca1024 {time.time() - t0:.0f}s")
            for t in TARGETS:
                for rk in ("r16", "r64", "r1024", "full"):
                    ctx = c1024 if rk == "r1024" else c256
                    out, vm, vv = ctx.predict(S.Y[t], S.valid[t], S.epi, RANKS[rk], True)
                    key = f"A|{g}|{t}|{rk}|V+E"
                    S.OOF.setdefault(key, np.full(N, np.nan))[S.te[f]] = out
                    S.INFO.setdefault(f"info|{g}|{t}|{rk}", []).append(vm / vv if vv and np.isfinite(vv) and vv > 0 else float("nan"))
            del c1024
            log(f"[A] {g} fold {f} done {time.time() - t0:.0f}s")
        # c2 on the full kernel, shuffled targets, V+E, one pass per grid
        c2f = res["controls"].setdefault("c2_full", {"rows": {}, "pass": True, "scope": "full kernel, V+E sets, targets shuffled within fold"})
        rng = np.random.default_rng(SEED + 123)
        for t in TARGETS:
            y = S.Y[t].copy()
            for f in range(N_FOLDS):
                te = np.flatnonzero((S.folds == f) & S.valid[t]); y[te] = S.Y[t][rng.permutation(te)]
            oofk = np.full(N, np.nan)
            for f in range(N_FOLDS):
                Str, Ste, sv = S.SC[g][f]
                oofk[S.te[f]] = Ctx(Kg, S.E, S.tr[f], S.te[f], Str, Ste, sv).predict(y, S.valid[t], S.epi, "full", True)[0]
            if S.kind[t] == "auc":
                pf = [R.auc_pooled(np.where(S.folds == f, y, np.nan), oofk) for f in range(N_FOLDS)]
                v = float(np.nanmean(pf)); row = {"value": v, "bar": "within 0.03 of 0.5", "pass": bool(abs(v - 0.5) <= 0.03)}
            else:
                v = R.r2_pooled(y, oofk); row = {"value": v, "bar": "<= 0.02", "pass": bool(v <= 0.02)}
            c2f["rows"][f"{t}|{g}+E"] = row
            c2f["pass"] = bool(c2f["pass"] and row["pass"])
        del Kg
        log(f"[A] {g} done {time.time() - t0:.0f}s")
    # preferred rank per grid and target: smallest mean normalised inner-validation MSE over the folds
    pref = {}
    for g in GRIDS:
        for t in TARGETS:
            sc = {rk: float(np.nanmean(S.INFO[f"info|{g}|{t}|{rk}"])) for rk in RANK_ORDER}
            pref[f"{g}|{t}"] = {"rank": min(sc, key=lambda k: (np.nan_to_num(sc[k], nan=1e9), RANK_ORDER.index(k))), "inner_val_mse_over_var": sc}
    S.pref = {k: v["rank"] for k, v in pref.items()}
    # metrics + paired bootstrap
    A = {"pref_rank": pref, "table": {}, "diff_V160E_minus_V40E": {}, "outcome_per_target": {}}
    for t in TARGETS:
        kind = S.kind[t]; y = S.Y[t]
        for rk in RANK_ORDER:
            bs = {g: boot_metric(kind, S.epi, y, S.OOF[f"A|{g}|{t}|{rk}|V+E"], S.Cb, S.n_ep) for g in GRIDS}
            for g in GRIDS:
                A["table"][f"{t}|{g}+E|{rk}"] = bs[g][1]
            d = bs["V160"][0] - bs["V40"][0]
            A["diff_V160E_minus_V40E"][f"{t}|{rk}"] = {"diff": float(bs["V160"][1] - bs["V40"][1]), "ci95": R.ci(d)}
        A["table"][f"{t}|E|floor_P1"] = metric_of(kind, y, S.OOF[f"E|{t}"])
        hides = [rk for rk in ("r1024", "full") if A["diff_V160E_minus_V40E"][f"{t}|{rk}"]["diff"] >= BAR and A["diff_V160E_minus_V40E"][f"{t}|{rk}"]["ci95"][0] > 0]
        A["outcome_per_target"][t] = {"label": "RANK-HIDES" if hides else "RANK-FLAT", "ranks_meeting_bar": hides}
    A["outcome"] = A["outcome_per_target"]["GAP"]["label"]
    res["A"] = A
    res["controls"]["c2_full"]["pass"] = bool(res["controls"]["c2_full"]["pass"])
    checkpoint("A")


def read_B(S, res, t0, checkpoint):
    N = len(S.beid)
    B = {"fracs": list(FRACS) + [1.0], "n_draws": CFG["N_DRAWS_B"], "curve": {}, "gain_last_doubling": {}, "outcome": {}}
    Bm = {}          # (g, t, rkind, frac) -> list of pooled metrics over draws ; rkind in r256, pref, E
    Boof = {}
    ntr = {}
    for g in GRIDS:
        Kg = load_K(g, S.idx_keep)
        extra = sorted({S.pref[f"{g}|{t}"] for t in TARGETS} - {"r256"}, key=RANK_ORDER.index)
        for d in range(CFG["N_DRAWS_B"]):
            for fr in FRACS:
                oofs = {(t, rk): np.full(N, np.nan) for t in TARGETS for rk in ("r256", "pref", "E")}
                for f in range(N_FOLDS):
                    trall = S.tr[f]
                    eps = np.unique(S.epi[trall])
                    sel = nested_subsets(eps, FRACS, SEED + 1000 + 10 * f + d)[fr]
                    tr = trall[np.isin(S.epi[trall], sel)]
                    ntr.setdefault((g, fr), []).append(len(sel))
                    for t in TARGETS:
                        ntr.setdefault((g, fr, t), []).append(len(set(S.epi[tr][S.valid[t][tr]])))
                    Str, Ste = build_scores(Kg, tr, S.te[f], 256)
                    sv = float(Str.var(0).mean())
                    ctx = Ctx(Kg, S.E, tr, S.te[f], Str, Ste, sv)
                    ctxs = {"r256": ctx}
                    for rk in extra:
                        if rk != "full":
                            ctxs[rk] = Ctx(Kg, S.E, tr, S.te[f], *build_scores(Kg, tr, S.te[f], RANKS[rk])[:2], sv)
                        else:
                            ctxs[rk] = ctx
                    for t in TARGETS:
                        oofs[(t, "r256")][S.te[f]] = ctx.predict(S.Y[t], S.valid[t], S.epi, 256, True)[0]
                        pk = S.pref[f"{g}|{t}"]
                        oofs[(t, "pref")][S.te[f]] = oofs[(t, "r256")][S.te[f]] if pk == "r256" else ctxs[pk].predict(S.Y[t], S.valid[t], S.epi, RANKS[pk], True)[0]
                        oofs[(t, "E")][S.te[f]] = e_only(S.E, S.Y[t], S.valid[t], S.epi, tr, S.te[f])
                    del ctxs, ctx
                for (t, rk), oof in oofs.items():
                    Boof[(g, t, rk, fr, d)] = oof
                    Bm.setdefault((g, t, rk, fr), []).append(metric_of(S.kind[t], S.Y[t], oof))
                log(f"[B] {g} draw {d} frac {fr} {time.time() - t0:.0f}s")
        del Kg
        for t in TARGETS:                       # fraction 1 = the A / phase-0 predictions
            pk = S.pref[f"{g}|{t}"]
            for rk, key in (("r256", f"A|{g}|{t}|r256|V+E"), ("pref", f"A|{g}|{t}|{pk}|V+E"), ("E", f"E|{t}")):
                Boof[(g, t, rk, 1.0, 0)] = S.OOF[key]
                Bm[(g, t, rk, 1.0)] = [metric_of(S.kind[t], S.Y[t], S.OOF[key])]
        checkpoint(f"B-{g}")
    for k, v in Boof.items():
        S.OOF["B|" + "|".join(map(str, k))] = v
    for (g, t, rk, fr), v in Bm.items():
        n_e = float(np.mean(ntr[(g, fr)])) if fr < 1 else float(np.mean([len(np.unique(S.epi[S.tr[f]])) for f in range(N_FOLDS)]))
        n_v = float(np.mean(ntr[(g, fr, t)])) if fr < 1 else float(np.mean([len(set(S.epi[S.tr[f]][S.valid[t][S.tr[f]]])) for f in range(N_FOLDS)]))
        B["curve"][f"{t}|{g}+E|{rk}|{fr}"] = {"mean": float(np.mean(v)), "sd_over_draws": float(np.std(v)) if len(v) > 1 else 0.0, "n_train_episodes": n_e,
                                               "n_train_episodes_with_valid_target": n_v}
    for g in GRIDS:
        for t in TARGETS:
            for rk in ("r256", "pref", "E"):
                kind = S.kind[t]
                halves = [Boof[(g, t, rk, 0.5, d)] for d in range(CFG["N_DRAWS_B"])]
                gain = last_doubling_gain(kind, S.epi, S.Y[t], Boof[(g, t, rk, 1.0, 0)], halves, S.Cb, S.n_ep)
                lab = "DATA-LIMITED" if (gain["diff"] >= BAR and gain["ci95"][0] > 0) else ("SATURATED" if gain["ci95"][1] < BAR else "UNDECIDED")
                rkname = rk if rk != "pref" else f"pref={S.pref[f'{g}|{t}']}"
                B["gain_last_doubling"][f"{t}|{g}+E|{rk}"] = {**gain, "label": lab, "rank_used": rkname if rk != "E" else "E floor, informational"}
    res["B"] = B
    checkpoint("B")


def read_C(S, res, t0, checkpoint):
    N = len(S.beid)
    nd = CFG["N_DRAWS_C"]
    C = {"budgets": list(BUDGETS), "n_random_draws": nd, "rows": {}, "outcome_per_set": {}, "outcome": {}}
    c3 = {"identity": {}, "extra_draw_inside_range": {}, "pass": True}
    arms = [("random", d) for d in range(nd)] + [("kcentre", 0), ("leverage", 0), ("random_c3_redo", 0), ("random_extra", nd)]
    M = {}
    for g in GRIDS:
        Kg = load_K(g, S.idx_keep)
        for b in BUDGETS:
            oofs = {(arm, t): np.full(N, np.nan) for arm in arms for t in TARGETS}
            nsel = {}
            for f in range(N_FOLDS):
                Str256, _, _ = S.SC[g][f]
                eps_tr = S.epi[S.tr[f]]
                k = max(1, int(round(b * len(np.unique(eps_tr)))))
                for arm in arms:
                    name, d = arm
                    if name in ("kcentre", "leverage"):
                        sel = select_episodes(name, Str256, eps_tr, k)
                    else:
                        sd = SEED + 5000 + 100 * f + (0 if name == "random_c3_redo" else d)
                        sel = select_episodes("random", Str256, eps_tr, k, seed=sd)
                    tr = S.tr[f][np.isin(eps_tr, sel)]
                    nsel.setdefault(arm, []).append((len(sel), len(tr)))
                    Str, Ste = build_scores(Kg, tr, S.te[f], 256)
                    ctx = Ctx(Kg, S.E, tr, S.te[f], Str, Ste, float(Str.var(0).mean()))
                    for t in TARGETS:
                        oofs[(arm, t)][S.te[f]] = ctx.predict(S.Y[t], S.valid[t], S.epi, 256, True)[0]
                log(f"[C] {g} budget {b} fold {f} {time.time() - t0:.0f}s")
            for (arm, t), oof in oofs.items():
                M[(g, b, t, arm)] = metric_of(S.kind[t], S.Y[t], oof)
                if arm[0] in ("kcentre", "leverage") or arm == ("random", 0):
                    S.OOF[f"C|{g}|{b}|{t}|{arm[0]}{arm[1]}"] = oof
            C["rows"][f"{g}|{b}|mean_selected_episodes_frames"] = {a[0] + str(a[1]): [float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))] for a, v in nsel.items()}
        del Kg
        checkpoint(f"C-{g}")
    for g in GRIDS:
        for t in TARGETS:
            for b in BUDGETS:
                rnd = np.array([M[(g, b, t, ("random", d))] for d in range(nd)])
                C["rows"][f"{t}|{g}+E|{b}"] = {"random_min": float(rnd.min()), "random_max": float(rnd.max()), "random_mean": float(rnd.mean()),
                                                "kcentre": M[(g, b, t, ("kcentre", 0))], "leverage": M[(g, b, t, ("leverage", 0))]}
                ident = abs(M[(g, b, t, ("random_c3_redo", 0))] - M[(g, b, t, ("random", 0))])
                c3["identity"][f"{t}|{g}+E|{b}"] = float(ident)
                ex = M[(g, b, t, ("random_extra", nd))]
                c3["extra_draw_inside_range"][f"{t}|{g}+E|{b}"] = bool(rnd.min() <= ex <= rnd.max())
                c3["pass"] = bool(c3["pass"] and ident <= 1e-9 and rnd.min() <= M[(g, b, t, ("random_c3_redo", 0))] <= rnd.max())
            rows = [C["rows"][f"{t}|{g}+E|{b}"] for b in BUDGETS]
            per = {}
            for rule in ("kcentre", "leverage"):
                per[rule] = {"above_best_random_at_both": all(r[rule] > r["random_max"] for r in rows), "below_worst_random_at_both": all(r[rule] < r["random_min"] for r in rows)}
            lab = "SELECT-HELPS" if any(v["above_best_random_at_both"] for v in per.values()) else ("SELECT-HURTS" if any(v["below_worst_random_at_both"] for v in per.values()) else "SELECT-FLAT")
            mixed = any(v["above_best_random_at_both"] for v in per.values()) and any(v["below_worst_random_at_both"] for v in per.values())
            C["outcome_per_set"][f"{t}|{g}+E"] = {"label": lab, "rules": per, "mixed_helps_and_hurts": bool(mixed)}
    for t in TARGETS:
        l = {g: C["outcome_per_set"][f"{t}|{g}+E"]["label"] for g in GRIDS}
        C["outcome"][t] = l["V40"] if l["V40"] == l["V160"] else {"V40+E": l["V40"], "V160+E": l["V160"]}
    res["C"] = C
    res["controls"]["c3"] = {**c3, "bar": "a random draw passed through the SELECT code path lies inside the range of the 20 random draws: here the draw-0 seed re-run through run path reproduces draw 0 to 1e-9 (R5); the extra independent draw is informational"}
    checkpoint("C")


# ----------------------------------------------------------------------------- outputs
def write_outputs(res, outdir, S, stopped):
    json.dump(res, open(os.path.join(outdir, "RESULT_RESCURVE.json"), "w", encoding="utf-8"), indent=1, default=float)
    np.savez_compressed(os.path.join(outdir, "OOF_RESCURVE.npz"), **{k: v.astype(np.float32) for k, v in S.OOF.items()})
    m = res["meta"]; c = res["controls"]; L = []
    L.append("# VERDICT_RESCURVE (PREREG_V8F_PROOF_ADD7)\n")
    L.append(f"Tier T0: a frozen-feature probe on logged frames, a perception diagnostic, not driving performance. STRATEGIC family: PAUSED (PI item 25), not applicable.")
    L.append(f"MEASURED, RESULT_RESCURVE.json, n frames {m['n_frames']}, episodes {m['n_episodes']}, folds {m['folds']}, B {m['B']}, smoke {m['smoke']}. Estimator: {m['estimator']}.\n")
    L.append("## Controls")
    if "c1" in c:
        L.append(f"* c1 (banked RESPROBE P1 reproduced to 1e-6 at rank 256): {('PASS' if c['c1']['pass'] else 'FAIL') if c['c1']['checked'] else 'NOT CHECKABLE on a subsample (banked values are full-set)'}, max abs diff {c['c1']['max_abs_diff']:.2e}")
    if "c2" in c:
        bad = {k: v["value"] for k, v in c["c2"]["rows"].items() if not v["pass"]}
        L.append(f"* c2 (shuffled targets, rank 256): {'PASS' if c['c2']['pass'] else 'FAIL'}; offenders {json.dumps(bad)}")
    if "c2_full" in c:
        bad = {k: v["value"] for k, v in c["c2_full"]["rows"].items() if not v["pass"]}
        L.append(f"* c2 full kernel (shuffled targets, V+E): {'PASS' if c['c2_full']['pass'] else 'FAIL'}; offenders {json.dumps(bad)}")
    if "c3" in c:
        L.append(f"* c3 (random draw through the SELECT path inside the 20-draw range): {'PASS' if c['c3']['pass'] else 'FAIL'}; max identity diff {max(c['c3']['identity'].values()):.1e}; extra independent draw inside range in {sum(c['c3']['extra_draw_inside_range'].values())}/{len(c['c3']['extra_draw_inside_range'])} cells (informational, chance outside is 2/21 per cell)")
    if stopped:
        L.append("\nSTOPPED: c1 failed, no read was made. The reads are not admissible.\n")
        open(os.path.join(outdir, "VERDICT_RESCURVE.md"), "w", encoding="utf-8").write("\n".join(L) + "\n"); return
    ok = all(c[k]["pass"] for k in ("c2", "c2_full", "c3")) and (c["c1"]["pass"] or not c["c1"]["checked"])
    if not ok:
        L.append("* A control missed its bar or could not be checked: the reads below are NOT admissible until it is explained.")
    A, B, C = res["A"], res["B"], res["C"]
    L.append(f"\n## A. RANK: **{A['outcome']}** (bar: V160+E minus V40+E on GAP >= +0.03 R2 with the 95 % interval excluding 0, at rank 1,024 or the full kernel)")
    for rk in ("r1024", "full"):
        r = A["diff_V160E_minus_V40E"][f"GAP|{rk}"]
        met = r["diff"] >= BAR and r["ci95"][0] > 0
        L.append(f"* GAP {rk}: {r['diff']:+.4f} CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]: {'PASS' if met else 'FAIL'}")
    L.append("\nMetric by rank (V40+E | V160+E | difference with 95 % CI); R2, AUROC for LEADP")
    for t in TARGETS:
        L.append(f"* {t} (E floor {A['table'][f'{t}|E|floor_P1']:+.3f}), per-target label {A['outcome_per_target'][t]['label']}:")
        for rk in RANK_ORDER:
            r = A["diff_V160E_minus_V40E"][f"{t}|{rk}"]
            L.append(f"    * {rk}: {A['table'][f'{t}|V40+E|{rk}']:.3f} | {A['table'][f'{t}|V160+E|{rk}']:.3f} | {r['diff']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
    L.append("\nRank the inner split prefers (smallest mean normalised inner-validation MSE over the 5 folds): " + "; ".join(f"{k} {v['rank']}" for k, v in A["pref_rank"].items()))
    L.append("\n## B. DATA (bar: last doubling 1/2 -> 1 gains >= 0.03 with the interval excluding 0 = DATA-LIMITED; upper end below 0.03 = SATURATED; else UNDECIDED)")
    for t in TARGETS:
        for g in GRIDS:
            for rk in ("r256", "pref"):
                r = B["gain_last_doubling"][f"{t}|{g}+E|{rk}"]
                L.append(f"* {t} {g}+E {r['rank_used']}: gain {r['diff']:+.4f} CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] **{r['label']}** (bars: >= 0.03 and lo > 0: {'PASS' if r['label'] == 'DATA-LIMITED' else 'FAIL'}; hi < 0.03: {'PASS' if r['label'] == 'SATURATED' else 'FAIL'})")
            r = B["gain_last_doubling"][f"{t}|{g}+E|E"]
            L.append(f"* {t} E floor (with the {g} folds): gain {r['diff']:+.4f} CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] {r['label']} (informational)")
    L.append("\nLearning curve: metric (mean over draws, sd) against training episodes (mean over folds; in brackets those with a valid target)")
    for t in TARGETS:
        for g in GRIDS:
            for rk in ("r256", "pref", "E"):
                cells = []
                for fr in B["fracs"]:
                    v = B["curve"][f"{t}|{g}+E|{rk}|{fr}"]
                    cells.append(f"{v['n_train_episodes']:.0f}({v['n_train_episodes_with_valid_target']:.0f}) ep: {v['mean']:.3f}+-{v['sd_over_draws']:.3f}")
                L.append(f"* {t} {g}+E {rk if rk != 'pref' else 'pref=' + str(res['A']['pref_rank'][g + '|' + t]['rank'])}: " + " | ".join(cells))
    L.append("\n## C. SELECT (T8): rules above the best / below the worst of the random draws at BOTH budgets")
    for t in TARGETS:
        o = C["outcome"][t]
        L.append(f"* {t}: **{json.dumps(o) if isinstance(o, dict) else o}**")
        for g in GRIDS:
            for b in BUDGETS:
                r = C["rows"][f"{t}|{g}+E|{b}"]
                L.append(f"    * {g}+E budget {b}: kcentre {r['kcentre']:.3f}, leverage {r['leverage']:.3f}, random min/mean/max {r['random_min']:.3f}/{r['random_mean']:.3f}/{r['random_max']:.3f}")
            p = C["outcome_per_set"][f"{t}|{g}+E"]
            L.append(f"    * {g}+E: {p['label']} (kcentre helps {p['rules']['kcentre']['above_best_random_at_both']}, hurts {p['rules']['kcentre']['below_worst_random_at_both']}; leverage helps {p['rules']['leverage']['above_best_random_at_both']}, hurts {p['rules']['leverage']['below_worst_random_at_both']}; mixed {p['mixed_helps_and_hurts']})")
    L.append("\n## What each outcome commits (card)")
    L.append("* RANK-HIDES: the RESPROBE grid comparison was rank-limited; the finer grid stays a live lever. RANK-FLAT with DATA-LIMITED: more labelled clips before a finer grid. RANK-FLAT with SATURATED: the content is not linearly present at either pooled grid; un-pooled tokens and the native-resolution crop on the pod are the remaining test.")
    L.append("* SELECT-HELPS: carried to the choice of training clips (own card, trained arm). SELECT-FLAT: no selection claim from T8 for this representation.")
    L.append("\n## Limits (card)")
    L.append("Probe-level statements about a frozen representation, 139 episodes. A null from these probes is not a null about what a trained reader can learn. The interval answers another draw of episodes; closed-form probes have no training seed. Deviations R1 onward are in CURVE_DEFS.md and are part of this read.")
    open(os.path.join(outdir, "VERDICT_RESCURVE.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", type=int, default=0, help="subsample of at most N frames by whole episodes")
    cmd_run(ap.parse_args())
