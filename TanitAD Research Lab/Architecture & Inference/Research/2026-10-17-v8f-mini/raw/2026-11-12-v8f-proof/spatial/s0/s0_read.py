"""S0 read (PREREG_V8F_PROOF_ADD12, Stage 0; choices in S0_DEFS.md, rules R1..R17). Tier T0: a world-model diagnostic, never driving performance.
CPU only. Subcommands:
    run      full read on the true labels -> RESULT_S0.json, VERDICT_S0.md (resumable per fold, stages/)
    smoke    12 episodes, lead labels permuted across rows -> smoke/ (no outcome name)
Environment: PYTHONPATH not needed (res/ and wm/ are added to sys.path here); python = the tanitad venv.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ROOT = "F:/TanitAD-scratch/v8proof"
for _p in (f"{ROOT}/res", f"{ROOT}/wm"):
    if _p not in sys.path:
        sys.path.append(_p)
import resprobe as RP  # noqa: E402  (numpy only at import)

ART = "E:/Projects/TanitAD-artifacts"
CLIPS = f"{ART}/sfresh_inputs/cache/eval/clips"
BANK = f"{ROOT}/res/cache/frames.npz"
LATENTS_FALLBACK = None
NCELL, CELLD = 40, 16
MEAN_D = 128
KSTEP = 10                       # 2.0 s = 10 cache steps of 0.2 s
NACT = KSTEP * 2                 # logged action numbers
NF = 16
SEED = 0
B_BOOT = 2000
KRR_CAP = 3000
LEAK_PCA_FIT = False             # the mutation test flips this (R5)
FULLRES_E_GAP = 0.133
TIER = "T0"
STATEMENT = "a world-model diagnostic, never driving performance"
FAMILIES = ("Four metric families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass).")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ===================================================================================== PCA and states (pure numpy)
def pca_basis(Xtr: np.ndarray, k: int):
    """Centred covariance eigenvectors. Xtr [n, d] float -> (mu [d], V [d, k]) with the k leading directions."""
    Xtr = np.asarray(Xtr, np.float64)
    mu = Xtr.mean(0)
    Xc = Xtr - mu
    C = Xc.T @ Xc / max(len(Xc) - 1, 1)
    w, U = np.linalg.eigh(C)
    k = min(k, U.shape[1])
    return mu, U[:, ::-1][:, :k].copy()


def fit_rows(tr: np.ndarray, n_rows: int) -> np.ndarray:
    """The rows a PCA may see: the training rows only (R5). LEAK_PCA_FIT removes the restriction (mutation test only)."""
    return np.arange(n_rows) if LEAK_PCA_FIT else np.asarray(tr)


def cell_block(X: np.ndarray, tr: np.ndarray, c: int, k: int = CELLD, add: np.ndarray | None = None):
    """Scores [N, k] of cell c for ALL rows with a basis fitted on the training rows. add [N, d] is a planted signal (control s2)."""
    xc = X[:, c, :].astype(np.float32)
    if add is not None:
        xc = xc + add.astype(np.float32)
    fr = fit_rows(tr, len(X))
    mu, V = pca_basis(xc[fr], k)
    return ((xc - mu.astype(np.float32)) @ V.astype(np.float32)).astype(np.float32)


def mean_token(X: np.ndarray, chunk: int = 1024) -> np.ndarray:
    out = np.zeros((len(X), X.shape[2]), np.float32)
    for a in range(0, len(X), chunk):
        out[a:a + chunk] = X[a:a + chunk].astype(np.float32).mean(1)
    return out


def mean_state(M: np.ndarray, tr: np.ndarray, n: int = MEAN_D) -> np.ndarray:
    fr = fit_rows(tr, len(M))
    mu, V = pca_basis(M[fr], n)
    return ((M - mu.astype(np.float32)) @ V.astype(np.float32)).astype(np.float32)


def grid_state(X: np.ndarray, tr: np.ndarray, k: int = CELLD, blocks: dict | None = None) -> np.ndarray:
    """S0g: per cell k directions, concatenated in cell order. blocks = precomputed {cell: [N,k]} overrides."""
    n_cell = X.shape[1]
    out = np.zeros((len(X), n_cell * k), np.float32)
    for c in range(n_cell):
        b = blocks[c] if blocks is not None and c in blocks else cell_block(X, tr, c, k)
        out[:, c * k:(c + 1) * k] = b
    return out


# ===================================================================================== ridge predictor and probes
def ridge_multi(F: np.ndarray, Y: np.ndarray, alphas):
    """Multi-output ridge with intercept. Returns list of (alpha, W [p, q], b [q]). Y of exact zeros gives W = 0, b = 0 exactly."""
    mu, ym = F.mean(0), Y.mean(0)
    Fc, Yc = F - mu, Y - ym
    A = Fc.T @ Fc
    c = Fc.T @ Yc
    sc = np.trace(A) / A.shape[0] + 1e-30
    ev, U = np.linalg.eigh(A)
    Uc = U.T @ c
    out = []
    for a in alphas:
        W = U @ (Uc / (ev + a * sc)[:, None])
        out.append((a, W, ym - mu @ W))
    return out


def fit_state_predictor(Z0, Zk, Act, ep, steps: int = KSTEP):
    """R6. Z0, Zk [n, d] state at g0 and g0+steps; Act [n, 2*NF...] logged action rows (first `steps` rows used). Returns predict(Z0, Act) -> state at g0+steps."""
    Z0 = np.asarray(Z0, np.float64); Zk = np.asarray(Zk, np.float64)
    A = np.asarray(Act, np.float64)[:, :2 * steps]
    s_state = float(np.sqrt(Z0.var(0).mean()) + 1e-12)
    if A.shape[1] > 0:
        a_mu, a_sd = A.mean(0), A.std(0) + 1e-9
    else:
        a_mu, a_sd = np.zeros(0), np.ones(0)

    def feats(Z, Ac):
        Ac = np.asarray(Ac, np.float64)[:, :2 * steps]
        return np.concatenate([Z, (Ac - a_mu) / a_sd * s_state], 1)

    F, Y = feats(Z0, A), Zk - Z0
    iv = RP.inner_split(ep, SEED)
    if iv.sum() == 0 or (~iv).sum() < 4:
        iv = np.arange(len(F)) % 5 == 0
    sols = ridge_multi(F[~iv], Y[~iv], RP.ALPHAS)
    errs = [float(((Y[iv] - (F[iv] @ W + b)) ** 2).sum(1).mean()) for (_, W, b) in sols]
    a_best = sols[int(np.argmin(errs))][0]
    _, W, b = ridge_multi(F, Y, [a_best])[0]

    def predict(Zq, Aq):
        return np.asarray(Zq, np.float64) + feats(Zq, Aq) @ W + b
    predict.alpha = a_best
    return predict


def _sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T, 0.0)


GAMMAS = (0.5, 1.0, 2.0)
LAMS = (1e-3, 1e-2, 1e-1, 1.0)


def fit_probe(X, y, ep, seed: int = SEED):
    """R7. Ridge or RBF kernel ridge, whichever has the lower validation error on an inner episode split of the training rows; refitted on all."""
    X = np.asarray(X, np.float64); y = np.asarray(y, np.float64)
    n = len(y)
    if n < 12:
        m = float(y.mean()) if n else 0.0
        return lambda Z: np.full(len(Z), m)
    mu = X.mean(0)
    sc = float(np.sqrt(X.var(0).mean()) + 1e-12)
    Xs = (X - mu) / sc
    iv = RP.inner_split(ep, seed)
    if iv.sum() < 3 or (~iv).sum() < 8:
        iv = np.arange(n) % 5 == 0
    itr, iva = np.flatnonzero(~iv), np.flatnonzero(iv)
    rng = np.random.default_rng(seed)
    # ridge candidates
    sols = RP.ridge_solve(Xs[itr], y[itr], RP.ALPHAS)
    r_err = [float(((y[iva] - (Xs[iva] @ w + b)) ** 2).mean()) for (_, w, b) in sols]
    best = ("ridge", min(r_err), sols[int(np.argmin(r_err))][0])
    # kernel ridge candidates
    sub = itr if len(itr) <= 2400 else np.sort(rng.choice(itr, 2400, replace=False))
    ys = y[sub] - y[sub].mean()
    D2 = _sqd(Xs[sub], Xs[sub])
    med = float(np.median(D2[np.triu_indices(len(sub), 1)])) if len(sub) > 1 else 1.0
    med = med if med > 0 else 1.0
    Dv = _sqd(Xs[iva], Xs[sub])
    for gm in GAMMAS:
        g = gm / med
        ev, U = np.linalg.eigh(np.exp(-g * D2))
        Uy = U.T @ ys
        Kv = np.exp(-g * Dv)
        for lam in LAMS:
            al = U @ (Uy / (ev + lam))
            e = float(((y[iva] - (Kv @ al + y[sub].mean())) ** 2).mean())
            if e < best[1]:
                best = ("krr", e, (g, lam))
    if best[0] == "ridge":
        _, w, b = RP.ridge_solve(Xs, y, [best[2]])[0]
        return lambda Z: ((np.asarray(Z, np.float64) - mu) / sc) @ w + b
    g, lam = best[2]
    fsub = np.arange(n) if n <= KRR_CAP else np.sort(rng.choice(n, KRR_CAP, replace=False))
    ym = y[fsub].mean()
    K = np.exp(-g * _sqd(Xs[fsub], Xs[fsub]))
    al = np.linalg.solve(K + lam * np.eye(len(fsub)), y[fsub] - ym)
    Xf = Xs[fsub]

    def pred(Z):
        Zs = (np.asarray(Z, np.float64) - mu) / sc
        return np.exp(-g * _sqd(Zs, Xf)) @ al + ym
    return pred


def fold_cv_probe(Xs, y, valid, ep, folds, seed: int = SEED):
    """Out-of-fold probe predictions (only for tests / the ego floor of r-b path uses fit_probe per fold directly)."""
    oof = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr = np.flatnonzero((folds != f) & valid)
        te = np.flatnonzero((folds == f) & valid)
        if len(tr) < 4 or len(te) == 0:
            continue
        oof[te] = fit_probe(Xs[tr], y[tr], ep[tr], seed + int(f))(Xs[te])
    return oof


# ===================================================================================== labels
def permute_within_fold(y: np.ndarray, folds: np.ndarray, seed: int) -> np.ndarray:
    """Permute the finite values of y among the finite rows of the same fold (control s1)."""
    out = y.copy()
    rng = np.random.default_rng(seed)
    for f in np.unique(folds):
        idx = np.flatnonzero((folds == f) & np.isfinite(y))
        out[idx] = y[idx][rng.permutation(len(idx))]
    return out


def permute_across(y: np.ndarray, seed: int) -> np.ndarray:
    out = y.copy()
    idx = np.flatnonzero(np.isfinite(y))
    out[idx] = y[idx][np.random.default_rng(seed).permutation(len(idx))]
    return out


# ===================================================================================== ego-only floor of r-a (FULLRES R7)
def e_floor_ra(E, y, valid, epi, folds, binary: bool):
    """Better (pooled OOF metric) of RP ridge P1, RP MLP P2 and a plain linear fit on the standardised E features. Returns (oof, detail)."""
    ff = []
    for f in range(int(folds.max()) + 1):
        te, tr = np.flatnonzero(folds == f), np.flatnonzero(folds != f)
        mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-9
        ff.append(((E[tr] - mu) / sd, (E[te] - mu) / sd, tr, te))
    cand = {"P1": RP.run_probe_cv(ff, y, epi, folds, "P1", binary, valid), "P2": RP.run_probe_cv(ff, y, epi, folds, "P2", binary, valid)}
    lin = np.full(len(y), np.nan)
    for (Ftr, Fte, tr, te) in ff:
        mtr, mte = valid[tr], valid[te]
        A = np.concatenate([Ftr[mtr], np.ones((mtr.sum(), 1))], 1)
        w = np.linalg.lstsq(A, y[tr][mtr], rcond=None)[0]
        lin[te[mte]] = np.concatenate([Fte[mte], np.ones((mte.sum(), 1))], 1) @ w
    cand["linear"] = lin
    metric = (lambda p: RP.auc_pooled(y[valid], p[valid])) if binary else (lambda p: RP.r2_pooled(y[valid], p[valid]))
    sc = {k: metric(v) for k, v in cand.items()}
    pick = max(sc, key=lambda k: -1e9 if not np.isfinite(sc[k]) else sc[k])
    return cand[pick], {"scores": sc, "chosen": pick}


# ===================================================================================== data
def load_data(smoke: bool, n_smoke: int = 12):
    """Bank rows (R2), tokens (R1), windows (R3). Returns a dict."""
    import torch
    Z = np.load(BANK)
    beid = Z["beid"].astype(str); bg = Z["bg"].astype(np.int64); E = Z["E"].astype(np.float64)
    okE = np.isfinite(E).all(1)
    clips_all = sorted(set(beid))
    clips = clips_all[:n_smoke] if smoke else clips_all
    keep = okE & np.isin(beid, clips)
    rows = np.flatnonzero(keep)
    beid, bg, E = beid[rows], bg[rows], E[rows]
    gap = Z["lead_gap"][rows].astype(np.float64); pres = Z["lead_present"][rows].astype(np.float64)
    uep = sorted(set(beid))
    epi = np.array([uep.index(e) for e in beid])
    # tokens
    t0 = time.time()
    X = np.zeros((len(rows), NCELL, 1024), np.float16)
    by = {}
    for i, (e, g) in enumerate(zip(beid, bg)):
        by.setdefault(e, []).append(i)
    for ci_, c in enumerate(uep):
        o = torch.load(f"{CLIPS}/{c}.pt", map_location="cpu", weights_only=True, mmap=True)
        ii = np.array(by[c]); gs = bg[ii]
        assert gs.max() < o["pool4x10"].shape[0], (c, int(gs.max()))
        X[ii] = o["pool4x10"][torch.as_tensor(gs)].to(torch.float32).numpy().astype(np.float16)
        del o
        if ci_ % 20 == 0:
            log(f"[tokens] {ci_}/{len(uep)} clips {time.time() - t0:.0f}s")
    # windows
    import wmceil as W
    L = torch.load(W.LATENTS, weights_only=False)
    okw = L["ok"].all(1).numpy()
    t = np.asarray(L["t"]).astype(np.int64)
    eid = np.array([str(e) for e in L["eid"]])
    in_bank_clips = np.isin(eid, uep)
    cnt = {"windows_eval141": int(len(okw)), "windows_16_real_frames": int(okw.sum()), "windows_dropped_clip_not_in_bank_set": int((okw & ~in_bank_clips).sum())}
    pos = {(e, int(g)): i for i, (e, g) in enumerate(zip(beid, bg))}
    sel = np.flatnonzero(okw & in_bank_clips)
    g0 = t[sel] - (NF - 1)
    ri0 = np.array([pos.get((eid[s], int(g)), -1) for s, g in zip(sel, g0)])
    rik = np.array([pos.get((eid[s], int(g) + KSTEP), -1) for s, g in zip(sel, g0)])
    ok2 = (ri0 >= 0) & (rik >= 0)
    cnt["windows_dropped_no_bank_row_at_g0_or_g0+10"] = int((~ok2).sum())
    sel, g0, ri0, rik = sel[ok2], g0[ok2], ri0[ok2], rik[ok2]
    A = L["A_raw"][sel][:, :KSTEP, :].numpy().astype(np.float64).reshape(len(sel), NACT)
    okA = np.isfinite(A).all(1)
    cnt["windows_dropped_nonfinite_actions"] = int((~okA).sum())
    sel, g0, ri0, rik, A = sel[okA], g0[okA], ri0[okA], rik[okA], A[okA]
    cnt["windows_used"] = int(len(sel))
    return {"X": X, "beid": beid, "bg": bg, "E": E, "gap": gap, "pres": pres, "uep": uep, "epi": epi, "w_eid": eid[sel], "w_g0": g0, "ri0": ri0,
            "rik": rik, "A": A, "counts": cnt, "tok_seconds": time.time() - t0}


# ===================================================================================== the fold loop
def make_label_sets(D, folds_b, folds_w, mode: str):
    """mode 'true' | 'smoke' (permuted across rows). Returns {name: labels dict}, name in {main, s1}."""
    gap, pres = D["gap"], D["pres"]
    gk = D["gap"][D["rik"]]                                        # lead gap at g0+10 (window)
    g0v = D["gap"][D["ri0"]]
    both = np.isfinite(gk) & np.isfinite(g0v)
    gk_b = np.where(both, gk, np.nan)
    if mode == "smoke":
        gap_m, pres_m, gk_m = permute_across(gap, 11), permute_across(pres, 12), permute_across(gk_b, 13)
    else:
        gap_m, pres_m, gk_m = gap, pres, gk_b
    main = {"gap": gap_m, "pres": pres_m, "gapk": gk_m}
    s1 = {"gap": permute_within_fold(gap_m, folds_b, 1), "pres": permute_within_fold(pres_m, folds_b, 2), "gapk": permute_within_fold(gk_m, folds_w, 3)}
    return {"main": main, "s1": s1}, both


RB_PROBE = "bank"                # note N4 (stage 0b): "bank" = the r-a probe of the fold (training-fold bank frames); "windows" = the first read (N3b)
PLANT_SIGMAS = 12.0              # S0_DEFS R13 s2: amplitude in natural standard deviations of the planted cell along the planted direction


def planted_setup(D):
    N = len(D["X"])
    rng = np.random.default_rng(5)
    v = rng.standard_normal(X_D(D)).astype(np.float32)
    v = v / np.linalg.norm(v)
    cell = int(rng.integers(0, NCELL))
    flag = (rng.random(N) < 0.5).astype(np.float32)
    sig_v = float((D["X"][:, cell].astype(np.float32) @ v).std())
    med = float(np.median(np.linalg.norm(D["X"][rng.integers(0, N, 2000), cell].astype(np.float32), axis=1)))
    cell_b = (cell // 10) * 10 + (9 - cell % 10)                    # R18 s2p: the mirror cell of the same row (other half of the 4x10 grid)
    side = (np.random.default_rng(6).random(N) < 0.5).astype(np.float32)
    return {"vec": v * PLANT_SIGMAS * sig_v, "flag": flag, "cell": cell, "scale": PLANT_SIGMAS * sig_v, "sigma_v": sig_v, "median_token_norm": med,
            "cell_b": cell_b, "side": side, "vec_f5": v * med}


def X_D(D):
    return D["X"].shape[2]


def process_fold(f, D, M, labs, both, folds_b, folds_w, pl, log_=log):
    """All fits of fold f. Returns dict of arrays (full length, nan outside the test part)."""
    t0 = time.time()
    X = D["X"]
    N, nw = len(X), len(D["ri0"])
    tr = np.flatnonzero(folds_b != f); te = np.flatnonzero(folds_b == f)
    wtr = np.flatnonzero(folds_w != f); wte = np.flatnonzero(folds_w == f)
    out = {}
    St = {"g": grid_state(X, tr), "m": mean_state(M, tr)}
    log_(f"fold {f}: states g {St['g'].shape} m {St['m'].shape} ({time.time() - t0:.0f}s)")
    epi = D["epi"]; w_epi = np.array([D["uep"].index(e) for e in D["w_eid"]])
    E, A, ri0, rik = D["E"], D["A"], D["ri0"], D["rik"]
    # windows: predictor + horizon-0 control
    pred_states, s4 = {}, {}
    for s in ("g", "m"):
        Z0, Zk = St[s][ri0], St[s][rik]
        fn = fit_state_predictor(Z0[wtr], Zk[wtr], A[wtr], w_epi[wtr])
        pred_states[s] = fn(Z0[wte], A[wte])
        fn0 = fit_state_predictor(Z0[wtr], Z0[wtr], A[wtr], w_epi[wtr], steps=0)
        s4[s] = float(np.abs(fn0(Z0[wte], A[wte]) - Z0[wte]).max()) if len(wte) else 0.0
    out["s4_maxabs_g"] = np.array([s4["g"]]); out["s4_maxabs_m"] = np.array([s4["m"]])
    log_(f"fold {f}: predictors done ({time.time() - t0:.0f}s)")
    for lab, L in labs.items():
        ra_fit = {}
        # r-a
        for s in ("g", "m"):
            for tg in ("gap", "pres"):
                y = L[tg]
                vtr = tr[np.isfinite(y[tr])]; vte = te[np.isfinite(y[te])]
                key = f"ra|{lab}|{s}|{tg}"
                if key not in out:
                    out[key] = np.full(N, np.nan)
                if len(vtr) >= 12 and len(vte):
                    fit = fit_probe(St[s][vtr], y[vtr], epi[vtr], SEED + f)
                    out[key][vte] = fit(St[s][vte])
                    if tg == "gap":
                        ra_fit[s] = fit
        # r-b
        yk = L["gapk"]
        wv_tr = wtr[np.isfinite(yk[wtr])]; wv_te = wte[np.isfinite(yk[wte])]
        for s in ("g", "m"):
            for nm in ("i", "ii", "iii"):
                out.setdefault(f"rb|{lab}|{s}|{nm}", np.full(nw, np.nan))
            if len(wv_tr) >= 12 and len(wv_te):
                Z0, Zk = St[s][ri0], St[s][rik]
                fit = ra_fit[s] if (RB_PROBE == "bank" and s in ra_fit) else fit_probe(Zk[wv_tr], yk[wv_tr], w_epi[wv_tr], SEED + f)
                pos_in_te = {int(w): i for i, w in enumerate(wte)}
                pi = np.array([pos_in_te[int(w)] for w in wv_te])
                out[f"rb|{lab}|{s}|i"][wv_te] = fit(pred_states[s][pi])
                out[f"rb|{lab}|{s}|ii"][wv_te] = fit(Z0[wv_te])
                out[f"rb|{lab}|{s}|iii"][wv_te] = fit(Zk[wv_te])
        out.setdefault(f"rb|{lab}|ego|iv", np.full(nw, np.nan))
        out.setdefault(f"rb|{lab}|ego|ivb", np.full(nw, np.nan))
        yg_ = L["gap"]
        vtr_ = tr[np.isfinite(yg_[tr])]
        if len(vtr_) >= 12 and len(wv_te):
            fit_e = fit_probe(E[vtr_], yg_[vtr_], epi[vtr_], SEED + f)
            out[f"rb|{lab}|ego|ivb"][wv_te] = fit_e(E[rik[wv_te]])
        if len(wv_tr) >= 12 and len(wv_te):
            Fe = np.concatenate([E[ri0], A], 1)
            fit = fit_probe(Fe[wv_tr], yk[wv_tr], w_epi[wv_tr], SEED + f)
            out[f"rb|{lab}|ego|iv"][wv_te] = fit(Fe[wv_te])
        log_(f"fold {f}: label set {lab} done ({time.time() - t0:.0f}s)")
    # s2 planted object (label free)
    p = pl["cell"]
    add = pl["flag"][:, None] * pl["vec"][None, :]
    blk = cell_block(X, tr, p, CELLD, add=add)
    Sg = St["g"].copy(); Sg[:, p * CELLD:(p + 1) * CELLD] = blk
    Mp = M + pl["flag"][:, None] * pl["vec"][None, :] / NCELL
    Sm = mean_state(Mp, tr)
    for s, Sx in (("g", Sg), ("m", Sm)):
        key = f"s2|{s}"
        out[key] = np.full(N, np.nan)
        fit = fit_probe(Sx[tr], pl["flag"][tr].astype(np.float64), epi[tr], SEED + f)
        out[key][te] = fit(Sx[te])
    # s2p planted POSITION (S0_DEFS R18; lemma group P, VC-P3 / VC-P4): the mean token gets the SAME shift for both positions
    pa, pb, side = pl["cell"], pl["cell_b"], pl["side"]
    Sg2 = St["g"].copy()
    Sg2[:, pa * CELLD:(pa + 1) * CELLD] = cell_block(X, tr, pa, CELLD, add=(1.0 - side)[:, None] * pl["vec_f5"][None, :])
    Sg2[:, pb * CELLD:(pb + 1) * CELLD] = cell_block(X, tr, pb, CELLD, add=side[:, None] * pl["vec_f5"][None, :])
    Sm2 = mean_state(M + pl["vec_f5"][None, :] / NCELL, tr)
    for s, Sx in (("g", Sg2), ("m", Sm2)):
        key = f"s2p|{s}"
        out[key] = np.full(N, np.nan)
        fit = fit_probe(Sx[tr], side[tr].astype(np.float64), epi[tr], SEED + f)
        out[key][te] = fit(Sx[te])
    log_(f"fold {f}: planted-object controls done ({time.time() - t0:.0f}s)")
    return out


def merge_oof(parts):
    out = {}
    for d in parts:
        for k, v in d.items():
            if k.startswith("s4_"):
                out.setdefault(k, []).append(v)
            elif k not in out:
                out[k] = v.copy()
            else:
                m = np.isfinite(v)
                out[k][m] = v[m]
    for k in list(out):
        if k.startswith("s4_"):
            out[k] = float(np.max(np.concatenate(out[k])))
    return out


# ===================================================================================== analysis
def paired(ep, y, pa, pb, C, n_ep):
    Sa, Sb = RP.ep_stats_r2(ep, y, pa, n_ep), RP.ep_stats_r2(ep, y, pb, n_ep)
    return RP.diff_row(RP.boot_r2(Sa, C), RP.boot_r2(Sb, C), RP.r2_pooled(y, pa), RP.r2_pooled(y, pb))


def single_r2(ep, y, p, C, n_ep):
    S = RP.ep_stats_r2(ep, y, p, n_ep)
    return {"r2": RP.r2_pooled(y, p), "ci95": RP.ci(RP.boot_r2(S, C)), "n": int((np.isfinite(y) & np.isfinite(p)).sum())}


def single_auc(ep, y, p, C):
    return {"auc": RP.auc_pooled(y, p), "ci95": RP.ci(RP.boot_auc(ep, y, p, C)), "n": int((np.isfinite(y) & np.isfinite(p)).sum())}


def check_s1(r2s: dict, bar: float = 0.02):
    mx = max(r2s.values())
    return {"pass": bool(all(v <= bar for v in r2s.values())), "max_r2": float(mx), "bar": bar, "values": r2s}


def check_s2(auc_g: float, auc_m: float):
    return {"pass": bool(auc_g > 0.95 and auc_m <= 0.60), "auc_S0g": float(auc_g), "auc_S0m": float(auc_m), "bars": "S0g > 0.95 and S0m <= 0.60"}


def check_s3(r2_i: dict, r2_ii: dict, r2_iii: dict, e_gap: float, tol: float = 0.03):
    ceil_ok = all(r2_iii[s] >= r2_i[s] and r2_iii[s] >= r2_ii[s] for s in r2_iii)
    e_ok = abs(e_gap - FULLRES_E_GAP) <= tol
    return {"pass": bool(ceil_ok and e_ok), "ceiling_ok": bool(ceil_ok), "ego_floor_gap_r2": float(e_gap), "fullres_E": FULLRES_E_GAP, "tol": tol, "r2_i": r2_i, "r2_ii": r2_ii, "r2_iii": r2_iii}


def check_s4(maxabs: dict):
    return {"pass": bool(all(v == 0.0 for v in maxabs.values())), "max_abs_diff": maxabs}


def outcome_names(ra_vs_e: dict, rb_i_ii: dict, rb_i_iv: dict, grid_vs_mean: dict):
    out = {}
    for s in ("g", "m"):
        nm = "S0g" if s == "g" else "S0m"
        out[nm] = {"lead": "S0-LEAD-KEPT" if ra_vs_e[s]["separated_pos"] else "S0-LEAD-LOST"}
        a, b = rb_i_ii[s]["separated_pos"], rb_i_iv[s]["separated_pos"]
        out[nm]["imagine"] = "S0-IMAGINE-LEAD" if (a and b) else ("S0-IMAGINE-EGO" if a else "S0-IMAGINE-FLAT")
    out["headline"] = "GRID-ADDS" if grid_vs_mean["separated_pos"] else "GRID-EQUAL"
    return out


def analyse(D, oof, labs, both, folds_b, folds_w, B, floors):
    n_ep = len(D["uep"])
    C = RP.boot_counts(n_ep, B, SEED)
    epi = D["epi"]; w_epi = np.array([D["uep"].index(e) for e in D["w_eid"]])
    res = {}
    for lab in ("main", "s1"):
        L = labs[lab]
        r = {"ra": {}, "rb": {}}
        yg, yp, yk = L["gap"], L["pres"], L["gapk"]
        fl = floors[lab]
        for s in ("g", "m"):
            r["ra"][f"gap|{s}"] = single_r2(epi, yg, oof[f"ra|{lab}|{s}|gap"], C, n_ep)
            r["ra"][f"pres|{s}"] = single_auc(epi, yp, oof[f"ra|{lab}|{s}|pres"], C)
            for nm in ("i", "ii", "iii"):
                r["rb"][f"{s}|{nm}"] = single_r2(w_epi, yk, oof[f"rb|{lab}|{s}|{nm}"], C, n_ep)
        r["ra"]["gap|E"] = single_r2(epi, yg, fl["gap"][0], C, n_ep); r["ra"]["gap|E"]["chosen"] = fl["gap"][1]
        r["ra"]["pres|E"] = single_auc(epi, yp, fl["pres"][0], C); r["ra"]["pres|E"]["chosen"] = fl["pres"][1]
        r["rb"]["ego|iv"] = single_r2(w_epi, yk, oof[f"rb|{lab}|ego|iv"], C, n_ep)
        r["rb"]["ego|ivb"] = single_r2(w_epi, yk, oof[f"rb|{lab}|ego|ivb"], C, n_ep)
        if lab == "main":
            r["ra_vs_E"] = {s: paired(epi, yg, oof[f"ra|main|{s}|gap"], fl["gap"][0], C, n_ep) for s in ("g", "m")}
            r["grid_vs_mean"] = paired(epi, yg, oof["ra|main|g|gap"], oof["ra|main|m|gap"], C, n_ep)
            r["rb_i_ii"] = {s: paired(w_epi, yk, oof[f"rb|main|{s}|i"], oof[f"rb|main|{s}|ii"], C, n_ep) for s in ("g", "m")}
            fl_key = "ivb" if r["rb"]["ego|ivb"]["r2"] > r["rb"]["ego|iv"]["r2"] else "iv"      # note N4: the HIGHER of the two ego floors decides
            r["rb_floor_used"] = fl_key
            r["rb_i_iv"] = {s: paired(w_epi, yk, oof[f"rb|main|{s}|i"], oof[f"rb|main|ego|{fl_key}"], C, n_ep) for s in ("g", "m")}
            r["rb_i_iv_as_built"] = {s: paired(w_epi, yk, oof[f"rb|main|{s}|i"], oof["rb|main|ego|iv"], C, n_ep) for s in ("g", "m")}
            r["rb_ii_iv"] = {s: paired(w_epi, yk, oof[f"rb|main|{s}|ii"], oof[f"rb|main|ego|{fl_key}"], C, n_ep) for s in ("g", "m")}
            r["rb_iii_i"] = {s: paired(w_epi, yk, oof[f"rb|main|{s}|iii"], oof[f"rb|main|{s}|i"], C, n_ep) for s in ("g", "m")}
        res[lab] = r
    # controls
    s1_vals = {}
    for k in ("gap|g", "gap|m", "gap|E"):
        s1_vals[f"ra {k}"] = res["s1"]["ra"][k]["r2"]
    for k, v in res["s1"]["rb"].items():
        s1_vals[f"rb {k}"] = v["r2"]
    flag = floors["planted_flag"]
    ctr = {"s1": check_s1(s1_vals)}
    ctr["s1"]["presence_auc_permuted"] = {k: res["s1"]["ra"][k]["auc"] for k in ("pres|g", "pres|m", "pres|E")}
    ctr["s2"] = check_s2(RP.auc_pooled(flag, oof["s2|g"]), RP.auc_pooled(flag, oof["s2|m"]))
    ctr["s2p"] = check_s2(RP.auc_pooled(floors["planted_side"], oof["s2p|g"]), RP.auc_pooled(floors["planted_side"], oof["s2p|m"]))
    ctr["s2p"]["target"] = "position of the planted object (cell or mirror cell), f5 amplitude; lemma group P predicts S0m at 0.5"
    ctr["s3"] = check_s3({s: res["main"]["rb"][f"{s}|i"]["r2"] for s in ("g", "m")}, {s: res["main"]["rb"][f"{s}|ii"]["r2"] for s in ("g", "m")},
                         {s: res["main"]["rb"][f"{s}|iii"]["r2"] for s in ("g", "m")}, res["main"]["ra"]["gap|E"]["r2"])
    ctr["s4"] = check_s4({"S0g": oof["s4_maxabs_g"], "S0m": oof["s4_maxabs_m"]})
    if RB_PROBE == "bank":                                    # note N4, s5: (iii) of the test windows equals the r-a out-of-fold prediction at the same bank frames
        yk_ = labs["main"]["gapk"]
        d5 = {}
        for s_ in ("g", "m"):
            a_ = oof[f"rb|main|{s_}|iii"]; b_ = oof[f"ra|main|{s_}|gap"][D["rik"]]
            ok_ = np.isfinite(a_) & np.isfinite(b_) & np.isfinite(yk_)
            d5[s_] = {"n": int(ok_.sum()), "max_abs_diff": float(np.abs(a_[ok_] - b_[ok_]).max()) if ok_.any() else float("nan")}
        ctr["s5"] = {"pass": bool(all(v["n"] > 0 and v["max_abs_diff"] <= 1e-9 for v in d5.values())), **d5, "bar": "max abs diff <= 1e-9 on rows where both exist"}
    res["controls"] = ctr
    res["controls_pass"] = bool(all(v["pass"] for v in ctr.values()))
    res["outcomes"] = outcome_names(res["main"]["ra_vs_E"], res["main"]["rb_i_ii"], res["main"]["rb_i_iv"], res["main"]["grid_vs_mean"])
    return res


def ci_s(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def write_verdict(path, res, meta, smoke: bool):
    L = []
    ctr = res["controls"]
    L.append(f"# VERDICT_S0{' (SMOKE: permuted labels)' if smoke else ''}: PREREG_V8F_PROOF_ADD12 Stage 0")
    L.append(f"Tier {TIER}: {STATEMENT}. Estimator: paired episode-cluster bootstrap over {meta['n_episodes']} episodes, B {meta['B']}, seed 0. {FAMILIES}")
    L.append("A negative of stage 0 is a negative about a LINEAR predictor on an untrained state; it does not close stage 1. r-c (pixels): " + meta["r_c"])
    L.append("")
    L.append("## CONTROLS (first)")
    for k in (("s2", "s2p", "s4") if smoke else [k_ for k_ in ("s1", "s2", "s2p", "s3", "s4", "s5") if k_ in ctr]):
        c = ctr[k]
        L.append(f"- {k}: {'PASS' if c['pass'] else 'FAIL'} :: " + json.dumps({a: b for a, b in c.items() if a != 'pass'}, default=float))
    if not smoke:
        L.append(f"- controls_pass: {res['controls_pass']}")
    L.append("")
    if smoke:
        L.append("SMOKE: lead labels permuted across rows; no outcome name and no R2 is quoted. Shapes and timings are in RESULT_S0_SMOKE.json.")
    elif not res["controls_pass"]:
        L.append("## OUTCOME NAMES WITHHELD: a control did not read its known value. Numbers are in RESULT_S0.json, not quotable.")
    else:
        m = res["main"]
        L.append(f"## r-a retention (bank frames, n gap rows {m['ra']['gap|g']['n']}, presence rows {m['ra']['pres|g']['n']}; {TIER}; {STATEMENT})")
        for s, nm in (("g", "S0g (640)"), ("m", "S0m (128)"), ("E", "ego floor")):
            a, b = m["ra"][f"gap|{s}"], m["ra"][f"pres|{s}"]
            L.append(f"- {nm}: gap R2 {a['r2']:+.3f} {ci_s(a['ci95'])}; presence AUROC {b['auc']:.3f} {ci_s(b['ci95'])}")
        for s in ("g", "m"):
            d = m["ra_vs_E"][s]
            L.append(f"- gap R2 S0{s} minus ego floor: {d['diff']:+.3f} {ci_s(d['ci95'])}")
        d = m["grid_vs_mean"]
        L.append(f"- gap R2 S0g minus S0m: {d['diff']:+.3f} {ci_s(d['ci95'])}")
        L.append(f"## r-b imagined lead at +2.0 s (windows with a lead at both times, n {m['rb']['g|i']['n']}; {TIER}; {STATEMENT})")
        for s in ("g", "m"):
            L.append(f"- S0{s}: (i) predicted {m['rb'][f'{s}|i']['r2']:+.3f}, (ii) copy {m['rb'][f'{s}|ii']['r2']:+.3f}, (iii) true future {m['rb'][f'{s}|iii']['r2']:+.3f}; (i)-(ii) {m['rb_i_ii'][s]['diff']:+.3f} {ci_s(m['rb_i_ii'][s]['ci95'])}; (i)-(iv) {m['rb_i_iv'][s]['diff']:+.3f} {ci_s(m['rb_i_iv'][s]['ci95'])}")
        L.append(f"- (iv) ego only: {m['rb']['ego|iv']['r2']:+.3f} {ci_s(m['rb']['ego|iv']['ci95'])}")
        L.append(f"- (iv-b) ego only, bank probe on the true future ego vector (note N4): {m['rb']['ego|ivb']['r2']:+.3f} {ci_s(m['rb']['ego|ivb']['ci95'])}; floor used by the outcome rule: {m.get('rb_floor_used')}; r-b probe rows: {RB_PROBE}")
        L.append("## OUTCOME NAMES")
        for nm in ("S0g", "S0m"):
            L.append(f"- {nm}: {res['outcomes'][nm]['lead']}; {res['outcomes'][nm]['imagine']}")
        L.append(f"- headline (S0g against S0m): {res['outcomes']['headline']}")
    L.append("")
    L.append("Counts: " + json.dumps(meta["counts"]))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def cmd_run(smoke: bool):
    t_all = time.time()
    out_dir = f"{HERE}/smoke" if smoke else HERE
    stage_dir = f"{out_dir}/stages"
    os.makedirs(stage_dir, exist_ok=True)
    B = 200 if smoke else B_BOOT
    D = load_data(smoke)
    N, nw = len(D["X"]), len(D["ri0"])
    folds_b = RP.make_folds(D["beid"])
    fold_of = {e: int(f) for e, f in zip(D["beid"], folds_b)}
    folds_w = np.array([fold_of[e] for e in D["w_eid"]])
    fold_check = None
    if not smoke:
        fw2 = RP.make_folds(D["w_eid"])
        fold_check = bool(all(fold_of[e] == int(f) for e, f in zip(D["w_eid"], fw2)))
    labs, both = make_label_sets(D, folds_b, folds_w, "smoke" if smoke else "true")
    M = mean_token(D["X"])
    pl = planted_setup(D)
    log(f"rows {N}, windows {nw}, episodes {len(D['uep'])}, lead-both windows {int(both.sum())}, planted cell {pl['cell']} scale {pl['scale']:.2f}")
    sig = {"N": N, "nw": nw, "smoke": smoke, "both": int(both.sum())}
    parts, times = [], {}
    for f in range(5):
        sp = f"{stage_dir}/fold{f}.npz"
        if os.path.exists(sp):
            z = np.load(sp, allow_pickle=True)
            if json.loads(str(z["sig"])) == sig:
                parts.append({k: z[k] for k in z.files if k != "sig"}); log(f"fold {f}: resumed"); continue
        t0 = time.time()
        o = process_fold(f, D, M, labs, both, folds_b, folds_w, pl)
        times[f"fold{f}"] = time.time() - t0
        np.savez(sp, sig=json.dumps(sig), **o)
        parts.append(o)
    oof = merge_oof(parts)
    # ego-only floor of r-a (FULLRES rule) per label set
    floors = {"planted_flag": pl["flag"], "planted_side": pl["side"]}
    for lab in ("main", "s1"):
        L = labs[lab]
        floors[lab] = {"gap": e_floor_ra(D["E"], L["gap"], np.isfinite(L["gap"]), D["epi"], folds_b, False),
                       "pres": e_floor_ra(D["E"], L["pres"], np.isfinite(L["pres"]), D["epi"], folds_b, True)}
    log("ego floors done")
    res = analyse(D, oof, labs, both, folds_b, folds_w, B, floors)
    meta = {"card": "PREREG_V8F_PROOF_ADD12.md Stage 0", "defs": "S0_DEFS.md", "tier": TIER, "statement": STATEMENT, "families": FAMILIES, "B": B,
            "n_episodes": len(D["uep"]), "counts": {**D["counts"], "bank_rows": N, "lead_both_windows": int(both.sum()), "gap_valid_bank_rows": int(np.isfinite(D["gap"]).sum()),
                                                      "presence_valid_bank_rows": int(np.isfinite(D["pres"]).sum())},
            "fold_map_equals_latvis_construction": fold_check, "smoke": smoke, "stage_seconds": times, "token_seconds": D["tok_seconds"],
            "r_c": "not run (S0_DEFS R14: frames64.u8 and LO_r128.i8 deleted; LATVIS prep took 1,895 s of CPU; decoders need the GPU)",
            "estimator": "paired episode-cluster bootstrap", "shapes": {"X": list(D["X"].shape), "S0g": [N, NCELL * CELLD], "S0m": [N, MEAN_D]}}
    res["meta"] = meta
    json.dumps(res, default=float)                                         # format check of the full result and the full verdict writer (smoke only, file removed)
    _chk = f"{out_dir}/_format_check.tmp"
    write_verdict(_chk, {**res, "controls_pass": True}, meta, False)
    os.remove(_chk)
    if smoke:   # never store R2 of permuted labels as results: keep controls s2 and s4 and shapes
        keep = {"meta": meta, "controls": {k: res["controls"][k] for k in ("s2", "s2p", "s4")}, "total_seconds": time.time() - t_all}
        with open(f"{out_dir}/RESULT_S0_SMOKE.json", "w", encoding="utf-8") as fh:
            json.dump(keep, fh, indent=1, default=float)
        write_verdict(f"{out_dir}/VERDICT_S0_SMOKE.md", {**res, "controls": {k: v for k, v in res["controls"].items()}}, meta, True)
        log("SMOKE shapes", meta["shapes"], "counts", meta["counts"])
        log("SMOKE control s4:", res["controls"]["s4"]); log("SMOKE control s2 (label free):", res["controls"]["s2"]); log("SMOKE control s2p (label free):", res["controls"]["s2p"])
        log(f"SMOKE total {time.time() - t_all:.0f}s, stage seconds {times}")
        return 0
    with open(f"{out_dir}/RESULT_S0.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    write_verdict(f"{out_dir}/VERDICT_S0.md", res, meta, False)
    log(f"done {time.time() - t_all:.0f}s; controls_pass {res['controls_pass']}")
    return 0 if res["controls_pass"] else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "smoke"])
    a = ap.parse_args()
    sys.exit(cmd_run(a.cmd == "smoke"))
