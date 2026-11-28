r"""leadhead_s0.py -- LEADHEAD stage 0 (card PREREG_V8F_PROOF_ADD21, definitions LEADHEAD_DEFS.md L1..L17).
Tier T0: a world-model diagnostic on logged frames, never driving performance.  CPU, zero training of the model.
Does a supervised lead head on the history of the grid state (five frames) read the gap and the closing speed well enough for the kinematic slot
(rules R1, R2, R3 of card ADD20) to imagine 2.0 s ahead?  Heads: HT temporal ridge, HM small network; direct read of the change of the gap.

  python leadhead_s0.py run [--arms P_s0 P_s1 G_s0 G_s1] [--heads HT HM] [--no-direct] [--no-horizon] [--B 2000] [--out-dir DIR]

Strictly: STRATEGIC PAUSED (PI item 25); LONGITUDINAL, LATERAL, TACTICAL are not measured by this read.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/")
OBJ = "F:/TanitAD-scratch/v8proof/spatial/objslot"
if OBJ not in sys.path:
    sys.path.insert(0, OBJ)
import objslot_s0 as O  # noqa: E402

RP = O.RP
TIER = O.TIER
FAMILIES = O.FAMILIES
CACHE = "C:/Users/Admin/v8proof_c/leadhead/cache"
GRID = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5)
DT = O.DT
S0F = O.S0F
SEED_IN = 100
NPCA = 64
EPOCHS = 300
HORIZONS = (2, 4, 6, 8, 10)
TARGETS = {"gap": ("gap_ok", "gap"), "c": ("c_ok", "c"), "dir": ("dir_ok", "delta")}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================================ history builder and fit sets (pure)
def history_index(beid, bg, offsets):
    """Row index of the bank row (episode, bg + offset) for every row and offset; -1 where that row is not in the bank (L1)."""
    pos = {(str(e), int(g)): i for i, (e, g) in enumerate(zip(beid, bg))}
    idx = np.full((len(beid), len(offsets)), -1, np.int64)
    for j, o in enumerate(offsets):
        for i, (e, g) in enumerate(zip(beid, bg)):
            idx[i, j] = pos.get((str(e), int(g) + o), -1)
    return idx


def gather(a, idx):
    out = np.asarray(a, np.float64)[np.where(idx >= 0, idx, 0)].copy()
    out[idx < 0] = np.nan
    return out


def build_fit_sets(beid, bg, gap, speed, step_max=O.STEP_MAX_M, hmin=O.H_MIN):
    """Labels and masks of every bank row (L1..L3). idx5: rows t-4..t. gap_ok / c_ok / dir_ok: the rows that carry that label."""
    idx5 = history_index(beid, bg, range(-4, 1))
    ok5 = (idx5 >= 0).all(1)
    G5 = gather(gap, idx5)
    m = O.hist_run(G5, step_max)
    c = O.ls_slope(G5, m)
    hok = ok5 & (m.sum(1) >= hmin)
    c = np.where(hok, c, np.nan)
    gap_ok = ok5 & np.isfinite(gap)
    idxF = history_index(beid, bg, range(0, 11))
    okF = (idxF >= 0).all(1)
    G11 = gather(gap, idxF)
    V11 = gather(speed, idxF)
    with np.errstate(invalid="ignore"):
        fin = np.isfinite(G11).all(1)
        jump = np.nanmax(np.abs(np.diff(G11, axis=1)), axis=1) > step_max if len(G11) else np.zeros(0, bool)
    lead = fin & ~jump
    dir_ok = ok5 & okF & lead
    ego = np.zeros(len(beid))
    for k in range(10):
        ego += DT * 0.5 * (V11[:, k] + V11[:, k + 1])
    Xego = 2.0 * V11[:, 0] - ego
    delta = G11[:, 10] - G11[:, 0]
    return {"idx5": idx5, "ok5": ok5, "gap": np.asarray(gap, np.float64), "gap_ok": gap_ok, "c": c, "c_ok": hok & np.isfinite(c),
            "dir_ok": dir_ok & np.isfinite(Xego), "delta": delta, "Xego": Xego}


def ego_term_windows(V, K=10):
    V4, es = O.ego_terms(V, K)
    return DT * K * V4 - es


def inner_folds(ep, k=4, seed=SEED_IN):
    u = np.array(sorted(set(map(str, ep))))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(u))
    fo = {u[p]: i % k for i, p in enumerate(perm)}
    return np.array([fo[str(e)] for e in ep], np.int64)


# ============================================================================================ heads (pure numpy / torch CPU)
def _std(X):
    mu = X.mean(0)
    sd = X.std(0)
    return mu, np.where(sd < 1e-12, 1.0, sd)


def ridge_paths(Xtr, ytr, Xte, lams):
    """Predictions of the ridge (standardised on Xtr, intercept) for every penalty, dual form."""
    mu, sd = _std(Xtr)
    A, Bm = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    ev, U = np.linalg.eigh(A @ A.T)
    ev = np.maximum(ev, 0.0)
    Uy = U.T @ (ytr - ym)
    KU = (Bm @ A.T) @ U
    return [KU @ (Uy / (ev + l)) + ym for l in lams]


def fit_ht(X, y, ep, seed=SEED_IN, extra=0):
    """HT (L5). The penalty multiplier is chosen by 4 inner folds by episode on the rows given (the fit rows) and nowhere else."""
    n = len(y)
    fo = inner_folds(ep, 4, seed)
    sse = np.zeros(len(GRID))
    for j in range(4):
        tr, va = fo != j, fo == j
        if va.sum() == 0 or tr.sum() < 5:
            continue
        P = ridge_paths(X[tr], y[tr], X[va], [g * tr.sum() / 1000.0 for g in GRID])
        for i, p in enumerate(P):
            sse[i] += float(((y[va] - p) ** 2).sum())
    ib = int(np.argmin(sse))
    lam = GRID[ib] * n / 1000.0
    mu, sd = _std(X)
    A = (X - mu) / sd
    ym = y.mean()
    ev, U = np.linalg.eigh(A @ A.T)
    w = A.T @ (U @ ((U.T @ (y - ym)) / (np.maximum(ev, 0.0) + lam)))

    def predict(Z):
        return ((np.asarray(Z, np.float64) - mu) / sd) @ w + ym
    return predict, {"g": GRID[ib], "lam": float(lam), "inner_mse": float(sse[ib] / n), "inner_mse_by_g": [float(s / n) for s in sse], "n_fit": int(n)}


def _train_mlp(P, yz, seed):
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(P.shape[1], 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt = torch.tensor(P, dtype=torch.float32)
    yt = torch.tensor(yz, dtype=torch.float32).reshape(-1, 1)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = ((net(Xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    net.eval()
    return net


def fit_hm(X, y, ep=None, seed=SEED_IN, extra=0):
    """HM (L6): standardise, PCA 64 by the Gram matrix of the fit rows, 64-64-1 GELU, Adam, 300 epochs full batch, 3 initialisations averaged."""
    import torch
    n = len(y)
    ncol = X.shape[1] - extra
    mu, sd = _std(X)
    A = (X - mu) / sd
    k = min(NPCA, n - 1)
    ev, U = np.linalg.eigh(A[:, :ncol] @ A[:, :ncol].T)
    o = np.argsort(ev)[::-1][:k]
    V = A[:, :ncol].T @ U[:, o] / np.sqrt(np.maximum(ev[o], 1e-12))
    P = A[:, :ncol] @ V
    s1 = np.where(P.std(0) < 1e-12, 1.0, P.std(0))
    P = P / s1
    if extra:
        P = np.c_[P, A[:, ncol:]]
    ym, ys = float(y.mean()), float(y.std()) or 1.0
    nets = [_train_mlp(P, (y - ym) / ys, s) for s in (0, 1, 2)]

    def predict(Z):
        B = (np.asarray(Z, np.float64) - mu) / sd
        Q = B[:, :ncol] @ V / s1[None, :]
        if extra:
            Q = np.c_[Q, B[:, ncol:]]
        Qt = torch.tensor(Q, dtype=torch.float32)
        with torch.no_grad():
            out = np.mean([nn(Qt).numpy().reshape(-1) for nn in nets], axis=0)
        return out.astype(np.float64) * ys + ym
    return predict, {"n_fit": int(n), "n_pca": int(k)}


def inner_mse_hm(X, y, ep, seed=SEED_IN, extra=0):
    fo = inner_folds(ep, 4, seed)
    sse, cnt = 0.0, 0
    for j in range(4):
        tr, va = fo != j, fo == j
        if va.sum() == 0 or tr.sum() < 10:
            continue
        pred, _ = fit_hm(X[tr], y[tr], ep[tr], seed, extra)
        sse += float(((y[va] - pred(X[va])) ** 2).sum())
        cnt += int(va.sum())
    return sse / max(cnt, 1)


def fit_head(kind, X, y, ep, seed, extra=0):
    return fit_ht(X, y, ep, seed, extra) if kind == "HT" else fit_hm(X, y, ep, seed, extra)


def fit_fold(kind, Sb, FS, epi, fit_rows, f, targets, perm_c=False, inner=False):
    """Fit the heads of one fold on its fit rows only (L3). Returns (models, info)."""
    models, info = {}, {}
    for tg in targets:
        okm, col = TARGETS[tg]
        r = fit_rows[FS[okm][fit_rows]]
        X = Sb[FS["idx5"][r]].reshape(len(r), -1)
        extra = 1 if tg == "dir" else 0
        if extra:
            X = np.c_[X, FS["Xego"][r]]
        y = FS[col][r].copy()
        if tg == "c" and perm_c:
            y = y[np.random.default_rng(7 + f).permutation(len(y))]
        pred, inf = fit_head(kind, X, y, epi[r], SEED_IN + f, extra)
        if kind == "HM" and inner and tg in ("c", "dir"):
            inf["inner_mse"] = inner_mse_hm(X, y, epi[r], SEED_IN + f, extra)
        inf["n_fit"] = int(len(r))
        inf["episodes_fit"] = int(len(set(map(str, epi[r]))))
        models[tg], info[tg] = pred, inf
    return models, info


def apply_models(models, Z1, Xego_w):
    """Head outputs at S0 from the states of frames 0 to 4 of the windows. Only those frames are ever sliced."""
    X = np.asarray(Z1[:, :S0F + 1, :], np.float64).reshape(len(Z1), -1)
    out = {}
    for tg, m in models.items():
        out[tg] = m(np.c_[X, Xego_w]) if tg == "dir" else m(X)
    return out


def crossfit(kind, Sb, FS, epi, folds_b, Z1, Xego_w, fold, targets, perm_c=False, inner=False):
    nw = len(Z1)
    out = {tg: np.full(nw, np.nan) for tg in targets}
    infos, models = {}, {}
    for f in range(5):
        fit_rows = np.flatnonzero(FS["ok5"] & (folds_b != f))
        te = np.flatnonzero(fold == f)
        mdl, inf = fit_fold(kind, Sb, FS, epi, fit_rows, f, targets, perm_c, inner)
        o = apply_models(mdl, Z1[te], Xego_w[te])
        for tg in targets:
            out[tg][te] = o[tg]
        infos[f], models[f] = inf, mdl
    return out, infos, models


def head_pipeline(g0, c, V, K, y, scored, fold):
    """The rule path of this script (L9): the head's gap and closing speed -> R0..R3. R3 fitted on the training folds' scored windows."""
    inp = O.rule_inputs(g0, c, V, K)
    pr = O.rule_preds(inp)
    pr["R3"], coefs = O.fit_r3(inp, y, scored, fold)
    return {"pred": pr, "scored": scored, "valid": scored, "y": y, "inp": inp, "coefs": coefs}


def label_c(G15):
    Gh = G15[:, :S0F + 1]
    m = O.hist_run(Gh)
    c = O.ls_slope(Gh, m)
    hok = m.sum(1) >= O.H_MIN
    return np.where(hok, c, np.nan), hok


# ============================================================================================ intervals
def rmse_row(y, p, valid, ep, n_ep, C):
    m = valid & np.isfinite(y) & np.isfinite(p) & (ep >= 0)
    n = np.zeros(n_ep)
    s = np.zeros(n_ep)
    np.add.at(n, ep[m], 1.0)
    np.add.at(s, ep[m], (y - p)[m] ** 2)
    pt = float(np.sqrt(s.sum() / max(n.sum(), 1)))
    b = np.sqrt((C @ s) / np.maximum(C @ n, 1))
    return {"rmse": pt, "ci95": RP.ci(b), "n": int(m.sum())}


def fidelity(g_h, c_h, G15, c_lab, valid, ep, n_ep, C):
    yg = G15[:, S0F]
    lg, _ = O.level_row(yg, g_h, valid, ep, n_ep, C)
    lc, _ = O.level_row(c_lab, c_h, valid, ep, n_ep, C)
    return {"gap": {"r2": lg["r2"], "r2_ci95": lg["ci95"], "n": lg["n"], **{"rmse_" + k: v for k, v in rmse_row(yg, g_h, valid, ep, n_ep, C).items()}},
            "closing": {"r2": lc["r2"], "r2_ci95": lc["ci95"], "n": lc["n"], **{"rmse_" + k: v for k, v in rmse_row(c_lab, c_h, valid, ep, n_ep, C).items()}}}


def banked_compare(PV, y, sc, ep, n_ep, C, arm, eid, g0):
    bz = np.load("%s/rb/OOF_RB_%s_R1.npz" % (O.CT, arm), allow_pickle=True)
    assert np.array_equal(bz["w_eid"].astype(str), eid) and np.array_equal(bz["w_g0"], g0)
    pi = bz["p_i"]
    out = {}
    for rr in ("R3", "R2", "R1"):
        pa = PV["pred"][rr]
        m2 = sc & np.isfinite(PV["pred"]["R3"]) & np.isfinite(pi) & np.isfinite(pa) & (ep >= 0)
        Sa = RP.ep_stats_r2(ep[m2], y[m2], pa[m2], n_ep)
        Sb = RP.ep_stats_r2(ep[m2], y[m2], pi[m2], n_ep)
        dd = RP.diff_row(RP.boot_r2(Sa, C), RP.boot_r2(Sb, C), RP.r2_pooled(y[m2], pa[m2]), RP.r2_pooled(y[m2], pi[m2]))
        dd["level_r2_rule"] = float(RP.r2_pooled(y[m2], pa[m2]))
        dd["level_r2_banked_rollout"] = float(RP.r2_pooled(y[m2], pi[m2]))
        dd["n"] = int(m2.sum())
        out[rr] = dd
    return out


# ============================================================================================ controls (synthetic / label path)
def control_h1(s_noise, seed=0, n_ep=60, L=30, d=640, amp=10.0, kind="HT"):
    """Planted state (L13 h1): same fit-row code, same inner penalty choice, cross-fit by episode."""
    rng = np.random.default_rng(seed)
    beid, bg, gap = [], [], []
    for e in range(n_ep):
        g_e, c_e = rng.uniform(15, 60), rng.normal(0, 1.5)
        for t in range(L):
            beid.append("syn%03d" % e)
            bg.append(t)
            gap.append(g_e + c_e * DT * t)
    beid, bg, gap = np.array(beid), np.array(bg), np.array(gap)
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    Sb = rng.standard_normal((len(gap), d)) * s_noise + (gap * amp * s_noise)[:, None] * u[None, :]
    FS = build_fit_sets(beid, bg, gap, np.full(len(gap), 10.0))
    folds_b = RP.make_folds(beid)
    epi = np.array([int(str(e)[3:]) for e in beid])
    pred = np.full(len(gap), np.nan)
    for f in range(5):
        fit_rows = np.flatnonzero(FS["ok5"] & (folds_b != f))
        mdl, _ = fit_fold(kind, Sb, FS, epi, fit_rows, f, ("c",))
        te = np.flatnonzero(FS["ok5"] & (folds_b == f))
        pred[te] = mdl["c"](Sb[FS["idx5"][te]].reshape(len(te), -1))
    m = FS["c_ok"] & np.isfinite(pred)
    r2 = float(RP.r2_pooled(FS["c"][m], pred[m]))
    return {"pass": bool(r2 > 0.9), "heldout_r2_closing": r2, "bar": 0.9, "n": int(m.sum()), "noise_sd_per_dim": float(s_noise), "amp_per_metre": amp, "kind": kind}


def control_h6(G15, V, fold, banked_r2_skill):
    c, hok = label_c(G15)
    sc, _ = O.scored_mask(G15, 10)
    valid = sc & hok
    y = G15[:, S0F + 10]
    P = head_pipeline(G15[:, S0F], c, V, 10, y, valid, fold)
    s = O.skill_point(y, P["pred"]["R2"], P["pred"]["R0"], valid)
    P2 = O.label_pipeline(G15, V, 10, fold)
    s2 = O.skill_point(y, P2["pred"]["R2"], P2["pred"]["R0"], P2["valid"])
    return {"pass": bool(abs(s - banked_r2_skill) <= 1e-6 and abs(s - s2) <= 1e-6), "r2_skill_this_path": s, "r2_skill_objslot_label_pipeline_now": s2,
            "r2_skill_banked_ADD20": banked_r2_skill, "n": int(valid.sum()), "tol": 1e-6}


def control_h2(c_h_perm, c_lab, valid):
    r2 = float(RP.r2_pooled(c_lab[valid], c_h_perm[valid]))
    return {"pass": bool(r2 <= 0.02), "heldout_r2_closing_shuffled_labels": r2, "bar": 0.02, "n": int(valid.sum())}


def control_h4(c_h_all, c_h_cf, c_lab, valid):
    ins = float(RP.r2_pooled(c_lab[valid], c_h_all[valid]))
    cf = float(RP.r2_pooled(c_lab[valid], c_h_cf[valid]))
    return {"pass": bool(ins > cf), "r2_in_sample_all_folds": ins, "r2_cross_fitted": cf, "n": int(valid.sum())}


def control_h5(models, Z1, Xego_w, V, y, sc, fold, base_out):
    """Replace every window state after frame 4 by noise: every head output and every rule output at S0 identical (L13 h5)."""
    rng = np.random.default_rng(3)
    Zm = Z1.copy()
    Zm[:, S0F + 1:, :] = rng.standard_normal(Zm[:, S0F + 1:, :].shape).astype(Zm.dtype) * 5
    nw = len(Z1)
    outm = {tg: np.full(nw, np.nan) for tg in base_out}
    for f in range(5):
        te = np.flatnonzero(fold == f)
        o = apply_models(models[f], Zm[te], Xego_w[te])
        for tg in outm:
            outm[tg][te] = o[tg]
    dh = max(float(np.nanmax(np.abs(outm[tg] - base_out[tg]))) for tg in outm)
    P0 = head_pipeline(base_out["gap"], base_out["c"], V, 10, y, sc, fold)
    P1 = head_pipeline(outm["gap"], outm["c"], V, 10, y, sc, fold)
    dr = max(float(np.nanmax(np.abs(P0["pred"][r] - P1["pred"][r]))) for r in ("R1", "R2", "R3"))
    return {"pass": bool(dh == 0.0 and dr == 0.0), "max_abs_change_of_head_outputs": dh, "max_abs_change_of_rule_outputs": dr}


# ============================================================================================ data of one arm
def window_rows(D, eid, g0):
    """Bank row of window frame f (g0 - 4 + f) for frames 0..4, -1 where absent."""
    pos = {(str(e), int(g)): i for i, (e, g) in enumerate(zip(D["beid"], D["bg"]))}
    return np.array([[pos.get((str(e), int(g) - 4 + f), -1) for f in range(5)] for e, g in zip(eid, g0)])


def hl_gaps(arm, D, folds_b, Z1, fold):
    os.makedirs(CACHE, exist_ok=True)
    p = "%s/HL_%s.npz" % (CACHE, arm)
    if os.path.exists(p):
        return np.load(p)["VG"], "cached"
    fits = O.fit_vision_probes(D, folds_b)
    VG = O.vision_gaps(fits, Z1, fold)
    np.savez(p, VG=VG)
    return VG, "computed"


# ============================================================================================ run
def run(arms, B, out_dir, kinds, do_direct, do_horizon):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    res = {"tier": TIER, "families": FAMILIES, "card": "PREREG_V8F_PROOF_ADD21", "B": B, "kinds_run": list(kinds), "direct_run": bool(do_direct),
           "horizon_run": bool(do_horizon), "arms": {}, "controls": {}}
    banked = json.load(open(OBJ + "/RESULT_OBJSLOT_S0.json", encoding="utf-8"))
    ref_r2 = banked["label"]["10"]["rules"]["R2"]["skill_over_copy"]["skill"]
    first = None
    ctrl = {}
    for arm in arms:
        log("arm", arm)
        D, lat = O.load_arm(arm)
        eid, g0, fold, folds_b = O.arm_windows(D)
        if first is None:
            first = (eid, g0, fold)
            G15, miss = O.load_label_frames(eid, g0)
            res["counts"] = {"windows": int(len(eid)), "label_frames_without_bank_row": int(miss)}
            V0, _, _ = O.speed_and_states(arm, eid, g0)
            ctrl["h6_label_path"] = control_h6(G15, V0, fold, ref_r2)
            ctrl["h1_planted_state"] = control_h1(float(D["Sb"].std(0).mean()))
            log("h1", ctrl["h1_planted_state"]["pass"], ctrl["h1_planted_state"]["heldout_r2_closing"], "h6", ctrl["h6_label_path"]["pass"])
        else:
            assert np.array_equal(first[0], eid) and np.array_equal(first[1], g0) and np.array_equal(first[2], fold)
        V, Z1, _ = O.speed_and_states(arm, eid, g0)
        Xw = ego_term_windows(V, 10)
        FS = build_fit_sets(D["beid"], D["bg"], D["gap"], D["E"][:, 0])
        epi = D["epi"]
        wr = window_rows(D, eid, g0)
        sdiff = max(float(np.abs(Z1[wr[:, f] >= 0, f, :].astype(np.float64) - D["Sb"][wr[wr[:, f] >= 0, f]]).max()) for f in range(5))
        n_frames_no_bank_row = int((wr < 0).sum())
        sc, cnt = O.scored_mask(G15, 10)
        c_lab, hok = label_c(G15)
        valid = sc & hok
        y = G15[:, S0F + 10]
        ep10, n10 = O.epi_of(eid, sc)
        C10 = RP.boot_counts(n10, B, O.SEED)
        A = {"counts": {"scored": cnt, "label_history_ok": int(valid.sum()), "label_history_dropped": int((sc & ~hok).sum()), "n_episodes_scored": n10,
                        "state_max_abs_diff_window_vs_bank_frames0to4": sdiff, "window_frames_0to4_without_bank_row": n_frames_no_bank_row}, "fit_rows": {}, "kinds": {}, "controls": {}}
        for f in range(5):
            trn = folds_b != f
            A["fit_rows"][f] = {"bank_rows_in_training_folds": int(trn.sum()), "with_five_frames": int((trn & FS["ok5"]).sum()),
                                "dropped_no_five_frames": int((trn & ~FS["ok5"]).sum()), "gap_rows": int((trn & FS["gap_ok"]).sum()),
                                "closing_rows": int((trn & FS["c_ok"]).sum()), "direct_rows": int((trn & FS["dir_ok"]).sum()),
                                "episodes": int(len(set(map(str, D["beid"][trn]))))}
        # h3
        VG, how = hl_gaps(arm, D, folds_b, Z1, fold)
        bz = np.load(OBJ + "/VG_%s.npz" % arm)
        dmax = float(np.nanmax(np.abs(VG - bz["VG"])))
        A["controls"]["h3_reference_fidelity"] = {"pass": bool(dmax <= 1e-6), "max_abs_diff_to_banked_VG": dmax, "HL_source": how}
        PL = head_pipeline(VG[:, S0F], O.ls_slope(VG, np.ones(VG.shape, bool)), V, 10, y, sc, fold)
        A["HL_reference"] = {"R3_skill_ADD20_style_over_HL_copy": O.skill_row(y, PL["pred"]["R3"], PL["pred"]["R0"], sc, ep10, n10, C10)[0]}
        targets = ("gap", "c", "dir") if do_direct else ("gap", "c")
        outs, allmodels = {}, {}
        for kind in kinds:
            ts = time.time()
            out, infos, models = crossfit(kind, D["Sb"], FS, epi, folds_b, Z1, Xw, fold, targets, inner=True)
            outs[kind], allmodels[kind] = out, models
            K_ = {"penalty_or_info": {f: {tg: {k: v for k, v in infos[f][tg].items() if k != "inner_mse_by_g"} for tg in infos[f]} for f in infos}}
            K_["inner_mse_closing_mean_over_folds"] = float(np.mean([infos[f]["c"]["inner_mse"] for f in infos]))
            if do_direct:
                K_["inner_mse_direct_mean_over_folds"] = float(np.mean([infos[f]["dir"]["inner_mse"] for f in infos]))
            K_["fidelity"] = fidelity(out["gap"], out["c"], G15, c_lab, valid, ep10, n10, C10)
            PV = head_pipeline(out["gap"], out["c"], V, 10, y, sc, fold)
            tb, _ = O.table(PV, ep10, n10, C10)
            K_["rules"] = tb
            K_["coefs"] = PV["coefs"]
            K_["banked"] = banked_compare(PV, y, sc, ep10, n10, C10, arm, eid, g0)
            if do_direct:
                pd = out["gap"] + out["dir"]
                m = sc & np.isfinite(pd) & np.isfinite(out["gap"])
                K_["direct"] = {"skill_over_copy": O.skill_row(y, pd, out["gap"], m, ep10, n10, C10)[0], "rmse_gap_2s": rmse_row(y, pd, m, ep10, n10, C10),
                                "rmse_copy_gap_2s": rmse_row(y, out["gap"], m, ep10, n10, C10)}
            if do_horizon:
                K_["horizon"] = {}
                for K in HORIZONS:
                    scK, cK = O.scored_mask(G15, K)
                    epK, nK = O.epi_of(eid, scK)
                    CK = RP.boot_counts(nK, B, O.SEED)
                    PK = head_pipeline(out["gap"], out["c"], V, K, G15[:, S0F + K], scK, fold)
                    tK, _ = O.table(PK, epK, nK, CK)
                    K_["horizon"][K] = {"n": int(scK.sum()), "rules": {r: tK[r]["skill_over_copy"] for r in ("R1", "R2", "R3")}}
            # h2 shuffled closing labels
            outp, _, _ = crossfit(kind, D["Sb"], FS, epi, folds_b, Z1, Xw, fold, ("c",), perm_c=True)
            A["controls"]["h2_shuffled_labels|" + kind] = control_h2(outp["c"], c_lab, valid)
            A["controls"]["h5_no_future_input|" + kind] = control_h5(models, Z1, Xw, V, y, sc, fold, out)
            if kind == "HT":
                fit_rows = np.flatnonzero(FS["ok5"])
                mdl, _ = fit_fold("HT", D["Sb"], FS, epi, fit_rows, 0, ("c",))
                call = apply_models(mdl, Z1, Xw)
                A["controls"]["h4_leak_detector|HT"] = control_h4(call["c"], out["c"], c_lab, valid)
            A["kinds"][kind] = K_
            log(arm, kind, "closing R2 %+.3f" % K_["fidelity"]["closing"]["r2"], "R3 skill %+.3f" % tb["R3"]["skill_over_copy"]["skill"], "(%.0f s)" % (time.time() - ts))
        if len(kinds) == 2:
            a, b = A["kinds"]["HT"]["inner_mse_closing_mean_over_folds"], A["kinds"]["HM"]["inner_mse_closing_mean_over_folds"]
            A["chosen_closing"] = "HT" if a <= b else "HM"
            if do_direct:
                a, b = A["kinds"]["HT"]["inner_mse_direct_mean_over_folds"], A["kinds"]["HM"]["inner_mse_direct_mean_over_folds"]
                A["chosen_direct"] = "HT" if a <= b else "HM"
        else:
            A["chosen_closing"] = kinds[0] + " (only head run in this stage)"
            A["chosen_direct"] = kinds[0] + " (only head run in this stage)"
        res["arms"][arm] = A
        write_all(res, arms, out_dir, final=False)
    for arm, A in res["arms"].items():
        for k, v in A["controls"].items():
            ctrl["%s|%s" % (k, arm)] = v
    res["controls"] = ctrl
    res["controls_pass"] = bool(all(v["pass"] for v in ctrl.values()))
    res["outcomes"] = outcomes(res, arms) if res["controls_pass"] else {"named": False, "reason": "a control failed; no outcome is named"}
    res["elapsed_s"] = round(time.time() - t0, 1)
    write_all(res, arms, out_dir, final=True)
    log("controls", "PASS" if res["controls_pass"] else "FAILED", "elapsed %.0f s" % res["elapsed_s"])
    return 0 if res["controls_pass"] else 2


def _pick(A, which):
    ch = A["chosen_" + which].split(" ")[0]
    return ch, A["kinds"][ch]


def sep_pos(row):
    return bool(row["separated_pos"]) and row["n"] >= 30 if "n" in row else bool(row["separated_pos"])


def outcomes(res, arms):
    o = {"named": True, "complete": bool(len(res["kinds_run"]) == 2 and res["direct_run"]), "per_type": {}}
    for typ in ("P", "G"):
        seeds = [a for a in arms if a.startswith(typ + "_")]
        if len(seeds) < 2:
            o["per_type"][typ] = "not named (needs both seeds; have %s)" % seeds
            continue
        d = {}
        pk = {a: _pick(res["arms"][a], "closing") for a in seeds}
        rc = all(pk[a][1]["fidelity"]["closing"]["r2"] > 0 and pk[a][1]["fidelity"]["closing"]["r2_ci95"][0] > 0 and pk[a][1]["fidelity"]["closing"]["n"] >= 30 for a in seeds)
        d["closing"] = "HEAD-READS-CLOSING" if rc else "HEAD-CLOSING-FLAT"
        d["head_used"] = {a: pk[a][0] for a in seeds}
        im = all(pk[a][1]["rules"]["R3"]["skill_over_copy"]["separated_pos"] for a in seeds)
        names = ["HEAD-SLOT-IMAGINES"] if im else ["HEAD-SLOT-FLAT"]
        if im and all(pk[a][1]["banked"]["R1"]["R3"]["separated_pos"] for a in seeds):
            names.append("HEAD-SLOT-BEATS-ROLLOUT")
        d["slot"] = names
        flags = {}
        for other in res["kinds_run"]:
            flags[other] = {"closing": ("HEAD-READS-CLOSING" if all(res["arms"][a]["kinds"][other]["fidelity"]["closing"]["r2_ci95"][0] > 0 for a in seeds) else "HEAD-CLOSING-FLAT"),
                            "slot": ("HEAD-SLOT-IMAGINES" if all(res["arms"][a]["kinds"][other]["rules"]["R3"]["skill_over_copy"]["separated_pos"] for a in seeds) else "HEAD-SLOT-FLAT")}
        d["same_names_with_each_head_run"] = flags
        if res["direct_run"]:
            pdk = {a: _pick(res["arms"][a], "direct") for a in seeds}
            dh = all(pdk[a][1]["direct"]["skill_over_copy"]["separated_pos"] for a in seeds)
            d["direct"] = "STATE-HOLDS-FUTURE" if dh else "STATE-LACKS-FUTURE"
            d["direct_head_used"] = {a: pdk[a][0] for a in seeds}
            d["direct_with_each_head_run"] = {k: ("STATE-HOLDS-FUTURE" if all(res["arms"][a]["kinds"][k]["direct"]["skill_over_copy"]["separated_pos"] for a in seeds) else "STATE-LACKS-FUTURE") for k in res["kinds_run"]}
        o["per_type"][typ] = d
    return o


# ============================================================================================ writing
def jdump(o, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(o, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


def _ci(c):
    return "[%+.3f, %+.3f]" % (c[0], c[1])


def _sk(row):
    return "%+.3f %s n=%d%s" % (row["skill"], _ci(row["ci95"]), row["n"], " (separated above 0)" if row["separated_pos"] else (" (separated below 0)" if row.get("separated_neg") else ""))


def write_all(res, arms, out_dir, final):
    if not final:
        res = dict(res)
        res["controls"] = {"%s|%s" % (k, arm): v for arm, A in res["arms"].items() for k, v in A["controls"].items()}
        res["outcomes"] = {"named": False, "reason": "partial write while the run is in progress"}
        res["controls_pass"] = None
    jdump(res, os.path.join(out_dir, "RESULT_LEADHEAD_S0.json"))
    write_verdict(os.path.join(out_dir, "VERDICT_LEADHEAD_S0.md"), res, [a for a in arms if a in res["arms"]], final)


def write_verdict(path, res, arms, final):
    kinds = res["kinds_run"]
    L = ["# VERDICT LEADHEAD stage 0 (PREREG_V8F_PROOF_ADD21)", "", "Tier " + TIER + ". " + FAMILIES + ". Zero training of the model; CPU; the banked states and labels.",
         "Heads run in this file: %s; direct read: %s; horizon profile: %s. Definitions: LEADHEAD_DEFS.md. Ego speeds of the rule are the logged ones (the action input of a rollout)." % (
             ", ".join(kinds), "run" if res["direct_run"] else "NOT run", "run" if res["horizon_run"] else "NOT run"), ""]
    if not final:
        L += ["**Partial write: the run was not finished when this file was written.**", ""]
    L += ["## Controls (first)", ""]
    for k, v in res["controls"].items():
        L.append("- %s: %s :: %s" % (k, "PASS" if v["pass"] else "FAIL", json.dumps({a: b for a, b in v.items() if a != "pass"}, default=str)))
    L.append("- controls_pass: %s" % res.get("controls_pass"))
    L.append("")
    if final and not res["controls_pass"]:
        L += ["**A control failed. No outcome is named.** The numbers are in RESULT_LEADHEAD_S0.json.", ""]
    L += ["## Per arm", ""]
    for arm in arms:
        A = res["arms"][arm]
        c = A["counts"]["scored"]
        L += ["### " + arm, "", "Scored windows %d of %d (episodes %d); with the label history rule %d (dropped %d). Window states equal the bank states of frames 0 to 4 (max abs diff %.2e where the bank row exists; %d of the window frames 0 to 4 have no bank row, their window state is used as it is)." % (
            c["scored"], c["windows"], A["counts"]["n_episodes_scored"], A["counts"]["label_history_ok"], A["counts"]["label_history_dropped"],
            A["counts"]["state_max_abs_diff_window_vs_bank_frames0to4"], A["counts"]["window_frames_0to4_without_bank_row"]), "",
            "Fit rows per fold (bank rows of the training folds; with five frames; dropped without; rows with a gap label; with a closing label; with a direct label):", ""]
        for f, r in A["fit_rows"].items():
            L.append("- fold %s: %d; %d; dropped %d; gap %d; closing %d; direct %d; episodes %d" % (f, r["bank_rows_in_training_folds"], r["with_five_frames"], r["dropped_no_five_frames"],
                                                                                         r["gap_rows"], r["closing_rows"], r["direct_rows"], r["episodes"]))
        hl = A["HL_reference"]["R3_skill_ADD20_style_over_HL_copy"]
        L += ["", "Reference HL (level probe, one frame; ADD20): R3 skill over its copy %s." % _sk(hl), ""]
        for kind in kinds:
            K_ = A["kinds"][kind]
            L += ["#### %s" % kind, ""]
            if kind == "HT":
                L.append("Chosen penalty multiplier g per fold (gap / closing%s): %s" % (" / direct" if res["direct_run"] else "", "; ".join(
                    "f%s %s" % (f, " / ".join("%g" % K_["penalty_or_info"][f][tg]["g"] for tg in (("gap", "c", "dir") if res["direct_run"] else ("gap", "c")))) for f in K_["penalty_or_info"])))
            L.append("Inner-fold mean squared error (selection of the head, fit rows only): closing speed %.4f%s." % (
                K_["inner_mse_closing_mean_over_folds"], ("; direct %.4f" % K_["inner_mse_direct_mean_over_folds"]) if "inner_mse_direct_mean_over_folds" in K_ else ""))
            F = K_["fidelity"]
            L += ["", "Fidelity on held-out windows with the label history rule (n=%d):" % F["gap"]["n"],
                  "- gap at S0: R2 %+.3f %s; RMSE %.2f m %s" % (F["gap"]["r2"], _ci(F["gap"]["r2_ci95"]), F["gap"]["rmse_rmse"], _ci(F["gap"]["rmse_ci95"])),
                  "- closing speed at S0: R2 %+.3f %s; RMSE %.2f m/s %s" % (F["closing"]["r2"], _ci(F["closing"]["r2_ci95"]), F["closing"]["rmse_rmse"], _ci(F["closing"]["rmse_ci95"])), ""]
            L.append("Rules over the copy of the same head (n=%d scored windows):" % K_["rules"]["R3"]["skill_over_copy"]["n"])
            for r in ("R1", "R2", "R3"):
                row = K_["rules"][r]
                L.append("- %s: skill over the head copy %s; level R2 %+.3f %s" % (r, _sk(row["skill_over_copy"]), row["level_r2"]["r2"], _ci(row["level_r2"]["ci95"])))
            L.append("- R0 (head copy): level R2 %+.3f %s" % (K_["rules"]["R0"]["level_r2"]["r2"], _ci(K_["rules"]["R0"]["level_r2"]["ci95"])))
            d = K_["banked"]["R3"]
            L.append("- R3 level R2 %+.3f minus banked rollout (i) [R1 file] %+.3f: %+.3f %s%s (n=%d)" % (d["level_r2_rule"], d["level_r2_banked_rollout"], d["diff"], _ci(d["ci95"]),
                     " (separated above 0)" if d["separated_pos"] else (" (separated below 0)" if d["separated_neg"] else ""), d["n"]))
            if "direct" in K_:
                dd = K_["direct"]
                L.append("- Direct read (gap + predicted change over 2.0 s) skill over the head copy: %s; RMSE %.2f m %s against the copy RMSE %.2f m" % (
                    _sk(dd["skill_over_copy"]), dd["rmse_gap_2s"]["rmse"], _ci(dd["rmse_gap_2s"]["ci95"]), dd["rmse_copy_gap_2s"]["rmse"]))
            L.append("")
        if len(kinds) == 2:
            L.append("Selection by the inner folds: closing-speed head = %s%s." % (A["chosen_closing"], ("; direct head = " + A["chosen_direct"]) if res["direct_run"] else ""))
            L.append("")
    L += ["## Outcomes (card section 4; the head is the one chosen by the inner folds of the fit rows)", ""]
    if final and res["controls_pass"]:
        o = res["outcomes"]
        if not o["complete"]:
            L.append("**Provisional: not every head or the direct read has been run in this file; names below are on the heads run and are not the card's final names.**")
        for typ in ("P", "G"):
            d = o["per_type"][typ]
            if isinstance(d, str):
                L.append("- Arm type %s: %s" % (typ, d))
                continue
            L.append("- Arm type %s (both seeds needed): closing **%s**; slot **%s**%s; heads used %s; with each head run: %s" % (
                typ, d["closing"], " + ".join(d["slot"]), ("; direct **%s** (heads %s; with each head: %s)" % (d["direct"], d["direct_head_used"], d["direct_with_each_head_run"])) if "direct" in d else "; direct read not run",
                d["head_used"], json.dumps(d["same_names_with_each_head_run"])))
        L += ["", "Share of the label ceiling (R3 skill of the head over 0.750, the label R2 skill of ADD20; decides nothing):", ""]
        for arm in arms:
            for kind in kinds:
                sk = res["arms"][arm]["kinds"][kind]["rules"]["R3"]["skill_over_copy"]["skill"]
                L.append("- %s %s: %+.3f" % (arm, kind, sk / 0.7499))
    else:
        L.append("No outcome is named (%s)." % ("a control failed" if final else "partial write"))
    if res["horizon_run"] and final:
        L += ["", "## Horizon profile (decides nothing; skill over the copy of the same head)", "", "| horizon s | arm | head | R1 | R2 | R3 |", "|---|---|---|---|---|---|"]
        for K in HORIZONS[:-1]:
            for arm in arms:
                for kind in kinds:
                    h = res["arms"][arm]["kinds"][kind]["horizon"][K]
                    L.append("| %.1f | %s (n=%d) | %s | %s | %s | %s |" % (DT * K, arm, h["n"], kind, *[("%+.3f %s" % (h["rules"][r]["skill"], _ci(h["rules"][r]["ci95"]))) for r in ("R1", "R2", "R3")]))
    L += ["", "## Limits", "",
          "- The heads are fitted on bank states and applied to window states, which differ from the bank states of the same frames by a median 0.015 per element (standard deviation of a dimension 0.864); LEADHEAD_DEFS.md L18.",
          "- Fit rows are few (see the counts per fold); a negative of HT and HM is a negative for heads of this size on this bank, and the third next step of the card tests the state before pooling.",
          "- The scored set is small (429 windows, 48 episodes); one training run per seed, so an interval answers another draw of episodes and the names need both seeds.",
          "- R1, R2, R3 and the direct read use the logged ego speeds of the future frames (the action input of a rollout); no goal input, nothing of the situation classifier.",
          "- The ADD20 label ceiling uses the privileged gap history; the head sees only the grid states. The share of the ceiling compares different inputs.", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=["P_s0", "P_s1", "G_s0", "G_s1"])
    r.add_argument("--heads", nargs="+", default=["HT", "HM"])
    r.add_argument("--no-direct", action="store_true")
    r.add_argument("--no-horizon", action="store_true")
    r.add_argument("--B", type=int, default=2000)
    r.add_argument("--out-dir", default=HERE)
    a = ap.parse_args(argv)
    return run(a.arms, a.B, a.out_dir, a.heads, not a.no_direct, not a.no_horizon)


if __name__ == "__main__":
    sys.exit(main())
