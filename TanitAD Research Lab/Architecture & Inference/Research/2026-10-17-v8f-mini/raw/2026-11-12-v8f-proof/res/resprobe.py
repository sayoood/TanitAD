"""RESPROBE: PREREG_V8F_PROOF_ADD1 (frozen 2026-11-13 01:40), hypothesis H-V8F-PROOF-4, theorem T5. Tier T0: a frozen-feature probe on
logged frames, a perception diagnostic, NOT driving performance. CPU only, <= 4 threads. Subcommands:
    build   features -> res/cache/ (two Gram matrices K40, K160 of the flattened tokens, f8 ego vector, labels)  [~5 min, job]
    run     probes, controls, bootstrap -> RESULT_RESPROBE.json + VERDICT_RESPROBE.md                          [job]
Implementation choices that the card leaves open are listed in IMPL (bottom of this docstring block, printed into the verdict):
  * unit of analysis: one bank frame = (episode, cache step g) of the LATSEM2 bank (labels_latsem2.npz), tokens at step g (the frame NOW);
  * PCA = kernel PCA on the Gram matrix of the flattened fp8->fp32 tokens, fitted on the training folds only (centred on the train mean),
    rank 256, randomized subspace iteration (q=4); test scores are the exact out-of-sample projection;
  * P1 ridge on the scores (alpha by an inner episode split, 6 values); P2 one hidden layer of 256 ReLU units, Adam, early stopping on the inner split;
  * +E: the 8 standardised f8 features, scaled to the mean score variance, appended to the scores;
  * the headline probe of a vision set is P1; the ego-only control E takes the better of P1/P2 per target (card). P2-headline label printed too;
  * R2 and AUROC are POOLED over the five out-of-fold predictions; the bootstrap resamples the 139 episodes (clusters) with paired counts.
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
ART = "E:/Projects/TanitAD-artifacts"
FR_CACHE = f"{ART}/sfresh_inputs/cache/eval/clips"
TRACKS = f"{ART}/sfresh_inputs/inputs/native_tracks.pt"
LABELS = f"{ART}/dagland/motion_c/theory/latsem2/labels_latsem2.npz"
SEED = 0
B_BOOT = 2000
N_FOLDS = 5
RANK = 256
GAP_EDGES = (6.3, 12.5, 25.0)
TARGETS = {"GAP": ("lead_gap", "r2"), "NEAR": ("nearest", "r2"), "COUNT": ("agents30", "r2"), "LEADP": ("lead_present", "auc")}
SETS = ("E", "V40", "V160", "V40+E", "V160+E")
PROBES = ("P1", "P2")


def set_threads(n=4):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(n)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


# ----------------------------------------------------------------------------- folds, metrics, bootstrap (pure numpy; unit-tested)
def make_folds(eid, k=N_FOLDS, seed=SEED):
    """Fold index per frame; the unit is the EPISODE (every frame of an episode lies in one fold)."""
    u = np.array(sorted(set(map(str, eid))))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(u))
    fold_of = {u[p]: i % k for i, p in enumerate(perm)}
    return np.array([fold_of[str(e)] for e in eid], np.int64)


def r2_pooled(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def auc_pooled(y, s):
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.empty(len(s)); o = np.argsort(s, kind="mergesort"); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_counts(n_ep, B=B_BOOT, seed=SEED):
    """[B, n_ep] multinomial episode-resampling counts, the SAME draw for every paired difference."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_ep, np.full(n_ep, 1.0 / n_ep), size=B).astype(np.float64)


def ep_stats_r2(ep, y, p, n_ep):
    """Per-episode sufficient statistics [n_ep, 4]: n, sum y, sum y^2, SSE (rows with finite y and p only)."""
    m = np.isfinite(y) & np.isfinite(p)
    S = np.zeros((n_ep, 4))
    e, yy, pp = ep[m], y[m], p[m]
    np.add.at(S[:, 0], e, 1.0); np.add.at(S[:, 1], e, yy); np.add.at(S[:, 2], e, yy * yy); np.add.at(S[:, 3], e, (yy - pp) ** 2)
    return S


def boot_r2(S, C):
    t = C @ S                                       # [B, 4]
    sst = t[:, 2] - t[:, 1] ** 2 / np.maximum(t[:, 0], 1)
    return 1.0 - t[:, 3] / sst


def boot_auc(ep, y, s, C):
    """Weighted AUROC for every bootstrap draw (episode counts as case weights); scores assumed continuous (ties ignored)."""
    m = np.isfinite(y) & np.isfinite(s)
    e, yy, ss = ep[m], y[m], s[m]
    o = np.argsort(ss, kind="mergesort")
    e, yy = e[o], yy[o]
    W = C[:, e]                                     # [B, n]
    wpos, wneg = W * (yy == 1), W * (yy == 0)
    cneg = np.cumsum(wneg, axis=1) - wneg
    return (wpos * cneg).sum(1) / (wpos.sum(1) * wneg.sum(1))


def ci(d):
    lo, hi = np.percentile(d, [2.5, 97.5])
    return [float(lo), float(hi)]


def diff_row(a_boot, b_boot, a_pt, b_pt):
    d = a_boot - b_boot
    c = ci(d)
    return {"diff": float(a_pt - b_pt), "ci95": c, "separated_pos": bool(c[0] > 0), "separated_neg": bool(c[1] < 0)}


# ----------------------------------------------------------------------------- kernel PCA + probes
def kpca_fit(K, tr, rank=RANK, q=4, seed=SEED):
    """Kernel PCA on the Gram matrix K (all frames), fitted on the rows ``tr`` only. Returns a projector f(idx) -> scores [len(idx), rank]
    (train rows give the exact PCA scores, other rows the exact out-of-sample projection)."""
    Ktr = np.asarray(K[np.ix_(tr, tr)], np.float64)
    n = len(tr)
    cm = Ktr.mean(0); gm = cm.mean()
    Kc = Ktr - cm[:, None] - cm[None, :] + gm
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(Kc @ rng.standard_normal((n, rank + 24)))[0]
    for _ in range(q):
        Q = np.linalg.qr(Kc @ Q)[0]
    lam, V = np.linalg.eigh(Q.T @ Kc @ Q)
    o = np.argsort(lam)[::-1][:rank]
    lam, U = np.maximum(lam[o], 1e-12), Q @ V[:, o]
    inv = 1.0 / np.sqrt(lam)

    def project(idx):
        Kx = np.asarray(K[np.ix_(idx, tr)], np.float64)
        Kxc = Kx - Kx.mean(1)[:, None] - cm[None, :] + gm
        return Kxc @ U * inv[None, :]
    return project, lam


def ridge_solve(Z, y, alphas):
    """Ridge with intercept for several alphas on centred Z: returns list of (alpha, w, b)."""
    mu, ym = Z.mean(0), y.mean()
    Zc, yc = Z - mu, y - ym
    A, c = Zc.T @ Zc, Zc.T @ yc
    sc = np.trace(A) / A.shape[0]
    out = []
    for a in alphas:
        w = np.linalg.solve(A + a * sc * np.eye(A.shape[0]), c)
        out.append((a, w, ym - mu @ w))
    return out


ALPHAS = (1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def inner_split(ep_tr, seed=SEED, frac=0.2):
    u = np.array(sorted(set(map(str, ep_tr))))
    rng = np.random.default_rng(seed + 17)
    val_eps = set(rng.choice(u, max(1, int(round(frac * len(u)))), replace=False))
    return np.array([str(e) in val_eps for e in ep_tr])


def fit_p1(Ztr, ytr, ep_tr):
    iv = inner_split(ep_tr)
    sols = ridge_solve(Ztr[~iv], ytr[~iv], ALPHAS)
    best = min(sols, key=lambda s: float(((ytr[iv] - (Ztr[iv] @ s[1] + s[2])) ** 2).mean()))
    a, w, b = ridge_solve(Ztr, ytr, [best[0]])[0]
    return lambda Z: Z @ w + b


def fit_p2(Ztr, ytr, ep_tr, binary, seed=SEED, hidden=256, max_epochs=80, patience=8, bs=256):
    import torch
    torch.manual_seed(seed)
    iv = inner_split(ep_tr)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    ym, ys = (0.0, 1.0) if binary else (float(ytr.mean()), float(ytr.std() + 1e-8))
    X = torch.tensor((Ztr - mu) / sd, dtype=torch.float32); Y = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss() if binary else torch.nn.MSELoss()
    itr, iva = torch.tensor(np.flatnonzero(~iv)), torch.tensor(np.flatnonzero(iv))
    best, best_sd, bad = 1e30, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(max_epochs):
        net.train()
        for ib in itr[torch.randperm(len(itr), generator=g)].split(bs):
            opt.zero_grad(); lossf(net(X[ib]).squeeze(1), Y[ib]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(lossf(net(X[iva]).squeeze(1), Y[iva]))
        if v < best - 1e-6:
            best, bad = v, 0
            best_sd = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_sd); net.eval()

    def pred(Z):
        with torch.no_grad():
            return net(torch.tensor((Z - mu) / sd, dtype=torch.float32)).squeeze(1).numpy().astype(np.float64) * ys + ym
    return pred


def assemble(S, Ef, with_e):
    """Feature matrix: PCA scores (+ the standardised f8 block scaled to the mean score variance)."""
    return S if not with_e else np.concatenate([S, Ef], 1)


def e_block(E, tr, scale_var):
    mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-9
    return (E - mu) / sd * np.sqrt(scale_var)


def run_probe_cv(F_by_fold, y, ep, folds, probe, binary, valid):
    """F_by_fold[f] = (Ftr_all, Fte_all, tr_idx, te_idx) feature matrices for the training/test frames of fold f; returns pooled OOF predictions."""
    oof = np.full(len(y), np.nan)
    for f, (Ftr, Fte, tr, te) in enumerate(F_by_fold):
        mtr, mte = valid[tr], valid[te]
        fit = fit_p1(Ftr[mtr], y[tr][mtr], ep[tr][mtr]) if probe == "P1" else fit_p2(Ftr[mtr], y[tr][mtr], ep[tr][mtr], binary, seed=SEED + f)
        oof[te[mte]] = fit(Fte[mte])
    return oof


# ----------------------------------------------------------------------------- build
def cmd_build(a):
    set_threads(a.threads)
    import torch
    torch.set_num_threads(a.threads)
    sys.path[:0] = [p for p in os.environ.get("PYTHONPATH", "").split(";") if p]
    import tanitad
    root = os.path.abspath(os.environ["TREE_ROOT"]).replace("\\", "/").lower()
    assert os.path.abspath(tanitad.__file__).replace("\\", "/").lower().startswith(root), tanitad.__file__
    from tanitad.data import v8f_mini_data as D
    os.makedirs(CACHE, exist_ok=True)
    L = np.load(LABELS)
    beid, bg = L["beid"].astype(str), L["bg"].astype(np.int64)
    tr = D.load_native_tracks(TRACKS)
    fk = D._fk()
    N = len(beid)
    kin = np.full((N, 3), np.nan)
    order = {}
    for i, (e, g) in enumerate(zip(beid, bg)):
        order.setdefault(e, []).append(i)
    clips = sorted(order)
    n_use = len(clips) if a.max_clips <= 0 else a.max_clips
    clips = clips[:n_use]
    sel = np.array(sorted(i for c in clips for i in order[c]))
    X40 = np.zeros((len(sel), 40 * 1024), np.float16); X160 = np.zeros((len(sel), 160 * 1024), np.float16)
    pos = {int(i): j for j, i in enumerate(sel)}
    t0 = time.time()
    for ci_, c in enumerate(clips):
        o = torch.load(f"{FR_CACHE}/{c}.pt", map_location="cpu", weights_only=True, mmap=True)
        rows = np.array(order[c]); gs = bg[rows]
        assert gs.max() < o["pool4x10"].shape[0], (c, gs.max())
        a40 = o["pool4x10"][torch.as_tensor(gs)].to(torch.float32).reshape(len(gs), -1).numpy()
        a160 = o["pool8x20"][torch.as_tensor(gs)].to(torch.float32).reshape(len(gs), -1).numpy()
        k3 = D.yaw_kin_from_track(tr[c])[0]
        for r, i in enumerate(rows):
            X40[pos[int(i)]] = a40[r]; X160[pos[int(i)]] = a160[r]; kin[i] = k3[bg[i]]
        del o
        if ci_ % 20 == 0:
            print(f"[build] {ci_}/{len(clips)} clips {time.time() - t0:.0f}s", flush=True)
    kk = kin[sel]
    E = fk.kin_features(kk).astype(np.float64)
    np.savez(f"{CACHE}/frames.npz", sel=sel, beid=beid[sel], bg=bg[sel], E=E, **{q: L[q][sel] for q in ("lead_present", "lead_gap", "agents30", "nearest")})
    for name, X in (("K40", X40), ("K160", X160)):
        K = np.zeros((len(sel), len(sel)), np.float32)
        step = 8192
        for s0 in range(0, X.shape[1], step):
            Xb = torch.from_numpy(X[:, s0:s0 + step].astype(np.float32))
            K += (Xb @ Xb.T).numpy()
        np.save(f"{CACHE}/{name}.npy", K)
        print(f"[build] {name} {K.shape} {time.time() - t0:.0f}s", flush=True)
    print("[build] done", flush=True)


# ----------------------------------------------------------------------------- run
def cmd_run(a):
    set_threads(a.threads)
    import torch
    torch.set_num_threads(a.threads)
    t0 = time.time()
    Z = np.load(f"{CACHE}/frames.npz")
    beid = Z["beid"].astype(str); E = Z["E"]
    keep = np.isfinite(E).all(1)
    idx_keep = np.flatnonzero(keep)
    beid = beid[keep]; E = E[keep]
    Y = {t: Z[v][keep].astype(np.float64) for t, (v, _) in TARGETS.items()}
    gap_valid_stats = {}
    Ks = {"V40": np.load(f"{CACHE}/K40.npy", mmap_mode="r"), "V160": np.load(f"{CACHE}/K160.npy", mmap_mode="r")}
    Ks = {k: np.asarray(v[np.ix_(idx_keep, idx_keep)]) for k, v in Ks.items()}
    uep = sorted(set(beid)); n_ep = len(uep); epi = np.array([uep.index(e) for e in beid])
    folds = make_folds(beid)
    C = boot_counts(n_ep)
    N = len(beid)
    # fold feature matrices per grid
    FS = {}
    for g in ("V40", "V160"):
        fl = []
        for f in range(N_FOLDS):
            te = np.flatnonzero(folds == f); tr = np.flatnonzero(folds != f)
            proj, lam = kpca_fit(Ks[g], tr)
            Str, Ste = proj(tr), proj(te)
            fl.append((Str, Ste, tr, te, float(Str.var(0).mean())))
        FS[g] = fl
        print(f"[run] kpca {g} {time.time() - t0:.0f}s", flush=True)

    def fold_feats(setname):
        out = []
        for f in range(N_FOLDS):
            te = np.flatnonzero(folds == f); tr = np.flatnonzero(folds != f)
            if setname == "E":
                mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-9
                out.append(((E[tr] - mu) / sd, (E[te] - mu) / sd, tr, te)); continue
            g = setname[:4] if setname.startswith("V160") else setname[:3]
            g = "V160" if setname.startswith("V160") else "V40"
            Str, Ste, _, _, sv = FS[g][f]
            if setname.endswith("+E"):
                Eb_tr, Eb_te = e_block(E, tr, sv)[tr], e_block(E, tr, sv)[te]
                out.append((np.concatenate([Str, Eb_tr], 1), np.concatenate([Ste, Eb_te], 1), tr, te))
            else:
                out.append((Str, Ste, tr, te))
        return out
    FF = {s: fold_feats(s) for s in SETS}
    rng = np.random.default_rng(SEED + 99)
    res = {"meta": {"n_frames": int(N), "n_episodes": int(n_ep), "folds": N_FOLDS, "B": B_BOOT, "rank": RANK, "tier": "T0 perception diagnostic, frozen-feature probe on logged frames, not driving performance",
                    "STRATEGIC": "PAUSED (PI decision queue item 25, 2026-09-19): not a pass, not omitted, not extended to the other families"},
           "oof": {}, "metric": {}, "k1": {}, "k1_perfold_mean": {}, "k2": {}}
    OOF = {}
    for tname, (_, kind) in TARGETS.items():
        y = Y[tname]; valid = np.isfinite(y); binary = kind == "auc"
        res["meta"][f"n_{tname}"] = int(valid.sum()); res["meta"][f"n_ep_{tname}"] = int(len(set(epi[valid])))
        for s in SETS:
            for p in PROBES:
                if s == "E" or True:
                    oof = run_probe_cv(FF[s], y, epi, folds, p, binary, valid)
                    # k1: targets shuffled within fold
                    ysh = y.copy()
                    for f in range(N_FOLDS):
                        te = np.flatnonzero((folds == f) & valid); ysh[te] = y[rng.permutation(te)]
                    oofk = run_probe_cv(FF[s], ysh, epi, folds, p, binary, valid)
                    OOF[(tname, s, p)] = oof
                    mt = (auc_pooled if binary else r2_pooled)
                    res["metric"][f"{tname}|{s}|{p}"] = mt(y, oof)
                    res["k1"][f"{tname}|{s}|{p}"] = mt(ysh, oofk)
                    pf = [mt(np.where(folds == f, ysh, np.nan), oofk) for f in range(N_FOLDS)]
                    res["k1_perfold_mean"][f"{tname}|{s}|{p}"] = float(np.nanmean(pf))
                    print(f"[run] {tname} {s} {p} {res['metric'][f'{tname}|{s}|{p}']:.4f} k1 {res['k1'][f'{tname}|{s}|{p}']:.4f} {time.time() - t0:.0f}s", flush=True)
    np.savez(f"{HERE}/OOF_RESPROBE.npz", beid=beid, **{"|".join(k): v for k, v in OOF.items()}, **{"y_" + t: v for t, v in Y.items()})
    # k2: synthetic target = a fixed random combination of the top-8 GLOBAL kernel PCs of V40 (a fixed linear function of the V40 tokens)
    projg, _ = kpca_fit(Ks["V40"], np.arange(N), rank=8)
    y2 = projg(np.arange(N)) @ np.random.default_rng(SEED + 5).standard_normal(8)
    oof2 = np.full(N, np.nan)
    for f in range(N_FOLDS):
        Str, Ste, tr, te, _ = FS["V40"][f]
        fit = fit_p1(Str, y2[tr], epi[tr]); oof2[te] = fit(Ste)
    res["k2"] = {"r2_P1": r2_pooled(y2, oof2), "bar": 0.95}
    k1_ok = all((v <= 0.02) if not k.startswith("LEADP") else abs(v - 0.5) <= 0.03 for k, v in res["k1"].items())
    k2_ok = res["k2"]["r2_P1"] >= 0.95
    res["controls"] = {"k1_all_within_bar": bool(k1_ok), "k2_ge_0.95": bool(k2_ok), "k1_worst": {k: v for k, v in res["k1"].items() if ((v > 0.02) if not k.startswith("LEADP") else abs(v - 0.5) > 0.03)}}
    # point + bootstrap comparisons
    def best_e(tname):
        kind = TARGETS[tname][1]
        sc = {p: res["metric"][f"{tname}|E|{p}"] for p in PROBES}
        return max(sc, key=sc.get)
    def boot_of(tname, s, p, mask=None):
        y = Y[tname]; oof = OOF[(tname, s, p)]
        yy = y if mask is None else np.where(mask, y, np.nan)
        if TARGETS[tname][1] == "auc":
            return boot_auc(epi, yy, oof, C), auc_pooled(yy, oof)
        return boot_r2(ep_stats_r2(epi, yy, oof, n_ep), C), r2_pooled(yy, oof)
    cmp_ = {}
    for tname in TARGETS:
        pe = best_e(tname); eb = boot_of(tname, "E", pe)
        cmp_[tname] = {"E_best_probe": pe}
        for hp in PROBES:
            blk = {}
            for g in ("V40", "V160"):
                vb = boot_of(tname, g + "+E", hp)
                blk[f"{g}+E minus E"] = diff_row(vb[0], eb[0], vb[1], eb[1])
            v40, v160 = boot_of(tname, "V40+E", hp), boot_of(tname, "V160+E", hp)
            blk["V160+E minus V40+E"] = diff_row(v160[0], v40[0], v160[1], v40[1])
            cmp_[tname][f"headline_{hp}"] = blk
    res["compare"] = cmp_
    # GAP strata
    ygap = Y["GAP"]; lo_edges = (0.0,) + GAP_EDGES; hi_edges = GAP_EDGES + (1e9,)
    strata = {}
    for lo, hi in zip(lo_edges, hi_edges):
        m = np.isfinite(ygap) & (ygap >= lo) & (ygap < hi)
        strata[f"[{lo},{hi})"] = {"n": int(m.sum()), "share": float(m.sum() / max(1, np.isfinite(ygap).sum()))}
    res["gap_distribution"] = strata
    near = np.isfinite(ygap) & (ygap < 12.5); far = np.isfinite(ygap) & (ygap >= 12.5)
    pe = best_e("GAP"); st = {}
    for hp in PROBES:
        for g in ("V40", "V160"):
            b = {}
            for nm, m in (("near<12.5", near), ("far>=12.5", far)):
                ev, vv = boot_of("GAP", "E", pe, m), boot_of("GAP", g + "+E", hp, m)
                b[nm] = {"gain_pt": float(vv[1] - ev[1]), "gain_ci": ci(vv[0] - ev[0]), "R2_V+E": float(vv[1]), "R2_E": float(ev[1])}
                b[nm]["_boot"] = vv[0] - ev[0]
            d = b["far>=12.5"]["_boot"] - b["near<12.5"]["_boot"]
            pt = b["far>=12.5"]["gain_pt"] - b["near<12.5"]["gain_pt"]
            for nm in b: b[nm].pop("_boot")
            b["far_minus_near_gain"] = {"diff": float(pt), "ci95": ci(d), "far_smaller": bool(pt < 0 and ci(d)[1] < 0)}
            st[f"{hp}|{g}"] = b
    res["gap_strata"] = st
    # outcomes
    out = {}
    for hp in PROBES:
        c = cmp_["GAP"][f"headline_{hp}"]
        p1 = bool(c["V160+E minus V40+E"]["separated_pos"])
        p2 = all(st[f"{hp}|{g}"]["far_minus_near_gain"]["far_smaller"] for g in ("V40", "V160"))
        absent = all(not c[f"{g}+E minus E"]["separated_pos"] for g in ("V40", "V160"))
        refuted = st[f"{hp}|V40"]["far>=12.5"]["gain_pt"] >= 0.20 and st[f"{hp}|V40"]["far>=12.5"]["R2_V+E"] - st[f"{hp}|V40"]["far>=12.5"]["R2_E"] >= 0.20
        if refuted: lab = "RES-REFUTED"
        elif p1 and p2: lab = "RES-CONFIRMED"
        elif p1 or p2: lab = "RES-PARTIAL"
        elif absent: lab = "GAP-ABSENT"
        else: lab = "UNLABELLED-BY-CARD (V beats E on GAP but neither P4a-1 nor P4a-2 holds)"
        p3 = {t: {"gain_pt": cmp_[t][f"headline_{hp}"]["V40+E minus E"]["diff"], "ci95": cmp_[t][f"headline_{hp}"]["V40+E minus E"]["ci95"],
                  "holds_ci_gt0_and_ge_0.05": bool(cmp_[t][f"headline_{hp}"]["V40+E minus E"]["separated_pos"] and cmp_[t][f"headline_{hp}"]["V40+E minus E"]["diff"] >= 0.05)} for t in ("LEADP", "COUNT")}
        out[hp] = {"label": lab, "P4a-1": p1, "P4a-2": bool(p2), "P4a-3": p3, "GAP-ABSENT": bool(absent), "RES-REFUTED": bool(refuted)}
    res["outcome"] = out
    res["meta"]["runtime_s"] = time.time() - t0
    json.dump(res, open(f"{HERE}/RESULT_RESPROBE.json", "w", encoding="utf-8"), indent=1)
    write_verdict(res)
    print("[run] done", flush=True)


def write_verdict(res):
    m = res["meta"]; o = res["outcome"]; L = []
    L.append("# VERDICT_RESPROBE (PREREG_V8F_PROOF_ADD1, H-V8F-PROOF-4, theorem T5)\n")
    L.append("Tier T0: a frozen-feature probe on logged frames. This is a perception diagnostic, not driving performance.")
    L.append("STRATEGIC family: PAUSED (PI decision queue item 25, 2026-09-19); not a pass, not omitted.\n")
    L.append(f"MEASURED, RESULT_RESPROBE.json (F:/TanitAD-scratch/v8proof/res/), tier T0: n frames {m['n_frames']}, episodes {m['n_episodes']}, folds {m['folds']}, B {m['B']}.\n")
    L.append("## Card outcome label (headline probe class in brackets)")
    for hp in PROBES:
        L.append(f"* [{hp}] **{o[hp]['label']}** (P4a-1 {o[hp]['P4a-1']}, P4a-2 {o[hp]['P4a-2']}, GAP-ABSENT {o[hp]['GAP-ABSENT']}, REFUTED-bar {o[hp]['RES-REFUTED']})")
    L.append("\nThe P1 label is the primary one (the card lists P1 first); the ego-only control E uses the better of P1/P2 per target.")
    L.append(f"\n## Controls\n* k1 (shuffled targets) all within bar: {res['controls']['k1_all_within_bar']}; offenders: {json.dumps(res['controls']['k1_worst'])}\n* k2 (synthetic fixed linear function of V40 = random combination of its top-8 global kernel PCs, P1) R2 {res['k2']['r2_P1']:.4f}, bar 0.95: {res['controls']['k2_ge_0.95']}")
    if not (res["controls"]["k1_all_within_bar"] and res["controls"]["k2_ge_0.95"]):
        L.append("* A control missed its bar: the read above is NOT admissible until the control is explained.")
    L.append("* k1 per-fold-mean variant (secondary, same predictions): worst R2 " + f"{max(v for k, v in res['k1_perfold_mean'].items() if not k.startswith('LEADP')):.3f}" + ", AUROC range " + f"[{min(v for k, v in res['k1_perfold_mean'].items() if k.startswith('LEADP')):.3f}, {max(v for k, v in res['k1_perfold_mean'].items() if k.startswith('LEADP')):.3f}]" + " (a pooled metric over folds with different base rates sits below chance by construction when targets are shuffled within folds)")
    L.append("\n## GAP far-stratum (>= 12.5 m) vision gain over E, P1 (REFUTED bar: V40+E gain >= 0.20)")
    for g in ("V40", "V160"):
        b = res["gap_strata"][f"P1|{g}"]["far>=12.5"]
        L.append(f"* {g}+E: R2 {b['R2_V+E']:.3f}, E {b['R2_E']:.3f}, gain {b['gain_pt']:+.3f} CI {b['gain_ci']}")
    L.append("\n## GAP true-gap distribution (lead present within 40 m)")
    for k, v in res["gap_distribution"].items(): L.append(f"* {k} m: n {v['n']}, share {v['share']:.3f}")
    L.append("\n## Metrics (pooled out-of-fold; R2, AUROC for LEADP)")
    for t in TARGETS:
        L.append(f"* {t}: " + "; ".join(f"{s}/{p} {res['metric'][f'{t}|{s}|{p}']:.3f}" for s in SETS for p in PROBES))
    L.append("\n## Paired episode-cluster bootstrap differences (headline P1)")
    for t in TARGETS:
        for k, v in res["compare"][t]["headline_P1"].items():
            L.append(f"* {t} {k}: {v['diff']:+.4f} CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    L.append("\n## GAP gain by stratum (P1)")
    for g in ("V40", "V160"):
        b = res["gap_strata"][f"P1|{g}"]
        L.append(f"* {g}+E minus E: near<12.5 {b['near<12.5']['gain_pt']:+.3f} {b['near<12.5']['gain_ci']}; far>=12.5 {b['far>=12.5']['gain_pt']:+.3f} {b['far>=12.5']['gain_ci']}; far minus near {b['far_minus_near_gain']['diff']:+.3f} CI {b['far_minus_near_gain']['ci95']}")
    L.append("\nNext lever, committed for every outcome: un-pooled 16x40 tokens and a native-resolution centre crop on the pod (P4a full, P4b, P4c).")
    L.append("\nImplementation choices the card leaves open (kernel PCA rank 256, ridge alpha by inner split, MLP 256 units, E scaling, P4a-3 'well above' read as CI above 0 and gain >= 0.05, REFUTED needs the far-stratum R2 gain of V40+E over E >= 0.20) are in the docstring of resprobe.py.")
    open(os.path.join(HERE, "VERDICT_RESPROBE.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-clips", type=int, default=0)
    a = ap.parse_args()
    {"build": cmd_build, "run": cmd_run}[a.cmd](a)
