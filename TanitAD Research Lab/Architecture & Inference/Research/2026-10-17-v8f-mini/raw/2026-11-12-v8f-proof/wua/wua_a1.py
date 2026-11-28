"""wua_a1.py -- card ADD155 section 1 (A1 beyond-ego probes) for ONE seed's extraction; CPU.

  python -I wua_a1.py --seed 0 [--mlp] [--controls-only]
Writes C:/Users/Admin/wua_work/a1/RESULT_A1_s<seed>[_mlp].json (+ OOF arrays npz). Labels across seeds: wua_card.py.
"""
import argparse, hashlib, json, os, sys, time, warnings
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

WORK = "C:/Users/Admin/wua_work"
EXT = f"{WORK}/ext"
LAB = f"{WORK}/labels/WUA_LABELS.npz"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
B_BOOT = 2000
H = 10                               # ego history frames before t
CONT = {"lead_gap": ["lead_gap"], "lead_closing": ["lead_closing"], "lead_rel_acc": ["lead_rel_acc"], "gap_change_2s": ["gap_change_2s"],
        "ag1_pos": ["ag1_x", "ag1_y"], "ag2_pos": ["ag2_x", "ag2_y"], "ag3_pos": ["ag3_x", "ag3_y"], "ag1_vel": ["ag1_vx", "ag1_vy"], "curv_ahead": ["curv_ahead"]}
EGO_T = {"speed": 0, "accel": 1, "yaw_rate": 2}
BIN = ("lead_present", "tl_red_vs_green")
CLS = ("lon7", "lat_cls")
MLP_T = ("lead_gap", "lead_closing", "lead_rel_acc", "gap_change_2s", "ag1_pos", "speed")
REPS = ("L1", "L2", "L3", "TOK-MEAN", "TOK-GRID")


def fold_of(clips):
    return np.array([int(hashlib.sha256(c.encode()).hexdigest()[:8], 16) % 5 for c in clips])


def zs(tr, te):
    m, s = tr.mean(0), tr.std(0) + 1e-8
    return (tr - m) / s, (te - m) / s


def inner_groups(ci_tr):
    u = np.unique(ci_tr)
    return {c: i % 3 for i, c in enumerate(np.random.default_rng(1).permutation(u))}


def fit_ridge(Xtr, ytr, ci_tr):
    g = inner_groups(ci_tr); gi = np.array([g[c] for c in ci_tr])
    best, ba = -np.inf, ALPHAS[0]
    for al in ALPHAS:
        sse = sst = 0.0
        for k in range(3):
            tr, te = gi != k, gi == k
            if tr.sum() < 5 or te.sum() < 2:
                continue
            a, b = zs(Xtr[tr], Xtr[te])
            p = Ridge(alpha=al).fit(a, ytr[tr]).predict(b).reshape(ytr[te].shape)
            sse += ((ytr[te] - p) ** 2).sum(); sst += ((ytr[te] - ytr[te].mean(0)) ** 2).sum()
        sc = 1 - sse / sst
        if sc > best:
            best, ba = sc, al
    return ba


def oof_ridge(X, y, ci, folds, only=None):
    P = np.full(y.shape, np.nan)
    for f in (range(5) if only is None else [only]):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() < 20:
            continue
        al = fit_ridge(X[tr], y[tr], ci[tr])
        a, b = zs(X[tr], X[te])
        P[te] = Ridge(alpha=al).fit(a, y[tr]).predict(b).reshape(P[te].shape)
    return P


def oof_logit(X, y, ci, folds, multi=False, only=None):
    """returns OOF class scores: P(y=1) for binary, predicted label for classes."""
    out = np.full(len(y), np.nan)
    for f in (range(5) if only is None else [only]):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        g = inner_groups(ci[tr]); gi = np.array([g[c] for c in ci[tr]])
        best, bc = np.inf, 1.0
        for al in ALPHAS:
            ll = 0.0
            for k in range(3):
                a_, b_ = gi != k, gi == k
                if len(np.unique(y[tr][a_])) < 2:
                    continue
                A, Bm = zs(X[tr][a_], X[tr][b_])
                m = LogisticRegression(C=1.0 / al, max_iter=2000).fit(A, y[tr][a_])
                pr = np.clip(m.predict_proba(Bm), 1e-9, 1)
                idx = np.searchsorted(m.classes_, y[tr][b_])
                ok = (idx < len(m.classes_)) & (m.classes_[np.minimum(idx, len(m.classes_) - 1)] == y[tr][b_])
                ll -= np.log(pr[np.arange(len(idx))[ok], idx[ok]]).sum() + 20.0 * (~ok).sum()
            if ll < best:
                best, bc = ll, 1.0 / al
        A, Bm = zs(X[tr], X[te])
        m = LogisticRegression(C=bc, max_iter=2000).fit(A, y[tr])
        out[te] = m.predict(Bm) if multi else m.predict_proba(Bm)[:, list(m.classes_).index(1.0)]
    return out


def oof_mlp(X, y, folds):
    P = np.full(y.shape, np.nan)
    for f in range(5):
        tr, te = folds != f, folds == f
        a, b = zs(X[tr], X[te])
        ym, ys = y[tr].mean(0), y[tr].std(0) + 1e-8
        m = MLPRegressor(hidden_layer_sizes=(64,), alpha=1e-3, early_stopping=True, validation_fraction=0.1, max_iter=300, random_state=0)
        p = m.fit(a, (y[tr] - ym) / ys).predict(b)
        P[te] = p.reshape(P[te].shape) * ys + ym
    return P


# ------------------------------------------------------------------ bootstrap (episode clusters = clips)
def boot_w(nclip, seed=0):
    rng = np.random.default_rng(seed)
    return rng.multinomial(nclip, np.full(nclip, 1.0 / nclip), size=B_BOOT).astype(np.float64)


def r2_stats(y, pf, pe, ci, W):
    y = y.reshape(len(y), -1); pf = pf.reshape(y.shape); pe = pe.reshape(y.shape)
    nc = W.shape[1]
    S = lambda v: np.bincount(ci, weights=v, minlength=nc)
    n = S(np.ones(len(y)))
    sy = [S(y[:, d]) for d in range(y.shape[1])]; syy = [S(y[:, d] ** 2) for d in range(y.shape[1])]
    sf = S(((y - pf) ** 2).sum(1)); se = S(((y - pe) ** 2).sum(1))

    def r2(w):
        N = w @ n
        sst = sum((w @ syy[d]) - (w @ sy[d]) ** 2 / N for d in range(y.shape[1]))
        return 1 - (w @ sf) / sst, 1 - (w @ se) / sst
    f0, e0 = r2(np.ones(nc))
    N = W @ n
    sst = sum((W @ syy[d]) - (W @ sy[d]) ** 2 / N for d in range(y.shape[1]))
    fb, eb = 1 - (W @ sf) / sst, 1 - (W @ se) / sst
    d = fb - eb
    return {"r2_full": float(f0), "r2_ego": float(e0), "delta": float(f0 - e0), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def auc_w(y, s, w):
    """weighted AUROC; w [B, n] or [n]."""
    o = np.argsort(s, kind="mergesort"); y, s = y[o], s[o]
    w = np.atleast_2d(w)[:, o]
    neg = w * (y == 0); pos = w * (y == 1)
    # ties: average rank via grouping
    _, start = np.unique(s, return_index=True)
    grp = np.repeat(np.arange(len(start)), np.diff(np.append(start, len(s))))
    gneg = np.zeros((w.shape[0], len(start)))
    np.add.at(gneg.T, grp, neg.T)
    gcum = np.cumsum(gneg, 1) - gneg
    below = gcum[:, grp] + 0.5 * gneg[:, grp]
    return (pos * below).sum(1) / (pos.sum(1) * neg.sum(1))


def auc_stats(y, sf, se, ci, W):
    Wf = W[:, ci]
    out = []
    for i in range(0, W.shape[0], 250):
        out.append(auc_w(y, sf, Wf[i:i + 250]) - auc_w(y, se, Wf[i:i + 250]))
    d = np.concatenate(out)
    f0, e0 = auc_w(y, sf, np.ones(len(y)))[0], auc_w(y, se, np.ones(len(y)))[0]
    return {"auc_full": float(f0), "auc_ego": float(e0), "delta": float(f0 - e0), "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))], "boot_nan": int(np.isnan(d).sum())}


def bacc_stats(y, pf, pe, ci, W):
    cls = np.unique(y)
    nc = W.shape[1]

    def ba(w, p):
        rec = []
        for c in cls:
            m = y == c
            tot = np.bincount(ci[m], minlength=nc).astype(float); cor = np.bincount(ci[m & (p == c)], minlength=nc).astype(float)
            rec.append((w @ cor) / np.maximum(w @ tot, 1e-9))
        return np.mean(rec, 0)
    d = ba(W, pf) - ba(W, pe)
    one = np.ones(nc)
    return {"bacc_full": float(ba(one, pf)), "bacc_ego": float(ba(one, pe)), "delta": float(ba(one, pf) - ba(one, pe)),
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--reps", default=",".join(REPS))
    a = ap.parse_args(argv)
    t0 = time.time()
    E = np.load(f"{EXT}/WUA_RH_s{a.seed}.npz")
    Lb = np.load(LAB)
    assert (E["clips"] == Lb["clips"]).all() and (E["clip_idx"] == Lb["clip_idx"]).all() and (E["t"] == Lb["t"]).all(), "extraction and labels misaligned"
    names = list(Lb["names"]); Y = Lb["Y"]
    ci_all, t_all = E["clip_idx"], E["t"].astype(int)
    clips = list(E["clips"]); folds_clip = fold_of(clips)
    v, acc, yr = E["v"], E["a"], E["yaw_rate"]
    # ego history [N, 33]; rows with t >= H only
    N = len(ci_all)
    keep = t_all >= H
    egoh = np.full((N, 3 * (H + 1)), np.nan)
    for i in np.where(keep)[0]:
        sl = slice(i - H, i + 1)
        egoh[i] = np.concatenate([v[sl], acc[sl], yr[sl]])
    keep &= np.isfinite(egoh).all(1)
    reps = {"L1": E["L1"], "L2": E["L2"], "L3": E["L3"], "TOK-MEAN": E["tokmean"]}
    want = a.reps.split(",")
    grid = np.load(f"{EXT}/WUA_RH_s{a.seed}_tokgrid.npy", mmap_mode="r") if "TOK-GRID" in want else None
    rows = np.where(keep)[0]
    ci, fo = ci_all[rows], folds_clip[ci_all[rows]]
    EG = egoh[rows]
    W = boot_w(len(clips))
    tgt = {}
    for k, cols in CONT.items():
        tgt[k] = ("cont", np.stack([Y[rows, names.index(c)] for c in cols], 1))
    for k, j in EGO_T.items():
        tgt[k] = ("cont", np.stack([(v, acc, yr)[j][rows]], 1))
    for k in BIN:
        tgt[k] = ("bin", Y[rows, names.index(k)])
    for k in CLS:
        tgt[k] = ("cls", Y[rows, names.index(k)])

    def tok_grid_feats():
        """per outer fold PCA(256) fitted on that fold's training rows; returns [n, 256] OOF-consistent features per fold (dict fold -> (Xtr_rows, X))."""
        Xg = np.zeros((len(rows), 256), np.float32)
        perfold = {}
        G = np.asarray(grid[rows], np.float32)
        for f in range(5):
            tr = fo != f
            p = PCA(n_components=256, svd_solver="randomized", random_state=0).fit(G[tr])
            perfold[f] = p.transform(G).astype(np.float32)
        del G
        return perfold

    res = {"card": "ADD155", "section": "A1", "seed": a.seed, "probe": "MLP" if a.mlp else "LIN", "n_rows": int(len(rows)), "cells": {}, "controls": {}}
    rep_feats = {}
    for r in want:
        if r == "TOK-GRID":
            rep_feats[r] = tok_grid_feats()
        else:
            rep_feats[r] = reps[r][rows].astype(np.float64)

    def feats(r, f=None):
        X = rep_feats[r]
        return X[f] if isinstance(X, dict) else X

    def temporal(r):
        """state at t-4..t from the full per-frame array (rows have t >= 10, so frames i-4..i are the same clip)."""
        full = reps[r].astype(np.float64)
        assert (ci_all[rows - 4] == ci_all[rows]).all()
        return np.concatenate([full[rows - s] for s in range(5)], 1)

    def run_cell(r, name, kind, y, mask):
        m = mask
        if m.sum() < 50 or len(np.unique(ci[m])) < 10:
            return {"n": int(m.sum()), "skipped": "n"}
        ego = EG[m]
        if isinstance(rep_feats[r], dict):          # TOK-GRID: fold-specific PCA
            out_f = np.full(y[m].shape if kind == "cont" else (m.sum(),), np.nan)
            for f in range(5):
                Xf = np.concatenate([feats(r, f)[m], ego], 1)
                if kind == "cont":
                    P = oof_ridge(Xf, y[m], ci[m], fo[m], only=f)
                else:
                    P = oof_logit(Xf, y[m], ci[m], fo[m], multi=(kind == "cls"), only=f)
                out_f[fo[m] == f] = P[fo[m] == f]
            pf = out_f
        else:
            X = np.concatenate([feats(r)[m], ego], 1)
            if a.mlp:
                X = np.concatenate([temporal(r)[m], ego], 1)
                pf = oof_mlp(X, y[m], fo[m])
            elif kind == "cont":
                pf = oof_ridge(X, y[m], ci[m], fo[m])
            else:
                pf = oof_logit(X, y[m], ci[m], fo[m], multi=(kind == "cls"))
        key = (name, kind, a.mlp)
        if key not in ego_cache:
            if a.mlp:
                ego_cache[key] = oof_mlp(ego, y[m], fo[m])
            elif kind == "cont":
                ego_cache[key] = oof_ridge(ego, y[m], ci[m], fo[m])
            else:
                ego_cache[key] = oof_logit(ego, y[m], ci[m], fo[m], multi=(kind == "cls"))
        pe = ego_cache[key]
        ok = np.isfinite(pf).reshape(len(pf), -1).all(1) & np.isfinite(pe).reshape(len(pe), -1).all(1)
        yy, cc = y[m][ok], ci[m][ok]
        if kind == "cont":
            st = r2_stats(yy, pf[ok], pe[ok], cc, W)
        elif kind == "bin":
            st = auc_stats(yy, pf[ok], pe[ok], cc, W)
        else:
            st = bacc_stats(yy, pf[ok], pe[ok], cc, W)
        st.update({"n": int(ok.sum()), "n_clips": int(len(np.unique(cc)))})
        return st

    ego_cache = {}
    targets = MLP_T if a.mlp else list(tgt)
    for r in want:
        if a.mlp and r == "TOK-GRID":
            continue
        for name in targets:
            kind, y = tgt[name]
            mask = np.isfinite(y.reshape(len(y), -1)).all(1)
            res["cells"][f"{r}|{name}"] = run_cell(r, name, kind, y, mask)
            print(f"[a1 s{a.seed}] {r} {name} {json.dumps(res['cells'][f'{r}|{name}'])[:160]} {time.time() - t0:.0f}s", flush=True)

    if not a.mlp:
        # ---------------- controls c1-c4
        rng = np.random.default_rng(7)
        nclip = len(clips)
        perm = np.arange(nclip)
        while (perm == np.arange(nclip)).any():
            perm = rng.permutation(nclip)
        lookup = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(ci, t_all[rows]))}
        src = np.array([lookup.get((int(perm[c]), int(t)), -1) for c, t in zip(ci, t_all[rows])])
        for r in [x for x in ("L1", "TOK-GRID") if x in want]:
            for name in ("lead_gap", "ag1_pos", "speed"):
                kind, y = tgt[name]
                ys = np.full(y.shape, np.nan); ok = src >= 0; ys[ok] = y[src[ok]]
                ego_cache.pop((name, kind, False), None)
                mask = np.isfinite(ys.reshape(len(ys), -1)).all(1)
                res["controls"][f"c1_shuffle|{r}|{name}"] = run_cell(r, name + "_shuf", kind, ys, mask)
        # c2 planted on L1
        if "L1" in want:
            Z = rep_feats["L1"]
            A_ = np.concatenate([EG, np.ones((len(EG), 1))], 1)
            beta, *_ = np.linalg.lstsq(A_, Z, rcond=None)
            Rz = Z - A_ @ beta
            w = np.random.default_rng(11).normal(size=Z.shape[1]); w /= np.linalg.norm(w)
            sgl = Rz @ w; sgl = (sgl - sgl.mean()) / sgl.std()
            yp = (sgl + np.random.default_rng(12).normal(size=len(sgl)))[:, None]
            res["controls"]["c2_planted|L1"] = run_cell("L1", "planted", "cont", yp, np.ones(len(yp), bool))
        # c3 noise state
        for r, dim in (("L1", 128), ("TOK-GRID", 256)):
            if r not in want:
                continue
            rep_feats["NOISE"] = np.random.default_rng(13).normal(size=(len(rows), dim))
            for name in [k for k, (kd, _) in tgt.items() if kd == "cont"]:
                kind, y = tgt[name]
                mask = np.isfinite(y.reshape(len(y), -1)).all(1)
                res["controls"][f"c3_noise{dim}|{name}"] = run_cell("NOISE", name, kind, y, mask)
        # c4 alignment
        kind, y = tgt["speed"]
        pe = oof_ridge(EG, y, ci, fo)
        res["controls"]["c4_ego_speed_r2"] = float(1 - ((y - pe) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    res["seconds"] = round(time.time() - t0, 1)
    res["ext_sha256"] = json.load(open(f"{EXT}/WUA_RH_s{a.seed}.json"))["npz_sha256"]
    res["tool_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    os.makedirs(f"{WORK}/a1", exist_ok=True)
    out = f"{WORK}/a1/RESULT_A1_s{a.seed}{'_mlp' if a.mlp else ''}{'' if a.reps == ','.join(REPS) else '_' + a.reps.replace(',', '+')}.json"
    json.dump(res, open(out, "w"), indent=1)
    print("WROTE", out, res["seconds"])


if __name__ == "__main__":
    main()
