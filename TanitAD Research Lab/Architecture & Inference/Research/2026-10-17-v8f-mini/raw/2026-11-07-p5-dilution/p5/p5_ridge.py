"""p5_ridge: BESIDE, DECIDES NOTHING (THEORY_CRAWL_AMEND_3.md section 4, last bullet): the R^2 of a (the SG +-0.5 s acceleration) on the FROZEN pooled tokens of the present frame,
a ridge fit on the train folds only and scored out of fold (5 folds, grouped by episode). Tokens = the train cache's ``pool4x10`` [T, 40, 1024] fp16 (the frozen encoder's pooled output the model's adapter reads),
frame t of each window of the 20,000-window sample. Two reductions of the 40 tokens (dev-box RAM): mean token (1,024-d) and the 4 row-means of the 4 x 10 grid (4,096-d).
Also: R^2 of a0 (reference), R^2 of the part of a that a0 does not carry (a - b a0, b the train-fold OLS slope), and a shuffled-target control (R^2 must be ~0).
usage: p5_ridge.py extract | fit        (CPU only, <= 4 BLAS threads)
"""
import os
import sys
import json
import time

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import p5_lib as L

SCR = L.HERE + "/scratch"


def extract():
    import torch
    torch.set_num_threads(2)
    t0 = time.time()
    os.makedirs(SCR, exist_ok=True)
    tab = np.load(L.HERE + "/raw/window_table_band.npz")
    rows = np.load(L.HERE + "/raw/p5_sample_rows.npy")
    clips, ci, t = tab["clips"], tab["clip_idx"][rows], tab["t"][rows]
    XA = np.zeros((len(rows), 1024), np.float16)
    XB = np.zeros((len(rows), 4 * 1024), np.float16)
    done = np.zeros(len(rows), bool)
    order = np.argsort(ci, kind="stable")
    cur, o = None, None
    for n, j in enumerate(order):
        if ci[j] != cur:
            cur = ci[j]
            o = torch.load(f"{L.TRAIN_CACHE}/train/clips/{clips[cur]}.pt", map_location="cpu", mmap=True, weights_only=True)
            g = o["pool4x10"]
            assert g.shape[1] == 40 and g.shape[2] == 1024, g.shape
        f = g[int(t[j])].float()                                  # [40, 1024]
        XA[j] = f.mean(0).numpy().astype(np.float16)
        XB[j] = f.reshape(4, 10, 1024).mean(1).reshape(-1).numpy().astype(np.float16)     # the 4 x 10 grid: rows of 10 tokens (token index = row * 10 + col; checked against the cache builder's reshape: UNVERIFIED, both reductions are linear in the same tokens)
        done[j] = True
        if n % 4000 == 0:
            print("extract", n, f"{time.time()-t0:.0f}s", flush=True)
    assert done.all()
    np.save(SCR + "/XA_mean_token.npy", XA)
    np.save(SCR + "/XB_row_means.npy", XB)
    print("extract done", f"{time.time()-t0:.0f}s", flush=True)


def oof_r2_multi(X, Ys, groups, folds=5, alphas=None):
    """Ridge on standardised features, several targets sharing one X. Per outer fold (grouped by episode): standardise on the TRAIN fold, pick alpha per target on one inner grouped hold-out of
    the train fold (eigendecomposition of the inner-train covariance), refit on the whole train fold, predict the held-out fold. Nothing of the held-out fold enters the fit or the alpha."""
    from sklearn.model_selection import GroupKFold
    alphas = np.asarray(alphas if alphas is not None else np.logspace(0, 6, 13))
    names = list(Ys)
    pred = {k: np.zeros(len(X)) for k in names}
    chosen = {k: [] for k in names}
    for tr, te in GroupKFold(n_splits=folds).split(X, groups=groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = ((X[tr] - mu) / sd).astype(np.float64), ((X[te] - mu) / sd).astype(np.float64)
        itr, iva = next(GroupKFold(n_splits=5).split(Xtr, groups=groups[tr]))
        mi = Xtr[itr].mean(0)
        Xi = Xtr[itr] - mi
        w, V = np.linalg.eigh(Xi.T @ Xi)
        Xv = Xtr[iva] - mi
        for k in names:
            y = Ys[k][tr]
            yi = y[itr]
            b = V.T @ (Xi.T @ (yi - yi.mean()))
            mse = [float(np.mean((y[iva] - (Xv @ (V @ (b / (w + al))) + yi.mean())) ** 2)) for al in alphas]
            chosen[k].append(float(alphas[int(np.argmin(mse))]))
        mt = Xtr.mean(0)
        Xc = Xtr - mt
        w, V = np.linalg.eigh(Xc.T @ Xc)
        Xe = Xte - mt
        for k, al in ((k, chosen[k][-1]) for k in names):
            y = Ys[k][tr]
            b = V.T @ (Xc.T @ (y - y.mean()))
            pred[k][te] = Xe @ (V @ (b / (w + al))) + y.mean()
    out = {}
    for k in names:
        y = Ys[k]
        out[k] = {"r2_oof": float(1 - ((y - pred[k]) ** 2).sum() / ((y - y.mean()) ** 2).sum()), "alpha_chosen": chosen[k], "pred": pred[k]}
    return out


def boot_r2(y, pred, groups, B=1000, seed=0):
    uniq, inv = np.unique(groups, return_inverse=True)
    s_res = np.bincount(inv, weights=(y - pred) ** 2, minlength=len(uniq))
    s_y, s_y2, n = (np.bincount(inv, weights=w, minlength=len(uniq)) for w in (y, y * y, np.ones(len(y))))
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        p = rng.choice(len(uniq), size=len(uniq), replace=True)
        N, Sy, Sy2, Sr = n[p].sum(), s_y[p].sum(), s_y2[p].sum(), s_res[p].sum()
        out[b] = 1 - Sr / (Sy2 - Sy * Sy / N)
    return [float(x) for x in np.percentile(out, [2.5, 97.5])]


def fit():
    t0 = time.time()
    tab = np.load(L.HERE + "/raw/window_table_band.npz")
    rows = np.load(L.HERE + "/raw/p5_sample_rows.npy")
    a, a0 = tab["a_pm0.5s"][rows], tab["a0_model"][rows]
    groups = tab["clip_idx"][rows]
    res = {"n_windows": int(len(rows)), "n_episodes": int(len(np.unique(groups))), "folds": 5, "estimator": "ridge, features standardised on the train fold, alpha from 13 values 1..1e6 chosen on one inner grouped hold-out of the train fold, refit on the whole train fold; R^2 pooled over out-of-fold predictions; episode-cluster bootstrap B 1,000 of the pooled R^2",
           "target": "a = SG order 2 +-0.5 s at the present tick", "frame": "present frame t of each window (pool4x10 tokens of the train cache)"}
    # partial target: the part of a that a0 does not carry (slope from the whole sample; descriptive, a global OLS slope)
    b = float(np.cov(a0, a)[0, 1] / a0.var())
    a_res = a - b * a0
    rng = np.random.default_rng(0)
    for nm, fn in (("A_mean_token_1024d", "XA_mean_token.npy"), ("B_row_means_4096d", "XB_row_means.npy")):
        X = np.load(f"{SCR}/{fn}").astype(np.float32)
        Ys = {"a": a, "a0_reference": a0, "a_minus_b_a0": a_res, "CONTROL_shuffled_a": a[rng.permutation(len(a))]}
        o = oof_r2_multi(X, Ys, groups)
        r = {}
        for tn, y in Ys.items():
            r[tn] = {"r2_oof": o[tn]["r2_oof"], "ci95": boot_r2(y, o[tn]["pred"], groups), "alpha_chosen": o[tn]["alpha_chosen"]}
            print(nm, tn, f"{r[tn]['r2_oof']:.4f}", r[tn]["ci95"], f"{time.time()-t0:.0f}s", flush=True)
        res[nm] = r
        json.dump(res, open(L.HERE + "/raw/P5_RIDGE.json", "w"), indent=1)          # partial result banked after each reduction
    res["b_slope_a_on_a0"] = b
    res["wall_s"] = time.time() - t0
    json.dump(res, open(L.HERE + "/raw/P5_RIDGE.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"extract": extract, "fit": fit}[sys.argv[1]]()
