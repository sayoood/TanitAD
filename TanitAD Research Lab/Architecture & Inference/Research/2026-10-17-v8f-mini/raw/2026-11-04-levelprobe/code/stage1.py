#!/usr/bin/env python3
"""stage1.py -- Stage 1 of PREREG_LEVELPROBE.md (H-LEVELPROBE-1), AS WRITTEN: can the logged speed at t0 be read from the frozen DINOv3 image grid of the t0 frame?  Zero training of any model but one ridge.
CPU only.  Features: probe 3's G8 token grid ONLY (pooled to G2x4, 8,192-d; ERRATA_1 E1), no ego, no nav.  Writes raw/STAGE1.json, raw/stage1_pred_eval608.npz.
python -I code/stage1.py            (needs FROZEN_SHA.txt with the card's sha)
"""
import hashlib
import json
import os
import sys
import time

TH = "16"
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = TH
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

ART = "E:/Projects/TanitAD-artifacts"
PK = f"{ART}/dagland/levelprobe"
SCR = "F:/TanitAD-scratch/levelprobe_scratch"
P3 = "F:/TanitAD-scratch/trigger_probe3_scratch"
RAW1 = f"{ART}/dagland/trigger_probe/raw"
GD_DIR = f"{ART}/dagland/gapdecomp"
D_IN, D_OUT = 163840, 8192
CS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
SEED_K1 = 1


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def pool_g2x4(x):
    """[n, 163840] (any float dtype; row-major 8x20 tokens x 1024) -> [n, 8192] float32: the mean of each 4-row x 5-column token block, order (block row, block column, channel).  ERRATA_1 E1."""
    x = np.asarray(x, np.float32)
    n = x.shape[0]
    assert x.shape == (n, D_IN), x.shape
    return x.reshape(n, 2, 4, 4, 5, 1024).mean(axis=(2, 4)).reshape(n, D_OUT)


def pool_reference(row):
    """Independent loop implementation for ONE window [163840] -> [8192] (float64 accumulation, compared with a tolerance)."""
    g = np.asarray(row, np.float64).reshape(8, 20, 1024)
    out = np.zeros((2, 4, 1024))
    for br in range(2):
        for bc in range(4):
            acc = np.zeros(1024)
            for r in range(4 * br, 4 * br + 4):
                for c in range(5 * bc, 5 * bc + 5):
                    acc += g[r, c]
            out[br, bc] = acc / 20.0
    return out.reshape(-1)


def pool_file(path, rows=None, chunk=400):
    mm = np.load(path, mmap_mode="r")
    assert mm.shape[1] == D_IN and mm.dtype == np.float16, (path, mm.shape, mm.dtype)
    n = mm.shape[0] if rows is None else len(rows)
    out = np.zeros((n, D_OUT), np.float32)
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        blk = mm[sl] if rows is None else mm[rows[sl]]
        out[sl] = pool_g2x4(blk)
    return out


def metrics(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    if len(y) == 0:
        return {"n": 0}
    e = p - y
    sst = float(((y - y.mean()) ** 2).sum())
    return {"n": int(len(y)), "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()), "r2": float(1 - (e ** 2).sum() / sst) if sst > 0 else None, "y_mean": float(y.mean()), "y_std": float(y.std())}


class Ridge:
    """Primal ridge with the card's alpha grid: alpha = c * trace(A^T A over the BASE features) / D, intercept unpenalised.  One eigh of the fold's Gram serves every alpha (and, when only the labels change, every label vector)."""

    def __init__(self, X, extra=None, idx=None, base_trace_cols=D_OUT):
        Xi = X if idx is None else X[idx]
        self.mu = Xi.mean(0, dtype=np.float64).astype(np.float32)
        A = Xi - self.mu
        self.nb = int(base_trace_cols)
        self.scale = float((A.astype(np.float64) ** 2).sum() / self.nb)       # trace(A^T A) / D over the base features
        if extra is not None:
            ex = extra if idx is None else extra[idx]
            self.emu = ex.mean(0).astype(np.float32)
            A = np.hstack([A, (ex - self.emu).astype(np.float32)])
        else:
            self.emu = None
        self.A = A
        G = (A.T @ A).astype(np.float64)
        self.lam, self.V = sla.eigh(G, driver="evd")
        del G
        self.n = A.shape[0]

    def path(self, y, Xte, extra_te=None):
        """-> {c: predictions on Xte} for every c in CS (labels y centred inside)."""
        ybar = float(np.mean(y))
        b = (self.A.T @ (np.asarray(y, np.float32) - np.float32(ybar))).astype(np.float64)
        t = self.V.T @ b
        B = Xte - self.mu
        if extra_te is not None:
            B = np.hstack([B, (extra_te - self.emu).astype(np.float32)])
        out = {}
        for c in CS:
            al = c * self.scale
            w = (self.V @ (t / (self.lam + al))).astype(np.float32)
            out[c] = (B @ w).astype(np.float64) + ybar
        return out


def fit_eval_multi(X, ys, folds, Xe, extra=None, extra_e=None, tag=""):
    """ys = {name: label vector}.  ONE eigh per fold (and one for the final fit) serves every label vector: the Gram does not depend on the labels.
    Per name: episode-grouped 5-fold CV on the TRAIN windows -> pooled out-of-fold RMSE per c -> best c -> refit on all TRAIN windows -> predictions on Xe.  Returns {name: (pred, c, {c: rmse})}."""
    t0 = time.time()
    n = len(folds)
    oof = {k: {c: np.zeros(n) for c in CS} for k in ys}
    for f in range(5):
        te, tr = np.where(folds == f)[0], np.where(folds != f)[0]
        R = Ridge(X, extra, tr)
        for k, y in ys.items():
            pr = R.path(y[tr], X[te], None if extra is None else extra[te])
            for c in CS:
                oof[k][c][te] = pr[c]
        del R
    best, rms = {}, {}
    for k, y in ys.items():
        rms[k] = {c: float(np.sqrt(((oof[k][c] - y) ** 2).mean())) for c in CS}
        best[k] = min(rms[k], key=rms[k].get)
    R = Ridge(X, extra, None)
    out = {}
    for k, y in ys.items():
        out[k] = (R.path(y, Xe, extra_e)[best[k]], best[k], rms[k])
        print(f"[{tag}:{k}] chosen c={best[k]} cv_rmse={rms[k][best[k]]:.4f}", flush=True)
    print(f"[{tag}] wall {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    fz = {ln.split()[0]: ln.split()[2] for ln in open(f"{PK}/FROZEN_SHA.txt", encoding="utf-8") if " sha256 " in ln}
    want, got = fz["PREREG_LEVELPROBE.md"], sha256_file(f"{PK}/PREREG_LEVELPROBE.md")
    if want != got:
        print(f"REFUSED: PREREG_LEVELPROBE.md {got} != frozen {want}", file=sys.stderr)
        return 2
    os.makedirs(SCR, exist_ok=True)
    res = {"hypothesis": "H-LEVELPROBE-1", "stage": 1, "tier": "T0-class probe (frozen features of logged frames); NOT driving performance", "evidence_class": "MEASURED (ours; artifacts in MANIFEST.txt)",
           "prereg_sha256": got, "feature": "G8 token grid pooled to G2x4 (mean of 4x5-token blocks), 8,192-d float32, ERRATA_1 E1", "alpha_grid_c": list(CS), "run_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "controls": {}}
    # ---- windows: labels, clips, folds (only the named columns are read)
    W = np.load(f"{RAW1}/windows_train.npz", allow_pickle=False)
    v_all, clip_i, sha12, clips = W["v0"], W["clip_i"], W["episode_sha12"], W["clips"]
    idx = np.concatenate([np.load(f"{RAW1}/sample_{p}.npz")["idx"] for p in ("pstop", "pmove")])
    y = v_all[idx].astype(np.float64)
    ci = clip_i[idx]
    folds = np.array([int(str(sha12[c]), 16) % 5 for c in ci])
    train_clips = {str(clips[c]) for c in ci}
    E = np.load(f"{RAW1}/eval141_features.npz", allow_pickle=False)
    sel, ye, eval_clips = E["sel"], E["v0"].astype(np.float64), {str(c) for c in E["clip"]}
    assert len(idx) == 24000 and len(ye) == 608 and y.min() >= 0 and y.max() < 6.0, (len(idx), len(ye), y.min(), y.max())
    assert not (train_clips & eval_clips), "REFUSED: fit and score episode sets overlap"
    res["assertions"] = {"n_train": int(len(idx)), "n_eval": int(len(ye)), "train_label_range": [float(y.min()), float(y.max())], "n_train_clips": len(train_clips), "n_eval_clips": len(eval_clips),
                         "train_eval_clip_sets_disjoint": True, "fold_sizes": [int((folds == f).sum()) for f in range(5)], "columns_read_windows_train": ["v0", "clip_i", "episode_sha12", "clips"],
                         "columns_read_eval141_features": ["sel", "v0", "clip"], "ego_nav_columns_loaded_into_fit": False}
    # ---- features (cached on F:)
    ft, fe = f"{SCR}/X_g2x4_train24000.npy", f"{SCR}/X_g2x4_eval608.npy"
    t0 = time.time()
    if not os.path.exists(ft):
        X = np.concatenate([pool_file(f"{P3}/G8_pstop.npy"), pool_file(f"{P3}/G8_pmove.npy")])
        np.save(ft, X)
    X = np.load(ft)
    if not os.path.exists(fe):
        np.save(fe, pool_file(f"{P3}/G8_eval141.npy"))
    Xe = np.load(fe)
    assert X.shape == (24000, D_OUT) and Xe.shape == (608, D_OUT) and np.isfinite(X).all() and np.isfinite(Xe).all()
    # pooling known answer: the fast implementation equals an independent loop on 3 train and 3 eval rows
    mmt, mme = np.load(f"{P3}/G8_pstop.npy", mmap_mode="r"), np.load(f"{P3}/G8_eval141.npy", mmap_mode="r")
    dev = max(float(np.abs(X[r] - pool_reference(mmt[r])).max()) for r in (0, 5000, 11999))
    dev = max(dev, max(float(np.abs(Xe[r] - pool_reference(mme[r])).max()) for r in (0, 300, 607)))
    res["controls"]["POOL_equals_independent_loop"] = {"max_abs_dev": dev, "PASS": bool(dev <= 1e-4)}
    res["feature_sha256"] = {"X_train": sha256_file(ft), "X_eval": sha256_file(fe)}
    print(f"features ready {time.time() - t0:.0f}s pool dev {dev:.2e}", flush=True)
    # ---- the rig's classes and M1 windows on the 608 eval windows
    sys.path.insert(0, GD_DIR)
    import gapdecomp as GD  # noqa: E402
    z = np.load(f"{ART}/dagland/kappa_t1/raw/arrays/SREP_DAGOACC_s0.npz", allow_pickle=True)
    cls_all, _ = GD.classify(z["v0"], z["g"])
    i18 = [str(q) for q in z["labels"]].index("M1 crawl share")
    m1_all = np.isfinite(z[f"a_{i18}"]) & np.isfinite(z[f"b_{i18}"])
    assert np.array_equal(z["v0"][sel], ye)
    cls_e, m1_e = cls_all[sel], m1_all[sel]
    subsets = {"all": np.ones(len(sel), bool), "M1_windows": m1_e, "non_M1_windows": ~m1_e}
    for k, nm in enumerate(GD.CLASSES):
        subsets[f"class_{nm}"] = cls_e == k

    def score(p):
        return {k: metrics(ye[m], p[m]) for k, m in subsets.items()}

    # ---- the main fit
    ysh = np.random.default_rng(SEED_K1).permutation(y)
    mk = fit_eval_multi(X, {"main": y, "K1": ysh}, folds, Xe, tag="MAIN+K1")
    pr, c, rm = mk["main"]
    res["main"] = {"chosen_c": c, "cv_rmse_by_c": {str(k): v for k, v in rm.items()}, "chosen_c_at_grid_edge": bool(c in (CS[0], CS[-1])), "eval": score(pr),
                   "pred_range": [float(pr.min()), float(pr.max())], "train_label_mean": float(y.mean())}
    np.savez(f"{PK}/raw/stage1_pred_eval608.npz", sel=sel, pred=pr, v0=ye, c=c)
    # ---- K3: the train-mean predictor
    pm = np.full(len(ye), y.mean())
    res["controls"]["K3_train_mean_baseline"] = {"eval": score(pm), "note": "reported baseline; R2 here is against the eval mean, so it is <= 0"}
    res["main"]["skill_vs_train_mean_rmse"] = {k: (None if res["controls"]["K3_train_mean_baseline"]["eval"][k]["n"] == 0 else 1 - res["main"]["eval"][k]["rmse"] / res["controls"]["K3_train_mean_baseline"]["eval"][k]["rmse"]) for k in subsets}
    # ---- K1: labels shuffled within the TRAIN set (same pipeline incl. the alpha search)
    p1, c1, rm1 = mk["K1"]
    r2k1 = score(p1)["all"]["r2"]
    res["controls"]["K1_shuffled_train_labels"] = {"seed": SEED_K1, "chosen_c": c1, "eval_all": score(p1)["all"], "threshold": "eval R2 <= 0.02", "PASS": bool(r2k1 <= 0.02)}
    # ---- K2: v_logged appended (scaled so its variance equals the mean squared feature norm)
    Xc = X - X.mean(0, dtype=np.float64).astype(np.float32)
    sc = float(np.sqrt((Xc.astype(np.float64) ** 2).sum() / len(X)))
    del Xc
    zt, ze = ((y - y.mean()) / y.std()), ((ye - y.mean()) / y.std())
    ex_t, ex_e = (sc * zt)[:, None].astype(np.float32), (sc * ze)[:, None].astype(np.float32)
    p2, c2, rm2 = fit_eval_multi(X, {"K2": y}, folds, Xe, extra=ex_t, extra_e=ex_e, tag="K2")["K2"]
    r2k2 = score(p2)["all"]["r2"]
    res["controls"]["K2_v_logged_appended"] = {"column_scale": sc, "chosen_c": c2, "eval_all": score(p2)["all"], "threshold": "eval R2 >= 0.99", "PASS": bool(r2k2 >= 0.99)}
    res["controls_all_pass"] = bool(all(v.get("PASS", True) for v in res["controls"].values()))
    json.dump(res, open(f"{PK}/raw/STAGE1.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    m = res["main"]["eval"]
    print("STAGE1 main eval all:", {k: round(v, 4) for k, v in m["all"].items() if isinstance(v, float)}, "| M1:", {k: round(v, 4) for k, v in m["M1_windows"].items() if isinstance(v, float)})
    print("controls:", {k: v.get("PASS") for k, v in res["controls"].items()}, "ALL PASS" if res["controls_all_pass"] else "A CONTROL FAILED")
    return 0 if res["controls_all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
