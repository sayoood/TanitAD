"""wua_a1_perm.py -- card ADD155 Amendment 3: refit-under-permutation reference for A1 (P-LIN), one seed x one target, all five representations.

  python -I wua_a1_perm.py --seed 0 --target lead_gap   -> C:/Users/Admin/wua_work/a1/PERM_A1_s<seed>_<target>.json
Draw -1 = the observed labels (must equal RESULT_A1's point value: the alignment check); draws 0..18 = clip derangements with rng seeds 100..118.
"""
import argparse, hashlib, json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np
from sklearn.decomposition import PCA
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wua_a1 as A1   # noqa: E402

R_DRAWS = 19
TGT = {"lead_gap": ("cont", ["lead_gap"]), "ag1_pos": ("cont", ["ag1_x", "ag1_y"]), "ag2_pos": ("cont", ["ag2_x", "ag2_y"]), "lead_present": ("bin", ["lead_present"])}


def derange(n, seed):
    rng = np.random.default_rng(seed)
    p = np.arange(n)
    while (p == np.arange(n)).any():
        p = rng.permutation(n)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--target", required=True, choices=list(TGT))
    a = ap.parse_args(argv)
    t0 = time.time()
    E = np.load(f"{A1.EXT}/WUA_RH_s{a.seed}.npz")
    Lb = np.load(A1.LAB)
    assert (E["clip_idx"] == Lb["clip_idx"]).all() and (E["t"] == Lb["t"]).all()
    names = list(Lb["names"]); Y = Lb["Y"]
    ci_all, t_all = E["clip_idx"], E["t"].astype(int)
    clips = list(E["clips"]); fclip = A1.fold_of(clips)
    v, acc, yr = E["v"], E["a"], E["yaw_rate"]
    H = A1.H
    keep = t_all >= H
    egoh = np.full((len(ci_all), 3 * (H + 1)), np.nan)
    for i in np.where(keep)[0]:
        egoh[i] = np.concatenate([v[i - H:i + 1], acc[i - H:i + 1], yr[i - H:i + 1]])
    keep &= np.isfinite(egoh).all(1)
    rows = np.where(keep)[0]
    ci, fo, EG = ci_all[rows], fclip[ci_all[rows]], egoh[rows]
    kind, cols = TGT[a.target]
    y0 = np.stack([Y[rows, names.index(c)] for c in cols], 1)
    if kind == "bin":
        y0 = y0[:, 0]
    lookup = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(ci, t_all[rows]))}
    reps = {r: E[k][rows].astype(np.float64) for r, k in (("L1", "L1"), ("L2", "L2"), ("L3", "L3"), ("TOK-MEAN", "tokmean"))}
    gc = f"{A1.WORK}/a1/gridpca_s{a.seed}.npz"           # per-fold PCA features, deterministic (random_state 0); cached so parallel jobs need not hold the 423 MB grid
    if os.path.exists(gc):
        z = np.load(gc); gridf = {f: z[f"f{f}"] for f in range(5)}
        assert (z["rows"] == rows).all()
    else:
        G = np.asarray(np.load(f"{A1.EXT}/WUA_RH_s{a.seed}_tokgrid.npy", mmap_mode="r")[rows], np.float32)
        gridf = {f: PCA(n_components=256, svd_solver="randomized", random_state=0).fit(G[fo != f]).transform(G).astype(np.float32) for f in range(5)}
        del G
        np.savez(gc + ".tmp.npz", rows=rows, **{f"f{f}": gridf[f] for f in range(5)})
        os.replace(gc + ".tmp.npz", gc)

    def score(y, pf, pe):
        if kind == "cont":
            yy = y.reshape(len(y), -1); sst = ((yy - yy.mean(0)) ** 2).sum()
            return float((1 - ((yy - pf.reshape(yy.shape)) ** 2).sum() / sst) - (1 - ((yy - pe.reshape(yy.shape)) ** 2).sum() / sst))
        return float(A1.auc_w(y, pf, np.ones(len(y)))[0] - A1.auc_w(y, pe, np.ones(len(y)))[0])

    def fit(X, y, m, only=None):
        if kind == "cont":
            return A1.oof_ridge(X, y[m], ci[m], fo[m], only=only)
        return A1.oof_logit(X, y[m], ci[m], fo[m], only=only)
    out = {"card": "ADD155", "amendment": 3, "seed": a.seed, "target": a.target, "draws": {}}
    for d in [-1] + list(range(R_DRAWS)):
        if d < 0:
            y = y0
        else:
            perm = derange(len(clips), 100 + d)
            src = np.array([lookup.get((int(perm[c]), int(t)), -1) for c, t in zip(ci, t_all[rows])])
            y = np.full(y0.shape, np.nan); ok = src >= 0; y[ok] = y0[src[ok]]
        m = np.isfinite(y.reshape(len(y), -1)).all(1)
        pe = fit(EG[m], y, m)
        res = {}
        for r in ("L1", "L2", "L3", "TOK-MEAN", "TOK-GRID"):
            if r == "TOK-GRID":
                pf = np.full(pe.shape, np.nan)
                for f in range(5):
                    P = fit(np.concatenate([gridf[f][m], EG[m]], 1), y, m, only=f)
                    sel = fo[m] == f
                    pf[sel] = P[sel]
            else:
                pf = fit(np.concatenate([reps[r][m], EG[m]], 1), y, m)
            ok2 = np.isfinite(pf.reshape(len(pf), -1)).all(1) & np.isfinite(pe.reshape(len(pe), -1)).all(1)
            res[r] = score(y[m][ok2], pf[ok2], pe[ok2])
        out["draws"][str(d)] = res
        print(f"[perm s{a.seed} {a.target}] draw {d} {json.dumps({k: round(x, 4) for k, x in res.items()})} {time.time() - t0:.0f}s", flush=True)
    obs = out["draws"]["-1"]
    out["observed"] = obs
    out["perm_max"] = {r: max(out["draws"][str(d)][r] for d in range(R_DRAWS)) for r in obs}
    out["exceeds_all"] = {r: bool(obs[r] > out["perm_max"][r]) for r in obs}
    A1r = json.load(open(f"{A1.WORK}/a1/RESULT_A1_s{a.seed}.json"))["cells"]
    out["alignment_vs_RESULT_A1"] = {r: abs(obs[r] - A1r[f"{r}|{a.target}"]["delta"]) for r in obs}
    out["am3_sha256"] = hashlib.sha256(open("F:/TanitAD-scratch/v8proof/PREREG_V8F_PROOF_ADD155_AM3.md", "rb").read()).hexdigest()
    out["tool_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{A1.WORK}/a1/PERM_A1_s{a.seed}_{a.target}.json", "w"), indent=1)
    print("WROTE", out["exceeds_all"], out["alignment_vs_RESULT_A1"])


if __name__ == "__main__":
    main()
