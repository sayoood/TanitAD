"""H-REFAV1-TRUNK-TRAJ -- the pre-registered ridge probe (SPEC.md beside the banked copy of this file).

Arms P0..P5 exactly as SPEC s3; outer 5-fold grouped-by-episode CV (fold = episode index mod 5); (lambda, PCA m)
chosen by inner 4-fold grouped CV on the TRAINING folds only, on ADE. Writes out-of-fold stride-4 dumps per arm
(the refav1_paired_delta format) for the 2 s families, and a 6 s ADE/FDE table with the paired bootstrap.
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, "C:/Users/Admin/refav1_fullgrid/analysis")
from build_explore import retime          # the SAME re-timing used for kd_x on the stride-40 grid

FEAT = "C:/Users/Admin/refav1_probe/feat_s4"
OUT = "C:/Users/Admin/refav1_probe/out"
LAMS = np.logspace(-2, 4, 13)
MS = (8, 16, 32, 64, 128)
N_OUTER, N_INNER = 5, 4


def load():
    files = sorted(glob.glob(f"{FEAT}/ep*.npz"))
    cols = {}
    for fi, f in enumerate(files):
        z = np.load(f)
        n = len(z["ws"])
        for k in ("trunk", "raw1", "raw2", "raw3", "kin", "g", "g6", "g6_ok", "v0", "ha", "ha0", "ha0_ext",
                  "ol", "ha0_6", "ha0_ext_6", "ws"):
            cols.setdefault(k, []).append(z[k])
        cols.setdefault("ep", []).append(np.full(n, fi))
        cols.setdefault("eid", []).append(np.full(n, int(z["eid"][0])))
        cols.setdefault("clip_index", []).append(np.full(n, int(z["clip_index"][0])))
        cols.setdefault("fname", []).append(np.array([os.path.basename(f)] * n))
    return {k: np.concatenate(v) for k, v in cols.items()}


def kdx(ha0, ha0_ext):
    damp = 0.5 * ha0 + 0.5 * ha0_ext
    return np.stack([retime(damp[i], ha0_ext[i]) for i in range(len(damp))])


def kin_feats(kin):
    v, a, k = kin[:, 0], kin[:, 1], kin[:, 2]
    return np.stack([v, a, k, v * v, v * a, v * k, v * v * k, a * k], 1)


def ade(pred, g):
    return np.linalg.norm(pred - g, axis=-1).mean(-1)


class Readout:
    """standardise -> (optional) PCA on the image block -> ridge with intercept; every statistic is fit on the
    rows given. The SVD is computed ONCE per row set and shared by every (lambda, m) of the grid."""

    def __init__(self, img, kf):
        self.kmu, self.ksd = kf.mean(0), kf.std(0) + 1e-8
        self.has_img = img is not None
        if self.has_img:
            self.imu, self.isd = img.mean(0), img.std(0) + 1e-6
            _, _, vt = np.linalg.svd((img - self.imu) / self.isd, full_matrices=False)
            self.vt = vt[: max(MS)]

    def design(self, img, kf, m):
        blocks = [(kf - self.kmu) / self.ksd]
        if self.has_img:
            blocks.append(((img - self.imu) / self.isd) @ self.vt[:m].T)
        return np.concatenate(blocks, 1)

    @staticmethod
    def ridge(X, y, lam):
        xmu, ymu = X.mean(0), y.mean(0)
        Xc, yc = X - xmu, y - ymu
        W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
        return xmu, ymu, W

    def fit(self, img, kf, y, lam, m):
        X = self.design(img, kf, m)
        self.lam, self.m, self.d = lam, m, X.shape[1]
        self.xmu, self.ymu, self.W = self.ridge(X, y, lam)
        return self

    def predict(self, img, kf):
        return (self.design(img, kf, self.m) - self.xmu) @ self.W + self.ymu


def fit_select(img, kf, y, base, g, groups, use_img, rng=None):
    """inner grouped CV on (img, kf, y) -> best (lam, m); refit on all rows given."""
    ug = np.unique(groups)
    inner = {e: i % N_INNER for i, e in enumerate(ug)}
    fold = np.array([inner[e] for e in groups])
    ms = MS if use_img else (0,)
    score = {(l, m): [] for l in LAMS for m in ms}
    for f in range(N_INNER):
        tr, te = fold != f, fold == f
        ro = Readout(img[tr] if use_img else None, kf[tr])
        for m in ms:
            Xtr = ro.design(img[tr] if use_img else None, kf[tr], m)
            Xte = ro.design(img[te] if use_img else None, kf[te], m)
            for lam in LAMS:
                xmu, ymu, W = Readout.ridge(Xtr, y[tr], lam)
                yh = (Xte - xmu) @ W + ymu
                score[(lam, m)].append(ade(base[te] + yh.reshape(-1, *base.shape[1:]), g[te]))
    best = min(score, key=lambda k_: float(np.concatenate(score[k_]).mean()))
    ro = Readout(img if use_img else None, kf).fit(img if use_img else None, kf, y, *best)
    return ro, best


def main():
    os.makedirs(OUT, exist_ok=True)
    D = load()
    n = len(D["ws"])
    base2 = kdx(D["ha0"], D["ha0_ext"]).astype(np.float64)
    base6 = kdx(D["ha0_6"], D["ha0_ext_6"]).astype(np.float64)
    g2, g6, ok6 = D["g"].astype(np.float64), D["g6"].astype(np.float64), D["g6_ok"].astype(bool)
    kf = kin_feats(D["kin"].astype(np.float64))
    trunk = D["trunk"].astype(np.float64)
    raw = np.concatenate([D["raw1"], D["raw2"], D["raw3"]], 1).astype(np.float64)
    y2 = (g2 - base2).reshape(n, -1)
    y6 = np.where(ok6[:, None, None], g6 - base6, 0.0).reshape(n, -1)
    outer = D["ep"] % N_OUTER
    arms = ["P0_kdx", "P1_const", "P2_kin", "P3_raw", "P4_trunk", "P5_shuf"]
    pred2 = {a: np.zeros_like(g2) for a in arms}
    pred6 = {a: np.zeros_like(g6) for a in arms}
    meta = {"n_windows": int(n), "n_episodes": int(len(np.unique(D["ep"]))), "n_6s_ok": int(ok6.sum()),
            "d_trunk": int(trunk.shape[1]), "d_raw": int(raw.shape[1]), "d_kin": int(kf.shape[1]),
            "folds": {}}
    rng = np.random.default_rng(0)
    for f in range(N_OUTER):
        tr, te = outer != f, outer == f
        fm = {"n_train": int(tr.sum()), "n_test": int(te.sum())}
        for a in arms:
            if a == "P0_kdx":
                pred2[a][te] = base2[te]; pred6[a][te] = base6[te]; continue
            if a == "P1_const":
                pred2[a][te] = base2[te] + y2[tr].mean(0).reshape(1, 10, 2)
                t6 = tr & ok6
                pred6[a][te] = base6[te] + y6[t6].mean(0).reshape(1, -1, 2)
                continue
            img = {"P2_kin": None, "P3_raw": raw, "P4_trunk": trunk, "P5_shuf": trunk}[a]
            use_img = img is not None
            img_tr = None
            if use_img:
                img_tr = img[tr]
                if a == "P5_shuf":
                    img_tr = img_tr[rng.permutation(len(img_tr))]
            r2, sel2 = fit_select(img_tr, kf[tr], y2[tr], base2[tr], g2[tr], D["ep"][tr], use_img)
            pred2[a][te] = base2[te] + r2.predict(img[te] if use_img else None, kf[te]).reshape(-1, 10, 2)
            t6 = tr & ok6
            img6 = None if not use_img else (img[t6] if a != "P5_shuf" else img[t6][rng.permutation(int(t6.sum()))])
            r6, sel6 = fit_select(img6, kf[t6], y6[t6], base6[t6], g6[t6], D["ep"][t6], use_img)
            pred6[a][te] = base6[te] + r6.predict(img[te] if use_img else None, kf[te]).reshape(-1, g6.shape[1], 2)
            fm[a] = {"sel_2s": [float(sel2[0]), int(sel2[1])], "d_2s": int(r2.d),
                     "sel_6s": [float(sel6[0]), int(sel6[1])], "d_6s": int(r6.d), "n_train_6s": int(t6.sum())}
        meta["folds"][f] = fm
        print(f"[probe] fold {f} done: {json.dumps(fm)[:300]}", flush=True)
    # known-value control: P0 must be kd_x exactly
    meta["control_P0_equals_kdx_max_abs"] = float(np.abs(pred2["P0_kdx"] - base2).max())
    # dumps in the refav1_paired_delta format (stride-4 grid), one dir per arm
    for a in arms:
        dd = f"{OUT}/dump_{a}"
        os.makedirs(dd, exist_ok=True)
        for fn in np.unique(D["fname"]):
            s = D["fname"] == fn
            np.savez(f"{dd}/{fn}", g=D["g"][s], v0=D["v0"][s].astype(np.float32),
                     cl=pred2[a][s].astype(np.float32), ha=D["ha"][s], ha0=D["ha0"][s], ha0_ext=D["ha0_ext"][s],
                     ol=D["ol"][s], ws=D["ws"][s].astype(np.int64), eid=D["eid"][s][:1].astype(np.int64),
                     clip_index=D["clip_index"][s][:1].astype(np.int64))
    # 6 s ADE / FDE, paired episode-cluster bootstrap (same estimator as the 2 s tool)
    sys.path.insert(0, "C:/Users/Admin/tipsnap/b3f7ea6f/taniteval")
    from taniteval.ci import paired_episode_cluster_bootstrap as pb
    eid6 = D["ep"][ok6]
    ade6 = {a: ade(pred6[a][ok6], g6[ok6]) for a in arms}
    fde6 = {a: np.linalg.norm(pred6[a][ok6][:, -1] - g6[ok6][:, -1], axis=-1) for a in arms}
    ade2 = {a: ade(pred2[a], g2) for a in arms}
    pairs = [("P4_trunk", "P2_kin"), ("P4_trunk", "P0_kdx"), ("P5_shuf", "P2_kin"), ("P1_const", "P2_kin"),
             ("P3_raw", "P2_kin"), ("P4_trunk", "P3_raw"), ("P2_kin", "P0_kdx"), ("P0_kdx", "P0_kdx")]
    res = {"meta": meta, "means_2s_ade": {a: float(v.mean()) for a, v in ade2.items()},
           "means_6s_ade": {a: float(v.mean()) for a, v in ade6.items()},
           "means_6s_fde": {a: float(v.mean()) for a, v in fde6.items()}, "pairs_6s": {}, "pairs_2s_ade": {}}
    for x, y in pairs:
        res["pairs_6s"][f"{x}-{y}"] = {"ade": pb(ade6[x], ade6[y], eid6), "fde": pb(fde6[x], fde6[y], eid6)}
        res["pairs_2s_ade"][f"{x}-{y}"] = pb(ade2[x], ade2[y], D["ep"])
    json.dump(res, open(f"{OUT}/probe_result.json", "w"), indent=1)
    print("ZZPROBE-DONEZZ", json.dumps({k: res[k] for k in ("means_2s_ade", "means_6s_ade")}), flush=True)


if __name__ == "__main__":
    main()
