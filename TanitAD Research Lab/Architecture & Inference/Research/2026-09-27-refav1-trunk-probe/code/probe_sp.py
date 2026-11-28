"""SPEC Amendment A1 -- the SPATIAL-pooling arms P6 / P7 / P8 against the same P2 (and P4 for reference).

Identical folds, lambda/m grid, inner CV and estimator as probe.py (imported, not re-derived); only the image
features change: 4x10 region means (40 x 1024) of the trunk field at t0 (P6), of the raw DINOv3 field at t0 and
t0-1 (P7, 2 x 40 x 1024), and P6 with rows permuted across training windows (P8). PCA by the Gram trick
(d >> n): the basis is fit on the training rows only, exactly as in probe.py.
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, "C:/Users/Admin/refav1_probe")
import probe as P                                     # LAMS, MS, N_OUTER, N_INNER, kdx, kin_feats, ade, load

FEAT_SP = "C:/Users/Admin/refav1_probe/feat_s4sp"
OUT = "C:/Users/Admin/refav1_probe/out_sp"


class GramReadout(P.Readout):
    """P.Readout with the image PCA computed from the n x n Gram matrix (identical basis, d >> n)."""

    def __init__(self, img, kf):
        self.kmu, self.ksd = kf.mean(0), kf.std(0) + 1e-8
        self.has_img = img is not None
        if self.has_img:
            self.imu = img.mean(0)
            self.isd = img.std(0) + 1e-6
            z = ((img - self.imu) / self.isd).astype(np.float32)
            G = (z @ z.T).astype(np.float64)
            ev, U = np.linalg.eigh(G)
            order = np.argsort(ev)[::-1][: max(P.MS)]
            ev, U = np.clip(ev[order], 1e-12, None), U[:, order]
            self.vt = ((z.T @ U.astype(np.float32)) / np.sqrt(ev).astype(np.float32)).T.astype(np.float64)

    def design(self, img, kf, m):
        blocks = [(kf - self.kmu) / self.ksd]
        if self.has_img:
            z = ((img - self.imu) / self.isd).astype(np.float32)
            blocks.append((z @ self.vt[:m].T.astype(np.float32)).astype(np.float64))
        return np.concatenate(blocks, 1)


def fit_select(img, kf, y, base, g, groups, use_img):
    ug = np.unique(groups)
    inner = {e: i % P.N_INNER for i, e in enumerate(ug)}
    fold = np.array([inner[e] for e in groups])
    ms = P.MS if use_img else (0,)
    score = {(l, m): [] for l in P.LAMS for m in ms}
    for f in range(P.N_INNER):
        tr, te = fold != f, fold == f
        ro = GramReadout(img[tr] if use_img else None, kf[tr])
        for m in ms:
            Xtr = ro.design(img[tr] if use_img else None, kf[tr], m)
            Xte = ro.design(img[te] if use_img else None, kf[te], m)
            for lam in P.LAMS:
                xmu, ymu, W = P.Readout.ridge(Xtr, y[tr], lam)
                yh = (Xte - xmu) @ W + ymu
                score[(lam, m)].append(P.ade(base[te] + yh.reshape(-1, *base.shape[1:]), g[te]))
    best = min(score, key=lambda k_: float(np.concatenate(score[k_]).mean()))
    ro = GramReadout(img if use_img else None, kf).fit(img if use_img else None, kf, y, *best)
    return ro, best


def main():
    os.makedirs(OUT, exist_ok=True)
    D = P.load()
    files = sorted(glob.glob(f"{FEAT_SP}/ep*.npz"))
    assert [os.path.basename(f) for f in files] == sorted(set(D["fname"])), "episode sets differ"
    sp = {k: [] for k in ("trunk_sp", "raw_sp1", "raw_sp2", "ws")}
    for f in files:
        z = np.load(f)
        for k in sp:
            sp[k].append(z[k])
    sp = {k: np.concatenate(v) for k, v in sp.items()}
    assert np.array_equal(sp["ws"], D["ws"]), "window order differs between the two extractions"
    n = len(D["ws"])
    trunk_sp = sp["trunk_sp"].reshape(n, -1).astype(np.float32)
    raw_sp = np.concatenate([sp["raw_sp1"].reshape(n, -1), sp["raw_sp2"].reshape(n, -1)], 1).astype(np.float32)
    base2 = P.kdx(D["ha0"], D["ha0_ext"]).astype(np.float64)
    base6 = P.kdx(D["ha0_6"], D["ha0_ext_6"]).astype(np.float64)
    g2, g6, ok6 = D["g"].astype(np.float64), D["g6"].astype(np.float64), D["g6_ok"].astype(bool)
    kf = P.kin_feats(D["kin"].astype(np.float64))
    y2 = (g2 - base2).reshape(n, -1)
    y6 = np.where(ok6[:, None, None], g6 - base6, 0.0).reshape(n, -1)
    outer = D["ep"] % P.N_OUTER
    arms = ["P2_kin", "P6_trunk_sp", "P7_raw_sp", "P8_shuf_sp"]
    pred2 = {a: np.zeros_like(g2) for a in arms}
    pred6 = {a: np.zeros_like(g6) for a in arms}
    meta = {"n_windows": int(n), "d_trunk_sp": int(trunk_sp.shape[1]), "d_raw_sp": int(raw_sp.shape[1]),
            "d_kin": int(kf.shape[1]), "folds": {}}
    rng = np.random.default_rng(0)
    for f in range(P.N_OUTER):
        tr, te = outer != f, outer == f
        fm = {"n_train": int(tr.sum()), "n_test": int(te.sum())}
        for a in arms:
            img = {"P2_kin": None, "P6_trunk_sp": trunk_sp, "P7_raw_sp": raw_sp, "P8_shuf_sp": trunk_sp}[a]
            use_img = img is not None
            img_tr = None
            if use_img:
                img_tr = img[tr]
                if a == "P8_shuf_sp":
                    img_tr = img_tr[rng.permutation(len(img_tr))]
            r2, sel2 = fit_select(img_tr, kf[tr], y2[tr], base2[tr], g2[tr], D["ep"][tr], use_img)
            pred2[a][te] = base2[te] + r2.predict(img[te] if use_img else None, kf[te]).reshape(-1, 10, 2)
            t6 = tr & ok6
            img6 = None if not use_img else (img[t6] if a != "P8_shuf_sp" else img[t6][rng.permutation(int(t6.sum()))])
            r6, sel6 = fit_select(img6, kf[t6], y6[t6], base6[t6], g6[t6], D["ep"][t6], use_img)
            pred6[a][te] = base6[te] + r6.predict(img[te] if use_img else None, kf[te]).reshape(-1, g6.shape[1], 2)
            fm[a] = {"sel_2s": [float(sel2[0]), int(sel2[1])], "d_2s": int(r2.d),
                     "sel_6s": [float(sel6[0]), int(sel6[1])], "d_6s": int(r6.d)}
        meta["folds"][f] = fm
        print(f"[probe_sp] fold {f} done: {json.dumps(fm)[:400]}", flush=True)
    # known-value control: P2 must reproduce probe.py's P2 EXACTLY (same folds, features, grid, code path)
    ref = {fn: np.load(f"C:/Users/Admin/refav1_probe/out/dump_P2_kin/{fn}")["cl"] for fn in np.unique(D["fname"])}
    p2_ref = np.concatenate([ref[fn] for fn in D["fname"][np.sort(np.unique(D["fname"], return_index=True)[1])]])
    meta["control_P2_reproduces_probe_py_max_abs"] = float(np.abs(pred2["P2_kin"].astype(np.float32) - p2_ref).max())
    print("[probe_sp] control P2 == probe.py P2, max|diff| =", meta["control_P2_reproduces_probe_py_max_abs"], flush=True)
    for a in arms:
        dd = f"{OUT}/dump_{a}"
        os.makedirs(dd, exist_ok=True)
        for fn in np.unique(D["fname"]):
            s = D["fname"] == fn
            np.savez(f"{dd}/{fn}", g=D["g"][s], v0=D["v0"][s].astype(np.float32),
                     cl=pred2[a][s].astype(np.float32), ha=D["ha"][s], ha0=D["ha0"][s], ha0_ext=D["ha0_ext"][s],
                     ol=D["ol"][s], ws=D["ws"][s].astype(np.int64), eid=D["eid"][s][:1].astype(np.int64),
                     clip_index=D["clip_index"][s][:1].astype(np.int64))
    sys.path.insert(0, "C:/Users/Admin/tipsnap/b3f7ea6f/taniteval")
    from taniteval.ci import paired_episode_cluster_bootstrap as pb
    eid6 = D["ep"][ok6]
    ade2 = {a: P.ade(pred2[a], g2) for a in arms}
    ade6 = {a: P.ade(pred6[a][ok6], g6[ok6]) for a in arms}
    fde6 = {a: np.linalg.norm(pred6[a][ok6][:, -1] - g6[ok6][:, -1], axis=-1) for a in arms}
    pairs = [("P6_trunk_sp", "P2_kin"), ("P8_shuf_sp", "P2_kin"), ("P7_raw_sp", "P2_kin"),
             ("P6_trunk_sp", "P7_raw_sp")]
    res = {"meta": meta, "means_2s_ade": {a: float(v.mean()) for a, v in ade2.items()},
           "means_6s_ade": {a: float(v.mean()) for a, v in ade6.items()}, "pairs_2s_ade": {}, "pairs_6s": {}}
    for x, y in pairs:
        res["pairs_2s_ade"][f"{x}-{y}"] = pb(ade2[x], ade2[y], D["ep"])
        res["pairs_6s"][f"{x}-{y}"] = {"ade": pb(ade6[x], ade6[y], eid6), "fde": pb(fde6[x], fde6[y], eid6)}
    json.dump(res, open(f"{OUT}/probe_sp_result.json", "w"), indent=1)
    print("ZZPROBESP-DONEZZ", json.dumps(res["means_2s_ade"]), flush=True)


if __name__ == "__main__":
    main()
