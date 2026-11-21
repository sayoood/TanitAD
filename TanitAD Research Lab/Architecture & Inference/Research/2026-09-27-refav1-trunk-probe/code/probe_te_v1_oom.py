"""SPEC Amendment A2 -- TRAIN -> EVAL. Readouts are FITTED on refav1-TRAIN clips (600, stride 8) and SCORED on the
141 EVAL episodes (stride 4, the identical 2,399 windows of s4 / A1). lambda / PCA m / MLP settings are chosen by
grouped inner CV on the TRAIN windows only; no eval window is ever fitted on.
Ridge arms reuse probe.py / probe_sp.py (imported). P9 is the pre-registered nonlinear arm (one hidden layer).
"""
import glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, "C:/Users/Admin/refav1_probe")
import probe as P
import probe_sp as S

TRAIN = os.environ.get("PROBE_TE_TRAIN", "C:/Users/Admin/refav1_trainprobe/feat_train_s8")
OUT = os.environ.get("PROBE_TE_OUT", "C:/Users/Admin/refav1_probe/out_te")


def load_train():
    cols = {}
    for fi, f in enumerate(sorted(glob.glob(f"{TRAIN}/ep*.npz"))):
        z = np.load(f)
        n = len(z["ws"])
        for k in ("trunk", "raw1", "raw2", "raw3", "kin", "g", "g6", "g6_ok", "ha0", "ha0_ext", "ha0_6", "ha0_ext_6",
                  "trunk_sp", "raw_sp1", "raw_sp2"):
            cols.setdefault(k, []).append(z[k])
        cols.setdefault("ep", []).append(np.full(n, fi))
    return {k: np.concatenate(v) for k, v in cols.items()}


def load_eval():
    D = P.load()
    sp = {k: [] for k in ("trunk_sp", "raw_sp1", "raw_sp2", "ws")}
    for f in sorted(glob.glob(f"{S.FEAT_SP}/ep*.npz")):
        z = np.load(f)
        for k in sp:
            sp[k].append(z[k])
    sp = {k: np.concatenate(v) for k, v in sp.items()}
    assert np.array_equal(sp["ws"], D["ws"])
    D.update({k: sp[k] for k in ("trunk_sp", "raw_sp1", "raw_sp2")})
    return D


def feats(D):
    n = len(D["kin"])
    return {"kin": P.kin_feats(D["kin"].astype(np.float64)),
            "trunk": D["trunk"].astype(np.float64),
            "raw": np.concatenate([D["raw1"], D["raw2"], D["raw3"]], 1).astype(np.float64),
            "trunk_sp": D["trunk_sp"].reshape(n, -1).astype(np.float32),
            "raw_sp": np.concatenate([D["raw_sp1"].reshape(n, -1), D["raw_sp2"].reshape(n, -1)], 1).astype(np.float32)}


class MLP(torch.nn.Module):
    def __init__(self, d_in, d_out, h=256):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, h), torch.nn.GELU(), torch.nn.Linear(h, d_out))
        torch.nn.init.zeros_(self.net[2].weight); torch.nn.init.zeros_(self.net[2].bias)   # starts AT the prior

    def forward(self, x):
        return self.net(x)


def mlp_curve(Xtr, ytr, Xva, base_va, g_va, wd, epochs, seed=0):
    torch.manual_seed(seed)
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym = ytr.mean(0)
    net = MLP(Xtr.shape[1], ytr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    Xt = torch.tensor((Xtr - xm) / xs, dtype=torch.float32); Yt = torch.tensor(ytr - ym, dtype=torch.float32)
    Xv = None if Xva is None else torch.tensor((Xva - xm) / xs, dtype=torch.float32)
    rng = np.random.default_rng(seed); curve = []
    for ep in range(epochs):
        perm = rng.permutation(len(Xt))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if Xva is not None:
            with torch.no_grad():
                yh = net(Xv).numpy() + ym
            curve.append(float(P.ade(base_va + yh.reshape(-1, *base_va.shape[1:]), g_va).mean()))
    return net, (xm, xs, ym), curve


def fit_mlp(img, kf, y, base, g, groups, ms=(16, 64), wds=(1e-4, 1e-2), epochs=60):
    """inner grouped CV over (m, wd, epoch); refit on all rows at the chosen epoch count."""
    ug = np.unique(groups); inner = {e: i % P.N_INNER for i, e in enumerate(ug)}
    fold = np.array([inner[e] for e in groups])
    score = {}
    for f in range(P.N_INNER):
        tr, te = fold != f, fold == f
        ro = S.GramReadout(img[tr], kf[tr])
        for m in ms:
            Xtr, Xte = ro.design(img[tr], kf[tr], m), ro.design(img[te], kf[te], m)
            for wd in wds:
                _, _, cur = mlp_curve(Xtr, y[tr], Xte, base[te], g[te], wd, epochs)
                score.setdefault((m, wd), []).append(np.array(cur) * te.sum())
    tot = len(groups)
    best, best_s = None, np.inf
    for k, v in score.items():
        c = np.sum(v, 0) / tot
        e = int(np.argmin(c))
        if c[e] < best_s:
            best, best_s = (k[0], k[1], e + 1), float(c[e])
    m, wd, ep = best
    ro = S.GramReadout(img, kf)
    X = ro.design(img, kf, m)
    net, st, _ = mlp_curve(X, y, None, None, None, wd, ep)
    return ro, net, st, best



def strata_masks(g, v0, base, dt=0.2):
    """SPEC A2-prime: strata from the GT path ONLY (plus kd_x's own error for 'hard'); fixed before the data."""
    d = g[:, -1] - g[:, -2]
    head = np.degrees(np.abs(np.arctan2(d[:, 1], d[:, 0])))
    v_term = np.linalg.norm(d, axis=-1) / dt
    e_base = P.ade(base, g)
    return {"turn": head > 15.0, "straight": head < 5.0,
            "stop": (v0 > 3.0) & (v_term < 1.0), "accelerate": (v_term - v0) > 1.5,
            "hard": e_base > np.percentile(e_base, 75)}

def main():
    os.makedirs(OUT, exist_ok=True)
    T, E = load_train(), load_eval()
    FT, FE = feats(T), feats(E)
    bT2, bE2 = P.kdx(T["ha0"], T["ha0_ext"]), P.kdx(E["ha0"], E["ha0_ext"])
    bT6, bE6 = P.kdx(T["ha0_6"], T["ha0_ext_6"]), P.kdx(E["ha0_6"], E["ha0_ext_6"])
    gT2, gE2 = T["g"].astype(np.float64), E["g"].astype(np.float64)
    gT6, gE6 = T["g6"].astype(np.float64), E["g6"].astype(np.float64)
    okT, okE = T["g6_ok"].astype(bool), E["g6_ok"].astype(bool)
    yT2 = (gT2 - bT2).reshape(len(gT2), -1)
    yT6 = np.where(okT[:, None, None], gT6 - bT6, 0.0).reshape(len(gT6), -1)
    meta = {"n_train_windows": int(len(gT2)), "n_train_episodes": int(len(np.unique(T["ep"]))),
            "n_train_6s": int(okT.sum()), "n_eval_windows": int(len(gE2)), "n_eval_6s": int(okE.sum()), "sel": {}}
    print("[probe_te]", json.dumps(meta), flush=True)
    rng = np.random.default_rng(0)
    arms = {"P0_kdx": None, "P2_kin": None, "P3_raw": "raw", "P4_trunk": "trunk", "P5_shuf": "trunk",
            "P6_trunk_sp": "trunk_sp", "P7_raw_sp": "raw_sp", "P8_shuf_sp": "trunk_sp", "P9_trunk_sp_mlp": "trunk_sp"}
    pred2, pred6 = {}, {}
    for a, key in arms.items():
        if a == "P0_kdx":
            pred2[a], pred6[a] = bE2.copy(), bE6.copy(); continue
        use_img = key is not None
        imgT = FT[key] if use_img else None
        imgE = FE[key] if use_img else None
        if a in ("P5_shuf", "P8_shuf_sp"):
            imgT = imgT[rng.permutation(len(imgT))]
        out = {}
        for hz, (y, b, g, ok, bE) in {"2s": (yT2, bT2, gT2, np.ones(len(gT2), bool), bE2),
                                      "6s": (yT6, bT6, gT6, okT, bE6)}.items():
            it = None if not use_img else imgT[ok]
            if a == "P9_trunk_sp_mlp":
                ro, net, st, best = fit_mlp(it, FT["kin"][ok], y[ok], b[ok], g[ok], T["ep"][ok])
                xm, xs, ym = st
                with torch.no_grad():
                    yh = net(torch.tensor((ro.design(imgE, FE["kin"], best[0]) - xm) / xs, dtype=torch.float32)).numpy() + ym
                out[hz] = (bE + yh.reshape(-1, *bE.shape[1:]), [int(best[0]), float(best[1]), int(best[2])])
            else:
                ro, best = S.fit_select(it, FT["kin"][ok], y[ok], b[ok], g[ok], T["ep"][ok], use_img)
                out[hz] = (bE + ro.predict(imgE, FE["kin"]).reshape(-1, *bE.shape[1:]), [float(best[0]), int(best[1])])
        pred2[a], pred6[a] = out["2s"][0], out["6s"][0]
        meta["sel"][a] = {"2s": out["2s"][1], "6s": out["6s"][1]}
        print(f"[probe_te] {a} sel={meta['sel'][a]} ADE2={P.ade(pred2[a], gE2).mean():.4f}", flush=True)
    for a in arms:
        dd = f"{OUT}/dump_{a}"; os.makedirs(dd, exist_ok=True)
        for fn in np.unique(E["fname"]):
            s = E["fname"] == fn
            np.savez(f"{dd}/{fn}", g=E["g"][s], v0=E["v0"][s].astype(np.float32), cl=pred2[a][s].astype(np.float32),
                     ha=E["ha"][s], ha0=E["ha0"][s], ha0_ext=E["ha0_ext"][s], ol=E["ol"][s],
                     ws=E["ws"][s].astype(np.int64), eid=E["eid"][s][:1].astype(np.int64),
                     clip_index=E["clip_index"][s][:1].astype(np.int64))
    sys.path.insert(0, "C:/Users/Admin/tipsnap/b3f7ea6f/taniteval")
    from taniteval.ci import paired_episode_cluster_bootstrap as pb
    ade2 = {a: P.ade(pred2[a], gE2) for a in arms}
    ade6 = {a: P.ade(pred6[a][okE], gE6[okE]) for a in arms}
    fde6 = {a: np.linalg.norm(pred6[a][okE][:, -1] - gE6[okE][:, -1], axis=-1) for a in arms}
    pairs = [("P6_trunk_sp", "P2_kin"), ("P4_trunk", "P2_kin"), ("P9_trunk_sp_mlp", "P2_kin"),
             ("P6_trunk_sp", "P7_raw_sp"), ("P9_trunk_sp_mlp", "P6_trunk_sp"), ("P2_kin", "P0_kdx"),
             ("P5_shuf", "P2_kin"), ("P8_shuf_sp", "P2_kin"), ("P3_raw", "P2_kin"), ("P0_kdx", "P0_kdx")]
    res = {"meta": meta, "means_2s_ade": {a: float(v.mean()) for a, v in ade2.items()},
           "means_6s_ade": {a: float(v.mean()) for a, v in ade6.items()}, "pairs_2s_ade": {}, "pairs_6s": {}}
    for x, y in pairs:
        res["pairs_2s_ade"][f"{x}-{y}"] = pb(ade2[x], ade2[y], E["ep"])
        res["pairs_6s"][f"{x}-{y}"] = {"ade": pb(ade6[x], ade6[y], E["ep"][okE]),
                                       "fde": pb(fde6[x], fde6[y], E["ep"][okE])}
    # A2-prime descriptive strata (no bar); a stratum with < 10 episode clusters is UNDERPOWERED, never read
    res["strata"] = {}
    v0E = E["v0"].astype(np.float64)
    for hz, g_, b_, ok_, pr in (("2s", gE2, bE2, np.ones(len(gE2), bool), pred2), ("6s", gE6, bE6, okE, pred6)):
        M = strata_masks(g_[ok_], v0E[ok_], b_[ok_])
        eid = E["ep"][ok_]
        for name, msk in M.items():
            n_cl = int(len(np.unique(eid[msk])))
            row = {"n_windows": int(msk.sum()), "n_clusters": n_cl}
            if n_cl < 10:
                row["status"] = "UNDERPOWERED"
            else:
                for x, y in (("P6_trunk_sp", "P2_kin"), ("P9_trunk_sp_mlp", "P2_kin"), ("P2_kin", "P0_kdx")):
                    ax = P.ade(pr[x][ok_][msk], g_[msk]); ay = P.ade(pr[y][ok_][msk], g_[msk])
                    fx = np.linalg.norm(pr[x][ok_][msk][:, -1] - g_[msk][:, -1], axis=-1)
                    fy = np.linalg.norm(pr[y][ok_][msk][:, -1] - g_[msk][:, -1], axis=-1)
                    row[f"{x}-{y}"] = {"ade": pb(ax, ay, eid[msk]), "fde": pb(fx, fy, eid[msk])}
            res["strata"][f"{hz}:{name}"] = row
    json.dump(res, open(f"{OUT}/probe_te_result.json", "w"), indent=1)
    print("ZZPROBETE-DONEZZ", json.dumps(res["means_2s_ade"]), flush=True)


if __name__ == "__main__":
    main()
