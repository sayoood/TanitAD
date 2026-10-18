"""HJ-UNITS D2 -- the refav1 trunk-probe s1 (global ridge, arms P0..P5) re-read with the `kd_x` base in a STATED unit.

PREREG: ../PREREG.md (written before the first run). One lever only: the unit of the base.
  --units kappa : base from the STORED banked ha0 / ha0_ext / ha0_ext_6 (wrong unit)  -> CONTROL, must equal the banked probe_result.json
  --units steer : base REBUILT from the stored `kin` = (v0, a0, steer0) with floors_kin(action_units="steer") -> TREATMENT
Everything else -- Readout, fit_select, LAMS, MS, folds, rng, arms, pairs, bootstrap -- is the banked
`2026-09-27-refav1-trunk-probe/code/probe.py` VERBATIM (only paths, the base builder and the controls are new).
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--units", default="both", choices=["kappa", "steer", "both"])
ap.add_argument("--repo", default="C:/Users/Admin/hjunits/tree", help="checkout root holding taniteval/")
ap.add_argument("--feat", default="D:/Archive/devbox-C/refav1_probe/feat_s4")
ap.add_argument("--banked", default="D:/Archive/devbox-C/refav1_probe/out")
ap.add_argument("--out", required=True)
A = ap.parse_args()
sys.path.insert(0, os.path.join(A.repo, "taniteval"))
from taniteval import floors_kin as fk                                         # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as pb                # noqa: E402

DT = 0.2
LAMS = np.logspace(-2, 4, 13)
MS = (8, 16, 32, 64, 128)
N_OUTER, N_INNER = 5, 4
R46 = {"ha0_ext_kappa_ade6": 5.356479020677318, "ha0_ext_steer_ade6": 2.838041984856346,
       "kd_x_steer_ade6": 2.939635049776786, "kd_x_kappa_ade6": 3.5583767272028215}   # refav1_units_check.json


def load():
    files = sorted(glob.glob(f"{A.feat}/ep*.npz"))
    cols, md5 = {}, hashlib.md5()
    for fi, f in enumerate(files):
        z = np.load(f)
        md5.update(os.path.basename(f).encode()); md5.update(z["ws"].tobytes()); md5.update(z["kin"].tobytes())
        n = len(z["ws"])
        for k in ("trunk", "raw1", "raw2", "raw3", "kin", "g", "g6", "g6_ok", "v0", "ha", "ha0", "ha0_ext",
                  "ol", "ha0_6", "ha0_ext_6", "ws"):
            cols.setdefault(k, []).append(z[k])
        cols.setdefault("ep", []).append(np.full(n, fi))
    D = {k: np.concatenate(v) for k, v in cols.items()}
    D["_files"], D["_md5"] = files, md5.hexdigest()
    return D


def kin_feats(kin):
    v, a, k = kin[:, 0], kin[:, 1], kin[:, 2]
    return np.stack([v, a, k, v * v, v * a, v * k, v * v * k, a * k], 1)


def ade(pred, g):
    return np.linalg.norm(pred - g, axis=-1).mean(-1)


_BASIS_CACHE = {}
_C0 = {}


def _gram_vt(Z, k):
    """top-k right singular vectors of Z [n,d] by the Gram trick (eigh of the SMALLER Gram matrix)."""
    n, d = Z.shape
    if n <= d:
        w, U = np.linalg.eigh(Z @ Z.T)
        w, U = w[::-1][:k], U[:, ::-1][:, :k]
        return (Z.T @ U / np.sqrt(np.maximum(w, 1e-300))).T
    w, V = np.linalg.eigh(Z.T @ Z)
    return V[:, ::-1][:, :k].T


class Readout:
    """standardise -> (optional) PCA on the image block -> ridge with intercept; every statistic is fit on the
    rows given. The PCA basis is computed ONCE per row set and shared by every (lambda, m) of the grid.
    AMENDMENT A1: the basis is the Gram-trick eigendecomposition (== the SVD's right vectors up to sign), cached by content."""

    def __init__(self, img, kf):
        self.kmu, self.ksd = kf.mean(0), kf.std(0) + 1e-8
        self.has_img = img is not None
        if self.has_img:
            self.imu, self.isd = img.mean(0), img.std(0) + 1e-6
            key = hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()
            if key not in _BASIS_CACHE:
                Z = (img - self.imu) / self.isd
                vt = _gram_vt(Z, max(MS))
                if len(_C0) < 2:                       # C0: Gram vs the banked SVD on the first two blocks seen
                    _, _, vs = np.linalg.svd(Z, full_matrices=False)
                    vs = vs[:max(MS)]
                    cos = np.abs(np.sum(vt[:64] * vs[:64], axis=1))
                    _C0[f"block{len(_C0)}"] = {"shape": list(Z.shape), "min_abs_cos_top64": float(cos.min()),
                                              "projector128_max_abs_diff": float(np.abs(vt.T @ vt - vs.T @ vs).max())}
                _BASIS_CACHE[key] = vt
            self.vt = _BASIS_CACHE[key]

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


def build_base(D, units):
    """(base2 [n,10,2], base6 [n,30,2], controls dict)."""
    kin = D["kin"].astype(np.float64)
    ctl = {}
    # --- the rebuild-from-kin in BOTH units (controls C2/C3 are measured in both modes)
    reb = {}
    for u in ("kappa", "steer"):
        reb[u] = {"he2": fk.ha0_ext_paths(kin, DT, 10, action_units=u), "he6": fk.ha0_ext_paths(kin, DT, 30, action_units=u)}
    ha0_2 = fk.ha0_paths(kin[:, 0], DT, 10); ha0_6 = fk.ha0_paths(kin[:, 0], DT, 30)
    g6, ok6 = D["g6"].astype(np.float64), D["g6_ok"].astype(bool)
    ctl["C2_kappa_rebuild_vs_stored_ha0_ext_max_abs_m"] = float(np.abs(reb["kappa"]["he2"] - D["ha0_ext"]).max())
    ctl["C2_kappa_rebuild_vs_stored_ha0_ext_6_max_abs_m"] = float(np.abs(reb["kappa"]["he6"] - D["ha0_ext_6"]).max())
    ctl["C2_steer_rebuild_vs_stored_ha0_ext_6_max_abs_m"] = float(np.abs(reb["steer"]["he6"] - D["ha0_ext_6"]).max())
    ctl["C2_ha0_rebuild_vs_stored_max_abs_m"] = float(max(np.abs(ha0_2 - D["ha0"]).max(), np.abs(ha0_6 - D["ha0_6"]).max()))
    for u in ("kappa", "steer"):
        ctl[f"C3_ha0_ext_{u}_ade6"] = float(ade(reb[u]["he6"][ok6], g6[ok6]).mean())
        ctl[f"C3_kd_x_{u}_ade6"] = float(ade(fk.kd_x(ha0_6, reb[u]["he6"])[ok6].astype(np.float64), g6[ok6]).mean())
        ctl[f"kd_x_{u}_ade2"] = float(ade(fk.kd_x(ha0_2, reb[u]["he2"]).astype(np.float64), D["g"].astype(np.float64)).mean())
    if units == "kappa":      # the banked path: stored arrays, exactly as probe.py::kdx
        b2 = fk.kd_x(D["ha0"], D["ha0_ext"]).astype(np.float64)
        b6 = fk.kd_x(D["ha0_6"], D["ha0_ext_6"]).astype(np.float64)
        # C4: against the banked P0 dump (cl == pred2["P0_kdx"] == base2 as float32)
        cl = np.concatenate([np.load(f)["cl"] for f in sorted(glob.glob(f"{A.banked}/dump_P0_kdx/ep*.npz"))])
        ctl["C4_base2_vs_banked_P0_dump_max_abs_m"] = float(np.abs(b2.astype(np.float32) - cl).max())
    else:
        b2 = fk.kd_x(ha0_2, reb["steer"]["he2"]).astype(np.float64)
        b6 = fk.kd_x(ha0_6, reb["steer"]["he6"]).astype(np.float64)
    return b2, b6, ctl


def main(units):
    t0 = time.time()
    os.makedirs(A.out, exist_ok=True)
    D = load()
    n = len(D["ws"])
    base2, base6, ctl = build_base(D, units)
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
    meta = {"units": units, "n_windows": int(n), "n_episodes": int(len(np.unique(D["ep"]))), "n_6s_ok": int(ok6.sum()),
            "d_trunk": int(trunk.shape[1]), "d_raw": int(raw.shape[1]), "d_kin": int(kf.shape[1]),
            "feat_dir": A.feat, "feat_md5_of_ws_kin": D["_md5"], "folds": {}}
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
        print(f"[reread {units}] fold {f} done {time.time()-t0:.0f}s", flush=True)
    meta["control_P0_equals_base_max_abs"] = float(np.abs(pred2["P0_kdx"] - base2).max())
    eid6 = D["ep"][ok6]
    ade6 = {a: ade(pred6[a][ok6], g6[ok6]) for a in arms}
    fde6 = {a: np.linalg.norm(pred6[a][ok6][:, -1] - g6[ok6][:, -1], axis=-1) for a in arms}
    ade2 = {a: ade(pred2[a], g2) for a in arms}
    pairs = [("P4_trunk", "P2_kin"), ("P4_trunk", "P0_kdx"), ("P5_shuf", "P2_kin"), ("P1_const", "P2_kin"),
             ("P3_raw", "P2_kin"), ("P4_trunk", "P3_raw"), ("P2_kin", "P0_kdx"), ("P0_kdx", "P0_kdx")]
    res = {"meta": meta, "controls": ctl, "means_2s_ade": {a: float(v.mean()) for a, v in ade2.items()},
           "means_6s_ade": {a: float(v.mean()) for a, v in ade6.items()},
           "means_6s_fde": {a: float(v.mean()) for a, v in fde6.items()}, "pairs_6s": {}, "pairs_2s_ade": {}}
    for x, y in pairs:
        res["pairs_6s"][f"{x}-{y}"] = {"ade": pb(ade6[x], ade6[y], eid6), "fde": pb(fde6[x], fde6[y], eid6)}
        res["pairs_2s_ade"][f"{x}-{y}"] = pb(ade2[x], ade2[y], D["ep"])
    res["checks"] = checks(res, units)
    res["C0_gram_vs_svd"] = dict(_C0)
    res["checks"]["C0_pass"] = bool(len(_C0) >= 1 and all(v["min_abs_cos_top64"] >= 1 - 1e-8 and v["projector128_max_abs_diff"] <= 1e-6 for v in _C0.values()))
    json.dump(res, open(f"{A.out}/probe_units_{units}.json", "w"), indent=1)
    print("ZZREREAD-DONEZZ", units, json.dumps(res["checks"]), f"{time.time()-t0:.0f}s", flush=True)


def checks(res, units):
    """The PREREG s4 controls and s5 classification, in code, written before the run."""
    c = res["controls"]; out = {}
    out["C2_pass"] = bool(c["C2_kappa_rebuild_vs_stored_ha0_ext_max_abs_m"] <= 2e-4
                          and c["C2_kappa_rebuild_vs_stored_ha0_ext_6_max_abs_m"] <= 2e-4
                          and c["C2_steer_rebuild_vs_stored_ha0_ext_6_max_abs_m"] > 1.0)
    out["C3_pass"] = bool(abs(c["C3_ha0_ext_kappa_ade6"] - R46["ha0_ext_kappa_ade6"]) <= 1e-3
                          and abs(c["C3_ha0_ext_steer_ade6"] - R46["ha0_ext_steer_ade6"]) <= 1e-3
                          and abs(c["C3_kd_x_steer_ade6"] - R46["kd_x_steer_ade6"]) <= 1e-3)
    p = res["pairs_2s_ade"]
    out["V_pass"] = bool(not (p["P5_shuf-P2_kin"]["hi"] < 0) and not (p["P1_const-P2_kin"]["hi"] < 0))
    if units == "kappa":
        try:
            bk = json.load(open(f"{A.banked}/probe_result.json"))
            d = max(max(abs(res["means_2s_ade"][a] - bk["means_2s_ade"][a]) for a in bk["means_2s_ade"]),
                    max(abs(res["means_6s_ade"][a] - bk["means_6s_ade"][a]) for a in bk["means_6s_ade"]),
                    max(abs(res["means_6s_fde"][a] - bk["means_6s_fde"][a]) for a in bk["means_6s_fde"]))
            q, b = p["P4_trunk-P2_kin"], bk["pairs_2s_ade"]["P4_trunk-P2_kin"]
            dp = max(abs(q[k] - b[k]) for k in ("delta", "lo", "hi"))
            out["C1_max_abs_mean_diff_m"], out["C1_P4P2_delta_lo_hi_max_abs"] = float(d), float(dp)
            out["C1_pass"] = bool(d <= 5e-5 and dp <= 5e-4)
        except Exception as e:                                          # a failed control is not a pass
            out["C1_pass"] = False; out["C1_error"] = repr(e)
        out["C4_pass"] = bool(c["C4_base2_vs_banked_P0_dump_max_abs_m"] <= 1e-5)
    else:
        q = p["P4_trunk-P2_kin"]
        out["primary_P4_minus_P2_ade2"] = {k: q[k] for k in ("delta", "lo", "hi", "separated")}
        out["classification_if_controls_pass"] = ("O-REVERSED" if q["hi"] < 0 else
                                                  ("O-STANDS-WORSE" if q["lo"] > 0 else "O-STANDS-NULL"))
    return out


if __name__ == "__main__":
    for u in (["kappa", "steer"] if A.units == "both" else [A.units]):
        main(u)
