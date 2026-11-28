"""wua_imag.py -- card ADD155 sections 2 (A2 ego-matched pairs), 3 (A3 imagination skill), 4 (A4 action counterfactuals) for ONE seed; CPU.

  python -I wua_imag.py --seed 0
Reads the extraction (wua_extract.py) and the labels (wua_labels.py); A2 runs its own common-action rollouts with the checkpoint (wua_extract.load).
Writes C:/Users/Admin/wua_work/imag/RESULT_IMAG_s<seed>.json.
"""
import argparse, hashlib, json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np
import torch
from sklearn.linear_model import Ridge
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wua_a1 as A1   # noqa: E402
import wua_extract as WX   # noqa: E402

WORK = A1.WORK
STRIDE = WX.STRIDE
A3_CELLS = [(1, 5), (1, 10), (2, 5), (2, 15), (3, 5), (3, 15)]
A4_CELLS = [(1, 10), (2, 5), (3, 3)]
VARS = {"speed": None, "lead_gap": ["lead_gap"], "ag1_pos": ["ag1_x", "ag1_y"], "lead_closing": ["lead_closing"]}
DT = 0.2


def fold_models(X, y, ci, fo):
    """per outer fold: (mean, std, fitted Ridge) trained on the other folds (alpha by inner CV, as A1)."""
    out = {}
    for f in range(5):
        tr = fo != f
        if tr.sum() < 20 or len(np.unique(ci[tr])) < 3:
            continue
        al = A1.fit_ridge(X[tr], y[tr], ci[tr])
        m, s = X[tr].mean(0), X[tr].std(0) + 1e-8
        out[f] = (m, s, Ridge(alpha=al).fit((X[tr] - m) / s, y[tr]))
    return out


def apply(models, f, X):
    if f not in models:
        return np.full((len(X), 1), np.nan)
    m, s, r = models[f]
    return r.predict((X - m) / s).reshape(len(X), -1)


def clip_sums(v, ci, nc):
    return np.bincount(ci, weights=v, minlength=nc)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    E = np.load(f"{A1.EXT}/WUA_RH_s{a.seed}.npz")
    Lb = np.load(A1.LAB)
    assert (E["clip_idx"] == Lb["clip_idx"]).all() and (E["t"] == Lb["t"]).all()
    names = list(Lb["names"]); Y = Lb["Y"]
    ci_all, t_all = E["clip_idx"], E["t"].astype(int)
    clips = list(E["clips"]); nc = len(clips); fclip = A1.fold_of(clips)
    key = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(ci_all, t_all))}
    lev = {1: E["L1"].astype(np.float64), 2: E["L2"].astype(np.float64), 3: E["L3"].astype(np.float64)}
    v, acc, yr = E["v"], E["a"], E["yaw_rate"]
    W = A1.boot_w(nc)

    def lab(var, idx):
        if var == "speed":
            return v[idx][:, None].astype(np.float64)
        return np.stack([Y[idx, names.index(c)] for c in VARS[var]], 1)

    # ---------- probes on encoded states (state alone), per level and variable, OOF fold models
    allidx = np.arange(len(ci_all))
    P = {}
    for lv in (1, 2, 3):
        for var in VARS:
            y = lab(var, allidx); ok = np.isfinite(y).all(1)
            P[(lv, var)] = fold_models(lev[lv][ok], y[ok], ci_all[ok], fclip[ci_all[ok]])
    # ---------- A3 and A4 on the anchors
    rk = list(E["roll_keys"]); R = E["roll"].astype(np.float64)
    ac, at = E["anc_clip"].astype(int), E["anc_t"].astype(int)
    af = fclip[ac]
    i0 = np.array([key[(c, t)] for c, t in zip(ac, at)])
    res = {"card": "ADD155", "seed": a.seed, "A3": {}, "A4": {}, "A2": {}, "controls": {}}

    def pred(lv, var, Z, fsel):
        out = np.full((len(Z), 2 if var == "ag1_pos" else 1), np.nan)
        for f in range(5):
            m = (fsel == f) & np.isfinite(Z).all(1)
            if m.any():
                out[m] = apply(P[(lv, var)], f, Z[m])
        return out

    for lv, k in A3_CELLS:
        h = k * STRIDE[lv]
        Zr = R[rk.index(f"base|{lv}|{k}")]
        fut = np.array([key.get((c, t + h), -1) for c, t in zip(ac, at)])
        for var in VARS:
            ok = (fut >= 0) & np.isfinite(Zr).all(1)
            ytrue = np.full((len(ac), 2 if var == "ag1_pos" else 1), np.nan)
            ytrue[ok] = lab(var, fut[ok])
            ok &= np.isfinite(ytrue).all(1)
            if ok.sum() < 50:
                res["A3"][f"L{lv}|k{k}|{var}"] = {"n": int(ok.sum()), "skipped": "n"}
                continue
            pi = pred(lv, var, Zr[ok], af[ok]); pc = pred(lv, var, lev[lv][i0[ok]], af[ok]); po = pred(lv, var, lev[lv][fut[ok]], af[ok])
            yt = ytrue[ok]; cc = ac[ok]
            gd = np.isfinite(pi).all(1) & np.isfinite(pc).all(1) & np.isfinite(po).all(1)
            pi, pc, po, yt, cc = pi[gd], pc[gd], po[gd], yt[gd], cc[gd]
            si = clip_sums(((yt - pi) ** 2).sum(1), cc, nc); sc = clip_sums(((yt - pc) ** 2).sum(1), cc, nc); so = clip_sums(((yt - po) ** 2).sum(1), cc, nc)
            n = clip_sums(np.ones(len(cc)), cc, nc)
            d = (W @ sc - W @ si) / (W @ n)
            sy = [clip_sums(yt[:, j], cc, nc) for j in range(yt.shape[1])]; syy = [clip_sums(yt[:, j] ** 2, cc, nc) for j in range(yt.shape[1])]
            sst = sum(syy[j].sum() - sy[j].sum() ** 2 / n.sum() for j in range(yt.shape[1]))
            res["A3"][f"L{lv}|k{k}|{var}"] = {"h_s": round(h * DT, 1), "n": int(len(cc)), "n_clips": int(len(np.unique(cc))),
                                             "mse_imag": float(si.sum() / n.sum()), "mse_copy": float(sc.sum() / n.sum()), "mse_oracle": float(so.sum() / n.sum()),
                                             "skill": float(1 - si.sum() / sc.sum()), "oracle_r2": float(1 - so.sum() / sst),
                                             "copy_minus_imag": float((sc.sum() - si.sum()) / n.sum()), "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}
    # A4
    for lv, k in A4_CELLS:
        h = k * STRIDE[lv] * DT
        for off, var, comp in (("lon", "lead_gap", 0), ("lon", "speed", 0), ("lat", "ag1_pos", 1)):
            Zp, Zm = R[rk.index(f"{off}p|{lv}|{k}")], R[rk.index(f"{off}m|{lv}|{k}")]
            ok = np.isfinite(Zp).all(1) & np.isfinite(Zm).all(1)
            y0 = lab(var, i0)
            ok &= np.isfinite(y0).all(1)
            if ok.sum() < 50:
                res["A4"][f"L{lv}|k{k}|{off}|{var}"] = {"n": int(ok.sum()), "skipped": "n"}
                continue
            meas = (pred(lv, var, Zp[ok], af[ok]) - pred(lv, var, Zm[ok], af[ok]))[:, comp]
            gd = np.isfinite(meas)
            ok[np.where(ok)[0][~gd]] = False
            meas = meas[gd]
            if var == "lead_gap":
                exp_ = np.full(ok.sum(), -(2.0) * h ** 2 / 2)
            elif var == "speed":
                exp_ = np.full(ok.sum(), 2.0 * h)
            else:
                x1 = y0[ok][:, 0]
                exp_ = -(v[i0[ok]] * 0.2 * h ** 2 / 2 + x1 * 0.2 * h)
            cc = ac[ok]
            sm, se = clip_sums(meas, cc, nc), clip_sums(exp_, cc, nc)
            sl = (W @ sm) / (W @ se)
            res["A4"][f"L{lv}|k{k}|{off}|{var}"] = {"h_s": round(h, 1), "n": int(ok.sum()), "n_clips": int(len(np.unique(cc))), "slope": float(sm.sum() / se.sum()),
                                                   "ci": [float(np.nanpercentile(sl, 2.5)), float(np.nanpercentile(sl, 97.5))],
                                                   "mean_measured": float(meas.mean()), "mean_expected": float(exp_.mean())}
    # c8 / c9
    c9 = {}
    for lv, k in A4_CELLS:
        b, z = R[rk.index(f"base|{lv}|{k}")], R[rk.index(f"zero|{lv}|{k}")]
        okb = np.isfinite(b).all(1)
        c9[f"L{lv}|k{k}"] = float(np.abs(b[okb] - z[okb]).max())
    res["controls"]["c9_zero_offset_maxabs"] = c9

    # ---------- A2 ego-matched pairs
    H = A1.H
    cand = np.where((t_all >= H) & (t_all + 10 <= 100) & np.isfinite(Y[:, names.index("gap_change_2s")]))[0]
    egv = np.stack([v[i - H:i + 1] for i in cand]); egy = np.stack([yr[i - H:i + 1] for i in cand]); ega = acc[cand]
    dg = Y[cand, names.index("gap_change_2s")]; g0 = Y[cand, names.index("lead_gap")]; cl0 = Y[cand, names.index("lead_closing")]
    cc_ = ci_all[cand]
    pairs = {"DIFF": [], "SAME": []}
    for ii in range(len(cand)):
        jj = np.arange(ii + 1, len(cand))
        jj = jj[cc_[jj] != cc_[ii]]
        if not len(jj):
            continue
        m = (np.abs(egv[jj] - egv[ii]).max(1) <= 0.5) & (np.abs(egy[jj] - egy[ii]).max(1) <= 0.05) & (np.abs(ega[jj] - ega[ii]) <= 0.5)
        jj = jj[m]
        dd = np.abs(dg[jj] - dg[ii])
        for j in jj[dd >= 2.0]:
            pairs["DIFF"].append((ii, j))
        for j in jj[dd <= 0.3]:
            pairs["SAME"].append((ii, j))
    for kd in pairs:                                   # at most 5 per clip pair, first in frame order
        cnt, keep = {}, []
        for ii, j in sorted(pairs[kd], key=lambda p: (min(cc_[p[0]], cc_[p[1]]), max(cc_[p[0]], cc_[p[1]]), t_all[cand[p[0]]], t_all[cand[p[1]]])):
            kk = (min(cc_[ii], cc_[j]), max(cc_[ii], cc_[j]))
            if cnt.get(kk, 0) < 5:
                cnt[kk] = cnt.get(kk, 0) + 1; keep.append((ii, j))
        pairs[kd] = np.array(keep, int).reshape(-1, 2)
    pure = np.array([(abs(g0[i] - g0[j]) <= 3.0 and np.isfinite(cl0[i]) and np.isfinite(cl0[j]) and abs(cl0[i] - cl0[j]) <= 1.0) for i, j in pairs["DIFF"]], bool)
    res["A2"]["n_pairs"] = {k: int(len(p)) for k, p in pairs.items()}; res["A2"]["n_pure"] = int(pure.sum())
    # scorers
    egoh = np.stack([np.concatenate([v[i - H:i + 1], acc[i - H:i + 1], yr[i - H:i + 1]]) for i in cand])
    fo_c = fclip[cc_]
    allg = np.where((t_all >= H) & np.isfinite(Y[:, names.index("lead_gap")]))[0]
    egoh_g = np.stack([np.concatenate([v[i - H:i + 1], acc[i - H:i + 1], yr[i - H:i + 1]]) for i in allg])
    gp = fold_models(np.concatenate([lev[1][allg], egoh_g], 1), Y[allg, names.index("lead_gap")][:, None], ci_all[allg], fclip[ci_all[allg]])
    gp2 = fold_models(np.concatenate([lev[2][allg], egoh_g], 1), Y[allg, names.index("lead_gap")][:, None], ci_all[allg], fclip[ci_all[allg]])
    enc = A1.oof_ridge(np.concatenate([lev[1][cand], egoh], 1), dg, cc_, fo_c)
    ego = A1.oof_ridge(egoh, dg, cc_, fo_c)
    m, cksha = WX.load(a.seed)
    wm = m.wm

    def gprobe(models, Z, rows_):
        out = np.full(len(rows_), np.nan)
        for f in range(5):
            s = fo_c[rows_] == f
            if s.any():
                out[s] = apply(models, f, np.concatenate([Z[s], egoh[rows_][s]], 1))[:, 0]
        return out

    def imag_scores(P2, models_l1, models_l2):
        """per pair member scores for IMAG-L1 / IMAG-L2 under the pair's common action."""
        ii, jj = P2[:, 0], P2[:, 1]
        ca = np.stack([(acc[cand[ii]] + acc[cand[jj]]) / 2, (yr[cand[ii]] + yr[cand[jj]]) / 2], 1)
        out = {}
        with torch.no_grad():
            for side, rr in (("i", ii), ("j", jj)):
                z1 = torch.tensor(lev[1][cand[rr]], dtype=torch.float32)
                a_sc = wm.scale_actions(torch.tensor(ca, dtype=torch.float32))[:, None].expand(-1, 10, -1).contiguous()
                r1 = wm.stack.rollout(1, z1, a_sc, context=1)[:, 9].double().numpy()
                encd = wm.stack.encode(torch.zeros(len(rr), 11, z1.shape[1]), a_sc)
                z2 = torch.tensor(lev[2][cand[rr]], dtype=torch.float32)
                r2 = wm.stack.rollout(2, z2, encd.act[1][:, :5], context=1)[:, 4].double().numpy()
                out[("IMAG-L1", side)] = gprobe(models_l1, r1, rr) - gprobe(models_l1, lev[1][cand[rr]], rr)
                out[("IMAG-L2", side)] = gprobe(models_l2, r2, rr) - gprobe(models_l2, lev[2][cand[rr]], rr)
                fut = np.array([key[(int(ci_all[cand[x]]), int(t_all[cand[x]]) + 10)] for x in rr])
                out[("ORACLE-L1", side)] = gprobe(models_l1, lev[1][fut], rr) - gprobe(models_l1, lev[1][cand[rr]], rr)
                out[("ENC-L1", side)] = enc[rr]
                out[("EGO", side)] = ego[rr]
        return out

    def auc_pairs(si, sj, ti, tj, wi, wj):
        s = np.sign(si - sj) * np.sign(ti - tj)
        val = np.where(s > 0, 1.0, np.where(s < 0, 0.0, 0.5))
        w = wi * wj
        return (w * val).sum(-1) / w.sum(-1)

    for kd in ("DIFF", "SAME"):
        P2 = pairs[kd]
        if len(P2) < 20:
            res["A2"][kd] = {"skipped": "n", "n": int(len(P2))}
            continue
        sc = imag_scores(P2, gp, gp2)
        ti, tj = dg[P2[:, 0]], dg[P2[:, 1]]
        ci_i, ci_j = cc_[P2[:, 0]], cc_[P2[:, 1]]
        wi, wj = W[:, ci_i], W[:, ci_j]
        out = {}
        aucb = {}
        for name in ("IMAG-L1", "IMAG-L2", "ENC-L1", "ORACLE-L1", "EGO"):
            si, sj = sc[(name, "i")], sc[(name, "j")]
            okp = np.isfinite(si) & np.isfinite(sj)
            a0 = float(auc_pairs(si[okp], sj[okp], ti[okp], tj[okp], np.ones(okp.sum()), np.ones(okp.sum())))
            ab = auc_pairs(si[okp], sj[okp], ti[okp], tj[okp], wi[:, okp], wj[:, okp])
            aucb[name] = (ab, okp)
            out[name] = {"auc": a0, "ci": [float(np.nanpercentile(ab, 2.5)), float(np.nanpercentile(ab, 97.5))], "n": int(okp.sum())}
            if kd == "DIFF" and pure.any():
                pp = okp & pure
                out[name]["pure_auc"] = float(auc_pairs(si[pp], sj[pp], ti[pp], tj[pp], np.ones(pp.sum()), np.ones(pp.sum()))) if pp.sum() else None
            if name == "IMAG-L1":
                out[name]["c7_swap"] = float(auc_pairs(sj[okp], si[okp], ti[okp], tj[okp], np.ones(okp.sum()), np.ones(okp.sum())))
        for name in ("IMAG-L1", "IMAG-L2", "ENC-L1", "ORACLE-L1"):
            ab, okp = aucb[name]; ae, oke = aucb["EGO"]
            if (okp == oke).all():
                d = ab - ae
                out[name]["minus_ego"] = float(out[name]["auc"] - out["EGO"]["auc"])
                out[name]["minus_ego_ci"] = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
        res["A2"][kd] = out
    res["ckpt_sha256"] = cksha
    res["ext_sha256"] = json.load(open(f"{A1.EXT}/WUA_RH_s{a.seed}.json"))["npz_sha256"]
    res["tool_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(f"{WORK}/imag", exist_ok=True)
    out = f"{WORK}/imag/RESULT_IMAG_s{a.seed}.json"
    json.dump(res, open(out, "w"), indent=1)
    print("WROTE", out, res["seconds"], json.dumps(res["A2"].get("n_pairs")))


if __name__ == "__main__":
    main()
