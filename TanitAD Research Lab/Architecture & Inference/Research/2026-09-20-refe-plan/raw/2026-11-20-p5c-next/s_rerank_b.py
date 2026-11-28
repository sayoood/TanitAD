#!/usr/bin/env python3
"""S-RERANK-B (PREREG_S_RERANK_B.md, sha256 7fa8d979... frozen before this run): within-token gain-over-pick switcher, tau by
5-fold GroupKFold on TUNE logs. Reuses design A's loader, features, read and bootstrap (s_rerank.py) unchanged.
Usage: s_rerank_b.py --arm C1=<rows glob>=<npz> --arm C0=... --arm base=... --out OUT.json [--boot 10000]"""
import argparse, json, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import s_rerank as A

TAUS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, np.inf]


def featB(d, mode):
    X = d["X"]; n = len(X); ar = np.arange(n); pk = d["pick"]
    head = X[..., :7]                                        # 6 sigmoid head outputs + navsim_v1 aggregate
    hp = head[ar, pk][:, None, :]
    F = [X, head - hp, np.broadcast_to(hp, head.shape)]
    gain = d["tr"]["pdms"] - d["tr"]["pdms"][ar, pk][:, None]
    if mode == "pos":
        F.append(gain[..., None].astype(np.float32))
    return np.concatenate(F, -1).astype(np.float32), gain


def labels(d, gain, mode, rng):
    if mode != "red":
        return gain
    pdms = d["tr"]["pdms"]
    perm = np.stack([r[rng.permutation(r.shape[0])] for r in pdms])
    return perm - perm[np.arange(len(perm)), d["pick"]][:, None]


def rows(XB, y, tok_idx, pick):
    m = np.ones(XB.shape[:2], bool); m[np.arange(len(XB)), pick] = False
    m &= np.isin(np.arange(len(XB)), tok_idx)[:, None]
    return XB[m], y[m]


def choose(pred, pick):
    p = pred.copy(); p[np.arange(len(p)), pick] = -np.inf
    k = p.argmax(1); return k, p[np.arange(len(p)), k]


def run(arms, mode, rng):
    D = []
    for d in arms:
        XB, gain = featB(d, mode); y = labels(d, gain, mode, rng)
        D.append((d, XB, y))
    # TUNE groups pooled across arms by log name (a log never straddles folds)
    tune_logs = sorted(set(np.concatenate([d["logs"][d["tune"]] for d in arms])))
    gid = {l: i for i, l in enumerate(tune_logs)}
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(tune_logs)), groups=np.arange(len(tune_logs))))
    oof = {d["name"]: np.full(d["X"].shape[:2], np.nan) for d in arms}
    for tr_g, te_g in folds:
        trs, tes = set(tr_g), set(te_g)
        Xs, ys = [], []
        for d, XB, y in D:
            ti = np.where(d["tune"] & np.array([gid.get(l, -1) in trs for l in d["logs"]]))[0]
            a, b = rows(XB, y, ti, d["pick"]); Xs.append(a); ys.append(b)
        m = HistGradientBoostingRegressor(**A.HP).fit(np.concatenate(Xs), np.concatenate(ys))
        for d, XB, y in D:
            te = np.where(d["tune"] & np.array([gid.get(l, -1) in tes for l in d["logs"]]))[0]
            oof[d["name"]][te] = m.predict(XB[te].reshape(-1, XB.shape[-1])).reshape(len(te), -1)
    score = {}
    for tau in TAUS:
        tot, cnt = 0.0, 0
        for d, XB, y in D:
            T = np.where(d["tune"])[0]; k, g = choose(oof[d["name"]][T], d["pick"][T])
            sel = np.where(g > tau, k, d["pick"][T])
            tot += d["tr"]["pdms"][T, sel].sum(); cnt += len(T)
        score[str(tau)] = tot / cnt * 100
    best = max(score.values())
    tau = max(t for t in TAUS if score[str(t)] == best)                       # ties -> larger tau
    Xs, ys = [], []
    for d, XB, y in D:
        a, b = rows(XB, y, np.where(d["tune"])[0], d["pick"]); Xs.append(a); ys.append(b)
    m = HistGradientBoostingRegressor(**A.HP).fit(np.concatenate(Xs), np.concatenate(ys))
    sels = {}
    for d, XB, y in D:
        pred = m.predict(XB.reshape(-1, XB.shape[-1])).reshape(XB.shape[:2])
        k, g = choose(pred, d["pick"]); sels[d["name"]] = np.where(g > tau, k, d["pick"])
    return sels, tau, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args(); t0 = time.time()
    arms = [A.load(s) for s in a.arm]; P = arms[0]
    eval_logs = set(P["logs"][~P["tune"]])
    for d in arms:
        assert not set(d["logs"][d["tune"]]) & eval_logs, f"{d['name']}: an EVAL log is in the training rows"
    rng = np.random.default_rng(20261120); res, taus, cv = {}, {}, {}
    for mode in ("main", "red", "pos"):
        sels, taus[mode], cv[mode] = run(arms, mode, rng)
        res[mode] = {d["name"]: A.read(d, sels[d["name"]], a.boot if (mode == "main" or d is P) else 1000) for d in arms}
        print("ZZRERANKB", mode, "tau", taus[mode], {k: round(v["pdms"]["point"], 3) for k, v in res[mode].items()},
              {k: v["n_differs_from_pick"] for k, v in res[mode].items()}, f"{time.time() - t0:.0f}s", flush=True)
    ctrl = {"RED_delta_point_le_0": res["red"][P["name"]]["pdms"]["point"] <= 0,
            "POS_regret_lt_1": res["pos"][P["name"]]["regret_x100"]["point"] < 1.0}
    pm = res["main"][P["name"]]; adverse = [k for k in A.SUBS if pm[k]["hi"] < 0]
    reps = {d["name"]: res["main"][d["name"]]["pdms"]["point"] for d in arms[1:]}
    if not all(ctrl.values()):
        verdict = "INVALID (control failed)"
    else:
        verdict = "SIGNAL (second design on the split)" if (pm["pdms"]["lo"] > 0 and not adverse and all(v > 0 for v in reps.values())) else "NO SIGNAL"
    out = {"card": "PREREG_S_RERANK_B.md", "hp": A.HP, "taus": taus, "cv_oof_tune_pdms_by_tau": cv, "controls": ctrl, "results": res,
           "eval_separated_adverse": adverse, "replicate_delta_points": reps, "verdict": verdict,
           "estimator": "paired log-cluster bootstrap over EVAL logs, 10000 draws (RED/POS replicates 1000), seed 20260927; LOGS only",
           "tier": "T1 executed (NAVSIM v1.1 LQR-tracked)", "evidence": "MEASURED (this read); DIAGNOSTIC: trained on navtest TUNE logs",
           "runtime_s": round(time.time() - t0, 1)}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    print("ZZRERANKB_VERDICT", verdict, "tau", taus["main"], "C1 delta", pm["pdms"], "adverse", adverse, "reps", reps, "controls", ctrl)
    return 0 if all(ctrl.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
