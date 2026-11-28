#!/usr/bin/env python3
"""S-RERANK (PREREG_S_RERANK.md, frozen sha256 c09e8bed... before this run): a learned re-ranker over C1's frozen outputs + geometry.

Trains HistGradientBoostingRegressor (fixed hyper-parameters) on TUNE-log proposals of C1 + C0 + base (label = tracked PDMS, T1),
selects argmax predicted PDMS, reads delta vs the navsim_v1 pick on EVAL logs (paired log-cluster bootstrap, 10,000, seed 20260927).
Controls: RED (labels permuted within token) must read delta point < 0; POS (true PDMS as a feature) must reach regret < 1.0.
Usage: s_rerank.py --arm C1=<rows glob>=<proposals.npz> --arm C0=... --arm base=... --out OUT.json [--boot 10000]
"""
import argparse, glob, hashlib, json, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

SUBS = ("nc", "dac", "ttc", "ep", "c")
HP = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=0)


def agg32(L):
    p = 1.0 / (1.0 + np.exp(-np.asarray(L, np.float32)))
    return p, (p[..., 0] * p[..., 1]) * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12


def features(P, L):
    """P [n,64,8,3] poses (x, y, heading), L [n,64,6] logits -> [n,64,F] float32.  Model outputs only."""
    p, g = agg32(L)
    n, K = g.shape
    rank = np.argsort(np.argsort(-g, 1), 1) / (K - 1)
    gap = g - g.max(1, keepdims=True)
    xy = P[..., :2].astype(np.float32)
    end = P[:, :, -1, :]
    p1 = P[:, :, 1, :2]
    seg = np.diff(np.concatenate([np.zeros_like(xy[:, :, :1]), xy], 2), axis=2)
    plen = np.linalg.norm(seg, axis=-1).sum(-1)
    dh = np.abs(np.diff(np.concatenate([np.zeros_like(P[:, :, :1, 2]), P[..., 2]], 2), axis=2)).max(-1)
    v = np.linalg.norm(seg, axis=-1) / 0.5
    acc = np.abs(np.diff(v, axis=-1)).max(-1) / 0.5
    jerk = np.abs(np.diff(v, n=2, axis=-1)).max(-1) / 0.25
    w = np.exp((g - g.max(1, keepdims=True)) * 50.0); w /= w.sum(1, keepdims=True)
    cons = (w[..., None, None] * xy).sum(1)
    dcons = np.linalg.norm(xy - cons[:, None], axis=-1).mean(-1)
    pk = g.argmax(1)
    dpick = np.linalg.norm(xy - xy[np.arange(n), pk][:, None], axis=-1).mean(-1)
    dmed = np.linalg.norm(end[..., :2] - np.median(end[..., :2], axis=1, keepdims=True), axis=-1)
    F = [p[..., i] for i in range(p.shape[-1])] + [g, rank, gap, end[..., 0], end[..., 1], end[..., 2], p1[..., 0], p1[..., 1], plen, dh,
                                                  acc, jerk, dcons, dpick, dmed]
    return np.stack(F, -1).astype(np.float32)


def load(spec):
    name, g, npz = spec.split("=", 2)
    R = {}
    for f in sorted(glob.glob(g)):
        for ln in open(f, encoding="utf-8"):
            r = json.loads(ln)
            if "error" not in r:
                R[r["token"]] = r
    z = np.load(npz, allow_pickle=False)
    tok = z["token"].astype(str)
    assert set(tok) == set(R), f"{name}: table tokens != dump tokens"
    logs = np.array([R[t]["log"] for t in tok])
    tr = {k: np.array([R[t]["tr"][k] for t in tok], np.float64) for k in SUBS + ("pdms",)}
    pick = z["pick"]
    assert (agg32(z["logits"])[1].argmax(1) == pick).all(), f"{name}: CTRL-S (float32) failed"
    X = features(z["proposals"], z["logits"])
    tune = np.array([int(hashlib.sha1(l.encode()).hexdigest()[-1], 16) % 2 == 0 for l in logs])
    return {"name": name, "tok": tok, "logs": logs, "tr": tr, "pick": pick, "X": X, "tune": tune}


def boot(x, logs, B, seed=20260927):
    ul = sorted(set(logs)); ix = [np.where(logs == l)[0] for l in ul]
    rng = np.random.default_rng(seed); bs = np.empty(B)
    for b in range(B):
        bs[b] = x[np.concatenate([ix[i] for i in rng.integers(0, len(ul), len(ul))])].mean()
    return {"point": float(x.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def read(d, sel, B):
    E = ~d["tune"]; ar = np.arange(len(d["tok"]))
    out = {k: boot((d["tr"][k][ar, sel] - d["tr"][k][ar, d["pick"]])[E] * 100, d["logs"][E], B) for k in ("pdms",) + SUBS}
    orc = d["tr"]["pdms"].max(1)
    out["regret_x100"] = boot((orc - d["tr"]["pdms"][ar, sel])[E] * 100, d["logs"][E], B)
    out["level_pdms_x100"] = float(d["tr"]["pdms"][ar, sel][E].mean() * 100)
    out["pick_level_pdms_x100"] = float(d["tr"]["pdms"][ar, d["pick"]][E].mean() * 100)
    out["n_differs_from_pick"] = int((sel != d["pick"])[E].sum()); out["n_tokens"] = int(E.sum()); out["n_logs"] = len(set(d["logs"][E]))
    return out


def fit_select(arms, mode, rng):
    Xs, ys = [], []
    for d in arms:
        T = d["tune"]; X = d["X"][T]; y = d["tr"]["pdms"][T].copy()
        if mode == "red":
            y = np.stack([r[rng.permutation(r.shape[0])] for r in y])
        if mode == "pos":
            X = np.concatenate([X, d["tr"]["pdms"][T][..., None].astype(np.float32)], -1)
        Xs.append(X.reshape(-1, X.shape[-1])); ys.append(y.reshape(-1))
    m = HistGradientBoostingRegressor(**HP).fit(np.concatenate(Xs), np.concatenate(ys))
    sels = {}
    for d in arms:
        X = d["X"] if mode != "pos" else np.concatenate([d["X"], d["tr"]["pdms"][..., None].astype(np.float32)], -1)
        pr = m.predict(X.reshape(-1, X.shape[-1])).reshape(X.shape[:2])
        sels[d["name"]] = pr.argmax(1)
    return sels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    t0 = time.time()
    arms = [load(s) for s in a.arm]
    P = arms[0]
    for d in arms:
        assert not set(d["logs"][d["tune"]]) & set(d["logs"][~d["tune"]]), "TUNE/EVAL log overlap"
    eval_logs = set(P["logs"][~P["tune"]])
    for d in arms:
        assert not set(d["logs"][d["tune"]]) & eval_logs, f"{d['name']}: an EVAL log is in the training rows"
    rng = np.random.default_rng(20261120)
    res = {}
    for mode in ("main", "red", "pos"):
        sels = fit_select(arms, mode, rng)
        res[mode] = {d["name"]: read(d, sels[d["name"]], a.boot if (mode == "main" or d is P) else 1000) for d in arms}
        print("ZZRERANK", mode, {k: round(v["pdms"]["point"], 3) for k, v in res[mode].items()}, f"{time.time() - t0:.0f}s", flush=True)
    ctrl = {"RED_delta_point_lt_0": res["red"][P["name"]]["pdms"]["point"] < 0,
            "POS_regret_lt_1": res["pos"][P["name"]]["regret_x100"]["point"] < 1.0}
    m = res["main"]; pm = m[P["name"]]
    adverse = [k for k in SUBS if pm[k]["hi"] < 0]
    reps = {d["name"]: m[d["name"]]["pdms"]["point"] for d in arms[1:]}
    if not all(ctrl.values()):
        verdict = "INVALID (control failed)"
    else:
        verdict = "SIGNAL" if (pm["pdms"]["lo"] > 0 and not adverse and all(v > 0 for v in reps.values())) else "NO SIGNAL"
    out = {"card": "PREREG_S_RERANK.md", "hp": HP, "controls": ctrl, "results": res, "eval_separated_adverse": adverse,
           "replicate_delta_points": reps, "verdict": verdict,
           "estimator": "paired log-cluster bootstrap over EVAL logs, 10000 draws (RED/POS replicates 1000), seed 20260927; LOGS only",
           "tier": "T1 executed (NAVSIM v1.1 LQR-tracked)", "evidence": "MEASURED (this read); DIAGNOSTIC: trained on navtest TUNE logs",
           "runtime_s": round(time.time() - t0, 1)}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print("ZZRERANK_VERDICT", verdict, "C1 delta", pm["pdms"], "adverse", adverse, "reps", reps, "controls", ctrl)
    return 0 if all(ctrl.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
