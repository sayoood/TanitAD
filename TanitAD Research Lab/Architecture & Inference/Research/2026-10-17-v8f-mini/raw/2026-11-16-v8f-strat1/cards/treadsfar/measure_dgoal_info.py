"""PREREG_V8F_DGOAL scoping measurement (labels only, zero training, CPU).

Question: does DETECTED strategic information (the augmented label release's weather, surface, illumination,
period of day, country) carry information about a strategic SPEED GOAL in the 6-30 s band, beyond the
current speed?  If it carries none, a goal derived from detected information has nothing to pass on.

Labels only (labels may use ego, PI 2026-08-03): v_now = |v| at an anchor t0 (egomotion), v_goal = mean |v|
over [t0 + 6 s, t0 + 30 s].  Anchors every 2 s where t0 + 30 s lies inside the log.  Detected info = the
clip's modal valid label per variable (strat_labels_table.npz, train split).  Model: least squares with a
clip-grouped 2-fold split (fold = clip index parity), so no clip is scored by a fit that saw it.

Run:  python -I measure_dgoal_info.py --out DGOAL_INFO.json   (a few minutes: 4,572 parquet reads)
"""
import argparse, hashlib, json, os
import numpy as np
import pandas as pd

LAB = "F:/TanitAD-scratch/strat1/read/inputs/strat_labels_table.npz"
EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/{}.parquet"
VARS = ("weather", "surface", "illumination", "period_of_day", "country")
T_LO, T_HI, STEP = 6.0, 30.0, 2.0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clip_mode(ids, valid):
    x = ids[valid.astype(bool)]
    if x.size == 0:
        return -1
    v, c = np.unique(x, return_counts=True)
    return int(v[np.argmax(c)])


def fit_cv(X, y, fold):
    pred = np.empty_like(y)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        w, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred[te] = X[te] @ w
    ss = ((y - y.mean()) ** 2).sum()
    return 1.0 - ((y - pred) ** 2).sum() / ss, float(np.sqrt(((y - pred) ** 2).mean()))


def onehot(ids, n):
    out = np.zeros((len(ids), n))
    ok = ids >= 0
    out[np.where(ok)[0], ids[ok]] = 1.0
    return out[:, 1:] if n > 1 else out          # drop one level (the intercept carries it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Z = np.load(LAB, allow_pickle=True)
    clips, off, n = Z["train_clip"], Z["train_off"], Z["train_n"]
    lab = {v: [] for v in VARS}
    rows = []
    missing = 0
    for ci, c in enumerate(clips):
        p = EGO.format(c)
        if not os.path.exists(p):
            missing += 1
            continue
        s = slice(int(off[ci]), int(off[ci]) + int(n[ci]))
        modes = {v: clip_mode(Z[f"train_{v}_id"][s], Z[f"train_{v}_valid"][s]) for v in VARS}
        d = pd.read_parquet(p, columns=["timestamp", "vx", "vy"])
        t = d["timestamp"].to_numpy(np.float64) * 1e-6
        sp = np.hypot(d["vx"].to_numpy(np.float64), d["vy"].to_numpy(np.float64))
        t0s = np.arange(max(t[0], 0.0), t[-1] - T_HI, STEP)
        for t0 in t0s:
            i0 = int(np.searchsorted(t, t0))
            m = (t >= t0 + T_LO) & (t <= t0 + T_HI)
            if m.sum() < 10:
                continue
            rows.append((ci, sp[i0], sp[m].mean()))
            for v in VARS:
                lab[v].append(modes[v])
    R = np.array(rows)
    ci, v_now, v_goal = R[:, 0].astype(int), R[:, 1], R[:, 2]
    fold = ci % 2
    L = {v: np.array(lab[v]) for v in VARS}
    nvoc = {v: len(Z[f"vocab_{v}"]) for v in VARS}
    base = np.stack([np.ones_like(v_now), v_now, v_now ** 2], 1)
    rep = {"inputs": {LAB: sha(LAB)}, "egomotion_dir": os.path.dirname(EGO), "clips_used": int(len(np.unique(ci))),
           "clips_missing_egomotion": missing, "anchors": int(len(R)), "band_s": [T_LO, T_HI], "anchor_step_s": STEP,
           "split": "clip-grouped 2-fold, fold = clip index parity", "v_goal_mean": float(v_goal.mean()),
           "v_now_mean": float(v_now.mean())}
    r2b, rmb = fit_cv(base, v_goal, fold)
    rep["R2_base_vnow"] = {"R2": round(r2b, 4), "rmse_mps": round(rmb, 4)}
    allX = [base]
    per = {}
    for v in VARS:
        X = np.concatenate([base, onehot(L[v], nvoc[v])], 1)
        r2, rm = fit_cv(X, v_goal, fold)
        per[v] = {"R2": round(r2, 4), "dR2_vs_base": round(r2 - r2b, 4), "rmse_mps": round(rm, 4),
                  "class_mean_dv": {str(Z[f"vocab_{v}"][k]): {"n_anchors": int((L[v] == k).sum()), "n_clips": int(len(np.unique(ci[L[v] == k]))),
                                                                  "mean_v_goal_minus_v_now": round(float((v_goal - v_now)[L[v] == k].mean()), 3) if (L[v] == k).any() else None}
                                    for k in range(nvoc[v])}}
        allX.append(onehot(L[v], nvoc[v]))
    rep["per_variable"] = per
    r2a, rma = fit_cv(np.concatenate(allX, 1), v_goal, fold)
    rep["all_detected"] = {"R2": round(r2a, 4), "dR2_vs_base": round(r2a - r2b, 4), "rmse_mps": round(rma, 4)}
    nc = [v for v in VARS if v != "country"]
    r2n, rmn = fit_cv(np.concatenate([base] + [onehot(L[v], nvoc[v]) for v in nc], 1), v_goal, fold)
    rep["detected_without_country"] = {"R2": round(r2n, 4), "dR2_vs_base": round(r2n - r2b, 4), "rmse_mps": round(rmn, 4)}
    # control: a permuted-label arm (clip labels deranged across clips, seed 0) must give dR2 <= ~0
    rng = np.random.default_rng(0)
    uc = np.unique(ci)
    perm = dict(zip(uc, rng.permutation(uc)))
    first = {c: int(np.where(ci == c)[0][0]) for c in uc}
    Lp = {v: np.array([L[v][first[perm[c]]] for c in ci]) for v in VARS}
    r2p, _ = fit_cv(np.concatenate([base] + [onehot(Lp[v], nvoc[v]) for v in VARS], 1), v_goal, fold)
    rep["CONTROL_permuted_labels"] = {"R2": round(r2p, 4), "dR2_vs_base": round(r2p - r2b, 4), "must": "dR2 <= 0.002"}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: rep[k] for k in rep if k != "per_variable"}, indent=1))
    print(json.dumps({v: {k: per[v][k] for k in ("R2", "dR2_vs_base")} for v in VARS}))


if __name__ == "__main__":
    main()
