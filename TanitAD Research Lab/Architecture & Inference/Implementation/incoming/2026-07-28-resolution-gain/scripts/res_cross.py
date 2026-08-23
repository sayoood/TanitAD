"""RESOLUTION-GAIN — the CROSS-LADDER contrasts, with proper paired intervals.

`res_eval.py` adjudicates each ladder against its own baseline, which is what the pre-registration
requires. Two cross-ladder contrasts are quoted in the report's headline and therefore need their
own paired CI rather than a difference of two point estimates:

  * `V5_640  vs B_today`  — the wide 120 deg / 640-token frame against today's deployed input.
    ⛔ C34: these differ in FIELD, PROJECTION and TOKEN COUNT. The contrast is reported as an
    end-to-end fact and is explicitly NOT attributed to any one of the three.
  * `D_today vs B_today`  — the same pair at **MATCHED angular resolution** (both 4.6426 px/deg).
    This is the one that licenses the *negative* attribution in the report: whatever the wide frame
    buys, it is not angular resolution, because resolution is held equal here.

Only the three arms involved are re-fitted; everything else (split, ranks, lambda search, estimator)
is `res_eval`'s own code, imported rather than re-implemented.

usage:  python res_cross.py <feats_dir> <labels_dir> <out_json>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import res_eval as RE                                                   # noqa: E402
from taniteval.rank_metrics import average_precision                    # noqa: E402

ARMS = ["V5_640", "D_today", "B_today"]
PAIRS = [("V5_640", "B_today"), ("D_today", "B_today"), ("V5_640", "D_today")]


def main():
    feats_dir, labels_dir, out_json = sys.argv[1:4]
    L = np.load(os.path.join(labels_dir, "fov_labels.npz"))
    meta = {m["k"]: m for m in json.load(open(os.path.join(labels_dir, "fov_meta.json")))}
    files = sorted(f for f in os.listdir(feats_dir) if f.startswith("clip_"))
    X = {a: [] for a in ARMS}
    Y = {s: [] for s in RE.SITS}
    V = {s: [] for s in RE.SITS}
    KIN, EID, SIDE = {"speed": [], "absyaw": []}, [], []
    for f in files:
        k = int(f[5:10])
        z = np.load(os.path.join(feats_dir, f))
        t = z["t"]
        for a in ARMS:
            X[a].append(z[a].astype(np.float32))
        for s in RE.SITS:
            Y[s].append(L[f"c{k}_y_{s}"][t])
            V[s].append(L[f"c{k}_valid_{s}"][t])
        K = RE.kinematics(L[f"c{k}_poses"].astype(np.float64))
        KIN["speed"].append(K["v"][t])
        KIN["absyaw"].append(np.abs(K["omega"])[t])
        EID.append(np.full(len(t), k))
        SIDE.append(np.full(len(t), meta[k]["side"] == "HELDOUT"))
    for a in ARMS:
        X[a] = np.concatenate(X[a])
    for s in RE.SITS:
        Y[s] = np.concatenate(Y[s]).astype(np.float64)
        V[s] = np.concatenate(V[s]).astype(bool)
    for t_ in KIN:
        KIN[t_] = np.concatenate(KIN[t_]).astype(np.float64)
    EID, HELD = np.concatenate(EID), np.concatenate(SIDE)

    out = {"arms": ARMS, "n_boot": RE.B_BOOT,
           "estimator": "paired clip-cluster bootstrap (taniteval.ci._draws), unit = clip",
           "C34_WARNING": "V5_640 vs B_today differs in FIELD, PROJECTION and TOKEN COUNT; "
                          "D_today vs B_today holds ANGULAR RESOLUTION equal at 4.6426 px/deg but "
                          "still differs in field, projection and token count. Neither contrast "
                          "attributes the effect to a single cause; the admissible reading is the "
                          "NEGATIVE one — it is not angular resolution.",
           "P1_AP": {}, "P2_R2": {}}

    # ---- P1: situation AP ----
    for s in RE.SITS:
        m_tr, m_he = V[s] & ~HELD, V[s] & HELD
        if m_tr.sum() < 100 or Y[s][m_tr].sum() < 5:
            continue
        chunks = np.array([meta[int(k)]["chunk"] for k in EID[m_tr]])
        uch = np.unique(chunks)
        fid = np.array([{c: i % 5 for i, c in enumerate(uch)}[c] for c in chunks])
        sc = {}
        for a in ARMS:
            mu, W, _ = RE.fit_pca(X[a][m_tr], RE.RANK)
            Ztr, Zhe = (X[a][m_tr] - mu) @ W, (X[a][m_he] - mu) @ W
            sd = Ztr.std(0, keepdims=True) + 1e-6
            Ztr, Zhe = Ztr / sd, Zhe / sd
            best, best_ap = RE.LAMBDAS[0], -1.0
            for lam in RE.LAMBDAS:
                oof = np.zeros(len(Ztr))
                for fo in range(5):
                    tr, va = fid != fo, fid == fo
                    if Y[s][m_tr][tr].sum() < 2 or Y[s][m_tr][va].sum() < 1:
                        continue
                    oof[va] = RE.ridge_pred(Ztr[va], RE.ridge_fit(Ztr[tr], Y[s][m_tr][tr], lam))
                v = average_precision(Y[s][m_tr], oof)
                if v > best_ap:
                    best, best_ap = lam, v
            sc[a] = RE.ridge_pred(Zhe, RE.ridge_fit(Ztr, Y[s][m_tr], best))
        out["P1_AP"][s] = {f"{a}_minus_{b}": RE.paired_delta(
            Y[s][m_he], sc[a], sc[b], EID[m_he], average_precision) for a, b in PAIRS}

    # ---- P2: kinematic R^2 ----
    m_tr, m_he = ~HELD, HELD
    chunks = np.array([meta[int(k)]["chunk"] for k in EID[m_tr]])
    uch = np.unique(chunks)
    fid = np.array([{c: i % 5 for i, c in enumerate(uch)}[c] for c in chunks])
    for tgt, yall in KIN.items():
        ytr, yhe, pr = yall[m_tr], yall[m_he], {}
        for a in ARMS:
            Xtr, Xhe = X[a][m_tr].astype(np.float64), X[a][m_he].astype(np.float64)
            mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
            Ztr, Zhe = (Xtr - mu) / sd, (Xhe - mu) / sd
            best, best_r2 = RE.LAMBDAS_REG[0], -1e18
            for lam in RE.LAMBDAS_REG:
                oof = np.zeros(len(Ztr))
                for fo in range(5):
                    tr, va = fid != fo, fid == fo
                    oof[va] = RE.ridge_pred(Ztr[va], RE.ridge_fit(Ztr[tr], ytr[tr], lam))
                v = RE.r2(ytr, oof)
                if v > best_r2:
                    best, best_r2 = lam, v
            pr[a] = RE.ridge_pred(Zhe, RE.ridge_fit(Ztr, ytr, best))
        out["P2_R2"][tgt] = {f"{a}_minus_{b}": RE.paired_delta(
            yhe, pr[a], pr[b], EID[m_he], RE.r2) for a, b in PAIRS}

    json.dump(out, open(out_json, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
