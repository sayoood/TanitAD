#!/usr/bin/env python3
"""TIER 3 — BOUND the training-side effect (a true answer needs a retrain).

Two questions, and they have OPPOSITE answers — which is the whole point.

Q1  INFORMATION-THEORETIC redundancy: given (yaw_rate, speed), how much is
    left to know about `steer`? The IDM v2 diagnosis reports
    corr(steer, omega/v) = 0.9865 on PhysicalAI and concludes the channel
    "carries zero information beyond (yaw_rate, speed)". Reproduced here as an
    R^2 with an EPISODE-DISJOINT fit (train clips -> val clips), because a
    pooled in-sample correlation is not a predictive claim.

Q2  What flagship-v1 ACTUALLY RECEIVES. Its action vector is
    `[steer, accel, v0/10]` (`registry.MODEL_REGISTRY` §1.2; `rollout.collect`
    -> `append_ego`). **`yaw_rate` is NOT an input channel** — only the
    `--dyn-input`/`--yaw-input` arms get `yr0`, and v1 is `--speed-input` only.
    So the operative question is R^2(steer | accel, v0), NOT R^2(steer | w, v).

Q1 says the label is redundant *with respect to the dataset*. Q2 says it is the
only explicit rotational signal *with respect to the model*. Both are measured
here; the Tier-2 `zeroed` arm is the behavioural referee.

Fit protocol: closed-form ridge (lambda by 5-fold episode-disjoint CV on the
TRAIN clips only), evaluated out-of-sample on the val clips. R^2 CI is an
episode-cluster bootstrap over val clips (`taniteval.ci` semantics, reimplemented
here so the script runs on the dev box without the pod's taniteval).
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SHIPPED_WB = 2.9
TARGET_HZ = 10.0
CLIP_SPAN_US = 20.133e6
N_TARGET = 201
N_STACK_DROP = 2
DT = 0.1


def quaternion_yaw(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def build_table(rows, ego_root: Path, cam_have: dict, limit=None):
    by_chunk = defaultdict(list)
    for r in rows:
        by_chunk[r["chunk"]].append(r)
    cols = defaultdict(list)
    for ci, (chunk, rs) in enumerate(sorted(by_chunk.items())):
        with zipfile.ZipFile(ego_root / f"egomotion.chunk_{chunk:04d}.zip") as zf:
            names = {n.split(".")[0]: n for n in zf.namelist()}
            for r in rs:
                nm = names.get(r["clip_id"])
                if nm is None:
                    continue
                with zf.open(nm) as fh:
                    ego = pd.read_parquet(io.BytesIO(fh.read()))
                t = ego["timestamp"].to_numpy(np.float64)
                o = np.argsort(t)
                t = t[o]

                def c(name):
                    return ego[name].to_numpy(np.float64)[o]

                cp = cam_have.get(r["clip_id"])
                if cp is not None:
                    ts = pd.read_parquet(cp)
                    tc = next(x for x in ts.columns if "time" in x.lower())
                    tf = ts[tc].to_numpy(np.float64)
                    nt = max(int((tf[-1] - tf[0]) / 1e6 * TARGET_HZ), 4)
                    tq = np.linspace(tf[0], tf[-1], nt)
                else:
                    tq = np.linspace(t[0], t[0] + CLIP_SPAN_US, N_TARGET)
                curv = np.interp(tq, t, c("curvature"))
                v = np.hypot(np.interp(tq, t, c("vx")), np.interp(tq, t, c("vy")))
                ax = np.interp(tq, t, c("ax"))
                yaw_u = np.unwrap(quaternion_yaw(c("qx"), c("qy"), c("qz"), c("qw")))
                yaw = np.interp(tq, t, yaw_u)
                # yaw-rate at the SAME 10 Hz grid the pipeline would see
                w = np.gradient(yaw, DT)
                steer = np.arctan(SHIPPED_WB * curv)
                k = N_STACK_DROP
                n = len(tq) - k
                cols["steer"].append(steer[k:].astype(np.float32))
                cols["v"].append(v[k:].astype(np.float32))
                cols["w"].append(w[k:].astype(np.float32))
                cols["accel"].append(ax[k:].astype(np.float32))
                cols["curv"].append(curv[k:].astype(np.float32))
                cols["ep"].append(np.full(n, r["idx"] + (10 ** 6 if r["split"] == "val" else 0),
                                          np.int64))
                cols["is_val"].append(np.full(n, r["split"] == "val", bool))
                cols["wb"].append(np.full(n, r["wheelbase"], np.float32))
        if limit and ci + 1 >= limit:
            break
        if (ci + 1) % 40 == 0:
            print(f"[t3] {ci+1}/{len(by_chunk)} chunks", flush=True)
    return {k: np.concatenate(v) for k, v in cols.items()}


def ridge_fit(X, y, lam):
    Xb = np.c_[X, np.ones(len(X))]
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam                        # do not penalise the intercept
    return np.linalg.solve(A, Xb.T @ y)


def ridge_pred(X, wgt):
    return np.c_[X, np.ones(len(X))] @ wgt


def r2(y, yh):
    ss = ((y - yh) ** 2).sum()
    return float(1.0 - ss / ((y - y.mean()) ** 2).sum())


def cv_lambda(X, y, ep, lams=(1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4)):
    uniq = np.unique(ep)
    rng = np.random.default_rng(0)
    fold = {e: i % 5 for i, e in enumerate(rng.permutation(uniq))}
    f = np.array([fold[e] for e in ep])
    best, bl = -np.inf, lams[0]
    for lam in lams:
        sc = []
        for k in range(5):
            tr, te = f != k, f == k
            sc.append(r2(y[te], ridge_pred(X[te], ridge_fit(X[tr], y[tr], lam))))
        m = float(np.mean(sc))
        if m > best:
            best, bl = m, lam
    return bl, best


def boot_r2(y, yh, ep, n_boot=2000, seed=0):
    uniq, idx = np.unique(ep, return_inverse=True)
    by = [np.where(idx == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([by[p] for p in pick])
        vals.append(r2(y[sel], yh[sel]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def probe(tab, feats, name, n_boot=2000, mask=None):
    tr, te = ~tab["is_val"], tab["is_val"]
    if mask is not None:
        tr, te = tr & mask, te & mask
    X = np.column_stack([tab[f] for f in feats]).astype(np.float64)
    y = tab["steer"].astype(np.float64)
    lam, cvr2 = cv_lambda(X[tr], y[tr], tab["ep"][tr])
    w = ridge_fit(X[tr], y[tr], lam)
    yh = ridge_pred(X[te], w)
    lo, hi = boot_r2(y[te], yh, tab["ep"][te], n_boot=n_boot)
    return {"features": list(feats), "name": name, "lambda": lam,
            "cv_r2_train": round(cvr2, 4),
            "r2_out_of_sample": round(r2(y[te], yh), 4),
            "ci95_episode_cluster_bootstrap": [round(lo, 4), round(hi, 4)],
            "rmse_rad": round(float(np.sqrt(((y[te] - yh) ** 2).mean())), 6),
            "n_train_windows": int(tr.sum()), "n_val_windows": int(te.sum()),
            "n_val_episodes": int(len(np.unique(tab["ep"][te]))),
            "estimator": "episode_cluster_bootstrap", "n_boot": n_boot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ego-root",
                    default=r"C:\Users\Admin\tanitad-data\physicalai\labels\egomotion")
    ap.add_argument("--cam-root",
                    default=r"C:\Users\Admin\tanitad-data\physicalai\r0\camera_front_wide")
    a = ap.parse_args()
    scratch, out = Path(a.scratch), Path(a.out)
    cam_root = Path(a.cam_root)
    cam_have = {p.name.split(".")[0]: p
                for p in cam_root.glob("*.timestamps.parquet")} if cam_root.exists() else {}
    rows = json.load(open(scratch / "corpus_dims.json", encoding="utf-8"))
    tab = build_table(rows, Path(a.ego_root), cam_have)
    print(f"[t3] table: {len(tab['steer'])} samples, "
          f"{int(tab['is_val'].sum())} val", flush=True)

    # eps for numerical safety in the omega/v ratio feature
    v_safe = np.maximum(tab["v"], 0.5)
    tab["w_over_v"] = (tab["w"] / v_safe).astype(np.float32)
    tab["v0_scaled"] = (tab["v"] / 10.0).astype(np.float32)

    res = {"_meta": {
        "evidence_class": "MEASURED",
        "corpus": "physicalai-train-e438721ae894 (fit) -> val-0c5f7dac3b11 order (test)",
        "n_samples": int(len(tab["steer"])),
        "protocol": ("closed-form ridge; lambda by 5-fold EPISODE-DISJOINT CV on "
                     "train clips only; R^2 evaluated out-of-sample on val clips; "
                     "CI = episode-cluster bootstrap over val clips, B=2000"),
        "flagship_v1_action_vector": ["steer_road_rad", "accel_mps2", "v0/10"],
        "note_yaw_rate_not_an_input": (
            "yaw_rate reaches the predictor ONLY on --yaw-input/--dyn-input arms "
            "(rollout.ego_action_channels). flagship-v1 is --speed-input only."),
    }}

    mv = tab["v"] > 2.0            # the mask idm2_diag_labels.py:116 uses
    res["_meta"]["frac_samples_v_gt_2mps"] = round(float(mv.mean()), 4)
    res["Q1_information_theoretic"] = {
        "with_yaw_rate_and_speed": probe(tab, ("w", "v"), "omega, v"),
        "with_ratio_feature": probe(tab, ("w_over_v",), "omega/v"),
        "with_ratio_plus": probe(tab, ("w_over_v", "w", "v"), "omega/v, omega, v"),
        "with_ratio_v_gt_2mps": probe(tab, ("w_over_v",), "omega/v  [v>2 m/s]",
                                      mask=mv),
        "with_ratio_plus_v_gt_2mps": probe(tab, ("w_over_v", "w", "v"),
                                           "omega/v, omega, v  [v>2 m/s]", mask=mv),
        "corr_steer_vs_w_over_v_v_gt_2mps": round(float(np.corrcoef(
            tab["steer"][mv], tab["w_over_v"][mv])[0, 1]), 4),
        "corr_steer_vs_w_over_v": round(float(np.corrcoef(
            tab["steer"], tab["w_over_v"])[0, 1]), 4),
        "corr_steer_vs_w": round(float(np.corrcoef(tab["steer"], tab["w"])[0, 1]), 4),
        "corr_steer_vs_curvature": round(float(np.corrcoef(
            tab["steer"], tab["curv"])[0, 1]), 4),
    }
    res["Q2_what_flagship_v1_receives"] = {
        "from_accel_and_v0": probe(tab, ("accel", "v0_scaled"), "accel, v0/10"),
        "from_accel_only": probe(tab, ("accel",), "accel"),
        "from_v0_only": probe(tab, ("v0_scaled",), "v0/10"),
        "from_accel_and_v0_v_gt_2mps": probe(tab, ("accel", "v0_scaled"),
                                             "accel, v0/10  [v>2 m/s]", mask=mv),
    }
    # how much of the steer channel's variance the wheelbase error moves,
    # expressed against the residual the other channels CANNOT explain
    q1 = res["Q1_information_theoretic"]["with_yaw_rate_and_speed"]["r2_out_of_sample"]
    q2 = res["Q2_what_flagship_v1_receives"]["from_accel_and_v0"]["r2_out_of_sample"]
    d = tab["steer"] * (tab["wb"] / SHIPPED_WB - 1.0)     # small-angle delta
    res["bound"] = {
        "steer_std_rad": round(float(tab["steer"].std()), 6),
        "delta_std_rad": round(float(d.std()), 6),
        "delta_std_over_steer_std": round(float(d.std() / tab["steer"].std()), 4),
        "unexplained_var_frac_given_omega_v": round(1 - q1, 4),
        "unexplained_var_frac_given_v1_inputs": round(1 - q2, 4),
        "delta_var_frac_of_steer_var": round(float((d.std() / tab["steer"].std()) ** 2), 5),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "tier3_redundancy.json").write_text(json.dumps(res, indent=2),
                                               encoding="utf-8")
    print("[t3] wrote", out / "tier3_redundancy.json")
    print(json.dumps({k: v for k, v in res.items() if k != "_meta"},
                     indent=1)[:2500])


if __name__ == "__main__":
    main()
