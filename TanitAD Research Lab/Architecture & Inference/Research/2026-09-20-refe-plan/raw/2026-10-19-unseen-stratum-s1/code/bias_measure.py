"""S1 step 2 (sampler bias): on the navtrain logs that sit in the val split (the 214), compare the navtrain tokens with ALL candidates of the
frozen rule, on features that need NO label: ego speed at the token, 4-s displacement and 4-s heading change of the LOGGED ego, and the nuPlan
scenario_tag types at the token.  Ego pose is DB geometry (a model INPUT at inference, not a label); no PDM score, no model output is read.
This script reads ego_pose / scenario_tag ONLY for navtrain logs, never for U.

    python bias_measure.py [<db_dir> <tag>]     -> raw/bias_<tag>.json
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import identity_check as IC  # noqa: E402
import unseen_token_rule as R  # noqa: E402

DB, TAG = IC.DB, "214"
if len(sys.argv) > 2:
    DB, TAG = sys.argv[1], sys.argv[2]
OUT = f"C:/Users/Admin/par18/unseen_s1/raw/bias_{TAG}.json"


def features(path, lp_idx):
    """per candidate lidar_pc index: speed, 4-s displacement, 4-s |heading change|, longitudinal accel, scenario tags (set)"""
    c = R._ro(path)
    try:
        rows = c.execute("SELECT e.x, e.y, e.vx, e.vy, e.qw, e.qx, e.qy, e.qz, e.acceleration_x FROM lidar_pc l JOIN ego_pose e ON e.token = l.ego_pose_token ORDER BY l.timestamp").fetchall()
        tags = collections.defaultdict(set)
        order = {r[0]: i for i, r in enumerate(c.execute("SELECT lower(hex(token)) FROM lidar_pc ORDER BY timestamp"))}
        for tok, ty in c.execute("SELECT lower(hex(lidar_pc_token)), type FROM scenario_tag"):
            tags[order[tok]].add(ty)
    finally:
        c.close()
    a = np.array(rows, dtype=float)
    yaw = np.arctan2(2 * (a[:, 4] * a[:, 7] + a[:, 5] * a[:, 6]), 1 - 2 * (a[:, 6] ** 2 + a[:, 7] ** 2))
    i = np.asarray(lp_idx)
    j = np.minimum(i + 80, len(a) - 1)        # +4 s at 20 Hz
    v = np.hypot(a[i, 2], a[i, 3])
    disp = np.hypot(a[j, 0] - a[i, 0], a[j, 1] - a[i, 1])
    dy = np.abs((yaw[j] - yaw[i] + np.pi) % (2 * np.pi) - np.pi) * 180 / np.pi
    return v, disp, dy, a[i, 8], [tags.get(int(k), set()) for k in i]


def q(x):
    return {k: round(float(v), 3) for k, v in zip(("p5", "p25", "p50", "p75", "p95", "mean"), list(np.percentile(x, [5, 25, 50, 75, 95])) + [np.mean(x)])}


def main():
    known = IC.load_known()
    logs = sorted(l for l in known if os.path.exists(f"{DB}/{l}.db"))
    V, D, Y, A, G, Lbl, LG = [], [], [], [], [], [], []
    for n, lg in enumerate(logs):
        rec = R.read_log(f"{DB}/{lg}.db")
        c = R.candidates(rec)
        v, d, y, a, g = features(f"{DB}/{lg}.db", c["lp_idx"])
        lab = np.array([t in known[lg] for t in c["tokens"]])
        V.append(v); D.append(d); Y.append(y); A.append(a); G.extend(g); Lbl.append(lab); LG.extend([n] * len(lab))
    V, D, Y, A = map(np.concatenate, (V, D, Y, A))
    Lbl, LG = np.concatenate(Lbl), np.array(LG)
    res = {"what": "label-free covariates of navtrain tokens vs all candidates of the frozen rule, on navtrain logs", "n_logs": len(logs), "n_candidates": int(len(Lbl)), "n_navtrain": int(Lbl.sum()),
           "evidence": "MEASURED (read-only DB: ego_pose, scenario_tag; no PDM score)"}
    feats = {"speed_mps": V, "disp_4s_m": D, "abs_dyaw_4s_deg": Y, "abs_acc_x": np.abs(A)}
    res["quantiles"] = {k: {"navtrain": q(x[Lbl]), "all_candidates": q(x), "non_navtrain_candidates": q(x[~Lbl])} for k, x in feats.items()}
    thr = {"stationary_speed_lt_0.5": V < 0.5, "slow_speed_lt_2": V < 2.0, "progress_4s_lt_2m": D < 2.0, "turning_dyaw_gt_15deg": Y > 15.0}
    res["shares"] = {k: {"navtrain": float(m[Lbl].mean()), "all_candidates": float(m.mean()), "non_navtrain": float(m[~Lbl].mean())} for k, m in thr.items()}
    tc_all, tc_nt = collections.Counter(), collections.Counter()
    for g, l in zip(G, Lbl):
        for t in g:
            tc_all[t] += 1
            if l:
                tc_nt[t] += 1
    types = [t for t, _ in tc_all.most_common(25)]
    res["scenario_tag_share_at_token"] = {t: {"navtrain": tc_nt[t] / Lbl.sum(), "all_candidates": tc_all[t] / len(Lbl)} for t in types}
    res["no_tag_share"] = {"navtrain": float(np.mean([len(g) == 0 for g, l in zip(G, Lbl) if l])), "all_candidates": float(np.mean([len(g) == 0 for g in G]))}
    # per-log navtrain density and what a uniform within-log draw sees
    dens = np.array([Lbl[LG == n].mean() for n in range(len(logs))])
    res["navtrain_density_per_log"] = {"p5": float(np.percentile(dens, 5)), "median": float(np.median(dens)), "p95": float(np.percentile(dens, 95)), "min": float(dens.min()), "max": float(dens.max())}
    # discriminability: how well do label-free features separate navtrain from the other candidates?  (grouped 5-fold CV by log)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    X = np.column_stack([V, D, Y, np.abs(A)] + [np.array([t in g for g in G], dtype=float) for t in types])
    auc = {}
    for name, cols in {"speed_only": [0], "disp_4s_only": [1], "dyaw_4s_only": [2], "speed+disp+dyaw+acc": [0, 1, 2, 3], "all_plus_scenario_tags": list(range(X.shape[1]))}.items():
        p = np.zeros(len(Lbl))
        for tr, te in GroupKFold(5).split(X, Lbl, LG):
            m = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, random_state=0).fit(X[tr][:, cols], Lbl[tr])
            p[te] = m.predict_proba(X[te][:, cols])[:, 1]
        auc[name] = float(roc_auc_score(Lbl, p))
    res["auc_navtrain_vs_other_candidates_(5-fold_by_log)"] = auc
    # candidate screens
    scr = {}
    for name, m in {"none(uniform over candidates)": np.ones(len(Lbl), bool), "speed>=1": V >= 1.0, "speed>=2": V >= 2.0, "progress_4s>=5m": D >= 5.0, "speed>=1 & progress_4s>=5m": (V >= 1.0) & (D >= 5.0)}.items():
        scr[name] = {"share_of_candidates_kept": float(m.mean()), "navtrain_recall": float(m[Lbl].mean()), "navtrain_share_of_kept": float(Lbl[m].mean()), "stationary_share_of_kept": float((V[m] < 0.5).mean()),
                     "speed_quantiles_kept": q(V[m]), "turning_share_kept": float((Y[m] > 15).mean())}
    res["screens"] = scr
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_candidates", "n_navtrain", "shares", "no_tag_share", "navtrain_density_per_log", "auc_navtrain_vs_other_candidates_(5-fold_by_log)")}, indent=1))
    for k, v in res["quantiles"].items():
        print(k, "navtrain", v["navtrain"], "| all", v["all_candidates"])
    for k, v in scr.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "speed_quantiles_kept"})
    print("tags", {t: (round(v["navtrain"], 3), round(v["all_candidates"], 3)) for t, v in list(res["scenario_tag_share_at_token"].items())[:12]})


if __name__ == "__main__":
    main()
