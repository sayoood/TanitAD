"""S1 step 2 (sampler): fit label-free importance weights on t0 EGO SPEED (an input at inference, not a label) so that a 4-tokens-per-log draw over
the rule's candidates reproduces navtrain's t0-speed distribution, then measure what that does and does NOT fix.

    FIT on the 214 navtrain logs of the val split; EVALUATE out-of-sample on the 248 top-level navtrain logs (weights frozen).
    DESIGN-weighted = mean over logs of the within-log share (the A1 estimator clusters on logs; a 4-per-log draw gives every log equal weight).
    Future-geometry features (4-s progress, 4-s heading change) are NOT used in the weights: they are the logged EXPERT FUTURE, i.e. the planning
    ground truth, and selecting on them is label-adjacent.  They are reported only as the RESIDUAL the t0-only sampler leaves.

    python sampler_check.py     -> raw/sampler_check.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bias_measure as B  # noqa: E402
import identity_check as IC  # noqa: E402
import unseen_token_rule as R  # noqa: E402

DB_TV = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval"
DB_TOP = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
OUT = "C:/Users/Admin/par18/unseen_s1/raw/sampler_check.json"
EDGES = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 1e9])


def per_log(db, known):
    out = []
    for lg in sorted(l for l in known if os.path.exists(f"{db}/{l}.db")):
        rec = R.read_log(f"{db}/{lg}.db")
        c = R.candidates(rec)
        v, d, y, a, g = B.features(f"{db}/{lg}.db", c["lp_idx"])
        lab = np.array([t in known[lg] for t in c["tokens"]])
        out.append({"log": lg, "v": v, "d": d, "y": y, "a": np.abs(a), "tag_stationary": np.array(["stationary" in s for s in g]), "tag_none": np.array([len(s) == 0 for s in g]),
                    "tag_inter": np.array([any("intersection" in t for t in s) for s in g]), "nav": lab, "bin": np.clip(np.searchsorted(EDGES, v, side="right") - 1, 0, len(EDGES) - 2)})
    return out


def design_hist(logs, w=None):
    """mean over logs of the within-log weighted share per speed bin; w None = uniform over candidates; w='nav' = navtrain tokens"""
    nb = len(EDGES) - 1
    acc = np.zeros(nb)
    n_used = 0
    for L in logs:
        if isinstance(w, str):
            m = L["nav"].astype(float)
        elif w is None:
            m = np.ones(len(L["v"]))
        else:
            m = w[L["bin"]]
        h = np.bincount(L["bin"], weights=m, minlength=nb)
        if h.sum() == 0:      # a log whose navtrain tokens are all off the rule's grid (the 7 phase-miss logs of the 248): no navtrain token to compare
            continue
        acc += h / h.sum()
        n_used += 1
    return acc / n_used


def design_share(logs, f, w=None):
    s = []
    for L in logs:
        if isinstance(w, str):
            m = L["nav"].astype(float)
        elif w is None:
            m = np.ones(len(L["v"]))
        else:
            m = w[L["bin"]]
        if m.sum() == 0:
            continue
        s.append(float((m * f(L)).sum() / m.sum()))
    return float(np.mean(s))


FEATS = {"stationary_speed_lt_0.5": lambda L: (L["v"] < 0.5).astype(float), "speed_mean_mps": lambda L: L["v"], "abs_acc_x_mean": lambda L: L["a"],
         "tag_stationary": lambda L: L["tag_stationary"].astype(float), "tag_intersection": lambda L: L["tag_inter"].astype(float), "tag_none": lambda L: L["tag_none"].astype(float),
         "RESID_progress_4s_lt_2m": lambda L: (L["d"] < 2.0).astype(float), "RESID_turning_dyaw_gt_15deg": lambda L: (L["y"] > 15.0).astype(float), "RESID_disp_4s_mean_m": lambda L: L["d"]}


def report(logs, w):
    r = {}
    for k, f in FEATS.items():
        r[k] = {"navtrain": design_share(logs, f, "nav"), "uniform_over_candidates": design_share(logs, f, None), "speed_weighted": design_share(logs, f, w)}
    return r


def main():
    known = IC.load_known()
    fit = per_log(DB_TV, known)
    oos = per_log(DB_TOP, known)
    print("fit logs", len(fit), "oos logs", len(oos), flush=True)
    target = design_hist(fit, "nav")
    w = np.ones(len(EDGES) - 1)
    for it in range(60):
        ach = design_hist(fit, w)
        w = w * np.where(ach > 0, target / np.maximum(ach, 1e-12), 1.0)
        w = w / w.mean()
    res = {"what": "t0-speed importance weights fitted on the 214 val-split navtrain logs; evaluated out-of-sample on the 248 top-level navtrain logs; design-weighted (equal log weight)",
           "evidence": "MEASURED (read-only DB ego_pose / scenario_tag on NAVTRAIN logs only)", "speed_bin_edges_mps": EDGES[:-1].tolist() + ["inf"], "weights": w.tolist(),
           "speed_hist_navtrain_design_fit": target.tolist(), "speed_hist_uniform_design_fit": design_hist(fit, None).tolist(), "speed_hist_weighted_design_fit": design_hist(fit, w).tolist(),
           "FIT_214": report(fit, w), "OOS_248": report(oos, w)}
    # a seeded 4-per-log draw (Efraimidis-Spirakis, weights w[bin]) reproduces the analytic shares?
    rng = np.random.default_rng(20261019)
    sim = {"n_rep": 200, "per_log": 4}
    for name, logs in (("FIT_214", fit), ("OOS_248", oos)):
        st, st_nav = [], []
        for _ in range(200):
            vals = []
            for L in logs:
                key = rng.random(len(L["v"])) ** (1.0 / w[L["bin"]])
                top = np.argpartition(-key, 4)[:4]
                vals.append((L["v"][top] < 0.5).mean())
            st.append(np.mean(vals))
        sim[name] = {"stationary_share_drawn_mean": float(np.mean(st)), "sd_over_reps": float(np.std(st))}
    res["simulated_draw"] = sim
    json.dump(res, open(OUT, "w"), indent=1)
    print("weights", np.round(w, 3).tolist())
    for part in ("FIT_214", "OOS_248"):
        print(part)
        for k, v in res[part].items():
            print("  %-30s navtrain %.3f | uniform %.3f | speed-weighted %.3f" % (k, v["navtrain"], v["uniform_over_candidates"], v["speed_weighted"]))
    print(sim)


if __name__ == "__main__":
    main()
