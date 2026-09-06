"""D-GSTR-1 P1b -- decompose `strategic_goal_loss` on the banked windows and put
the model's own score beside the NO-INFORMATION FLOORS it must beat.

⛔ A loss number alone says nothing. Three floors, each a KNOWN value:
   F1 CONST-STRAIGHT  a predictor that always says (1, 0). If the head scores
                      WORSE than this, it is not "unconverged", it is anti-fit.
   F2 CONST-BEST      the best fixed unit bearing = direction of the mean target
                      (the closed-form optimum of a cosine loss for a constant).
   F3 CONST-DIST      the best fixed dist_pref = the MEDIAN target (L1 optimum).
Plus a SHAPE assertion: the dumped `gstr_nav_true[:, :2]` must be a UNIT vector,
or it is not the model's bearing and nothing below applies.

ZERO GPU. ASCII-only.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/nvidia/navpred/stack")
from tanitad.data.lan import (LAN_FEATS_PER_ANCHOR, LanConfig,  # noqa: E402
                              lan_window_features)

DUMP = os.environ.get("GSTR_DUMP",
                      "/home/nvidia/navpred/navflip_dump/decisions")
TOP = os.path.dirname(DUMP)
OUT = os.environ.get("GSTR_OUT2", "/home/nvidia/navroute/GSTR_LOSS_DECOMP.json")
NBOOT = int(os.environ.get("NBOOT", "2000"))
SEED = int(os.environ.get("SEED", "0"))


def goal_target(f, k):
    f = np.asarray(f, dtype=np.float64).reshape(k, LAN_FEATS_PER_ANCHOR)
    valid = f[:, 3] > 0.5
    first = int(np.argmax(valid.astype(np.float64)))
    picked = f[first]
    n = np.linalg.norm(picked[:2])
    return picked[:2] / max(n, 1e-6), first / max(k - 1, 1) * 2.0 - 1.0, \
        bool(valid.any())


def boot(ep, x, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    ue = np.unique(ep)
    idx = {e: np.where(ep == e)[0] for e in ue}
    d = np.empty(nboot)
    for i in range(nboot):
        pick = rng.choice(ue, size=ue.size, replace=True)
        d[i] = x[np.concatenate([idx[e] for e in pick])].mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": round(float(x.mean()), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "n": int(x.size),
            "n_ep": int(ue.size)}


def main():
    cfg = LanConfig()
    k = cfg.k
    TC, TS, TD, MC, MS, MD, V, EP = [], [], [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(DUMP, "*.npz"))):
        top = os.path.join(TOP, os.path.basename(f))
        if not os.path.exists(top):
            continue
        t = np.load(top, allow_pickle=True)
        z = np.load(f, allow_pickle=True)
        need = ("ep_poses", "ws", "gstr_nav_true", "pose_last")
        if any(q not in z.files for q in need):
            continue
        poses = np.asarray(z["ep_poses"], dtype=np.float64)
        ws = np.asarray(z["ws"]).astype(np.int64).reshape(-1)
        pl = np.asarray(z["pose_last"], dtype=np.float64).reshape(len(ws), 4)
        if not np.allclose(poses[ws], pl, atol=1e-5, rtol=0):
            print("# SKIP index mismatch", f, file=sys.stderr)
            continue
        g = np.asarray(z["gstr_nav_true"], dtype=np.float64)
        ci = (int(np.asarray(t["clip_index"]).reshape(-1)[0])
              if "clip_index" in t.files else -1)
        for i, w in enumerate(ws.tolist()):
            b, d, av = goal_target(lan_window_features(poses, w, cfg), k)
            TC.append(b[0]); TS.append(b[1]); TD.append(d); V.append(av)
            MC.append(g[i, 0]); MS.append(g[i, 1]); MD.append(g[i, 2])
            EP.append(ci)
    TC, TS, TD = map(np.array, (TC, TS, TD))
    MC, MS, MD = map(np.array, (MC, MS, MD))
    V = np.array(V, dtype=bool); EP = np.array(EP)

    res = {"tool": "D-GSTR-1 P1b strategic_goal_loss decomposition",
           "dump": DUMP, "evidence_class": "MEASURED (ours)",
           "tier": "T1 for the model output; the target is a LABEL",
           "estimator": f"paired episode-cluster bootstrap, {NBOOT} resamples, "
                        f"seed {SEED}",
           "n_windows": int(TC.size), "n_valid_label": int(V.sum())}

    # ---- SHAPE assertion: is the dumped vector a UNIT bearing? ----
    nrm = np.hypot(MC, MS)
    res["SHAPE_ASSERTION"] = {
        "rule": "gstr_nav_true[:, :2] must be a UNIT vector (refc_v3.py:981) "
                "or it is not the model's bearing",
        "norm_min": round(float(nrm.min()), 6),
        "norm_max": round(float(nrm.max()), 6),
        "PASS": bool(np.allclose(nrm, 1.0, atol=1e-4)),
        "model_cos_frac_positive": round(float((MC > 0).mean()), 4),
        "model_cos_min": round(float(MC.min()), 4),
        "model_cos_median": round(float(np.median(MC)), 4),
        "model_dist_pref_median": round(float(np.median(MD)), 4),
        "model_dist_pref_min": round(float(MD.min()), 4),
        "model_dist_pref_max": round(float(MD.max()), 4),
    }

    m = V
    ep = EP[m]
    l_bear_model = 1.0 - (MC[m] * TC[m] + MS[m] * TS[m])
    l_bear_straight = 1.0 - TC[m]
    mu = np.array([TC[m].mean(), TS[m].mean()])
    mu = mu / max(np.linalg.norm(mu), 1e-12)
    l_bear_best = 1.0 - (mu[0] * TC[m] + mu[1] * TS[m])
    l_dist_model = np.abs(MD[m] - TD[m])
    dmed = float(np.median(TD[m]))
    l_dist_best = np.abs(dmed - TD[m])

    res["LOSS_DECOMPOSITION"] = {
        "_definition": "strategic_goal_loss = l_bear + l_dist; "
                       "l_bear = 1 - cos(model, target) in [0, 2]; "
                       "l_dist = |tanh(g2) - dist_target| in [0, 2]; "
                       "masked by label validity",
        "l_bear_MODEL": boot(ep, l_bear_model),
        "l_bear_FLOOR_const_straight_(1,0)": boot(ep, l_bear_straight),
        "l_bear_FLOOR_const_best": boot(ep, l_bear_best),
        "_best_const_bearing": [round(float(mu[0]), 4), round(float(mu[1]), 4)],
        "l_dist_MODEL": boot(ep, l_dist_model),
        "l_dist_FLOOR_const_median": boot(ep, l_dist_best),
        "_dist_median_target": dmed,
        "TOTAL_MODEL": round(float((l_bear_model + l_dist_model).mean()), 4),
        "TOTAL_FLOOR_const": round(float(
            (l_bear_straight + l_dist_best).mean()), 4),
    }
    d = l_bear_model - l_bear_straight
    res["MODEL_MINUS_FLOOR_bearing"] = boot(ep, d)
    res["VERDICT"] = (
        "WORSE THAN A CONSTANT STRAIGHT-AHEAD PREDICTOR"
        if res["MODEL_MINUS_FLOOR_bearing"]["lo"] > 0 else
        "not separated from the constant floor")

    # angle view, degrees
    ang_m = np.degrees(np.arctan2(MS[m], MC[m]))
    ang_t = np.degrees(np.arctan2(TS[m], TC[m]))
    res["ANGLES_deg"] = {
        "model": {"p5": round(float(np.percentile(ang_m, 5)), 3),
                  "median": round(float(np.median(ang_m)), 3),
                  "p95": round(float(np.percentile(ang_m, 95)), 3),
                  "frac_positive_LEFT": round(float((ang_m > 0).mean()), 4)},
        "target": {"p5": round(float(np.percentile(ang_t, 5)), 3),
                   "median": round(float(np.median(ang_t)), 3),
                   "p95": round(float(np.percentile(ang_t, 95)), 3),
                   "frac_positive_LEFT": round(float((ang_t > 0).mean()), 4)},
        "mean_signed_offset_model_minus_target":
            round(float((ang_m - ang_t).mean()), 3),
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
