"""stopcue_read.py -- PREREAD_V8F_STOPCUE (theory/stopcue/PREREAD_V8F_STOPCUE.md). Zero training, zero model passes.

Reads H-V8F-STOPDEC-1's per-window npz files and asks whether T's goal head RANKS stop windows (AUROC), per seed, with an
episode-cluster bootstrap (B 2000, seed 0). Usage: stopcue_read.py --out <json>
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "otband", "read"))
import numpy as np  # noqa: E402
from topk_read import BL  # noqa: E402  (taniteval from the read tree)

STOPDEC_OUT = os.path.join(os.path.dirname(HERE), "stopdec", "out")
TAGS = ("PARITY_DAGOACC_s0", "PARITY_DAGOACC_s1")
STOP_I = 6                                   # GOAL22 index of STOP_POINT
LON = ("HOLD", "CREEP", "STOP", "FOLLOW", "DECELERATE", "ACCELERATE", "KEEP")
NB, SEED = 2000, 0
PRESENT_AUC, PRESENT_LO, ABSENT_AUC = 0.85, 0.80, 0.70


def auroc(score, label):
    """Mann-Whitney AUROC, ties counted one half. NaN if a class is empty."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def avg_precision(score, label):
    s, y = np.asarray(score, float), np.asarray(label, bool)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    yy = y[o]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def recall_at_fpr(score, label, fpr=0.05):
    s, y = np.asarray(score, float), np.asarray(label, bool)
    thr = np.quantile(s[~y], 1.0 - fpr)
    return float((s[y] > thr).mean()), float(thr)


def self_test():
    y = np.array([0, 0, 1, 1, 0, 1], bool)
    assert auroc(y.astype(float), y) == 1.0, "planted perfect score must read 1.0"
    assert auroc(np.zeros(len(y)), y) == 0.5, "constant score must read 0.5"
    assert auroc(1.0 - y.astype(float), y) == 0.0
    return {"perfect": 1.0, "constant": 0.5, "inverted": 0.0}


def read_arm(path):
    z = np.load(path, allow_pickle=False)
    has = z["has"]
    gp, lp = z["goal_prob"], z["lon_prob"]
    assert np.all((gp >= 0) & (gp <= 1)), "goal_prob outside [0, 1]"
    v_end, v2 = z["v_end_gt"], z["v2_gt"]
    idx = np.where(has)[0]
    pos = z["stop_hit"][idx].astype(bool)
    score = gp[idx, STOP_I]
    reg_stop = (v_end[idx] < 0.5)
    reg_slow = reg_stop | ((v_end[idx] - v2[idx]) < -1.0)
    s_lstop = lp[idx, LON.index("STOP")]
    s_lslow = lp[idx, LON.index("HOLD")] + lp[idx, LON.index("STOP")] + lp[idx, LON.index("DECELERATE")]
    eid = z["eid"][idx]

    def stat(sel):
        sel = np.asarray(sel)
        return {"auroc_goal_stop": auroc(score[sel], pos[sel]),
                "auroc_lon_stop_vs_regime_stop": auroc(s_lstop[sel], reg_stop[sel]),
                "auroc_lon_slow_vs_regime_stop_or_decel": auroc(s_lslow[sel], reg_slow[sel]),
                "ap_goal_stop": avg_precision(score[sel], pos[sel])}
    r = BL.CI.stat_cluster_bootstrap(list(range(len(idx))), eid, stat, n_boot=NB, seed=SEED)
    rec, thr = recall_at_fpr(score, pos, 0.05)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"npz": os.path.basename(path), "npz_sha256": sha, "n_has": int(has.sum()), "n_pos": int(pos.sum()), "n_neg": int((~pos).sum()),
            "n_regime_stop": int(reg_stop.sum()), "n_regime_stop_or_decel": int(reg_slow.sum()),
            "prevalence": float(pos.mean()), "stats": r,
            "recall_at_fpr_0p05_eval_chosen": rec, "threshold_at_fpr_0p05": thr}


def classify(a):
    d = a["stats"]["auroc_goal_stop"]
    if a["n_pos"] < 50:
        return "NOT-EVALUATED"
    if d["point"] >= PRESENT_AUC and d["lo"] >= PRESENT_LO:
        return "CUE-PRESENT"
    if d["point"] < ABSENT_AUC:
        return "CUE-ABSENT"
    return "PARTIAL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"tool": "stopcue_read.py", "card": "theory/stopcue/PREREAD_V8F_STOPCUE.md", "tier": "T1-kin first tick, eval-141; a decision-head diagnostic, never driving performance",
           "estimator": "taniteval.ci.stat_cluster_bootstrap, B=2000, seed 0, by episode; another draw of episodes only",
           "bars": {"present_auc": PRESENT_AUC, "present_lo": PRESENT_LO, "absent_auc": ABSENT_AUC}, "self_test": self_test(), "arms": {}}
    for tag in TAGS:
        res["arms"][tag] = read_arm(os.path.join(STOPDEC_OUT, f"stopdec_{tag}.npz"))
        res["arms"][tag]["class"] = classify(res["arms"][tag])
        print(json.dumps({"tag": tag, "n_pos": res["arms"][tag]["n_pos"], "auroc": res["arms"][tag]["stats"]["auroc_goal_stop"], "class": res["arms"][tag]["class"]}), flush=True)
    cls = {res["arms"][t]["class"] for t in TAGS}
    res["verdict"] = {"outcome": cls.pop() if len(cls) == 1 else "SEEDS-DISAGREE", "per_seed": {t: res["arms"][t]["class"] for t in TAGS}}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["verdict"]), flush=True)


if __name__ == "__main__":
    main()
