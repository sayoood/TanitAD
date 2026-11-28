"""l2_roc.py -- PREREG_L2_ROC.md (sha256 5a84da9a...): AUC of the braking score dv = v_end - v0, GT-decel vs GT-cruise, compliant
S6 windows, refcv8+stage / refcv8 os / refcv7, paired episode-cluster bootstrap over episodes. CPU only."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import l2_common as C  # noqa: E402


def dv_of(P, v0, dt):
    P = np.asarray(P, np.float64)
    return np.linalg.norm(P[:, -1] - P[:, -2], axis=-1) / dt - v0


def cls_of(dv):
    return np.where(dv <= -1.5, "decel", np.where(dv >= 1.5, "accel", "cruise"))


def auc(pos, neg):
    """P(pos < neg) + 0.5 P(tie): lower score = more braking, positives are GT-decel."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    s = allv[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[len(neg):]
    u_pos_gt_neg = rp.sum() - len(pos) * (len(pos) + 1) / 2   # count of (pos > neg) + 0.5 ties
    return float(1.0 - u_pos_gt_neg / (len(pos) * len(neg)))


def main():
    out = {"prereg_sha256": open(Path(__file__).parent / "PREREG_L2_ROC.sha256").read().split()[0], "seeds": {}}
    zg = json.load(open("C:/Users/Admin/refcv8_pf/refcv9_l2/result/zero_gpu_nocal/ZERO_GPU.json"))
    for seed in (0, 1):
        E = C.load_eval_pf(seed)
        m = E["compliant"]
        dt, v0 = E["dt"], E["v0"]
        dg = dv_of(E["g"], v0, dt)
        gt = cls_of(dg)
        arms = {"stage": E["stage_banked"], "os": E["os"], "r7": E["r7"]}
        dvs = {a: dv_of(P, v0, dt) for a, P in arms.items()}
        dvs["GT_C1"] = dg
        dvs["CONST_C2"] = np.zeros_like(dg)
        pos = m & (gt == "decel")
        neg = m & (gt == "cruise")
        eid = E["eid"]
        eps = np.unique(eid[pos | neg])
        res = {"n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "n_episodes": int(len(eps)), "checks": E["checks"]}
        res["auc"] = {a: auc(d[pos], d[neg]) for a, d in dvs.items()}
        fdmd = {}
        for a in ("stage", "os", "r7"):
            pc = cls_of(dvs[a])
            fdmd[a] = {"fd": float((pc[neg] == "decel").mean()), "md": float((pc[pos] != "decel").mean())}
            ref = zg["seeds"][str(seed)]["rate_cells"][f"compliant|{a}"]
            fdmd[a]["C3_match"] = bool(abs(fdmd[a]["fd"] - ref["fd"]) <= 1e-4 and abs(fdmd[a]["md"] - ref["md"]) <= 1e-4)
        res["fd_md"] = fdmd
        res["C1_gt_auc_is_1"] = res["auc"]["GT_C1"] == 1.0
        res["C2_const_auc_is_half"] = res["auc"]["CONST_C2"] == 0.5
        # paired episode-cluster bootstrap
        rng = np.random.default_rng(0)
        idx_by_ep = {e: np.flatnonzero((eid == e) & (pos | neg)) for e in eps}
        B = 2000
        d_stage, d_os = np.empty(B), np.empty(B)
        for b in range(B):
            draw = rng.choice(eps, size=len(eps), replace=True)
            ii = np.concatenate([idx_by_ep[e] for e in draw])
            p, n = ii[pos[ii]], ii[neg[ii]]
            a7 = auc(dvs["r7"][p], dvs["r7"][n])
            d_stage[b] = auc(dvs["stage"][p], dvs["stage"][n]) - a7
            d_os[b] = auc(dvs["os"][p], dvs["os"][n]) - a7
        res["paired_vs_r7"] = {
            "stage": {"delta": res["auc"]["stage"] - res["auc"]["r7"], "ci95": [float(np.percentile(d_stage, 2.5)), float(np.percentile(d_stage, 97.5))]},
            "os": {"delta": res["auc"]["os"] - res["auc"]["r7"], "ci95": [float(np.percentile(d_os, 2.5)), float(np.percentile(d_os, 97.5))]}}
        res["estimator"] = "paired episode-cluster bootstrap over episodes, B 2000, seed 0 (answers: another draw of EPISODES)"
        out["seeds"][str(seed)] = res
        print("ZZSEED", seed, json.dumps({k: res[k] for k in ("n_pos", "n_neg", "n_episodes", "auc", "paired_vs_r7", "C1_gt_auc_is_1", "C2_const_auc_is_half")}))
        print("ZZFDMD", seed, json.dumps(res["fd_md"]))
    json.dump(out, open(Path(__file__).parent / "L2_ROC.json", "w"), indent=1)
    print("ZZROCDONEZZ")


if __name__ == "__main__":
    main()
