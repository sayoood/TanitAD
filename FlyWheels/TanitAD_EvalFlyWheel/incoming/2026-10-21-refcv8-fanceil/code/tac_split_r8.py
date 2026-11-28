#!/usr/bin/env python
"""tac_split_r8.py -- TACTICAL-proxy split of the fan ceiling (BESIDE; T0 for anything using GT).

On the battery's S6 windows (6 s GT valid): the heading CLASS at 6 s of a plan = left / straight / right by its terminal heading
(5 s -> 6 s segment, the S6 slots 5 and 6) with a +-0.15 rad dead band (a PROXY of the manoeuvre, not the tactical head's
label). Reports: how often the SELECTED candidate's class matches GT's, how often the S6-ORACLE candidate's does, and how
often SOME candidate in the fan has GT's class (class coverage). STRATEGIC family: no instrument applies to a fan ceiling
(it has no route / goal readout of its own) -- stated, not dropped.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def cls(p):
    d = p[..., -1, :] - p[..., -2, :]
    th = np.arctan2(d[..., 1], d[..., 0])
    st = np.hypot(d[..., 0], d[..., 1]) < 0.05
    c = np.where(th > 0.15, 1, np.where(th < -0.15, -1, 0))
    return np.where(st, 0, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan-npz", required=True)
    ap.add_argument("--bank-s6", required=True)
    ap.add_argument("--taniteval-root", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.taniteval_root)
    from taniteval import ci as C
    z = np.load(a.fan_npz, allow_pickle=True)
    keys = [(int(c), int(w)) for c, w in zip(z["clip_index"], z["ws"])]
    s6 = {}
    for f in sorted(glob.glob(os.path.join(a.bank_s6, "ep*.npz"))):
        q = np.load(f, allow_pickle=True)
        ci = int(np.asarray(q["clip_index"]).ravel()[0])
        eid = str(int(np.asarray(q["eid"]).ravel()[0]))
        for j, w in enumerate(np.asarray(q["ws"]).ravel()):
            s6[(ci, int(w))] = (q["g"][j].astype(np.float64), eid)
    idx = np.asarray([i for i, k in enumerate(keys) if k in s6])
    g = np.stack([s6[keys[i]][0] for i in idx])
    eid = [s6[keys[i]][1] for i in idx]
    fan = np.asarray(z["anchor_traj"], np.float64)[idx][:, :, [1, 3, 4, 5, 6, 7]]
    sel = np.asarray(z["sel_idx"]).astype(np.int64)[idx]
    B = len(idx)
    ar = np.arange(B)
    cg = cls(g)
    cf = cls(fan)                                                   # [B, N]
    de = np.linalg.norm(fan - g[:, None], axis=-1).mean(-1)
    orr = de.argmin(1)
    m_sel = (cf[ar, sel] == cg).astype(float)
    m_or = (cf[ar, orr] == cg).astype(float)
    cov = (cf == cg[:, None]).any(1).astype(float)
    bt = lambda v, t: dict(C.episode_cluster_bootstrap(v, eid, n_boot=2000, seed=0), metric=t)
    turn = cg != 0
    out = {"n_s6_windows": int(B), "gt_class_counts": {"left": int((cg == 1).sum()), "straight": int((cg == 0).sum()),
                                                        "right": int((cg == -1).sum())},
           "selected_class_matches_gt": bt(m_sel, "selected heading-class == GT"),
           "s6_oracle_class_matches_gt_T0": bt(m_or, "S6-oracle heading-class == GT"),
           "fan_covers_gt_class_T0": bt(cov, "some candidate has GT's heading-class"),
           "on_GT_turns_only": {"n": int(turn.sum()),
                                "selected_match": round(float(m_sel[turn].mean()), 4) if turn.any() else None,
                                "oracle_match": round(float(m_or[turn].mean()), 4) if turn.any() else None,
                                "coverage": round(float(cov[turn].mean()), 4) if turn.any() else None},
           "strategic_family": "NOT APPLICABLE to a fan ceiling: no route/goal readout is produced by this instrument "
                               "(the battery's own strategic family stands as banked)",
           "defn": "heading class at 6 s, +-0.15 rad dead band, a PROXY of the manoeuvre", "script": os.path.abspath(__file__)}
    json.dump(out, open(a.out_json, "w", encoding="utf-8"), indent=1)
    print("ZZTACZZ n=%d sel=%.4f or=%.4f cov=%.4f turns n=%d sel=%s or=%s cov=%s" % (
        B, out["selected_class_matches_gt"]["mean"], out["s6_oracle_class_matches_gt_T0"]["mean"],
        out["fan_covers_gt_class_T0"]["mean"], out["on_GT_turns_only"]["n"], out["on_GT_turns_only"]["selected_match"],
        out["on_GT_turns_only"]["oracle_match"], out["on_GT_turns_only"]["coverage"]))


if __name__ == "__main__":
    main()
