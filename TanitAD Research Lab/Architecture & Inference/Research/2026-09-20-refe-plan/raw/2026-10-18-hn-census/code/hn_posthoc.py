#!/usr/bin/env python3
"""P6-HN step 0 -- POST-HOC diagnostics.  NOT pre-registered; written AFTER the census was read (the census numbers prompted two questions).
Nothing here is a headline, nothing changes the section-6 reading.  Every number says POST-HOC.

Q1  Is the expert ITSELF a hard negative in the sets where a proximate hard negative exists?  (A near-expert hard negative is a CONTRAST only if the expert is safe.)
    The teacher plan has no PDM label; the sets carry the teacher's own simulator calculators (`teacher_targets`: collision.NuPlanCollision.info, dac.violation).  First checked: do the same
    calculators on the 64 proposals agree with the banked PDM labels?  (sets `targets[k]`: collision info vs nc == 0, dac.violation vs dac == 0.)
Q2  How concentrated is the proximate-hard-negative supply across logs?  (per_set.csv, point.)

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I hn_posthoc.py --joined ../raw/hn_joined.npz --per-set ../raw/per_set.csv --out ../raw/posthoc.json
"""
import argparse
import csv
import glob
import json
import sys

import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hn_census as H  # noqa: E402

SETS = "D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/sets/onpolicy_r*_hw*.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", required=True)
    ap.add_argument("--per-set", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.joined)
    N = len(z["pdms"])
    key2i = {(str(l), str(t), int(s), int(r)): i for i, (l, t, s, r) in enumerate(zip(z["keys_log"], z["keys_token"], z["keys_step"], z["keys_rank"]))}
    p_col = np.full((N, 64), np.nan); p_dacv = np.full((N, 64), np.nan)
    t_col = np.full(N, np.nan); t_dacv = np.full(N, np.nan); t_off = np.full(N, np.nan)
    for p in sorted(glob.glob(SETS)):
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            i = key2i[(d["log_name"], d["token"], int(d["step"]), int(d["rank"]))]
            p_col[i] = [t.get("collision.NuPlanCollision.info", np.nan) for t in d["targets"]]
            p_dacv[i] = [t.get("dac.violation", np.nan) for t in d["targets"]]
            tt = d["teacher_targets"]
            t_col[i] = tt.get("collision.NuPlanCollision.info", np.nan); t_dacv[i] = tt.get("dac.violation", np.nan); t_off[i] = tt.get("off_road.OffRoad.info", np.nan)
    nc, dac, pdms = z["nc"], z["dac"], z["pdms"]
    R = {"_what": "POST-HOC diagnostics (NOT pre-registered; written after the census was read).", "defined_frac": {"p_col": float((~np.isnan(p_col)).mean()), "t_col": float((~np.isnan(t_col)).mean())}}
    ok = ~np.isnan(p_col)
    R["Q1_calculator_vs_PDM_label_on_proposals"] = {
        "collision_info>0 vs nc==0": {"both": int(((p_col > 0) & (nc == 0) & ok).sum()), "calc_only": int(((p_col > 0) & (nc != 0) & ok).sum()),
                                       "label_only": int(((p_col <= 0) & (nc == 0) & ok).sum()), "neither": int(((p_col <= 0) & (nc != 0) & ok).sum())},
        "dac.violation>0 vs dac==0": {"both": int(((p_dacv > 0) & (dac == 0) & ok).sum()), "calc_only": int(((p_dacv > 0) & (dac != 0) & ok).sum()),
                                       "label_only": int(((p_dacv <= 0) & (dac == 0) & ok).sum()), "neither": int(((p_dacv <= 0) & (dac != 0) & ok).sum())}}
    ade, _ = H.ade_fde(z["proposals"], z["teacher"])
    hn = H.hard_negative(nc, dac, pdms)
    ulogs, log_idx = np.unique(z["keys_log"], return_inverse=True)
    B = H.Boot(log_idx, len(ulogs))
    teacher_unsafe = (t_col > 0) | (t_dacv > 0)
    R["Q1_teacher_unsafe_by_its_own_calculators"] = {"share_sets": B.ratio(teacher_unsafe.astype(float)), "n_sets": int(teacher_unsafe.sum()),
                                                      "collision": int((t_col > 0).sum()), "dac_violation": int((t_dacv > 0).sum()), "offroad_info>0": int((t_off > 0).sum())}
    for d in H.GRID:
        npx, nhn = H.counts_at(ade, hn, d)
        has = nhn >= 1
        R[f"Q1_d={d}"] = {"share_sets_with_proximate_HN": B.ratio(has.astype(float)),
                          "share_sets_with_proximate_HN_AND_teacher_safe(contrast)": B.ratio((has & ~teacher_unsafe).astype(float)),
                          "share_of_proximate_HN_sets_where_teacher_is_unsafe": B.ratio((has & teacher_unsafe).astype(float), has.astype(float)),
                          "share_of_proximate_HN_proposals_in_teacher_unsafe_sets": B.ratio(nhn * teacher_unsafe, nhn)}
    # Q2 concentration (per_set.csv)
    rows = list(csv.DictReader(open(a.per_set, encoding="utf-8")))
    logs = np.array([r["log"] for r in rows]); n05 = np.array([int(r["n_HN_d0.5"]) for r in rows])
    cnt = {l: int((n05[logs == l] >= 1).sum()) for l in np.unique(logs)}
    tot = sum(cnt.values())
    top = sorted(cnt.items(), key=lambda kv: -kv[1])
    cum, k50 = 0, 0
    for k, (l, c) in enumerate(top, 1):
        cum += c
        if cum >= (tot + 1) // 2:
            k50 = k; break
    R["Q2_concentration_d0.5"] = {"sets_with_proximate_HN": tot, "k50_logs": k50, "n_logs": len(cnt), "top3": [{"log": l, "sets": c, "share": c / tot} for l, c in top[:3]],
                                  "sets_by_n_HN": {"0": int((n05 == 0).sum()), "1-4": int(((n05 >= 1) & (n05 <= 4)).sum()), "5-15": int(((n05 >= 5) & (n05 <= 15)).sum()), "16+": int((n05 >= 16).sum())}}

    # Q3 where do the siblings' ZERO picks sit relative to the teacher?  (point; each scorer READ SEPARATELY; n ~ 100 zero picks each)
    _, fde = H.ade_fde(z["proposals"], z["teacher"])
    key2z = key2i
    R["Q3_pick_distance_to_teacher(point)"] = {}
    for arm in ("N-s0", "N-s1", "N-s2"):
        sc = np.load(f"D:/Projects/TanitAD-artifacts/s1_read_cache/arms/{arm}/dump/scores_update400_B.npz")
        idx = np.array([key2z[(str(l), str(t), int(s_), int(r))] for l, t, s_, r in zip(sc["log"], sc["token"], sc["step"], sc["rank"])])
        p = 1.0 / (1.0 + np.exp(-sc["scores"].astype(np.float64)[..., :6]))
        agg = p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12.0
        pick = np.empty(N, int); pick[idx] = agg.argmax(1)
        ar = np.arange(N)
        zp = pdms[ar, pick] == 0
        q = lambda v: {k: float(np.quantile(v, kq)) for k, kq in (("q25", .25), ("q50", .5), ("q75", .75))}
        R["Q3_pick_distance_to_teacher(point)"][arm] = {"n_zero_picks": int(zp.sum()), "ADE_zero_picks": q(ade[ar, pick][zp]), "ADE_nonzero_picks": q(ade[ar, pick][~zp]),
                                                       "FDE_zero_picks": q(fde[ar, pick][zp]), "FDE_nonzero_picks": q(fde[ar, pick][~zp]),
                                                       "zero_pick_is_DAC0": int(((dac[ar, pick] == 0) & zp).sum()), "zero_pick_is_NC0_only": int(((nc[ar, pick] == 0) & (dac[ar, pick] == 1) & zp).sum())}
    json.dump(R, open(a.out, "w", encoding="utf-8"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(R, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
