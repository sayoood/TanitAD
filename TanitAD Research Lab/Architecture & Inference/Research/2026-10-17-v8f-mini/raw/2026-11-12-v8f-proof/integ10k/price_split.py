# price_split.py : descriptive attribution on the existing dumps of card PREREG_V8F_PROOF_ADD46 (row H-V8F-PROOF-101). POST HOC: it names nothing; it ranks
# the levers for the next arm. Tier T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance.
# Question: where does the price of the reader-fed rule (RR) above the label-fed rule (RLb) sit: in the lead windows (the mapped gap is smaller than the label gap)
# or in the windows without a lead track (the reader's presence fires where the label has no lead)?
# Per window: lag = arc of the logged ego at 6 s minus arc progress of the rollout at 6 s (positive = the rollout is behind), with the functions of the headway reader.
# Run from the headway folder: ops/pycap.sh ../integ10k/price_split.py   (numpy only)
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import headway_read_r2 as H  # noqa: E402

ROOT = "F:/TanitAD-scratch/v8proof/integ10k/pull"
OUT = "F:/TanitAD-scratch/v8proof/integ10k/read/PRICE_SPLIT.json"
ARMS, SEEDS = ("I10R0", "I10RLb", "I10RR", "I10RRx"), (0, 1)
ENG_M = 2.0                                       # a window counts as engaged when the rollout is more than 2 m further behind than the run without the rule

Wd, missing, where = H.load_world([ROOT], [(a, s) for a in ARMS for s in SEEDS], 141)
assert not missing, missing
ref = Wd[("I10R0", 0)]
J = H.load_join()
L = H.label_windows(J, ref["clip_index"], ref["ws"])
lead = np.asarray(L["lead"], bool)
res = {"what": "post hoc split of the price of the rule between lead windows and the other windows; card ADD46 dumps; T1-kin; names nothing",
       "n_windows": int(len(lead)), "n_lead": int(lead.sum()), "n_other": int((~lead).sum()), "engaged_bar_m": ENG_M, "per_seed": {}}


def lag6(d):
    g, cl = np.asarray(d["g"], np.float64), np.asarray(d["cl"], np.float64)
    return (H.logged_arc(g) - H.progress(g, cl))[:, -1], np.linalg.norm(cl[:, -1] - g[:, -1], axis=-1)


for s in SEEDS:
    lag, err = {}, {}
    for a in ARMS:
        lag[a], err[a] = lag6(Wd[(a, s)])
    R = {}
    for name, A, B in (("RR_minus_R0", "I10RR", "I10R0"), ("RLb_minus_R0", "I10RLb", "I10R0"), ("RR_minus_RLb", "I10RR", "I10RLb"), ("RRx_minus_R0", "I10RRx", "I10R0")):
        dl, de = lag[A] - lag[B], err[A] - err[B]
        r = {}
        for part, m in (("lead", lead), ("other", ~lead), ("all", np.ones_like(lead))):
            r[part] = {"n": int(m.sum()), "mean_lag_diff_m": float(np.nanmean(dl[m])), "mean_err6_diff_m": float(np.nanmean(de[m])),
                       "share_engaged": float(np.mean(dl[m] > ENG_M)), "sum_err6_diff_m": float(np.nansum(de[m]))}
        tot = r["all"]["sum_err6_diff_m"]
        r["share_of_err6_cost_in_other_windows"] = float(r["other"]["sum_err6_diff_m"] / tot) if tot else None
        R[name] = r
    res["per_seed"][str(s)] = R
json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
for s in SEEDS:
    for name, r in res["per_seed"][str(s)].items():
        print("s%d %-13s lead: lag %+6.2f m err6 %+6.2f m engaged %.3f | other: lag %+6.2f m err6 %+6.2f m engaged %.3f | cost share in other windows %s" % (
            s, name, r["lead"]["mean_lag_diff_m"], r["lead"]["mean_err6_diff_m"], r["lead"]["share_engaged"], r["other"]["mean_lag_diff_m"], r["other"]["mean_err6_diff_m"],
            r["other"]["share_engaged"], ("%.3f" % r["share_of_err6_cost_in_other_windows"]) if r["share_of_err6_cost_in_other_windows"] is not None else "n/a"))
print("n lead %d other %d" % (res["n_lead"], res["n_other"]))
print("ZZ PRICE-SPLIT DONE")
