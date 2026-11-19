# price_split_q50.py : the split named in section 5 of card PREREG_V8F_PROOF_ADD47 for the outcome KEEPS-CUT-PRICE-REMAINS: where does the remaining price of the
# reader-fed rule at level 0.50 (RQ50) above the label-fed rule (RLb) sit, in the lead windows or in the other windows. Same definitions as price_split.py
# (per window: 6 s position error difference and lag difference; a group's cost = the sum of the 6 s error differences over its windows).
# Convention, written into TIMES_NOTE.md before this tool ran: the deciding share is pooled over the two model seeds (sums over both seeds); if the pooled sum over
# all windows is not above 0 or the share lies outside [0, 1] the split is "not resolvable" and the branch "else" of the card applies (the reader, card ADD45).
# Tier T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance. Names nothing.
# Run from the headway folder: ops/pycap.sh ../integ10k/price_split_q50.py   (roots from PS_ROOT / PS_OUT if set)
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import headway_read_r2 as H  # noqa: E402

ROOT = os.environ.get("PS_ROOT", "F:/TanitAD-scratch/v8proof/integ10k/pull")
OUT = os.environ.get("PS_OUT", "F:/TanitAD-scratch/v8proof/integ10k/read_price/PRICE_SPLIT_Q50.json")
ARMS, SEEDS = ("I10R0", "I10RLb", "I10RR", "I10RQ35", "I10RQ50", "I10RRaw"), (0, 1)
PAIRS = (("RQ50_minus_RLb", "I10RQ50", "I10RLb"), ("RQ35_minus_RLb", "I10RQ35", "I10RLb"), ("RRaw_minus_RLb", "I10RRaw", "I10RLb"), ("RR_minus_RLb", "I10RR", "I10RLb"),
         ("RQ50_minus_R0", "I10RQ50", "I10R0"), ("RLb_minus_R0", "I10RLb", "I10R0"))
DECIDING = "RQ50_minus_RLb"
ENG_M = 2.0                                       # a window counts as engaged when the rollout is more than 2 m further behind than the other run of the pair

Wd, missing, where = H.load_world([ROOT], [(a, s) for a in ARMS for s in SEEDS], 141)
assert not missing, missing
ref = Wd[("I10R0", 0)]
J = H.load_join()
L = H.label_windows(J, ref["clip_index"], ref["ws"])
lead = np.asarray(L["lead"], bool)
res = {"card": "PREREG_V8F_PROOF_ADD47", "what": "split of the remaining price between lead windows and the other windows (section 5 of the card); T1-kin; names nothing",
       "n_windows": int(len(lead)), "n_lead": int(lead.sum()), "n_other": int((~lead).sum()), "engaged_bar_m": ENG_M, "per_seed": {}, "pooled": {}}


def lag6(d):
    g, cl = np.asarray(d["g"], np.float64), np.asarray(d["cl"], np.float64)
    return (H.logged_arc(g) - H.progress(g, cl))[:, -1], np.linalg.norm(cl[:, -1] - g[:, -1], axis=-1)


pool = {name: {"lead": 0.0, "other": 0.0, "all": 0.0} for name, _, _ in PAIRS}
for s in SEEDS:
    lag, err = {}, {}
    for a in ARMS:
        lag[a], err[a] = lag6(Wd[(a, s)])
    R = {}
    for name, A, B in PAIRS:
        dl, de = lag[A] - lag[B], err[A] - err[B]
        r = {}
        for part, m in (("lead", lead), ("other", ~lead), ("all", np.ones_like(lead))):
            r[part] = {"n": int(m.sum()), "mean_lag_diff_m": float(np.nanmean(dl[m])), "mean_err6_diff_m": float(np.nanmean(de[m])),
                       "share_engaged": float(np.mean(dl[m] > ENG_M)), "sum_err6_diff_m": float(np.nansum(de[m]))}
            pool[name][part] += r[part]["sum_err6_diff_m"]
        tot = r["all"]["sum_err6_diff_m"]
        r["share_of_err6_cost_in_other_windows"] = float(r["other"]["sum_err6_diff_m"] / tot) if tot else None
        R[name] = r
    res["per_seed"][str(s)] = R
for name, p in pool.items():
    share = (p["other"] / p["all"]) if p["all"] > 0 else None
    res["pooled"][name] = {"sum_err6_diff_m": p, "share_in_other_windows": share, "resolvable": bool(share is not None and 0.0 <= share <= 1.0)}
d = res["pooled"][DECIDING]
res["deciding"] = {"pair": DECIDING, "resolvable": d["resolvable"], "share_in_other_windows": d["share_in_other_windows"],
                   "next_arm": ("presence threshold" if (d["resolvable"] and d["share_in_other_windows"] > 0.5) else "reader (card ADD45)")}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
for s in SEEDS:
    for name, r in res["per_seed"][str(s)].items():
        print("s%d %-15s lead: lag %+6.2f m err6 %+6.2f m engaged %.3f | other: lag %+6.2f m err6 %+6.2f m engaged %.3f | share other %s" % (
            s, name, r["lead"]["mean_lag_diff_m"], r["lead"]["mean_err6_diff_m"], r["lead"]["share_engaged"], r["other"]["mean_lag_diff_m"], r["other"]["mean_err6_diff_m"],
            r["other"]["share_engaged"], ("%.3f" % r["share_of_err6_cost_in_other_windows"]) if r["share_of_err6_cost_in_other_windows"] is not None else "n/a"))
for name, p in res["pooled"].items():
    print("pooled %-15s sum err6 lead %+9.1f m other %+9.1f m all %+9.1f m | share other %s | resolvable %s" % (
        name, p["sum_err6_diff_m"]["lead"], p["sum_err6_diff_m"]["other"], p["sum_err6_diff_m"]["all"],
        ("%.3f" % p["share_in_other_windows"]) if p["share_in_other_windows"] is not None else "n/a", p["resolvable"]))
print("deciding %s" % json.dumps(res["deciding"]))
print("n lead %d other %d" % (res["n_lead"], res["n_other"]))
print("ZZ PRICE-SPLIT-Q50 DONE")
