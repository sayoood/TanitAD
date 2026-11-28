# unsafe_cause_b8.py : the split of the unsafe windows repeated on the runs of card ADD57 (braking limit of the lead rule 8.0 and 10.0 m/s2), as that card commits for
# the name RAISES-UNSAFE ("the split of the unsafe windows is repeated to see which windows were added"). Descriptive, names nothing.
# Tier T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance.
# Convention written into TIMES_NOTE.md before this ran. Per arm and model seed the split of unsafe_cause.py (first unsafe step k*, tick in force q = (k* - 1) // 10,
# kind, depth, rule state at the tick in force), for the arms R0, RLc, RLcB8, RLcB10, RQ50, RQ50B8.
# Added, per pair (limit 6.0 -> higher limit: RLc -> RLcB8, RLc -> RLcB10, RQ50 -> RQ50B8) and model seed, on the windows scored in both runs:
#   both / removed / added   unsafe in both runs / only at 6.0 / only at the higher limit
#   for the added windows, in the run with the higher limit: tick of the first unsafe step, kind, depth, rule state at the tick in force and at the tick before it;
#     beside, from both runs: the smallest time gap of the window (median and minimum; and how many had it under 1.1 s at 6.0 already, that is near the bar of 1.0 s),
#     and at the first unsafe step of the higher-limit run the ego speed and the gap, higher limit minus 6.0 (median)
#   for the removed windows the same with the roles of the two runs exchanged
#   over all windows scored in both runs: the mean ego speed, higher limit minus 6.0, at the steps 5, 10, 15, 20, 25, 30 (1.0 s ... 6.0 s), and the mean gap difference there
# Run from the headway folder of the pod read kit (numpy; roots from PS_ROOT / PS_OUT): python unsafe_cause_b8.py
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import headway_read_r2 as H  # noqa: E402

ROOT = os.environ.get("PS_ROOT", "F:/TanitAD-scratch/v8proof/integ10k/pull")
OUT = os.environ.get("PS_OUT", "F:/TanitAD-scratch/v8proof/integ10k/read_brake/UNSAFE_CAUSE_B8.json")
ARMS, SEEDS, M_STEPS, SHALLOW_TAU, NEAR_TAU = ("I10R0", "I10RLc", "I10RLcB8", "I10RLcB10", "I10RQ50", "I10RQ50B8"), (0, 1), 10, 0.8, 1.1
PAIRS = (("I10RLc", "I10RLcB8"), ("I10RLc", "I10RLcB10"), ("I10RQ50", "I10RQ50B8"))
STATES = ("no_lead", "sat", "capped", "unchanged")
STEPS = (5, 10, 15, 20, 25, 30)


def rule_rows(arm, k):
    L = np.load(os.path.join(ROOT, "%s_s%d" % (arm, k), "ticks", "rule_log_%s_s%d.npz" % (arm, k)))
    lkey = np.asarray(L["ci"], np.int64) * 1000000 + np.asarray(L["t0"], np.int64)
    row = {(int(x), int(t)): i for i, (x, t) in enumerate(zip(lkey, np.asarray(L["tick"], np.int64)))}
    return row, np.asarray(L["present"]).astype(bool), np.asarray(L["changed"]).astype(bool), np.asarray(L["sat_exec"], np.float64)


def state_of(rr, w, t):
    row, pres, chg, sx = rr
    if t < 0:
        return "first_tick"
    i = row.get((int(w), int(t)))
    if i is None or not pres[i]:
        return "no_lead"
    if sx[i] > 0:
        return "sat"
    return "capped" if chg[i] else "unchanged"


def counts(st, keys):
    st = np.array(st)
    return {s: int((st == s).sum()) for s in keys}


def med(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return round(float(np.median(x)), 4) if len(x) else None


def describe(idx, wkey, new, old, rr_new):
    """idx: windows unsafe in the run `new` and safe in the run `old`. Split in the run `new`; the two runs side by side."""
    if len(idx) == 0:
        return {"n": 0}
    kstar = new["un"][idx].argmax(1) + 1
    q = (kstar - 1) // M_STEPS
    kind_gap = new["un_gap"][idx, kstar - 1]
    shallow = (~new["un_gap"][idx].any(1)) & (new["tmin"][idx] >= SHALLOW_TAU)
    st = [state_of(rr_new, w, t) for w, t in zip(wkey[idx], q)]
    sb = [state_of(rr_new, w, t - 1) for w, t in zip(wkey[idx], q)]
    return {"n": int(len(idx)), "first_unsafe_in_tick": {str(t): int((q == t).sum()) for t in (0, 1, 2)}, "kind": {"gap": int(kind_gap.sum()), "time": int((~kind_gap).sum())},
            "depth": {"shallow": int(shallow.sum()), "deep": int((~shallow).sum())}, "median_first_unsafe_step": med(kstar),
            "rule_state_at_tick_in_force": counts(st, STATES), "rule_state_at_tick_before": counts(sb, STATES + ("first_tick",)),
            "smallest_time_gap_s_in_this_run": {"median": med(new["tmin"][idx]), "min": round(float(np.min(new["tmin"][idx])), 4)},
            "smallest_time_gap_s_in_the_other_run": {"median": med(old["tmin"][idx]), "min": round(float(np.min(old["tmin"][idx])), 4), "under_%.1f_s" % NEAR_TAU: int((old["tmin"][idx] < NEAR_TAU).sum())},
            "at_first_unsafe_step_this_minus_other": {"ego_speed_mps_median": med(new["vm"][idx, kstar - 1] - old["vm"][idx, kstar - 1]), "gap_m_median": med(new["gm"][idx, kstar - 1] - old["gm"][idx, kstar - 1])}}


Wd, missing, where = H.load_world([ROOT], [(a, s) for a in ARMS for s in SEEDS], 141)
assert not missing, missing
J = H.load_join()
res = {"what": "split of the unsafe windows on the runs of card ADD57 and the windows added or removed by the higher braking limit; descriptive, names nothing", "tier": "T1-kin",
       "shallow_time_gap_s": SHALLOW_TAU, "near_bar_time_gap_s": NEAR_TAU, "per_seed": {}}
for k in SEEDS:
    ref = Wd[("I10R0", k)]
    Lw = H.label_windows(J, ref["clip_index"], ref["ws"])
    wkey = np.asarray(ref["clip_index"], np.int64) * 1000000 + np.asarray(ref["ws"], np.int64)
    A = {}
    for a in ARMS:
        assert np.array_equal(np.asarray(Wd[(a, k)]["clip_index"]), np.asarray(ref["clip_index"])) and np.array_equal(np.asarray(Wd[(a, k)]["ws"]), np.asarray(ref["ws"])), "windows differ: " + a
        c = H.window_cells(Wd[(a, k)]["g"], Wd[(a, k)]["cl"], Lw)
        U = np.asarray(c["U"], np.float64)
        gm = np.asarray(c["_gm"], np.float64)
        vm = H.step_speeds(np.asarray(Wd[(a, k)]["cl"], np.float64))
        with np.errstate(invalid="ignore", divide="ignore"):
            tm = gm / np.maximum(vm, H.V_FLOOR)
            un_gap = np.isfinite(gm) & (gm < H.UNSAFE_GAP_M)
            un = un_gap | (np.isfinite(tm) & (tm < H.UNSAFE_TAU_S))
            tmin = np.min(np.where(np.isfinite(tm), tm, np.inf), 1)
        assert un[U == 1].any(1).all(), "a window with U = 1 has no unsafe step in this recomputation"
        A[a] = {"U": U, "gm": gm, "vm": vm, "un": un, "un_gap": un_gap, "tmin": tmin}
    U0 = A["I10R0"]["U"]
    S = {"arms": {}, "pairs": {}}
    for a in ARMS:
        U = A[a]["U"]
        idx = np.flatnonzero(U == 1)
        kstar = A[a]["un"][idx].argmax(1) + 1
        q = (kstar - 1) // M_STEPS
        rec = {"n_scored": int(np.isfinite(U).sum()), "n_unsafe": int(len(idx)), "U": float(np.nanmean(U)), "also_unsafe_without_rule": int((U0[idx] == 1).sum()),
               "first_unsafe_in_tick": {str(t): int((q == t).sum()) for t in (0, 1, 2)}}
        if a != "I10R0":
            rr = rule_rows(a, k)
            rec["rule_state_at_tick_in_force"] = counts([state_of(rr, w, t) for w, t in zip(wkey[idx], q)], STATES)
        S["arms"][a] = rec
        print("s%d %-10s unsafe %3d of %d (U %.4f) | also unsafe without rule %d | tick %s | state %s" % (
            k, a, rec["n_unsafe"], rec["n_scored"], rec["U"], rec["also_unsafe_without_rule"], json.dumps(rec["first_unsafe_in_tick"]), json.dumps(rec.get("rule_state_at_tick_in_force", {}))))
    for old, new in PAIRS:
        Uo, Un = A[old]["U"], A[new]["U"]
        sc = np.isfinite(Uo) & np.isfinite(Un)
        both, rem, add = np.flatnonzero(sc & (Uo == 1) & (Un == 1)), np.flatnonzero(sc & (Uo == 1) & (Un == 0)), np.flatnonzero(sc & (Uo == 0) & (Un == 1))
        dv = A[new]["vm"][sc] - A[old]["vm"][sc]
        dg = A[new]["gm"][sc] - A[old]["gm"][sc]
        with np.errstate(invalid="ignore"):
            rec = {"n_scored_in_both": int(sc.sum()), "both": int(len(both)), "removed": int(len(rem)), "added": int(len(add)),
                   "added_windows_in_the_higher_limit_run": describe(add, wkey, A[new], A[old], rule_rows(new, k)),
                   "removed_windows_in_the_6.0_run": describe(rem, wkey, A[old], A[new], rule_rows(old, k)),
                   "mean_ego_speed_mps_higher_minus_6.0_at_step": {str(s): round(float(np.nanmean(dv[:, s - 1])), 4) for s in STEPS},
                   "mean_gap_m_higher_minus_6.0_at_step": {str(s): round(float(np.nanmean(dg[:, s - 1])), 4) for s in STEPS},
                   "windows_with_another_track": int((np.nanmax(np.abs(dv), 1) > 1e-9).sum())}
        S["pairs"]["%s_to_%s" % (old, new)] = rec
        print("s%d %s -> %s: both %d removed %d added %d | windows whose track differs %d of %d" % (k, old, new, rec["both"], rec["removed"], rec["added"], rec["windows_with_another_track"], rec["n_scored_in_both"]))
        print("      added   %s" % json.dumps(rec["added_windows_in_the_higher_limit_run"]))
        print("      removed %s" % json.dumps(rec["removed_windows_in_the_6.0_run"]))
        print("      mean ego speed diff by step %s | mean gap diff by step %s" % (json.dumps(rec["mean_ego_speed_mps_higher_minus_6.0_at_step"]), json.dumps(rec["mean_gap_m_higher_minus_6.0_at_step"])))
    res["per_seed"][str(k)] = S
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
print("ZZ UNSAFE-CAUSE-B8 DONE")
