# unsafe_cause.py : descriptive split of the windows that stay unsafe under the lead rule on the two 10k checkpoints (banked arms of cards ADD46 to ADD48; no new run).
# Names nothing. Tier T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance.
# Convention written into TIMES_NOTE.md before this ran. Per arm (R0 no rule, RLb, RLc label-fed without / with the closing speed, RQ50 reader-fed) and model seed, for
# every scored window that is unsafe under the arm (the headway reader's U = 1):
#   k*    the first unsafe step (1 .. 30, 0.2 s each); the plan in force is the one of tick q = (k* - 1) // 10
#   kind  "gap" if the gap at k* is under 2.0 m, else "time" (time gap under 1.0 s)
#   depth "shallow" if the window never has a gap under 2.0 m and its smallest time gap is at least 0.8 s, else "deep"
#   rule state at tick q, from the rule log row of the window (arms with a rule):
#     no_lead   the rule has no lead at that tick (source absent or not finite)
#     sat       a lead and the braking limit was reached on an executed step (sat_exec > 0)
#     capped    a lead, the plan was changed, the limit not reached
#     unchanged a lead, the plan was left as the model gave it
# Also per arm: of its unsafe windows, how many are unsafe under no rule as well (kept) and how many are new (safe under no rule).
# Run from the headway folder of the pod read kit (numpy; roots from PS_ROOT / PS_OUT): python unsafe_cause.py
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import headway_read_r2 as H  # noqa: E402

ROOT = os.environ.get("PS_ROOT", "F:/TanitAD-scratch/v8proof/integ10k/pull")
OUT = os.environ.get("PS_OUT", "F:/TanitAD-scratch/v8proof/integ10k/read_gaprate/UNSAFE_CAUSE.json")
ARMS, SEEDS, M_STEPS, SHALLOW_TAU = ("I10R0", "I10RLb", "I10RLc", "I10RQ50"), (0, 1), 10, 0.8
STATES = ("no_lead", "sat", "capped", "unchanged")


def rlog(arm, k):
    return np.load(os.path.join(ROOT, "%s_s%d" % (arm, k), "ticks", "rule_log_%s_s%d.npz" % (arm, k)))


Wd, missing, where = H.load_world([ROOT], [(a, s) for a in ARMS for s in SEEDS], 141)
assert not missing, missing
J = H.load_join()
res = {"what": "descriptive split of the windows unsafe under each arm by first unsafe step, kind, depth and the rule state at the tick in force; names nothing", "tier": "T1-kin",
       "shallow_time_gap_s": SHALLOW_TAU, "per_seed": {}}
for k in SEEDS:
    ref = Wd[("I10R0", k)]
    Lw = H.label_windows(J, ref["clip_index"], ref["ws"])
    wkey = np.asarray(ref["clip_index"], np.int64) * 1000000 + np.asarray(ref["ws"], np.int64)
    cells = {a: H.window_cells(Wd[(a, k)]["g"], Wd[(a, k)]["cl"], Lw) for a in ARMS}
    U0 = np.asarray(cells["I10R0"]["U"], np.float64)
    S = {}
    for a in ARMS:
        c = cells[a]
        U = np.asarray(c["U"], np.float64)
        gm = np.asarray(c["_gm"], np.float64)
        vm = H.step_speeds(np.asarray(Wd[(a, k)]["cl"], np.float64))
        with np.errstate(invalid="ignore", divide="ignore"):
            tm = gm / np.maximum(vm, H.V_FLOOR)
            un_gap = np.isfinite(gm) & (gm < H.UNSAFE_GAP_M)
            un = un_gap | (np.isfinite(tm) & (tm < H.UNSAFE_TAU_S))
        idx = np.flatnonzero(U == 1)
        assert un[idx].any(1).all(), "a window with U = 1 has no unsafe step in this recomputation"
        kstar = un[idx].argmax(1) + 1
        q = (kstar - 1) // M_STEPS
        kind_gap = un_gap[idx, kstar - 1]
        with np.errstate(invalid="ignore"):
            tmin = np.nanmin(np.where(np.isfinite(tm[idx]), tm[idx], np.inf), 1)
        shallow = (~un_gap[idx].any(1)) & (tmin >= SHALLOW_TAU)
        rec = {"n_scored": int(np.isfinite(U).sum()), "n_unsafe": int(len(idx)), "U": float(np.nanmean(U)),
               "also_unsafe_without_rule": int((U0[idx] == 1).sum()), "new_against_no_rule": int((U0[idx] == 0).sum()),
               "first_unsafe_in_tick": {str(t): int((q == t).sum()) for t in (0, 1, 2)}, "kind": {"gap": int(kind_gap.sum()), "time": int((~kind_gap).sum())},
               "depth": {"shallow": int(shallow.sum()), "deep": int((~shallow).sum())}, "median_first_unsafe_step": float(np.median(kstar)) if len(idx) else None}
        if a != "I10R0":
            L = rlog(a, k)
            lkey = np.asarray(L["ci"], np.int64) * 1000000 + np.asarray(L["t0"], np.int64)
            ltick = np.asarray(L["tick"], np.int64)
            row = {(int(x), int(t)): i for i, (x, t) in enumerate(zip(lkey, ltick))}
            pres, chg, sx = np.asarray(L["present"]).astype(bool), np.asarray(L["changed"]).astype(bool), np.asarray(L["sat_exec"], np.float64)
            st = []
            for w, t in zip(wkey[idx], q):
                i = row.get((int(w), int(t)))
                if i is None or not pres[i]:
                    st.append("no_lead")
                elif sx[i] > 0:
                    st.append("sat")
                elif chg[i]:
                    st.append("capped")
                else:
                    st.append("unchanged")
            st = np.array(st)
            rec["rule_state_at_tick_in_force"] = {s: int((st == s).sum()) for s in STATES}
            rec["rule_state_by_depth"] = {d: {s: int(((st == s) & m).sum()) for s in STATES} for d, m in (("shallow", shallow), ("deep", ~shallow))}
            rec["rule_state_by_tick"] = {str(t): {s: int(((st == s) & (q == t)).sum()) for s in STATES} for t in (0, 1, 2)}
        S[a] = rec
        print("s%d %-8s unsafe %3d of %d (U %.4f) | kept %d new %d | tick %s | kind %s | depth %s | state %s" % (
            k, a, rec["n_unsafe"], rec["n_scored"], rec["U"], rec["also_unsafe_without_rule"], rec["new_against_no_rule"], json.dumps(rec["first_unsafe_in_tick"]), json.dumps(rec["kind"]),
            json.dumps(rec["depth"]), json.dumps(rec.get("rule_state_at_tick_in_force", {}))))
        if "rule_state_by_depth" in rec:
            print("      by depth %s | by tick %s" % (json.dumps(rec["rule_state_by_depth"]), json.dumps(rec["rule_state_by_tick"])))
    res["per_seed"][str(k)] = S
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
print("ZZ UNSAFE-CAUSE DONE")
