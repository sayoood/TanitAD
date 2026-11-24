"""hw3_name.py -- PREREG_V8F_RHHW3 section 3 floor F3 and section 4 name.  Reads the frozen reader's RESULT_HEADWAY.json (3 seeds) and the per-seed arm U levels."""
import itertools, json, os, sys
import numpy as np
NAMES = ("HW3-BLOCKED", "HW3-COST-REAL", "HW3-COST-BELOW-FLOOR", "HW3-NOT-SEPARATED")

def f3(rh, b0):
    d = [abs(x - y) for x, y in itertools.combinations(rh, 2)] + [abs(x - y) for x, y in itertools.combinations(b0, 2)]
    return float(max(d))

def cell_verdict(c, F3):
    """worse = D > 0 (higher unsafe rate). WORSE: interval excludes 0 on worse side, all seed points worse, |D| > F3."""
    ds = c["d_s"]
    worse = c["A_lo"] > 0 and all(x > 0 for x in ds) and abs(c["D"]) > F3
    better = c["A_hi"] < 0 and all(x < 0 for x in ds) and abs(c["D"]) > F3
    v = "WORSE" if worse else "BETTER" if better else "NOT-SEPARATED"
    straddle = v == "NOT-SEPARATED" and (c["A_lo"] > 0 or c["A_hi"] < 0)
    return v, straddle

def name(r0_pass, dumps_ok, c, F3):
    if not r0_pass or not dumps_ok:
        return "HW3-BLOCKED", None
    v, st = cell_verdict(c, F3)
    if v == "WORSE":
        return "HW3-COST-REAL", v
    if v == "NOT-SEPARATED" and ((st and c["A_lo"] > 0) or all(x > 0 for x in c["d_s"])):
        return "HW3-COST-BELOW-FLOOR", v
    return "HW3-NOT-SEPARATED", v

def plant():
    out = {}
    mk = lambda D, lo, hi, ds: dict(D=D, A_lo=lo, A_hi=hi, d_s=ds)
    out["blocked"] = name(False, True, mk(.03, .01, .05, [.03] * 3), .01)[0]
    out["cost_real_+0.03_smallF3"] = name(True, True, mk(.03, .01, .05, [.03, .031, .029]), .005)[0]
    out["below_floor_all_worse_big_F3"] = name(True, True, mk(.03, .01, .05, [.03, .031, .029]), .06)[0]
    out["below_floor_straddle_worse"] = name(True, True, mk(.01, .002, .02, [.03, -.01, .02]), .06)[0]
    out["control_0.0"] = name(True, True, mk(0.0, -.01, .01, [.0, .001, -.001]), .01)[0]
    out["missing_dump"] = name(True, False, mk(.03, .01, .05, [.03] * 3), .01)[0]
    exp = dict(blocked="HW3-BLOCKED", **{"cost_real_+0.03_smallF3": "HW3-COST-REAL", "below_floor_all_worse_big_F3": "HW3-COST-BELOW-FLOOR",
               "below_floor_straddle_worse": "HW3-COST-BELOW-FLOOR", "control_0.0": "HW3-NOT-SEPARATED", "missing_dump": "HW3-BLOCKED"})
    return out, all(out[k] == exp[k] for k in exp) and len(set(out.values())) == 4

if __name__ == "__main__":
    P, ok = plant()
    print("PLANT", ok, P)
    if len(sys.argv) < 2:
        sys.exit(0 if ok else 1)
    res = json.load(open(sys.argv[1]))
    # per-seed arm U levels supplied by hw3_levels.json
    lv = json.load(open(sys.argv[2]))
    r0 = json.load(open(sys.argv[3]))["pass"]
    out = {"R0_pass": r0, "plant_pass": ok, "plant": P}
    for form, tag in (("K", "R0KHb20"), ("Cstar", "R0C10XHb20")):
        t = "I10RH_%s_minus_I10B0_%s" % (tag, tag)
        c = res["pairs"][t]["cells"]
        F3 = f3(lv["RH_" + tag], lv["B0_" + tag])
        out[form] = {"U": c["U"], "C": c["C"], "F": c["F"], "F3": F3, "arm_U_levels": {"RH": lv["RH_" + tag], "B0": lv["B0_" + tag]},
                     "U_verdict": cell_verdict(c["U"], F3)}
    nm, v = name(r0 and ok, True, out["K"]["U"], out["K"]["F3"])
    out["name"] = nm
    out["K_cell_verdict"] = v
    json.dump(out, open("F:/TanitAD-scratch/rollhist/hw3/RESULT_HW3.json", "w"), indent=1)
    print(nm, v)
