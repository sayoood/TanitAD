#!/usr/bin/env python
"""level_names.py -- names of note PREREG_V8F_PROOF_ADD37_N6 section 4 (level arms) and of note N5 section 3 (dose arms), read from the outputs of imag_read_r2.py.
It computes nothing new: it collects the numbers of the reads and applies the frozen rules in their order. Tier T0 (imagination diagnostics on logged frames);
never driving performance. STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument).

  python level_names.py --read-dir DIR --rlft-dir DIR_OF_RLFT_READ --arms RLVa:level:0.001 RLVb:level:0.01 RLAh:dose:0.0003 [--rla3-dir DIR] [--out FILE]

  DIR/<ARM>/RESULT_IMAG_<ARM>.json, DIR/PAIR_<ARM>_RLFT/PAIR_IMAG_<ARM>_minus_RLFT.json, DIR/PAIR_<ARM>_RLAm/PAIR_IMAG_<ARM>_minus_RLAm.json

KEEPS THE LEVEL (note N4): at each base the mean absolute error of the rollout read is at most 1.15 times that of RLFT, and the two-level interval of
H1(arm) minus H1(RLFT) does not lie below 0. A reader control that failed ends the read of its arm with no name.
"""
import argparse
import io
import json
import os
import sys

BAR = 1.15


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if os.path.isfile(p) else None


def responds(r):
    return "IMAG-RESPONDS" in [x.strip() for x in str(r["names"].get("H2", "")).split("+")]


def mae_i(r):
    return [float(s["i"]) for s in r["H1"]["mae_m"]["per_seed"]]


def read_arm(read_dir, rlft, arm, kind, w, rla3=None):
    o = {"arm": arm, "kind": kind, "weight": w}
    r = jl(os.path.join(read_dir, arm, "RESULT_IMAG_%s.json" % arm))
    pf = jl(os.path.join(read_dir, "PAIR_%s_RLFT" % arm, "PAIR_IMAG_%s_minus_RLFT.json" % arm))
    pm = jl(os.path.join(read_dir, "PAIR_%s_RLAm" % arm, "PAIR_IMAG_%s_minus_RLAm.json" % arm))
    if not (r and pf and pm):
        o["name"] = None
        o["why"] = "a read is missing (arm read %s, pair against RLFT %s, pair against RLAm %s)" % (bool(r), bool(pf), bool(pm))
        return o
    o["controls_pass"] = bool(r.get("controls_pass"))
    o["controls"] = {k: bool(v.get("pass")) for k, v in r.get("controls", {}).items() if isinstance(v, dict) and "pass" in v}
    b2 = r["H2"]["b_T2.0"]
    o["slope_2s"] = {"per_base": b2.get("per_seed"), "pooled": b2.get("pooled"), "interval": b2.get("ci_two_level")}
    o["slope_1s_per_base"] = r["H2"].get("b_T1.0", {}).get("per_seed")
    o["H2_name"] = r["names"].get("H2")
    o["H2_overshoot_seeds"] = r["names"].get("H2_overshoot_seeds")
    o["responds"] = responds(r)
    ma, mb = mae_i(r), mae_i(rlft)
    o["mae_rollout_m"] = ma
    o["mae_rollout_rlft_m"] = mb
    o["mae_ratio"] = [a / b for a, b in zip(ma, mb)]
    d = pf["H1_a_minus_b"]
    o["H1_minus_rlft"] = {"pooled": d["pooled"], "interval": d["ci"]}
    o["keeps_level"] = bool(all(x <= BAR for x in o["mae_ratio"]) and not (d["ci"][1] < 0))
    o["keeps_level_parts"] = {"ratio_at_most_1.15_per_base": [bool(x <= BAR) for x in o["mae_ratio"]], "H1_difference_not_below_0": bool(not (d["ci"][1] < 0))}
    o["pair_against_rlft_names"] = pf.get("names")
    o["pair_against_rlam_names"] = pm.get("names")
    dm = pm.get("abs_dist_from_1_a_minus_b", {})
    o["dist_from_1_minus_rlam"] = {"pooled": dm.get("pooled"), "interval": dm.get("ci")}
    nm = pm.get("names")
    o["better_than_rlam"] = bool(isinstance(nm, (list, tuple)) and "RL-RESPONDS-BETTER" in nm)
    if rla3:
        o["beside_mae_rollout_rla3_m"] = mae_i(rla3)
    if not o["controls_pass"]:
        o["name"] = None
        o["why"] = "a reader control failed: no name"
    elif kind == "level":
        if not o["responds"]:
            o["name"] = "LEVEL-TERM-LOSES-RESPONSE-%s" % w
        elif not o["keeps_level"]:
            o["name"] = "LEVEL-TERM-LOSES-LEVEL-%s" % w
        elif o["better_than_rlam"]:
            o["name"] = "LEVEL-TERM-WORKS-%s" % w
        else:
            o["name"] = "LEVEL-TERM-KEEPS-LEVEL-NO-GAIN-%s" % w
    else:
        if o["responds"] and o["keeps_level"] and o["better_than_rlam"]:
            o["name"] = "DOSE-STRONGER-%s" % w
        elif not o["keeps_level"]:
            o["name"] = "DOSE-BREAKS-AT-%s" % w
        else:
            o["name"] = None
            o["why"] = "responds %s, keeps the level, not named better than RLAm: no name for this weight" % o["responds"]
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--read-dir", required=True)
    ap.add_argument("--rlft-dir", required=True)
    ap.add_argument("--rla3-dir", default="")
    ap.add_argument("--arms", nargs="+", required=True, help="ARM:level|dose:WEIGHT")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    rlft = jl(os.path.join(a.rlft_dir, "RESULT_IMAG_RLFT.json"))
    if not rlft:
        print("no RLFT read in", a.rlft_dir)
        return 3
    rla3 = jl(os.path.join(a.rla3_dir, "RESULT_IMAG_RLA3.json")) if a.rla3_dir else None
    res = {"tier": "T0 (imagination diagnostics on logged frames; never driving performance)",
           "family_line": "STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument)",
           "rules": "note PREREG_V8F_PROOF_ADD37_N6 section 4 (level arms, in order) and note N5 section 3 (dose arms); KEEPS THE LEVEL of note N4, bar %.2f" % BAR,
           "rlft": {"seeds": rlft.get("seeds"), "controls_pass": rlft.get("controls_pass"), "H2_name": rlft["names"].get("H2"), "mae_rollout_m": mae_i(rlft)}, "arms": {}}
    f4 = lambda x: None if x is None else ([round(float(v), 4) for v in x] if isinstance(x, (list, tuple)) else round(float(x), 4))   # noqa: E731
    for spec in a.arms:
        arm, kind, w = spec.split(":")
        assert kind in ("level", "dose"), spec
        o = read_arm(a.read_dir, rlft, arm, kind, w, rla3)
        res["arms"][arm] = o
        print("== %s (%s, weight %s): NAME %s%s" % (arm, kind, w, o.get("name"), (" | " + o["why"]) if o.get("why") else ""))
        if "slope_2s" in o:
            print("   controls %s %s" % ("PASS" if o["controls_pass"] else "FAIL", json.dumps(o["controls"])))
            print("   slope 2.0 s: pooled %s interval %s per base %s | %s | slope 1.0 s per base %s" % (f4(o["slope_2s"]["pooled"]), f4(o["slope_2s"]["interval"]), f4(o["slope_2s"]["per_base"]), o["H2_name"], f4(o["slope_1s_per_base"])))
            print("   rollout MAE m: arm %s RLFT %s ratio %s (bar %.2f) | H1 minus RLFT %s %s | keeps the level %s %s" % (f4(o["mae_rollout_m"]), f4(o["mae_rollout_rlft_m"]), f4(o["mae_ratio"]), BAR,
                  f4(o["H1_minus_rlft"]["pooled"]), f4(o["H1_minus_rlft"]["interval"]), o["keeps_level"], json.dumps(o["keeps_level_parts"])))
            print("   pair against RLAm: names %s, distance from 1 minus RLAm %s %s | pair against RLFT names %s | beside RLA3 MAE %s" % (o["pair_against_rlam_names"], f4(o["dist_from_1_minus_rlam"]["pooled"]),
                  f4(o["dist_from_1_minus_rlam"]["interval"]), o["pair_against_rlft_names"], f4(o.get("beside_mae_rollout_rla3_m"))))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
