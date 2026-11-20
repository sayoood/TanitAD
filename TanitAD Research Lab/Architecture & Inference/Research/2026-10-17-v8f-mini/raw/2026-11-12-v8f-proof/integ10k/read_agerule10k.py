#!/usr/bin/env python
"""read_agerule10k.py -- reader of part A of card PREREG_V8F_PROOF_ADD71 (AGERULE10K).

It is the reader of card ADD70 under its notes N3 and N4 (read_longroll10k.py through read_longroll10k_n3.py and read_longroll10k_n4.py; note N1 of card ADD71) with the run and pair tables of card ADD71. What differs, and only this:
  * tables: three new runs at P = 30 (both model seeds), six named pairs, three pairs on fresh windows and two rule pairs beside, no P = 60 run (that grid read and control l11 are stated
    as not applicable);
  * control l8 for R0C10XP30Hb20 expects the causal readout on the executed path with gain 1 (the reader's own check expects gain 0 for every readout that is not the pinned one);
  * control a1: the pair R0KP30Hb20 minus R0KHb20 read here equals the pair AGE-K of the banked result of card ADD70 read under its note N4 (file --add70-n3-json) on every headway share and family cell to 1e-9;
    when a1 fails the names, pairs, cells, readings and the beside block are removed from the result and not printed;
  * the sentences of section 2 of the card.
Usage: the arguments of read_longroll10k.py plus --add70-n3-json <file>.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys

import read_longroll10k_n4 as N4  # noqa: E402  (the reader of card ADD70 under its notes N3 and N4: own start point of a pre-rolled window)

N3 = N4.N3
R = N3.R
np = R.np
R.CARD, R.PRE = "PREREG_V8F_PROOF_ADD71", "AGERULE10K"
R.RUNS = {"RLcP30Hb20": ("RLc", "pinned", 30, "deadreckon", "30", "plan", None, 0.2), "RQ50P30Hb20": ("RR", "pinned", 30, "deadreckon", "30", "plan", None, 0.2),
          "R0C10XP30Hb20": ("R0", "CX", 30, "deadreckon", "30", "plan", None, 0.2)}
R.BANKED = dict(R.BANKED)
R.BANKED.update({"R0P30Hb20": ("R0", "pinned", 30, "deadreckon", "30", "plan", None, 0.2), "R0KP30Hb20": ("R0", "K", 30, "deadreckon", "30", "plan", None, 0.2),
                 "RLcKF5P30Hb20": ("RLc", "K", 30, "deadreckon", "30", "plan", 5, 0.2), "RQ50KF5P30Hb20": ("RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2),
                 "RLcHb20": ("RLc", "pinned", 0, "deadreckon", "full", "plan", None, 0.2), "RQ50Hb20": ("RR", "pinned", 0, "deadreckon", "full", "plan", None, 0.2),
                 "R0C10XHb20": ("R0", "CX", 0, "deadreckon", "full", "plan", None, 0.2)})
R.ALL_RUNS = dict(R.RUNS)
R.ALL_RUNS.update(R.CTRL_RUNS)
R.ALL_RUNS.update(R.BANKED)
R.NEW_RUN_LIST = [(r, k) for r in R.RUNS for k in R.SEEDS] + [(r, 0) for r in R.CTRL_RUNS]
R.NAMED_PAIRS = {"LABEL-CAND-VS-RULE-AT-AGE": ("RLcKF5P30Hb20", "RLcP30Hb20"), "READER-CAND-VS-RULE-AT-AGE": ("RQ50KF5P30Hb20", "RQ50P30Hb20"), "ACC-AT-AGE": ("R0C10XP30Hb20", "R0KP30Hb20"),
                 "AGE-LABEL-RULE": ("RLcP30Hb20", "RLcHb20"), "AGE-READER-RULE": ("RQ50P30Hb20", "RQ50Hb20"), "AGE-ACC": ("R0C10XP30Hb20", "R0C10XHb20")}
A1 = {"A1-AGE-K-AS-CARD-ADD70": "AGE-K", "A1-AGE-CONV-AS-CARD-ADD70": "AGE-CONV"}                  # pair here -> pair of the banked result (the second also brings the spread run R0Hb20 into the grid read)
R.FRESH_PAIRS = {"LABEL-CAND-VS-RULE-AT-FRESH": ("RLcKF5Hb20", "RLcHb20"), "READER-CAND-VS-RULE-AT-FRESH": ("RQ50KF5Hb20", "RQ50Hb20"), "ACC-AT-FRESH": ("R0C10XHb20", "R0KHb20"),
                 "A1-AGE-K-AS-CARD-ADD70": ("R0KP30Hb20", "R0KHb20"), "A1-AGE-CONV-AS-CARD-ADD70": ("R0P30Hb20", "R0Hb20")}
R.DOD = {"LABEL-CAND-VS-RULE-AT-AGE": ("LABEL-CAND-VS-RULE-AT-AGE", "LABEL-CAND-VS-RULE-AT-FRESH"), "READER-CAND-VS-RULE-AT-AGE": ("READER-CAND-VS-RULE-AT-AGE", "READER-CAND-VS-RULE-AT-FRESH"),
         "ACC-AT-AGE": ("ACC-AT-AGE", "ACC-AT-FRESH")}
R.RULE_AT_AGE = {"LABEL-RULE-IN-CONVENTION-AT-AGE": ("RLcP30Hb20", "R0P30Hb20"), "READER-RULE-IN-CONVENTION-AT-AGE": ("RQ50P30Hb20", "R0P30Hb20")}
R.P60_PAIRS = {}
CX_PROBLEM = "readout is not K (causal_g, gain 0, path-exec)"


def consequences_add71(Rd):
    out = []
    for nm in ("LABEL-CAND-VS-RULE-AT-AGE", "READER-CAND-VS-RULE-AT-AGE"):
        rd = Rd[nm]
        better = [f for f in R.FAMS if rd["fam"][f] == "BETTER"]
        worse = [f for f in R.FAMS if rd["fam"][f] == "WORSE"]
        if better and not worse and rd["U"] != "HIGHER":
            out.append(("HOLDS", "%s: that candidate form keeps its reading against the convention with the rule at an age of 6 to 12 s (BETTER on %s, no family WORSE, U %s)" % (nm, ",".join(better), rd["U"])))
        elif worse or rd["U"] == "HIGHER":
            out.append(("LIMIT", "%s: the specification states that limit at age (WORSE on %s; U %s)" % (nm, ",".join(worse) or "no family", rd["U"])))
        else:
            out.append(("AS-WRITTEN", "%s: reported as written (%s; U %s)" % (nm, rd["fam"], rd["U"])))
    rd = Rd["ACC-AT-AGE"]
    worse = [f for f in R.FAMS if rd["fam"][f] == "WORSE"]
    if worse:
        out.append(("C1-AT-AGE", "ACC-AT-AGE: the cost of handing the model its own acceleration holds at age (WORSE on %s); its values are the line-model reference of acceptance check C1 at age" % ",".join(worse)))
    else:
        out.append(("C1-FRESH-ONLY", "ACC-AT-AGE: no family WORSE; at age the term does not separate from the drift on these windows, C1 is read on fresh windows only"))
    for nm in ("AGE-LABEL-RULE", "AGE-READER-RULE", "AGE-ACC"):
        out.append(("NAMED-ONLY", "%s: named and reported as written; decides nothing by itself (%s; U %s)" % (nm, Rd[nm]["fam"], Rd[nm]["U"])))
    return out


def file_controls_add71(a, keys_by_grid, names, log):
    C = _n3_file_controls(a, keys_by_grid, names, log)
    if not any(R.ALL_RUNS[r][2] == 60 for r, _ in R.NEW_RUN_LIST):
        C["l11_segments"] = {"PASS": True, "note": "not applicable: no P = 60 run in this card", "runs": {}}
    l8 = C.get("l8_readout_logs") or {}
    fixed = {}
    for run, k in R.NEW_RUN_LIST:
        if R.ALL_RUNS[run][1] != "CX":
            continue
        key = R.key_of(run, k)
        ent = (l8.get("runs") or {}).get(key)
        RL = R.jl(os.path.join(R.tick_dir(a.pull, run, k), "readout_log_%s.json" % key))
        if ent is None or RL is None:
            continue
        dv, dg = RL.get("max_abs_speed_minus_last_step"), RL.get("max_abs_acceleration_minus_gain_executed")
        cx = bool(RL.get("readout") == "causal_g" and RL.get("acc_gain") == 1.0 and RL.get("path_exec") is True and dv is not None and dg is not None and dv <= R.TOL and dg <= R.TOL)
        fixed[key] = {"causal_g_gain_1_path_exec": cx, "acc_gain": RL.get("acc_gain"), "max_abs_speed_minus_last_step": dv, "max_abs_acceleration_minus_gain_executed": dg}
        if cx and CX_PROBLEM in ent.get("problems", []):
            ent["problems"] = [p for p in ent["problems"] if p != CX_PROBLEM]
        elif not cx and CX_PROBLEM not in ent.get("problems", []):
            ent.setdefault("problems", []).append("readout is not the causal readout on the executed path with gain 1")
    if l8:
        l8["cx_runs_checked_for_gain_1"] = fixed
        l8["PASS"] = bool(l8.get("runs") and all(not v.get("problems") for v in l8["runs"].values()))
    return C


def read_grid_add71(a, E, grid_name, keys, runs, spread_run, pairs, dod_defs, log):
    if grid_name == "60" and not pairs:
        return {"restriction": {}, "controls": {"ALL_PASS": True, "note": "not applicable: no P = 60 run in this card"}, "pairs": {}, "levels": {}, "counts": {}, "dod": {}, "missing_or_incomplete": [],
                "spread": {"headway": {"U": None}}}
    return _n3_read_grid(a, E, grid_name, keys, runs, spread_run, pairs, dod_defs, log)


def maxabs(x, y):
    return float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))


def a1_check(res, banked):
    """the pair R0KP30Hb20 minus R0KHb20 of this read against the pair AGE-K of the banked result of card ADD70"""
    try:
        if banked.get("controls_all_pass") is not True:
            return {"PASS": False, "why": "the banked result is not admitted"}
        d, di, n, same_cells = [], [], 0, True
        for nm, bnm in A1.items():
            here = res["beside"]["AT_FRESH_pairs"][nm]
            bh, bc = banked["pairs"][bnm]["headway"], banked["cells"][bnm]
            if not (here["A"] == banked["pairs"][bnm]["A"] and here["B"] == banked["pairs"][bnm]["B"]):
                return {"PASS": False, "why": "%s names other runs than %s of the banked result" % (nm, bnm)}
            d += [maxabs(here["headway"][c]["pooled"], bh[c]["pooled"]) for c in ("U", "C", "F")] + [maxabs(here["headway"][c]["per_seed"], bh[c]["per_seed"]) for c in ("U", "C", "F")]
            same_cells = same_cells and sorted(here["cells"]) == sorted(bc)
            d += [maxabs(here["cells"][c]["D"], bc[c]["D"]) for c in bc if c in here["cells"]] + [maxabs(here["cells"][c]["per_seed"], bc[c]["per_seed"]) for c in bc if c in here["cells"]]
            di += [maxabs(here["cells"][c]["interval"], bc[c]["interval"]) for c in bc if c in here["cells"]] + [maxabs(here["headway"][c]["interval"], bh[c]["interval"]) for c in ("U", "C", "F")]
            n += len(bc)
        return {"PASS": bool(same_cells and d and max(d) <= 1e-9), "largest_difference_of_values": max(d), "largest_difference_of_interval_ends_reported_not_judged": max(di), "cells_compared": n,
                "pairs_compared": dict(A1), "tolerance": 1e-9}
    except Exception as ex:                                                                   # noqa: BLE001
        return {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:200])}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    i = argv.index("--add70-n3-json")
    banked_path = argv[i + 1]
    argv = argv[:i] + argv[i + 2:]
    out_path = argv[argv.index("--out") + 1]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = N3.main(argv)
    text = buf.getvalue()
    if rc != 0 or not os.path.exists(out_path):
        print(text)
        print("ZZ READ-AGERULE rc %s (no result file)" % rc)
        return rc or 3
    res = json.load(io.open(out_path, encoding="utf-8"))
    res, text = N4.finish(res, text, None)                                                    # controls n1 to n3 of note N4; removes the names when one fails
    banked = R.jl(banked_path)
    if res.get("controls_all_pass") is True:
        a1 = a1_check(res, banked) if banked else {"PASS": False, "why": "banked result of card ADD70 missing"}
    else:
        a1 = {"PASS": None, "why": "not computed: a control of the reader did not pass"}
    res["controls"]["a1_age_k_equals_card_add70"] = a1
    ok = bool(res.get("controls_all_pass") is True and a1.get("PASS") is True)
    res["controls_all_pass"] = ok
    res["card"] = R.CARD
    if ok:
        print(text.replace("== controls (e1, l1 to l12)", "== controls (e1, l1 to l12; a1 below)"))
        print("   a1_age_k_equals_card_add70 PASS %s" % json.dumps(a1))
    else:
        for k in ("names", "pairs", "cells", "families", "strategic", "beside", "reported_no_bar"):
            res[k] = None
        cut = text.find("== names")
        print(text if cut < 0 else text[:cut])
        print("   a1_age_k_equals_card_add70 %s %s" % ("FAIL" if a1.get("PASS") is False else "NOT COMPUTED", json.dumps(a1)))
        print("== a control did not pass: no name, no level, no difference; the read is reported as written")
    json.dump(res, io.open(out_path, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZ READ-AGERULE rc %s controls_all_pass %s" % (rc, ok))
    return 0


_n3_file_controls = R.file_controls
_n3_read_grid = R.read_grid
R.file_controls = file_controls_add71
R.read_grid = read_grid_add71
R.consequences = consequences_add71

if __name__ == "__main__":
    sys.exit(main())
