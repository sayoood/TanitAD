#!/usr/bin/env python3
"""dagoacc_t1_selftest.py -- self-test and mutation proof of the DAG-OACC dev-box T1 chain (dagoacc_read.py, dagoacc_t1_pullcheck.py, dagoacc_t1_chain.sh).  Plain script, no pytest, CPU, NO ROLLOUT.
The default run is SYNTHETIC: it never hashes or opens a DAG-OACC checkpoint or eval file (the reads stay blind until the box-2 pulls land).  It reads only BANKED DAG material, the pinned inputs, the box-2 w_acc stage record and the
box-2 GATE record (a gate record is a training-time bookkeeping file, not a result).

    python dagoacc_t1_selftest.py                      logic (L), pull check (P), chain dry-run scenarios (C) and the reader's REHEARSAL on BANKED data (H: DAG-0 stands in for DAG-OACC; ~1 min) on the files beside this script -> exit 0 iff every check is as expected
    python dagoacc_t1_selftest.py --dir DIR --only L,P  the same on the files in DIR (the mutation runner uses this)
    python dagoacc_t1_selftest.py --mutate [--workers 2] [--json mutation.json]
        MUTATION PROOF: each mutant is ONE textual defect in a COPY of dagoacc_read.py / dagoacc_t1_pullcheck.py / dagoacc_t1_chain.sh.  It is KILLED iff some check goes RED on the copy; an anchor that does
        not occur exactly once is an ANCHOR ERROR; the UNMUTATED control must be green.  Chain mutants run only the scenario that targets them.
    python dagoacc_t1_selftest.py --real-pulls         ONLY once the box-2 pulls exist: pull check COMPLETE on both runs (hashes them), then the reader's --preflight (C-O1 / C-O2 / C-O3 on the run records).  Opens no rollout output.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
BASH = "C:/Program Files/Git/bin/bash.exe"
ART = "E:/Projects/TanitAD-artifacts"
DAGART = f"{ART}/v8f-dagger"
OACCART = f"{ART}/v8f-dagoacc"
R2 = "C:/Users/Admin/box2_ship/gate_oaf_assert_r2.py"
SCRATCH = "F:/TanitAD-scratch/dagoacc_t1_tmp"
RES: list = []


def check(name, cond, detail=""):
    RES.append((name, bool(cond), str(detail)[:300]))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "   <- " + str(detail)[:200]), flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def rowd(delta, sep, lo=None, hi=None, family=None):
    r = {"delta": delta, "lo": delta - 0.1 if lo is None else lo, "hi": delta + 0.1 if hi is None else hi, "separated": sep}
    if family:
        r["family"] = family
    return r


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def fn_ast(path, name):
    """ast.dump of the top-level function ``name`` without its docstring (so the copy's comments do not matter, its code does)."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            body = n.body[1:] if (n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str)) else n.body
            return ast.dump(ast.Module(body=body, type_ignores=[]))
    return None


def clean_table(over=None):
    """A Read A table (DAG-OACC - P10) in which every row is computed and nothing is separated-worse; ADE6 clears -2F; every family has a separated-better row."""
    FR = TABLE_FR
    t = {}
    for fam, lab, d in FR.READ_A_ROWS:
        better = {"PRIMARY": -0.5, "LONGITUDINAL": -0.3, "LATERAL": -0.2, "TACTICAL": 0.1, "STRATEGIC": -0.1}[fam]
        t[lab] = {**rowd(better, True), "family": fam, "status": None}
        del t[lab]["status"]
    t[FR.M1_LABEL] = {**rowd(-0.01, False), "family": "crawl"}
    for lab, r in (over or {}).items():
        if r is None:
            t.pop(lab, None)
        else:
            t[lab] = {**t.get(lab, {"family": "x"}), **r}
    return t


TABLE_FR = None


# ================================================================================================================ L: logic
def run_logic(D):
    global TABLE_FR
    FR = load("fr_under_test", os.path.join(D, "dagoacc_read.py"))
    TABLE_FR = FR
    two = 0.2415
    check("L0 F, 2F, F/2, B, tolerance are the PREREG's", (FR.F_M, FR.TWO_F, FR.HALF_F, FR.N_BOOT, FR.W_REL_TOL) == (0.1207, 0.2415, 0.0604, 2000, 1e-4), (FR.F_M, FR.TWO_F, FR.HALF_F, FR.N_BOOT, FR.W_REL_TOL))
    txt = open(os.path.join(HERE, "inputs", "PREREG_V8F_DAG_OACC.md"), encoding="utf-8").read()
    check("L1 the real pinned PREREG: section 2 + 4 cells verbatim, no problem", FR.prereg_problems(txt) == [], FR.prereg_problems(txt))
    bad_cells = {
        "the PRIMARY-ACC bar changed (< 0 -> <= 0)": txt.replace("< 0 and separated, at both seeds", "<= 0 and separated, at both seeds"),
        "the KEEP bound changed": txt.replace("upper bound ≤ +F/2 = +0.0604 m, at both seeds", "upper bound ≤ +F/2 = +0.0605 m, at both seeds"),
        "an outcome condition changed": txt.replace("PRIMARY-ACC fails at ≥ 1 seed", "PRIMARY-ACC fails at both seeds"),
        "two outcome rows swapped": txt.replace("| **DOMINANT** |", "| **TMPX** |").replace("| **LONGITUDINAL FIXED** |", "| **DOMINANT** |").replace("| **TMPX** |", "| **LONGITUDINAL FIXED** |"),
        "a section-2 row dropped": txt.replace("| **LON-CLEAR** | every LONGITUDINAL row of Read A, DAG-OACC − P10 | none separated-worse, at both seeds |\n", ""),
        "the order sentence changed": txt.replace("REFUSED → NO EFFECT → COSTS POSITION", "REFUSED → COSTS POSITION → NO EFFECT"),
        "control C-O3 dropped": txt.replace("**C-O3, the weight:**", "**C-O9, the weight:**"),
        "the C-O3 tolerance changed": txt.replace("to within 1e-4 relative", "to within 1e-3 relative"),
    }
    ne = [ln for ln in txt.split(chr(10)) if ln.startswith("| **NOT FIXED**")]
    bad_cells["an extra outcome row appended"] = txt.replace(ne[0], ne[0] + chr(10) + ne[0].replace("NOT FIXED", "EXTRA ROW"), 1) if len(ne) == 1 else txt
    for nm, t in bad_cells.items():
        check(f"L1b PREREG text with {nm} -> a problem is reported", t != txt and FR.prereg_problems(t) != [], "no problem reported" if t != txt else "the mutation did not change the text")
    ok, _ = FR.check_text_pin(os.path.join(HERE, "inputs", "PREREG_V8F_DAG_OACC.md"), FR.PREREG_OACC_SHA256, "x")
    ok2, _ = FR.check_text_pin(os.path.join(HERE, "inputs", "PREREG_V8F_DAG_REREAD.md"), FR.PREREG_OACC_SHA256, "x")
    check("L1c the pin check passes the pinned file and refuses another file", ok and not ok2, (ok, ok2))
    # --- PRIMARY-ACC and KEEP
    check("L2 primary_acc: separated and delta < 0 passes", FR.primary_acc(rowd(-0.2, True)), "")
    check("L2 primary_acc: delta exactly 0 does not pass (strict)", not FR.primary_acc(rowd(0.0, True)), "")
    check("L2 primary_acc: negative but not separated does not pass (a point is not a CI)", not FR.primary_acc(rowd(-0.9, False)), "")
    check("L2 primary_acc: separated-worse does not pass", not FR.primary_acc(rowd(0.3, True)), "")
    check("L3 keep_ok: upper bound exactly +F/2 passes (inclusive)", FR.keep_ok({"hi": 0.0604}), "")
    check("L3 keep_ok: upper bound just above +F/2 fails", not FR.keep_ok({"hi": 0.0605}), "")
    check("L3 keep_ok: reads the UPPER bound, not the point", not FR.keep_ok({"delta": -1.0, "lo": -2.0, "hi": 0.2}) and FR.keep_ok({"delta": 0.05, "lo": 0.04, "hi": 0.06}), "")
    check("L3 keep_ok: a missing, NaN or non-numeric bound fails", not FR.keep_ok({}) and not FR.keep_ok({"hi": float("nan")}) and not FR.keep_ok({"hi": None}) and not FR.keep_ok({"hi": True}), "")
    check("L4 sep_worse: error row, positive separated is worse; negative separated is not", FR.sep_worse(rowd(0.1, True), "lower") and not FR.sep_worse(rowd(-0.1, True), "lower"), "")
    check("L4 sep_worse: agreement row (TAC), NEGATIVE separated is worse", FR.sep_worse(rowd(-0.1, True), "higher") and not FR.sep_worse(rowd(0.1, True), "higher"), "")
    check("L4 sep_worse: an unseparated row is never worse", not FR.sep_worse(rowd(9.0, False), "lower") and not FR.sep_worse(rowd(-9.0, False), "higher"), "")
    check("L4 sep_better is the mirror", FR.sep_better(rowd(-0.1, True), "lower") and FR.sep_better(rowd(0.1, True), "higher") and not FR.sep_better(rowd(-0.1, False), "lower"), "")
    # --- LON-CLEAR, lateral / crawl rows, D1-D3 on synthetic Read A tables
    t0 = clean_table()
    check("L5 lon_status: a clean table is clear", FR.lon_status(t0)["clear"] and FR.lon_status(t0)["worse"] == [], FR.lon_status(t0))
    for lab in FR.LON_ROWS:
        tt = clean_table({lab: rowd(0.2, True)})
        check(f"L5 lon_status: {lab} separated-worse -> not clear, named", (not FR.lon_status(tt)["clear"]) and FR.lon_status(tt)["worse"] == [lab], FR.lon_status(tt))
    check("L5 lon_status: accel MAE separated-better is clear (the lever's own move)", FR.lon_status(clean_table({"accel MAE": rowd(-0.9, True)}))["clear"], "")
    tt = clean_table({"DK-HW": {"status": "NOT COMPUTED"}})
    check("L5 lon_status: a NOT COMPUTED longitudinal row is not clear and is listed (never silently passing)", (not FR.lon_status(tt)["clear"]) and FR.lon_status(tt)["not_computed"] == ["DK-HW"], FR.lon_status(tt))
    tt = clean_table({"DK-TG": None})
    check("L5 lon_status: a MISSING longitudinal row is not clear", not FR.lon_status(tt)["clear"], FR.lon_status(tt))
    check("L5 lon_status: lateral / tactical / crawl rows worse do not affect LON-CLEAR", FR.lon_status(clean_table({"curvature MAE": rowd(0.2, True), "TAC-LAT": rowd(-0.2, True), FR.M1_LABEL: rowd(0.1, True)}))["clear"], "")
    check("L5 LON_ROWS are the six LONGITUDINAL rows of Read A", FR.LON_ROWS == ("speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG"), FR.LON_ROWS)
    check("L6 latcrawl_worse: clean -> none", FR.latcrawl_worse(t0) == [], FR.latcrawl_worse(t0))
    check("L6 latcrawl_worse: the six lateral rows are each caught", all(FR.latcrawl_worse(clean_table({lab: rowd(0.2, True)})) == [lab] for lab in FR.LAT_ROWS), "")
    check("L6 latcrawl_worse: the crawl share M1 is caught", FR.latcrawl_worse(clean_table({FR.M1_LABEL: rowd(0.2, True)})) == [FR.M1_LABEL], "")
    check("L6 latcrawl_worse: a TACTICAL or longitudinal row worse is NOT a lateral / crawl row", FR.latcrawl_worse(clean_table({"TAC-LON": rowd(-0.2, True), "accel MAE": rowd(0.2, True)})) == [], "")
    d = FR.d_flags(t0)
    check("L7 d_flags: the clean table gives D1, D2, D3", d["D1"] and d["D2"] and d["D3"], d)
    check("L7 d_flags: D1 exactly -2F and separated holds (inclusive)", FR.d_flags(clean_table({"ADE6": rowd(-two, True)}))["D1"], "")
    check("L7 d_flags: D1 just above -2F fails", not FR.d_flags(clean_table({"ADE6": rowd(-0.2414, True)}))["D1"], "")
    check("L7 d_flags: D1 needs the interval to separate", not FR.d_flags(clean_table({"ADE6": rowd(-0.5, False)}))["D1"], "")
    check("L7 d_flags: D2 fails on a separated-worse table row (L2 is a table row)", not FR.d_flags(clean_table({"L2 curvature MAE (GT>=5.2)": rowd(0.1, True)}))["D2"], "")
    check("L7 d_flags: D2 does NOT count the crawl share M1 (build_report.verdict)", FR.d_flags(clean_table({FR.M1_LABEL: rowd(0.3, True)}))["D2"], "")
    check("L7 d_flags: a NOT COMPUTED row fails D2", not FR.d_flags(clean_table({"S-POS@30": {"status": "NOT COMPUTED"}}))["D2"], "")
    check("L7 d_flags: a TACTICAL agreement row negative-separated fails D2", not FR.d_flags(clean_table({"TAC-LAT": rowd(-0.2, True)}))["D2"], "")
    tt = clean_table({lab: rowd(0.0, False) for lab in ("TAC-LAT", "TAC-LON")})
    check("L7 d_flags: a family with no separated-better row fails D3 (all four families are needed)", not FR.d_flags(tt)["D3"] and FR.d_flags(tt)["D2"], FR.d_flags(tt)["separated_better_rows_by_family"])
    tt = clean_table({lab: rowd(0.0, False) for lab in ("S-POS@30", "S-POS@50")})
    check("L7 d_flags: one STRATEGIC row separated-better still gives D3", FR.d_flags(tt)["D3"], "")
    check("L7 crawl_rows_fail: M1 or L2 separated-worse at one seed fails the crawl rows; clean passes",
          FR.crawl_rows_fail({0: t0, 1: clean_table({FR.M1_LABEL: rowd(0.2, True)})}) and FR.crawl_rows_fail({0: clean_table({"L2 curvature MAE (GT>=5.2)": rowd(0.2, True)}), 1: t0}) and not FR.crawl_rows_fail({0: t0, 1: t0}), "")
    # --- the decision, on seed flags
    def sfl(pa=True, keep=True, d1=True, d2=True, d3=True, lon=True, lat=False):
        return {"PA": pa, "KEEP": keep, "D1": d1, "D2": d2, "D3": d3, "LON_CLEAR": lon, "LATCRAWL_WORSE": lat}
    G = sfl()

    def dec(a, b, cf=(), reading=None, m1=None):
        return FR.decide(list(cf), {0: a, 1: b}, reading, m1)["OUTCOME"]
    check("L8 DOMINANT: both seeds pass everything", dec(G, G) == "DOMINANT", dec(G, G))
    check("L8 REFUSED precedes everything", dec(sfl(pa=False), G, cf=["C-O3"]) == "REFUSED" and dec(G, G, cf=["C-O1"]) == "REFUSED", "")
    check("L8 NO EFFECT: PRIMARY-ACC fails at seed 1 only", dec(G, sfl(pa=False)) == "NO EFFECT", dec(G, sfl(pa=False)))
    check("L8 NO EFFECT: PRIMARY-ACC fails at seed 0 only", dec(sfl(pa=False), G) == "NO EFFECT", dec(sfl(pa=False), G))
    check("L8 NO EFFECT precedes COSTS POSITION (PA fails at one seed, KEEP at the other)", dec(sfl(pa=False), sfl(keep=False)) == "NO EFFECT", dec(sfl(pa=False), sfl(keep=False)))
    check("L8 COSTS POSITION: KEEP fails at seed 0 only", dec(sfl(keep=False), G) == "COSTS POSITION", dec(sfl(keep=False), G))
    check("L8 COSTS POSITION: KEEP fails at seed 1 only", dec(G, sfl(keep=False)) == "COSTS POSITION", dec(G, sfl(keep=False)))
    check("L8 COSTS POSITION precedes DOMINANT", dec(sfl(keep=False), sfl(keep=False)) == "COSTS POSITION", "")
    check("L8 DOMINANT needs D1 at both seeds", dec(G, sfl(d1=False)) != "DOMINANT", dec(G, sfl(d1=False)))
    check("L8 DOMINANT needs D2 at both seeds", dec(sfl(d2=False), G) != "DOMINANT", dec(sfl(d2=False), G))
    check("L8 DOMINANT needs D3 at both seeds", dec(G, sfl(d3=False)) != "DOMINANT", dec(G, sfl(d3=False)))
    LF0 = sfl(d2=False, lat=True)
    check("L8 LONGITUDINAL FIXED: LON-CLEAR at both, a lateral / crawl row worse at both", dec(LF0, LF0) == "LONGITUDINAL FIXED", dec(LF0, LF0))
    check("L8 LONGITUDINAL FIXED: the lateral / crawl row at ONE seed suffices (reading a)", dec(LF0, sfl()) == "LONGITUDINAL FIXED", dec(LF0, sfl()))
    check("L8 LONGITUDINAL FIXED needs LON-CLEAR at BOTH seeds -> a longitudinal row worse at one seed is NOT FIXED", dec(LF0, sfl(d2=False, lon=False, lat=True)) == "NOT FIXED", dec(LF0, sfl(d2=False, lon=False, lat=True)))
    check("L8 NOT FIXED: a longitudinal row worse at seed 0 only", dec(sfl(d2=False, lon=False), G) == "NOT FIXED", dec(sfl(d2=False, lon=False), G))
    check("L8 NOT FIXED: a longitudinal row worse at seed 1 only", dec(G, sfl(d2=False, lon=False)) == "NOT FIXED", dec(G, sfl(d2=False, lon=False)))
    check("L8 DOMINANT precedes LONGITUDINAL FIXED (M1 worse, D1-D3 hold): as written, DOMINANT", dec(sfl(lat=True), sfl(lat=True)) == "DOMINANT", dec(sfl(lat=True), sfl(lat=True)))
    check("L8 reading (b): DOMINANT needing the crawl rows to pass moves that cell to LONGITUDINAL FIXED",
          dec(sfl(lat=True), sfl(lat=True), reading={"dom_crawl": True}, m1={0: True, 1: True}) == "LONGITUDINAL FIXED", "")
    check("L8 reading (a): LONGITUDINAL FIXED at both seeds only -> the one-seed cell is UNPLACED", dec(LF0, sfl(), reading={"lf": "both"}) in ("UNPLACED", "DOMINANT") and dec(LF0, sfl(), reading={"lf": "both"}) != "LONGITUDINAL FIXED", "")
    U = sfl(d1=False)
    check("L8 UNPLACED (the PREREG gap): PA, KEEP, LON-CLEAR hold, no lateral / crawl row worse, D1 short", dec(U, G) == "UNPLACED", dec(U, G))
    check("L8 UNPLACED: D3 short, nothing else", dec(G, sfl(d3=False)) == "UNPLACED", dec(G, sfl(d3=False)))
    check("L8 UNPLACED: a TACTICAL / STRATEGIC row worse (D2 short), LON-CLEAR, no lateral worse", dec(sfl(d2=False), G) == "UNPLACED", dec(sfl(d2=False), G))
    check("L8 UNPLACED is never silently turned into NOT FIXED / NOT FIXED needs a longitudinal row", dec(U, U) == "UNPLACED", dec(U, U))
    d = FR.decide([], {0: G, 1: G})
    check("L8 matching rows are returned in table order", d["matching_rows_in_table_order"] == ["DOMINANT"], d["matching_rows_in_table_order"])
    d2 = FR.decide(["C-O2-GATE:DAGOACC_s0"], {0: sfl(pa=False), 1: sfl(pa=False)})
    check("L8 several matching rows come back in table order, the first wins", d2["matching_rows_in_table_order"] == ["REFUSED", "NO EFFECT"] and d2["OUTCOME"] == "REFUSED", d2["matching_rows_in_table_order"])
    try:
        FR.decide([], {0: G})
        check("L8 decide refuses a call without both seeds", False, "no error")
    except ValueError:
        check("L8 decide refuses a call without both seeds", True)
    ob = FR.outcome_block(FR.decide([], {0: G, 1: G}))
    check("L8 outcome_block carries the PREREG's condition and committed next verbatim", ob["condition"] == "PRIMARY-ACC and KEEP pass; Read A's D1, D2 and D3 at both seeds" and ob["committed_next"].startswith("the dominance claim at T1-kin over the kinematic floor"), ob)
    obn = FR.outcome_block(FR.decide([], {0: sfl(d2=False, lon=False), 1: G}))
    check("L8 NOT FIXED's committed next is the HACC re-fit", obn["OUTCOME"] == "NOT FIXED" and obn["committed_next"].startswith("next: HACC's rule re-fitted on DAG-OACC"), obn)
    obu = FR.outcome_block(FR.decide([], {0: U, 1: G}))
    check("L8 an UNPLACED block carries no outcome text, only the gap note", obu["OUTCOME"] == "UNPLACED" and obu["condition"] is None and obu["committed_next"] is None and "PREREG gap" in obu["note"], obu)
    class StubCtl:
        def __init__(self, f):
            self.f = f

        def failed(self):
            return list(self.f)
    oc = FR.outcome_from(StubCtl([]), {0: G, 1: G}, {0: False, 1: False})
    check("L8 outcome_from: no control failed and clean flags -> DOMINANT, with the ambiguity block", oc["OUTCOME"] == "DOMINANT" and "ambiguity" in oc and oc["ambiguity"]["differs"] is False, oc["OUTCOME"])
    oc = FR.outcome_from(StubCtl(["C-O3", "C-O1-ONELEVER:DAGOACC_s0"]), {0: G, 1: G}, {0: False, 1: False})
    check("L8 outcome_from: a failed control reaches decide() -> REFUSED (the controls are not dropped on the way)", oc["OUTCOME"] == "REFUSED" and oc["ambiguity"]["outcome"] == "REFUSED", oc["OUTCOME"])
    oc = FR.outcome_from(StubCtl([]), {0: G, 1: sfl(pa=False)}, {0: False, 1: False})
    check("L8 outcome_from: the seed flags and the controls both reach decide() (PA fails at seed 1 -> NO EFFECT)", oc["OUTCOME"] == "NO EFFECT", oc["OUTCOME"])
    class StubCtl2(StubCtl):
        def __init__(self):
            self.items = {}

        def add(self, name, ok, detail=""):
            self.items[name] = bool(ok)

        def failed(self):
            return sorted(k for k, v in self.items.items() if not v)
    import types

    def verdict_stub(tabs_by_seed):                     # build_report.verdict's contract: {seed: {D1, D2, D3}}
        return {s: {k: FR.d_flags(tabs_by_seed[s], FR.DIRECTIONS)[k] for k in ("D1", "D2", "D3")} for s in (0, 1)}
    Astub = types.SimpleNamespace(BR={"verdict": verdict_stub}, DIRECTIONS=FR.DIRECTIONS)
    od = lambda acc, ade: {FR.ACC_LABEL: acc, FR.ADE_LABEL: ade}                                                  # noqa: E731
    good_od = od(rowd(-0.3, True), rowd(0.0, False, hi=0.03))
    c2 = StubCtl2()
    sf2, ex2, rows2, m1w2 = FR.compose(Astub, c2, {"OP": {0: clean_table({FR.M1_LABEL: rowd(0.2, True)}), 1: clean_table()}, "OD": {0: good_od, 1: good_od}})
    check("L8c compose: the crawl share M1 separated-worse at seed 0 only sets m1w {0: True, 1: False}; PRIMARY-ACC and KEEP come from the DAG-OACC - DAG rows; every compose control passes",
          m1w2 == {0: True, 1: False} and sf2[0]["PA"] and sf2[0]["KEEP"] and sf2[0]["LATCRAWL_WORSE"] and not sf2[1]["LATCRAWL_WORSE"] and c2.failed() == [], (m1w2, c2.failed()))
    c2 = StubCtl2()
    sf2, _, _, _ = FR.compose(Astub, c2, {"OP": {0: clean_table(), 1: clean_table()}, "OD": {0: od(rowd(0.3, True), rowd(0.2, True, lo=0.1, hi=0.3)), 1: good_od}})
    check("L8c compose: PRIMARY-ACC (accel MAE) and KEEP (ADE6) are read from their OWN rows (seed 0: accel worse, ADE6 hi 0.3 -> PA False, KEEP False; seed 1 passes)",
          (sf2[0]["PA"], sf2[0]["KEEP"], sf2[1]["PA"], sf2[1]["KEEP"]) == (False, False, True, True), sf2)
    c2 = StubCtl2()
    FR.compose(Astub, c2, {"OP": {0: clean_table({"DK-HW": {"status": "NOT COMPUTED"}}), 1: clean_table()}, "OD": {0: good_od, 1: good_od}})
    check("L8c compose: a NOT COMPUTED longitudinal row fails LON-ROWS-COMPUTED at that seed only", c2.failed() == ["LON-ROWS-COMPUTED:s0"], c2.failed())
    c2 = StubCtl2()
    FR.compose(Astub, c2, {"OP": {0: clean_table(), 1: clean_table()}, "OD": {0: good_od, 1: od({"status": "NOT COMPUTED", "delta": None, "lo": None, "hi": None, "separated": False}, rowd(0.0, False, hi=0.03))}})
    check("L8c compose: a NOT COMPUTED PRIMARY-ACC row fails LON-ROWS-COMPUTED (it never counts as passing)", c2.failed() == ["LON-ROWS-COMPUTED:s1"], c2.failed())
    c2 = StubCtl2()
    Awrong = types.SimpleNamespace(BR={"verdict": lambda tabs_by_seed: {s: {"D1": False, "D2": False, "D3": False} for s in (0, 1)}}, DIRECTIONS=FR.DIRECTIONS)
    FR.compose(Awrong, c2, {"OP": {0: clean_table(), 1: clean_table()}, "OD": {0: good_od, 1: good_od}})
    check("L8c compose: this reader's D1 / D2 / D3 disagreeing with build_report.verdict fails READA-VERDICT-LINK at both seeds", c2.failed() == ["READA-VERDICT-LINK:s0", "READA-VERDICT-LINK:s1"], c2.failed())
    check("L8 outcome_rc: 0 placed outcomes, 2 REFUSED, 3 UNPLACED", (FR.outcome_rc({"OUTCOME": "NOT FIXED"}), FR.outcome_rc({"OUTCOME": "REFUSED"}), FR.outcome_rc({"OUTCOME": "UNPLACED"}), FR.outcome_rc({"OUTCOME": "DOMINANT"})) == (0, 2, 3, 0), "")
    check("L8 default reading: LF at >= 1 seed, DOMINANT as written", FR.DEFAULT_READING == {"lf": "any", "dom_crawl": False}, FR.DEFAULT_READING)
    amb = FR.ambiguity([], {0: sfl(lat=True), 1: sfl(lat=True)}, {0: True, 1: True})
    check("L8 ambiguity: reports the alternatives and that the word differs (DOMINANT -> LONGITUDINAL FIXED)", amb["outcome"] == "DOMINANT" and amb["differs"] and amb["alternatives"]["dominant_needs_crawl_rows_to_pass"] == "LONGITUDINAL FIXED", amb)
    amb2 = FR.ambiguity([], {0: G, 1: G}, {0: False, 1: False})
    check("L8 ambiguity: no difference on a clean DOMINANT", not amb2["differs"], amb2)
    r = subprocess.run([PY, "-I", os.path.join(HERE, "enumerate_cells.py")], capture_output=True, text=True, env={**os.environ, "DAGOACC_READ_DIR": D})
    check("L9 enumerate_cells: 524,288 cells, decide() == the independent table on every cell, every row reached; the PREREG gap is exactly 255 unplaced cells (rc 3)",
          r.returncode == 3 and "cells: 524288" in r.stdout and "disagreements with the independent table: 0" in r.stdout and "unplaced cells: 255" in r.stdout and "every table row reached by at least one cell: True" in r.stdout,
          r.stdout[-400:] + r.stderr[-200:])
    # --- C-O1 on the real banked DAG argv
    dag = {s: jl(f"{DAGART}/M2b_s{s}_dag/argv.json")["argv"] for s in (0, 1)}
    W = 49.6969
    mine = lambda s: [("D:\\tanitad\\runs\\v8f_mini_full\\M2b_s%d_dagoacc" % s if (i and dag[s][i - 1] == "--out") else t) for i, t in enumerate(dag[s])] + ["--w-acc", "49.6969"]      # noqa: E731
    check("L10 one_lever: DAG's banked argv + --w-acc W (only --out differing) is clean at both seeds", FR.one_lever_problems(mine(0), dag[0], W) == [] and FR.one_lever_problems(mine(1), dag[1], W) == [], FR.one_lever_problems(mine(0), dag[0], W))
    a = mine(0); j = a.index("--steps"); a2 = a[:-2][:j] + ["--w-acc", "49.6969"] + a[:-2][j:]
    check("L10 one_lever: --w-acc anywhere in the argv (not only last) is the same lever", FR.one_lever_problems(a2, dag[0], W) == [], FR.one_lever_problems(a2, dag[0], W))
    check("L10 one_lever: no --w-acc is reported", FR.one_lever_problems(mine(0)[:-2], dag[0], W) != [], "")
    check("L10 one_lever: --w-acc twice is reported", FR.one_lever_problems(mine(0) + ["--w-acc", "49.6969"], dag[0], W) != [], "")
    check("L10 one_lever: a --w-acc value other than the stage's W is reported", FR.one_lever_problems(mine(0)[:-1] + ["42.9311"], dag[0], W) != [], "")
    check("L10 one_lever: a non-numeric / zero / negative / missing --w-acc value is reported", all(FR.one_lever_problems(mine(0)[:-1] + [v], dag[0], W) != [] for v in ("abc", "0", "-5", "nan")) and FR.one_lever_problems(mine(0)[:-1], dag[0], W) != [], "")
    check("L10 one_lever: a second lever (an extra flag) is reported", FR.one_lever_problems(mine(0) + ["--w-tac", "1.0"], dag[0], W) != [], "")
    b = mine(0); b[b.index("--lr") + 1] = "2e-4"
    check("L10 one_lever: a changed shared flag value (--lr) is reported", FR.one_lever_problems(b, dag[0], W) != [], "")
    b = mine(0); k = b.index("--dagger-p"); del b[k:k + 2]
    check("L10 one_lever: a DROPPED flag (--dagger-p) is reported", FR.one_lever_problems(b, dag[0], W) != [], "")
    b = mine(0); b[b.index("--dagger-bank-sha256") + 1] = BANK_OTHER
    check("L10 one_lever: a changed bank sha256 is reported", FR.one_lever_problems(b, dag[0], W) != [], "")
    check("L10 one_lever: a reference that already names --w-acc is reported", FR.one_lever_problems(mine(0), dag[0] + ["--w-acc", "1"], W) != [], "")
    check("L10 bank_problems: DAG's iteration-1 bank at both seeds is clean", FR.bank_problems(mine(0), 0) == [] and FR.bank_problems(mine(1), 1) == [], FR.bank_problems(mine(0), 0))
    check("L10 bank_problems: the other seed's bank sha256, another bank directory, or none is reported", FR.bank_problems(mine(0), 1) != [] and FR.bank_problems([x.replace("dagger_bank\\s0", "dagger_bank_flat\\s0") for x in mine(0)], 0) != []
          and FR.bank_problems(mine(0)[:mine(0).index("--dagger-bank-sha256")], 0) != [], "")
    b = mine(0); b[b.index("--dagger-bank-sha256") + 1] = BANK_OTHER
    pb = FR.bank_problems(b, 0)
    check("L10 bank_problems: the RIGHT bank directory with another sha256 is reported ON THE SHA256 (independent of the one-lever comparison)", pb != [] and any("bank-sha256" in x for x in pb), pb)
    # --- C-O2 on the real GATE record of M2b_s0_dagoacc (a training-time bookkeeping file, not a result) and on mutated copies
    gp = f"{OACCART}/M2b_s0_dagoacc/gate/selfcheck.json"
    sc = jl(gp)
    check("L11 wacc_problems (r2 form): the REAL gate record of M2b_s0_dagoacc passes at W = 49.6969 (a trained start: residual_zero_init False, read_o_grad_diff > 0)", FR.wacc_problems(sc, W) == [] and sc["levers"]["residual_zero_init"] is False, FR.wacc_problems(sc, W))

    def prob(mod, w=W):
        s = copy.deepcopy(sc)
        mod(s)
        return FR.wacc_problems(s, w)
    check("L11 a built w_acc other than the argv's is reported", prob(lambda s: s["levers"].__setitem__("w_acc", 42.9311)) != [] and FR.wacc_problems(sc, 42.9311) != [], "")
    check("L11 an acc term of exactly 0 in G-LIVE is reported", prob(lambda s: s["live"]["terms"].__setitem__("acc", 0.0)) != [], "")
    check("L11 an acc term absent from G-LIVE is reported", prob(lambda s: s["live"]["terms"].pop("acc")) != [], "")
    check("L11 an infinite acc term is reported", prob(lambda s: s["live"]["terms"].__setitem__("acc", float("inf"))) != [], "")
    check("L11 O's head gradient 0 is reported", prob(lambda s: s["acc_probe"].__setitem__("head_o_grad_diff", 0.0)) != [], "")
    check("L11 live.head_o_grad_step0 0 is reported", prob(lambda s: s["live"].__setitem__("head_o_grad_step0", 0.0)) != [], "")
    check("L11 the probe itself not passing is reported", prob(lambda s: s["acc_probe"].__setitem__("pass", False)) != [], "")
    check("L11 the term not entering the total at the declared weight is reported", prob(lambda s: s["acc_probe"].__setitem__("contribution", 0.0)) != [], "")
    check("L11 r2 form: residual_zero_init False and read_o_grad_diff 0 is reported (a trained head MUST pass the term's gradient to read_o)", prob(lambda s: s["acc_probe"].__setitem__("read_o_grad_diff", 0.0)) != [], "")
    check("L11 r2 form: residual_zero_init True and read_o_grad_diff exactly 0.0 is the identity and passes", prob(lambda s: (s["levers"].__setitem__("residual_zero_init", True), s["acc_probe"].__setitem__("read_o_grad_diff", 0.0))) == [], "")
    check("L11 r2 form: residual_zero_init True and read_o_grad_diff 864.79 is reported (the first attempt's STOP)", prob(lambda s: (s["levers"].__setitem__("residual_zero_init", True), s["acc_probe"].__setitem__("read_o_grad_diff", 864.79))) != [], "")
    check("L11 r2 form: residual_zero_init neither True nor False is reported (not decidable)", prob(lambda s: s["levers"].__setitem__("residual_zero_init", None)) != [], "")
    for k in ("control_repeat_total_diff", "control_repeat_grad_diff", "control_prior_grad_diff", "control_wm_grad_diff"):
        check(f"L11 the registered control {k} must read EXACTLY 0.0", prob(lambda s, k=k: s["acc_probe"].__setitem__(k, 1e-12)) != [], "")
    check("L11 live.missing_terms non-empty is reported", prob(lambda s: s["live"].__setitem__("missing_terms", ["acc"])) != [], "")
    if os.path.isfile(R2):
        check("L11 AST: this reader's wacc_problems IS box2_ship/gate_oaf_assert_r2.py's wacc_problems (code, docstring aside)", fn_ast(os.path.join(D, "dagoacc_read.py"), "wacc_problems") == fn_ast(R2, "wacc_problems") is not None,
              "the two ASTs differ")
        check("L11 AST: this reader's finite() IS r2's finite()", fn_ast(os.path.join(D, "dagoacc_read.py"), "finite") == fn_ast(R2, "finite") is not None, "")
        check("L11 the r2 asserter is the pinned file", hashlib.sha256(open(R2, "rb").read()).hexdigest() == FR.R2_ASSERT_SHA256, "")
    else:
        check("L11 AST comparison with r2 skipped: box2_ship/gate_oaf_assert_r2.py is not on this machine", True, "")
    s_ok = {"levers": {"w_acc": W}}
    check("L12 runrec_wacc_problems: summary and selfcheck both carry the built w_acc = W -> clean", FR.runrec_wacc_problems(s_ok, s_ok, W) == [], "")
    check("L12 a run record without a built w_acc is reported", FR.runrec_wacc_problems({"levers": {}}, s_ok, W) != [] and FR.runrec_wacc_problems(s_ok, {}, W) != [], "")
    check("L12 a run record with another built w_acc is reported", FR.runrec_wacc_problems({"levers": {"w_acc": 42.9311}}, s_ok, W) != [] and FR.runrec_wacc_problems(s_ok, {"levers": {"w_acc": 0.0}}, W) != [], "")
    # --- C-O3 on the REAL box-2 w_acc record vs the offline reference, and mutated copies
    box, off = jl(f"{OACCART}/wacc/w_acc.json"), jl(f"{HERE}/inputs/wacc_offline_DAG_s0.json")
    check("L13 wacc_c03: the REAL box-2 w_acc record (49.6969) equals the offline reference within 1e-4 relative, on DAG_s0's checkpoint and the banked fit plans", FR.wacc_c03_problems(box, off, 49.6969, True) == [], FR.wacc_c03_problems(box, off, 49.6969, True))

    def c03(bm=None, om=None, txt=49.6969, shaok=True):
        b, o = copy.deepcopy(box), copy.deepcopy(off)
        if bm:
            bm(b)
        if om:
            om(o)
        return FR.wacc_c03_problems(b, o, txt, shaok)
    ww = lambda b, v: (b.__setitem__("w_acc", v), b.__setitem__("w_acc_raw", v))                                                       # noqa: E731
    ok_lo = c03(lambda b: ww(b, float(f"{off['w_acc'] * (1 + 0.99e-4):.6g}")), txt=float(f"{off['w_acc'] * (1 + 0.99e-4):.6g}"))
    bad_hi = c03(lambda b: ww(b, float(f"{off['w_acc'] * (1 + 1.5e-4):.6g}")), txt=float(f"{off['w_acc'] * (1 + 1.5e-4):.6g}"))
    check("L13 0.99e-4 relative (rounded to 6 figures) passes the tolerance clause; 1.5e-4 is reported", not any("relative difference" in x for x in ok_lo) and any("relative difference" in x for x in bad_hi), (ok_lo, bad_hi))
    check("L13 OACC's M2b value 42.9311 (the disclosed alternative) is REFUSED", any("relative difference" in x for x in c03(lambda b: ww(b, 42.9311), txt=42.9311)), "")
    check("L13 a w_acc.txt that disagrees with the json is reported", c03(txt=49.6970) != [] and c03(txt=None) != [], "")
    check("L13 a box value fitted on another checkpoint than DAG_s0's is reported", c03(lambda b: b["source"].__setitem__("ckpt_sha256", "0" * 64)) != [], "")
    check("L13 other fit windows (box or offline) are reported", c03(lambda b: b["banked_crosscheck"].__setitem__("fit_windows_sha256", "0" * 64)) != [] and c03(om=lambda o: o["source"].__setitem__("fit_windows_sha256", "0" * 64)) != [], "")
    check("L13 another window count is reported", c03(lambda b: b.__setitem__("n_windows", 9000)) != [] and c03(om=lambda o: o.__setitem__("n_windows", 9000)) != [], "")
    check("L13 a w_acc that is not the raw value rounded to 6 figures is reported", c03(lambda b: b.__setitem__("w_acc_raw", 49.70)) != [] and c03(lambda b: b.__setitem__("w_acc_raw", None)) != [], "")
    check("L13 an offline reference that is not the pinned file is reported", c03(shaok=False) != [], "")
    check("L13 non-finite values are reported", c03(lambda b: b.__setitem__("w_acc", float("nan"))) != [] and c03(om=lambda o: o.__setitem__("w_acc", 0.0)) != [], "")
    check("L13 round6: 6 significant figures (OACC R3)", FR.round6(49.69688832944211) == 49.6969 and FR.round6(1234567.0) == 1234570.0, FR.round6(49.69688832944211))
    # --- rollout argv, rows_diff, T0 families, routing, SPEC
    bt = [x for x in open(f"{ART}/dagreread/argv/FLAT_s0_m10_causal_heads.argv", encoding="utf-8").read().split("\n") if x != ""]
    nt = list(bt)
    for i, t in enumerate(nt):
        if t in ("--ckpt", "--tag", "--out-dir", "--dump-ticks"):
            nt[i + 1] = nt[i + 1] + "_X"
    check("L14 rollout_argv_problems: differing only in --ckpt/--tag/--out-dir/--dump-ticks is clean", FR.rollout_argv_problems(nt, bt) == [], FR.rollout_argv_problems(nt, bt))
    nt2 = list(bt); nt2[nt2.index("--limit") + 1] = "logged"
    check("L14 a changed --limit is reported", FR.rollout_argv_problems(nt2, bt) != [], "")
    nt3 = list(bt); nt3[nt3.index("--scene") + 1] = "frozen"
    check("L14 a changed --scene is reported", FR.rollout_argv_problems(nt3, bt) != [], "")
    check("L14 a different token count is reported", FR.rollout_argv_problems(bt + ["--x"], bt) != [], "")
    ra = {"ADE6": {"delta": -0.5, "lo": -0.6, "hi": -0.4, "separated": True, "n_windows": 2399, "mean_a": 2.0, "mean_b": 2.5}, "_M2": {"M2": 0.1}}
    rb = copy.deepcopy(ra)
    check("L15 rows_diff: equal rows -> no difference, max 0.0; a metadata label is skipped", FR.rows_diff(ra, rb) == ([], 0.0), FR.rows_diff(ra, rb))
    rb["ADE6"]["delta"] = -0.5 + 1e-6
    check("L15 rows_diff: a 1e-6 difference is reported", FR.rows_diff(ra, rb)[0] != [], "")
    rb = copy.deepcopy(ra); rb["ADE6"]["separated"] = False
    check("L15 rows_diff: a flipped separated flag is reported", FR.rows_diff(ra, rb)[0] != [], "")
    rb = copy.deepcopy(ra); rb["ADE6"]["n_windows"] = 2398
    check("L15 rows_diff: a different n is reported", FR.rows_diff(ra, rb)[0] != [], "")
    rb = copy.deepcopy(ra); del rb["ADE6"]
    check("L15 rows_diff: a row on one side only is reported", FR.rows_diff(ra, rb)[0] != [], "")
    fa = {"longitudinal": {"speed_mae_mps": 0.9, "target_speed_acc": {"within_0.5_mps": 0.5}, "note": "x", "ok": True}, "lateral": {"a": 1.0}}
    fb = {"longitudinal": {"speed_mae_mps": 1.0, "target_speed_acc": {"within_0.5_mps": 0.4}}, "strategic": {"b": 2.0}}
    fd = FR.families_diff(fa, fb)
    check("L16 families_diff: oacc - dag leaf by leaf, strings and booleans skipped, one-sided leaves listed", abs(fd["leaves"]["longitudinal.speed_mae_mps"]["diff"] - (-0.1)) < 1e-12 and fd["only_in_oacc"] == ["lateral.a"]
          and fd["only_in_dag"] == ["strategic.b"] and "longitudinal.note" not in fd["leaves"] and "longitudinal.ok" not in fd["leaves"], fd)
    check("L16 guards_beside: R-T0 point == F/2 holds (inclusive); just above fails", FR.guards_beside(0.0604)["R-T0_point_le_F_over_2"] and not FR.guards_beside(0.0605)["R-T0_point_le_F_over_2"], "")
    check("L17 family_of routes controls to C-O1..C-O4", (FR.family_of("C-O3"), FR.family_of("C-O2-GATE:DAGOACC_s0"), FR.family_of("C-O1-ONELEVER:DAGOACC_s0"), FR.family_of("RC-DAG:s0"), FR.family_of("C2-IDENTITY:x"), FR.family_of("T1-K:K1:x"),
                                                         FR.family_of("SUMMARY:DAGOACC_s0"), FR.family_of("LON-ROWS-COMPUTED:s0")) == ("C-O3", "C-O2", "C-O1", "C-O4", "C-O4", "C-O4", "C-O1", "C-O4"), "")
    import re
    bsrc = open(os.path.join(HERE, "readA_code", "build_report.py"), encoding="utf-8").read()
    spec = ast.literal_eval(re.search(r"SPEC = (\[.*?\])\n\n\n", bsrc, re.S).group(1))
    check("L18 READ_A_ROWS (family, label, direction) equal build_report.SPEC (PREREG_V8F_DAG_REREAD section 2)", tuple((f, l, d) for f, l, _, _, d in spec) == FR.READ_A_ROWS, "")
    # --- d_flags / lon_status / latcrawl_worse on the REAL banked Read A tables (DAG - P10), against the banked verdict
    ba = jl(f"{ART}/dagreread/raw/READ_A.json")["READ_A"]
    for s in (0, 1):
        rows, ver = ba["rows"][f"s{s}"], ba["per_seed_verdict"][f"s{s}"]
        d = FR.d_flags(rows)
        check(f"L19 d_flags on the banked DAG - P10 table s{s} reproduces the banked D1 / D2 / D3 and the separated-worse rows", (d["D1"], d["D2"], d["D3"]) == (ver["D1"], ver["D2"], ver["D3"]) and d["separated_worse_rows"] == sorted(ver["separated_worse_rows"]),
              (d["D1"], d["D2"], d["D3"], ver["D1"], ver["D2"], ver["D3"], d["separated_worse_rows"], ver["separated_worse_rows"]))
        lon = FR.lon_status(rows)
        banked_lon = sorted(l for l in FR.LON_ROWS if rows[l].get("separated_worse"))
        check(f"L19 lon_status on the banked DAG - P10 table s{s} names the banked separated-worse longitudinal rows", sorted(lon["worse"]) == banked_lon and lon["clear"] == (not banked_lon), (lon, banked_lon))
        lc = FR.latcrawl_worse(rows)
        banked_lc = sorted(l for l in FR.LAT_ROWS + (FR.M1_LABEL,) if rows[l].get("separated_worse"))
        check(f"L19 latcrawl_worse on the banked DAG - P10 table s{s} names the banked lateral / crawl rows", sorted(lc) == banked_lc, (lc, banked_lc))
    # --- the build/seed-flag glue
    f, ex = FR.seed_flags_of(rowd(-0.3, True), rowd(0.0, False, hi=0.05), t0)
    check("L20 seed_flags_of composes the rows into the seed's flags", f == {"PA": True, "KEEP": True, "D1": True, "D2": True, "D3": True, "LON_CLEAR": True, "LATCRAWL_WORSE": False}, f)
    f2, _ = FR.seed_flags_of(rowd(-0.3, True), rowd(0.0, False, hi=0.07), clean_table({"yaw-rate MAE": rowd(0.1, True)}))
    check("L20 seed_flags_of: KEEP from the upper bound, lateral worse -> D2 False and LATCRAWL_WORSE True", f2["KEEP"] is False and f2["D2"] is False and f2["LATCRAWL_WORSE"] is True and f2["LON_CLEAR"] is True, f2)


BANK_OTHER = "ab" * 32


# ================================================================================================================ P: pull check
def run_pull(D):
    PC = load("pc_under_test", os.path.join(D, "dagoacc_t1_pullcheck.py"))
    tmp = tempfile.mkdtemp(prefix="dopull_", dir=SCRATCH)
    dag = {s: jl(f"{DAGART}/M2b_s{s}_dag/argv.json")["argv"] for s in (0, 1)}

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    def mkrun(art, seed, wacc=True, **kw):
        d = os.path.join(art, PC.run_name("dagoacc", seed))
        os.makedirs(os.path.join(d, "gate"), exist_ok=True)
        argv = [("D:\\tanitad\\runs\\v8f_mini_full\\" + kw.get("out", PC.run_name("dagoacc", seed)) if (i and dag[seed][i - 1] == "--out") else t) for i, t in enumerate(dag[seed])]
        if kw.get("seed_arg") is not None:
            argv[argv.index("--seed") + 1] = str(kw["seed_arg"])
        if kw.get("init"):
            argv[argv.index("--init-ckpt") + 1] = kw["init"]
        if kw.get("p") is not None:
            argv[argv.index("--dagger-p") + 1] = str(kw["p"])
        if kw.get("bank"):
            argv[argv.index("--dagger-bank") + 1] = kw["bank"]
        if kw.get("bank_sha"):
            argv[argv.index("--dagger-bank-sha256") + 1] = kw["bank_sha"]
        if kw.get("lr"):
            argv[argv.index("--lr") + 1] = kw["lr"]
        if kw.get("drop_bank_flags"):
            k = argv.index("--dagger-seed"); del argv[k:k + 2]
        argv += kw.get("w_tokens", ["--w-acc", "49.6969"])
        argv += kw.get("extra", [])
        files = {"ckpt_last.pt": b"ckpt" + bytes([seed]) + kw.get("ck", b"dagoacc"), f"eval_{PC.eval_tag('dagoacc', seed)}.json": b"{}", f"eval_{PC.eval_tag('dagoacc', seed)}.npz": b"npz",
                 "selfcheck.json": b"{}", "train_log.jsonl": b"{}\n", "launch_gate_token.json": b"{}",
                 "summary.json": json.dumps({"done": kw.get("done", True), "arm": "M2b", "seed": kw.get("sm_seed", seed), "levers": {"dagger_p": 0.1, "init_ckpt": {"sha256": kw.get("init_sha", PC.M2B_CKPT_SHA[seed])},
                                                                                                                                    "dagger_bank_sha256": kw.get("lv_bank", PC.BANK1_SHA[seed]), "w_acc": 49.6969}}).encode(),
                 "argv.json": json.dumps({"argv": argv}).encode()}
        for n, b in files.items():
            open(os.path.join(d, n), "wb").write(b)
        names = kw.get("sums_names") or PC.base_files("dagoacc", seed)
        if kw.get("fresh"):
            fr = [f"eval_{PC.eval_tag('dagoacc', seed)}-fresh.json", f"eval_{PC.eval_tag('dagoacc', seed)}-fresh.npz"]
            names = names + fr
            for n in fr:
                open(os.path.join(d, n), "wb").write(b"fresh")
        lines = "".join(f"{sha(open(os.path.join(d, n), 'rb').read()) if os.path.exists(os.path.join(d, n)) else '0' * 64} *{n}\n" for n in names)
        if kw.get("bad_sum"):
            lines = lines.replace(lines[:8], "deadbeef", 1)
        open(os.path.join(d, "SHA256SUMS.txt"), "w", newline="\n").write(lines)
        if wacc:
            os.makedirs(os.path.join(art, "wacc"), exist_ok=True)
            open(os.path.join(art, "wacc", "w_acc.json"), "w").write(json.dumps({"w_acc": kw.get("wjson", 49.6969)}))
            open(os.path.join(art, "wacc", "w_acc.txt"), "w").write(str(kw.get("wtxt", 49.6969)) + "\n")
        return d, os.path.getmtime(os.path.join(d, "SHA256SUMS.txt"))

    def st(art, seed, now=None, dag_art=DAGART):
        return PC.check(art, "dagoacc", seed, min_age_s=3.0, now=now, dag_art=dag_art)[0]

    def ck(name, kw, want, seed=0, wacc=True):
        art = os.path.join(tmp, f"X{len(RES)}")
        d, t = mkrun(art, seed, wacc=wacc, **kw)
        got = PC.check(art, "dagoacc", seed, min_age_s=3.0, now=t + 10, dag_art=DAGART)
        check(name, got[0] == want, got[:2])
    art = os.path.join(tmp, "A")
    d, t = mkrun(art, 0)
    check("P1 a synthetic complete DAG-OACC s0 pull (DAG's argv + --w-acc 49.6969) is COMPLETE", st(art, 0, now=t + 10) == "COMPLETE", PC.check(art, "dagoacc", 0, now=t + 10, dag_art=DAGART)[1])
    art_b, tb = os.path.join(tmp, "A1"), None
    d1, tb = mkrun(art_b, 1)
    check("P1 the same at seed 1", st(art_b, 1, now=tb + 10) == "COMPLETE", PC.check(art_b, "dagoacc", 1, now=tb + 10, dag_art=DAGART)[1])
    check("P1 the sums are the 8 names, no oracle eval", len(PC.base_files("dagoacc", 0)) == 8 and not any("tacorc" in n for n in PC.base_files("dagoacc", 0)) and "eval_M2b-dagoacc_s0.json" in PC.base_files("dagoacc", 0), PC.base_files("dagoacc", 0))
    check("P2 a SHA256SUMS.txt younger than 3 s is INCOMPLETE (may still be filling)", st(art, 0, now=t + 1) == "INCOMPLETE", st(art, 0, now=t + 1))
    ck("P3 7 of the 8 names listed -> INCOMPLETE", {"sums_names": PC.base_files("dagoacc", 0)[:-1]}, "INCOMPLETE")
    ck("P3 the 9-name FLAT-DAG form (a tacorc name) -> INCOMPLETE (not the exact set)", {"sums_names": PC.base_files("dagoacc", 0) + ["eval_M2b-dagoacc_s0-tacorc.npz"]}, "INCOMPLETE")
    art3 = os.path.join(tmp, "C"); d3, t3 = mkrun(art3, 0); os.remove(os.path.join(d3, "eval_M2b-dagoacc_s0.npz"))
    check("P4 a listed file absent -> INCOMPLETE", st(art3, 0, now=t3 + 10) == "INCOMPLETE", st(art3, 0, now=t3 + 10))
    art4 = os.path.join(tmp, "D"); d4, t4 = mkrun(art4, 0); open(os.path.join(d4, "ckpt_last.pt"), "wb").write(b"half")
    check("P5 a file whose sha256 differs from the sums -> INCOMPLETE", st(art4, 0, now=t4 + 10) == "INCOMPLETE", st(art4, 0, now=t4 + 10))
    ck("P6 summary.json done false -> INCOMPLETE", {"done": False}, "INCOMPLETE")
    ck("P7 a box-2 fresh-age eval in the sums -> WRONG (never waited for)", {"fresh": True}, "WRONG")
    ck("P8 argv --dagger-p 0 (a DAG-0 run) -> WRONG", {"p": 0.0}, "WRONG")
    ck("P9 a dagger flag dropped (--dagger-seed) -> WRONG", {"drop_bank_flags": True}, "WRONG")
    ck("P10 the FLAT-DAG bank (dagger_bank_flat) -> WRONG", {"bank": "D:\\tanitad\\runs\\v8f_mini_full\\dagger_bank_flat\\s0"}, "WRONG")
    ck("P10 an iteration-2 bank (dagger_bank2) -> WRONG", {"bank": "D:\\tanitad\\runs\\v8f_mini_full\\dagger_bank2\\s0"}, "WRONG")
    ck("P11 a bank sha256 other than DAG's iteration-1 one (C-O1) -> WRONG", {"bank_sha": "ab" * 32}, "WRONG")
    ck("P11 summary.json built dagger_bank_sha256 other than DAG's -> WRONG", {"lv_bank": "cd" * 32}, "WRONG")
    ck("P12 init-ckpt = DAG_s0 (iteration 2 style) -> WRONG", {"init": "D:\\tanitad\\runs\\v8f_mini_full\\M2b_s0_dag\\ckpt_last.pt"}, "WRONG")
    ck("P12 init-ckpt = FLAT_s0 -> WRONG", {"init": "D:\\tanitad\\runs\\v8f_mini_full\\M2b_s0_flat\\ckpt_last.pt"}, "WRONG")
    ck("P13 summary.json levers.init_ckpt.sha256 not M2b_s0's -> WRONG", {"init_sha": "00" * 32}, "WRONG")
    ck("P14 --out naming another run -> WRONG", {"out": "M2b_s0_dag"}, "WRONG")
    ck("P15 argv --seed 0 in a seed-1 directory -> WRONG", {"seed_arg": 0}, "WRONG", seed=1)
    ck("P15 summary.json seed 1 in a seed-0 directory -> WRONG", {"sm_seed": 1}, "WRONG")
    ck("P16 NO --w-acc in the argv (a DAG run in a dagoacc directory) -> WRONG (C-O1)", {"w_tokens": []}, "WRONG")
    ck("P16 --w-acc twice -> WRONG", {"w_tokens": ["--w-acc", "49.6969", "--w-acc", "49.6969"]}, "WRONG")
    ck("P16 --w-acc other than the w_acc stage's value (OACC's M2b 42.9311) -> WRONG", {"w_tokens": ["--w-acc", "42.9311"]}, "WRONG")
    ck("P16 a second lever (--w-tac 1.0) beside --w-acc -> WRONG (C-O1 one lever)", {"extra": ["--w-tac", "1.0"]}, "WRONG")
    ck("P16 a changed shared flag (--lr 2e-4) -> WRONG (C-O1 one lever)", {"lr": "2e-4"}, "WRONG")
    art5 = os.path.join(tmp, "E5"); d5, t5 = mkrun(art5, 0)
    ck_sha = sha(open(os.path.join(d5, "ckpt_last.pt"), "rb").read())
    check("P17 control: a checkpoint that is not DAG_s0's is COMPLETE", PC.check(art5, "dagoacc", 0, min_age_s=3.0, now=t5 + 10, dag_art=DAGART)[0] == "COMPLETE", "")
    keep = dict(PC.DAG_CKPT_SHA)
    PC.DAG_CKPT_SHA[0] = ck_sha                                                                          # the pulled checkpoint now IS "DAG_s0's"
    got17 = PC.check(art5, "dagoacc", 0, min_age_s=3.0, now=t5 + 10, dag_art=DAGART)
    PC.DAG_CKPT_SHA.update(keep)
    check("P17 the pulled checkpoint IS DAG_s0's checkpoint (nothing was fine-tuned with the lever) -> WRONG", got17[0] == "WRONG" and "nothing was fine-tuned" in got17[1], got17[:2])
    check("P18 the w_acc stage record not pulled -> INCOMPLETE", st(os.path.join(tmp, "E6"), 0) == "INCOMPLETE" and PC.check(os.path.join(tmp, "E6"), "dagoacc", 0)[0] == "INCOMPLETE", "")
    art7 = os.path.join(tmp, "E7"); d7, t7 = mkrun(art7, 0, wacc=False)
    check("P18 a run pulled before the w_acc stage record -> INCOMPLETE (the lever's value is unknown)", PC.check(art7, "dagoacc", 0, now=t7 + 10, dag_art=DAGART)[0] == "INCOMPLETE", PC.check(art7, "dagoacc", 0, now=t7 + 10, dag_art=DAGART)[:2])
    ck("P19 w_acc.json and w_acc.txt that disagree -> WRONG", {"wtxt": 42.9311}, "WRONG")
    check("P20 an unknown arm or an absent directory -> INCOMPLETE", PC.check(art, "dag2", 0)[0] == "INCOMPLETE" and st(art, 1) == "INCOMPLETE", "")
    check("P21 DAG's banked argv missing -> INCOMPLETE (C-O1 has no control)", PC.check(art, "dagoacc", 0, now=t + 10, dag_art=os.path.join(tmp, "nodag"))[0] == "INCOMPLETE", "")
    # identity_problems on its own (DAG's banked argv withheld): each clause is exercised by its OWN message, not by the one-lever comparison that also catches most of them
    def idp(kw, seed=0):
        a_ = os.path.join(tmp, f"I{len(RES)}")
        d_, _ = mkrun(a_, seed, **kw)
        return PC.identity_problems("dagoacc", seed, d_, 49.6969, None)
    ONLY = "DAG's banked argv could not be read"
    base_msgs = idp({})
    check("P22a identity_problems: a faithful DAG-OACC record gives only the 'DAG argv withheld' note (every identity clause passes)", len(base_msgs) == 1 and ONLY in base_msgs[0], base_msgs)
    for nm, kw, frag, sd in (("--dagger-p 0", {"p": 0.0}, "--dagger-p", 0), ("a wrong bank directory", {"bank": "D:\\tanitad\\runs\\v8f_mini_full\\dagger_bank_flat\\s0"}, "ITERATION-1 bank", 0),
                             ("a wrong bank sha256 in the argv", {"bank_sha": "ab" * 32}, "the bank sha256 is identical", 0), ("--seed of the other seed", {"seed_arg": 0}, "argv --seed", 1),
                             ("summary.json of another seed", {"sm_seed": 1}, "summary.json says arm", 0), ("init = DAG_s0", {"init": "D:\\x\\M2b_s0_dag\\ckpt_last.pt"}, "--init-ckpt", 0),
                             ("--out of another run", {"out": "M2b_s0_dag"}, "does not end in", 0), ("a dropped dagger flag", {"drop_bank_flags": True}, "does not name", 0),
                             ("summary.json built another bank sha256", {"lv_bank": "cd" * 32}, "summary.json built dagger_bank_sha256", 0),
                             ("summary.json init sha256 not M2b_s0's", {"init_sha": "00" * 32}, "levers.init_ckpt.sha256", 0)):
        m_ = idp(kw, sd)
        check(f"P22a identity_problems: {nm} is reported by its own clause ({frag!r})", any(frag in x for x in m_), m_)
    # DAG's banked bank with a consistent forgery (bank.json, SHA256SUMS.txt and PROVENANCE.json all moved together): only the pin against DAG's declared sha256 catches it
    fk = os.path.join(tmp, "FB2")
    shutil.copytree(f"{DAGART}/bank_s0", os.path.join(fk, "bank_s0"), ignore=shutil.ignore_patterns("*.npz"))
    old = PC.sha256_file(os.path.join(fk, "bank_s0", "bank.json"))
    open(os.path.join(fk, "bank_s0", "bank.json"), "a").write(" ")
    new = PC.sha256_file(os.path.join(fk, "bank_s0", "bank.json"))
    sp = os.path.join(fk, "bank_s0", "SHA256SUMS.txt")
    open(sp, "w", newline="\n").write(open(sp, encoding="utf-8").read().replace(old, new))
    pp = os.path.join(fk, "bank_s0", "PROVENANCE.json")
    pr = jl(pp); pr["bank_sha256"] = new; json.dump(pr, open(pp, "w"))
    check("P24b a bank.json forged consistently (SUMS and PROVENANCE moved with it) is still WRONG: it no longer hashes to the sha256 DAG's argv declared", PC.dagbank_status(fk, 0)[0] == "WRONG" and "DAG's argv declared" in PC.dagbank_status(fk, 0)[1], PC.dagbank_status(fk, 0))
    # the w_acc stage record, DAG's banked banks
    check("P22 wacc_status: OK with the value on a consistent record", PC.wacc_status(art) == ("OK", 49.6969), PC.wacc_status(art))
    bad_art = os.path.join(tmp, "W1"); os.makedirs(os.path.join(bad_art, "wacc")); open(os.path.join(bad_art, "wacc", "w_acc.json"), "w").write('{"w_acc": -3}'); open(os.path.join(bad_art, "wacc", "w_acc.txt"), "w").write("-3\n")
    check("P22 wacc_status: a non-positive w_acc is WRONG; absent files are INCOMPLETE", PC.wacc_status(bad_art)[0] == "WRONG" and PC.wacc_status(os.path.join(tmp, "nowhere"))[0] == "INCOMPLETE", "")
    check("P23 the REAL w_acc stage record is OK, value 49.6969", PC.wacc_status(OACCART) == ("OK", 49.6969), PC.wacc_status(OACCART))
    check("P24 DAG's REAL iteration-1 banks s0 and s1 are COMPLETE (bank.json sha256 == the sha256 DAG's argv declared)", [PC.dagbank_status(DAGART, s)[0] for s in (0, 1)] == ["COMPLETE", "COMPLETE"], [PC.dagbank_status(DAGART, s) for s in (0, 1)])
    fake = os.path.join(tmp, "FB"); shutil.copytree(f"{DAGART}/bank_s0", os.path.join(fake, "bank_s0"), ignore=shutil.ignore_patterns("*.npz"))
    pj = os.path.join(fake, "bank_s0", "bank.json"); open(pj, "a").write(" ")
    check("P24 a bank.json that no longer hashes to DAG's declared sha256 is WRONG", PC.dagbank_status(fake, 0)[0] == "WRONG", PC.dagbank_status(fake, 0))
    check("P24 a missing bank is INCOMPLETE", PC.dagbank_status(os.path.join(tmp, "nowhere"), 0)[0] == "INCOMPLETE", "")
    FR = load("fr_for_pull", os.path.join(D, "dagoacc_read.py"))
    variants = [dag[0] + ["--w-acc", "49.6969"], dag[0], dag[0] + ["--w-acc", "49.6969", "--w-acc", "49.6969"], dag[0] + ["--w-acc", "42.9311"], dag[0] + ["--w-acc", "49.6969", "--w-tac", "1"],
                dag[0][:-2] + ["--w-acc", "49.6969"], dag[0] + ["--w-acc", "abc"], dag[0] + ["--w-acc", "0"], dag[0] + ["--w-acc"]]
    agree = [bool(PC.one_lever_problems(v, dag[0], 49.6969)) == bool(FR.one_lever_problems(v, dag[0], 49.6969)) for v in variants]
    check("P25 one_lever_problems: the pull check and the reader (which must not trust the pull check) agree on the verdict for nine argv variants, and they are not all the same verdict", all(agree) and len({bool(PC.one_lever_problems(v, dag[0], 49.6969)) for v in variants}) == 2, agree)
    shutil.rmtree(tmp, ignore_errors=True)


# ================================================================================================================ C: chain dry-run scenarios
def syn_art():
    """A SYNTHETIC pull directory for the dry-run scenarios: only the w_acc stage record (a bookkeeping file) is copied, so the chain's pull check finds INCOMPLETE pulls and never opens or hashes a DAG-OACC checkpoint or eval file."""
    d = f"{SCRATCH}/syn_art"
    if not os.path.isdir(f"{d}/wacc"):
        os.makedirs(f"{d}/wacc", exist_ok=True)
        for n in ("w_acc.json", "w_acc.txt"):
            shutil.copyfile(f"{OACCART}/wacc/{n}", f"{d}/wacc/{n}")
    return d


def chain(D, env=None):
    e = {**os.environ, "DOT1_TEST_CFREE_MB": "3000", "DOT1_TEST_EFREE_MB": "3000", "DOT1_B2_OUT": "off", "DOT1_ART": syn_art()}
    for k, v in (env or {}).items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = v
    r = subprocess.run([BASH, "-lc", f"bash '{D.replace(chr(92), '/')}/dagoacc_t1_chain.sh' --dry-run < /dev/null"], capture_output=True, text=True, env=e, timeout=600)
    return r.returncode, r.stdout + r.stderr


def run_chain(D, which=None):
    S = which or ("base", "cgate", "egate", "prereg", "offline", "reread", "tree", "cfg")
    if "base" in S:
        rc, o = chain(D)
        check("C1 baseline dry-run ends ZZDAGOACCT1-DRY-OK, rc 0", rc == 0 and "ZZDAGOACCT1-DRY-OK" in o, o[-300:])
        cmds = [ln for ln in o.splitlines() if "[lane s" in ln]
        check("C2 two rollout commands: DAG-OACC s0 (lane s0), DAG-OACC s1 (lane s1), tags never DAG_s..", len(cmds) == 2 and all(f"--tag DAGOACC_s{s}_m10_causal_heads" in c for c, s in zip(cmds, (0, 1))) and "[lane s0]" in cmds[0] and "[lane s1]" in cmds[1], cmds)
        check("C3 every rollout is --limit causal --dump-heads --dump-ticks, m=10, deadreckon, logged scene, periodic, on CPU, on the pulled checkpoint",
              all(x in c for c in cmds for x in ("--limit causal", "--dump-heads", "--dump-ticks", "--replan-steps 10", "--nav deadreckon", "--scene logged", "--schedule periodic", "--device cpu")) and
              all(f"--ckpt {syn_art()}/M2b_s{s}_dagoacc/ckpt_last.pt" in c for c, s in zip(cmds, (0, 1))), cmds[:1])
        check("C4 no logged-bucket rollout and no fresh-T0 command is planned", "--limit logged" not in o.split("== rollouts")[1].split("== NOT run here")[0] and "--ages fresh" not in o, "")
        check("C1b the scenarios run on a SYNTHETIC pull directory: both pulls INCOMPLETE and the w_acc record OK, so no DAG-OACC checkpoint or eval file is opened or hashed",
              "M2b_s0_dagoacc  INCOMPLETE" in o and "M2b_s1_dagoacc  INCOMPLETE" in o and "w_acc record  OK 49.6969" in o, o[:300])
        check("C5 the pulls and the w_acc record are reported by name", all(f"M2b_s{s}_dagoacc" in o for s in (0, 1)) and "w_acc record" in o, "")
        check("C6 the reader command carries the OACC PREREG, the offline reference and the M2b init slots, and writes only under v8f-dagoacc-t1", "--prereg-oacc" in o and "--offline-wacc" in o and "--init-sha" in o and "v8f-dagoacc-t1/read/DAGOACC_READ.json" in o, "")
        check("C7 the dry run says the box-2 DONE gate and the w_acc / banked-bank re-verification come after the lanes", "box-2 ZZV8FMINI-DONE" in o and "w_acc record + DAG's two banked banks" in o, "")
        check("C7 DAG's banks and banked outputs are verified in the dry run, and no bank wait exists", "iteration-1 bank_s0" in o and "iteration-1 bank_s1" in o and "DAG's banked outputs present and pinned" in o, "")
        rcb, ob = chain(D, {"DOT1_B2_OUT": None})
        check("C8 the default box-2 watcher is the box-2 chain's own log chain_v8f_dagoacc_r2.out (read-only)", rcb == 0 and "box-2 watcher: C:/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out" in ob, ob[-200:])
    if "cgate" in S:
        rc1, o1 = chain(D, {"DOT1_TEST_CFREE_MB": "2047"})
        rc2, o2 = chain(D, {"DOT1_TEST_CFREE_MB": "2048"})
        check("C9 C: gate: 2047 MB free STOPs (rc 1, ZZDAGOACCT1-STOP)", rc1 == 1 and "ZZDAGOACCT1-STOP" in o1 and "C: has 2047 MB" in o1, o1[-200:])
        check("C9 C: gate: exactly 2048 MB passes", rc2 == 0 and "ZZDAGOACCT1-DRY-OK" in o2, o2[-200:])
    if "egate" in S:
        rc1, o1 = chain(D, {"DOT1_TEST_EFREE_MB": "1499"})
        rc2, o2 = chain(D, {"DOT1_TEST_EFREE_MB": "1500"})
        check("C10 E: gate: 1499 MB free STOPs", rc1 == 1 and "E: has 1499 MB" in o1, o1[-200:])
        check("C10 E: gate: exactly 1500 MB passes", rc2 == 0 and "ZZDAGOACCT1-DRY-OK" in o2, o2[-200:])
    if "prereg" in S:
        t = open(os.path.join(HERE, "inputs", "PREREG_V8F_DAG_OACC.md"), "rb").read().replace(b"0.0604", b"0.0605", 1)
        p = os.path.join(SCRATCH, "PREREG_tampered.md")
        open(p, "wb").write(t)
        rc, o = chain(D, {"DOT1_PREREGO": p})
        check("C11 a tampered PREREG (one number changed) STOPs on its pin", rc == 1 and "ZZDAGOACCT1-STOP" in o and "not the pinned text" in o, o[-300:])
    if "offline" in S:
        wd = os.path.join(SCRATCH, "offline_copy")
        shutil.rmtree(wd, ignore_errors=True)
        os.makedirs(wd)
        for f in os.listdir(D):
            if os.path.isfile(os.path.join(D, f)) and f.endswith((".py", ".sh", ".cmd")):
                shutil.copyfile(os.path.join(D, f), os.path.join(wd, f))
        for sub in ("inputs", "readA_code"):
            shutil.copytree(os.path.join(D, sub), os.path.join(wd, sub))
        p = os.path.join(wd, "inputs", "wacc_offline_DAG_s0.json")
        open(p, "wb").write(open(p, "rb").read().replace(b"49.6969", b"49.6970", 1))
        rc, o = chain(wd)
        check("C11 a tampered offline w_acc reference (C-O3's input) STOPs on the CHAIN's own pin (not only later in the reader)", rc == 1 and "ZZDAGOACCT1-STOP" in o and "is not the pinned wacc_offline_DAG_s0.json" in o, o[-300:])
        shutil.rmtree(wd, ignore_errors=True)
    if "reread" in S:
        wd = os.path.join(SCRATCH, "reread_copy")
        shutil.rmtree(wd, ignore_errors=True)
        os.makedirs(wd)
        for f in os.listdir(D):
            if os.path.isfile(os.path.join(D, f)) and f.endswith((".py", ".sh", ".cmd")):
                shutil.copyfile(os.path.join(D, f), os.path.join(wd, f))
        for sub in ("inputs", "readA_code"):
            shutil.copytree(os.path.join(D, sub), os.path.join(wd, sub))
        p = os.path.join(wd, "inputs", "PREREG_V8F_DAG_REREAD.md")
        open(p, "wb").write(open(p, "rb").read().replace(b"D1", b"D9", 1))
        rc, o = chain(wd)
        check("C11 a tampered PREREG_V8F_DAG_REREAD (Read A's rows and C1-C4) STOPs on the chain's own pin", rc == 1 and "ZZDAGOACCT1-STOP" in o and "PREREG_V8F_DAG_REREAD.md" in o and "is not the pinned text" in o, o[-300:])
        shutil.rmtree(wd, ignore_errors=True)
    if "tree" in S:
        rc, o = chain(D, {"DOT1_TREE": "E:/Projects/TanitAD-artifacts/no_such_tree"})
        check("C12 a missing tool tree STOPs", rc == 1 and "ZZDAGOACCT1-STOP" in o, o[-200:])
    if "cfg" in S:
        src = os.path.join(ART, "dagt1_tree_b43c7a9a")
        res = {}
        for nm in ("control", "no_cfg", "no_wacc"):
            mt = os.path.join(SCRATCH, "mintree_" + nm)
            shutil.rmtree(mt, ignore_errors=True)
            for f in ("stack/scripts/v8f_t1_eval.py", "stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_dagger.py", "stack/tanitad/models/v8f_mini.py"):
                os.makedirs(os.path.dirname(os.path.join(mt, f)), exist_ok=True)
                shutil.copyfile(os.path.join(src, f), os.path.join(mt, f))
            os.makedirs(os.path.join(mt, "taniteval"), exist_ok=True)
            if nm == "no_cfg":
                tp = os.path.join(mt, "stack/scripts/v8f_t1_eval.py")
                tx = open(tp, encoding="utf-8", newline="").read()
                assert tx.count('V8FMiniConfig(**ck["cfg"])') >= 1
                open(tp, "w", encoding="utf-8", newline="").write(tx.replace('V8FMiniConfig(**ck["cfg"])', 'V8FMiniConfig(**ck["cfg_x"])'))
            if nm == "no_wacc":
                tp = os.path.join(mt, "stack/tanitad/models/v8f_mini.py")
                tx = open(tp, encoding="utf-8", newline="").read()
                assert tx.count("    w_acc: float") == 1
                open(tp, "w", encoding="utf-8", newline="").write(tx.replace("    w_acc: float", "    w_accX: float"))
            res[nm] = chain(D, {"DOT1_TREE": mt.replace(chr(92), "/")})
            shutil.rmtree(mt, ignore_errors=True)
        check("C14 control: a four-file tree whose tool builds from ck[cfg] and whose config declares w_acc passes both guards (and stops later, at --help, for lack of imports)",
              res["control"][0] == 1 and "does not build the model" not in res["control"][1] and "declares no w_acc" not in res["control"][1] and "--help did not run" in res["control"][1], res["control"][1][-250:])
        check("C14 a tool that does NOT build the model from the checkpoint's cfg STOPs on that guard", res["no_cfg"][0] == 1 and "does not build the model from the checkpoint's cfg" in res["no_cfg"][1], res["no_cfg"][1][-250:])
        check("C15 a config class that declares no w_acc (a DAG-OACC checkpoint's cfg would not load) STOPs on that guard", res["no_wacc"][0] == 1 and "declares no w_acc" in res["no_wacc"][1], res["no_wacc"][1][-250:])
    if "syntax" in S or which is None:
        r = subprocess.run([BASH, "-n", os.path.join(D, "dagoacc_t1_chain.sh")], capture_output=True, text=True)
        check("C13 bash -n dagoacc_t1_chain.sh (the real-run branch is at least syntactically valid)", r.returncode == 0, r.stderr[:200])


# ================================================================================================================ H: rehearsal on BANKED data
GOLDEN_REH = "9111eca96f598095e7ef7745ed09de8175fb3d4901c8dd7899961160337a207c"      # sha256 of the canonical json of {OUTCOME, rows, t0_beside, controls_PASS} of the DAG-0-vs-DAG rehearsal (deterministic: two runs, same hash)


def run_rehearsal(D):
    """The reader's --rehearsal on BANKED data (DAG-0 stands in for DAG-OACC against DAG; no DAG-OACC file is opened): the whole Read-A / beside / T0 row pipeline and decide() end to end, against a pinned characterisation hash."""
    TT = f"{ART}/dagt1_tree_b43c7a9a"
    PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
    SF = f"{ART}/sfresh_inputs"
    out = tempfile.mkdtemp(prefix="reh_", dir=SCRATCH)
    cmd = [PY, "-I", os.path.join(D, "dagoacc_read.py"), "--rehearsal", "--out", f"{out}/REHEARSAL.json", "--tool-root", TT, "--t1hd-read", f"{TT}/{PK}/raw/2026-10-26-t1hdread/t1hd_read.py",
           "--t1k-read", f"{TT}/{PK}/raw/2026-10-26-t1kin-hacc/t1k_read.py", "--eval-cache", f"{SF}/cache", "--tracks", f"{SF}/inputs/native_tracks.pt", "--eval-grid", f"{TT}/{PK}/raw/eval_grid_windows.json",
           "--eval-join", f"{TT}/{PK}/raw/join_eval141.npz"]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"}, timeout=900)
    lines = r.stdout.splitlines()
    npass, nfail = sum(1 for ln in lines if ln.startswith("PASS ")), sum(1 for ln in lines if ln.startswith("FAIL "))
    check("H1 the rehearsal on banked data runs to the end, rc 0, no traceback", r.returncode == 0 and "Traceback" not in r.stderr, (r.returncode, r.stderr[-300:]))
    check("H2 every rehearsal control passes: 17 PASS, 0 FAIL (SPEC-LABELS, C4, C3, C2, RC-DAG (C1), LON-ROWS-COMPUTED, READA-VERDICT-LINK, T0-CKPT, R-T0-GRID, per seed)", npass == 17 and nfail == 0, (npass, nfail))
    jp = f"{out}/REHEARSAL.json"
    if not os.path.isfile(jp):
        check("H3 the rehearsal wrote its json", False, "no file")
        shutil.rmtree(out, ignore_errors=True)
        return
    j = jl(jp)
    check("H3 the output is labelled REHEARSAL and NOT a DAG-OACC result", j.get("REHEARSAL") is True and "NOT a DAG-OACC result" in j.get("note", ""), j.get("note"))
    canon = json.dumps({k: j.get(k) for k in ("OUTCOME", "rows", "t0_beside", "controls_PASS")}, sort_keys=True, default=str)
    check("H4 the rehearsal's outcome, Read-A tables, beside rows, T0 rows and controls equal the pinned characterisation (sha256)", hashlib.sha256(canon.encode()).hexdigest() == GOLDEN_REH, hashlib.sha256(canon.encode()).hexdigest())
    check("H5 the stand-in reads NO EFFECT (DAG-0 against DAG: PRIMARY-ACC cannot pass for a run without the lever)", j["OUTCOME"]["OUTCOME"] == "NO EFFECT", j["OUTCOME"]["OUTCOME"])
    t0 = j.get("t0_beside", {})
    check("H6 R-T0 and the four-family T0 leaves are computed for both seeds, and R-T0's guard is evaluated", sorted(t0) == ["s0", "s1"] and all("R-T0" in v and len(v["four_families_T0"]["leaves"]) > 100 and "R-T0_point_le_F_over_2" in v["guards"] for v in t0.values()), sorted(t0))
    shutil.rmtree(out, ignore_errors=True)


# ================================================================================================================ M: mutation proof
READ_MUTANTS = [
    ("R01 primary_acc: < 0 becomes <= 0", 'return bool(row["separated"] and row["delta"] < 0)', 'return bool(row["separated"] and row["delta"] <= 0)'),
    ("R02 primary_acc: separation dropped", 'return bool(row["separated"] and row["delta"] < 0)', 'return bool(row["delta"] < 0)'),
    ("R03 keep_ok: <= +F/2 becomes < +F/2", "and hi <= HALF_F)", "and hi < HALF_F)"),
    ("R04 keep_ok: reads the lower bound", "and hi <= HALF_F)", 'and row.get("lo", 1.0) <= HALF_F)'),
    ("R05 HALF_F 0.0604 -> 0.0605", "F_M, TWO_F, HALF_F = 0.1207, 0.2415, 0.0604", "F_M, TWO_F, HALF_F = 0.1207, 0.2415, 0.0605"),
    ("R06 TWO_F 0.2415 -> 0.2414", "F_M, TWO_F, HALF_F = 0.1207, 0.2415, 0.0604", "F_M, TWO_F, HALF_F = 0.1207, 0.2414, 0.0604"),
    ("R07 sep_worse: direction flipped", 'return bool(row["separated"] and ((row["delta"] > 0) if direction == "lower" else (row["delta"] < 0)))', 'return bool(row["separated"] and ((row["delta"] < 0) if direction == "lower" else (row["delta"] > 0)))'),
    ("R08 sep_worse: separation dropped", 'return bool(row["separated"] and ((row["delta"] > 0) if direction == "lower" else (row["delta"] < 0)))', 'return bool(((row["delta"] > 0) if direction == "lower" else (row["delta"] < 0)))'),
    ("R09 lon_status: NOT COMPUTED rows do not block LON-CLEAR", 'return {"clear": not worse and not notc,', 'return {"clear": not worse,'),
    ("R10 LON_ROWS: accel MAE dropped", 'LON_ROWS = tuple(l for f, l, _ in READ_A_ROWS if f == "LONGITUDINAL")', 'LON_ROWS = tuple(l for f, l, _ in READ_A_ROWS if f == "LONGITUDINAL" and l != "accel MAE")'),
    ("R11 latcrawl_worse: the crawl share M1 dropped", "for l in LAT_ROWS + (M1_LABEL,) if", "for l in LAT_ROWS if"),
    ("R12 d_flags: D1 bound exclusive", 'd1 = bool(sep_better(ade, "lower") and ade["delta"] <= -TWO_F)', 'd1 = bool(sep_better(ade, "lower") and ade["delta"] < -TWO_F)'),
    ("R13 d_flags: D1 separation not required", 'd1 = bool(sep_better(ade, "lower") and ade["delta"] <= -TWO_F)', 'd1 = bool(ade["delta"] <= -TWO_F)'),
    ("R14 d_flags: D2 counts the crawl share", 'rows = {l: r for l, r in table.items() if r.get("family") != "crawl"}', "rows = dict(table)"),
    ("R15 d_flags: a NOT COMPUTED row passes D2", '"D2": bool(not worse and not notc)', '"D2": bool(not worse)'),
    ("R16 d_flags: D3 needs only one family", '"D3": all(len(v) >= 1 for v in fams.values())', '"D3": any(len(v) >= 1 for v in fams.values())'),
    ("R17 decide: NO EFFECT needs both seeds to fail", '"NO EFFECT": not pa,', '"NO EFFECT": not any(sf[s]["PA"] for s in SEEDS),'),
    ("R18 decide: COSTS POSITION needs both seeds to fail KEEP", '"COSTS POSITION": pa and not kp,', '"COSTS POSITION": pa and not any(sf[s]["KEEP"] for s in SEEDS),'),
    ("R19 decide: DOMINANT ignores D3", 'd123 = all(sf[s]["D1"] and sf[s]["D2"] and sf[s]["D3"] for s in SEEDS)', 'd123 = all(sf[s]["D1"] and sf[s]["D2"] for s in SEEDS)'),
    ("R20 decide: DOMINANT needs only one seed", 'd123 = all(sf[s]["D1"]', 'd123 = any(sf[s]["D1"]'),
    ("R21 decide: LONGITUDINAL FIXED ignores LON-CLEAR", '"LONGITUDINAL FIXED": pa and kp and lon_all and lat,', '"LONGITUDINAL FIXED": pa and kp and lat,'),
    ("R22 decide: LONGITUDINAL FIXED drops the lateral / crawl clause", '"LONGITUDINAL FIXED": pa and kp and lon_all and lat,', '"LONGITUDINAL FIXED": pa and kp and lon_all,'),
    ("R23 decide: NOT FIXED needs a longitudinal row worse at BOTH seeds", '"NOT FIXED": pa and kp and not lon_all}', '"NOT FIXED": pa and kp and not any(sf[s]["LON_CLEAR"] for s in SEEDS)}'),
    ("R24 decide: REFUSED never matches", '"REFUSED": bool(controls_failed),', '"REFUSED": False,'),
    ("R25 decide: the LAST matching row wins", 'out = matching[0] if matching else "UNPLACED"', 'out = matching[-1] if matching else "UNPLACED"'),
    ("R26 decide: an unplaced cell is called NOT FIXED", 'out = matching[0] if matching else "UNPLACED"', 'out = matching[0] if matching else "NOT FIXED"'),
    ("R27 DEFAULT_READING: LF at both seeds", 'DEFAULT_READING = {"lf": "any", "dom_crawl": False}', 'DEFAULT_READING = {"lf": "both", "dom_crawl": False}'),
    ("R28 outcome_rc: UNPLACED exits 0", '{"UNPLACED": 3, "REFUSED": 2}', '{"UNPLACED": 0, "REFUSED": 2}'),
    ("R29 outcome_rc: REFUSED exits 0", '{"UNPLACED": 3, "REFUSED": 2}', '{"UNPLACED": 3, "REFUSED": 0}'),
    ("R30 OUTCOME_TABLE: the NOT FIXED committed next altered", "next: HACC's rule re-fitted on DAG-OACC", "next: HACC's rule re-fitted on DAG-OACC2"),
    ("R31 one_lever: --out not dropped from the comparison", 'rest, ref = drop_flag(argv, ("--out", "--w-acc")), drop_flag(ref_argv, ("--out",))', 'rest, ref = drop_flag(argv, ("--w-acc",)), drop_flag(ref_argv, ("--out",))'),
    ("R32 one_lever: --w-acc may be named any number of times >= 1", "    if len(n) != 1:", "    if len(n) < 1:"),
    ("R33 one_lever: the value is not compared with W", "        elif w is not None and vf != w:", "        elif False:"),
    ("R34 one_lever: the rest of the argv is not compared", "    if rest != ref:\n        k = next(", "    if False:\n        k = next("),
    ("R35 bank_problems: the bank sha256 not compared", " or v[0][0] != BANK1_SHA[seed]:", ":"),
    ("R36 wacc_problems: an acc term of 0 accepted", 'or not finite(live["terms"]["acc"]) or live["terms"]["acc"] == 0.0:', 'or not finite(live["terms"]["acc"]):'),
    ("R37 wacc_problems: O's head gradient not required", '    if not (finite(ap.get("head_o_grad_diff")) and ap["head_o_grad_diff"] > 0.0):', "    if False:"),
    ("R38 wacc_problems: read_o must pass the gradient (False branch) dropped", "        if not (finite(rd) and rd > 0.0):", "        if False:"),
    ("R39 wacc_problems: the zero-init identity (True branch) dropped", "        if not (finite(rd) and rd == 0.0):", "        if False:"),
    ("R40 wacc_problems: the controls need not read exactly 0.0", "        if ap.get(k) != 0.0:", "        if False:"),
    ("R41 wacc_problems: the built w_acc not compared", '    if lv.get("w_acc") != W:', "    if False:"),
    ("R42 runrec_wacc_problems: the built w_acc not compared", "        elif lv[\"w_acc\"] != W:", "        elif False:"),
    ("R43 C-O3 tolerance 1e-4 -> 1e-2", "W_REL_TOL = 1e-4", "W_REL_TOL = 1e-2"),
    ("R44 C-O3: the fitted checkpoint not checked", '    if src.get("ckpt_sha256") != DAG_CKPT_SHA[0]:', "    if False:"),
    ("R45 C-O3: round6 not checked", "    if not finite(raw) or round6(raw) != wb:", "    if not finite(raw):"),
    ("R46 C-O3: the txt not compared", "    if txt_val is None or txt_val != wb:", "    if False:"),
    ("R47 round6: 5 significant figures", 'return float(f"{x:.6g}")', 'return float(f"{x:.5g}")'),
    ("R48 rollout_argv_problems: any token may differ", "            if flag not in ROLLOUT_PATH_FLAGS:", "            if False:"),
    ("R49 rollout_argv_problems: --limit allowed to differ", 'ROLLOUT_PATH_FLAGS = ("--ckpt", "--tag", "--out-dir", "--dump-ticks")', 'ROLLOUT_PATH_FLAGS = ("--ckpt", "--tag", "--out-dir", "--dump-ticks", "--limit")'),
    ("R50 rows_diff: tolerance 1e-9 -> huge", "                if d > tol:", "                if d > 1e3:"),
    ("R51 rows_diff: separated flag not compared", '        if bool(ra.get("separated")) != bool(rb.get("separated")):', "        if False:"),
    ("R52 prereg_problems: cell comparison dropped", "            if r != tuple(w):", "            if False:"),
    ("R53 prereg_problems: row count not compared", "        if len(rows) != len(want):", "        if False:"),
    ("R54 check_text_pin: always passes", "return got == want_sha, (", "return True, ("),
    ("R55 ambiguity: differs always False", '"differs": any(v != base for v in alts.values())', '"differs": False'),
    ("R56 families_diff: sign flipped", '"diff": a[p] - b[p]', '"diff": b[p] - a[p]'),
    ("R57 guards_beside: R-T0 bound exclusive", "bool(r_t0_point <= HALF_F)", "bool(r_t0_point < HALF_F)"),
    ("R58 crawl_rows_fail: M1 dropped from the crawl rows", 'CRAWL_ROWS = ("L2 curvature MAE (GT>=5.2)", M1_LABEL)', 'CRAWL_ROWS = ("L2 curvature MAE (GT>=5.2)",)'),
    ("R59 family_of: C-O3 routed to C-O1", 'if cid.startswith(("C-O3",)):\n        return "C-O3"', 'if cid.startswith(("C-O3",)):\n        return "C-O1"'),
    ("R60 decide: the seeds guard dropped", "    if tuple(sorted(sf)) != SEEDS:", "    if False:"),
    ("R61 flatten_numeric: booleans admitted", "isinstance(d, (int, float)) and not isinstance(d, bool) and math.isfinite(d)", "isinstance(d, (int, float)) and math.isfinite(d)"),
]
PULL_MUTANTS = [
    ("P01 base_files: the train log dropped from the 8 names", '"selfcheck.json", "summary.json", "argv.json", "train_log.jsonl", "launch_gate_token.json"]', '"selfcheck.json", "summary.json", "argv.json", "launch_gate_token.json"]'),
    ("P02 identity: init-ckpt DAG_s accepted", 'endswith(f"M2b_s{seed}/ckpt_last.pt")', 'endswith(f"M2b_s{seed}_dag/ckpt_last.pt")'),
    ("P03 identity: --dagger-p 0 accepted", "    if pv != DAGGER_P:", "    if False:"),
    ("P04 identity: the iteration-1 bank directory not enforced", 'endswith(f"dagger_bank/s{seed}")', "endswith(f\"s{seed}\")"),
    ("P05 identity: the bank sha256 not compared", "    if bs != BANK1_SHA[seed]:", "    if False:"),
    ("P06 identity: the one-lever rule not applied", '        bad += [f"C-O1: {x}" for x in one_lever_problems(argv, dag_argv, w)]', "        pass"),
    ("P07 check: a box-2 fresh eval accepted", "    if carried:", "    if False:"),
    ("P08 identity: recorded init sha not compared", "    if ics is not None and ics != M2B_CKPT_SHA[seed]:", "    if False:"),
    ("P09 read_sums: the age gate dropped", "    if age < min_age_s:", "    if False:"),
    ("P10 check: a half-pulled file accepted", '            return "INCOMPLETE", f"{name} sha256 {got[:12]} != the listed', '            pass\n            return "INCOMPLETE" if False else None, f"{name} sha256 {got[:12]} != the listed'),
    ("P11 identity: --out basename not checked", "    if out is None or os.path.basename(norm(out)) != run_name(arm, seed):", "    if False:"),
    ("P12 identity: --seed not checked", '    if one(fl, "--seed") != str(seed):', "    if False:"),
    ("P13 check: done false accepted", '        if name == "summary.json" and rec.get("done") is not True:', "        if False:"),
    ("P14 one_lever: the rest of the argv not compared", "    if rest != ref:\n        k = next(", "    if False:\n        k = next("),
    ("P15 one_lever: --w-acc value not compared with the stage's W", "        elif w is not None and vf != w:", "        elif False:"),
    ("P16 check: the w_acc stage record not required", '    if st == "INCOMPLETE":\n        return "INCOMPLETE", str(w), ""', '    if False:\n        return "INCOMPLETE", str(w), ""'),
    ("P17 wacc_status: txt vs json not compared", "    if tv != w:", "    if False:"),
    ("P18 dagbank_status: the sha256 not compared with DAG's declared one", "    if got != BANK1_SHA[seed]:", "    if False:"),
    ("P20 check: a pulled checkpoint identical to DAG's accepted", '    if listed["ckpt_last.pt"] in (DAG_CKPT_SHA[seed], M2B_CKPT_SHA[seed]):', "    if False:"),
    ("P19 identity: summary.json seed not compared", 'if sm.get("arm") != "M2b" or sm.get("seed") != seed:', 'if sm.get("arm") != "M2b":'),
]
CHAIN_MUTANTS = [
    ("K01 C: gate -ge becomes -gt", '[ "$CF0" -ge "$CMIN" ] ||', '[ "$CF0" -gt "$CMIN" ] ||', ("cgate",)),
    ("K02 E: gate -ge becomes -gt", '[ "$EF0" -ge "$EMIN" ] ||', '[ "$EF0" -gt "$EMIN" ] ||', ("egate",)),
    ("K03 PREREG pin not enforced", '[ "$(sha256sum < "$PREREGO" | cut -c1-64)" = "$PREREGO_SHA" ] ||', "true ||", ("prereg",)),
    ("K04 rollouts at --limit logged", "--replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal", "--replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit logged", ("base",)),
    ("K05 --dump-heads dropped", "--limit causal --dump-ticks \"$OUTm/ticks\" --dump-heads --device cpu)", "--limit causal --dump-ticks \"$OUTm/ticks\" --device cpu)", ("base",)),
    ("K06 a run dropped from RUNS", "RUNS=(dagoacc:0 dagoacc:1)", "RUNS=(dagoacc:0)", ("base",)),
    ("K07 the DAG-OACC tag collides with the banked DAG tag", 'tagof() { echo "DAGOACC_s${2}_m10_causal_heads"; }', 'tagof() { echo "DAG_s${2}_m10_causal_heads"; }', ("base",)),
    ("K08 rollouts on CUDA (cpu device dropped)", "--dump-heads --device cpu)", "--dump-heads)", ("base",)),
    ("K09 the second lane runs seed 0 again", "lane 1 & PB=$!", "lane 0 & PB=$!", ("REALRUN",)),
    ("K10 the tool-cfg guard dropped", '[ "$n_cfg" -ge 1 ] ||', "true ||", ("cfg",)),
    ("K11 the default box-2 watcher path altered", "chain_v8f_dagoacc_r2.out}", "chain_v8f_dagoacc.out}", ("base",)),
    ("K12 the scene flag flipped to frozen", "--nav deadreckon --scene logged --schedule periodic --limit causal", "--nav deadreckon --scene frozen --schedule periodic --limit causal", ("base",)),
    ("K13 the w_acc-in-config guard dropped", '[ "$n_wacc" -ge 1 ] ||', "true ||", ("cfg",)),
    ("K14 the offline reference pin not enforced", '[ "$(sha256sum < "$OFFLINE" | cut -c1-64)" = "$OFFLINE_SHA" ] ||', "true ||", ("offline",)),
    ("K16 the PREREG_V8F_DAG_REREAD pin not enforced", '[ "$(sha256sum < "$PREREGR" | cut -c1-64)" = "$PREREGR_SHA" ] ||', "true ||", ("reread",)),
    ("K15 the pull check called with another seed", 'out=$("$PY" "$Hm/dagoacc_t1_pullcheck.py" run "$ARTm" "$arm" "$s" --dag-art "$DAG_ARTm" 2>&1 < /dev/null); rc=$?\n    if [ $rc -eq 0 ]; then PULL_SHA',
     'out=$("$PY" "$Hm/dagoacc_t1_pullcheck.py" run "$ARTm" "$arm" 0 --dag-art "$DAG_ARTm" 2>&1 < /dev/null); rc=$?\n    if [ $rc -eq 0 ]; then PULL_SHA', ("REALRUN",)),
]


REH_MUTANTS = [
    ("H01 layer_a: the DAG-OACC - DAG table is built the other way round", '"OD": (o["cl"], g["cl"], P["OACC"], P["DAG"])', '"OD": (g["cl"], o["cl"], P["DAG"], P["OACC"])'),
    ("H02 compose: PRIMARY-ACC and KEEP rows swapped", "f, ex = seed_flags_of(tOD[ACC_LABEL], tOD[ADE_LABEL], tOP, A.DIRECTIONS)", "f, ex = seed_flags_of(tOD[ADE_LABEL], tOD[ACC_LABEL], tOP, A.DIRECTIONS)"),
    ("H03 t0_rows: R-T0 computed DAG - DAG-OACC", 'zo["ade6"], zc["ade6"], zo["eid"], N_BOOT)', 'zc["ade6"], zo["ade6"], zo["eid"], N_BOOT)'),
    ("H04 t0_rows: the R-T0 guard reads the negated point", '"guards": guards_beside(r["point"]),', '"guards": guards_beside(-r["point"]),'),
    ("H05 compose: the crawl-share flag never set", 'm1w[s] = M1_LABEL in ex["latcrawl_worse"]', "m1w[s] = False"),
    ("H06 compose: the Read-A link reads the wrong table", 'vv = A.BR["verdict"](tabs["OP"])[s]', 'vv = A.BR["verdict"](tabs["OD"])[s]'),
    ("H07 layer_a: the DAG - P10 table is built from the DAG-OACC - P10 pair", '"DP": A.table(comps["DP"], tacdk["DP"], banked_dagread, "DAG")}', '"DP": A.table(comps["OP"], tacdk["OP"], banked_dagread, "DAG")}'),
    ("H08 layer_a: C3 (P10's tracks read no model) reads the wrong arrays", 'bool(np.array_equal(o["fl"], g["fl"]))', 'bool(np.array_equal(o["cl"], g["cl"]))'),
]
READ_MUTANTS_X = [
    ("R62 outcome_from: failed controls not passed to decide()", "dec = decide(failed, sf, None, m1w)", "dec = decide([], sf, None, m1w)"),
    ("R63 outcome_from: the ambiguity block built without the failed controls", 'dec["ambiguity"] = ambiguity(failed, sf, m1w)', 'dec["ambiguity"] = ambiguity([], sf, m1w)'),
]
READ_MUTANTS += READ_MUTANTS_X


def one_mutant(kind, mid, anchor, repl, scenarios, workdir_root):
    t0 = time.time()
    wd = tempfile.mkdtemp(prefix=f"mut_{mid[:3]}_", dir=workdir_root)
    try:
        for f in os.listdir(HERE):
            sp = os.path.join(HERE, f)
            if os.path.isfile(sp) and f.endswith((".py", ".sh", ".cmd")):
                shutil.copyfile(sp, os.path.join(wd, f))
        for sub in ("inputs", "readA_code"):
            shutil.copytree(os.path.join(HERE, sub), os.path.join(wd, sub))
        target = {"read": "dagoacc_read.py", "reh": "dagoacc_read.py", "pull": "dagoacc_t1_pullcheck.py", "chain": "dagoacc_t1_chain.sh"}[kind]
        p = os.path.join(wd, target)
        txt = open(p, encoding="utf-8", newline="").read()
        if txt.count(anchor) != 1:
            return {"id": mid, "status": "ANCHOR ERROR", "count": txt.count(anchor), "seconds": time.time() - t0}
        open(p, "w", encoding="utf-8", newline="").write(txt.replace(anchor, repl))
        if scenarios == ("REALRUN",):
            return {"id": mid, "status": "UNVERIFIED", "why": "the defect sits in the real-run branch (lanes / wait loop), which a dry run never executes and which would need a rollout", "seconds": round(time.time() - t0, 1)}
        only = {"read": "L", "reh": "L,H", "pull": "P", "chain": "C"}[kind]
        cmd = [PY, "-I", os.path.join(HERE, "dagoacc_t1_selftest.py"), "--dir", wd, "--only", only] + (["--chain-scenarios", ",".join(scenarios)] if (kind == "chain" and scenarios) else [])
        env = {**os.environ, "DAGOACC_SELFTEST_INNER": "1"}
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
        fails = [ln[5:].split("   <-")[0] for ln in r.stdout.splitlines() if ln.startswith("FAIL ")]
        crashed = r.returncode not in (0, 1) or "Traceback" in r.stderr
        killed = bool(fails) or crashed
        return {"id": mid, "status": "KILLED" if killed else "SURVIVED", "killed_by": (fails[:3] if fails else ["(crash) " + r.stderr.strip().splitlines()[-1][:160]] if crashed else []), "n_red": len(fails), "seconds": round(time.time() - t0, 1)}
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def mutate(workers, out_json):
    os.makedirs(SCRATCH, exist_ok=True)
    root = tempfile.mkdtemp(prefix="mutroot_", dir=SCRATCH)
    r = subprocess.run([PY, "-I", os.path.join(HERE, "dagoacc_t1_selftest.py"), "--only", "L,P,H"], capture_output=True, text=True, env={**os.environ, "DAGOACC_SELFTEST_INNER": "1"}, timeout=900)
    ctrl_ok = r.returncode == 0
    npass_c = sum(1 for ln in r.stdout.splitlines() if ln.startswith("PASS "))
    nfail_c = sum(1 for ln in r.stdout.splitlines() if ln.startswith("FAIL "))
    print(f"CONTROL (unmutated, L+P+H): {'green' if ctrl_ok else 'RED'}  ({npass_c} pass, {nfail_c} fail)", flush=True)
    jobs = ([("read", m[0], m[1], m[2], None) for m in READ_MUTANTS] + [("reh", m[0], m[1], m[2], None) for m in REH_MUTANTS] + [("pull", m[0], m[1], m[2], None) for m in PULL_MUTANTS]
            + [("chain", m[0], m[1], m[2], m[3]) for m in CHAIN_MUTANTS])
    res = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one_mutant, k, i, a, b, sc, root) for k, i, a, b, sc in jobs]
        for f in futs:
            x = f.result()
            res.append(x)
            print(f"{x['status']:12s} {x['id']}  {x.get('killed_by', x.get('count', ''))}", flush=True)
    shutil.rmtree(root, ignore_errors=True)
    killed = sum(1 for x in res if x["status"] == "KILLED")
    unver = [x["id"] for x in res if x["status"] == "UNVERIFIED"]
    print(f"MUTATION: {killed}/{len(res) - len(unver)} killable mutants killed ({len(res)} in all; {len(unver)} UNVERIFIED real-run-only: {unver}); survived {[x['id'] for x in res if x['status'] == 'SURVIVED']}; anchor errors {[x['id'] for x in res if x['status'] == 'ANCHOR ERROR']}; control {'green' if ctrl_ok else 'RED'}")
    if out_json:
        json.dump({"control_green": ctrl_ok, "killed": killed, "total": len(res), "mutants": res}, open(out_json, "w"), indent=1)
    return 0 if (ctrl_ok and killed == len(res) - len(unver)) else 1


def real_pulls():
    """ONLY once the box-2 pulls exist: pull check COMPLETE on both runs (this HASHES the checkpoints and eval files), then the reader's --preflight (C-O1 / C-O2 / C-O3 on the run records)."""
    PC = load("pc_real", os.path.join(HERE, "dagoacc_t1_pullcheck.py"))
    M = load("man_real", os.path.join(HERE, "dagoacc_t1_manifest.py"))
    for s in (0, 1):
        d = os.path.join(OACCART, PC.run_name("dagoacc", s))
        if not os.path.isfile(os.path.join(d, "SHA256SUMS.txt")):
            print(f"--real-pulls: {d} has no SHA256SUMS.txt yet: the pulls have not landed, nothing to check")
            return 2
    for s in (0, 1):
        st, why, sha = PC.check(OACCART, "dagoacc", s, dag_art=DAGART)
        check(f"RP1 the REAL pull M2b_s{s}_dagoacc is COMPLETE", st == "COMPLETE", why)
    m = M.build(OACCART, f"{ART}/v8f-dagoacc-t1", M.parse_runs(M.DEFAULT_RUNS), DAGART, f"{ART}/v8f-dagger-t1", f"{HERE}/inputs")
    os.makedirs(SCRATCH, exist_ok=True)
    mp = os.path.join(SCRATCH, "runs_preflight.json")
    json.dump(m, open(mp, "w"), indent=1)
    shas = M.initsha(OACCART, M.parse_runs(M.DEFAULT_RUNS))
    TT = f"{ART}/dagt1_tree_b43c7a9a"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": f"{TT}/stack;{TT}/taniteval;{TT}/stack/scripts", "TANITEVAL_STACK_OVERRIDE": f"{TT}/stack".replace("/", "\\")}
    r = subprocess.run([PY, "-I", os.path.join(HERE, "dagoacc_read.py"), "--preflight", "--manifest", mp, "--tool-root", TT, "--init-sha", f"0={shas[0]}", "--init-sha", f"1={shas[1]}"], capture_output=True, text=True, env=env, timeout=900)
    fails = [ln for ln in r.stdout.splitlines() if ln.startswith("FAIL ")]
    check("RP2 the reader's --preflight (C-O1, C-O2, C-O3 and the recipe's run-record controls) on the REAL pulled records: no control fails", r.returncode == 0 and not fails, (fails[:3], r.stderr[-300:]))
    print(r.stdout[-3000:])
    bad = [n for n, ok, _ in RES if not ok]
    print(f"REAL-PULLS: {len(RES) - len(bad)}/{len(RES)} as expected" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--only", default="L,P,C,H")
    ap.add_argument("--chain-scenarios")
    ap.add_argument("--mutate", action="store_true")
    ap.add_argument("--real-pulls", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--json")
    a = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    if a.mutate:
        return mutate(min(a.workers, 2), a.json)
    if a.real_pulls:
        return real_pulls()
    only = a.only.split(",")
    t0 = time.time()
    if "L" in only:
        run_logic(a.dir)
    if "P" in only:
        run_pull(a.dir)
    if "C" in only:
        run_chain(a.dir, a.chain_scenarios.split(",") if a.chain_scenarios else None)
    if "H" in only:
        run_rehearsal(a.dir)
    bad = [n for n, ok, _ in RES if not ok]
    print(f"SELFTEST: {len(RES) - len(bad)}/{len(RES)} checks as expected ({time.time() - t0:.0f} s)" + (f"; FAILED: {bad}" if bad else ""))
    if a.json:
        json.dump([{"check": n, "ok": ok, "detail": d} for n, ok, d in RES], open(a.json, "w"), indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
