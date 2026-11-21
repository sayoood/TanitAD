#!/usr/bin/env python3
"""flatdag_t1_selftest.py -- self-test and mutation proof of the FLAT-DAG dev-box T1 chain (flatdag_read.py, flatdag_t1_pullcheck.py, flatdag_t1_chain.sh).  Plain script, no pytest, CPU, NO ROLLOUT, no eval_* opened.

    python flatdag_t1_selftest.py                      logic (L), pull check (P), chain dry-run scenarios (C) on the files beside this script          -> exit 0 iff every check is as expected
    python flatdag_t1_selftest.py --dir DIR --only L,P  the same on the files in DIR (the mutation runner uses this)
    python flatdag_t1_selftest.py --mutate [--workers 2] [--json mutation.json]
        MUTATION PROOF: each mutant is ONE textual defect in a COPY of flatdag_read.py / flatdag_t1_pullcheck.py / flatdag_t1_chain.sh.  It is KILLED iff some check goes RED on the copy; an anchor that does
        not occur exactly once is an ANCHOR ERROR; the UNMUTATED control must be green.  Chain mutants run only the scenario that targets them.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
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
SCRATCH = "F:/TanitAD-scratch/flatdag_t1_tmp"
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


# ================================================================================================================ L: logic
def run_logic(D):
    FR = load("fr_under_test", os.path.join(D, "flatdag_read.py"))
    two = 0.2415
    # --- the PREREG's committed numbers and tables
    check("L0 F, 2F, F/2, B, COVERAGE, swap constants are the PREREG's", (FR.F_M, FR.TWO_F, FR.HALF_F, FR.N_BOOT, FR.COVERAGE_MIN, FR.SWAP_P, FR.SWAP_TOL) == (0.1207, 0.2415, 0.0604, 2000, 0.5, 0.1, 0.02),
          (FR.F_M, FR.TWO_F, FR.HALF_F, FR.N_BOOT, FR.COVERAGE_MIN, FR.SWAP_P, FR.SWAP_TOL))
    txt = open(os.path.join(HERE, "inputs", "PREREG_V8F_FLATDAG.md"), encoding="utf-8").read()
    check("L1 the real pinned PREREG: section 2 + 4 cells verbatim, no problem", FR.prereg_problems(txt) == [], FR.prereg_problems(txt))
    bad_cells = {
        "a bar changed in the PRIMARY row": txt.replace("≤ −2F = −0.2415 m, separated, at both seeds", "≤ −2F = −0.2414 m, separated, at both seeds"),
        "an outcome condition changed": txt.replace("ADE6 DAG − FLAT-DAG separated-worse at ≥ 1 seed", "ADE6 DAG − FLAT-DAG separated-worse at both seeds"),
        "two outcome rows swapped": txt.replace("| **FLAT BETTER** |", "| **TMPX** |").replace("| **HIERARCHY BEATS FLAT** |", "| **FLAT BETTER** |").replace("| **TMPX** |", "| **HIERARCHY BEATS FLAT** |"),
        "a section-2 row dropped": txt.replace("| S-POS | against CTRV, both arms | beside |\n", ""),
        "the order sentence changed": txt.replace("REFUSED → BLOCKED → FLAT BETTER → HIERARCHY BEATS FLAT → NOT ESTABLISHED", "REFUSED → BLOCKED → HIERARCHY BEATS FLAT → FLAT BETTER → NOT ESTABLISHED"),
        "control C-F3 dropped": txt.replace("**C-F3, the cut survives the fine-tune:**", "**C-F9, the cut survives the fine-tune:**"),
    }
    ne = [ln for ln in txt.split(chr(10)) if ln.startswith("| **NOT ESTABLISHED**")]
    bad_cells["an extra outcome row appended"] = txt.replace(ne[0], ne[0] + chr(10) + ne[0].replace("NOT ESTABLISHED", "EXTRA ROW"), 1) if len(ne) == 1 else txt
    for nm, t in bad_cells.items():
        check(f"L1b PREREG text with {nm} -> a problem is reported", t != txt and FR.prereg_problems(t) != [], "no problem reported" if t != txt else "the mutation did not change the text")
    ok, _ = FR.check_text_pin(os.path.join(HERE, "inputs", "PREREG_V8F_FLATDAG.md"), FR.PREREG_FLATDAG_SHA256, "x")
    ok2, _ = FR.check_text_pin(os.path.join(HERE, "inputs", "PREREG_V8F_DAG_REREAD.md"), FR.PREREG_FLATDAG_SHA256, "x")
    check("L1c the pin check passes the pinned file and refuses another file", ok and not ok2, (ok, ok2))
    # --- PRIMARY H-T1
    st = FR.primary_state
    check("L2 primary_state: exactly -2F and separated is MET (inclusive)", st(rowd(-two, True)) == "MET", st(rowd(-two, True)))
    check("L2 primary_state: just above -2F is SHORT", st(rowd(-0.2414, True)) == "SHORT", st(rowd(-0.2414, True)))
    check("L2 primary_state: past -2F but not separated is NS (a bar is not a CI)", st(rowd(-0.30, False)) == "NS", st(rowd(-0.30, False)))
    check("L2 primary_state: separated-worse is WORSE; not separated positive is NS", st(rowd(0.01, True)) == "WORSE" and st(rowd(0.5, False)) == "NS", (st(rowd(0.01, True)), st(rowd(0.5, False))))
    check("L2 primary_met agrees", FR.primary_met(rowd(-0.25, True)) and not FR.primary_met(rowd(-0.20, True)) and not FR.primary_met(rowd(-0.25, False)), "")
    check("L3 sep_worse: error row, positive separated is worse; negative separated is not", FR.sep_worse(rowd(0.1, True), "lower") and not FR.sep_worse(rowd(-0.1, True), "lower"), "")
    check("L3 sep_worse: agreement row (TAC), NEGATIVE separated is worse", FR.sep_worse(rowd(-0.1, True), "higher") and not FR.sep_worse(rowd(0.1, True), "higher"), "")
    check("L3 sep_worse: an unseparated row is never worse", not FR.sep_worse(rowd(9.0, False), "lower") and not FR.sep_worse(rowd(-9.0, False), "higher"), "")
    check("L3 sep_better is the mirror", FR.sep_better(rowd(-0.1, True), "lower") and FR.sep_better(rowd(0.1, True), "higher") and not FR.sep_better(rowd(-0.1, False), "lower"), "")
    # --- the family rows
    tab = {"ADE6": rowd(-0.5, True, family="PRIMARY"), "accel MAE": rowd(0.9, True, family="LONGITUDINAL"), "TAC-LAT": rowd(-0.2, True, family="TACTICAL"),
           "M1 crawl share": rowd(0.05, True, family="crawl"), "S-POS@30": {"family": "STRATEGIC", "status": "NOT COMPUTED"}, "curvature MAE": rowd(-0.1, True, family="LATERAL")}
    dirs = {"ADE6": "lower", "accel MAE": "lower", "TAC-LAT": "higher", "M1 crawl share": "lower", "curvature MAE": "lower"}
    w, nc = FR.family_flags(tab, dirs, include_crawl=True)
    check("L4 family_flags: accel (error up), TAC-LAT (agreement down) and M1 are separated-worse; the NOT COMPUTED row is listed and never worse", w == ["M1 crawl share", "TAC-LAT", "accel MAE"] and nc == ["S-POS@30"], (w, nc))
    w2, _ = FR.family_flags(tab, dirs, include_crawl=False)
    check("L4 family_flags: the table-only reading leaves M1 out", w2 == ["TAC-LAT", "accel MAE"], w2)
    w3, _ = FR.family_flags({"unlisted error row": rowd(0.4, True, family="LATERAL"), "unlisted error row b": rowd(-0.4, True, family="LATERAL")}, {}, True)
    check("L4 family_flags: a label with no declared direction is read as an ERROR row (up = worse), never as agreement", w3 == ["unlisted error row"], w3)
    tab_ok = {k: v for k, v in tab.items() if k in ("ADE6", "curvature MAE")}
    check("L4 family_flags: no worse row on a clean table", FR.family_flags(tab_ok, dirs)[0] == [], FR.family_flags(tab_ok, dirs))
    # --- the decision
    M, S, N, Wr = rowd(-0.30, True), rowd(-0.10, True), rowd(-0.30, False), rowd(0.20, True)
    nofam = {0: [], 1: []}

    def dec(cf, bl, a, b, fam=nofam):
        return FR.decide(cf, bl, {0: a, 1: b}, fam)["OUTCOME"]
    check("L5 HIERARCHY BEATS FLAT: both seeds MET, no family row worse", dec([], [], M, M) == "HIERARCHY BEATS FLAT", dec([], [], M, M))
    check("L5 one seed MET and the other SHORT -> NOT ESTABLISHED", dec([], [], M, S) == "NOT ESTABLISHED", dec([], [], M, S))
    check("L5 one seed MET and the other not separated -> NOT ESTABLISHED", dec([], [], M, N) == "NOT ESTABLISHED", dec([], [], M, N))
    check("L5 a family row separated-worse at seed 0 only blocks the claim -> NOT ESTABLISHED", dec([], [], M, M, {0: ["accel MAE"], 1: []}) == "NOT ESTABLISHED", "")
    check("L5 a family row separated-worse at seed 1 only blocks the claim -> NOT ESTABLISHED", dec([], [], M, M, {0: [], 1: ["TAC-LAT"]}) == "NOT ESTABLISHED", "")
    check("L5 FLAT BETTER if ADE6 is separated-worse at seed 1 only (even with seed 0 MET)", dec([], [], M, Wr) == "FLAT BETTER", dec([], [], M, Wr))
    check("L5 FLAT BETTER if ADE6 is separated-worse at seed 0 only", dec([], [], Wr, M) == "FLAT BETTER", dec([], [], Wr, M))
    check("L5 FLAT BETTER precedes HIERARCHY BEATS FLAT (a worse seed cannot be a claim)", dec([], [], Wr, Wr) == "FLAT BETTER", "")
    check("L5 BLOCKED precedes FLAT BETTER and HIERARCHY BEATS FLAT", dec([], ["COVERAGE s0 = 0.4 < 0.5"], Wr, M) == "BLOCKED" and dec([], ["BUILD-SWAP"], M, M) == "BLOCKED", "")
    check("L5 REFUSED precedes everything", dec(["C-F2"], ["BUILD-SWAP"], Wr, M) == "REFUSED" and dec(["C-F4"], [], M, M) == "REFUSED", "")
    check("L5 NOT ESTABLISHED is the catch-all (SHORT, SHORT)", dec([], [], S, S) == "NOT ESTABLISHED", "")
    d = FR.decide([], [], {0: M, 1: M}, nofam)
    check("L5 matching rows are returned in table order, NOT ESTABLISHED last", d["matching_rows_in_table_order"] == ["HIERARCHY BEATS FLAT", "NOT ESTABLISHED"], d["matching_rows_in_table_order"])
    try:
        FR.decide([], [], {0: M}, {0: []})
        check("L5 decide refuses a call without both seeds", False, "no error")
    except ValueError:
        check("L5 decide refuses a call without both seeds", True)
    ob = FR.outcome_block(FR.decide([], [], {0: M, 1: M}, nofam))
    check("L5 outcome_block carries the PREREG's condition and committed next verbatim", ob["condition"] == "PRIMARY H-T1 at both seeds, and no family row separated-worse"
          and ob["committed_next"].startswith("the first hierarchy-over-flat claim of v8F"), ob)
    check("L5 outcome_rc: 0 placed outcomes, 2 REFUSED, 4 BLOCKED, 3 UNPLACED", (FR.outcome_rc({"OUTCOME": "NOT ESTABLISHED"}), FR.outcome_rc({"OUTCOME": "REFUSED"}), FR.outcome_rc({"OUTCOME": "BLOCKED"}), FR.outcome_rc({"OUTCOME": "UNPLACED"})) == (0, 2, 4, 3), "")
    r = subprocess.run([PY, "-I", os.path.join(HERE, "enumerate_cells.py")], capture_output=True, text=True, env={**os.environ, "FLATDAG_READ_DIR": D})
    check("L6 enumerate_cells: 128 cells, none unplaced, decide() == the independent table, every row reached", r.returncode == 0 and "cells: 128" in r.stdout, r.stdout[-300:] + r.stderr[-200:])
    # --- the beside guards
    check("L7 guards_beside: R-T0 point == F/2 holds (inclusive); just above fails; R-ADE separated-worse fails", FR.guards_beside(rowd(0.0, False), 0.0604)["R-T0"] and not FR.guards_beside(rowd(0.0, False), 0.0605)["R-T0"]
          and not FR.guards_beside(rowd(0.2, True), 0.0)["R-ADE"] and FR.guards_beside(rowd(-0.2, True), 0.0)["R-ADE"], "")
    # --- the build check on the REAL selfcheck / summary of a pulled run (read-only) and on mutated copies
    base = os.path.join(ART, "v8f-flatdag", "M2b_s0_flatdag")
    sm, sc = json.load(open(os.path.join(base, "summary.json"), encoding="utf-8")), json.load(open(os.path.join(base, "selfcheck.json"), encoding="utf-8"))

    def failing(sm_, sc_, p):
        return sorted(n for n, ok, _ in FR.build_check("x", sm_, sc_, p) if not ok)
    check("L8 build_check passes on the real pulled FLAT-DAG s0 record (cut built, e_code dead, O invariant, swap share in 0.1 +- 0.02)", failing(sm, sc, 0.1) == [], failing(sm, sc, 0.1))
    m1 = copy.deepcopy(sm); m1["levers"]["o_interface"] = "full"; s1 = copy.deepcopy(sc); s1["levers"]["o_interface"] = "full"
    check("L8 o_interface built full -> BUILD-CUT-LEVER fails (named)", "BUILD-CUT-LEVER" in failing(m1, s1, 0.1), failing(m1, s1, 0.1))
    s2 = copy.deepcopy(sc); s2["live"]["dead_params"] = []; s2["live"]["declared_dead_but_alive"] = ["e_code.weight"]
    check("L8 e_code alive -> BUILD-CUT-GRAD fails (named)", "BUILD-CUT-GRAD" in failing(sm, s2, 0.1), failing(sm, s2, 0.1))
    s2c = copy.deepcopy(sc); s2c["live"]["dead_params"] = []
    s2d = copy.deepcopy(sc); s2d["live"]["dead_params"] = ["e_code.weight", "other.weight"]
    check("L8 e_code NOT among the measured dead parameters (declared only) -> BUILD-CUT-GRAD fails", "BUILD-CUT-GRAD" in failing(sm, s2c, 0.1), failing(sm, s2c, 0.1))
    check("L8 a second dead parameter beside e_code -> BUILD-CUT-GRAD fails", "BUILD-CUT-GRAD" in failing(sm, s2d, 0.1), failing(sm, s2d, 0.1))
    s2b = copy.deepcopy(sc); s2b["live"]["unexpected_dead"] = ["O.w"]
    check("L8 an unexpectedly dead parameter -> BUILD-CUT-GRAD fails", "BUILD-CUT-GRAD" in failing(sm, s2b, 0.1), failing(sm, s2b, 0.1))
    s3 = copy.deepcopy(sc); s3["o_interface_probe"]["perturbations"]["gt_code"]["max_abs_dtraj60"] = 1e-3
    check("L8 O moves under a perturbation -> BUILD-CUT-INVARIANT fails (named)", "BUILD-CUT-INVARIANT" in failing(sm, s3, 0.1), failing(sm, s3, 0.1))
    s3b = copy.deepcopy(sc); s3b["o_interface_probe"]["pass"] = False
    check("L8 the probe itself not passing -> BUILD-CUT-INVARIANT fails", "BUILD-CUT-INVARIANT" in failing(sm, s3b, 0.1), failing(sm, s3b, 0.1))
    for n_, e_, ok_ in ((14000, 144182, False), (10000, 144182, False), (14581, 144182, True), (17000, 144182, True), (18000, 144182, False)):
        m4 = copy.deepcopy(sm); m4["dagger_n"], m4["dagger_eligible"] = n_, e_
        check(f"L8 swap share {n_}/{e_} = {n_ / e_:.4f}: {'inside' if ok_ else 'outside'} 0.1 +- 0.02", ("BUILD-SWAP" not in failing(m4, sc, 0.1)) == ok_ if n_ not in (14000,) else ("BUILD-SWAP" not in failing(m4, sc, 0.1)) == (abs(n_ / e_ - 0.1) <= 0.02), failing(m4, sc, 0.1))
    m5 = copy.deepcopy(sm); m5["dagger_n"], m5["dagger_eligible"] = 3, 0
    check("L8 p = 0 arm with dagger_n 3 -> BUILD-SWAP fails; with 0 passes", "BUILD-SWAP" in failing(m5, sc, 0.0) and "BUILD-SWAP" not in failing({**sm, "dagger_n": 0, "dagger_eligible": 0}, sc, 0.0), "")
    # --- argv controls
    fa = FR.argv_flags(json.load(open(os.path.join(HERE, "inputs", "flat_argv_s0.json"), encoding="utf-8"))["argv"])
    aj = FR.argv_flags(json.load(open(os.path.join(base, "argv.json"), encoding="utf-8"))["argv"])
    check("L9 flat_argv_problems: the real FLAT-DAG s0 argv is FLAT's plus exactly the recipe", FR.flat_argv_problems(aj, fa, True) == [], FR.flat_argv_problems(aj, fa, True))
    a1 = copy.deepcopy(aj); a1["--w-tac"] = [["1.0"]]
    check("L9 an extra non-recipe flag is reported", FR.flat_argv_problems(a1, fa, True) != [], "")
    a2 = copy.deepcopy(aj); a2.pop("--o-interface")
    check("L9 --o-interface not named is reported", FR.flat_argv_problems(a2, fa, True) != [], "")
    fa_no = {k: v for k, v in fa.items() if k != "--o-interface"}
    aj_no = {k: v for k, v in aj.items() if k != "--o-interface"}
    check("L9 --o-interface named by neither side (a banked FLAT argv without it) -> still reported (cut must be NAMED)", FR.flat_argv_problems(aj_no, fa_no, True) != [], FR.flat_argv_problems(aj_no, fa_no, True))
    a3 = copy.deepcopy(aj); a3["--o-interface"] = [["full"]]
    check("L9 --o-interface full is reported", FR.flat_argv_problems(a3, fa, True) != [], "")
    a4 = copy.deepcopy(aj); a4["--batch"] = [["32"]]
    check("L9 a shared flag with another value than FLAT's is reported", FR.flat_argv_problems(a4, fa, True) != [], "")
    a5 = copy.deepcopy(aj); a5.pop("--dagger-bank")
    check("L9 a missing recipe flag is reported for flatdag; flatdag0 must NOT carry the bank flags", FR.flat_argv_problems(a5, fa, True) != [] and FR.flat_argv_problems(aj, fa, False) != [], "")
    bt = [x for x in open(os.path.join(ART, "dagreread", "argv", "FLAT_s0_m10_causal_heads.argv"), encoding="utf-8").read().split("\n") if x != ""]
    nt = list(bt)
    for i, t in enumerate(nt):
        if t in ("--ckpt", "--tag", "--out-dir", "--dump-ticks"):
            nt[i + 1] = nt[i + 1] + "_X"
    check("L10 rollout_argv_problems: differing only in --ckpt/--tag/--out-dir/--dump-ticks is clean", FR.rollout_argv_problems(nt, bt) == [], FR.rollout_argv_problems(nt, bt))
    nt2 = list(bt); nt2[nt2.index("--limit") + 1] = "logged"
    check("L10 a changed --limit is reported", FR.rollout_argv_problems(nt2, bt) != [], "")
    nt3 = list(bt); nt3[nt3.index("--scene") + 1] = "frozen"
    check("L10 a changed --scene is reported", FR.rollout_argv_problems(nt3, bt) != [], "")
    check("L10 a different token count is reported", FR.rollout_argv_problems(bt + ["--x"], bt) != [], "")
    # --- the oracle identity row (numpy)
    import numpy as np

    class PB:
        calls = 0

        @staticmethod
        def f(a, b, eid, n_boot=2000, seed=0):
            d = float(np.mean(a - b))
            return {"delta": d, "lo": d, "hi": d}
    rng = np.random.default_rng(0)
    own = {"eid": np.repeat(np.arange(5), 4), "ade6": rng.random(20), "pred": rng.random((20, 3, 2))}
    orc = {k: v.copy() for k, v in own.items()}
    js_ok = {"tac_manipulation": {"mode": "oracle", "frac_code_changed": 0.4}}
    pr, idr = FR.oracle_identity_problems(np, own, orc, js_ok, PB.f)
    check("L11 oracle identity: equal arrays on a code-changing oracle -> clean", pr == [] and idr["delta"] == 0.0, pr)
    o2 = {k: v.copy() for k, v in orc.items()}; o2["ade6"][3] += 1e-9
    check("L11 one window differs by 1e-9 -> reported", FR.oracle_identity_problems(np, own, o2, js_ok, PB.f)[0] != [], "")
    o3 = {k: v.copy() for k, v in orc.items()}; o3["pred"][0, 0, 0] += 1e-9
    check("L11 a prediction differs by 1e-9 -> reported", FR.oracle_identity_problems(np, own, o3, js_ok, PB.f)[0] != [], "")
    check("L11 mode absmask (not an oracle) -> reported", FR.oracle_identity_problems(np, own, orc, {"tac_manipulation": {"mode": "absmask", "frac_code_changed": 0.4}}, PB.f)[0] != [], "")
    check("L11 frac_code_changed 0 (a zero would say nothing) -> reported", FR.oracle_identity_problems(np, own, orc, {"tac_manipulation": {"mode": "oracle", "frac_code_changed": 0.0}}, PB.f)[0] != [], "")
    check("L11 no json at all -> reported", FR.oracle_identity_problems(np, own, orc, None, PB.f)[0] != [], "")
    o4 = {k: v.copy() for k, v in orc.items()}; o4["eid"] = o4["eid"][::-1].copy()
    check("L11 another eval grid -> reported", FR.oracle_identity_problems(np, own, o4, js_ok, PB.f)[0] != [], "")
    # --- rows_diff
    ra = {"ADE6": {"delta": -0.5, "lo": -0.6, "hi": -0.4, "separated": True, "n_windows": 2399, "mean_a": 2.0, "mean_b": 2.5}, "_M2": {"M2": 0.1}}
    rb = copy.deepcopy(ra)
    check("L12 rows_diff: equal rows -> no difference, max 0.0; a metadata label is skipped", FR.rows_diff(ra, rb) == ([], 0.0), FR.rows_diff(ra, rb))
    rb["ADE6"]["delta"] = -0.5 + 1e-6
    check("L12 rows_diff: a 1e-6 difference is reported", FR.rows_diff(ra, rb)[0] != [], "")
    rb = copy.deepcopy(ra); rb["ADE6"]["separated"] = False
    check("L12 rows_diff: a flipped separated flag is reported", FR.rows_diff(ra, rb)[0] != [], "")
    rb = copy.deepcopy(ra); rb["ADE6"]["n_windows"] = 2398
    check("L12 rows_diff: a different n is reported", FR.rows_diff(ra, rb)[0] != [], "")
    rb = copy.deepcopy(ra); del rb["ADE6"]
    check("L12 rows_diff: a row on one side only is reported", FR.rows_diff(ra, rb)[0] != [], "")
    # --- the g_argv substitution and the COVERAGE routing, on stubs
    class DRs:
        BANKED_M2B_SHA = {}

        @staticmethod
        def one_value(flags, name):
            occ = flags.get(name)
            return (True, occ[0][0]) if occ and len(occ) == 1 and len(occ[0]) == 1 else (False, None)

        @staticmethod
        def g_argv(tag, arm, seed, r):
            v = r["flags"]["--init-ckpt"][0][0]
            return v.endswith(f"M2b_s{seed}/ckpt_last.pt"), "orig"

    ga = FR.make_g_argv(DRs)
    okflat = ga("t", "DAG", 0, {"flags": {"--init-ckpt": [["D:\\tanitad\\runs\\v8f_mini_full\\M2b_s0_flat\\ckpt_last.pt"]]}})
    badm2b = ga("t", "DAG", 0, {"flags": {"--init-ckpt": [["D:\\tanitad\\runs\\v8f_mini_full\\M2b_s0\\ckpt_last.pt"]]}})
    baddag = ga("t", "DAG", 0, {"flags": {"--init-ckpt": [["D:\\tanitad\\runs\\v8f_mini_full\\M2b_s0_dag\\ckpt_last.pt"]]}})
    wrongseed = ga("t", "DAG", 0, {"flags": {"--init-ckpt": [["D:\\tanitad\\runs\\v8f_mini_full\\M2b_s1_flat\\ckpt_last.pt"]]}})
    check("L13 g_argv: init FLAT_s0 is accepted and handed to the reader's own clause", okflat[0] is True, okflat)
    check("L13 g_argv: init M2b_s0, DAG_s0 or FLAT_s1 for seed 0 is refused", not badm2b[0] and not baddag[0] and not wrongseed[0], (badm2b, baddag, wrongseed))

    class DRb:
        def __init__(self, frac):
            self.frac = frac

        def load_bank(self, rep, bf, what=""):
            return {"x": 1}

        def bank_stage(self, ctl, s, B, sha, ev, tr):
            pass

        def coverage_of(self, B):
            return {"eligible_fraction": self.frac}
    man = {"bank": {"0": {"report": __file__, "bank_file": __file__}, "1": {"report": __file__, "bank_file": __file__}}}
    for frac, want in ((0.49, 2), (0.5, 0), (0.51, 0)):
        _, bl = FR.stage_banks(DRb(frac), None, man, {0: "a", 1: "b"}, [], [], {})
        check(f"L14 COVERAGE {frac}: BLOCKED reasons = {want} (the minimum is inclusive 0.5)", len(bl) == want and all("COVERAGE" in b for b in bl), bl)
    check("L15 family_of routes controls to C-F1..C-F4", FR.family_of("K-B2:s0") == "C-F1" and FR.family_of("FLAT-ARGV:FLATDAG_s0") == "C-F2" and FR.family_of("ORACLE-IDENTITY:FLATDAG_s0") == "C-F3" and FR.family_of("RC-B2:s0") == "C-F4", "")


# ================================================================================================================ P: pull check
def run_pull(D):
    PC = load("pc_under_test", os.path.join(D, "flatdag_t1_pullcheck.py"))
    tmp = tempfile.mkdtemp(prefix="fdpull_", dir=SCRATCH)
    FLAT = PC.FLAT_CKPT_SHA

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    def mkrun(art, arm, seed, **kw):
        d = os.path.join(art, PC.run_name(arm, seed))
        os.makedirs(os.path.join(d, "gate"), exist_ok=True)
        p = kw.get("p", 0.1 if arm == "flatdag" else 0.0)
        argv = ["--arm", "M2b", "--seed", str(kw.get("seed_arg", seed)), "--out", "D:\\tanitad\\runs\\v8f_mini_full\\" + kw.get("out", PC.run_name(arm, seed)),
                "--init-ckpt", kw.get("init", f"D:\\tanitad\\runs\\v8f_mini_full\\M2b_s{seed}_flat\\ckpt_last.pt"), "--steps", "3000"]
        if kw.get("cut", True):
            argv += ["--o-interface", "cut"]
        argv += ["--dagger-p", str(p)]
        if arm == "flatdag" or kw.get("bank_on_0"):
            argv += ["--dagger-bank", kw.get("bank", f"D:\\tanitad\\runs\\v8f_mini_full\\dagger_bank_flat\\s{seed}"), "--dagger-bank-sha256", "ab" * 32, "--dagger-seed", str(seed)]
        files = {"ckpt_last.pt": b"ckpt" + bytes([seed]) + arm.encode(), f"eval_{PC.eval_tag(arm, seed)}.json": b"{}", f"eval_{PC.eval_tag(arm, seed)}.npz": b"npz",
                 f"eval_{PC.eval_tag(arm, seed)}-tacorc.npz": b"orc", "selfcheck.json": b"{}", "train_log.jsonl": b"{}\n", "launch_gate_token.json": b"{}",
                 "summary.json": json.dumps({"done": kw.get("done", True), "arm": "M2b", "seed": seed, "levers": {"dagger_p": p, "init_ckpt": {"sha256": kw.get("init_sha", FLAT[seed])}}}).encode(),
                 "argv.json": json.dumps({"argv": argv}).encode()}
        for n, b in files.items():
            open(os.path.join(d, n), "wb").write(b)
        names = [n for n in PC.base_files(arm, seed)] if not kw.get("sums_names") else kw["sums_names"]
        if kw.get("fresh"):
            names = names + PC.box2_fresh_files(arm, seed)
            for n in PC.box2_fresh_files(arm, seed):
                open(os.path.join(d, n), "wb").write(b"fresh")
        lines = "".join(f"{sha(open(os.path.join(d, n), 'rb').read()) if os.path.exists(os.path.join(d, n)) else '0' * 64} *{n}\n" for n in names)
        if kw.get("bad_sum"):
            lines = lines.replace(lines[:8], "deadbeef", 1)
        open(os.path.join(d, "SHA256SUMS.txt"), "w", newline="\n").write(lines)
        t = os.path.getmtime(os.path.join(d, "SHA256SUMS.txt"))
        return d, t

    def st(art, arm, seed, now=None):
        return PC.check(art, arm, seed, min_age_s=3.0, now=now)[0]
    art = os.path.join(tmp, "A")
    d, t = mkrun(art, "flatdag", 0)
    check("P1 a synthetic complete FLAT-DAG s0 pull is COMPLETE", st(art, "flatdag", 0, now=t + 10) == "COMPLETE", PC.check(art, "flatdag", 0, now=t + 10)[1])
    check("P1 the sums are 9 names with the oracle -tacorc.npz", len(PC.base_files("flatdag", 0)) == 9 and "eval_M2b-flatdag_s0-tacorc.npz" in PC.base_files("flatdag", 0), PC.base_files("flatdag", 0))
    check("P2 a SHA256SUMS.txt younger than 3 s is INCOMPLETE (may still be filling)", st(art, "flatdag", 0, now=t + 1) == "INCOMPLETE", st(art, "flatdag", 0, now=t + 1))
    art2 = os.path.join(tmp, "B"); mkrun(art2, "flatdag0", 0, sums_names=PC.base_files("flatdag0", 0)[:-1])
    check("P3 8 of the 9 names listed -> INCOMPLETE", st(art2, "flatdag0", 0, now=time.time() + 10) == "INCOMPLETE", st(art2, "flatdag0", 0, now=time.time() + 10))
    art3 = os.path.join(tmp, "C"); d3, t3 = mkrun(art3, "flatdag", 0); os.remove(os.path.join(d3, "eval_M2b-flatdag_s0.npz"))
    check("P4 a listed file absent -> INCOMPLETE", st(art3, "flatdag", 0, now=t3 + 10) == "INCOMPLETE", st(art3, "flatdag", 0, now=t3 + 10))
    art4 = os.path.join(tmp, "D"); d4, t4 = mkrun(art4, "flatdag", 0); open(os.path.join(d4, "ckpt_last.pt"), "wb").write(b"half")
    check("P5 a file whose sha256 differs from the sums -> INCOMPLETE", st(art4, "flatdag", 0, now=t4 + 10) == "INCOMPLETE", st(art4, "flatdag", 0, now=t4 + 10))
    art5 = os.path.join(tmp, "E"); d5, t5 = mkrun(art5, "flatdag", 0, done=False)
    check("P6 summary.json done false -> INCOMPLETE", st(art5, "flatdag", 0, now=t5 + 10) == "INCOMPLETE", st(art5, "flatdag", 0, now=t5 + 10))
    art6 = os.path.join(tmp, "F"); d6, t6 = mkrun(art6, "flatdag", 0, fresh=True)
    check("P7 a box-2 fresh-age eval in the sums -> WRONG (never waited for)", st(art6, "flatdag", 0, now=t6 + 10) == "WRONG", st(art6, "flatdag", 0, now=t6 + 10))
    art7 = os.path.join(tmp, "G"); d7, t7 = mkrun(art7, "flatdag0", 0, bank_on_0=True, p=0.1)
    check("P8 flatdag0's directory holding a --dagger-p 0.1 run -> WRONG", st(art7, "flatdag0", 0, now=t7 + 10) == "WRONG", PC.check(art7, "flatdag0", 0, now=t7 + 10)[1])
    art8 = os.path.join(tmp, "H"); d8, t8 = mkrun(art8, "flatdag", 0, p=0.0)
    check("P9 flatdag's directory holding a --dagger-p 0 run -> WRONG", st(art8, "flatdag", 0, now=t8 + 10) == "WRONG", PC.check(art8, "flatdag", 0, now=t8 + 10)[1])
    art7b = os.path.join(tmp, "G2"); d7b, t7b = mkrun(art7b, "flatdag0", 0, p=0.1)
    check("P8b flatdag0's directory holding a --dagger-p 0.1 run WITHOUT the bank flags -> WRONG (by the p alone)", st(art7b, "flatdag0", 0, now=t7b + 10) == "WRONG", PC.check(art7b, "flatdag0", 0, now=t7b + 10)[1])
    art7c = os.path.join(tmp, "G3"); d7c, t7c = mkrun(art7c, "flatdag0", 0, p=0.0, bank_on_0=True)
    check("P8c flatdag0's directory naming the bank flags with --dagger-p 0 -> WRONG (by the bank flags alone)", st(art7c, "flatdag0", 0, now=t7c + 10) == "WRONG", PC.check(art7c, "flatdag0", 0, now=t7c + 10)[1])
    art9 = os.path.join(tmp, "I"); d9, t9 = mkrun(art9, "flatdag", 0, cut=False)
    check("P10 argv without --o-interface cut -> WRONG", st(art9, "flatdag", 0, now=t9 + 10) == "WRONG", PC.check(art9, "flatdag", 0, now=t9 + 10)[1])
    art10 = os.path.join(tmp, "J"); d10, t10 = mkrun(art10, "flatdag", 0, init="D:\\tanitad\\runs\\v8f_mini_full\\M2b_s0\\ckpt_last.pt")
    check("P11 init-ckpt = M2b_s0 (not FLAT_s0) -> WRONG", st(art10, "flatdag", 0, now=t10 + 10) == "WRONG", PC.check(art10, "flatdag", 0, now=t10 + 10)[1])
    art10b = os.path.join(tmp, "J2"); d10b, t10b = mkrun(art10b, "flatdag", 0, init="D:\\tanitad\\runs\\v8f_mini_full\\M2b_s0_dag\\ckpt_last.pt")
    check("P11 init-ckpt = DAG_s0 -> WRONG", st(art10b, "flatdag", 0, now=t10b + 10) == "WRONG", PC.check(art10b, "flatdag", 0, now=t10b + 10)[1])
    art11 = os.path.join(tmp, "K"); d11, t11 = mkrun(art11, "flatdag", 0, bank="D:\\tanitad\\runs\\v8f_mini_full\\dagger_bank\\s0")
    check("P12 the M2b/iteration-1 bank (not FLAT's own) -> WRONG", st(art11, "flatdag", 0, now=t11 + 10) == "WRONG", PC.check(art11, "flatdag", 0, now=t11 + 10)[1])
    art12 = os.path.join(tmp, "L"); d12, t12 = mkrun(art12, "flatdag", 0, out="M2b_s0_flatdag0")
    check("P13 --out naming another run -> WRONG", st(art12, "flatdag", 0, now=t12 + 10) == "WRONG", PC.check(art12, "flatdag", 0, now=t12 + 10)[1])
    art13 = os.path.join(tmp, "M"); d13, t13 = mkrun(art13, "flatdag", 0, init_sha="00" * 32)
    check("P14 summary.json levers.init_ckpt.sha256 not FLAT_s0's -> WRONG", st(art13, "flatdag", 0, now=t13 + 10) == "WRONG", PC.check(art13, "flatdag", 0, now=t13 + 10)[1])
    art14 = os.path.join(tmp, "N"); d14, t14 = mkrun(art14, "flatdag", 1, seed_arg=0)
    check("P15 argv --seed 0 in a seed-1 directory -> WRONG", st(art14, "flatdag", 1, now=t14 + 10) == "WRONG", PC.check(art14, "flatdag", 1, now=t14 + 10)[1])
    check("P16 an unknown arm or an absent directory -> INCOMPLETE", st(art, "dag2", 0) == "INCOMPLETE" and st(art, "flatdag", 1) == "INCOMPLETE", "")
    real = [st(os.path.join(ART, "v8f-flatdag"), a, s) for a, s in (("flatdag", 0), ("flatdag0", 0), ("flatdag", 1))]
    check("P17 the REAL pulls M2b_s0_flatdag, M2b_s0_flatdag0, M2b_s1_flatdag are COMPLETE", real == ["COMPLETE"] * 3, real)
    rb = [PC.check_bank_status(os.path.join(ART, "v8f-flatdag"), s, expect_shards=4)[0] for s in (0, 1)]
    check("P18 the REAL banks s0 and s1 are COMPLETE (11 files, bank.json == the gate's declared sha256)", rb == ["COMPLETE"] * 2, rb)
    check("P19 a bank with the wrong expected shard count is WRONG", PC.check_bank_status(os.path.join(ART, "v8f-flatdag"), 0, expect_shards=3)[0] == "WRONG", "")
    shutil.rmtree(tmp, ignore_errors=True)


# ================================================================================================================ C: chain dry-run scenarios
def chain(D, env=None, extra=None):
    e = {**os.environ, "FDT1_TEST_CFREE_MB": "3000", "FDT1_TEST_EFREE_MB": "3000", "FDT1_B2_OUT": "off"}
    for k, v in (env or {}).items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = v
    r = subprocess.run([BASH, "-lc", f"bash '{D.replace(chr(92), '/')}/flatdag_t1_chain.sh' --dry-run < /dev/null"], capture_output=True, text=True, env=e, timeout=600)
    return r.returncode, r.stdout + r.stderr


def run_chain(D, which=None):
    S = which or ("base", "cgate", "egate", "prereg", "tree", "cfg")
    if "base" in S:
        rc, o = chain(D)
        check("C1 baseline dry-run ends ZZFLATDAGT1-DRY-OK, rc 0", rc == 0 and "ZZFLATDAGT1-DRY-OK" in o, o[-300:])
        cmds = [ln for ln in o.splitlines() if "[lane s" in ln]
        check("C2 four rollout commands: FLAT-DAG s0, FLAT-DAG-0 s0 (lane s0), FLAT-DAG s1, FLAT-DAG-0 s1 (lane s1), tags never FLAT_s..", len(cmds) == 4 and all(f"--tag {t}_m10_causal_heads" in c for c, t in zip(cmds, ("FLATDAG_s0", "FLATDAG0_s0", "FLATDAG_s1", "FLATDAG0_s1")))
              and "[lane s0]" in cmds[0] and "[lane s0]" in cmds[1] and "[lane s1]" in cmds[2] and "[lane s1]" in cmds[3], cmds)
        check("C3 every rollout is --limit causal --dump-heads --dump-ticks, m=10, deadreckon, logged scene, periodic, on CPU, on the pulled checkpoint",
              all(x in c for c in cmds for x in ("--limit causal", "--dump-heads", "--dump-ticks", "--replan-steps 10", "--nav deadreckon", "--scene logged", "--schedule periodic", "--device cpu")) and
              all(f"v8f-flatdag/M2b_s{s}_{a}/ckpt_last.pt" in c for c, (a, s) in zip(cmds, (("flatdag", 0), ("flatdag0", 0), ("flatdag", 1), ("flatdag0", 1)))), cmds[:1])
        check("C4 no logged-bucket rollout and no fresh-T0 command is planned", "--limit logged" not in o.split("== rollouts")[1].split("== NOT run here")[0] and "--ages fresh" not in o, "")
        check("C5 the pulls are reported by name (four run states and two banks)", all(f"M2b_s{s}_{a}" in o for a in ("flatdag", "flatdag0") for s in (0, 1)) and "bank_s0" in o and "bank_s1" in o, "")
        check("C6 the reader command carries the FLAT shas slots, the pinned PREREGs and writes only under v8f-flatdag-t1", "--prereg-flatdag" in o and "--init-sha" in o and "v8f-flatdag-t1/read/FLATDAG_READ.json" in o, "")
        check("C7 the dry run says the box-2 DONE gate and the bank gate come after the lanes", "box-2 ZZV8FMINI-DONE" in o and "bank gate" in o, "")
        rcb, ob = chain(D, {"FDT1_B2_OUT": None})
        check("C8 the default box-2 watcher is the box-2 chain's own log (read-only)", rcb == 0 and "box-2 watcher: C:/Users/Admin/box2_ship/logs/chain_v8f_flatdag.out" in ob, ob[-200:])
    if "cgate" in S:
        rc1, o1 = chain(D, {"FDT1_TEST_CFREE_MB": "2047"})
        rc2, o2 = chain(D, {"FDT1_TEST_CFREE_MB": "2048"})
        check("C9 C: gate: 2047 MB free STOPs (rc 1, ZZFLATDAGT1-STOP)", rc1 == 1 and "ZZFLATDAGT1-STOP" in o1 and "C: has 2047 MB" in o1, o1[-200:])
        check("C9 C: gate: exactly 2048 MB passes", rc2 == 0 and "ZZFLATDAGT1-DRY-OK" in o2, o2[-200:])
    if "egate" in S:
        rc1, o1 = chain(D, {"FDT1_TEST_EFREE_MB": "1499"})
        rc2, o2 = chain(D, {"FDT1_TEST_EFREE_MB": "1500"})
        check("C10 E: gate: 1499 MB free STOPs", rc1 == 1 and "E: has 1499 MB" in o1, o1[-200:])
        check("C10 E: gate: exactly 1500 MB passes", rc2 == 0 and "ZZFLATDAGT1-DRY-OK" in o2, o2[-200:])
    if "prereg" in S:
        t = open(os.path.join(HERE, "inputs", "PREREG_V8F_FLATDAG.md"), "rb").read().replace(b"0.2415", b"0.2414", 1)
        p = os.path.join(SCRATCH, "PREREG_tampered.md")
        open(p, "wb").write(t)
        rc, o = chain(D, {"FDT1_PREREGF": p})
        check("C11 a tampered PREREG (one number changed) STOPs on its pin", rc == 1 and "ZZFLATDAGT1-STOP" in o and "not the pinned text" in o, o[-300:])
    if "tree" in S:
        rc, o = chain(D, {"FDT1_TREE": "E:/Projects/TanitAD-artifacts/no_such_tree"})
        check("C12 a missing tool tree STOPs", rc == 1 and "ZZFLATDAGT1-STOP" in o, o[-200:])
    if "cfg" in S:
        src = os.path.join(ART, "dagt1_tree_b43c7a9a")
        res = {}
        for nm, mod in (("control", False), ("no_cfg", True)):
            mt = os.path.join(SCRATCH, "mintree_" + nm)
            shutil.rmtree(mt, ignore_errors=True)
            for f in ("stack/scripts/v8f_t1_eval.py", "stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_dagger.py"):
                os.makedirs(os.path.dirname(os.path.join(mt, f)), exist_ok=True)
                shutil.copyfile(os.path.join(src, f), os.path.join(mt, f))
            os.makedirs(os.path.join(mt, "taniteval"), exist_ok=True)
            if mod:
                tp = os.path.join(mt, "stack/scripts/v8f_t1_eval.py")
                tx = open(tp, encoding="utf-8", newline="").read()
                assert tx.count('V8FMiniConfig(**ck["cfg"])') >= 1
                open(tp, "w", encoding="utf-8", newline="").write(tx.replace('V8FMiniConfig(**ck["cfg"])', 'V8FMiniConfig(**ck["cfg_x"])'))
            res[nm] = chain(D, {"FDT1_TREE": mt.replace(chr(92), "/")})
            shutil.rmtree(mt, ignore_errors=True)
        check("C14 control: a three-file tree whose tool builds from ck[cfg] passes the cfg guard (and stops later, at --help, for lack of imports)", res["control"][0] == 1 and "does not build the model" not in res["control"][1] and "--help did not run" in res["control"][1], res["control"][1][-250:])
        check("C14 a tool that does NOT build the model from the checkpoint's cfg STOPs on that guard", res["no_cfg"][0] == 1 and "does not build the model from the checkpoint's cfg" in res["no_cfg"][1], res["no_cfg"][1][-250:])
    if "syntax" in S or which is None:
        r = subprocess.run([BASH, "-n", os.path.join(D, "flatdag_t1_chain.sh")], capture_output=True, text=True)
        check("C13 bash -n flatdag_t1_chain.sh (the real-run branch is at least syntactically valid)", r.returncode == 0, r.stderr[:200])


# ================================================================================================================ M: mutation proof
READ_MUTANTS = [
    ("R01 primary_met: <= -2F becomes < -2F", 'return bool(row["separated"] and row["delta"] <= -TWO_F)', 'return bool(row["separated"] and row["delta"] < -TWO_F)'),
    ("R02 primary_met: separation dropped", 'return bool(row["separated"] and row["delta"] <= -TWO_F)', 'return bool(row["delta"] <= -TWO_F)'),
    ("R03 TWO_F 0.2415 -> 0.2414", "F_M, TWO_F, HALF_F = 0.1207, 0.2415, 0.0604", "F_M, TWO_F, HALF_F = 0.1207, 0.2414, 0.0604"),
    ("R04 primary_state: WORSE/SHORT inverted", 'return "WORSE" if row["delta"] > 0 else "SHORT"', 'return "WORSE" if row["delta"] < 0 else "SHORT"'),
    ("R05 sep_worse: direction flipped", 'return bool(row["separated"] and ((row["delta"] > 0) if direction == "lower" else (row["delta"] < 0)))', 'return bool(row["separated"] and ((row["delta"] < 0) if direction == "lower" else (row["delta"] > 0)))'),
    ("R06 sep_worse: separation dropped", 'return bool(row["separated"] and ((row["delta"] > 0) if direction == "lower" else (row["delta"] < 0)))', 'return bool(((row["delta"] > 0) if direction == "lower" else (row["delta"] < 0)))'),
    ("R07 family_rows: NOT COMPUTED rows admitted", 'if r.get("status") != "NOT COMPUTED" and (include_crawl', 'if (include_crawl'),
    ("R08 family_rows: M1 always counted", '(include_crawl or r.get("family") != "crawl")', 'True'),
    ("R09 family_flags: direction default flipped to higher", 'sep_worse(r, directions.get(lab, "lower"))', 'sep_worse(r, directions.get(lab, "higher"))'),
    ("R10 decide: FLAT BETTER needs both seeds", 'worse_any = any(st[s] == "WORSE" for s in SEEDS)', 'worse_any = all(st[s] == "WORSE" for s in SEEDS)'),
    ("R11 decide: the claim needs only one seed MET", 'met_both = all(st[s] == "MET" for s in SEEDS)', 'met_both = any(st[s] == "MET" for s in SEEDS)'),
    ("R12 decide: family rows read at seed 0 only", 'fam_none = not any(fam_worse[s] for s in SEEDS)', 'fam_none = not fam_worse[0]'),
    ("R13 decide: BLOCKED never matches", '"BLOCKED": bool(blocked)', '"BLOCKED": False'),
    ("R14 decide: REFUSED never matches", '"REFUSED": bool(controls_failed)', '"REFUSED": False'),
    ("R15 decide: NOT ESTABLISHED no longer the catch-all", '"NOT ESTABLISHED": True}', '"NOT ESTABLISHED": False}'),
    ("R16 decide: FLAT BETTER ignores seed 1", 'worse_any = any(st[s] == "WORSE" for s in SEEDS)', 'worse_any = st[0] == "WORSE"'),
    ("R17 outcome_rc: BLOCKED exits 0", '{"UNPLACED": 3, "BLOCKED": 4, "REFUSED": 2}', '{"UNPLACED": 3, "BLOCKED": 0, "REFUSED": 2}'),
    ("R18 OUTCOME_TABLE: the HIERARCHY row's committed next altered", '"the first hierarchy-over-flat claim of v8F, at T1-kin: register it and signal the PI"', '"the first hierarchy-over-flat claim of v8F: register it"'),
    ("R19 build_check: a built 'full' accepted as the cut", 'built == "cut"', 'built in ("cut", "full")'),
    ("R20 build_check: e_code dead no longer required", 'dead == ["e_code.weight"] and decl == ["e_code.weight"]', 'decl == ["e_code.weight"]'),
    ("R21 build_check: O's traj60 invariance not read", 'v.get("max_abs_dtraj60") == 0.0 and', 'v.get("max_abs_dtraj60") is not None and'),
    ("R22 build_check: swap tolerance 0.02 -> 0.2", "SWAP_P, SWAP_TOL = 0.1, 0.02", "SWAP_P, SWAP_TOL = 0.1, 0.2"),
    ("R23 build_check: flatdag0 may swap", "ok = isinstance(n, int) and not isinstance(n, bool) and n == 0", "ok = isinstance(n, int) and not isinstance(n, bool)"),
    ("R24 flat_argv_problems: extra flags allowed", "    if extra != want:", "    if False:"),
    ("R25 flat_argv_problems: --o-interface cut not required", '    if ic != [["cut"]]:', "    if False:"),
    ("R26 flat_argv_problems: shared flags not compared", '    if diff:\n        bad.append(f"shared flags', '    if False:\n        bad.append(f"shared flags'),
    ("R27 rollout_argv_problems: any token may differ", "            if flag not in ROLLOUT_PATH_FLAGS:", "            if False:"),
    ("R28 rollout_argv_problems: --limit allowed to differ", 'ROLLOUT_PATH_FLAGS = ("--ckpt", "--tag", "--out-dir", "--dump-ticks")', 'ROLLOUT_PATH_FLAGS = ("--ckpt", "--tag", "--out-dir", "--dump-ticks", "--limit")'),
    ("R29 oracle identity: an absmask eval accepted as the oracle", 'if m.get("mode") != "oracle" or not', 'if False or not'),
    ("R30 oracle identity: frac_code_changed 0 accepted", 'm["frac_code_changed"] > 0.0', 'm["frac_code_changed"] >= 0.0'),
    ("R31 oracle identity: predictions not compared", "np.array_equal(a_o, a_n) and worst_pred == 0.0 and", "np.array_equal(a_o, a_n) and"),
    ("R32 oracle identity: eval grid not compared", 'if not np.array_equal(np.asarray(z_own["eid"]), np.asarray(z_orc["eid"])):', "if False:"),
    ("R33 guards_beside: R-T0 bound exclusive", 'bool(r_t0_point <= HALF_F)', 'bool(r_t0_point < HALF_F)'),
    ("R34 rows_diff: tolerance 1e-9 -> huge", "                if d > tol:", "                if d > 1e3:"),
    ("R35 rows_diff: separated flag not compared", '        if bool(ra.get("separated")) != bool(rb.get("separated")):', "        if False:"),
    ("R36 prereg_problems: cell comparison dropped", "            if r != tuple(w):", "            if False:"),
    ("R37 prereg_problems: row count not compared", "        if len(rows) != len(want):", "        if False:"),
    ("R38 check_text_pin: always passes", "return got == want_sha, (", "return True, ("),
    ("R39 stage_banks: COVERAGE minimum 0.5 -> 0.4", "COVERAGE_MIN = 0.5", "COVERAGE_MIN = 0.4"),
    ("R40 make_g_argv: the M2b tail accepted", 'want_tail = f"M2b_s{seed}_flat/ckpt_last.pt"', 'want_tail = f"M2b_s{seed}/ckpt_last.pt"'),
    ("R41 make_g_argv: the tail check dropped", "        if not tail_norm(v).endswith(want_tail):", "        if False:"),
    ("R42 build_check: the probe's own pass not required", 'bool(pr.get("pass") is True and pr.get("o_interface") == "cut" and inv)', 'bool(pr.get("o_interface") == "cut" and inv)'),
    ("R43 primary bar: inclusive bound on ROW 'NS' -> a past-bar unseparated row counts", 'if not row["separated"]:\n        return "NS"', 'if False:\n        return "NS"'),
    ("R44 family_of: C-F1 prefix dropped", 'if cid.startswith(("BANK-", "K-B", "COVERAGE")):', 'if cid.startswith(("BANK-", "COVERAGE")):'),
]
PULL_MUTANTS = [
    ("P01 base_files: the oracle npz dropped from the 9 names", 'f"eval_{tg}.npz", f"eval_{tg}-tacorc.npz", "selfcheck.json"', 'f"eval_{tg}.npz", "selfcheck.json"'),
    ("P02 identity: init-ckpt M2b accepted", 'endswith(f"M2b_s{seed}_flat/ckpt_last.pt")', 'endswith(f"M2b_s{seed}/ckpt_last.pt")'),
    ("P03 identity: --o-interface cut not required", 'if one(fl, "--o-interface") != "cut":', "if False:"),
    ("P04 identity: the iteration-1 bank accepted", 'endswith(f"dagger_bank_flat/s{seed}")', 'endswith(f"dagger_bank/s{seed}")'),
    ("P05 identity: flatdag0 may carry p > 0", 'elif arm == "flatdag0" and pv != 0.0:', "elif False:"),
    ("P06 identity: flatdag may carry p 0", 'elif arm == "flatdag" and not pv > 0.0:', "elif False:"),
    ("P07 check: a box-2 fresh eval accepted", "    if carried:", "    if False:"),
    ("P08 identity: recorded init sha not compared", "    if ics is not None and ics != FLAT_CKPT_SHA[seed]:", "    if False:"),
    ("P09 read_sums: the age gate dropped", "    if age < min_age_s:", "    if False:"),
    ("P10 check: a half-pulled file accepted", '            return "INCOMPLETE", f"{name} sha256 {got[:12]} != the listed', '            pass\n            return "INCOMPLETE" if False else None, f"{name} sha256 {got[:12]} != the listed'),
    ("P11 identity: --out basename not checked", "    if out is None or os.path.basename(norm(out)) != run_name(arm, seed):", "    if False:"),
    ("P12 identity: --seed not checked", '    if one(fl, "--seed") != str(seed):', "    if False:"),
    ("P13 check: done false accepted", '        if name == "summary.json" and rec.get("done") is not True:', "        if False:"),
    ("P14 identity: flatdag0 may name bank flags", "            if n in fl:\n                bad.append(f\"flatdag0's argv names {n}", "            if False:\n                bad.append(f\"flatdag0's argv names {n}"),
    ("P15 bank: the shard count not enforced", "    if expect_shards is not None and n != expect_shards:", "    if False:"),
]
CHAIN_MUTANTS = [
    ("K01 C: gate -ge becomes -gt", '[ "$CF0" -ge "$CMIN" ] ||', '[ "$CF0" -gt "$CMIN" ] ||', ("cgate",)),
    ("K02 E: gate -ge becomes -gt", '[ "$EF0" -ge "$EMIN" ] ||', '[ "$EF0" -gt "$EMIN" ] ||', ("egate",)),
    ("K03 PREREG pin not enforced", '[ "$(sha256sum < "$PREREGF" | cut -c1-64)" = "$PREREGF_SHA" ] ||', "true ||", ("prereg",)),
    ("K04 rollouts at --limit logged", "--replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal", "--replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit logged", ("base",)),
    ("K05 --dump-heads dropped", "--limit causal --dump-ticks \"$OUTm/ticks\" --dump-heads --device cpu)", "--limit causal --dump-ticks \"$OUTm/ticks\" --device cpu)", ("base",)),
    ("K06 a run dropped from RUNS", "RUNS=(flatdag:0 flatdag0:0 flatdag:1 flatdag0:1)", "RUNS=(flatdag:0 flatdag0:0 flatdag:1)", ("base",)),
    ("K07 the FLAT-DAG tag collides with the banked FLAT tag", "if [ \"$1\" = flatdag ]; then echo FLATDAG; else echo FLATDAG0; fi", "if [ \"$1\" = flatdag ]; then echo FLAT; else echo FLATDAG0; fi", ("base",)),
    ("K08 rollouts on CUDA (cpu device dropped)", "--dump-heads --device cpu)", "--dump-heads)", ("base",)),
    ("K09 the second lane runs seed 0 again", "lane 1 & PB=$!", "lane 0 & PB=$!", ("REALRUN",)),
    ("K10 the tool-cfg guard dropped", '[ "$n_cfg" -ge 1 ] ||', "true ||", ("cfg",)),
    ("K11 the default box-2 watcher path altered", "C:/Users/Admin/box2_ship/logs/chain_v8f_flatdag.out}", "C:/Users/Admin/box2_ship/logs/chain_v8f_dagoacc.out}", ("base",)),
    ("K12 the scene flag flipped to frozen", "--nav deadreckon --scene logged --schedule periodic --limit causal", "--nav deadreckon --scene frozen --schedule periodic --limit causal", ("base",)),
    ("K13 the pull check called for the wrong arm", 'out=$("$PY" "$Hm/flatdag_t1_pullcheck.py" run "$ARTm" "$arm" "$s" 2>&1 < /dev/null); rc=$?\n    if [ $rc -eq 0 ]; then PULL_SHA', 'out=$("$PY" "$Hm/flatdag_t1_pullcheck.py" run "$ARTm" flatdag "$s" 2>&1 < /dev/null); rc=$?\n    if [ $rc -eq 0 ]; then PULL_SHA', ("REALRUN",)),
]


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
        target = {"read": "flatdag_read.py", "pull": "flatdag_t1_pullcheck.py", "chain": "flatdag_t1_chain.sh"}[kind]
        p = os.path.join(wd, target)
        txt = open(p, encoding="utf-8", newline="").read()
        if txt.count(anchor) != 1:
            return {"id": mid, "status": "ANCHOR ERROR", "count": txt.count(anchor), "seconds": time.time() - t0}
        open(p, "w", encoding="utf-8", newline="").write(txt.replace(anchor, repl))
        if scenarios == ("REALRUN",):
            return {"id": mid, "status": "UNVERIFIED", "why": "the defect sits in the real-run branch (lanes / wait loop), which a dry run never executes and which would need a rollout", "seconds": round(time.time() - t0, 1)}
        only = {"read": "L", "pull": "P", "chain": "C"}[kind]
        cmd = [PY, "-I", os.path.join(HERE, "flatdag_t1_selftest.py"), "--dir", wd, "--only", only] + (["--chain-scenarios", ",".join(scenarios)] if (kind == "chain" and scenarios) else [])
        env = {**os.environ, "FLATDAG_SELFTEST_INNER": "1"}
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
    # the unmutated control must be green first
    r = subprocess.run([PY, "-I", os.path.join(HERE, "flatdag_t1_selftest.py"), "--only", "L,P"], capture_output=True, text=True, env={**os.environ, "FLATDAG_SELFTEST_INNER": "1"}, timeout=900)
    ctrl_ok = r.returncode == 0
    print(f"CONTROL (unmutated, L+P): {'green' if ctrl_ok else 'RED'}  ({r.stdout.count('PASS ')} pass, {r.stdout.count('FAIL ')} fail)", flush=True)
    jobs = [("read", m[0], m[1], m[2], None) for m in READ_MUTANTS] + [("pull", m[0], m[1], m[2], None) for m in PULL_MUTANTS] + [("chain", m[0], m[1], m[2], m[3]) for m in CHAIN_MUTANTS]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--only", default="L,P,C")
    ap.add_argument("--chain-scenarios")
    ap.add_argument("--mutate", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--json")
    a = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    if a.mutate:
        return mutate(min(a.workers, 2), a.json)
    only = a.only.split(",")
    t0 = time.time()
    if "L" in only:
        run_logic(a.dir)
    if "P" in only:
        run_pull(a.dir)
    if "C" in only:
        run_chain(a.dir, a.chain_scenarios.split(",") if a.chain_scenarios else None)
    bad = [n for n, ok, _ in RES if not ok]
    print(f"SELFTEST: {len(RES) - len(bad)}/{len(RES)} checks as expected ({time.time() - t0:.0f} s)" + (f"; FAILED: {bad}" if bad else ""))
    if a.json:
        json.dump([{"check": n, "ok": ok, "detail": d} for n, ok, d in RES], open(a.json, "w"), indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
