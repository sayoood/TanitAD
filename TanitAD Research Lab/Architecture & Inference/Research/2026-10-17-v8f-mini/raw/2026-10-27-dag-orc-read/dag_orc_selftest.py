#!/usr/bin/env python3
"""dag_orc_selftest.py -- the proof for dag_orc_read.py (plain script, no pytest, CPU, no rollout, no network).

    python dag_orc_selftest.py [--reader PATH] [--quiet]        exit 0 iff every check is as expected

Everything runs on SYNTHETIC inputs (constructed Read-A tables, run records, files in a temp dir) plus three cheap real files (the PREREG, the frozen P0_RF0.json, the banked DAG - P10 table).
Groups:
  L  the PREREG pin: raw sha256, every table cell verbatim under its heading (one registered cell edited at a time), every coded phrase, the section-7 order
  B  the bar 2 * max(F_r, RF0_r, RF_r): each term binds somewhere, the factor, F only on ADE6, inclusive, direction
  O  the outcome table: EVERY row of section 7 reached by one constructed case; the guards, R-KEEP and R-NOBASE each flip the outcome; the committed next is the PREREG's cell
  P  the partition: all 2 x 2^18 = 524,288 cells match exactly one row, against (i) an independent derivation from the table's words and (ii) the draft's enumerate_cells.py logic; the PREREG's counts
  K  the run-record controls K-PAIR / K-INIT / K-REACH / K-BUILD / K-P0 / K-ZERO (identity) / K-RC (diff + evidence) and the registered-argv check, each with a deliberate regression
  F  RF0: the frozen file loads; a changed file, a wrong .sha256, a wrong schema, a missing row are refused; the mtime rule
  S  slot discovery; the dose cascade; the families report; compose_all on synthetic tables (RF0 / RF each bind)
  M  closing checks added after the first mutation run (UNPLACED on 0 / 2 matches, each prereg step alone, extra table row, REP argv, seed-1 floor, RC evidence, controls flag)
  R  real banked data: |D_s| = 14 / 13 and C_s from the banked DAG - P10 table
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
import time

sys.dont_write_bytecode = True
HERE = os.environ.get("DAGORC_HOME", os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
ap = argparse.ArgumentParser()
ap.add_argument("--reader", default=f"{HERE}/dag_orc_read.py")
ap.add_argument("--quiet", action="store_true")
ap.add_argument("--no-cells", action="store_true", help="skip group P (the 524,288-cell partition)")
ARGS = ap.parse_args()
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:       # noqa: BLE001
    pass

spec = importlib.util.spec_from_file_location("dag_orc_read_under_test", ARGS.reader)
R = importlib.util.module_from_spec(spec)
sys.modules["dag_orc_read_under_test"] = R
spec.loader.exec_module(R)

RESULTS = []
TMP = tempfile.mkdtemp(prefix=f"dagorc_st_{os.getpid()}_", dir=os.environ.get("DAGORC_TMP", "F:/TanitAD-scratch/dag_orc_read_tmp"))


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    if not ARGS.quiet or not cond:
        print(("ok   " if cond else "FAIL ") + name + ((" :: " + str(detail)[:200]) if (detail and not cond) else ""), flush=True)


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:       # noqa: BLE001
        return False
    return False


# ---------------------------------------------------------------------------------------------------------------- synthetic Read-A tables
def row(delta=0.0, sep=False, lo=None, hi=None):
    return {"delta": delta, "lo": delta - 0.5 if lo is None else lo, "hi": delta + 0.5 if hi is None else hi, "separated": sep}


def neutral():
    return {l: row() for l in R.LABELS}


RF0_SMALL = {l: 0.01 for l in R.LABELS}
RF_SMALL = {l: 0.01 for l in R.LABELS}


def base_seed(s):
    """Everything passes: C rows improved by 1.0 (separated) for X - CTRL, X - P10 neutral, DAG - P10 separated-worse on C and on the two exempt rows (as the banked table), guards neutral."""
    C = R.CRAWL_SET[s]
    dag_p10 = neutral()
    for l in C + R.EXEMPT:
        dag_p10[l] = row(+0.05, True)
    x_ctrl = neutral()
    for l in C:
        x_ctrl[l] = row(-1.0, True)
    return {"t": {"X_P10": neutral(), "X_CTRL": x_ctrl, "X_DAG": neutral(), "DAG_P10": dag_p10},
            "g": {"R-TLAT": row(), "R-TLON": row(), "T0_point": 0.0, "KEEP_point": 0.0}}


def compose_pair(mod=None, rf0=None, rf=None):
    """mod(case_by_seed) edits the two base seeds in place.  -> per_seed dict."""
    cs = {s: base_seed(s) for s in R.SEEDS}
    if mod:
        mod(cs)
    return {s: R.compose_seed(s, cs[s]["t"], cs[s]["g"], rf0 or RF0_SMALL, rf or RF_SMALL) for s in R.SEEDS}


def outcome_of(mod=None, rf0=None, rf=None, ctrl=True):
    ps = compose_pair(mod, rf0, rf)
    d = R.decide_arm(ctrl, ps)
    return d["OUTCOME"], d, ps


# ---------------------------------------------------------------------------------------------------------------- L: the PREREG pin
PREREG = f"{HERE}/PREREG_V8F_DAG_ORC.md"
text = open(PREREG, encoding="utf-8").read()
check("L1 the real PREREG passes prereg_problems", R.prereg_problems(PREREG) == [], R.prereg_problems(PREREG))
check("L2 the real PREREG hashes to PREREG_SHA256 (raw bytes)", hashlib.sha256(open(PREREG, "rb").read()).hexdigest() == R.PREREG_SHA256)
bad_file = os.path.join(TMP, "PREREG_changed.md")
open(bad_file, "w", encoding="utf-8", newline="\n").write(text.replace("F = 0.1207 m", "F = 0.1208 m", 1))
check("L3 a PREREG with one changed number is refused by the raw-byte pin", any("not the pinned" in p for p in R.prereg_problems(bad_file)))
check("L4 a missing PREREG is refused", R.prereg_problems(os.path.join(TMP, "nope.md")) != [])
caught = tot = 0
for key, tb in R.PREREG_CELLS.items():
    for i, rw in enumerate(tb["rows"]):
        for j in range(len(rw)):
            c2 = copy.deepcopy(R.PREREG_CELLS)
            c2[key]["rows"][i][j] = rw[j] + "X"
            tot += 1
            caught += bool(R.table_problems(text, c2))
check(f"L5 every one of the {tot} registered table cells, edited one at a time, is caught by table_problems", caught == tot and tot > 100, f"{caught}/{tot}")
c2 = copy.deepcopy(R.PREREG_CELLS)
c2["7."]["heading"] = c2["7."]["heading"] + " (edited)"
check("L6 a registered heading that is not in the PREREG is caught", bool(R.table_problems(text, c2)))
c2 = copy.deepcopy(R.PREREG_CELLS)
c2["4."]["rows"].append(["R-EXTRA", "x", "y"])
check("L7 a registered row the PREREG does not have is caught", bool(R.table_problems(text, c2)))
t_edit = text.replace("| **HARMS** | controls pass;", "| **HARMS** | controls pas;", 1)
check("L8 an edited section-7 line in the PREREG TEXT is caught under its heading", t_edit != text and bool(R.table_problems(t_edit)))
t_edit = text.replace("| **R-NEW** | no row of D_s **fires** | guard |\n", "", 1)
check("L9 a deleted section-4 row in the PREREG TEXT is caught", t_edit != text and bool(R.table_problems(t_edit)))
# a cell moved under another heading is not found there: swap the 7. table's heading text into section 4's place
t_edit = text.replace("## 4. The rows", "## 4. The rows (moved)", 1)
check("L10 a changed heading (the table no longer sits under its registered heading) is caught", bool(R.table_problems(t_edit)))
ph_missing = [ph for ph in R.PREREG_PHRASES if R.phrase_problems(text, (ph,))]
check("L11 every coded phrase is verbatim in the real PREREG", not ph_missing, ph_missing)
def edit_in_section(txt, key, phrase):
    sep = chr(10) + "## "
    parts = txt.split(sep)
    for i, part in enumerate(parts):
        if part.startswith(key):
            parts[i] = re.sub("[ \t\n]+".join(re.escape(w) for w in phrase.split(" ")), lambda m: phrase[:-1] + "~", part)
            return sep.join(parts)
    return txt


caught = sum(bool(R.phrase_problems(edit_in_section(text, ph[0], ph[1]), (ph,))) for ph in R.PREREG_PHRASES)
check(f"L12 every one of the {len(R.PREREG_PHRASES)} coded phrases, edited in the text, is caught", caught == len(R.PREREG_PHRASES), caught)
check("L13 section 7 has the 8 rows in the PREREG's order", R.OUTCOME_NAMES == ("REFUSED", "HARMS", "FIXED / ADOPTED", "FIXED / NOT ADOPTED", "PARTIAL / ADOPTED", "PARTIAL / NOT ADOPTED", "PARTIAL / ONE SEED", "NO EFFECT"), R.OUTCOME_NAMES)
check("L14 the adopting rows are exactly the two '/ ADOPTED' rows", set(R.ADOPTING) == {n for n in R.OUTCOME_NAMES if n.endswith("/ ADOPTED")} == {"FIXED / ADOPTED", "PARTIAL / ADOPTED"})
check("L15 LABELS are Read A's 19 rows, 3 STRATEGIC, TAC rows higher-is-better", len(R.LABELS) == 19 and len(set(R.LABELS)) == 19 and sum(R.FAMILY[l] == "STRATEGIC" for l in R.LABELS) == 3
      and [l for l in R.LABELS if R.DIRECTION[l] == "higher"] == ["TAC-LAT", "TAC-LON"])
check("L16 the constants: F 0.1207, F/2 0.0604, p 0.3 / 0.1, tolerance 0.02, REP seeds 10 / 11, DAG_s shas, crawl sets, exempt rows, |D_s|",
      R.F_ADE6 == 0.1207 and R.HALF_F == 0.0604 and R.P_A == 0.3 and R.P_B == 0.1 and R.REACH_TOL == 0.02 and R.REP_SEED == {0: 10, 1: 11}
      and R.DAG_CKPT_SHA[0].startswith("d186fd0bcfa7ee9f") and R.DAG_CKPT_SHA[1].startswith("1d12ec2eb825990d") and R.CTRL_CKPT_PREFIX == {0: "603b9151f5119886", 1: "f4234ac11854076b"}
      and R.CRAWL_SET[0] == ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share") and R.CRAWL_SET[1] == R.CRAWL_SET[0] + ("yaw-rate MAE",)
      and R.EXEMPT == ("accel MAE", "DK-VIOL") and R.DOMAIN_N == {0: 14, 1: 13} and R.SEEDS == (0, 1))

# ---------------------------------------------------------------------------------------------------------------- B: the bar
rf0 = {l: 0.0 for l in R.LABELS}
rfz = {l: 0.0 for l in R.LABELS}
r0, r1 = dict(rf0, **{"curvature MAE": 0.03}), dict(rfz, **{"curvature MAE": 0.02})
check("B1 RF0 binds: bar = 2 * RF0 when RF0 > RF and F = 0 (non-ADE row)", abs(R.bar_for("curvature MAE", r0, rfz) - 0.06) < 1e-15)
check("B2 RF binds: bar = 2 * RF when RF > RF0", abs(R.bar_for("curvature MAE", rf0, r1) - 0.04) < 1e-15 and abs(R.bar_for("curvature MAE", r0, r1) - 0.06) < 1e-15)
check("B3 the larger of RF0 and RF wins", abs(R.bar_for("curvature MAE", dict(rf0, **{"curvature MAE": 0.01}), r1) - 0.04) < 1e-15)
check("B4 F binds on ADE6 only: 2 * 0.1207 = 0.2414 with both floors below F", abs(R.bar_for("ADE6", dict(rf0, ADE6=0.0138), dict(rfz, ADE6=0.05)) - 0.2414) < 1e-12)
check("B5 on ADE6 a floor above F wins", abs(R.bar_for("ADE6", dict(rf0, ADE6=0.2), rfz) - 0.4) < 1e-12)
check("B6 F is NOT applied to a non-ADE row (two zero floors give bar 0)", R.bar_for("curvature MAE", rf0, rfz) == 0.0 and R.bar_for("M1 crawl share", rf0, rfz) == 0.0)
check("B7 the factor is 2 (not 1)", R.bar_for("speed MAE", dict(rf0, **{"speed MAE": 0.5}), rfz) == 1.0)
check("B8 floors enter as magnitudes (a negative value is |value|)", R.bar_for("speed MAE", dict(rf0, **{"speed MAE": -0.5}), rfz) == 1.0 and R.bar_for("speed MAE", rf0, dict(rfz, **{"speed MAE": -0.25})) == 0.5)
check("B9 clears is inclusive at the bar and not below it", R.clears("curvature MAE", row(-0.0600, True), 0.06) and not R.clears("curvature MAE", row(-0.0599, True), 0.06))
check("B10 clears needs separation", not R.clears("curvature MAE", row(-5.0, False), 0.06))
check("B11 clears needs the BETTER direction (worse never clears)", not R.clears("curvature MAE", row(+5.0, True), 0.06))
check("B12 direction: an agreement row (TAC-LAT) clears upward and not downward", R.clears("TAC-LAT", row(+0.05, True), 0.04) and not R.clears("TAC-LAT", row(-0.05, True), 0.04))
check("B13 a bar of exactly 0 (two bit-identical floors): clears iff separated-better", R.clears("curvature MAE", row(-1e-9, True), 0.0) and not R.clears("curvature MAE", row(-1.0, False), 0.0))
check("B14 sep_worse / sep_better are strict (a zero separated delta is neither)", not R.sep_worse("curvature MAE", row(0.0, True)) and not R.sep_better("curvature MAE", row(0.0, True)))
check("B15 sep_worse / sep_better respect direction", R.sep_worse("TAC-LON", row(-0.1, True)) and R.sep_better("TAC-LON", row(+0.1, True)) and R.sep_worse("ADE6", row(0.1, True)) and R.sep_better("ADE6", row(-0.1, True)))
check("B16 a NOT COMPUTED row cannot be judged (it raises)", raises(lambda: R.sep_worse("ADE6", {"status": "NOT COMPUTED"}), R.RowNotComputed) and raises(lambda: R.worsening("ADE6", None), R.RowNotComputed))
check("B17 fires is inclusive at the bar", R.fires("heading MAE", row(0.5, True), row(0.0, False), row(0.02, True), 0.02) and not R.fires("heading MAE", row(0.5, True), row(0.0, False), row(0.0199, True), 0.02))
check("B18 fires needs all three: X-P10 worse, DAG-P10 NOT worse, X-CTRL worse",
      not R.fires("heading MAE", row(0.0, False), row(0.0, False), row(0.5, True), 0.02) and not R.fires("heading MAE", row(0.5, True), row(0.4, True), row(0.5, True), 0.02)
      and not R.fires("heading MAE", row(0.5, True), row(0.0, False), row(0.5, False), 0.02))
check("B19 fires on an agreement row uses the negative direction", R.fires("TAC-LAT", row(-0.3, True), row(0.0, False), row(-0.1, True), 0.05) and not R.fires("TAC-LAT", row(-0.3, True), row(0.0, False), row(+0.1, True), 0.0))
dp = neutral()
dp["curvature MAE"] = row(0.05, True)
dp["accel MAE"] = row(0.05, True)
dp["speed MAE"] = row(-0.5, True)
dp["TAC-LAT"] = row(-0.2, True)
dom = R.domain(dp)
check("B20 domain drops DAG-P10 separated-worse rows (error and agreement) and the exempt rows, keeps improved/neutral ones",
      "curvature MAE" not in dom and "accel MAE" not in dom and "TAC-LAT" not in dom and "speed MAE" in dom and "DK-VIOL" not in dom and "ADE6" in dom and len(dom) == 19 - 4)
dp2 = neutral()
check("B21 exempt rows are excluded even when DAG-P10 is NOT worse on them", "accel MAE" not in R.domain(dp2) and "DK-VIOL" not in R.domain(dp2) and len(R.domain(dp2)) == 17)

# ---------------------------------------------------------------------------------------------------------------- O: the outcome table, every row by a constructed case
def set_rows(cs, s, pair, label_rows):
    for l, rw in label_rows.items():
        cs[s]["t"][pair][l] = rw


def fail_move(cs, s):                       # R-MOVE fails at seed s: one C row of X - CTRL is not separated-better
    cs[s]["t"]["X_CTRL"]["curvature MAE"] = row(-1.0, False)


def fail_crawl(cs, s):                      # R-CRAWL fails at s: a C row of X - P10 is separated-worse
    cs[s]["t"]["X_P10"]["curvature MAE"] = row(+0.3, True)


def fail_keep(cs, s):
    cs[s]["g"]["KEEP_point"] = 0.0605


def fail_nobase(cs, s):                     # X - P10 and X - DAG both separated-worse on a domain row, X - CTRL neutral (so R-NEW does not fire)
    cs[s]["t"]["X_P10"]["cross-track MAE"] = row(+0.4, True)
    cs[s]["t"]["X_DAG"]["cross-track MAE"] = row(+0.4, True)


def fail_tlat(cs, s):
    cs[s]["g"]["R-TLAT"] = row(-0.05, True)


def fail_tlon(cs, s):
    cs[s]["g"]["R-TLON"] = row(-0.05, True)


def fail_ade(cs, s):
    cs[s]["t"]["X_CTRL"]["ADE6"] = row(+0.3, True)


def fail_t0(cs, s):
    cs[s]["g"]["T0_point"] = 0.0605


def fail_new(cs, s):
    cs[s]["t"]["X_P10"]["heading MAE"] = row(+0.5, True)
    cs[s]["t"]["X_CTRL"]["heading MAE"] = row(+0.5, True)


def both(f):
    return lambda cs: [f(cs, s) for s in R.SEEDS]


def one(f, s):
    return lambda cs: f(cs, s)


o, d, _ = outcome_of()
check("O1 the baseline reads FIXED / ADOPTED and adopts", o == "FIXED / ADOPTED" and d["adopts"] and d["matching_rows_in_table_order"] == ["FIXED / ADOPTED"], o)
o, d, _ = outcome_of(ctrl=False)
check("O2 a failed control reads REFUSED (decided first; nothing else matches)", o == "REFUSED" and d["matching_rows_in_table_order"] == ["REFUSED"] and not d["adopts"])
GUARD_CASES = {"R-TLAT": fail_tlat, "R-TLON": fail_tlon, "R-ADE": fail_ade, "R-T0": fail_t0, "R-NEW": fail_new}
for g, f in GUARD_CASES.items():
    for s in R.SEEDS:
        o, d, ps = outcome_of(one(f, s))
        check(f"O3 guard {g} failing at seed {s} alone reads HARMS (and that guard is the failed one)", o == "HARMS" and ps[s]["guards"][g] is False and sum(not v for v in ps[s]["guards"].values()) == 1
              and not d["adopts"], (o, ps[s]["guards"]))
o, d, _ = outcome_of(lambda cs: [fail_tlat(cs, 0), fail_new(cs, 1), fail_keep(cs, 0)])
check("O4 HARMS wins over a failed adoption test and over a failed R-MOVE (precedence: guards before the rest)", o == "HARMS")
o, d, _ = outcome_of(both(fail_move))
check("O5 R-MOVE at neither seed (guards held) reads NO EFFECT", o == "NO EFFECT" and not d["adopts"], o)
for s in R.SEEDS:
    o, d, _ = outcome_of(one(fail_move, s))
    check(f"O6 R-MOVE at exactly one seed (fails at seed {s}) reads PARTIAL / ONE SEED and never adopts", o == "PARTIAL / ONE SEED" and not d["adopts"], o)
o, d, _ = outcome_of(lambda cs: [fail_move(cs, 0), fail_crawl(cs, 1)])
check("O7 R-MOVE at one seed with R-CRAWL failing at the OTHER seed is still PARTIAL / ONE SEED", o == "PARTIAL / ONE SEED", o)
o, d, _ = outcome_of(lambda cs: [fail_move(cs, 0), fail_crawl(cs, 0), fail_keep(cs, 1), fail_nobase(cs, 1)])
check("O8 one-seed R-MOVE never adopts whatever the adoption tests read", o == "PARTIAL / ONE SEED" and not d["adopts"], o)
for s in R.SEEDS:
    o, d, _ = outcome_of(one(fail_crawl, s))
    check(f"O9 R-MOVE at both seeds, R-CRAWL failing at seed {s}, adoption holding: PARTIAL / ADOPTED", o == "PARTIAL / ADOPTED" and d["adopts"], o)
    o, d, _ = outcome_of(lambda cs, s=s: [fail_crawl(cs, s), fail_keep(cs, 1 - s)])
    check(f"O10 same with R-KEEP failing at seed {1 - s}: PARTIAL / NOT ADOPTED", o == "PARTIAL / NOT ADOPTED" and not d["adopts"], o)
    o, d, _ = outcome_of(lambda cs, s=s: [fail_crawl(cs, s), fail_nobase(cs, s)])
    check("O11 same with R-NOBASE failing: PARTIAL / NOT ADOPTED", o == "PARTIAL / NOT ADOPTED" and not d["adopts"], o)
for tst, f in (("R-KEEP", fail_keep), ("R-NOBASE", fail_nobase)):
    for s in R.SEEDS:
        o, d, ps = outcome_of(one(f, s))
        check(f"O12 {tst} failing at seed {s} alone flips FIXED / ADOPTED to FIXED / NOT ADOPTED", o == "FIXED / NOT ADOPTED" and not d["adopts"] and ps[s]["flags"][tst] is False and ps[1 - s]["flags"][tst] is True, o)
o, d, _ = outcome_of(both(fail_keep))
check("O13 both adoption tests failing at both seeds: FIXED / NOT ADOPTED", o == "FIXED / NOT ADOPTED")
# every committed next is the PREREG's cell, with the arm-specific split of HARMS
for name in R.OUTCOME_NAMES:
    cond, nxt = R.table_row(name)
    want = next(r[2] for r in R.PREREG_CELLS["7."]["rows"][1:] if r[0] == name)
    check(f"O14 the committed next of {name} is the PREREG's cell", nxt == want and R.next_for_arm(name, "A").startswith(nxt[:10]) or name == "HARMS")
na, nb_ = R.next_for_arm("HARMS", "A"), R.next_for_arm("HARMS", "B")
check("O15 HARMS' committed next splits: A reads B, B goes to NEXT-CRAWL on DAG states", na.startswith("A: not adopted; read B") and "NEXT-CRAWL" not in na and nb_.startswith("B: not adopted") and "NEXT-CRAWL on DAG states" in nb_, (na, nb_))
blk = R.outcome_block(R.decide_arm(True, compose_pair()), "A")
check("O16 outcome_block carries the condition and the next of the row", blk["condition"] == R.table_row("FIXED / ADOPTED")[0] and blk["committed_next"] == R.table_row("FIXED / ADOPTED")[1] and blk["arm"] == "A")
check("O17 a reading matching no row (a gap) is UNPLACED, not a guessed row", R.decide_orc(True, {g: {0: True, 1: True} for g in R.GUARDS}, (True, True), (True, True), (True, True), (True, True))["OUTCOME"] == "FIXED / ADOPTED"
      and R.outcome_block({"OUTCOME": "UNPLACED", "matching_rows_in_table_order": [], "adopts": False, "flags": {}}, "A")["committed_next"] is None)

# ---------------------------------------------------------------------------------------------------------------- O (guards by value): the guard definitions on their edges
def only(f):
    return outcome_of(f)[2]


ps = only(lambda cs: cs[0]["g"].update({"T0_point": 0.0604}))
check("O18 R-T0 is inclusive at +F/2 = 0.0604", ps[0]["guards"]["R-T0"] is True)
ps = only(lambda cs: cs[0]["g"].update({"T0_point": 0.06040001}))
check("O19 R-T0 above 0.0604 fails", ps[0]["guards"]["R-T0"] is False and ps[1]["guards"]["R-T0"] is True)
ps = only(lambda cs: cs[0]["g"].update({"KEEP_point": 0.0604}))
check("O20 R-KEEP is inclusive at 0.0604 and fails above it", ps[0]["flags"]["R-KEEP"] is True and only(lambda cs: cs[0]["g"].update({"KEEP_point": 0.06041}))[0]["flags"]["R-KEEP"] is False)
ps = only(lambda cs: cs[0]["g"].update({"T0_point": 0.0605, "KEEP_point": 0.0}))
check("O21 R-T0 and R-KEEP read different points (T0 point vs the T1-kin ADE6 X - DAG point)", ps[0]["guards"]["R-T0"] is False and ps[0]["flags"]["R-KEEP"] is True)
ps = only(lambda cs: cs[0]["g"].update({"KEEP_point": 0.0605, "T0_point": 0.0}))
check("O22 and the other way round", ps[0]["guards"]["R-T0"] is True and ps[0]["flags"]["R-KEEP"] is False)
ps = only(lambda cs: cs[0]["g"].update({"R-TLAT": row(+0.2, True)}))
check("O23 R-TLAT: a separated IMPROVEMENT is not a failure (macro-F1 is higher-is-better)", ps[0]["guards"]["R-TLAT"] is True)
ps = only(lambda cs: cs[0]["g"].update({"R-TLAT": row(-0.2, False)}))
check("O24 R-TLAT: a non-separated drop is not a failure", ps[0]["guards"]["R-TLAT"] is True)
ps = only(lambda cs: cs[0]["t"]["X_CTRL"].update({"ADE6": row(-0.3, True)}))
check("O25 R-ADE: a separated improvement is not a failure; reads X - CTRL (not X - P10)", ps[0]["guards"]["R-ADE"] is True
      and only(lambda cs: cs[0]["t"]["X_P10"].update({"ADE6": row(+0.3, True)}))[0]["guards"]["R-ADE"] is True)
ps = only(lambda cs: cs[0]["t"]["X_CTRL"].update({"ADE6": row(+0.0001, True)}))
check("O26 R-ADE: any separated worsening fails (no bar on the guard)", ps[0]["guards"]["R-ADE"] is False)
# R-NEW: X - P10 worse, DAG - P10 not, X - CTRL worse >= bar; bar floor-aware
def new_case(delta_ctrl, rf0_=None, rf_=None, label="heading MAE", dag_worse=False, x_p10_worse=True):
    def mod(cs):
        cs[0]["t"]["X_P10"][label] = row(0.4 if x_p10_worse else 0.0, x_p10_worse)
        cs[0]["t"]["X_CTRL"][label] = row(delta_ctrl, True)
        if dag_worse:
            cs[0]["t"]["DAG_P10"][label] = row(0.4, True)
    return compose_pair(mod, rf0_, rf_)[0]
check("O27 R-NEW fires at exactly the bar (inclusive) and not below it", new_case(0.02)["guards"]["R-NEW"] is False and new_case(0.0199)["guards"]["R-NEW"] is True)
check("O28 R-NEW's bar rises with RF0 (0.03 -> bar 0.06) and with RF", new_case(0.05, dict(RF0_SMALL, **{"heading MAE": 0.03}))["guards"]["R-NEW"] is True
      and new_case(0.05, None, dict(RF_SMALL, **{"heading MAE": 0.03}))["guards"]["R-NEW"] is True and new_case(0.05)["guards"]["R-NEW"] is False)
check("O29 R-NEW does not fire on a row that DAG - P10 already had worse (outside the domain)", new_case(0.5, dag_worse=True)["guards"]["R-NEW"] is True)
check("O30 R-NEW does not fire when X - P10 is not separated-worse", new_case(0.5, x_p10_worse=False)["guards"]["R-NEW"] is True)
check("O31 R-NEW on ADE6 uses F: worsening 0.2413 does not fire, 0.2414 does", new_case(0.2413, label="ADE6")["guards"]["R-NEW"] is True and new_case(0.2414, label="ADE6")["guards"]["R-NEW"] is False)
check("O32 R-NEW on a non-ADE row has no F: worsening 0.0201 fires (bar 0.02)", new_case(0.0201)["guards"]["R-NEW"] is False)
def exempt_case(mod_):
    return compose_pair(mod_)[0]
ps0 = exempt_case(lambda cs: [cs[0]["t"]["X_P10"].update({"accel MAE": row(+0.9, True)}), cs[0]["t"]["X_CTRL"].update({"accel MAE": row(+0.9, True)}), cs[0]["t"]["DAG_P10"].update({"accel MAE": row(0.0, False)})])
check("O33 accel MAE is exempt from R-NEW (not in the domain) but REPORTED", ps0["guards"]["R-NEW"] is True and ps0["exempt_reported"]["accel MAE"]["X_minus_CTRL_separated_worse"] is True and "accel MAE" not in ps0["domain"])
ps0 = exempt_case(lambda cs: [cs[0]["t"]["X_P10"].update({"DK-VIOL": row(+0.9, True)}), cs[0]["t"]["X_DAG"].update({"DK-VIOL": row(+0.9, True)})])
check("O34 DK-VIOL is exempt from R-NOBASE but REPORTED", ps0["flags"]["R-NOBASE"] is True and ps0["exempt_reported"]["DK-VIOL"]["X_minus_P10_separated_worse"] is True)
def nb_case(delta_dag, rf0_=None, rf_=None, label="cross-track MAE", x_p10_worse=True, dag_sep=True, ctrl_worse=False):
    def mod(cs):
        cs[0]["t"]["X_P10"][label] = row(0.4 if x_p10_worse else 0.0, x_p10_worse)
        cs[0]["t"]["X_DAG"][label] = row(delta_dag, dag_sep)
        if ctrl_worse:
            cs[0]["t"]["X_CTRL"][label] = row(0.5, True)
    return compose_pair(mod, rf0_, rf_)[0]
check("O35 R-NOBASE: a violation at exactly the bar (inclusive), none below", nb_case(0.02)["flags"]["R-NOBASE"] is False and nb_case(0.0199)["flags"]["R-NOBASE"] is True)
check("O36 R-NOBASE needs BOTH X - P10 and X - DAG separated-worse", nb_case(0.5, x_p10_worse=False)["flags"]["R-NOBASE"] is True and nb_case(0.5, dag_sep=False)["flags"]["R-NOBASE"] is True)
check("O37 R-NOBASE's bar rises with RF0 and RF", nb_case(0.05, dict(RF0_SMALL, **{"cross-track MAE": 0.03}))["flags"]["R-NOBASE"] is True and nb_case(0.05, None, dict(RF_SMALL, **{"cross-track MAE": 0.03}))["flags"]["R-NOBASE"] is True
      and nb_case(0.05)["flags"]["R-NOBASE"] is False)
check("O38 R-NOBASE reads X - DAG for the worsening and X - P10 only for the sign (X - CTRL is irrelevant to it)", nb_case(0.5, ctrl_worse=False)["flags"]["R-NOBASE"] is False and nb_case(0.5, ctrl_worse=True)["flags"]["R-NOBASE"] is False)
c_row = compose_pair(lambda cs: cs[0]["t"]["X_CTRL"].update({"curvature MAE": row(-0.06, True)}), dict(RF0_SMALL, **{"curvature MAE": 0.03}))[0]
check("O39 R-MOVE: improvement 0.06 clears a bar of 0.06 (RF0 0.03) and not 0.0601", c_row["flags"]["R-MOVE"] is True
      and compose_pair(lambda cs: cs[0]["t"]["X_CTRL"].update({"curvature MAE": row(-0.0599, True)}), dict(RF0_SMALL, **{"curvature MAE": 0.03}))[0]["flags"]["R-MOVE"] is False)
check("O40 R-MOVE needs EVERY row of C_s (one weak row fails it); seed 1 has the yaw-rate row, seed 0 does not",
      compose_pair(lambda cs: cs[1]["t"]["X_CTRL"].update({"yaw-rate MAE": row(-0.001, True)}))[1]["flags"]["R-MOVE"] is False
      and compose_pair(lambda cs: cs[0]["t"]["X_CTRL"].update({"yaw-rate MAE": row(+9.0, True)}))[0]["flags"]["R-MOVE"] is True)
check("O41 R-CRAWL reads X - P10 on C_s only: a worse non-C row does not fail it; the yaw-rate row fails it at seed 1 only",
      compose_pair(lambda cs: cs[0]["t"]["X_P10"].update({"heading MAE": row(+0.4, True)}))[0]["flags"]["R-CRAWL"] is True
      and compose_pair(lambda cs: [cs[s]["t"]["X_P10"].update({"yaw-rate MAE": row(+0.4, True)}) for s in R.SEEDS])[1]["flags"]["R-CRAWL"] is False
      and compose_pair(lambda cs: [cs[s]["t"]["X_P10"].update({"yaw-rate MAE": row(+0.4, True)}) for s in R.SEEDS])[0]["flags"]["R-CRAWL"] is True)
check("O42 R-CRAWL: improved-but-separated X - P10 on a C row is not 'worse'; non-separated worse is not 'worse'", compose_pair(lambda cs: cs[0]["t"]["X_P10"].update({"curvature MAE": row(-0.4, True)}))[0]["flags"]["R-CRAWL"] is True
      and compose_pair(lambda cs: cs[0]["t"]["X_P10"].update({"curvature MAE": row(+0.4, False)}))[0]["flags"]["R-CRAWL"] is True)
check("O43 a NOT COMPUTED row in the judged set raises (the read is REFUSED, never a pass)",
      raises(lambda: compose_pair(lambda cs: cs[0]["t"]["X_CTRL"].update({"curvature MAE": {"status": "NOT COMPUTED"}})), R.RowNotComputed)
      and raises(lambda: compose_pair(lambda cs: cs[0]["t"]["X_P10"].update({"heading MAE": {"status": "NOT COMPUTED"}})), R.RowNotComputed)
      and raises(lambda: compose_pair(lambda cs: cs[0]["t"]["DAG_P10"].update({"S-POS@30": {"status": "NOT COMPUTED"}})), R.RowNotComputed))

# ---------------------------------------------------------------------------------------------------------------- P: the partition, all cells
if not ARGS.no_cells:
    src = ast.parse(open(f"{HERE}/ref/enumerate_cells.py", encoding="utf-8").read())
    ns = {"SEEDS": (0, 1), "GUARDS": ("R-TLAT", "R-TLON", "R-ADE", "R-T0", "R-NEW")}
    for node in src.body:
        if isinstance(node, ast.FunctionDef) and node.name == "outcome_rows":
            exec(compile(ast.Module([node], []), "enumerate_cells.py", "exec"), ns)
    draft_rows = ns["outcome_rows"]
    from collections import Counter
    cnt, bad_one, bad_ind, bad_draft, bad_adopt = Counter(), 0, 0, 0, 0
    GU = R.GUARDS
    t0 = time.time()
    for bits in itertools.product((True, False), repeat=19):
        ctrl = bits[0]
        gb = bits[1:11]
        held = {g: {s: gb[2 * i + s] for s in (0, 1)} for i, g in enumerate(GU)}
        rc, rm, kp, nb2 = bits[11:13], bits[13:15], bits[15:17], bits[17:19]
        d = R.decide_orc(ctrl, held, rc, rm, kp, nb2)
        # (i) independent derivation from the words of section 7
        if not ctrl:
            exp = "REFUSED"
        elif not all(gb):
            exp = "HARMS"
        else:
            n = sum(rm)
            if n == 0:
                exp = "NO EFFECT"
            elif n == 1:
                exp = "PARTIAL / ONE SEED"
            else:
                exp = ("FIXED" if all(rc) else "PARTIAL") + (" / ADOPTED" if (all(kp) and all(nb2)) else " / NOT ADOPTED")
        hit = [k for k, v in draft_rows(ctrl, held, rc, rm, kp, nb2).items() if v]
        bad_one += len(d["matching_rows_in_table_order"]) != 1
        bad_ind += d["OUTCOME"] != exp
        bad_draft += hit != [d["OUTCOME"]]
        cnt[d["OUTCOME"]] += 1
        if d["adopts"] != (d["OUTCOME"].endswith("/ ADOPTED")) or (d["adopts"] and not (all(rm) and all(kp) and all(nb2))):
            bad_adopt += 1
        if all(rm) and all(kp) and all(nb2) and all(gb) and ctrl and not d["adopts"]:
            bad_adopt += 1
    want = {"REFUSED": 262144, "HARMS": 261888, "FIXED / ADOPTED": 1, "FIXED / NOT ADOPTED": 15, "PARTIAL / ADOPTED": 3, "PARTIAL / NOT ADOPTED": 45, "PARTIAL / ONE SEED": 128, "NO EFFECT": 64}
    check("P1 all 524,288 cells match EXACTLY ONE row of section 7", bad_one == 0 and sum(cnt.values()) == 524288, (bad_one, sum(cnt.values())))
    check("P2 every cell equals the independent derivation from the words of section 7", bad_ind == 0, bad_ind)
    check("P3 every cell equals the draft enumerate_cells.py logic (exactly that one row)", bad_draft == 0, bad_draft)
    check("P4 the per-row cell counts are the PREREG's (262,144 / 261,888 / 1 / 15 / 3 / 45 / 128 / 64)", dict(cnt) == want, dict(cnt))
    check("P5 adoption is reachable ONLY with R-MOVE, R-KEEP and R-NOBASE at both seeds, and always then (guards held, controls pass), checked on every cell", bad_adopt == 0, bad_adopt)
    # the 16 (R-CRAWL, R-MOVE) cells with adoption projected away = DAGGER2 Amendment 1: FIXED 1 / PARTIAL 11 / NO EFFECT 4
    proj = Counter()
    for rc_, rm_ in itertools.product(itertools.product((True, False), repeat=2), repeat=2):
        d = R.decide_orc(True, {g: {0: True, 1: True} for g in GU}, rc_, rm_, (True, True), (True, True))["OUTCOME"]
        proj["FIXED" if d.startswith("FIXED") else "PARTIAL" if d.startswith("PARTIAL") else d] += 1
    check("P6 with adoption projected away the 16 guarded cells are DAGGER2 Amendment 1's: FIXED 1 / PARTIAL 11 / NO EFFECT 4", dict(proj) == {"FIXED": 1, "PARTIAL": 11, "NO EFFECT": 4}, dict(proj))
    if not ARGS.quiet:
        print(f"     (partition: {time.time() - t0:.1f} s)")

# ---------------------------------------------------------------------------------------------------------------- K: run-record controls, each with a deliberate regression
ctrl_argv = ["--arm", "M2b", "--seed", "0", "--grid", "pool4x10", "--out", "D:\\runs\\M2b_s0_dag20", "--init-ckpt", "D:\\runs\\M2b_s0_dag\\ckpt_last.pt", "--steps", "3000", "--lr", "1e-4",
             "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", "0", "--dagger-p", "0"]
real_argv_path = "E:/Projects/TanitAD-artifacts/v8f-dagger2/M2b_s0_dag20/argv.json"
if os.path.isfile(real_argv_path):
    ctrl_argv = json.load(open(real_argv_path, encoding="utf-8"))["argv"]
CF = R.parse_flags(ctrl_argv)


def argv_of(flags, out="D:\\runs\\X"):
    toks = []
    for k, occs in flags.items():
        for vals in occs:
            toks += [k] + list(vals)
    return toks


def flags_A(s, base, p=0.3, **kw):
    f = R.a_flags(base, s, p)
    f["--out"] = [["D:\\runs\\M2b_s%d_dagorc" % s]]
    for k, v in kw.items():
        f["--" + k.replace("_", "-")] = v
    return f


for s in R.SEEDS:
    cf = R.parse_flags(json.load(open(f"E:/Projects/TanitAD-artifacts/v8f-dagger2/M2b_s{s}_dag20/argv.json", encoding="utf-8"))["argv"]) if os.path.isfile(real_argv_path) else CF
    ok, _ = R.k_pair("A", s, flags_A(s, cf), cf)
    check(f"K1 A_s{s} = DAG2-0 + p 0.3 + the five law flags passes K-PAIR", ok and True)
    check(f"K2 DAG2-0's own argv does not pass as A (no lever)", not R.k_pair("A", s, cf, cf)[0])
    wrong = flags_A(s, cf)
    wrong["--ego-perturb-p"] = [["0.1"]]
    check("K3 A with p = 0.1 fails K-PAIR", not R.k_pair("A", s, wrong, cf)[0])
    for law in ("--ego-perturb-vscale", "--ego-perturb-vmax", "--ego-perturb-adrop", "--ego-perturb-yawrate-std", "--ego-perturb-seed"):
        w = flags_A(s, cf)
        del w[law]
        check(f"K4 A without {law} fails K-PAIR", not R.k_pair("A", s, w, cf)[0])
        w = flags_A(s, cf)
        w[law] = [["9.9"]]
        check(f"K5 A with {law} 9.9 fails K-PAIR", not R.k_pair("A", s, w, cf)[0])
    w = flags_A(s, cf)
    w["--ego-perturb-seed"] = [[str(1 - s)]]
    check("K6 A with the other seed's --ego-perturb-seed fails K-PAIR", not R.k_pair("A", s, w, cf)[0])
    w = flags_A(s, cf)
    w["--steps"] = [["2999"]]
    check("K7 A differing in one non-lever flag (--steps) fails K-PAIR", not R.k_pair("A", s, w, cf)[0])
    w = flags_A(s, cf)
    w["--extra"] = [["1"]]
    check("K8 A with an extra flag fails K-PAIR", not R.k_pair("A", s, w, cf)[0])
    w = flags_A(s, cf)
    w["--out"] = [["elsewhere"]]
    check("K9 --out is ignored by K-PAIR", R.k_pair("A", s, w, cf)[0])
    w = flags_A(s, cf)
    w["--ego-perturb-p"] = [["0.30"]]
    check("K10 numeric normalisation: 0.30 == 0.3", R.k_pair("A", s, w, cf)[0])
    rep = {k: v for k, v in cf.items()}
    rep["--seed"] = [[str(R.REP_SEED[s])]]
    rep["--out"] = [["D:\\runs\\M2b_s%d_rep" % s]]
    check(f"K11 REP_s{s} = DAG2-0 with --seed {R.REP_SEED[s]} passes", R.k_pair("REP", s, rep, cf)[0])
    w = dict(rep)
    w["--seed"] = [[str(s)]]
    check("K12 REP with the control's own seed fails", not R.k_pair("REP", s, w, cf)[0])
    w = dict(rep)
    w["--seed"] = [[str(R.REP_SEED[1 - s])]]
    check("K13 REP with the other seed's REP seed fails", not R.k_pair("REP", s, w, cf)[0])
    w = dict(rep)
    w["--ego-perturb-p"] = [["0.3"]]
    check("K14 REP carrying the lever fails", not R.k_pair("REP", s, w, cf)[0])
    w = dict(rep)
    w["--lr"] = [["1e-3"]]
    check("K15 REP differing in another flag fails", not R.k_pair("REP", s, w, cf)[0])
    fb = flags_A(s, cf, p=0.1)
    check("K16 B_s = A_s with p 0.1 passes (against A's record too)", R.k_pair("B", s, fb, cf, flags_A(s, cf))[0])
    check("K17 B with p 0.3 fails", not R.k_pair("B", s, flags_A(s, cf), cf, flags_A(s, cf))[0])
    w = flags_A(s, cf, p=0.1)
    w["--ego-perturb-vmax"] = [["7.0"]]
    check("K18 B with another law value fails", not R.k_pair("B", s, w, cf, flags_A(s, cf))[0])
    w = flags_A(s, cf, p=0.1)
    w["--ego-perturb-vmax"] = [["8.0"]]
    a_other = flags_A(s, cf)
    a_other["--ego-perturb-vmax"] = [["7.0"]]
    check("K19 B must also equal A's record except p (A's record differs in a law flag -> fails)", not R.k_pair("B", s, w, cf, a_other)[0])
    check("K20 an unknown arm is refused", not R.k_pair("Z", s, cf, cf)[0])
    reg = R.k_registered("CTRL", s, cf)
    check("K21 the registered fine-tune check passes on DAG2-0's banked argv", reg[0], reg[1])
    for nm, val in (("--steps", "2999"), ("--lr", "2e-4"), ("--warmup", "99"), ("--p-gt-start", "0.3"), ("--p-gt-end", "0.1"), ("--dagger-p", "0.1"), ("--seed", str(1 - s))):
        w = {k: v for k, v in cf.items()}
        w[nm] = [[val]]
        check(f"K22 a registered flag changed ({nm} {val}) fails the argv check", not R.k_registered("CTRL", s, w)[0])
    w = {k: v for k, v in cf.items()}
    w["--init-ckpt"] = [[f"D:\\runs\\M2b_s{s}\\ckpt_last.pt"]]
    check("K23 an init that is M2b_s (not DAG_s) fails", not R.k_registered("CTRL", s, w)[0])
    w = {k: v for k, v in cf.items()}
    del w["--dagger-p"]
    check("K24 a DEFAULT-disabling flag left unnamed fails (it must be named on the argv)", not R.k_registered("CTRL", s, w)[0])
    check("K25 A's registered check wants --ego-perturb-p 0.3 and REP's wants 0", R.k_registered("A", s, flags_A(s, cf))[0] and not R.k_registered("REP", s, flags_A(s, cf))[0] and not R.k_registered("A", s, cf)[0])
    # K-INIT
    good = [("summary:levers.init_ckpt.sha256", R.DAG_CKPT_SHA[s]), ("selfcheck:levers.init_ckpt.sha256", R.DAG_CKPT_SHA[s])]
    check("K26 K-INIT: every recorded init sha256 equal to DAG_s passes", R.k_init("A", s, good)[0])
    check("K27 K-INIT: one wrong record fails", not R.k_init("A", s, good + [("x", "0" * 64)])[0])
    check("K28 K-INIT: the other seed's DAG sha fails", not R.k_init("A", s, [("x", R.DAG_CKPT_SHA[1 - s])])[0])
    check("K29 K-INIT: no record fails (the init is unproven)", not R.k_init("A", s, [])[0])
# K-REACH
check("K30 K-REACH A: 24,720 / 82,457 (ORC's measured 0.2998) passes", R.k_reach("A", 0.3, 24720, 82457)[0])
check("K31 K-REACH A: the 0.02 band is inclusive at 0.32 and fails at 0.3201", R.k_reach("A", 0.3, 32000, 100000)[0] and not R.k_reach("A", 0.3, 32010, 100000)[0] and R.k_reach("A", 0.3, 28000, 100000)[0] and not R.k_reach("A", 0.3, 27990, 100000)[0])
check("K32 K-REACH B: 0.1 +- 0.02", R.k_reach("B", 0.1, 8200, 82457)[0] and not R.k_reach("B", 0.1, 24720, 82457)[0] and not R.k_reach("B", 0.1, 4000, 82457)[0])
check("K33 K-REACH REP: exactly 0 passes, 1 fails", R.k_reach("REP", 0.0, 0, 82457)[0] and not R.k_reach("REP", 0.0, 1, 82457)[0])
check("K34 K-REACH: built p must equal the argv p (A built at 0.0 fails: the lever never reached training)", not R.k_reach("A", 0.0, 24720, 82457)[0] and not R.k_reach("A", None, 24720, 82457)[0])
check("K35 K-REACH: eligible 0 / n > eligible / non-numbers fail", not R.k_reach("A", 0.3, 0, 0)[0] and not R.k_reach("A", 0.3, 5, 4)[0] and not R.k_reach("A", 0.3, None, 82457)[0])
check("K36 K-REACH: A perturbing NOTHING (n = 0) fails (the lever never fired)", not R.k_reach("A", 0.3, 0, 82457)[0])
# K-BUILD
sha_a, sha_b = "a" * 64, "b" * 64
bld = {"launch_commit": "0" * 40, "base_commit": "1d4f4fbb11922f4dd62b51871d898c03e0c88635", "files": {f: {"sha256_launch": sha_a, "sha256_base": sha_a} for f in R.BUILD_FILES}, "changed": []}
check("K37 K-BUILD: byte-identical files pass", R.k_build(bld)[0])
b2 = copy.deepcopy(bld)
b2["files"][R.BUILD_FILES[0]]["sha256_launch"] = sha_b
check("K38 K-BUILD: a changed file with no listed hunk fails", not R.k_build(b2)[0])
b2["changed"] = [{"file": R.BUILD_FILES[0], "hunk": "h1", "behind_flag": "--x", "default_off": True, "bit_identity_test": "PASS"}]
check("K39 K-BUILD: a changed file whose hunks sit behind default-off flags with PASSing tests passes", R.k_build(b2)[0])
for key, val in (("default_off", False), ("bit_identity_test", "FAIL"), ("behind_flag", "")):
    b3 = copy.deepcopy(b2)
    b3["changed"][0][key] = val
    check(f"K40 K-BUILD: a hunk with {key} = {val!r} fails", not R.k_build(b3)[0])
b3 = copy.deepcopy(bld)
b3["launch_commit"] = "abc"
check("K41 K-BUILD: a launch commit that is not 40-hex fails", not R.k_build(b3)[0])
b3 = copy.deepcopy(bld)
b3["base_commit"] = "deadbeef"
check("K42 K-BUILD: a base commit other than 1d4f4fbb fails", not R.k_build(b3)[0])
b3 = copy.deepcopy(bld)
del b3["files"][R.BUILD_FILES[2]]
check("K43 K-BUILD: a missing file record fails; so does no record at all", not R.k_build(b3)[0] and not R.k_build(None)[0])
# identity / RC controls
z = {l: {"delta": 0.0, "lo": 0.0, "hi": 0.0, "separated": False} for l in R.LABELS}
check("K44 identity: all rows exactly +0 [0, 0] passes", R.is_exact_zero(z))
z2 = copy.deepcopy(z)
z2["ADE6"]["delta"] = 1e-12
check("K45 identity: one row at 1e-12 fails", not R.is_exact_zero(z2))
z2 = copy.deepcopy(z)
z2["S-POS@80"]["hi"] = 0.0001
check("K46 identity: a nonzero CI end fails", not R.is_exact_zero(z2) and not R.is_exact_zero({}))
a_, b_ = copy.deepcopy(z), copy.deepcopy(z)
check("K47 RC: identical tables have 0 differences", R.rc_diffs(a_, b_) == [])
b_["ADE6"]["delta"] = 0.0002
check("K48 RC: a delta off by 2e-4 is a difference (tolerance 1e-4)", len(R.rc_diffs(a_, b_)) == 1)
b_ = copy.deepcopy(z)
b_["ADE6"]["delta"] = 0.00009
check("K49 RC: a delta off by 9e-5 is within tolerance", R.rc_diffs(a_, b_) == [])
b_ = copy.deepcopy(z)
b_["TAC-LAT"]["separated"] = True
check("K50 RC: a separated flip is a difference", len(R.rc_diffs(a_, b_)) == 1)
b_ = copy.deepcopy(z)
del b_["DK-HW"]
check("K51 RC: a missing row is a difference; NOT COMPUTED on one side only is a difference", len(R.rc_diffs(a_, b_)) == 1 and len(R.rc_diffs({"a": {"status": "NOT COMPUTED"}}, {"a": z["ADE6"]})) == 1)
check("K52 floors_from: max over seeds of |delta|, per row", R.floors_from({0: dict(z, ADE6=row(-0.3)), 1: dict(z, ADE6=row(0.2))})["ADE6"] == 0.3 and R.floors_from({0: z, 1: z})["ADE6"] == 0.0)
check("K53 floors_from of a table with a NOT COMPUTED row raises", raises(lambda: R.floors_from({0: dict(z, ADE6={"status": "NOT COMPUTED"}), 1: z}), R.RowNotComputed))

# ---------------------------------------------------------------------------------------------------------------- F: RF0
RF0_REAL = f"{HERE}/P0_RF0.json"
probs, rf0v = R.rf0_problems(RF0_REAL)
check("F1 the frozen P0_RF0.json loads (sha256 pinned, .sha256 beside it, 19 rows, RF0 = max over seeds)", probs == [] and rf0v is not None and list(rf0v) == list(R.LABELS), probs)
raw = open(RF0_REAL, "rb").read()
tmp = os.path.join(TMP, "rf0")
os.makedirs(tmp, exist_ok=True)


def put(name, data, sha=None):
    p = os.path.join(tmp, name)
    open(p, "wb").write(data)
    open(os.path.splitext(p)[0] + ".sha256", "w", encoding="utf-8", newline="\n").write(f"{sha or hashlib.sha256(data).hexdigest()}  {name}\n")
    return p


check("F2 a byte-identical copy loads (the pin is on content, not on the path)", R.rf0_problems(put("copy.json", raw))[0] == [])
ch = bytearray(raw)
ch[raw.index(b"0.0138")] = ord("9")
p = put("changed.json", bytes(ch))
pr_, v_ = R.rf0_problems(p)
check("F3 a CHANGED RF0 file is refused by the pin even with a matching .sha256 beside it", v_ is None and any("pins" in x for x in pr_), pr_)
p = put("shamismatch.json", raw, sha="0" * 64)
check("F4 a .sha256 that does not match the file is refused", R.rf0_problems(p)[1] is None)
p = put("nosha.json", raw)
os.remove(os.path.splitext(p)[0] + ".sha256")
check("F5 a missing .sha256 is refused; so is a missing file", R.rf0_problems(p)[1] is None and R.rf0_problems(os.path.join(tmp, "absent.json"))[1] is None)
d = json.loads(raw)
for nm, fn in (("schema", lambda d: d.__setitem__("schema", "p0-rf0-v2")), ("a missing row", lambda d: (d["rows"].pop("DK-HW"), d["RF0"].pop("DK-HW"))),
               ("RF0 not the max over seeds", lambda d: d["RF0"].__setitem__("ADE6", 0.0001)), ("a negative RF0", lambda d: d["RF0"].__setitem__("ADE6", -0.01)),
               ("a failed P-0 control", lambda d: d["controls"]["IDENTITY:s0"].__setitem__("PASS", False)), ("no controls", lambda d: d.__setitem__("controls", {})),
               ("a non-finite RF0", lambda d: d["RF0"].__setitem__("ADE6", float("nan")))):
    d2 = copy.deepcopy(d)
    fn(d2)
    data = json.dumps(d2, indent=1).encode()
    p = put("v_" + nm.replace(" ", "_") + ".json", data)
    pr_, v_ = R.rf0_problems(p, pin=hashlib.sha256(data).hexdigest())
    check(f"F6 a self-consistent file with {nm} is refused on content", v_ is None and pr_, pr_)
d2 = copy.deepcopy(d)
data = json.dumps(d2, indent=1).encode()
pr_, v_ = R.rf0_problems(put("v_ok.json", data), pin=hashlib.sha256(data).hexdigest())
check("F7 the same content re-serialised passes when its own sha is the pin (the content checks accept a valid file)", pr_ == [] and v_ is not None, pr_)
check("F8 the frozen RF0 values: ADE6 0.0138, curvature 0.0263, M1 0.0048, L2 0.0068, yaw-rate 0.5535", rf0v["ADE6"] == 0.0138 and rf0v["curvature MAE"] == 0.0263 and rf0v["M1 crawl share"] == 0.0048
      and rf0v["L2 curvature MAE (GT>=5.2)"] == 0.0068 and rf0v["yaw-rate MAE"] == 0.5535)
# K-P0 mtime rule
froz = put("mt.json", raw)
roll = os.path.join(tmp, "rollout_a.npz")
open(roll, "wb").write(b"x")
now = time.time()
os.utime(froz, (now - 100, now - 100))
os.utime(roll, (now - 50, now - 50))
os.utime(os.path.splitext(froz)[0] + ".sha256", (now - 100, now - 100))
check("F9 K-P0: RF0 older than every rollout file passes", R.k_p0(froz, [roll], check_inputs=False)[0] == [])
os.utime(roll, (now - 200, now - 200))
check("F10 K-P0: a rollout file older than RF0 fails (RF0 must predate every rollout)", R.k_p0(froz, [roll], check_inputs=False)[0] != [])
os.utime(roll, (now - 100, now - 100))
check("F11 K-P0: a rollout file with the same mtime fails (strictly later is required)", R.k_p0(froz, [roll], check_inputs=False)[0] != [])
check("F12 K-P0 on a changed RF0 fails", R.k_p0(os.path.join(tmp, "changed.json"), [], check_inputs=False)[0] != [])
# K-RC evidence
ev = os.path.join(TMP, "rc_ev.json")
me = hashlib.sha256(open(ARGS.reader, "rb").read()).hexdigest()
open(ev, "w").write(json.dumps({"reader_sha256": me, "differences": 0, "PASS": True}))
# rc_evidence_ok hashes ITS OWN file (abspath(__file__) of the module under test)
check("K54 K-RC evidence from this reader with 0 differences passes", R.rc_evidence_ok(ev)[0])
open(ev, "w").write(json.dumps({"reader_sha256": "0" * 64, "differences": 0, "PASS": True}))
check("K55 K-RC evidence produced by ANOTHER reader (any edit invalidates it) fails", not R.rc_evidence_ok(ev)[0])
open(ev, "w").write(json.dumps({"reader_sha256": me, "differences": 1, "PASS": False}))
check("K56 K-RC evidence with a difference fails; so does a missing file", not R.rc_evidence_ok(ev)[0] and not R.rc_evidence_ok(os.path.join(TMP, "none.json"))[0])
open(ev, "w").write(json.dumps({"reader_sha256": me, "differences": 0, "PASS": False}))
check("K57 K-RC evidence with PASS false fails even at 0 listed differences", not R.rc_evidence_ok(ev)[0])

# ---------------------------------------------------------------------------------------------------------------- S: slots, cascade, families, compose_all
def mk_dump(path, n):
    os.makedirs(path, exist_ok=True)
    for i in range(n):
        open(os.path.join(path, f"ep{i:04d}.npz"), "wb").write(b"x")


def mk_entry(base, arm, n_dump=141, omit=()):
    os.makedirs(base, exist_ok=True)
    e = {"train_dir": base}
    for f in ("argv.json", "summary.json", "selfcheck.json", "tok.json"):
        open(os.path.join(base, f), "w").write("{}")
    e["token"] = os.path.join(base, "tok.json")
    for k in ("t1", "t1_logged"):
        d = os.path.join(base, k)
        mk_dump(d, n_dump)
        open(os.path.join(base, f"{k}.json"), "w").write("{}")
        open(os.path.join(base, f"{k}.npz"), "w").write("x")
        e[k] = {"json": os.path.join(base, f"{k}.json"), "ticks": os.path.join(base, f"{k}.npz"), "dump_dir": d}
    for k in ("t0", "t0_fresh"):
        for f in ("json", "npz"):
            open(os.path.join(base, f"{k}.{f}"), "w").write("x")
        e[k] = {"json": os.path.join(base, f"{k}.json"), "npz": os.path.join(base, f"{k}.npz")}
    for o in omit:
        k, _, f = o.partition(".")
        if f:
            os.remove(e[k][f])
        else:
            e.pop(k, None)
    return e


slot_dir = os.path.join(TMP, "slots")
man = {"arms": {"A": {"0": mk_entry(f"{slot_dir}/A0", "A"), "1": mk_entry(f"{slot_dir}/A1", "A", n_dump=140)}, "CTRL": {"0": mk_entry(f"{slot_dir}/C0", "CTRL")},
                "DAG": {"0": mk_entry(f"{slot_dir}/D0", "DAG")}}}
sl = R.discover(man)
check("S1 discover: a complete entry is PRESENT", sl["A"][0]["state"] == "PRESENT" and sl["CTRL"][0]["state"] == "PRESENT" and sl["DAG"][0]["state"] == "PRESENT", sl["A"][0])
check("S2 discover: a dump dir with 140 of 141 episodes is PARTIAL", sl["A"][1]["state"] == "PARTIAL" and any("140" in m for m in sl["A"][1]["missing"]))
check("S3 discover: an arm absent from the manifest is MISSING (no entry), per seed", sl["REP"][0]["state"] == "MISSING" and sl["B"][1]["state"] == "MISSING" and sl["CTRL"][1]["state"] == "MISSING")
man2 = copy.deepcopy(man)
os.remove(man2["arms"]["A"]["0"]["t0"]["npz"])
check("S4 discover: a missing T0 eval file makes the slot PARTIAL", R.discover(man2)["A"][0]["state"] == "PARTIAL")
man2 = copy.deepcopy(man)
man2["arms"]["A"]["0"].pop("t1_logged")
check("S5 discover: a missing logged lane makes a new arm PARTIAL (K4 needs it)", R.discover(man2)["A"][0]["state"] == "PARTIAL")
man2 = copy.deepcopy(man)
man2["arms"]["DAG"]["0"].pop("t1_logged")
check("S6 discover: the banked DAG slot needs no logged lane and no T0", R.discover(man2)["DAG"][0]["state"] == "PRESENT")
man2 = copy.deepcopy(man)
os.remove(man2["arms"]["A"]["0"]["token"])
check("S7 discover: a missing launch-gate token makes a new arm PARTIAL", R.discover(man2)["A"][0]["state"] == "PARTIAL")
man2 = copy.deepcopy(man)
os.remove(os.path.join(man2["arms"]["A"]["0"]["train_dir"], "summary.json"))
check("S8 discover: a missing summary.json makes it PARTIAL", R.discover(man2)["A"][0]["state"] == "PARTIAL")
check("S9 slot_summary shape", R.slot_summary(sl)["A"] == {"s0": "PRESENT", "s1": "PARTIAL"})
# cascade
def blk(name):
    return R.outcome_block({"OUTCOME": name, "matching_rows_in_table_order": [name], "adopts": name in R.ADOPTING, "flags": {}}, "A")


okc = True
for a_row, b_row in itertools.product(R.OUTCOME_NAMES[1:], R.OUTCOME_NAMES[1:] + (None,)):
    c = R.cascade(blk(a_row), blk(b_row) if b_row else None)
    if a_row != "HARMS":
        okc &= c["final_arm"] == "A" and c["final"] == a_row
    elif b_row is None:
        okc &= c["final_arm"] == "A" and c["final"] == "HARMS" and "PENDING" in c["b_status"]
    else:
        okc &= c["final_arm"] == "B" and c["final"] == b_row
check("S10 the dose cascade: B decides ONLY after A's HARMS; otherwise B is beside; A HARMS with no B is PENDING", okc)
c = R.cascade(blk("NO EFFECT"), blk("FIXED / ADOPTED"))
check("S11 a B read beside an A that is not HARMS never promotes B over A", c["final_arm"] == "A" and "beside" in c["b_status"] and "never pooled" in c["b_status"])
fr = R.families_report({"X-CTRL": neutral(), "X-P10": neutral()})
check("S12 families_report: four families, tier and estimator stamps, STRATEGIC not paused", list(fr) == ["LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"] and fr["STRATEGIC"]["paused"] is False
      and "OPEN loop" in fr["LATERAL"]["tier"] and "bootstrap" in fr["TACTICAL"]["estimator"] and set(fr["LONGITUDINAL"]["rows"]) == {l for l in R.LABELS if R.FAMILY[l] == "LONGITUDINAL"})
# compose_all on a synthetic res: RF0 / RF each bind
def mk_res(rf0_, rf_, x_ctrl_curv):
    cs = {s: base_seed(s) for s in R.SEEDS}
    for s in R.SEEDS:
        cs[s]["t"]["X_CTRL"]["curvature MAE"] = row(x_ctrl_curv, True)
    return {"xs": ["A"], "tabs": {"A_P10": {s: cs[s]["t"]["X_P10"] for s in R.SEEDS}, "A_CTRL": {s: cs[s]["t"]["X_CTRL"] for s in R.SEEDS}, "A_DAG": {s: cs[s]["t"]["X_DAG"] for s in R.SEEDS},
                                  "DAG_P10": {s: cs[s]["t"]["DAG_P10"] for s in R.SEEDS}}, "g": {"A": {s: cs[s]["g"] for s in R.SEEDS}}, "floors": {"RF0": rf0_, "RF": rf_}}


ca = lambda r: R.compose_all(r, True)["A"]["decision"]["OUTCOME"]      # noqa: E731
check("S13 compose_all: with both floors small the baseline FIXED / ADOPTED", ca(mk_res(RF0_SMALL, RF_SMALL, -0.05)) == "FIXED / ADOPTED")
check("S14 compose_all: RF0 alone raises the bar over the improvement (NO EFFECT)", ca(mk_res(dict(RF0_SMALL, **{"curvature MAE": 0.03}), RF_SMALL, -0.05)) == "NO EFFECT")
check("S15 compose_all: RF alone raises the bar over the improvement (NO EFFECT)", ca(mk_res(RF0_SMALL, dict(RF_SMALL, **{"curvature MAE": 0.03}), -0.05)) == "NO EFFECT")
check("S16 compose_all: RF0 is read as RF0 and RF as RF (a swapped pair with different values still binds either way)",
      ca(mk_res(dict(RF0_SMALL, **{"curvature MAE": 0.03}), dict(RF_SMALL, **{"curvature MAE": 0.001}), -0.05)) == "NO EFFECT"
      and ca(mk_res(dict(RF0_SMALL, **{"curvature MAE": 0.001}), dict(RF_SMALL, **{"curvature MAE": 0.03}), -0.05)) == "NO EFFECT"
      and ca(mk_res(dict(RF0_SMALL, **{"curvature MAE": 0.001}), dict(RF_SMALL, **{"curvature MAE": 0.001}), -0.05)) == "FIXED / ADOPTED")
# the pure helpers around the heavy path
check("S17 parse_flags keeps every value of every occurrence", R.parse_flags(["--a", "1", "2", "--b", "--a", "3"]) == {"--a": [["1", "2"], ["3"]], "--b": [[]]})
check("S18 a_flags replaces --ego-perturb-p and adds the five law flags with the seed", R.a_flags({"--ego-perturb-p": [["0"]], "--lr": [["1e-4"]]}, 1)["--ego-perturb-seed"] == [["1"]]
      and R.a_flags({"--ego-perturb-p": [["0"]]}, 1)["--ego-perturb-p"] == [["0.3"]] and set(R.a_flags({}, 0)) == {"--ego-perturb-p", "--ego-perturb-seed", *R.LAW_A})

# ---------------------------------------------------------------------------------------------------------------- M: the closing checks the first mutation run asked for (each kills a mutant that survived it)
# decide_orc: exactly one matching row, else UNPLACED -- two matches AND zero matches
_orig_rows = R.outcome_rows
try:
    R.outcome_rows = lambda *a, **k: {"HARMS": True, "NO EFFECT": True}
    two = R.decide_orc(True, {g: {0: True, 1: True} for g in R.GUARDS}, (True, True), (True, True), (True, True), (True, True))["OUTCOME"]
    R.outcome_rows = lambda *a, **k: {}
    zero = R.decide_orc(True, {g: {0: True, 1: True} for g in R.GUARDS}, (True, True), (True, True), (True, True), (True, True))["OUTCOME"]
finally:
    R.outcome_rows = _orig_rows
check("M1 decide_orc: two matching rows -> UNPLACED (never the first of them)", two == "UNPLACED", two)
check("M2 decide_orc: zero matching rows -> UNPLACED", zero == "UNPLACED", zero)
# prereg_problems: a file that is its OWN pin still goes through the table check AND the phrase check
t_tab = text.replace("| **HARMS** | controls pass;", "| **HARMS** | controls pas;", 1)
PH_PROSE = next(ph for ph in R.PREREG_PHRASES if edit_in_section(text, ph[0], ph[1]) != text and not R.table_problems(edit_in_section(text, ph[0], ph[1])))        # a phrase that lives in prose, not in a registered table
t_phr = edit_in_section(text, PH_PROSE[0], PH_PROSE[1])
check("M3 the table-edited copy is caught by table_problems and NOT by phrase_problems (so only the table step can catch it)", t_tab != text and bool(R.table_problems(t_tab)) and not R.phrase_problems(t_tab))
check("M4 the phrase-edited copy is caught by phrase_problems and NOT by table_problems (so only the phrase step can catch it)", t_phr != text and bool(R.phrase_problems(t_phr)) and not R.table_problems(t_phr))
for nm, tt, pat in (("tab", t_tab, "M5"), ("phr", t_phr, "M6")):
    pth = os.path.join(TMP, f"PREREG_{nm}.md")
    open(pth, "w", encoding="utf-8", newline="\n").write(tt)
    own = hashlib.sha256(open(pth, "rb").read()).hexdigest()
    check(f"{pat} prereg_problems(pin = the copy's own sha256) still reports the {nm}-edit (the pin is not the only gate)", bool(R.prereg_problems(pth, pin=own)) and not any("not the pinned" in p for p in R.prereg_problems(pth, pin=own)))
good_own = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
check("M7 prereg_problems with the real file's own sha256 as the pin is clean", R.prereg_problems(PREREG, pin=good_own) == [])
# table_problems: an EXTRA row after the registered last row is a difference (the cell compare alone cannot see it)
lines_ = text.split(chr(10))
i_ne = next(i for i, ln in enumerate(lines_) if ln.startswith("| **NO EFFECT** |"))
t_extra_end = chr(10).join(lines_[:i_ne + 1] + ["| **EXTRA** | a | b |"] + lines_[i_ne + 1:])
check("M8 table_problems: one extra row after the registered last row of section 7 is caught (row count)", t_extra_end != text and any("rows, the reader codes" in p for p in R.table_problems(t_extra_end)), R.table_problems(t_extra_end))
# phrase_problems: a phrase registered under a section the PREREG lacks is a problem, not a crash and not a pass
check("M9 phrase_problems: a phrase under a missing section is reported", R.phrase_problems(text, (("99.", "no such section"),)) != [])
# k_registered: REP names seed 10 / 11 and p 0; A names p 0.3
def cf_of(s):
    pth = f"E:/Projects/TanitAD-artifacts/v8f-dagger2/M2b_s{s}_dag20/argv.json"
    return R.parse_flags(json.load(open(pth, encoding="utf-8"))["argv"]) if os.path.isfile(pth) else CF


for s in R.SEEDS:
    cfs = cf_of(s)
    rep = {k: v for k, v in cfs.items()}
    rep["--seed"] = [[str(R.REP_SEED[s])]]
    check(f"M10 k_registered REP_s{s}: --seed {R.REP_SEED[s]}, p 0 passes", R.k_registered("REP", s, rep)[0], R.k_registered("REP", s, rep)[1])
    w = {k: v for k, v in rep.items()}
    w["--seed"] = [[str(s)]]
    check(f"M11 k_registered REP_s{s} with the BASE seed {s} fails (REP is a different seed)", not R.k_registered("REP", s, w)[0])
    w = {k: v for k, v in rep.items()}
    w["--ego-perturb-p"] = [["0.3"]]
    check(f"M12 k_registered REP_s{s} with p 0.3 fails (REP carries no perturbation)", not R.k_registered("REP", s, w)[0])
    check(f"M13 k_registered A_s{s} keeps the base seed {s}, not the REP seed", R.k_registered("A", s, flags_A(s, cfs))[0] and not R.k_registered("A", s, dict(flags_A(s, cfs), **{"--seed": [[str(R.REP_SEED[s])]]}))[0])
# floors_from: a binding seed 1
zz = {l: row() for l in R.LABELS}
check("M14 floors_from: seed 1 binding (|d| larger than seed 0's) sets the floor", R.floors_from({0: dict(zz, ADE6=row(0.2)), 1: dict(zz, ADE6=row(-0.3))})["ADE6"] == 0.3)
# rc_evidence_ok: differences must be exactly 0 even when PASS is claimed
open(ev, "w").write(json.dumps({"reader_sha256": me, "differences": 1, "PASS": True}))
check("M15 K-RC evidence claiming PASS with 1 difference fails", not R.rc_evidence_ok(ev)[0])
open(ev, "w").write(json.dumps({"reader_sha256": me, "differences": 0, "PASS": True}))
check("M16 K-RC evidence with 0 differences and PASS passes again", R.rc_evidence_ok(ev)[0])
# compose_all passes the controls flag through: a failed control makes the arm REFUSED
check("M17 compose_all(res, False) -> REFUSED (the controls flag is the caller's, never hard-coded)", R.compose_all(mk_res(RF0_SMALL, RF_SMALL, -0.05), False)["A"]["decision"]["OUTCOME"] == "REFUSED"
      and R.compose_all(mk_res(RF0_SMALL, RF_SMALL, -0.05), True)["A"]["decision"]["OUTCOME"] == "FIXED / ADOPTED")

# ---------------------------------------------------------------------------------------------------------------- R: real banked data
banked =json.load(open("E:/Projects/TanitAD-artifacts/dagreread/raw/READ_A.json", encoding="utf-8"))["READ_A"]["rows"]
for s in R.SEEDS:
    rows_s = banked[f"s{s}"]
    dom_s = R.domain({l: rows_s[l] for l in R.LABELS})
    check(f"R1 the banked DAG - P10 table gives |D_{s}| = {R.DOMAIN_N[s]} (PREREG section 4)", len(dom_s) == R.DOMAIN_N[s], (len(dom_s), dom_s))
    derived = sorted(l for l, r in rows_s.items() if r.get("separated_worse") and l not in R.EXEMPT)
    check(f"R2 the banked separated-worse rows (minus accel / DK-VIOL) are the PREREG's crawl set C_{s}", derived == sorted(R.CRAWL_SET[s]), derived)
    check(f"R3 every Read-A label of the banked table has the reader's direction", all(R.DIRECTION[l] == rows_s[l].get("direction") for l in R.LABELS))

# ---------------------------------------------------------------------------------------------------------------- report
shutil.rmtree(TMP, ignore_errors=True)
n_bad = sum(1 for _, ok, _ in RESULTS if not ok)
print(f"SELFTEST {len(RESULTS) - n_bad}/{len(RESULTS)} checks as expected" + (f"; FAILED: {[n for n, ok, _ in RESULTS if not ok][:8]}" if n_bad else ""))
sys.exit(1 if n_bad else 0)
