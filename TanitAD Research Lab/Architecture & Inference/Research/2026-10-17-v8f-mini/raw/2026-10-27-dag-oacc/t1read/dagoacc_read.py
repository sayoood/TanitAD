#!/usr/bin/env python3
"""dagoacc_read.py -- the dev-box T1-kin read of PREREG_V8F_DAG_OACC (claim row H-V8F-DAG-OACC-1).  BUILT AND SELF-TESTED, NOT RUN ON REAL DAG-OACC OUTPUTS.

PREREG (binding, nothing here changes it): Project Steering/PREREG_V8F_DAG_OACC.md @ 95063711, raw-blob sha256 PREREG_OACC_SHA256 below.  Its section 2 rows and section 4 outcome table are checked CELL BY CELL against the pinned
text (static check).  Read A is PREREG_V8F_DAG_REREAD section 2 VERBATIM (rows, D1 / D2 / D3, crawl rows, readers, estimator), its controls C1-C4 are C-O4 (section 4 there, pinned by raw sha256 too).

NOTHING IS RE-DERIVED.  Every number comes from a function of an existing reader, run on these arms:
  DAG-OACC  M2b_s<S>_dagoacc: DAG_s<S>'s argv + --w-acc W, nothing else; ONE new T1-kin rollout per seed                                  (manifest arm DAGOACC)
  DAG       the CONTROL, BANKED (v8f-dagger-t1/ dumps and ticks; read, never re-run)                                                      (manifest control DAG)
  P10       the kinematic floor, carried in every dump (`fl`)
  layer A   Read A's driver code over the pinned reader sources: comps / L1 / L2 / M1 rows (readA.rows_block), TAC + DK rows (t1kf_read_a1.prep / rows), the Read A table (build_report.collect / verdict / outcome), as
            dag2_read.ReadA (imported unchanged, sha256-pinned).  Pairs: DAG-OACC - P10 (Read A, LON-CLEAR), DAG-OACC - DAG (PRIMARY-ACC, KEEP, beside), DAG - P10 (re-computed from DAG's banked dumps and required to
            equal the banked READ_A.json: control RC-DAG = C1).
  layer G   dag_read (pinned): the run-record controls (SUMMARY, INIT-CKPT, ARGV, DAGGER-REACHED, GATE-TOKEN), the T1-HEADS pipeline (Pack: S-POS vs CTRV, K1 K2 K3 K5 K6 K7 of the new rollout).  There is NO bank here
            (DAG's iteration-1 bank is BANKED; its sha256 identity is C-O1) and no K-B control.
The only new logic is the PREREG's composition and it is the part the self-test mutates: PRIMARY-ACC, KEEP, LON-CLEAR, the D1 / D2 / D3 flags, C-O1's one-lever rule, C-O2's gate form (r2), C-O3's weight check and the
outcome partition (decide()).

THE DECISION (section 4, applied in order, first match wins): REFUSED -> NO EFFECT -> COSTS POSITION -> DOMINANT -> LONGITUDINAL FIXED -> NOT FIXED.
AMENDMENT 1 (A1-2026-10-27; AMEND1.md, raw sha256 AMEND1_SHA256 below, future repo path AMEND1_REPO_PATH) IS CODED AND PINNED.  The cell enumeration (enumerate_cells.py) found two defects in section 4 before any DAG-OACC row existed:
255 of 524,288 cells matched no row, and two clauses admitted two readings.  The amendment closes both; rows, bars, controls, the order, the weight rule and W do not change:
  (1) LONGITUDINAL FIXED := PRIMARY-ACC and KEEP pass; LON-CLEAR at both seeds; not DOMINANT.  (The old clause "a lateral or crawl row still separated-worse", and with it the at-one-seed / at-both-seeds question (a), is dropped: it decides nothing.)
  (2) DOMINANT := PRIMARY-ACC and KEEP pass; Read A's D1, D2 and D3 at both seeds; and Read A's crawl rows pass (L2 and M1 not separated-worse at either seed).  L2 is a LATERAL row inside D2, so D2 already carries it; the
      crawl share M1 is outside D2 and is the clause this reader adds.  A cell with D1-D3 held and M1 separated-worse reads LONGITUDINAL FIXED and is reported "D1-D3 held, crawl open" (`crawl_open`).
  (3) LONGITUDINAL FIXED's committed next is rewritten (DAGGER2 read HARMS, so DAG2-OACC is no longer the committed route): see OUTCOME_TABLE.
  The result names, per seed, which of DOMINANT's clauses failed (`dominant_clauses_failed_per_seed`: D1 / D2 / D3 / crawl).  The UNAMENDED section-4 outcome (the pre-amendment logic, verbatim: decide_unamended) is computed
  beside as `outcome_unamended`; it is descriptive and decides nothing.  The reader REFUSES unless AMEND1.md is byte-identical to the pin and still contains, verbatim, the conditions and the committed next it codes.
  (c) STRATEGIC S-POS rows are against CTRV WITHIN a run (no paired DAG-OACC - DAG instrument exists): in the DAG-OACC - DAG table they are NOT COMPUTED; in the DAG-OACC - P10 table they come from the new rollout's
      T1-HEADS pack.  A longitudinal row can never be NOT COMPUTED (all six are paired instruments); if one were, LON-CLEAR cannot pass and control LON-ROWS-COMPUTED REFUSES.
  (d) UNPLACED is a DEFENSIVE branch only.  After Amendment 1 every cell is placed in exactly one row (enumerate_cells.py: 0 unplaced of 524,288 cells, decide() equal to an independent table on every cell); reaching UNPLACED
      would be a BUG in decide() (rc 3, written, no outcome word invented), never a result.
  (e) The T0 "four families" beside is the box-2 eval json's `raw.families` block (DAG-OACC and DAG side by side, leaf by leaf, descriptive: no paired T0 four-family instrument exists, as dag2_read says).
  (f) T1-HEADS K4 (tick 0 == the arm's own fresh-age T0) needs a logged-bucket rollout and a fresh T0 eval that PREREG section 5 does not schedule: K4 is reported NOT RUN; K1 K2 K3 K5 K6 K7 and GRID of the new
      rollout ARE controls here (a rollout that fails its own instrument checks is not read).

    python dagoacc_read.py --manifest runs.json --out DAGOACC_READ.json --tool-root <tree> ... (see build_parser)   rc 0 outcome written | 2 REFUSED (nothing written) | 3 UNPLACED (written)
    python dagoacc_read.py --static-check ...     pins, PREREG cells and static inputs only; touches no run output
    python dagoacc_read.py --preflight ...        the run-record controls C-O1 / C-O2 / C-O3 on whatever DAG-OACC runs are pulled (no eval_* and no rollout output is opened)
    python dagoacc_read.py --rehearsal ...        the WHOLE row pipeline and decide() on BANKED data: DAG-0 stands in for DAG-OACC against DAG (labelled REHEARSAL, NOT a DAG-OACC result)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
H2 = HERE.replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts"
# ---------------------------------------------------------------------------------------------------------------- PREREG_V8F_DAG_OACC (committed numbers)
PREREG_OACC_ID = "Project Steering/PREREG_V8F_DAG_OACC.md @ 95063711"
PREREG_OACC_SHA256 = "bd722b7a1a9ae419fdd6ebaf437f5657396664670ea4b4f1074420a80e7f169d"
PREREG_REREAD_SHA256 = "7acaa4dd25373d554d9d2cdc68d0c9b94687167e343cceec0e8cf0939d7657fa"       # PREREG_V8F_DAG_REREAD.md @ 9b91b654 (with Amendment 1): Read A's rows (section 2) and C1-C4 (section 4)
AMEND1_ID = "A1-2026-10-27 (Amendment 1 to PREREG_V8F_DAG_OACC.md, written by the Master Mind before any DAG-OACC T1-kin rollout or value)"
AMEND1_SHA256 = "44d7e879e5349e5791d1a6ad9ff18559cfd72ea63ce3ef634dcde16e9a33373e"                 # AMEND1.md, raw bytes
AMEND1_REPO_PATH = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-27-dagoacc-amend1/AMEND1.md"     # where it will live in the repo (this reader reads the E: copy, pinned by the sha256 above)
D2_READ_SHA256 = "1425e8a3895c442b9841f3bcd6fd0893021d583e06485fde171b04bfa73fbc20"              # dag2_read.py as built for PREREG_V8F_DAGGER2 (imported unchanged)
OFFLINE_WACC_SHA256 = "f30aa20978eb5d54727eb5feeb804c13dfe9d6fb7f213ebfd813e1143010a901"         # wacc_offline_DAG_s0.json: C-O3's offline reference (w_acc 49.6969)
R2_ASSERT_SHA256 = "e1890346b92980a8f4842e488a01ab6c00c26ebf81d2a15f00664f331f130e8f"            # box2_ship/gate_oaf_assert_r2.py: the form of C-O2's gate check (wacc_problems copied from it)
FITPLANS_SHA256 = "71a57ac94208fefc4e284811623c065b003aa3807fd092bafc5941c7227ecbab"             # fitplans_DAG_s0.npz: DAG_s0's banked fit windows
CLAIM_ROW = "H-V8F-DAG-OACC-1"
SEEDS = (0, 1)
F_M, TWO_F, HALF_F = 0.1207, 0.2415, 0.0604       # F (MODEL_REGISTRY section 17), 2F = D1's bar, F/2 = KEEP's bound
N_BOOT = 2000
W_REL_TOL = 1e-4                                   # C-O3
DAG_CKPT_SHA = {0: "d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1", 1: "1d12ec2eb825990d57dcc45dda50b5178d1d733961b47ab0e3d50e6155fa39ef"}
M2B_CKPT_SHA = {0: "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", 1: "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
BANK1_SHA = {0: "197a68bb540be5a7d157942e9157e1c8e60ca60d82b1ca4fb4fcb68edccfaa8f", 1: "36c7aea964dafe9423253aa479c2fc6cdfc93848395b593315cba812f5f04d4d"}
DAG_ARGV_SHA = {0: "50e9d01d3dee881367a4eb1433903b05cc093c83d7bccc32a518e2290d9f3ce3", 1: "38a591e3e5a05c6f72669a268aaf99591af167758296720d650c46ebba4541b3"}
ROLLOUT_PATH_FLAGS = ("--ckpt", "--tag", "--out-dir", "--dump-ticks")      # the only flags in which a new rollout's argv may differ from the registered one
SPOS_KEY = "S-POS vs CTRV (ticks>=1, within this run)"
SPOS_ROWS = ("S-POS@30", "S-POS@50", "S-POS@80")
ACC_LABEL, ADE_LABEL, M1_LABEL = "accel MAE", "ADE6", "M1 crawl share"
# Read A's rows, family and direction (PREREG_V8F_DAG_REREAD section 2 = build_report.SPEC; a control compares this with the extracted SPEC)
READ_A_ROWS = (
    ("PRIMARY", "ADE6", "lower"),
    ("LONGITUDINAL", "speed MAE", "lower"), ("LONGITUDINAL", "along-track MAE", "lower"), ("LONGITUDINAL", "accel MAE", "lower"),
    ("LONGITUDINAL", "DK-VIOL", "lower"), ("LONGITUDINAL", "DK-HW", "lower"), ("LONGITUDINAL", "DK-TG", "lower"),
    ("LATERAL", "heading MAE", "lower"), ("LATERAL", "yaw-rate MAE", "lower"), ("LATERAL", "curvature MAE", "lower"),
    ("LATERAL", "cross-track MAE", "lower"), ("LATERAL", "L1 lateral-accel MAE", "lower"), ("LATERAL", "L2 curvature MAE (GT>=5.2)", "lower"),
    ("TACTICAL", "TAC-LAT", "higher"), ("TACTICAL", "TAC-LON", "higher"),
    ("STRATEGIC", "S-POS@30", "lower"), ("STRATEGIC", "S-POS@50", "lower"), ("STRATEGIC", "S-POS@80", "lower"),
)
LON_ROWS = tuple(l for f, l, _ in READ_A_ROWS if f == "LONGITUDINAL")
LAT_ROWS = tuple(l for f, l, _ in READ_A_ROWS if f == "LATERAL")
FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
CRAWL_ROWS = ("L2 curvature MAE (GT>=5.2)", M1_LABEL)           # the crawl rows (REREAD section 2): L2 is in D2, M1 is not
DIRECTIONS = {l: d for _, l, d in READ_A_ROWS}
DIRECTIONS[M1_LABEL] = "lower"

# The PREREG's section 2 rows and section 4 table AS PRINTED (markdown bold markers are markup, not text); a control checks every cell against the pinned file.
ROWS_2 = (
    ("PRIMARY-ACC", "accel MAE, DAG-OACC − DAG", "< 0 and separated, at both seeds"),
    ("KEEP", "ADE6, DAG-OACC − DAG", "upper bound ≤ +F/2 = +0.0604 m, at both seeds"),
    ("LON-CLEAR", "every LONGITUDINAL row of Read A, DAG-OACC − P10", "none separated-worse, at both seeds"),
    ("Read A", "`PREREG_V8F_DAG_REREAD.md` §2 verbatim (rows, D1 / D2 / D3, crawl rows, readers, estimator), arm = DAG-OACC", "as there"),
    ("beside", "DAG-OACC − DAG on every Read-A row; the four families at T0; S-POS against CTRV; R-T0 (T0 ADE6, DAG-OACC − DAG)", "descriptive"),
)
OUTCOME_TABLE_PREREG = (
    ("REFUSED", "a control fails", "fix and re-run"),
    ("NO EFFECT", "PRIMARY-ACC fails at ≥ 1 seed",
     "the accel error is not reachable through O's loss on DAG's states at this capacity. Next: OACC's NO EFFECT next, an O-capacity read, under its own PREREG"),
    ("COSTS POSITION", "PRIMARY-ACC passes; KEEP fails at ≥ 1 seed",
     "not adopted. The trade has then appeared on three routes. Next: the same term restricted to the plan's along-track component, so that lateral smoothing cannot cost position in curves (HYPOTHESIS), under its own PREREG"),
    ("DOMINANT", "PRIMARY-ACC and KEEP pass; Read A's D1, D2 and D3 at both seeds",
     "the dominance claim at T1-kin over the kinematic floor, on all four families, at both seeds, is registered for DAG-OACC. Signal the PI. Evidence videos are re-rendered on DAG-OACC. FLAT-DAG stays the attribution read"),
    ("LONGITUDINAL FIXED", "PRIMARY-ACC and KEEP pass; LON-CLEAR at both seeds; a lateral or crawl row still separated-worse",
     "the term is the trained longitudinal lever and is adopted. It composes with DAgger iteration 2: next is DAG2's recipe plus this W (DAG2-OACC), read on Read A, as `PREREG_V8F_DAGGER2`'s FIXED / PARTIAL rows commit"),
    ("NOT FIXED", "PRIMARY-ACC and KEEP pass; a longitudinal row still separated-worse at ≥ 1 seed",
     "next: HACC's rule re-fitted on DAG-OACC (`PREREG_V8F_DAG_HACC`'s recipe, zero training), read with KEEP against DAG-OACC"),
)       # section 4 of PREREG_V8F_DAG_OACC.md AS PRINTED (what the static check compares with the pinned PREREG, cell by cell)
# Amendment 1 rewrites exactly these rows (name -> (condition, committed next)); every other row is the PREREG's, unchanged.
_LF_NEXT = ("the term is adopted and DAG-OACC becomes the base; next is the lever of the row still open. "
            "If the open rows are crawl rows (L2, curvature, yaw-rate, M1; the list PREREG_V8F_DAG_REREAD §5 uses), the lever is DAG-ORC (PREREG_V8F_DAG_ORC.md), re-based on DAG-OACC by an amendment of its own PREREG before any DAG-ORC data. "
            "Otherwise, the failing family's lever, under its own PREREG.")
AMEND1_ROWS = {
    "DOMINANT": ("PRIMARY-ACC and KEEP pass; Read A's D1, D2 and D3 at both seeds; and Read A's crawl rows pass (L2 and M1 not separated-worse at either seed)", None),      # None: the PREREG's committed next, unchanged
    "LONGITUDINAL FIXED": ("PRIMARY-ACC and KEEP pass; LON-CLEAR at both seeds; not DOMINANT", _LF_NEXT),
}
OUTCOME_TABLE = tuple((n, AMEND1_ROWS[n][0] if n in AMEND1_ROWS else c, (AMEND1_ROWS[n][1] or nx) if n in AMEND1_ROWS else nx) for n, c, nx in OUTCOME_TABLE_PREREG)       # the table IN FORCE (Amendment 1 applied)
OUTCOME_NAMES = tuple(r[0] for r in OUTCOME_TABLE_PREREG)
CONTROL_IDS_3 = ("C-O1", "C-O2", "C-O3", "C-O4")
UNPLACED_TEXT = ("UNPLACED: no row of section 4 (as amended by Amendment 1) matches this reading.  Amendment 1 makes the table complete (enumerate_cells.py: 0 unplaced cells of 524,288), so this branch is DEFENSIVE and must be "
                 "unreachable: reaching it is a BUG in decide(), not a result.  No outcome word is invented; stop and report it.")


# ---------------------------------------------------------------------------------------------------------------- pure logic (no numpy, no torch): this is what the self-test mutates
def sep_worse(row, direction="lower"):
    """separated AND the point on the wrong side.  'lower' = an error row (positive delta is worse); 'higher' = an agreement / F1 row (negative delta is worse)."""
    return bool(row["separated"] and ((row["delta"] > 0) if direction == "lower" else (row["delta"] < 0)))


def sep_better(row, direction="lower"):
    return bool(row["separated"] and ((row["delta"] < 0) if direction == "lower" else (row["delta"] > 0)))


def primary_acc(row):
    """PRIMARY-ACC at one seed: accel MAE, DAG-OACC - DAG: < 0 (strict) and separated."""
    return bool(row["separated"] and row["delta"] < 0)


def keep_ok(row):
    """KEEP at one seed: ADE6, DAG-OACC - DAG: the UPPER bound of the interval <= +F/2 = +0.0604 m (inclusive).  A missing or NaN bound fails."""
    hi = row.get("hi")
    return bool(isinstance(hi, (int, float)) and not isinstance(hi, bool) and hi <= HALF_F)


def lon_status(table, directions=DIRECTIONS):
    """LON-CLEAR at one seed: every LONGITUDINAL row of Read A (DAG-OACC - P10) NOT separated-worse.  A row that is missing or NOT COMPUTED cannot be judged and is returned under 'not_computed' (it never counts as clear)."""
    notc = [l for l in LON_ROWS if l not in table or table[l].get("status") == "NOT COMPUTED"]
    worse = [l for l in LON_ROWS if l not in notc and sep_worse(table[l], directions.get(l, "lower"))]
    return {"clear": not worse and not notc, "worse": worse, "not_computed": notc}


def latcrawl_worse(table, directions=DIRECTIONS):
    """The LATERAL rows (heading, yaw-rate, curvature, cross-track, L1, L2) and the crawl share M1 that are separated-worse at one seed (DAG-OACC - P10)."""
    return [l for l in LAT_ROWS + (M1_LABEL,) if l in table and table[l].get("status") != "NOT COMPUTED" and sep_worse(table[l], directions.get(l, "lower"))]


def d_flags(table, directions=DIRECTIONS):
    """Read A's D1 / D2 / D3 at one seed, PREREG_V8F_DAG_REREAD section 2: D1 ADE6 separated-better with point <= -2F; D2 no row of the table separated-worse (M1 is the crawl share, not a table row; a NOT COMPUTED row
    fails D2); D3 each of the four families has at least one separated-better row."""
    ade = table[ADE_LABEL]
    d1 = bool(sep_better(ade, "lower") and ade["delta"] <= -TWO_F)
    rows = {l: r for l, r in table.items() if r.get("family") != "crawl"}
    worse = sorted(l for l, r in rows.items() if r.get("status") != "NOT COMPUTED" and sep_worse(r, directions.get(l, "lower")))
    notc = sorted(l for l, r in rows.items() if r.get("status") == "NOT COMPUTED")
    fams = {f: sorted(l for l, r in rows.items() if r.get("family") == f and r.get("status") != "NOT COMPUTED" and sep_better(r, directions.get(l, "lower"))) for f in FAMILIES}
    return {"D1": d1, "D2": bool(not worse and not notc), "D3": all(len(v) >= 1 for v in fams.values()), "separated_worse_rows": worse, "not_computed": notc, "separated_better_rows_by_family": fams}


def crawl_rows_fail(tables):
    """The crawl rows (REREAD section 2) FAIL iff L2 or M1 is separated-worse at >= 1 seed.  ``tables`` = {seed: Read A table of DAG-OACC - P10}."""
    return any(sep_worse(tables[s][l], DIRECTIONS[l]) for s in SEEDS for l in CRAWL_ROWS if l in tables[s] and tables[s][l].get("status") != "NOT COMPUTED")


def seed_flags_of(pa_row, keep_row, table, directions=DIRECTIONS):
    """One seed's flags from its rows: PRIMARY-ACC, KEEP, LON-CLEAR, D1-D3, lateral / crawl rows separated-worse."""
    d = d_flags(table, directions)
    lon = lon_status(table, directions)
    lc = latcrawl_worse(table, directions)
    return {"PA": primary_acc(pa_row), "KEEP": keep_ok(keep_row), "D1": d["D1"], "D2": d["D2"], "D3": d["D3"], "LON_CLEAR": lon["clear"], "LATCRAWL_WORSE": bool(lc)}, {"d": d, "lon": lon, "latcrawl_worse": lc}


def dominant_clauses_failed(sf, m1_worse):
    """Per seed: which of DOMINANT's per-seed clauses fail.  D1 / D2 / D3 are Read A's; 'crawl' = the crawl share M1 separated-worse (an L2 failure is a LATERAL row inside D2 and is named there)."""
    return {f"s{s}": [n for n, ok in (("D1", sf[s]["D1"]), ("D2", sf[s]["D2"]), ("D3", sf[s]["D3"]), ("crawl", not m1_worse[s])) if not ok] for s in SEEDS}


def decide(controls_failed, sf, m1_worse):
    """Section 4 AS AMENDED BY AMENDMENT 1, applied in order, coded cell by cell.  controls_failed: list of failed control ids; sf[s] = seed_flags_of()[0] for both seeds; m1_worse[s]: bool, the crawl share M1
    separated-worse at seed s (REQUIRED for both seeds: DOMINANT's crawl clause reads it).  Returns the outcome with EVERY matching row in table order (the first wins).  Outside REFUSED the rows are mutually exclusive
    (LONGITUDINAL FIXED carries `not dominant` itself, it does not rely on the order).  No matching row = UNPLACED, a defensive branch that Amendment 1 makes unreachable (enumerate_cells.py)."""
    if tuple(sorted(sf)) != SEEDS:
        raise ValueError(f"decide needs exactly seeds {SEEDS}")
    if not isinstance(m1_worse, dict) or tuple(sorted(m1_worse)) != SEEDS:
        raise ValueError(f"decide needs m1_worse for exactly seeds {SEEDS}: DOMINANT's crawl clause reads it")
    pa = all(sf[s]["PA"] for s in SEEDS)
    kp = all(sf[s]["KEEP"] for s in SEEDS)
    d123 = all(sf[s]["D1"] and sf[s]["D2"] and sf[s]["D3"] for s in SEEDS)
    crawl_pass = not any(m1_worse[s] for s in SEEDS)
    dominant = pa and kp and d123 and crawl_pass
    lon_all = all(sf[s]["LON_CLEAR"] for s in SEEDS)
    match = {"REFUSED": bool(controls_failed),
             "NO EFFECT": not pa,
             "COSTS POSITION": pa and not kp,
             "DOMINANT": dominant,
             "LONGITUDINAL FIXED": pa and kp and lon_all and not dominant,
             "NOT FIXED": pa and kp and not lon_all}
    matching = [n for n in OUTCOME_NAMES if match[n]]
    out = matching[0] if matching else "UNPLACED"
    crawl_open = bool(pa and kp and d123 and not crawl_pass)
    return {"OUTCOME": out, "matching_rows_in_table_order": matching, "amendment": "A1-2026-10-27",
            "crawl_open": crawl_open,
            "dominant_clauses_failed_per_seed": dominant_clauses_failed(sf, m1_worse),
            "flags": {"PRIMARY-ACC_per_seed": {f"s{s}": sf[s]["PA"] for s in SEEDS}, "KEEP_per_seed": {f"s{s}": sf[s]["KEEP"] for s in SEEDS}, "LON-CLEAR_per_seed": {f"s{s}": sf[s]["LON_CLEAR"] for s in SEEDS},
                      "D1_per_seed": {f"s{s}": sf[s]["D1"] for s in SEEDS}, "D2_per_seed": {f"s{s}": sf[s]["D2"] for s in SEEDS}, "D3_per_seed": {f"s{s}": sf[s]["D3"] for s in SEEDS},
                      "lateral_or_crawl_row_separated_worse_per_seed": {f"s{s}": sf[s]["LATCRAWL_WORSE"] for s in SEEDS}, "crawl_share_M1_separated_worse_per_seed": {f"s{s}": bool(m1_worse[s]) for s in SEEDS},
                      "controls_failed": list(controls_failed)}}


def decide_unamended(controls_failed, sf, m1_worse=None):
    """The section-4 logic AS WRITTEN IN THE PREREG before Amendment 1 (the old decide(), with its default reading: LONGITUDINAL FIXED needs a lateral or crawl row separated-worse at >= 1 seed; DOMINANT is Read A's D1-D3
    alone), verbatim.  DESCRIPTIVE ONLY: it is reported beside the outcome (`outcome_unamended`) and decides nothing.  A cell no old row matches reads UNPLACED, as it did."""
    if tuple(sorted(sf)) != SEEDS:
        raise ValueError(f"decide_unamended needs exactly seeds {SEEDS}")
    pa = all(sf[s]["PA"] for s in SEEDS)
    kp = all(sf[s]["KEEP"] for s in SEEDS)
    d123 = all(sf[s]["D1"] and sf[s]["D2"] and sf[s]["D3"] for s in SEEDS)
    lon_all = all(sf[s]["LON_CLEAR"] for s in SEEDS)
    lat = any(sf[s]["LATCRAWL_WORSE"] for s in SEEDS)
    match = {"REFUSED": bool(controls_failed),
             "NO EFFECT": not pa,
             "COSTS POSITION": pa and not kp,
             "DOMINANT": pa and kp and d123,
             "LONGITUDINAL FIXED": pa and kp and lon_all and lat,
             "NOT FIXED": pa and kp and not lon_all}
    matching = [n for n in OUTCOME_NAMES if match[n]]
    return {"OUTCOME": matching[0] if matching else "UNPLACED", "matching_rows_in_table_order": matching}


def outcome_block(d):
    out = d["OUTCOME"]
    if out == "UNPLACED":
        return {**d, "condition": None, "committed_next": None, "note": UNPLACED_TEXT}
    cond, nxt = next((c, n) for nm, c, n in OUTCOME_TABLE if nm == out)
    blk = {**d, "condition": cond, "committed_next": nxt}
    if out == "LONGITUDINAL FIXED":
        fails = d["dominant_clauses_failed_per_seed"]
        blk["reported_as"] = ("D1–D3 held, crawl open" if d["crawl_open"] else
                              "not DOMINANT: " + "; ".join(f"seed {k[1:]} fails {'/'.join(v) if v else 'nothing'}" for k, v in fails.items()))
    return blk


def outcome_rc(dec):
    return {"UNPLACED": 3, "REFUSED": 2}.get(dec["OUTCOME"], 0)


def unamended_beside(controls_failed, sf, m1_worse, amended_outcome):
    """The UNAMENDED section-4 outcome beside the amended one.  Descriptive: nothing reads it back."""
    u = decide_unamended(controls_failed, sf, m1_worse)
    return {"OUTCOME": u["OUTCOME"], "matching_rows_in_table_order": u["matching_rows_in_table_order"], "differs_from_amended": u["OUTCOME"] != amended_outcome, "descriptive_only": True,
            "note": "the section-4 table as printed in PREREG_V8F_DAG_OACC.md before Amendment 1, computed beside; it decides nothing (the amended outcome is the result)"}


def argv_flags(argv):
    out, i = {}, 0
    while i < len(argv):
        t = str(argv[i])
        if t.startswith("--"):
            j, vals = i + 1, []
            while j < len(argv) and not str(argv[j]).startswith("--"):
                vals.append(str(argv[j]))
                j += 1
            out.setdefault(t, []).append(vals)
            i = j
        else:
            i += 1
    return out


def drop_flag(argv, names):
    o, k = [], 0
    while k < len(argv):
        if argv[k] in names:
            k += 2
        else:
            o.append(argv[k])
            k += 1
    return o


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def one_lever_problems(argv, ref_argv, w):
    """C-O1 on the argv text (the same rule dagoacc_t1_pullcheck applies, restated here because the reader must not trust the pull check): DAG-OACC's argv == DAG's banked argv except --out and --w-acc,
    --w-acc named exactly once with the value ``w``."""
    bad = []
    if "--w-acc" in ref_argv:
        bad.append("the reference (DAG's banked argv) already names --w-acc")
    n = [i for i, t in enumerate(argv) if t == "--w-acc"]
    if len(n) != 1:
        bad.append(f"--w-acc is named {len(n)} time(s), exactly once is the lever")
    else:
        v = argv[n[0] + 1] if n[0] + 1 < len(argv) else None
        try:
            vf = float(v)
        except (TypeError, ValueError):
            vf = None
        if vf is None or not finite(vf) or vf <= 0.0:
            bad.append(f"--w-acc value {v!r} is not a finite positive number")
        elif w is not None and vf != w:
            bad.append(f"--w-acc {v} != the box-2 w_acc stage's value {w!r}")
    rest, ref = drop_flag(argv, ("--out", "--w-acc")), drop_flag(ref_argv, ("--out",))
    if rest != ref:
        k = next((i for i in range(min(len(rest), len(ref))) if rest[i] != ref[i]), min(len(rest), len(ref)))
        bad.append(f"the argv differs from DAG's banked argv in more than --out / --w-acc ({len(rest)} vs {len(ref)} tokens; first difference at token {k}: {rest[k:k + 2]} vs {ref[k:k + 2]})")
    return bad


def bank_problems(argv, seed):
    """C-O1's second half: the iteration-1 bank (DAG's own) and its sha256 are identical to DAG's."""
    fl = argv_flags(argv)
    bad = []
    v = fl.get("--dagger-bank-sha256")
    if not v or len(v) != 1 or len(v[0]) != 1 or v[0][0] != BANK1_SHA[seed]:
        bad.append(f"--dagger-bank-sha256 {v} is not DAG's iteration-1 bank {BANK1_SHA[seed][:16]}...")
    b = fl.get("--dagger-bank")
    if not b or len(b) != 1 or len(b[0]) != 1 or not str(b[0][0]).replace("\\", "/").rstrip("/").endswith(f"dagger_bank/s{seed}"):
        bad.append(f"--dagger-bank {b} is not .../dagger_bank/s{seed} (DAG's iteration-1 bank)")
    return bad


def wacc_problems(sc: dict, W: float) -> list:
    """C-O2 on the gate's record: the acceleration term runs (finite, non-zero), the built weight is the argv's, O's head gets its gradient, and the controls read exactly 0.0.
    COPIED from box2_ship/gate_oaf_assert_r2.py (sha256 R2_ASSERT_SHA256): the r2 form -- read_o_grad_diff is keyed on levers.residual_zero_init (True -> exactly 0.0; False -> finite and > 0).  A test compares the two on the AST."""
    bad = []
    lv, live, ap = sc["levers"], sc["live"], sc.get("acc_probe")
    if lv.get("w_acc") != W:
        bad.append(f"built levers.w_acc {lv.get('w_acc')!r} != the argv's {W!r}")
    if "acc" not in (live.get("terms") or {}) or not finite(live["terms"]["acc"]) or live["terms"]["acc"] == 0.0:
        bad.append(f"G-LIVE: the acc term is {(live.get('terms') or {}).get('acc')!r} (must be in the loss dict, finite and non-zero)")
    if live.get("missing_terms") not in ([], None):
        bad.append(f"G-LIVE missing_terms {live.get('missing_terms')}")
    if not isinstance(ap, dict) or ap.get("pass") is not True:
        return bad + [f"acc_probe {ap}"]
    if ap.get("w_acc_declared") != W or ap.get("w_probe") != W:
        bad.append(f"acc_probe declared / probe weight {ap.get('w_acc_declared')!r} / {ap.get('w_probe')!r} != {W!r}")
    if not (finite(ap.get("acc")) and ap["acc"] > 0.0):
        bad.append(f"acc_probe.acc {ap.get('acc')!r} is not finite and non-zero")
    if not (finite(ap.get("contribution")) and finite(ap.get("w_x_acc")) and abs(ap["contribution"] - ap["w_x_acc"]) <= 1e-4 * max(1.0, abs(ap["w_x_acc"])) and ap["contribution"] > 0.0):
        bad.append(f"the total moved by {ap.get('contribution')!r} for w x acc = {ap.get('w_x_acc')!r}: the term does not enter the loss at the declared weight")
    if not (finite(ap.get("head_o_grad_diff")) and ap["head_o_grad_diff"] > 0.0):
        bad.append(f"O's head receives gradient {ap.get('head_o_grad_diff')!r} from the acceleration term (must be > 0)")
    if not (finite(live.get("head_o_grad_step0")) and live["head_o_grad_step0"] > 0.0):
        bad.append(f"live.head_o_grad_step0 {live.get('head_o_grad_step0')!r}")
    for k in ("control_repeat_total_diff", "control_repeat_grad_diff", "control_prior_grad_diff", "control_wm_grad_diff"):
        if ap.get(k) != 0.0:
            bad.append(f"acc_probe control {k} = {ap.get(k)!r} (must be exactly 0.0)")
    rd, zi = ap.get("read_o_grad_diff"), lv.get("residual_zero_init")
    if zi is True:
        if not (finite(rd) and rd == 0.0):
            bad.append(f"acc_probe read_o_grad_diff = {rd!r} (levers.residual_zero_init is True: O's last layer is zero, so read_o gets exactly 0.0)")
    elif zi is False:
        if not (finite(rd) and rd > 0.0):
            bad.append(f"acc_probe read_o_grad_diff = {rd!r} (levers.residual_zero_init is False: a trained head passes the term's gradient to read_o, it must be finite and > 0)")
    else:
        bad.append(f"levers.residual_zero_init = {zi!r} is neither True nor False: the read_o_grad_diff reading {rd!r} cannot be judged")
    return bad


def runrec_wacc_problems(summary, selfcheck, W):
    """C-O2's second half: THE RUN'S OWN record carries the built w_acc, equal to the argv's (summary.json levers and the run's selfcheck.json levers)."""
    bad = []
    for nm, rec in (("summary.json", summary), ("selfcheck.json", selfcheck)):
        lv = rec.get("levers") if isinstance(rec.get("levers"), dict) else {}
        if "w_acc" not in lv:
            bad.append(f"{nm} levers carries no w_acc (the run's record must show the BUILT weight)")
        elif lv["w_acc"] != W:
            bad.append(f"{nm} levers.w_acc {lv['w_acc']!r} != the argv's {W!r}")
    return bad


def round6(x):
    """OACC's R3: rounded to 6 significant figures."""
    return float(f"{x:.6g}")


def wacc_c03_problems(box, offline, txt_val, offline_sha_ok=True):
    """C-O3: the box-2 value equals the offline recomputation on DAG_s0's banked fit plans within 1e-4 RELATIVE.  Also: the box value was fitted on DAG_s0's checkpoint (B = DAG), on the banked fit windows (same sha256 as
    the offline route's), and is the raw value rounded to 6 significant figures; the txt agrees with the json.  A mismatch is REFUSED, never resolved by picking one."""
    bad = []
    wb, wo = box.get("w_acc"), offline.get("w_acc")
    if not (finite(wb) and finite(wo) and wo > 0.0):
        return [f"w_acc box {wb!r} / offline {wo!r}: not finite numbers"]
    rel = abs(wb - wo) / abs(wo)
    if not rel <= W_REL_TOL:
        bad.append(f"box w_acc {wb!r} vs offline {wo!r}: relative difference {rel:.3e} > {W_REL_TOL}")
    if txt_val is None or txt_val != wb:
        bad.append(f"w_acc.txt {txt_val!r} != w_acc.json {wb!r}")
    src = box.get("source") if isinstance(box.get("source"), dict) else {}
    if src.get("ckpt_sha256") != DAG_CKPT_SHA[0]:
        bad.append(f"the box value was fitted on checkpoint {str(src.get('ckpt_sha256'))[:16]}, B is DAG_s0 ({DAG_CKPT_SHA[0][:16]})")
    cc = box.get("banked_crosscheck") if isinstance(box.get("banked_crosscheck"), dict) else {}
    osrc = offline.get("source") if isinstance(offline.get("source"), dict) else {}
    if cc.get("fit_windows_sha256") != FITPLANS_SHA256 or osrc.get("fit_windows_sha256") != FITPLANS_SHA256:
        bad.append(f"fit windows sha256: box {str(cc.get('fit_windows_sha256'))[:12]}, offline {str(osrc.get('fit_windows_sha256'))[:12]}, banked fitplans_DAG_s0.npz {FITPLANS_SHA256[:12]}")
    if box.get("n_windows") != offline.get("n_windows") or box.get("n_windows") != 9582:
        bad.append(f"n_windows box {box.get('n_windows')!r} / offline {offline.get('n_windows')!r}, registered 9582")
    raw = box.get("w_acc_raw")
    if not finite(raw) or round6(raw) != wb:
        bad.append(f"w_acc {wb!r} is not w_acc_raw {raw!r} rounded to 6 significant figures (OACC R3)")
    if not offline_sha_ok:
        bad.append("the offline reference is not the pinned wacc_offline_DAG_s0.json")
    return bad


def rollout_argv_problems(new_tokens, banked_tokens):
    """Control C-ARGV: the new rollout's argv equals the registered rollout argv (FLAT's banked B-2 rollout, which equals DAG's in every setting flag, REREAD Amendment 1) except in --ckpt, --tag, --out-dir,
    --dump-ticks (token by token)."""
    if len(new_tokens) != len(banked_tokens):
        return [f"{len(new_tokens)} tokens, the registered rollout has {len(banked_tokens)}"]
    bad = []
    for i, (x, y) in enumerate(zip(new_tokens, banked_tokens)):
        if x != y:
            flag = new_tokens[i - 1] if i else None
            if flag not in ROLLOUT_PATH_FLAGS:
                bad.append(f"token {i} after {flag!r}: {x!r} vs the registered {y!r}")
    return bad


def rows_diff(a, b, tol=1e-9):
    """Two row dicts {label: row}: same labels, same separated flag, delta / lo / hi (and the means, n) within tol.  Labels that hold no row (metadata) are skipped.  -> (differences, max |diff|)."""
    bad, mx = [], 0.0
    is_row = lambda r: isinstance(r, dict) and "delta" in r      # noqa: E731
    for lab in sorted(set(a) | set(b)):
        ra, rb = a.get(lab), b.get(lab)
        if not is_row(ra) and not is_row(rb):
            continue
        if not (is_row(ra) and is_row(rb)):
            bad.append(f"{lab}: present on one side only")
            continue
        if ra.get("status") == "NOT COMPUTED" or rb.get("status") == "NOT COMPUTED":
            continue
        for k in ("delta", "lo", "hi", "mean_a", "mean_b"):
            if k in ra and k in rb and ra[k] is not None and rb[k] is not None:
                d = abs(float(ra[k]) - float(rb[k]))
                mx = max(mx, d)
                if d > tol:
                    bad.append(f"{lab}.{k}: {ra[k]} vs {rb[k]}")
        for k in ("n_windows", "n_episodes"):
            if k in ra and k in rb and ra[k] != rb[k]:
                bad.append(f"{lab}.{k}: {ra[k]} vs {rb[k]}")
        if bool(ra.get("separated")) != bool(rb.get("separated")):
            bad.append(f"{lab}.separated: {ra.get('separated')} vs {rb.get('separated')}")
    return bad, mx


def flatten_numeric(d, prefix=""):
    """{dotted path: float} of the numeric leaves of a nested dict (booleans and strings skipped)."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten_numeric(v, f"{prefix}{k}."))
    elif isinstance(d, (int, float)) and not isinstance(d, bool) and math.isfinite(d):
        out[prefix[:-1]] = float(d)
    return out


def families_diff(fa, fb):
    """The four families at T0, DAG-OACC (fa) against DAG (fb), leaf by leaf: {path: {oacc, dag, diff}} on the common numeric leaves, and the paths on one side only.  Descriptive: no CI exists at this layer."""
    a, b = flatten_numeric(fa), flatten_numeric(fb)
    common = {p: {"oacc": a[p], "dag": b[p], "diff": a[p] - b[p]} for p in sorted(set(a) & set(b))}
    return {"leaves": common, "only_in_oacc": sorted(set(a) - set(b)), "only_in_dag": sorted(set(b) - set(a))}


def guards_beside(r_t0_point):
    """R-T0 beside: DAGGER's definition applied to DAG-OACC - DAG at T0 (point <= +F/2, inclusive).  Descriptive here: the PREREG gives it no bar."""
    return {"R-T0_point_le_F_over_2": bool(r_t0_point <= HALF_F), "R-T0_point": float(r_t0_point), "bound_F_over_2": HALF_F}


def family_of(cid):
    """The PREREG section-3 control group of a control id."""
    if cid.startswith(("C-O3",)):
        return "C-O3"
    if cid.startswith(("C-O2",)):
        return "C-O2"
    if cid.startswith(("RC-DAG", "C2-IDENTITY", "C3-FLOOR", "C4-WINDOWS", "T1-K", "T1-CONFIG", "C-ARGV", "C-DONE", "LON-ROWS", "READA-", "SPEC-", "R-T0-GRID", "T0-CKPT", "PINS", "PREREG")):
        return "C-O4"
    return "C-O1"


# ---------------------------------------------------------------------------------------------------------------- the PREREG text
def table_after(text, heading):
    """The rows of the first markdown table after ``heading`` (markdown bold stripped), header and separator rows dropped."""
    t = text.replace("**", "")
    if t.count(heading) != 1:
        return None
    rows = []
    for ln in t.split(heading, 1)[1].splitlines():
        ln = ln.strip()
        if ln.startswith("|"):
            rows.append(tuple(c.strip() for c in ln.strip("|").split("|")))
        elif rows:
            break
    return [r for r in rows if r[0] not in ("read", "outcome") and not set("".join(r)) <= set("-: ")]


def prereg_problems(text):
    """-> [] iff section 2's five rows and section 4's six rows equal ROWS_2 / OUTCOME_TABLE_PREREG cell by cell, the section-3 controls are named, and F / F/2 / the order sentence are the printed values."""
    probs = []
    for heading, want, label in (("## 2. The read", ROWS_2, "section 2"), ("## 4. Outcomes, committed now", OUTCOME_TABLE_PREREG, "section 4")):
        rows = table_after(text, heading)
        if rows is None:
            probs.append(f"the PREREG has no unique heading {heading!r}")
            continue
        if len(rows) != len(want):
            probs.append(f"{label}: the PREREG has {len(rows)} rows, the reader codes {len(want)}")
        for i, w in enumerate(want):
            r = rows[i] if i < len(rows) else None
            if r != tuple(w):
                probs.append(f"{label} row {w[0]}: the PREREG reads {r!r}, the reader codes {tuple(w)!r}")
    for c in CONTROL_IDS_3:
        if f"**{c}" not in text:
            probs.append(f"section 3 control {c} is not named")
    for frag in ("F = 0.1207 m", "Applied in this order: REFUSED → NO EFFECT → COSTS POSITION → DOMINANT → LONGITUDINAL FIXED → NOT FIXED", "paired\nepisode-cluster bootstrap, B = 2,000, per seed",
                 "STRATEGIC is reported; it is not paused for v8F", "to within 1e-4 relative"):
        if frag not in text:
            probs.append(f"the PREREG no longer contains {frag!r}")
    return probs


def amend1_norm(text):
    """AMEND1.md reduced to plain words: markdown emphasis and backticks removed, whitespace collapsed (the file wraps lines and indents bullets)."""
    return " ".join(text.replace("*", "").replace("`", "").split())


def amend1_problems(text):
    """-> [] iff AMEND1.md still contains, verbatim (emphasis, backticks and line wrapping aside), the condition of every row it rewrites and LONGITUDINAL FIXED's committed next, as OUTCOME_TABLE codes them; and the coded
    table equals the PREREG's printed table on every row Amendment 1 does not rewrite."""
    probs, norm = [], amend1_norm(text)
    for name, (cond, nxt) in AMEND1_ROWS.items():
        for what, frag in (("condition", cond), ("committed next", nxt)):
            if frag is not None and amend1_norm(frag) not in norm:
                probs.append(f"AMEND1 no longer contains {name}'s {what} as the reader codes it: {frag[:90]!r}")
    for (n, c, nx), (n0, c0, nx0) in zip(OUTCOME_TABLE, OUTCOME_TABLE_PREREG):
        if n != n0 or (n not in AMEND1_ROWS and (c, nx) != (c0, nx0)):
            probs.append(f"the coded row {n!r} differs from the PREREG's printed row although Amendment 1 does not rewrite it")
    if tuple(AMEND1_ROWS) != ("DOMINANT", "LONGITUDINAL FIXED") or len(OUTCOME_TABLE) != len(OUTCOME_TABLE_PREREG):
        probs.append("Amendment 1 rewrites exactly DOMINANT and LONGITUDINAL FIXED and adds or removes no row")
    return probs


def check_text_pin(path, want_sha, what):
    if not os.path.isfile(path):
        return False, f"{what}: {path} is missing"
    raw = open(path, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    return got == want_sha, (f"{what}: raw sha256 {got[:12]}... == the pin" if got == want_sha else f"{what}: {path} has sha256 {got[:16]}..., pinned {want_sha[:16]}...")


# ---------------------------------------------------------------------------------------------------------------- pins
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_d2(path):
    """dag2_read.py, imported unchanged after its pin was checked (Read A's driver class ReadA, pin tables, load_dag_read)."""
    if not os.path.isfile(path) or sha256_file(path) != D2_READ_SHA256:
        raise SystemExit(f"REFUSED: {path} is not the pinned dag2_read.py (sha256 {D2_READ_SHA256[:12]}...)")
    return load_module("dag2_read", path)


DEFAULTS = {
    "dag_read": "C:/Users/Admin/dagread/dag_read.py",
    "dag2_read": f"{ART}/dag2t1/dag2_read.py",
    "readers": f"{ART}/dagreread/reader_src",
    "reada_code": f"{H2}/readA_code",
    "lead_block": f"{H2}/inputs/b1_eval_lead_block_k30.npz",
    "dag_t1": f"{ART}/v8f-dagger-t1",
    "dag_art": f"{ART}/v8f-dagger",
    "dagreread": f"{ART}/dagreread",
    "prereg_oacc": f"{H2}/inputs/PREREG_V8F_DAG_OACC.md",
    "prereg_reread": f"{H2}/inputs/PREREG_V8F_DAG_REREAD.md",
    "amend1": f"{ART}/dagland/dagoacc_amend1/AMEND1.md",
    "offline_wacc": f"{H2}/inputs/wacc_offline_DAG_s0.json",
    "r2_assert": "C:/Users/Admin/box2_ship/gate_oaf_assert_r2.py",
}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("manifest", "out", "t1hd_read", "t1k_read", "eval_cache", "tracks", "eval_grid", "eval_join", "work_dir"):
        ap.add_argument("--" + k.replace("_", "-"))
    ap.add_argument("--tool-root", default=f"{ART}/dagt1_tree_b43c7a9a")
    ap.add_argument("--init-sha", action="append", help="SEED=sha256 of the checkpoint DAG-OACC started from (the runs' own records; must equal M2b_s<S>)")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v)
    ap.add_argument("--static-check", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--rehearsal", action="store_true")
    return ap


def static_paths(a):
    return {**a.D2.static_paths(a)}


def own_pin_problems(a):
    bad = []
    for path, sha, what in ((a.prereg_oacc, PREREG_OACC_SHA256, "PREREG_V8F_DAG_OACC.md"), (a.prereg_reread, PREREG_REREAD_SHA256, "PREREG_V8F_DAG_REREAD.md"),
                            (a.amend1, AMEND1_SHA256, "AMEND1.md (Amendment 1 to PREREG_V8F_DAG_OACC)"),
                            (a.offline_wacc, OFFLINE_WACC_SHA256, "wacc_offline_DAG_s0.json (C-O3's offline reference)")):
        ok, det = check_text_pin(path, sha, what)
        if not ok:
            bad.append(det)
    if os.path.isfile(a.prereg_oacc):
        bad += [f"PREREG_V8F_DAG_OACC: {p}" for p in prereg_problems(open(a.prereg_oacc, encoding="utf-8").read())]
    if os.path.isfile(a.amend1):
        bad += [f"AMEND1: {p}" for p in amend1_problems(open(a.amend1, encoding="utf-8").read())]
    for s in SEEDS:
        p = os.path.join(a.dag_art, f"M2b_s{s}_dag", "argv.json")
        if not os.path.isfile(p) or sha256_file(p) != DAG_ARGV_SHA[s]:
            bad.append(f"DAG_s{s}'s banked argv.json at {p} is missing or not the pinned file {DAG_ARGV_SHA[s][:12]}...")
    return bad


# ---------------------------------------------------------------------------------------------------------------- the run-record controls (C-O1, C-O2, C-O3)
def tail_norm(p):
    return str(p).replace("\\", "/")


def load_dag_argv(a, s):
    return list(json.load(open(os.path.join(a.dag_art, f"M2b_s{s}_dag", "argv.json"), encoding="utf-8"))["argv"])


def wacc_value(manifest):
    return json.load(open(manifest["wacc"]["box"], encoding="utf-8")).get("w_acc")


def stage_c03(a, ctl, manifest):
    """C-O3 once (the weight is one value for both seeds): the box-2 w_acc record vs the offline recomputation."""
    box = json.load(open(manifest["wacc"]["box"], encoding="utf-8"))
    off = json.load(open(manifest["wacc"]["offline"], encoding="utf-8"))
    txt = open(manifest["wacc"]["txt"], encoding="utf-8").read().strip()
    try:
        tv = float(txt)
    except ValueError:
        tv = None
    ok_sha = sha256_file(manifest["wacc"]["offline"]) == OFFLINE_WACC_SHA256
    ctl.run("C-O3", lambda: (lambda bad: (not bad, bad or {"box_w_acc": box.get("w_acc"), "offline_w_acc": off.get("w_acc"), "tolerance_relative": W_REL_TOL,
                                                         "fitted_on": "DAG_s0's checkpoint, banked fit plans fitplans_DAG_s0.npz"}))(wacc_c03_problems(box, off, tv, ok_sha)))
    return box.get("w_acc")


def stage_runs(a, DR, ctl, manifest, init_sha, W, state):
    """C-O1 (one lever, same bank) and C-O2 (the lever runs) per seed, plus the recipe's run-record controls (SUMMARY, INIT-CKPT, ARGV, DAGGER-REACHED, GATE-TOKEN)."""
    ki, kn, ke, kp = "init_ckpt.sha256", "dagger_n", "dagger_eligible", "dagger_p"
    recs = {}
    for s in SEEDS:
        e = DR.entry(manifest, "DAGOACC", s)
        tag = f"DAGOACC_s{s}"
        ctl.add(f"M2B-CKPT:s{s}", init_sha.get(s) == M2B_CKPT_SHA[s], {"passed": str(init_sha.get(s))[:16], f"M2b_s{s}": M2B_CKPT_SHA[s][:16]})
        if not os.path.isfile(os.path.join(e["train_dir"], "summary.json")):
            state[tag] = "not pulled"
            continue
        tr = DR.train_records(e, ki)
        recs[s] = tr
        ctl.run(f"SUMMARY:{tag}", lambda tr=tr, s=s, tag=tag: DR.g_summary(tag, s, tr))
        ctl.run(f"INIT-CKPT:{tag}", lambda tr=tr, s=s, tag=tag: DR.g_init(tag, s, tr, init_sha.get(s, ""), ki))
        ctl.run(f"ARGV:{tag}", lambda tr=tr, s=s, tag=tag: DR.g_argv(tag, "DAG", s, tr))
        ctl.run(f"DAGGER-REACHED:{tag}", lambda tr=tr, tag=tag: DR.g_reached(tag, "DAG", tr, kn, ke, kp))
        ctl.run(f"GATE-TOKEN:{tag}", lambda tr=tr, e=e, tag=tag: DR.g_token(tag, tr, DR.read_token(e.get("token"))))
        dag_argv = load_dag_argv(a, s)
        ctl.run(f"C-O1-ONELEVER:{tag}", lambda tr=tr, d=dag_argv, W=W: (lambda bad: (not bad, bad or "DAG's banked argv + --w-acc W, only --out differing"))(one_lever_problems(tr["argv"], d, W)))
        ctl.run(f"C-O1-BANK:{tag}", lambda tr=tr, s=s: (lambda bad: (not bad, bad or f"the iteration-1 bank, sha256 {BANK1_SHA[s][:16]}... identical to DAG's"))(bank_problems(tr["argv"], s)))
        gsc = os.path.join(e["gate_dir"], "selfcheck.json")
        ctl.run(f"C-O2-GATE:{tag}", lambda gsc=gsc, W=W: (lambda bad: (not bad, bad or "the gate shows the acc term finite and non-zero, O's head gradient > 0, the controls exactly 0.0 (r2 form)"))(
            wacc_problems(json.load(open(gsc, encoding="utf-8")), W)))
        ctl.run(f"C-O2-RUNREC:{tag}", lambda tr=tr, W=W: (lambda bad: (not bad, bad or f"the run's own record carries the built w_acc {W!r}, equal to the argv's"))(
            runrec_wacc_problems(tr["summary.json"], tr["selfcheck.json"], W)))
    return recs


def r2_form_control(a, ctl, manifest, W):
    """C-O2 'match r2's form': when the r2 asserter file is present and is the pinned one, its own wacc_problems and this reader's copy agree on every seed's gate record."""
    p = a.r2_assert
    if not os.path.isfile(p):
        ctl.add("C-O2-R2-FORM", True, {"skipped": f"{p} is not on this machine (box2_ship); the copy is pinned by an AST test in the self-test"})
        return
    if sha256_file(p) != R2_ASSERT_SHA256:
        ctl.add("C-O2-R2-FORM", False, f"{p} is not the pinned r2 asserter (sha256 {R2_ASSERT_SHA256[:12]}...)")
        return
    m = load_module("gate_oaf_assert_r2", p)
    res = {}
    for s in SEEDS:
        gsc = os.path.join(manifest["arms"]["DAGOACC"][str(s)]["gate_dir"], "selfcheck.json")
        if os.path.isfile(gsc):
            sc = json.load(open(gsc, encoding="utf-8"))
            res[f"s{s}"] = (m.wacc_problems(sc, W) == wacc_problems(sc, W))
    ctl.add("C-O2-R2-FORM", bool(res) and all(res.values()), {"agree_with_r2_wacc_problems": res})


# ---------------------------------------------------------------------------------------------------------------- the read
def main(argv=None):
    a = build_parser().parse_args(argv)
    t_start = time.time()
    D2 = load_d2(a.dag2_read)
    a.D2 = D2
    a.dag_read_json = f"{a.dag_t1}/read/DAG_READ.json"
    a.reada_banked = f"{a.dagreread}/raw/READ_A.json"
    a.t1kf_banked = f"{a.dagreread}/raw/t1kf_DAG/T1KF_READ_A1_DAG.json"
    bad = D2.pin_problems(static_paths(a)) + own_pin_problems(a)
    if a.static_check:
        print(json.dumps({"pins_checked": len(static_paths(a)) + 6, "problems": bad, "prereg_cells_verbatim": [r[0] for r in ROWS_2] + list(OUTCOME_NAMES), "amend1_rows_verbatim": list(AMEND1_ROWS)}, indent=1, default=str))
        print("DAGOACC-READ-STATIC-OK" if not bad else "DAGOACC-READ-STATIC-REFUSED")
        return 0 if not bad else 2
    if bad:
        print("DAGOACC READ REFUSED (nothing written): pinned inputs:", file=sys.stderr)
        for b in bad:
            print("  -", b, file=sys.stderr)
        return 2
    DR = D2.load_dag_read(a.dag_read)
    try:
        if a.rehearsal:
            return rehearsal(a, DR, D2, t_start)
        return run(a, DR, D2, t_start)
    except DR.Refusal as e:
        print("DAGOACC READ REFUSED (nothing written):", file=sys.stderr)
        for m_ in e.msgs:
            print("  -", m_, file=sys.stderr)
        return 2
    except (OSError, KeyError, ValueError) as e:
        print(f"DAGOACC READ INPUT ERROR (nothing written): REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


def refused(ctl):
    fam = {}
    for k in ctl.failed():
        fam.setdefault(family_of(k), []).append(k)
    print("DAGOACC READ REFUSED (nothing written):", file=sys.stderr)
    for f in sorted(fam):
        print(f"  [{f}]", file=sys.stderr)
        for k in fam[f]:
            print("   -", k, str(ctl.items[k]["detail"])[:300], file=sys.stderr)
    return 2


def dag_dump_dir(a, s):
    return f"{a.dag_t1}/dump_DAG_s{s}_m10_causal_heads"


def layer_a(a, A, DR, ctl, oacc, dag, dagread_oacc, spos_arm, label):
    """Read A's rows on the pairs; the controls C1 (RC-DAG), C2, C3, C4.  oacc[s], dag[s] = per-window dumps (causal).  ``label`` names the new arm in control ids (DAGOACC, or REHEARSAL's stand-in)."""
    np, T, RA = A.np, A.T, A.RA
    banked_reada = json.load(open(a.reada_banked, encoding="utf-8"))["READ_A"]["rows"]
    banked_dagread = json.load(open(a.dag_read_json, encoding="utf-8"))
    comps = {k: {} for k in ("OP", "OD", "DP")}
    tacdk = {k: {} for k in comps}
    for s in SEEDS:
        o, g = oacc[s], dag[s]
        same = all(np.array_equal(o[k], g[k]) for k in ("clip_index", "ws", "eid", "g", "v0"))
        ctl.add(f"C4-WINDOWS:{label}-vs-DAG:s{s}", bool(same and len(o["eid"]) == DR.N_WINDOWS and len(np.unique(o["eid"])) == DR.N_EPISODES),
                {"n_windows": int(len(o["eid"])), "n_episodes": int(len(np.unique(o["eid"]))), "note": "clip_index, ws, eid, g, v0 bit-equal to the banked DAG's"})
        ctl.add(f"C3-FLOOR:{label}-vs-DAG:s{s}", bool(np.array_equal(o["fl"], g["fl"])), "P10's tracks read no model: max |delta| must be exactly 0.0")
        cx = A.tacdk_context(g)
        P = {"OACC": T.prep(label, o["cl"], cx), "DAG": T.prep("DAG", g["cl"], cx), "P10": T.prep("P10", g["fl"], cx)}
        for k, (X, Y, PX, PY) in {"OP": (o["cl"], o["fl"], P["OACC"], P["P10"]), "OD": (o["cl"], g["cl"], P["OACC"], P["DAG"]), "DP": (g["cl"], g["fl"], P["DAG"], P["P10"])}.items():
            comps[k][s], tacdk[k][s] = A.pair_rows(X, Y, g, PX, PY)
        idc = RA.ident_block(o["cl"], o["g"], o["eid"])
        idt = T.rows(P["OACC"], P["OACC"], o["eid"])
        ok = bool(all(v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0 for v in idc.values()) and all(T.is_zero(v) for v in idt.values()))
        ctl.add(f"C2-IDENTITY:{label}-{label}:s{s}", ok, f"({label}, {label}) reads +0 [0, 0] on every comps and TAC / DK row")
    tabs = {"OP": A.table(comps["OP"], tacdk["OP"], dagread_oacc, spos_arm), "OD": A.table(comps["OD"], tacdk["OD"], banked_dagread, None), "DP": A.table(comps["DP"], tacdk["DP"], banked_dagread, "DAG")}
    for s in SEEDS:
        diffs = a.D2.rows_equal(tabs["DP"][s], banked_reada[f"s{s}"])
        ctl.add(f"RC-DAG:s{s}", not diffs, {"rows_compared": len(banked_reada[f"s{s}"]), "tolerance": 1e-4, "differences": diffs[:6], "note": "C1: the readers re-computed on the banked arm (DAG - P10) reproduce READ_A.json"})
    return tabs


def compose(A, ctl, tabs):
    """The PREREG's composition from the Read A tables.  -> (outcome block, per-seed rows, flags)."""
    sf, extra, rows, m1w = {}, {}, {}, {}
    for s in SEEDS:
        tOP, tOD = tabs["OP"][s], tabs["OD"][s]
        ctl.add(f"LON-ROWS-COMPUTED:s{s}", all(l in tOP and tOP[l].get("status") != "NOT COMPUTED" for l in LON_ROWS) and all(l in tOD and tOD[l].get("status") != "NOT COMPUTED" for l in (ACC_LABEL, ADE_LABEL)),
                "the six LONGITUDINAL rows of DAG-OACC - P10 and the PRIMARY-ACC / KEEP rows of DAG-OACC - DAG are all computed (a row that cannot be computed never counts as passing)")
        f, ex = seed_flags_of(tOD[ACC_LABEL], tOD[ADE_LABEL], tOP, A.DIRECTIONS)
        sf[s], extra[s] = f, ex
        m1w[s] = M1_LABEL in ex["latcrawl_worse"]
        vv = A.BR["verdict"](tabs["OP"])[s]
        ctl.add(f"READA-VERDICT-LINK:s{s}", bool(vv["D1"] == f["D1"] and vv["D2"] == f["D2"] and vv["D3"] == f["D3"]), {"build_report.verdict": {k: vv[k] for k in ("D1", "D2", "D3")}, "this reader": {k: f[k] for k in ("D1", "D2", "D3")}})
        rows[f"s{s}"] = {"PRIMARY-ACC": {"row": {k: tOD[ACC_LABEL].get(k) for k in ("delta", "lo", "hi", "separated", "n", "mean_arm", "mean_ref")}, "passes": f["PA"], "bar": "delta < 0 and separated"},
                         "KEEP": {"row": {k: tOD[ADE_LABEL].get(k) for k in ("delta", "lo", "hi", "separated", "n", "mean_arm", "mean_ref")}, "passes": f["KEEP"], "bar": f"upper bound <= +{HALF_F} m"},
                         "LON-CLEAR": {**ex["lon"], "rows": {l: {k: tOP[l].get(k) for k in ("delta", "lo", "hi", "separated", "n")} for l in LON_ROWS if l in tOP and tOP[l].get("status") != "NOT COMPUTED"}},
                         "READ_A": {"D1": f["D1"], "D2": f["D2"], "D3": f["D3"], **ex["d"], "lateral_or_crawl_rows_separated_worse": ex["latcrawl_worse"],
                                    "table_DAGOACC_minus_P10": tOP},
                         "beside_DAGOACC_minus_DAG_table": tOD}
    return sf, extra, rows, m1w


def outcome_from(ctl, sf, m1w):
    failed = sorted(ctl.failed())
    dec = decide(failed, sf, m1w)
    dec["outcome_unamended"] = unamended_beside(failed, sf, m1w, dec["OUTCOME"])
    return dec


def t0_rows(a, DR, T, ctl, manifest, label="DAGOACC"):
    """R-T0 (T0 ADE6, DAG-OACC - DAG) and the four families at T0 (the box-2 eval json's raw.families), beside."""
    out = {}
    for s in SEEDS:
        eo = DR.entry(manifest, "DAGOACC", s)
        ec = manifest["control"]["DAG"][str(s)]
        jo, zo, oko = DR.t0_arrays(eo["t0"], eo["ckpt_tail"])
        jc, zc, okc = DR.t0_arrays(ec["t0"], ec["ckpt_tail"])
        ctl.add(f"T0-CKPT:{label}_s{s}", oko and okc, f"T0 eval json ckpt {jo.get('ckpt')!r} / DAG's {jc.get('ckpt')!r} vs .../{eo['ckpt_tail']} and .../{ec['ckpt_tail']}")
        same = bool(len(zo["eid"]) == len(zc["eid"]) and (zo["eid"] == zc["eid"]).all())
        ctl.add(f"R-T0-GRID:s{s}", same, "the two T0 evals are on the same eval-141 grid (eid equal)")
        if not same:
            continue
        r = DR.err_pair_row(T, f"R-T0 T0 own-code ADE6 (DAG-OACC - DAG) s{s}", zo["ade6"], zc["ade6"], zo["eid"], N_BOOT)
        r["bar_point_le"] = HALF_F
        out[f"s{s}"] = {"R-T0": r, "guards": guards_beside(r["point"]),
                        "four_families_T0": families_diff((jo.get("raw") or {}).get("families", {}), (jc.get("raw") or {}).get("families", {})),
                        "note": "T0 (teacher-forced own-code, eval-141), descriptive: raw.families of box 2's standard eval, DAG-OACC against DAG leaf by leaf; no paired T0 four-family instrument exists"}
    return out


def run(a, DR, D2, t_start):
    ctl = DR.Ctl()
    ctl.add("PINS", True, "pinned inputs re-hashed (incl. AMEND1.md); PREREG_V8F_DAG_OACC section 2 and 4 cells verbatim; AMEND1's rewritten rows verbatim")
    A = D2.ReadA(a)
    spec_ok = tuple((f, l, d) for f, l, _, _, d in A.BR["SPEC"]) == READ_A_ROWS
    ctl.add("SPEC-LABELS", spec_ok, "this reader's Read A rows / families / directions equal build_report.SPEC (PREREG_V8F_DAG_REREAD section 2)")
    init_sha = {s: v.lower() for s, v in DR.parse_kv(a.init_sha, "--init-sha").items()}
    manifest = DR.load_manifest(a.manifest)
    if sorted(manifest["arms"]) != ["DAGOACC"] or sorted(manifest.get("control", {})) != ["DAG"]:
        raise DR.Refusal(f"the manifest must hold exactly the arm DAGOACC and the control DAG, it holds {sorted(manifest['arms'])} / {sorted(manifest.get('control', {}))}")
    state = {}
    W = stage_c03(a, ctl, manifest)
    recs = stage_runs(a, DR, ctl, manifest, init_sha, W, state)
    r2_form_control(a, ctl, manifest, W)
    if a.preflight:
        for k, v in ctl.items.items():
            print(("PASS " if v["PASS"] else "FAIL ") + k + "  " + str(v["detail"])[:170].replace(chr(10), " "))
        print(json.dumps({"state": state, "controls": len(ctl.items), "failed": sorted(ctl.failed())}, indent=1, default=str))
        if a.out:
            write_out(a, {"preflight": True, "state": state, "controls_PASS": ctl.flags(), "controls": ctl.items, "provenance": provenance(a, t_start)}, 0)
        return 0 if not ctl.failed() else 1
    if len(recs) != 2:
        raise DR.Refusal([f"both seeds' runs are read; pulled: {sorted(recs)}"])
    if ctl.failed():
        return refused(ctl)
    # ------------------------------------------------------------------------------------------------ the rollouts (T1-HEADS pipeline), windows, C-O4
    T = DR.load_t1hd(a.t1hd_read, a.tool_root)
    DR.load_t1k(a.t1k_read, a.tool_root)
    work = a.work_dir or os.path.join(os.path.dirname(os.path.abspath(a.out)), "_dagoacc_work")
    os.makedirs(work, exist_ok=True)
    ns = T.parse(["--mode", "real", "--tool-root", a.tool_root, "--eval-cache", a.eval_cache, "--tracks", a.tracks, "--eval-grid", a.eval_grid, "--eval-join", a.eval_join,
                  "--banked-dir", ".", "--out", os.path.join(work, "t1hd_unused.json")])
    backend = DR.RealBackend(T, ns, N_BOOT, work)
    packs, oacc, dag = {}, {}, {}
    for s in SEEDS:
        e = DR.entry(manifest, "DAGOACC", s)
        tag = f"DAGOACC_s{s}"
        t1 = e["t1"]
        t1json = DR.jload(DR.need(t1["json"], f"{tag} t1v8f json"))
        pk = backend.analyze_t1(t1, t1.get("tag") or tag, s)
        packs[s] = pk
        for k, (ok, det) in pk.controls.items():
            ctl.add(f"T1-K:{k}:{tag}", ok, det)
        ctl.run(f"T1-CONFIG:{tag}", lambda t1json=t1json, e=e, pk=pk, tag=tag: DR.g_t1cfg(tag, t1json, e["ckpt_tail"], "causal", pk.meta))
        oacc[s] = A.dump(t1["dump_dir"])
        dag[s] = A.dump(dag_dump_dir(a, s))
        nt = [x for x in open(t1["argv"], encoding="utf-8").read().split("\n") if x != ""] if os.path.isfile(t1["argv"]) else None
        bt = [x for x in open(f"{a.dagreread}/argv/FLAT_s{s}_m10_causal_heads.argv", encoding="utf-8").read().split("\n") if x != ""]
        ctl.add(f"C-ARGV:{tag}", nt is not None and not rollout_argv_problems(nt, bt), {"problems": rollout_argv_problems(nt, bt) if nt is not None else f"{t1['argv']} missing",
                                                                                      "compared_with": f"dagreread/argv/FLAT_s{s}_m10_causal_heads.argv (the registered T1-kin rollout; equal to DAG's in every setting flag)"})
        dn = os.path.join(os.path.dirname(t1["json"]), f"done_{t1['tag']}.txt")
        dr = DR.jload(dn)
        ck = os.path.join(e["train_dir"], "ckpt_last.pt")
        ctl.add(f"C-DONE:{tag}", bool(dr.get("ckpt_sha256") == dr.get("ckpt_sha256_before") == dr.get("ckpt_sha256_after") == sha256_file(ck)),
                {"done_record": {k: str(dr.get(k))[:16] for k in ("ckpt_sha256", "ckpt_sha256_before", "ckpt_sha256_after")}, "pulled_checkpoint": sha256_file(ck)[:16]})
    dagread_oacc = {"beside": {"per_seed": {f"s{s}": {"per_run_t1heads": {"DAGOACC": {SPOS_KEY: packs[s].extra[SPOS_KEY]}}} for s in SEEDS}}}
    tabs = layer_a(a, A, DR, ctl, oacc, dag, dagread_oacc, "DAGOACC", "DAGOACC")
    t0 = t0_rows(a, DR, T, ctl, manifest)
    if ctl.failed():
        return refused(ctl)
    sf, extra, rows, m1w = compose(A, ctl, tabs)
    if ctl.failed():
        return refused(ctl)
    dec = outcome_from(ctl, sf, m1w)
    banked_dagread = json.load(open(a.dag_read_json, encoding="utf-8"))
    beside = {"per_seed": {f"s{s}": {"S-POS_vs_CTRV": {"DAGOACC": packs[s].extra.get(SPOS_KEY), "DAG_banked": banked_dagread["beside"]["per_seed"][f"s{s}"]["per_run_t1heads"]["DAG"][SPOS_KEY]},
                                     "crawl_fraction_ticks_ge1": {"DAGOACC": packs[s].extra.get("crawl_fraction_ticks_ge1")},
                                     "T0": t0.get(f"s{s}"), "DAG_minus_P10_banked_context": tabs["DP"][s]} for s in SEEDS},
              "crawl_rows_fail_REREAD": crawl_rows_fail({s: tabs["OP"][s] for s in SEEDS}),
              "readA_outcome_if_applied_REREAD_section_5": A.BR["outcome"](A.BR["verdict"](tabs["OP"])),
              "K4": "NOT RUN (needs a logged-bucket rollout and a fresh-age T0 eval; PREREG section 5 schedules neither and section 3 does not require K4)"}
    out = {"OUTCOME": outcome_block(dec), "controls_PASS": ctl.flags(), "rows": rows, "beside": beside, "controls": ctl.items, "state": state,
           "tier": "T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic; plain arms); R-T0 and the T0 families at T0 (box 2's standard eval). Never 'closed loop'; the lead follows its logged future.",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0, 141 episodes (2,399 windows), per seed, paired. It answers 'would another draw of EPISODES say this?'; the training-seed question is "
                        "answered by the two seeds and F = 0.1207 m (H-ESTIM-SEED-1: no replicate arm was trained, so a separated row is a result about one training run per seed); the planner is deterministic.",
           "four_families": "ADE6 + LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC rows of Read A, per seed (never pooled); STRATEGIC is reported, not paused for v8F",
           "w_acc": W, "provenance": provenance(a, t_start)}
    return write_out(a, out, outcome_rc(dec))


def rehearsal(a, DR, D2, t_start):
    """The whole row pipeline and decide() on BANKED data only: DAG-0 stands in for DAG-OACC against DAG (no DAG-OACC file is opened).  Labelled REHEARSAL; the outcome word is NOT a DAG-OACC result."""
    ctl = DR.Ctl()
    A = D2.ReadA(a)
    ctl.add("SPEC-LABELS", tuple((f, l, d) for f, l, _, _, d in A.BR["SPEC"]) == READ_A_ROWS, "this reader's Read A rows equal build_report.SPEC")
    oacc = {s: A.dump(f"{a.dag_t1}/dump_DAG0_s{s}_m10_causal_heads") for s in SEEDS}
    dag = {s: A.dump(dag_dump_dir(a, s)) for s in SEEDS}
    banked_dagread = json.load(open(a.dag_read_json, encoding="utf-8"))
    tabs = layer_a(a, A, DR, ctl, oacc, dag, banked_dagread, "DAG0", "REHEARSAL-DAG0")
    sf, extra, rows, m1w = compose(A, ctl, tabs)
    # the T0 beside rows (R-T0, the four families at T0) through the SAME t0_rows code the real run uses, on banked evals: DAG-0's T0 eval stands in for DAG-OACC's, DAG's is the control
    man = {"arms": {"DAGOACC": {str(s): {"ckpt_tail": f"M2b_s{s}_dag0/ckpt_last.pt", "t0": {"json": f"{a.dag_art}/M2b_s{s}_dag0/eval_M2b-dag0_s{s}.json", "npz": f"{a.dag_art}/M2b_s{s}_dag0/eval_M2b-dag0_s{s}.npz"}} for s in SEEDS}},
           "control": {"DAG": {str(s): {"ckpt_tail": f"M2b_s{s}_dag/ckpt_last.pt", "t0": {"json": f"{a.dag_art}/M2b_s{s}_dag/eval_M2b-dag_s{s}.json", "npz": f"{a.dag_art}/M2b_s{s}_dag/eval_M2b-dag_s{s}.npz"}} for s in SEEDS}}}
    t0 = t0_rows(a, DR, DR.load_t1hd(a.t1hd_read, a.tool_root), ctl, man, "REHEARSAL-DAG0")                  # the same T the real run passes (t1hd_read), NOT A.T
    for k, v in ctl.items.items():
        print(("PASS " if v["PASS"] else "FAIL ") + k + "  " + str(v["detail"])[:150].replace(chr(10), " "))
    dec = outcome_from(ctl, sf, m1w)
    print(json.dumps({"REHEARSAL (DAG-0 as the stand-in for DAG-OACC; banked data only; NOT a DAG-OACC result)": dec["OUTCOME"], "matching": dec["matching_rows_in_table_order"], "flags": dec["flags"],
                      "outcome_unamended": dec["outcome_unamended"], "crawl_open": dec["crawl_open"], "dominant_clauses_failed_per_seed": dec["dominant_clauses_failed_per_seed"], "LON-CLEAR": {f"s{s}": extra[s]["lon"] for s in SEEDS}, "lateral_or_crawl_worse": {f"s{s}": extra[s]["latcrawl_worse"] for s in SEEDS}}, indent=1, default=str))
    if a.out:
        write_out(a, {"REHEARSAL": True, "note": "DAG-0 stands in for DAG-OACC against DAG, banked data only; NOT a DAG-OACC result", "OUTCOME": outcome_block(dec), "controls_PASS": ctl.flags(), "rows": rows, "t0_beside": t0,
                      "provenance": provenance(a, t_start)}, 0)
    return 0 if not ctl.failed() else 1


def provenance(a, t_start):
    return {"prereg": PREREG_OACC_ID, "prereg_sha256": PREREG_OACC_SHA256, "amendment": AMEND1_ID, "amend1_sha256": AMEND1_SHA256, "amend1_file_sha256": sha256_file(a.amend1), "amend1_repo_path": AMEND1_REPO_PATH, "claim_row": CLAIM_ROW, "reader_sha256": sha256_file(os.path.abspath(__file__)), "dag2_read_sha256": D2_READ_SHA256,
            "manifest_sha256": sha256_file(a.manifest) if a.manifest and os.path.isfile(a.manifest) else None, "wall_s": time.time() - t_start, "evidence_class": "MEASURED (ours; artifact paths in the manifest)"}


def write_out(a, out, rc):
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    oc = out.get("OUTCOME")
    if isinstance(oc, dict):
        print(json.dumps({"OUTCOME": oc["OUTCOME"], "matching_rows": oc["matching_rows_in_table_order"], "committed_next": oc.get("committed_next"), "reported_as": oc.get("reported_as"), "crawl_open": oc.get("crawl_open"),
                          "dominant_clauses_failed_per_seed": oc.get("dominant_clauses_failed_per_seed"), "outcome_unamended": oc.get("outcome_unamended"), "note": oc.get("note")},
                         indent=1))
    print(f"wrote {a.out}  sha256 {sha256_file(a.out)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
