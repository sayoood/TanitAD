#!/usr/bin/env python3
"""dag_orc_read.py -- the dev-box T1-kin read of PREREG_V8F_DAG_ORC (claim row H-V8F-DAG-ORC-1).  BUILT AND SELF-TESTED, NOT RUN ON DAG-ORC OUTPUTS (none exist).

PREREG (binding, nothing here changes it): Project Steering/PREREG_V8F_DAG_ORC.md @ 3b7b404e, raw-byte sha256 PREREG_SHA256 below.  Arms: A (DAG-ORC, p = 0.3), CTRL (= DAG2-0, banked), REP
(DAG2-0's recipe, --seed 10 / 11), B (p = 0.1, read only on A's HARMS), against DAG (the base) and P10 (the floor planner), both seeds.

NOTHING IS RE-DERIVED.  This file composes the EXISTING readers and adds only the PREREG's own composition:
  ref/dag2_read.py   Read A's layer (ReadA: comps / L1 / L2 / M1 rows, TAC + DK rows, the Read-A table), unchanged, sha-pinned (the staged copy of dag2t1/dag2_read.py)
  ref/dag_read.py    the DAGGER reader's helpers (Ctl, RealBackend = the T1-HEADS pipeline, g_summary / g_init / g_token / g_t1cfg, f1_pair_row, err_pair_row, t0_arrays), sha-pinned
  P0_RF0.json        the banked replicate floor RF0 (p0_rf0.py), frozen BEFORE any DAG-ORC rollout; this reader loads it only if its sha256 equals RF0_SHA256

The only new logic -- the part dag_orc_selftest.py mutates -- is: the bar 2 * max(F_r, RF0_r, RF_r) (inclusive), R-CRAWL, R-MOVE (clears), the guards R-TLAT / R-TLON / R-ADE / R-T0 /
R-NEW (fires), the adoption tests R-KEEP / R-NOBASE, the outcome table coded cell by cell from section 7 (A first; B only on A's HARMS), the PREREG pin (raw sha256 + every table cell
verbatim under its heading), the run-record controls K-PAIR / K-INIT / K-REACH / K-BUILD / K-P0 / K-ZERO / K-RC, and the slot discovery.

    python dag_orc_read.py --static-check                                       pins, PREREG text, RF0 pin; touches no run output
    python dag_orc_read.py --dry-run --manifest runs_dry.json                   + slot discovery on the REAL banked inputs; ends OK with the DAG-ORC slots reported missing
    python dag_orc_read.py --rc-dag2 --out RC_DAG2.json                         K-RC: this reader's analysis on DAG2 / DAG2-0, floors 0, adoption off, vs the banked DAG2_READ.json
    python dag_orc_read.py --manifest runs.json --out DAG_ORC_READ.json         THE READ (REFUSED rc 2: nothing written | rc 0 outcome written | rc 3 UNPLACED | rc 4 slots missing)

Manifest (runs.json): {"seeds": [0, 1], "arms": {"A"|"CTRL"|"REP"|"B"|"DAG": {"0": ENTRY, "1": ENTRY}}, "build": K-BUILD record}.  ENTRY = {ckpt_tail, train_dir, token,
t1: {tag, ticks, dump_dir, json}, t1_logged: {..}, t0: {json, npz} (box 2's standard eval), t0_fresh: {json, npz}} exactly as dag2t1_manifest.py writes them.  CTRL and DAG are the banked runs.
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import math
import os
import re
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:       # noqa: BLE001
        pass

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
REF = f"{HERE}/ref"
ART = "E:/Projects/TanitAD-artifacts"
TREE = f"{ART}/dagt1_tree_b43c7a9a"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"

# ---------------------------------------------------------------------------------------------------------------- PREREG_V8F_DAG_ORC (committed numbers)
PREREG_ID = "Project Steering/PREREG_V8F_DAG_ORC.md @ 3b7b404e"
PREREG_SHA256 = "38bc688b173cdb8c94b8a4ce37a29876cb7b9b1cf8170043afc22290aca0c4ee"     # raw bytes of the 3b7b404e blob
RF0_SHA256 = "1608d70b21b6b6a45136a6d96828f70b5bbabd9c0ceb84db136927593c79ad0f"        # P0_RF0.json, frozen 2026-10-27 by p0_rf0.py (section 3, launch precondition 1)
RF0_SCHEMA = "p0-rf0-v1"
CLAIM_ROW = "H-V8F-DAG-ORC-1"
SEEDS = (0, 1)
F_ADE6 = 0.1207                  # section 3: F_r for ADE6 (F_M2b, MODEL_REGISTRY section 17, H-V8F-SEED-1); every other row: 0
HALF_F = 0.0604                  # +F/2: R-T0 and R-KEEP, inclusive, on the point
P_A, P_B, REACH_TOL = 0.3, 0.1, 0.02                    # section 6, K-REACH
LAW_A = {"--ego-perturb-vscale": ["0.5", "1.0"], "--ego-perturb-vmax": ["8.0"], "--ego-perturb-adrop": ["1.5"], "--ego-perturb-yawrate-std": ["2.0"]}    # section 1; plus --ego-perturb-seed <s>
REP_SEED = {0: 10, 1: 11}
STEPS, LR, WARMUP, PGT = 3000, 1e-4, 100, 0.2
KEY_INIT = "init_ckpt.sha256"
N_BOOT, N_WINDOWS, N_EPISODES = 2000, 2399, 141
DAG_CKPT_SHA = {0: "d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1", 1: "1d12ec2eb825990d57dcc45dda50b5178d1d733961b47ab0e3d50e6155fa39ef"}    # section 1: DAG_s
CTRL_CKPT_PREFIX = {0: "603b9151f5119886", 1: "f4234ac11854076b"}                       # section 1: DAG2-0_s ckpt sha256 (prefix, as the PREREG prints it)
BUILD_BASE_COMMIT = "1d4f4fbb"                                                           # section 6, K-BUILD
BUILD_FILES = ("stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/models/v8f_mini.py")
ARMS = ("A", "CTRL", "REP", "B", "DAG")
REQUIRED_SLOTS = ("A", "CTRL", "REP", "DAG")             # B is read only on A's HARMS (section 7)

LABELS = ("ADE6", "speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG", "heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE",
          "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)", "TAC-LAT", "TAC-LON", "S-POS@30", "S-POS@50", "S-POS@80", "M1 crawl share")          # Read A's 19 rows (build_report.SPEC + M1)
DIRECTION = {l: "lower" for l in LABELS}
DIRECTION.update({"TAC-LAT": "higher", "TAC-LON": "higher"})
FAMILY = {"ADE6": "PRIMARY", "speed MAE": "LONGITUDINAL", "along-track MAE": "LONGITUDINAL", "accel MAE": "LONGITUDINAL", "DK-VIOL": "LONGITUDINAL", "DK-HW": "LONGITUDINAL",
          "DK-TG": "LONGITUDINAL", "heading MAE": "LATERAL", "yaw-rate MAE": "LATERAL", "curvature MAE": "LATERAL", "cross-track MAE": "LATERAL", "L1 lateral-accel MAE": "LATERAL",
          "L2 curvature MAE (GT>=5.2)": "LATERAL", "TAC-LAT": "TACTICAL", "TAC-LON": "TACTICAL", "S-POS@30": "STRATEGIC", "S-POS@50": "STRATEGIC", "S-POS@80": "STRATEGIC",
          "M1 crawl share": "crawl"}
SPOS_LABELS = ("S-POS@30", "S-POS@50", "S-POS@80")
SPOS_KEY = "S-POS vs CTRV (ticks>=1, within this run)"
CRAWL_SET = {0: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"),                    # section 4: the rows separated-worse for DAG - P10 (RESULT_V8F_DAG_REREAD section 1)
             1: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share", "yaw-rate MAE")}
EXEMPT = ("accel MAE", "DK-VIOL")                                                                       # section 4: belong to DAG-OACC; reported, exempt from R-NEW and R-NOBASE
DOMAIN_N = {0: 14, 1: 13}                                                                               # section 4: |D_s|
GUARDS = ("R-TLAT", "R-TLON", "R-ADE", "R-T0", "R-NEW")
ADOPT_TESTS = ("R-KEEP", "R-NOBASE")
TIER_T1 = "T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan; plain arms). Never 'closed loop'."
TIER_T0 = "T0 (teacher-forced own-code, eval-141, ages cycle: box 2's standard eval). Never 'driving performance'."
ESTIMATOR = ("taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0, 141 episodes / 2,399 windows, per seed, paired, on the same windows and matched cells. It answers 'would another "
             "draw of EPISODES say this?'. The training-run question is answered by the two seeds plus the replicate floors RF0 / RF (section 3); the planner is deterministic, so no inference-seed question arises.")

# The tables of the PREREG that this reader codes, cell by cell (markdown bold markers stripped, \| unescaped).  prereg_problems() compares every one with the pinned text, UNDER ITS HEADING.
PREREG_CELLS_JSON = r'''{
"1.": {
"heading": "1. The arms (box 2; each through the trainer's own launch gate; both seeds)",
"rows": [
[
"arm",
"what",
"role"
],
[
"A (DAG-ORC)",
"DAG2-0's argv, plus `--ego-perturb-p 0.3 --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed <s>`, with its own `--out`",
"decisive"
],
[
"CTRL = DAG2-0_s",
"reused, not retrained. Banked run `v8f-dagger2/M2b_s{s}_dag20/`, ckpt sha256 `603b9151f5119886\u2026` / `f4234ac11854076b\u2026`",
"the control: the same 3,000-step fine-tune from DAG_s without the lever"
],
[
"REP",
"DAG2-0's argv with `--seed 10` (init DAG_s0) / `--seed 11` (init DAG_s1), with its own `--out`",
"the replicate: the control's recipe run again (\u00a73)"
],
[
"B (DAG-ORC-01)",
"A's argv with `--ego-perturb-p 0.1`",
"the dose fallback. Read only if A is HARMS (\u00a77); otherwise reported beside"
]
]
},
"3.": {
"heading": "3. The replicate floors and the bar (every Read-A row, defined now)",
"rows": [
[
"symbol",
"definition",
"when it is computed"
],
[
"RF0_r",
"max over s of |d_r,s(DAG-0 \u2212 ORC-0)|. Banked M2b-init pair: same argv except `--out` and the default-off `--dagger-p 0`; same seed; builds `b43c7a9a` / `853f2fc3`. Causal dumps: `v8f-dagger-t1/dump_DAG0_s*_m10_causal_heads`, `v8f-orecover-t1/dump_ORC0_s*_m10_causal_heads`",
"P-0, before any DAG-ORC rollout is read, 0 GPU. Its output is frozen by sha256 into the reader's config. Its inputs must match their `PINS.txt` sha256"
],
[
"RF_r",
"max over s of |d_r,s(REP \u2212 CTRL)|",
"after REP's rollouts, before any A or B row is read"
],
[
"F_r",
"ADE6: F = 0.1207 m (F_M2b, `MODEL_REGISTRY.md` \u00a717, `H-V8F-SEED-1`). Every other row: 0, because no floor is registered and the row-equivalent is RF0_r itself",
"now"
],
[
"bar_r",
"2 \u00d7 max(F_r, RF0_r, RF_r), one value per row, applied at both seeds, inclusive. For every non-ADE row this is 2 \u00d7 max(RF0_r, RF_r), in the row's own units",
"as its inputs"
]
]
},
"4.": {
"heading": "4. The rows",
"rows": [
[
"row",
"contrast, per seed",
"role"
],
[
"R-CRAWL",
"every row of C_s is not separated-worse for X \u2212 P10",
"decisive"
],
[
"R-MOVE",
"every row of C_s clears for X \u2212 CTRL (separated-better AND \u2265 bar_r)",
"decisive"
],
[
"R-TLAT / R-TLON",
"the T head's lat / lon macro-F1 at ticks \u2265 1 (the T1-HEADS definition, matched cells), X \u2212 CTRL: not separated-worse",
"guard"
],
[
"R-ADE",
"T1-kin plain ADE6, X \u2212 CTRL: not separated-worse",
"guard"
],
[
"R-T0",
"T0 own-code ADE6, X \u2212 CTRL: point \u2264 +F/2 = 0.0604 m",
"guard"
],
[
"R-NEW",
"no row of D_s fires",
"guard"
],
[
"R-KEEP",
"T1-kin ADE6, X \u2212 DAG: point \u2264 +F/2 = 0.0604 m",
"adoption"
],
[
"R-NOBASE",
"no row r of D_s has X \u2212 P10 separated-worse AND X \u2212 DAG separated-worse with the worsening \u2265 bar_r",
"adoption"
]
]
},
"5.": {
"heading": "5. The four metric families (all reported, per seed, per family, never pooled)",
"rows": [
[
"family",
"T1-kin (primary): Read A rows, X \u2212 CTRL, X \u2212 P10, X \u2212 DAG",
"T0 (box 2's eval): X \u2212 CTRL, beside",
"not computable, with the reason"
],
[
"LONGITUDINAL",
"speed MAE (target speed), along-track MAE, accel MAE; distance keeping DK-VIOL / DK-HW / DK-TG (`read_dk`, LEAD windows only)",
"speed, accel, along-track MAE; headway, time gap, TTC, LEAD-VIOL (`orc_families.py`, unchanged CLI)",
"none. The anti-echo block is reported beside every longitudinal number"
],
[
"LATERAL",
"heading, yaw-rate, curvature, cross-track, L1 lateral-accel, L2",
"heading, curvature, yaw-rate, cross-track",
"none"
],
[
"TACTICAL",
"TAC-LAT / TAC-LON (plan-derived decision agreement); R-TLAT / R-TLON (T-head macro-F1, ticks \u2265 1)",
"lat / lon head macro-F1 and accuracy; goal-head macro-F1, 22 tokens (`v8f_paired_families.py`)",
"T1-kin goal-head F1 and head accuracy: the paired tool reads eval `run` dumps, not tick dumps (as ORC \u00a73.6). Anchor selection: UNAVAILABLE by design (one path per window)"
],
[
"STRATEGIC (\u26d4 not paused for v8F, PI ruling 2026-10-17)",
"S-POS@30 / 50 / 80 against CTRV, per run",
"S head checkpoint error and strategic rows of the eval JSON",
"paired S-POS across arms is not computed (it is read against CTRV within a run, as in DAGGER2)"
]
]
},
"7.": {
"heading": "7. Outcomes, committed now (arm X = A; X = B only on A's HARMS)",
"rows": [
[
"outcome",
"condition",
"committed next"
],
[
"REFUSED",
"any control fails",
"fix and re-read; no row is reported"
],
[
"HARMS",
"controls pass; a guard (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW) fails at \u2265 1 seed",
"A: not adopted; read B under this same table (train B first if the PI dropped it from the queue). B: not adopted; the perturbation cannot be dosed under the guards; NEXT-CRAWL on DAG states (\u00a78)"
],
[
"FIXED / ADOPTED",
"guards held; R-MOVE and R-CRAWL at both seeds; R-KEEP and R-NOBASE at both seeds",
"adopt X as the base. Re-run Read A on X: DOMINANT there is the dominance claim. Then X with DAG-OACC's lever, if DAG-OACC is adopted"
],
[
"FIXED / NOT ADOPTED",
"guards held; R-MOVE and R-CRAWL at both seeds; R-KEEP or R-NOBASE fails at \u2265 1 seed",
"the lever is attributed, but X cannot replace DAG. Next: DAG-ORC-SWAP, under its own PREREG: DAG2's argv plus this perturbation at X's dose, control DAG2 (banked). The swap kept own-state skill (DAG2 \u2212 DAG2-0 ADE6 \u22120.22 / \u22120.16 m)"
],
[
"PARTIAL / ADOPTED",
"guards held; R-MOVE at both seeds; R-CRAWL fails at \u2265 1 seed; R-KEEP and R-NOBASE at both seeds",
"adopt X. The crawl rows that still fail R-CRAWL go to NEXT-CRAWL on X's states"
],
[
"PARTIAL / NOT ADOPTED",
"guards held; R-MOVE at both seeds; R-CRAWL fails at \u2265 1 seed; R-KEEP or R-NOBASE fails at \u2265 1 seed",
"as FIXED / NOT ADOPTED: DAG-ORC-SWAP"
],
[
"PARTIAL / ONE SEED",
"guards held; R-MOVE at exactly one seed",
"not adopted, and never called an effect of the lever (`H-ESTIM-SEED-1`). NEXT-CRAWL on DAG states"
],
[
"NO EFFECT",
"guards held; R-MOVE at neither seed",
"own-state levers have now failed twice on DAG (DAgger iteration 2, then the perturbation). The state channel is eliminated as the crawl's lever. NEXT-CRAWL on DAG states"
]
]
}
}'''
PREREG_CELLS = json.loads(PREREG_CELLS_JSON)
PREREG_PHRASES = (      # (section prefix, verbatim phrase): the constants and rule texts this reader codes outside the tables
    ("3.", "F = 0.1207 m"), ("3.", "2 × max(F_r, RF0_r, RF_r)"), ("3.", "0.38⁴ ≈ 0.02"), ("3.", "inclusive"), ("3.", "Its output is frozen by sha256"),
    ("4.", "14 rows at s0 and 13 at s1"), ("4.", "accel MAE and DK-VIOL"), ("4.", "curvature MAE, L2, M1."), ("4.", "curvature MAE, L2, M1, yaw-rate MAE."), ("4.", "0.0604 m"),
    ("1.", "d186fd0bcfa7ee9f"), ("1.", "1d12ec2eb825990d"), ("1.", "603b9151f5119886"), ("1.", "f4234ac11854076b"), ("1.", "--seed 10"), ("1.", "--seed 11"),
    ("1.", "--ego-perturb-p 0.3 --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed <s>"),
    ("1.", "--steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0 --dagger-p 0"),
    ("2.", "B = 2,000"), ("2.", "141 episodes / 2,399 windows"),
    ("6.", "K-GATE"), ("6.", "K-PAIR"), ("6.", "K-INIT"), ("6.", "K-REACH"), ("6.", "K-BUILD"), ("6.", "K-P0"), ("6.", "K-ZERO"), ("6.", "K-RC"),
    ("6.", "0.3 ± 0.02 for A and 0.1 ± 0.02 for B, and is exactly 0 for REP"), ("6.", "REP − REP reads exactly 0.0 with CI [0, 0] on every row"),
    ("6.", "RF0's frozen sha256 predates every A / REP / B rollout's mtime"), ("6.", "0 differences"),
    ("7.", "REFUSED is decided first, then HARMS"), ("7.", "adoption is decided last"), ("7.", "B is a second shot at the guards, taken only after A's HARMS"),
    ("5.", "not paused"),
)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------- the PREREG pin: raw sha256 + every table cell verbatim under its heading
def split_md_row(ln):
    ln = ln.strip()
    body, cells, cur, i = ln[1:-1], [], "", 0
    while i < len(body):
        if body[i] == chr(92) and i + 1 < len(body) and body[i + 1] == "|":
            cur += "|"
            i += 2
            continue
        if body[i] == "|":
            cells.append(cur)
            cur = ""
            i += 1
            continue
        cur += body[i]
        i += 1
    cells.append(cur)
    return [c.strip().replace("**", "") for c in cells]


def md_sections(text):
    parts = re.split(r"^## ", text, flags=re.M)
    return {p.split("\n", 1)[0]: p for p in parts[1:]}


def first_table(sec):
    rows, started = [], False
    for ln in sec.splitlines():
        if ln.startswith("|"):
            started = True
            rows.append(split_md_row(ln))
        elif started:
            break
    return [r for r in rows if not set("".join(r)) <= set("-: ")]


def table_problems(text, cells=None):
    """-> [] iff, for each registered section, the section HEADING is verbatim and the FIRST table under it equals the registered rows cell by cell."""
    cells = PREREG_CELLS if cells is None else cells
    secs, probs = md_sections(text), []
    for key, want in cells.items():
        got_h = secs.get(want["heading"])
        if got_h is None:
            probs.append(f"section {key}: the heading {want['heading']!r} is not in the PREREG")
            continue
        got = first_table(got_h)
        if len(got) != len(want["rows"]):
            probs.append(f"section {key}: the table has {len(got)} rows, the reader codes {len(want['rows'])}")
        for i, w in enumerate(want["rows"]):
            g = got[i] if i < len(got) else None
            if g != w:
                probs.append(f"section {key} row {i} ({w[0]!r}): the PREREG reads {g!r}, the reader codes {w!r}")
    return probs


def phrase_problems(text, phrases=PREREG_PHRASES):
    secs = md_sections(text)
    probs = []
    for key, ph in phrases:
        sec = next((v for k, v in secs.items() if k.startswith(key)), None)
        if sec is None or re.sub(r"\s+", " ", ph) not in re.sub(r"\s+", " ", sec.replace("**", "")):
            probs.append(f"section {key}: the phrase {ph!r} is not verbatim under its heading")
    return probs


def prereg_problems(path, pin=None):
    """The file is the pinned one (raw-byte sha256 of the 3b7b404e blob; ``pin`` overrides it for tests only), every registered table cell is verbatim under its heading, and every coded phrase
    is where it is coded from."""
    pin = PREREG_SHA256 if pin is None else pin
    if not os.path.isfile(path):
        return [f"{path} is missing"]
    raw = open(path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != pin:
        return [f"{path} is not the pinned PREREG_V8F_DAG_ORC.md (sha256 {hashlib.sha256(raw).hexdigest()[:12]}..., pinned {pin[:12]}...)"]
    text = raw.decode("utf-8")
    return table_problems(text) + phrase_problems(text)


# ---------------------------------------------------------------------------------------------------------------- the outcome table (section 7), coded from the registered cells
OUTCOME_TABLE = tuple(tuple(r) for r in PREREG_CELLS["7."]["rows"][1:])          # (outcome, condition, committed next), the PREREG's order
OUTCOME_NAMES = tuple(r[0] for r in OUTCOME_TABLE)
ADOPTING = ("FIXED / ADOPTED", "PARTIAL / ADOPTED")
UNPLACED_TEXT = ("UNPLACED: this reading matches no row (or more than one row) of the PREREG's section-7 table. enumerate_cells.py proved the table complete and exclusive over its 524,288 "
                 "cells, so this path is unreachable through real cells: a reader / PREREG gap, not a result. No outcome word is invented; the PI / Master Mind decides, WITHOUT looking at which way the numbers fall.")


def table_row(name):
    return next((c, n) for nm, c, n in OUTCOME_TABLE if nm == name)


def outcome_rows(ctrl, held, rc, rm, keep, nobase):
    """Section 7, each row as its own literal predicate.  ``held[g][s]`` (g in GUARDS), ``rc[s]`` R-CRAWL, ``rm[s]`` R-MOVE, ``keep[s]`` R-KEEP, ``nobase[s]`` R-NOBASE: booleans per seed.
    Precedence (section 7): REFUSED first, then HARMS; with the guards held R-MOVE partitions the rest (both seeds / one seed / neither), R-CRAWL splits FIXED from PARTIAL, adoption is last."""
    guards_held = all(held[g][s] for g in GUARDS for s in SEEDS)
    rm_both, rm_n, rc_both = all(rm[s] for s in SEEDS), sum(bool(rm[s]) for s in SEEDS), all(rc[s] for s in SEEDS)
    adopt_ok = all(keep[s] and nobase[s] for s in SEEDS)
    return {
        "REFUSED": not ctrl,
        "HARMS": ctrl and not guards_held,
        "FIXED / ADOPTED": ctrl and guards_held and rm_both and rc_both and adopt_ok,
        "FIXED / NOT ADOPTED": ctrl and guards_held and rm_both and rc_both and not adopt_ok,
        "PARTIAL / ADOPTED": ctrl and guards_held and rm_both and not rc_both and adopt_ok,
        "PARTIAL / NOT ADOPTED": ctrl and guards_held and rm_both and not rc_both and not adopt_ok,
        "PARTIAL / ONE SEED": ctrl and guards_held and rm_n == 1,
        "NO EFFECT": ctrl and guards_held and rm_n == 0,
    }


def decide_orc(ctrl, held, rc, rm, keep, nobase):
    """-> {"OUTCOME", "matching_rows_in_table_order", "adopts", "flags"}.  Exactly one row must match; otherwise UNPLACED (defensive; unreachable through real cells)."""
    r = outcome_rows(ctrl, held, rc, rm, keep, nobase)
    matching = [n for n in OUTCOME_NAMES if r.get(n)]
    out = matching[0] if len(matching) == 1 else "UNPLACED"
    flags = {"controls_pass": bool(ctrl), "guards_held_per_seed": {f"s{s}": {g: bool(held[g][s]) for g in GUARDS} for s in SEEDS},
             "R-CRAWL": {f"s{s}": bool(rc[s]) for s in SEEDS}, "R-MOVE": {f"s{s}": bool(rm[s]) for s in SEEDS},
             "R-KEEP": {f"s{s}": bool(keep[s]) for s in SEEDS}, "R-NOBASE": {f"s{s}": bool(nobase[s]) for s in SEEDS}}
    return {"OUTCOME": out, "matching_rows_in_table_order": matching, "adopts": out in ADOPTING, "flags": flags}


def next_for_arm(name, arm):
    """The committed next of row ``name`` for arm ``arm``: HARMS is the one cell with an arm-specific text ('A: ... B: ...')."""
    cond, nxt = table_row(name)
    if name == "HARMS":
        i = nxt.find("B: not adopted")
        return nxt[:i].strip() if arm == "A" else nxt[i:].strip()
    return nxt


def outcome_block(d, arm):
    out = d["OUTCOME"]
    if out == "UNPLACED":
        return {**d, "arm": arm, "condition": None, "committed_next": None, "note": UNPLACED_TEXT}
    cond, _ = table_row(out)
    return {**d, "arm": arm, "condition": cond, "committed_next": next_for_arm(out, arm)}


def cascade(a_block, b_block):
    """Section 7, the dose cascade.  A is read first.  B decides only after A's HARMS; otherwise B is a dose-response row beside that decides nothing and never promotes B over A.
    -> {"final_arm", "final", "b_status"}."""
    if a_block["OUTCOME"] != "HARMS":
        return {"final_arm": "A", "final": a_block["OUTCOME"], "b_status": ("not read" if b_block is None else "beside: dose-response row, decides nothing, never pooled with A")}
    if b_block is None:
        return {"final_arm": "A", "final": "HARMS", "b_status": "PENDING: train B (p = 0.1) through its gate and read it under this same table"}
    return {"final_arm": "B", "final": b_block["OUTCOME"], "b_status": "decisive after A's HARMS (labelled p = 0.1; own floors and seeds; never pooled with A)"}


# ---------------------------------------------------------------------------------------------------------------- rows, bars, clears, fires (section 3 and 4)
class RowNotComputed(Exception):
    pass


def _row(label, row):
    if row is None or row.get("status") == "NOT COMPUTED":
        raise RowNotComputed(label)
    return row


def worsening(label, row):
    """Positive = worse, in the row's own units: an error row (lower is better) worsens by +delta, an agreement row (higher is better) by -delta."""
    d = float(_row(label, row)["delta"])
    return d if DIRECTION[label] == "lower" else -d


def sep_worse(label, row):
    return bool(_row(label, row)["separated"] and worsening(label, row) > 0)


def sep_better(label, row):
    return bool(_row(label, row)["separated"] and worsening(label, row) < 0)


def bar_for(label, rf0, rf):
    """bar_r = 2 * max(F_r, RF0_r, RF_r); F_r = F_ADE6 for ADE6, 0 for every other row (its row-equivalent is RF0_r itself).  One value per row, applied at both seeds."""
    return 2.0 * max(F_ADE6 if label == "ADE6" else 0.0, abs(rf0[label]), abs(rf[label]))


def clears(label, row, bar):
    """R-MOVE, per row: X - CTRL is separated-better AND the improvement is >= bar_r (inclusive)."""
    return bool(sep_better(label, row) and -worsening(label, row) >= bar)


def fires(label, x_p10, dag_p10, x_ctrl, bar):
    """R-NEW, per row of D_s: (1) X - P10 separated-worse, (2) DAG - P10 NOT separated-worse (the row is in the domain), (3) X - CTRL separated-worse with the worsening >= bar_r (inclusive)."""
    return bool(sep_worse(label, x_p10) and not sep_worse(label, dag_p10) and sep_worse(label, x_ctrl) and worsening(label, x_ctrl) >= bar)


def domain(dag_p10):
    """D_s: every Read-A row not separated-worse for DAG - P10 at s, minus the exempt rows."""
    return [l for l in LABELS if l not in EXEMPT and not sep_worse(l, dag_p10[l])]


def compose_seed(s, t, g, rf0, rf):
    """The PREREG's per-seed composition from the readers' rows.
    t = {"X_P10", "X_CTRL", "X_DAG", "DAG_P10"}: {Read-A label: row}; g = {"R-TLAT": F1 row, "R-TLON": F1 row, "T0_point": float, "KEEP_point": float}.  -> flags + detail."""
    bars = {l: bar_for(l, rf0, rf) for l in LABELS}
    c = CRAWL_SET[s]
    crawl_rows = {l: not sep_worse(l, t["X_P10"][l]) for l in c}
    move_rows = {l: clears(l, t["X_CTRL"][l], bars[l]) for l in c}
    d_s = domain(t["DAG_P10"])
    fire_rows = {l: fires(l, t["X_P10"][l], t["DAG_P10"][l], t["X_CTRL"][l], bars[l]) for l in d_s}
    nobase_rows = {l: bool(sep_worse(l, t["X_P10"][l]) and sep_worse(l, t["X_DAG"][l]) and worsening(l, t["X_DAG"][l]) >= bars[l]) for l in d_s}
    f1 = lambda r: bool(r["separated"] and float(r["delta"]) < 0)      # noqa: E731  macro-F1, higher is better: separated-worse = separated and negative
    guards = {"R-TLAT": not f1(g["R-TLAT"]), "R-TLON": not f1(g["R-TLON"]), "R-ADE": not sep_worse("ADE6", t["X_CTRL"]["ADE6"]),
              "R-T0": bool(g["T0_point"] <= HALF_F), "R-NEW": not any(fire_rows.values())}
    flags = {"R-CRAWL": all(crawl_rows.values()), "R-MOVE": all(move_rows.values()), "R-KEEP": bool(g["KEEP_point"] <= HALF_F), "R-NOBASE": not any(nobase_rows.values())}
    exempt = {l: {"X_minus_P10_separated_worse": sep_worse(l, t["X_P10"][l]), "DAG_minus_P10_separated_worse": sep_worse(l, t["DAG_P10"][l]),
                  "X_minus_CTRL_separated_worse": sep_worse(l, t["X_CTRL"][l])} for l in EXEMPT}
    return {"flags": flags, "guards": guards, "bars": bars, "crawl_set": list(c), "R-CRAWL_rows": crawl_rows, "R-MOVE_rows": move_rows, "domain": d_s,
            "R-NEW_fires": fire_rows, "R-NOBASE_violations": nobase_rows, "exempt_reported": exempt, "T0_point": g["T0_point"], "KEEP_point": g["KEEP_point"]}


def decide_arm(ctrl_ok, per_seed):
    """per_seed[s] = compose_seed(...)."""
    held = {g: {s: per_seed[s]["guards"][g] for s in SEEDS} for g in GUARDS}
    f = lambda k: {s: per_seed[s]["flags"][k] for s in SEEDS}        # noqa: E731
    return decide_orc(ctrl_ok, held, f("R-CRAWL"), f("R-MOVE"), f("R-KEEP"), f("R-NOBASE"))


# ---------------------------------------------------------------------------------------------------------------- RF0: loaded only if it is the frozen file
def rf0_problems(path, sha_file=None, pin=None):
    """-> (problems, rf0 dict).  The file's sha256 must equal the pin AND the .sha256 beside it; schema and the 19 labels must be exact; every RF0 value a finite number >= 0 that equals the row's
    own max over seeds of |delta|."""
    pin = RF0_SHA256 if pin is None else pin
    if not os.path.isfile(path):
        return [f"{path} is missing"], None
    got = sha256_file(path)
    if got != pin:
        return [f"{path}: sha256 {got[:16]}..., the reader pins {pin[:16]}... (RF0 is frozen; a changed file is refused)"], None
    sf = sha_file or (os.path.splitext(path)[0] + ".sha256")
    if not os.path.isfile(sf):
        return [f"{sf} is missing"], None
    rec = open(sf, encoding="utf-8").read().split()
    if not rec or rec[0] != got:
        return [f"{sf} records {rec[:1]}, the file hashes to {got[:16]}..."], None
    try:
        d = json.load(open(path, encoding="utf-8"))
    except ValueError as e:
        return [f"{path} is not JSON: {e}"], None
    bad = []
    if d.get("schema") != RF0_SCHEMA:
        bad.append(f"schema {d.get('schema')!r}, expected {RF0_SCHEMA!r}")
    rows, rf0 = d.get("rows") or {}, d.get("RF0") or {}
    if list(rows) != list(LABELS) or list(rf0) != list(LABELS):
        bad.append(f"the rows are {list(rows)}, the reader needs exactly {list(LABELS)}")
        return bad, None
    for l in LABELS:
        v = rf0[l]
        ds = [abs(float(rows[l]["per_seed"][f"s{s}"]["delta"])) for s in SEEDS]
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
            bad.append(f"RF0[{l}] = {v!r} is not a finite number >= 0")
        elif v != max(ds):
            bad.append(f"RF0[{l}] = {v} is not max over seeds of |d| = {max(ds)}")
    failed = [k for k, c in (d.get("controls") or {}).items() if not c.get("PASS")]
    if failed or not d.get("controls"):
        bad.append(f"P-0's own controls: failed {failed} / none recorded")
    return bad, (None if bad else {l: float(rf0[l]) for l in LABELS})


def k_p0(rf0_path, rollout_files, check_inputs=True):
    """K-P0: RF0 is the frozen file; its inputs (the tool files of each PINS.txt, and the dump digests it recorded) still hash as recorded; its mtime predates every A / REP / B rollout file."""
    probs, rf0 = rf0_problems(rf0_path)
    detail = {}
    if rf0 is not None and check_inputs:
        d = json.load(open(rf0_path, encoding="utf-8"))
        for k, v in d["inputs"].items():
            if k.startswith("PINS.txt:"):
                lines = [ln.strip() for ln in open(v["path"], encoding="utf-8") if ln.strip()]
                tree = next(ln.split("=", 1)[1] for ln in lines if ln.startswith("tree="))
                for ln in lines:
                    if ln.startswith("file="):
                        rel, sha = ln[5:].rsplit(" ", 1)
                        p = f"{tree}/{rel}"
                        if not os.path.isfile(p) or sha256_file(p) != sha:
                            probs.append(f"{k}: {p} no longer hashes to its PINS.txt value")
                if sha256_file(v["path"]) != v["sha256"]:
                    probs.append(f"{k}: PINS.txt itself changed")
            elif k.startswith("dump:"):
                fs = sorted(glob.glob(os.path.join(v["path"], "ep*.npz")))
                h = hashlib.sha256()
                for f in fs:
                    h.update(os.path.basename(f).encode() + b"\0" + sha256_file(f).encode() + b"\n")
                if len(fs) != v["n_files"] or h.hexdigest() != v["sha256_of_file_hashes"]:
                    probs.append(f"{k}: the dump directory no longer hashes to the recorded digest ({len(fs)} files)")
    if rf0 is not None:
        t_frozen = os.path.getmtime(rf0_path)
        late = [f for f in rollout_files if os.path.isfile(f) and os.path.getmtime(f) <= t_frozen]
        detail["rollout_files_checked"] = sum(os.path.isfile(f) for f in rollout_files)
        if late:
            probs.append(f"{len(late)} A / REP / B rollout file(s) are not newer than the frozen RF0 (e.g. {late[0]}): RF0 must predate every rollout")
    return probs, rf0, detail


# ---------------------------------------------------------------------------------------------------------------- run-record controls (pure on the records dag_read.train_records returns)
def parse_flags(argv):
    """argv -> {flag: [values-of-occurrence-1, ...]}; EVERY value after a flag is kept (the DAGGER reader's own parse_flags)."""
    out, i, argv = {}, 0, list(argv)
    while i < len(argv):
        t = argv[i]
        if t.startswith("--"):
            j, vals = i + 1, []
            while j < len(argv) and not argv[j].startswith("--"):
                vals.append(argv[j])
                j += 1
            out.setdefault(t, []).append(vals)
            i = j
        else:
            i += 1
    return out


def _norm(vs):
    """Occurrence lists of token lists -> numerically normalised (so '0.30' == '0.3', '1e-4' == '0.0001'); non-numeric tokens (paths) compare as strings."""
    def tok(x):
        try:
            return repr(float(x))
        except ValueError:
            return str(x).replace("\\", "/")
    return [[tok(x) for x in occ] for occ in vs]


def flag_diff(fa, fb, ignore=("--out",)):
    """Flags (parse_flags dicts) whose normalised values differ, ignoring ``ignore``."""
    ka = {k: _norm(v) for k, v in fa.items() if k not in ignore}
    kb = {k: _norm(v) for k, v in fb.items() if k not in ignore}
    return sorted(k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k))


def a_flags(ctrl_flags, seed, p=P_A):
    """The flags A must carry: DAG2-0's, plus --ego-perturb-p <p> (replacing its 0) and the five law flags (section 1)."""
    fl = {k: v for k, v in ctrl_flags.items()}
    fl["--ego-perturb-p"] = [[repr(p)]]
    for k, v in LAW_A.items():
        fl[k] = [list(v)]
    fl["--ego-perturb-seed"] = [[str(seed)]]
    return fl


def k_pair(arm, seed, flags, ctrl_flags, a_flags_rec=None):
    """K-PAIR (section 6).  A_s: DAG2-0_s's argv except --out, --ego-perturb-p = 0.3 and the five A-only law flags (values in section 1).  REP_s: except --out and --seed (10 / 11).
    B_s: A_s's argv except --out and --ego-perturb-p = 0.1.  -> (ok, problems)."""
    if arm == "A":
        want = a_flags(ctrl_flags, seed, P_A)
        d = flag_diff(flags, want)
        return not d, ([f"A_s{seed}'s argv differs from DAG2-0's + the registered law in {d}"] if d else "argv(A) = argv(DAG2-0) + --ego-perturb-p 0.3 + the five law flags")
    if arm == "REP":
        want = {k: v for k, v in ctrl_flags.items()}
        want["--seed"] = [[str(REP_SEED[seed])]]
        d = flag_diff(flags, want)
        return not d, ([f"REP_s{seed}'s argv differs from DAG2-0's with --seed {REP_SEED[seed]} in {d}"] if d else f"argv(REP) = argv(DAG2-0) with --seed {REP_SEED[seed]}")
    if arm == "B":
        want = a_flags(ctrl_flags, seed, P_B)
        d = flag_diff(flags, want)
        bad = [f"B_s{seed}'s argv differs from DAG2-0's + the registered law at p = 0.1 in {d}"] if d else []
        if a_flags_rec is not None:
            d2 = flag_diff(flags, a_flags_rec)
            if d2 != ["--ego-perturb-p"]:
                bad.append(f"B_s{seed}'s argv differs from A_s{seed}'s in {d2}, expected exactly ['--ego-perturb-p']")
        return not bad, bad or "argv(B) = argv(A) with --ego-perturb-p 0.1"
    return False, [f"unknown arm {arm!r}"]


def one_value(flags, name):
    occ = flags.get(name)
    if not occ or len(occ) != 1 or len(occ[0]) != 1:
        return False, None
    return True, occ[0][0]


def k_registered(arm, seed, flags):
    """The fine-tune is NAMED on the argv (a default would not be a test): steps, lr, warmup, p-gt, --ego-perturb-p, --dagger-p 0, --arm M2b, --seed, --init-ckpt tail."""
    bad = []
    want_seed = REP_SEED[seed] if arm == "REP" else seed
    p = {"A": P_A, "B": P_B, "REP": 0.0, "CTRL": 0.0}[arm]
    for name, w in (("--steps", STEPS), ("--lr", LR), ("--warmup", WARMUP), ("--p-gt-start", PGT), ("--p-gt-end", PGT), ("--ego-perturb-p", p), ("--dagger-p", 0.0), ("--seed", want_seed)):
        ok, v = one_value(flags, name)
        try:
            same = ok and float(v) == float(w)
        except (TypeError, ValueError):
            same = False
        if not same:
            bad.append(f"{name} {v!r}, registered {w}")
    ok, v = one_value(flags, "--arm")
    if not ok or v != "M2b":
        bad.append(f"--arm {v!r}, expected M2b")
    ok, v = one_value(flags, "--init-ckpt")
    if not ok or not str(v).replace("\\", "/").endswith(f"M2b_s{seed}_dag/ckpt_last.pt"):
        bad.append(f"--init-ckpt {v!r} is not .../M2b_s{seed}_dag/ckpt_last.pt (the init is DAG_s{seed})")
    return not bad, bad or "argv names the registered fine-tune"


def k_init(arm, seed, init_values):
    """K-INIT: every recorded init-checkpoint sha256 equals DAG_s's.  ``init_values`` = [(where, sha)] from dag_read.init_sha_values."""
    if not init_values:
        return False, "no record of the init checkpoint's sha256: the init is unproven"
    bad = [f"{w} = {str(v)[:16]}" for w, v in init_values if str(v) != DAG_CKPT_SHA[seed]]
    return not bad, (f"init sha256 differs from DAG_s{seed} ({DAG_CKPT_SHA[seed][:16]}...): " + "; ".join(bad)) if bad else {"init_sha256": DAG_CKPT_SHA[seed][:16] + "...", "sources": [w for w, _ in init_values]}


def k_reach(arm, built_p, n, eligible, nav_kept=None):
    """K-REACH: the lever REACHED training.  ego_perturb_n / ego_perturb_eligible lies in 0.3 +- 0.02 for A, 0.1 +- 0.02 for B, and is exactly 0 for REP (n == 0); the built p equals the argv p."""
    want = {"A": P_A, "B": P_B, "REP": 0.0}[arm]
    bad, det = [], {"ego_perturb_n": n, "ego_perturb_eligible": eligible, "ego_perturb_nav_kept": nav_kept}
    if built_p is None or abs(float(built_p) - want) > 1e-12:
        bad.append(f"built ego_perturb_p = {built_p!r}, argv names {want}")
    if not (isinstance(n, (int, float)) and isinstance(eligible, (int, float)) and eligible > 0 and 0 <= n <= eligible):
        bad.append(f"ego_perturb_n {n!r} / ego_perturb_eligible {eligible!r}: not a count pair with 0 <= n <= eligible, eligible > 0")
        return False, bad
    if arm == "REP":
        if n != 0:
            bad.append(f"REP ego_perturb_n = {n}, must be exactly 0")
    else:
        ratio = n / eligible
        det["ratio"] = ratio
        if not (want - REACH_TOL - 1e-12 <= ratio <= want + REACH_TOL + 1e-12):
            bad.append(f"{arm} perturbed fraction {n}/{eligible} = {ratio:.4f} outside {want} +- {REACH_TOL}")
    return not bad, bad or det


def k_build(rec):
    """K-BUILD: the launch commit's v8f_mini_train.py, v8f_mini_data.py and v8f_mini.py are byte-identical to 1d4f4fbb's, or every changed hunk sits behind a default-off flag whose bit-identity
    test passes on the launch commit.  ``rec`` = {"launch_commit", "base_commit", "files": {path: {"sha256_launch", "sha256_base"}}, "changed": [{file, hunk, behind_flag, default_off, bit_identity_test}]}.
    The diff is listed in the result (the record is returned verbatim)."""
    if not rec:
        return False, "no K-BUILD record in the manifest (build.launch_commit, build.files, build.changed)"
    bad = []
    if not re.fullmatch(r"[0-9a-f]{40}", str(rec.get("launch_commit") or "")):
        bad.append(f"launch_commit {rec.get('launch_commit')!r} is not a 40-hex commit")
    if not str(rec.get("base_commit") or "").startswith(BUILD_BASE_COMMIT):
        bad.append(f"base_commit {rec.get('base_commit')!r} is not {BUILD_BASE_COMMIT}")
    files = rec.get("files") or {}
    for f in BUILD_FILES:
        fr = files.get(f)
        if not fr or not fr.get("sha256_launch") or not fr.get("sha256_base"):
            bad.append(f"{f}: no launch / base sha256")
            continue
        if fr["sha256_launch"] != fr["sha256_base"]:
            hunks = [h for h in (rec.get("changed") or []) if h.get("file") == f]
            if not hunks:
                bad.append(f"{f} differs from {BUILD_BASE_COMMIT} and no changed hunk is listed")
            for h in hunks:
                if h.get("default_off") is not True or not h.get("behind_flag") or h.get("bit_identity_test") != "PASS":
                    bad.append(f"{f}: hunk {h.get('hunk')!r} is not behind a default-off flag with a PASSing bit-identity test")
    return not bad, bad or {"record": rec}


def rc_diffs(a, b, tol=1e-4):
    """Two Read-A row dicts (label -> row) equal: same labels, same separated flag, delta / lo / hi within tol.  -> list of differences (the DAG2 reader's rows_equal)."""
    bad = []
    for lab in sorted(set(a) | set(b)):
        ra, rb = a.get(lab), b.get(lab)
        if ra is None or rb is None:
            bad.append(f"{lab}: present on one side only")
            continue
        if ra.get("status") == "NOT COMPUTED" or rb.get("status") == "NOT COMPUTED":
            if ra.get("status") != rb.get("status"):
                bad.append(f"{lab}: NOT COMPUTED on one side only")
            continue
        for k in ("delta", "lo", "hi"):
            if abs(float(ra[k]) - float(rb[k])) > tol:
                bad.append(f"{lab}.{k}: {ra[k]} vs {rb[k]}")
        if bool(ra["separated"]) != bool(rb["separated"]):
            bad.append(f"{lab}.separated: {ra['separated']} vs {rb['separated']}")
    return bad


def is_exact_zero(rows):
    """K-ZERO / identity: every row reads exactly +0 [0, 0]."""
    return bool(rows) and all(r.get("status") == "NOT COMPUTED" or (r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0) for r in rows.values())


def floors_from(tables):
    """RF_r = max over seeds of |d_r,s(REP - CTRL)|; tables[s] = {label: row}."""
    return {l: max(abs(float(_row(l, tables[s][l])["delta"])) for s in SEEDS) for l in LABELS}


# ---------------------------------------------------------------------------------------------------------------- slot discovery (pure filesystem; no run output is parsed)
def slot_needs(arm, e):
    """The files one slot entry must name and have on disk.  CTRL and DAG are banked runs; A / REP / B are new."""
    need = []
    if arm != "DAG":
        td = e.get("train_dir")
        need += [("train_dir", td)] + [(f"train_dir/{f}", os.path.join(td, f) if td else None) for f in ("argv.json", "summary.json", "selfcheck.json")]
        need.append(("token", e.get("token")))
        for k in ("t0", "t0_fresh"):
            for f in ("json", "npz"):
                need.append((f"{k}.{f}", (e.get(k) or {}).get(f)))
        lanes = ("t1", "t1_logged")
    else:
        lanes = ("t1",)
        need.append(("train_dir", e.get("train_dir")))
    for k in lanes:
        t = e.get(k) or {}
        need += [(f"{k}.json", t.get("json")), (f"{k}.ticks", t.get("ticks"))]
        dd = t.get("dump_dir")
        need.append((f"{k}.dump_dir", dd))
    return need


def discover(manifest, arms=ARMS, seeds=SEEDS):
    """-> {arm: {seed: {"state": PRESENT | MISSING | PARTIAL, "missing": [what, ...]}}}."""
    out = {}
    for arm in arms:
        out[arm] = {}
        for s in seeds:
            e = ((manifest.get("arms") or {}).get(arm) or {}).get(str(s))
            if not e:
                out[arm][s] = {"state": "MISSING", "missing": ["no manifest entry"]}
                continue
            miss = []
            for what, p in slot_needs(arm, e):
                if not p or not os.path.exists(p):
                    miss.append(f"{what}: {p!r}")
                elif what.endswith("dump_dir") and len(glob.glob(os.path.join(p, "ep*.npz"))) < N_EPISODES:
                    miss.append(f"{what}: {p!r} holds {len(glob.glob(os.path.join(p, 'ep*.npz')))} of {N_EPISODES} episodes")
            out[arm][s] = {"state": "PRESENT" if not miss else "PARTIAL", "missing": miss}
    return out


def slot_summary(slots):
    return {arm: {f"s{s}": v["state"] for s, v in per.items()} for arm, per in slots.items()}


def families_report(rows_by_pair):
    """Section 5: group the Read-A rows of each contrast by family, with the tier and estimator stamps; STRATEGIC is NOT PAUSED.  ``rows_by_pair`` = {pair name: {label: row}} for one seed."""
    out = {}
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
        labs = [l for l in LABELS if FAMILY[l] == fam]
        out[fam] = {"tier": TIER_T1, "estimator": "paired episode-cluster bootstrap (B=2000, seed 0)", "rows": {l: {pn: ({k: rows[l].get(k) for k in ("delta", "lo", "hi", "separated")} if rows.get(l) and rows[l].get("status") != "NOT COMPUTED" else "NOT COMPUTED") for pn, rows in rows_by_pair.items()} for l in labs}}
    out["STRATEGIC"]["paused"] = False
    out["STRATEGIC"]["ruling"] = "NOT paused for v8F (PI ruling 2026-10-17)"
    return out


# ---------------------------------------------------------------------------------------------------------------- the heavy path (layers over the existing readers; imported lazily)
def mods():
    """-> (dag2_read, dag_read): the staged, sha-pinned copies (REF).  Imported lazily: the pure logic above needs neither."""
    if REF not in sys.path:
        sys.path.insert(0, REF)
    import dag2_read as D2
    if "dag_read" not in sys.modules:
        DR = D2.load_dag_read(f"{REF}/dag_read.py")
    else:
        DR = sys.modules["dag_read"]
    return D2, DR


REF_PINS = {"ref/dag2_read.py": "1425e8a3895c442b9841f3bcd6fd0893021d583e06485fde171b04bfa73fbc20", "ref/dag_read.py": "1cf56c7c30195ac47718bb07feef11531a6309cb01e0b28a34f926c6a696084c",
            "ref/readA_code/rr_lib.py": "c3800878cae80bf3018284528e68ed5f88dcebbab7d1125749942735e6cebc99", "ref/readA_code/readA.py": "722913b785f20e0458daed322ecd486fbc2a33106b44b1294f6b59d08d936fa6",
            "ref/readA_code/build_report.py": "c4e94920487381bfb22d06acdff2d79cbb7d97ba52ebb0707003c3e0968e820a",
            "ref/b1_eval_lead_block_k30.npz": "7a879baa2b6f4c140c2b64c9c098a38f84ecb04926da61fa8e96cfffc33073d9"}


def default_args(**kw):
    import types
    d = dict(tool_root=TREE, readers=f"{ART}/dagreread/reader_src", reada_code=f"{REF}/readA_code", lead_block=f"{REF}/b1_eval_lead_block_k30.npz",
             eval_join=f"{TREE}/{PK}/raw/join_eval141.npz", eval_grid=f"{TREE}/{PK}/raw/eval_grid_windows.json", eval_cache=f"{ART}/sfresh_inputs/cache", tracks=f"{ART}/sfresh_inputs/inputs/native_tracks.pt",
             t1hd_read=f"{TREE}/{PK}/raw/2026-10-26-t1hdread/t1hd_read.py", t1k_read=f"{TREE}/{PK}/raw/2026-10-26-t1kin-hacc/t1k_read.py",
             dag_read=f"{REF}/dag_read.py", dag_read_json=f"{ART}/v8f-dagger-t1/read/DAG_READ.json", reada_banked=f"{ART}/dagreread/raw/READ_A.json",
             t1kf_banked=f"{ART}/dagreread/raw/t1kf_DAG/T1KF_READ_A1_DAG.json", prereg=f"{HERE}/PREREG_V8F_DAG_ORC.md", rf0=f"{HERE}/P0_RF0.json", work_dir=None, n_boot=N_BOOT)
    d.update(kw)
    return types.SimpleNamespace(**d)


def static_problems(a):
    """Pins (the staged readers, Read A's banked inputs), the PREREG text and RF0.  -> list of problems."""
    D2, DR = mods()
    bad = [f"{k}: {p} is missing or has another sha256" for k, p in ((k, f"{HERE}/{k}") for k in REF_PINS)
           if not os.path.isfile(p) or sha256_file(p) != REF_PINS[k]]
    bad += D2.pin_problems(D2.static_paths(a))
    bad += [f"PREREG: {p}" for p in prereg_problems(a.prereg)]
    pr, _ = rf0_problems(a.rf0)
    bad += [f"RF0: {p}" for p in pr]
    return bad


class Rig:
    """The readers, constructed once: ReadA (Read A's rows), the T1-HEADS pipeline (RealBackend), t1k_read (comps)."""

    def __init__(self, a):
        self.a = a
        self.D2, self.DR = mods()
        self.A = self.D2.ReadA(a)                               # Read A first, as dag2_read does (it fixes the tool-root stack)
        self.np = self.A.np
        self.T = self.DR.load_t1hd(a.t1hd_read, a.tool_root)
        self.K = self.DR.load_t1k(a.t1k_read, a.tool_root)
        work = a.work_dir or os.path.join(HERE, "_work")
        os.makedirs(work, exist_ok=True)
        ns = self.T.parse(["--mode", "real", "--tool-root", a.tool_root, "--eval-cache", a.eval_cache, "--tracks", a.tracks, "--eval-grid", a.eval_grid, "--eval-join", a.eval_join,
                           "--banked-dir", ".", "--out", os.path.join(work, "t1hd_unused.json")])
        self.backend = self.DR.RealBackend(self.T, ns, a.n_boot, work)

    # -- Read A rows for X - Y on the windows of d_ref; STRATEGIC left NOT COMPUTED (filled by spos_pair)
    def pair_table(self, seed_dumps, cx_cache, name_x, name_y, d_ref):
        A, T = self.A, self.A.T
        def prep(n):
            if n not in cx_cache["P"]:
                src = seed_dumps[n]["fl"] if n == "P10" else seed_dumps[n]["cl"]
                cx_cache["P"][n] = T.prep(n, src, cx_cache["cx"])
            return cx_cache["P"][n]
        X = seed_dumps[name_x]["fl"] if name_x == "P10" else seed_dumps[name_x]["cl"]
        Y = seed_dumps[name_y]["fl"] if name_y == "P10" else seed_dumps[name_y]["cl"]
        return A.pair_rows(X, Y, d_ref, prep(name_x), prep(name_y))

    def spos_contrasts(self, tick_path):
        T = self.T
        R = self.backend._cache[tick_path][0]
        P = {nm: T.per_window(R["err"][nm]["pos_err_m"]) for nm in ("S", "CTRV")}
        return {f"S-POS@{arc}": P["S"][1][:, m] - P["CTRV"][1][:, m] for m, arc in enumerate(T.ARCS)}

    def spos_pair(self, tick_x, tick_y, ci):
        """The STRATEGIC rows of X - Y: the difference of the two runs' CTRV contrasts on the same windows, through the same estimator (T.srow -> paired episode-cluster bootstrap)."""
        cx, cy = self.spos_contrasts(tick_x), self.spos_contrasts(tick_y)
        out = {}
        for lab in SPOS_LABELS:
            r = self.T.srow(lab + " (difference of CTRV contrasts)", cx[lab], cy[lab], ci, self.a.n_boot, record=False)
            out[lab] = {"family": "STRATEGIC", "direction": "lower", "delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": bool(r["separated"]), "n": f"{r['n_windows']}/{r['n_episodes']}",
                        "mean_arm": None, "mean_ref": None, "separated_better": bool(r["separated"] and r["delta"] < 0), "separated_worse": bool(r["separated"] and r["delta"] > 0),
                        "kind": "difference of the two runs' (S - CTRV, ticks>=1) contrasts on the same windows"}
        return out


def with_spos(table, spos_rows):
    t = dict(table)
    t.update(spos_rows)
    return t


def main_static(a, as_json=True):
    bad = static_problems(a)
    print(json.dumps({"problems": bad, "prereg_sha256": PREREG_SHA256, "rf0_sha256": RF0_SHA256}, indent=1))
    print("DAG-ORC-READ-STATIC-OK" if not bad else "DAG-ORC-READ-STATIC-REFUSED")
    return 0 if not bad else 2


def banked_ctrl_dag_manifest():
    """The dry-run manifest: CTRL = DAG2-0 (v8f-dagger2 + v8f-dagger2-t1), DAG = DAG (v8f-dagger + v8f-dagger-t1), both banked; A / REP / B absent (they do not exist yet)."""
    m2 = json.load(open(f"{ART}/v8f-dagger2-t1/runs.json", encoding="utf-8"))
    m1 = json.load(open(f"{ART}/v8f-dagger-t1/runs.json", encoding="utf-8"))
    return {"seeds": [0, 1], "arms": {"CTRL": m2["arms"]["DAG0"], "DAG": m1["arms"]["DAG"]}, "provenance": "dry-run manifest: CTRL := DAG2-0 (banked), DAG := DAG (banked); A / REP / B absent"}


def main_dry(a, manifest_path, out):
    """--dry-run: the static checks, K-P0 against the banked inputs, slot discovery on the REAL banked CTRL and DAG slots, and the run-record controls of CTRL.  Ends OK with the DAG-ORC slots reported missing."""
    t0 = time.time()
    D2, DR = mods()
    ctl = DR.Ctl()
    bad = static_problems(a)
    ctl.add("STATIC", not bad, bad or "staged readers, Read A's banked inputs, PREREG text, RF0 pin")
    pr, rf0, det = k_p0(a.rf0, [])
    ctl.add("K-P0", not pr, pr or {"RF0_sha256": RF0_SHA256[:16] + "...", "inputs": "PINS.txt tool files + the recorded dump digests re-hashed", **det})
    man = json.load(open(manifest_path, encoding="utf-8")) if manifest_path else banked_ctrl_dag_manifest()
    slots = discover(man)
    present = {arm: all(slots[arm][s]["state"] == "PRESENT" for s in SEEDS) for arm in ARMS}
    ctl.add("SLOTS-BANKED", present["CTRL"] and present["DAG"], {"CTRL": slot_summary(slots)["CTRL"], "DAG": slot_summary(slots)["DAG"]})
    # CTRL's run records through the same controls a new run gets (summary, init, gate token, registered fine-tune)
    for s in SEEDS:
        e = man["arms"]["CTRL"][str(s)]
        rec = DR.train_records(e, KEY_INIT)
        tag = f"CTRL_s{s}"
        ctl.run(f"SUMMARY:{tag}", lambda rec=rec, s=s, tag=tag: DR.g_summary(tag, s, rec))
        ctl.run(f"K-INIT:{tag}", lambda rec=rec, s=s: k_init("CTRL", s, DR.init_sha_values(rec, KEY_INIT)))
        ctl.run(f"ARGV:{tag}", lambda rec=rec, s=s: k_registered("CTRL", s, rec["flags"]))
        ctl.run(f"K-GATE:{tag}", lambda rec=rec, e=e, tag=tag: DR.g_token(tag, rec, DR.read_token(e.get("token"))))
        sha = sha256_file(os.path.join(e["train_dir"], "ckpt_last.pt")) if os.path.isfile(os.path.join(e["train_dir"], "ckpt_last.pt")) else None
        ctl.add(f"CTRL-CKPT:{tag}", bool(sha and sha.startswith(CTRL_CKPT_PREFIX[s])), {"ckpt_sha256": (sha or "")[:16], "PREREG": CTRL_CKPT_PREFIX[s]})
        # K-PAIR's expected A / REP / B argv, built from CTRL's own banked argv (shows the controls would hold on a correct run)
        ctl.add(f"K-PAIR-selfcheck:{tag}", k_pair("A", s, a_flags(rec["flags"], s, P_A), rec["flags"])[0] and k_pair("REP", s, {**rec["flags"], "--seed": [[str(REP_SEED[s])]]}, rec["flags"])[0]
                and k_pair("B", s, a_flags(rec["flags"], s, P_B), rec["flags"], a_flags(rec["flags"], s, P_A))[0] and not k_pair("A", s, rec["flags"], rec["flags"])[0],
                "A / REP / B argv built from DAG2-0's banked argv pass K-PAIR; DAG2-0's own argv does not pass as A")
    res = {"DRY_RUN": True, "controls_PASS": ctl.flags(), "controls": ctl.items, "slots": slot_summary(slots), "slots_detail": {arm: {f"s{s}": v for s, v in per.items()} for arm, per in slots.items()},
           "new_arms_missing": {arm: [f"s{s}" for s in SEEDS if slots[arm][s]["state"] != "PRESENT"] for arm in ("A", "REP", "B")}, "wall_s": round(time.time() - t0, 1),
           "prereg": PREREG_ID, "prereg_sha256": PREREG_SHA256, "rf0_sha256": RF0_SHA256, "reader_sha256": sha256_file(os.path.abspath(__file__))}
    ok = not ctl.failed() and all(slots[arm][s]["state"] != "PRESENT" for arm in ("A", "REP", "B") for s in SEEDS)
    for k, v in ctl.items.items():
        print(("PASS " if v["PASS"] else "FAIL ") + k, str(v["detail"])[:160].replace("\n", " "))
    print("DAG-ORC slots (A / REP / B) reported missing:", json.dumps(res["new_arms_missing"]))
    if out:
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"wrote {out}  sha256 {sha256_file(out)}")
    print("ZZDAG-ORC-READ-DRY-OK" if ok else "ZZDAG-ORC-READ-DRY-FAIL")
    return 0 if ok else 1


def norm_path(p):
    return str(p).replace("\\", "/").rstrip("/")


def analyze_slot(rig, ctl, arm, seed, e, records):
    """One run through the readers' own controls.  ``records`` = 'full' (A / REP / B: K-GATE, K-INIT, the registered argv) | 'basic' (summary, init, gate token only) | 'none' (the banked DAG run).
    -> rec {tag, entry, pack, dump (T1-HEADS loader), t0 (arrays | None)}; adds every control to ``ctl``."""
    DR, np = rig.DR, rig.np
    tag, nb = f"{arm}_s{seed}", rig.a.n_boot
    rec = {"arm": arm, "seed": seed, "tag": tag, "entry": e}
    if records != "none":
        tr = DR.train_records(e, KEY_INIT)
        rec["train"] = tr
        run_seed = REP_SEED[seed] if arm == "REP" else seed
        ctl.run(f"SUMMARY:{tag}", lambda: DR.g_summary(tag, run_seed, tr))
        ctl.run(f"K-INIT:{tag}", lambda: k_init(arm, seed, DR.init_sha_values(tr, KEY_INIT)))
        ctl.run(f"K-GATE:{tag}", lambda: DR.g_token(tag, tr, DR.read_token(e.get("token"))))
        if records == "full":
            ctl.run(f"ARGV:{tag}", lambda: k_registered(arm, seed, tr["flags"]))
            if arm in ("A", "B", "REP"):
                def reach(tr=tr):
                    bp = DR.find_key(tr["sources"], "ego_perturb.ego_perturb_p")
                    ns, es, nk = DR.find_key(tr["sources"], "ego_perturb_n"), DR.find_key(tr["sources"], "ego_perturb_eligible"), DR.find_key(tr["sources"], "ego_perturb_nav_kept")
                    for what, found in (("ego_perturb_p", bp), ("ego_perturb_n", ns), ("ego_perturb_eligible", es)):
                        if not found:
                            return False, f"no built value {what!r} recorded by the trainer"
                        if len({json.dumps(v) for _, v in found}) > 1:
                            return False, f"{what!r} recorded with conflicting values {found}"
                    return k_reach(arm, bp[0][1], ns[0][1], es[0][1], nk[0][1] if nk else None)
                ctl.run(f"K-REACH:{tag}", reach)
    ckpt_tail = e.get("ckpt_tail") or (f"{os.path.basename(norm_path(e['train_dir']))}/ckpt_last.pt" if e.get("train_dir") else f"{tag}/ckpt_last.pt")
    rec["ckpt_tail"] = ckpt_tail
    t1 = e["t1"]
    rec["t1json"] = DR.jload(DR.need(t1["json"], f"{tag} t1v8f json"))
    rec["dump"] = DR.load_dump(t1["dump_dir"])
    rec["pack"] = pk = rig.backend.analyze_t1(t1, t1.get("tag") or tag, seed)
    for k, (ok, det) in pk.controls.items():
        ctl.add(f"{k}:{tag}", ok, det)
    ctl.run(f"T1-CONFIG:{tag}", lambda: DR.g_t1cfg(tag, rec["t1json"], ckpt_tail, "causal", pk.meta))
    if arm != "DAG":
        lg = e.get("t1_logged")
        if lg is None:
            raise DR.Refusal(f"{tag}: manifest has no 't1_logged' (K4 needs the --limit logged run's tick 0)")
        lpk = rig.backend.analyze_t1_light(lg, lg.get("tag") or f"{tag}-logged", seed)
        for k, (ok, det) in lpk.controls.items():
            ctl.add(f"{k}-logged:{tag}", ok, det)
        lj = DR.jload(DR.need(lg["json"], f"{tag} logged t1v8f json"))
        ctl.run(f"T1-CONFIG-logged:{tag}", lambda: DR.g_t1cfg(tag, lj, ckpt_tail, "logged", lpk.meta))
        t0f = e.get("t0_fresh")
        if not t0f:
            raise DR.Refusal(f"{tag}: manifest has no 't0_fresh' (K4 compares tick 0 with the arm's own T0 eval at --ages fresh)")
        ok, det = rig.backend.k4(lg, pk, t0f, seed)
        ctl.add(f"K4:{tag}", ok, det)
        t0e = e.get("t0")
        if not t0e:
            raise DR.Refusal(f"{tag}: manifest has no 't0' (box 2's standard T0 eval)")
        rec["t0json"], rec["t0arr"], t0_ok = DR.t0_arrays(t0e, ckpt_tail)
        ctl.add(f"T0-CKPT:{tag}", t0_ok, f"T0 eval json ckpt {rec['t0json'].get('ckpt')!r} vs .../{ckpt_tail}")
        t0fj = DR.jload(DR.need(t0f["json"], "fresh T0 json"))
        ctl.add(f"T0F-CKPT:{tag}", DR.tail2(str(t0fj.get("ckpt", ""))) == ckpt_tail, f"fresh T0 eval json ckpt {t0fj.get('ckpt')!r} vs .../{ckpt_tail}")
        ctl.add(f"T0-AGES:{tag}", (rec["t0json"].get("ages"), t0fj.get("ages")) == ("cycle", "fresh"), {"t0": rec["t0json"].get("ages"), "t0_fresh": t0fj.get("ages")})
    d = rec["dump"]
    ctl.add(f"DUMP-WINDOWS:{tag}", bool(len(d["ws"]) == pk.N and np.array_equal(d["clip_index"], pk.ci) and np.array_equal(d["ws"], pk.t0)), {"dump_n": int(len(d["ws"])), "ticks_n": int(pk.N)})
    ctl.add(f"GRID-COUNTS:{tag}", bool(pk.N == N_WINDOWS and len(np.unique(d["eid"])) == N_EPISODES), {"windows": int(pk.N), "episodes": int(len(np.unique(d["eid"])))})
    js = (rec["t1json"].get("paired_decision_grade") or {}).get("paired_model_minus_floor_t1", {}).get("ade_m")

    def r2():
        if js is None:
            return False, "the run json carries no paired_model_minus_floor_t1.ade_m to reproduce"
        A_, B_ = rig.K.comps(d["cl"], d["g"])["ade6_m"], rig.K.comps(d["fl"], d["g"])["ade6_m"]
        mk = np.isfinite(A_) & np.isfinite(B_)
        r = rig.T.PB(A_[mk], B_[mk], d["eid"][mk], n_boot=nb, seed=0)
        return abs(r["delta"] - float(js["delta"])) <= 1e-6, {"reader_delta": r["delta"], "analysis_delta": js["delta"]}
    ctl.run(f"R2-ADE6:{tag}", r2)
    return rec


def read_core(rig, ctl, man, mode, rf0, floors_dir=None):
    """The whole analysis.  mode 'read': arms A, CTRL, REP, DAG (+ B when present); mode 'rc': A := DAG2, CTRL := DAG2-0, DAG := DAG, floors 0, no REP.
    -> dict of tables, G rows, floors, per-arm compose and decisions (nothing is written here)."""
    DR, A, T, K, np, nb = rig.DR, rig.A, rig.T, rig.K, rig.np, rig.a.n_boot
    rc = mode == "rc"
    xs = ["A"] + (["B"] if (not rc and all(((man["arms"].get("B") or {}).get(str(s))) for s in SEEDS)) else [])
    arms = xs + ["CTRL", "DAG"] + ([] if rc else ["REP"])
    recs = {arm: {} for arm in arms}
    for arm in arms:
        for s in SEEDS:
            e = (man["arms"].get(arm) or {}).get(str(s))
            if not e:
                raise DR.Refusal(f"manifest has no {arm} seed {s}")
            recs[arm][s] = analyze_slot(rig, ctl, arm, s, e, "none" if arm == "DAG" else ("basic" if rc else "full"))
    # ---- K-PAIR (needs CTRL's flags) and the banked CTRL checkpoint
    for s in SEEDS:
        td = recs["CTRL"][s]["entry"]["train_dir"]
        sums = os.path.join(td, "SHA256SUMS.txt")
        sha = next((ln.split()[0] for ln in open(sums, encoding="utf-8") if ln.strip().endswith("ckpt_last.pt")), None) if os.path.isfile(sums) else None
        ctl.add(f"CTRL-CKPT:s{s}", bool(sha and (rc or sha.startswith(CTRL_CKPT_PREFIX[s]))), {"ckpt_sha256": (sha or "")[:16], "PREREG": CTRL_CKPT_PREFIX[s]})
        if not rc:
            ctrl_fl = recs["CTRL"][s]["train"]["flags"]
            for arm in ("A", "REP", "B"):
                if arm in arms:
                    ctl.run(f"K-PAIR:{arm}_s{s}", lambda arm=arm, s=s: k_pair(arm, s, recs[arm][s]["train"]["flags"], ctrl_fl, recs["A"][s]["train"]["flags"] if arm == "B" else None))
    # ---- Read A dumps, windows, tables (per seed)
    dumps = {s: {arm: A.dump(recs[arm][s]["entry"]["t1"]["dump_dir"]) for arm in arms} for s in SEEDS}
    cache = {}
    for s in SEEDS:
        ref = dumps[s]["CTRL"]
        for arm in arms:
            if arm == "CTRL":
                continue
            d = dumps[s][arm]
            same = all(np.array_equal(d[k], ref[k]) for k in ("clip_index", "ws", "eid", "g", "v0"))
            ctl.add(f"C4-WINDOWS:{arm}-vs-CTRL:s{s}", bool(same and len(d["eid"]) == N_WINDOWS and len(np.unique(d["eid"])) == N_EPISODES), "clip_index, ws, eid, g, v0 bit-equal; 2399 windows / 141 episodes")
            ctl.add(f"C3-FLOOR:{arm}-vs-CTRL:s{s}", bool(np.array_equal(d["fl"], ref["fl"])), "P10's tracks read no model: bit-equal")
        cache[s] = {"cx": A.tacdk_context(ref), "P": {}}
    for s in SEEDS:                                     # P10's dump = CTRL's `fl`
        dumps[s]["P10"] = dumps[s]["CTRL"]

    def fake_dagread(names):
        return {"beside": {"per_seed": {f"s{s}": {"per_run_t1heads": {n: recs[n][s]["pack"].extra for n in names}} for s in SEEDS}}}

    def pair_tab(x, y):
        """Read A's 19 rows for X - Y at both seeds.  STRATEGIC: for P10 the within-run (S - CTRV) contrast of X; otherwise the difference of the two runs' CTRV contrasts on the same windows."""
        cj, tj = {}, {}
        for s in SEEDS:
            cj[s], tj[s] = rig.pair_table(dumps[s], cache[s], x, y, dumps[s]["CTRL"])
        within = y == "P10"
        t = A.table(cj, tj, fake_dagread([x]) if within else {}, x if within else None)
        if not within:
            for s in SEEDS:
                t[s] = with_spos(t[s], rig.spos_pair(recs[x][s]["pack"].ticks_path, recs[y][s]["pack"].ticks_path, recs[x][s]["pack"].ci))
        return t

    # ---- PHASE 1, the floors (section 3: RF "after REP's rollouts, before any A or B row is read"): REP - CTRL, then the file, BEFORE any X contrast is computed
    rf0z = {l: 0.0 for l in LABELS}
    tabs = {}
    ident = {}
    if rc:
        rf0_, rf_ = rf0z, rf0z
    else:
        rf0_ = rf0
        ident["REP"] = pair_tab("REP", "REP")
        ctl.add("K-ZERO:REP-REP", all(is_exact_zero(ident["REP"][s]) for s in SEEDS), "REP - REP reads exactly 0.0 with CI [0, 0] on every row")
        ctl.add("K-ZERO:RF-of-REP-as-CTRL", all(v == 0.0 for v in floors_from({s: ident["REP"][s] for s in SEEDS}).values()), "feeding REP's dump as CTRL makes RF exactly 0.0 on every row")
        tabs["REP_CTRL"] = pair_tab("REP", "CTRL")
        rf_ = floors_from({s: tabs["REP_CTRL"][s] for s in SEEDS})
    floors = {"RF0": rf0_, "RF": rf_, "F": {l: (F_ADE6 if l == "ADE6" else 0.0) for l in LABELS}, "bar": {l: bar_for(l, rf0_, rf_) for l in LABELS}}
    if floors_dir and not rc:
        os.makedirs(floors_dir, exist_ok=True)
        fp = os.path.join(floors_dir, "RF_REP.json")
        with open(fp, "w", encoding="utf-8", newline=chr(10)) as f:
            json.dump({"RF": rf_, "REP_minus_CTRL": {f"s{s}": {l: {k: tabs["REP_CTRL"][s][l].get(k) for k in ("delta", "lo", "hi", "separated")} for l in LABELS} for s in SEEDS}}, f, indent=1)
        floors["RF_file"] = {"path": fp, "sha256": sha256_file(fp), "note": "written before any A or B contrast was computed"}
    # ---- PHASE 2, the X rows (A, then B), the base and the floor planner
    for arm in arms:
        if arm not in ident:
            ident[arm] = pair_tab(arm, arm)
        for s in SEEDS:
            ctl.add(f"C2-IDENTITY:{arm}-{arm}:s{s}", is_exact_zero(ident[arm][s]), "(X, X) reads +0 [0, 0] on every Read-A row (comps, TAC / DK, S-POS)")
    for x in xs:
        for k, (p, q) in ((f"{x}_P10", (x, "P10")), (f"{x}_CTRL", (x, "CTRL")), (f"{x}_DAG", (x, "DAG"))):
            tabs[k] = pair_tab(p, q)
    tabs["DAG_P10"], tabs["CTRL_P10"] = pair_tab("DAG", "P10"), pair_tab("CTRL", "P10")
    d_ = [domain(tabs["DAG_P10"][s_]) for s_ in SEEDS]
    ctl.add("DOMAIN-COUNT", (len(d_[0]), len(d_[1])) == (DOMAIN_N[0], DOMAIN_N[1]), {"n_s0": len(d_[0]), "n_s1": len(d_[1]), "PREREG": DOMAIN_N})
    # ---- guards' G rows, per X
    res = {"tabs": tabs, "floors": floors, "recs": recs, "xs": xs, "g": {}}
    items = []
    for x in xs:
        g_rows = {}
        for s in SEEDS:
            px, pc = recs[x][s]["pack"], recs["CTRL"][s]["pack"]
            g = {}
            for hd, nm in (("lat", "R-TLAT"), ("lon", "R-TLON")):
                r, (a_, b_, eid2, red) = DR.f1_pair_row(T, px, pc, hd, nb, matched=True)
                g[nm] = r
                items.append((f"{nm} {x} s{s}", a_, eid2, red))
            ade_link = DR.err_pair_row(T, f"R-ADE T1-kin ADE6 ({x} - CTRL) s{s}", K.comps(recs[x][s]["dump"]["cl"], recs[x][s]["dump"]["g"])["ade6_m"],
                                       K.comps(recs["CTRL"][s]["dump"]["cl"], recs["CTRL"][s]["dump"]["g"])["ade6_m"], recs[x][s]["dump"]["eid"], nb)
            ctl.add(f"ADE-LINK:{x}:s{s}", abs(float(tabs[f"{x}_CTRL"][s]["ADE6"]["delta"]) - float(ade_link["delta"])) <= 1e-9, {"readA_rows_block": tabs[f"{x}_CTRL"][s]["ADE6"]["delta"], "layer_G_R-ADE": ade_link["delta"]})
            t0 = DR.err_pair_row(T, f"R-T0 T0 own-code ADE6 ({x} - CTRL) s{s}", recs[x][s]["t0arr"]["ade6"], recs["CTRL"][s]["t0arr"]["ade6"], recs[x][s]["t0arr"]["eid"], nb)
            ctl.add(f"T0-WINDOWS:{x}:s{s}", bool(np.array_equal(recs[x][s]["t0arr"]["eid"], recs["CTRL"][s]["t0arr"]["eid"]) and len(recs[x][s]["t0arr"]["ade6"]) == N_WINDOWS), "T0 eid identical window for window")
            ra = tabs[f"{x}_DAG"][s]["ADE6"]
            g["T0_point"], g["KEEP_point"] = t0["point"], float(ra["mean_arm"]) - float(ra["mean_ref"])
            g["R-T0_row"], g["R-ADE_row"] = t0, ade_link
            g_rows[s] = g
            a6 = K.comps(recs[x][s]["dump"]["cl"], recs[x][s]["dump"]["g"])["ade6_m"]
            mk = np.isfinite(a6)
            items += [(f"R-ADE {x} s{s}", a6[mk], recs[x][s]["dump"]["eid"][mk], "mean"), (f"R-T0 {x} s{s}", recs[x][s]["t0arr"]["ade6"], recs[x][s]["t0arr"]["eid"], "mean")]
        res["g"][x] = g_rows
    ok1, det1 = DR.k1_own(T, items, nb)
    ctl.add("K1-ORC-ROWS", ok1, det1)
    return res


def compose_all(res, ctl_ok):
    """Per X: compose_seed for both seeds, decide, outcome block.  Then the cascade."""
    out = {}
    for x in res["xs"]:
        per_seed = {}
        for s in SEEDS:
            t = {"X_P10": res["tabs"][f"{x}_P10"][s], "X_CTRL": res["tabs"][f"{x}_CTRL"][s], "X_DAG": res["tabs"][f"{x}_DAG"][s], "DAG_P10": res["tabs"]["DAG_P10"][s]}
            per_seed[s] = compose_seed(s, t, res["g"][x][s], res["floors"]["RF0"], res["floors"]["RF"])
        out[x] = {"per_seed": per_seed, "decision": outcome_block(decide_arm(ctl_ok, per_seed), x)}
    return out


def trim(rows):
    return {l: ({k: r.get(k) for k in ("delta", "lo", "hi", "separated", "n", "mean_arm", "mean_ref", "family", "direction", "separated_better", "separated_worse", "kind")} if r.get("status") != "NOT COMPUTED" else r) for l, r in rows.items()}


def families_block(res, x):
    return {f"s{s}": families_report({"X-CTRL": res["tabs"][f"{x}_CTRL"][s], "X-P10": res["tabs"][f"{x}_P10"][s], "X-DAG": res["tabs"][f"{x}_DAG"][s]}) for s in SEEDS}


def provenance(a, t_start, manifest_path):
    return {"prereg": PREREG_ID, "prereg_sha256": PREREG_SHA256, "claim_row": CLAIM_ROW, "rf0_sha256": RF0_SHA256, "reader_sha256": sha256_file(os.path.abspath(__file__)),
            "ref_pins": REF_PINS, "manifest_sha256": sha256_file(manifest_path) if manifest_path and os.path.isfile(manifest_path) else None, "wall_s": round(time.time() - t_start, 1),
            "evidence_class": "MEASURED (ours; artifact paths in the manifest)"}


def main_read(a, manifest_path, out_path, rc_evidence):
    """THE READ.  REFUSED (rc 2) writes NOTHING; missing required slots = rc 4 (nothing written); otherwise the outcome and every row are written (rc 0, or 3 for UNPLACED)."""
    t_start = time.time()
    D2, DR = mods()
    bad = static_problems(a)
    if bad:
        print("DAG-ORC READ REFUSED (nothing written): pinned inputs / PREREG / RF0:", file=sys.stderr)
        for b in bad:
            print("  -", b, file=sys.stderr)
        return 2
    man = json.load(open(manifest_path, encoding="utf-8"))
    slots = discover(man)
    miss = {arm: {f"s{s}": slots[arm][s] for s in SEEDS if slots[arm][s]["state"] != "PRESENT"} for arm in REQUIRED_SLOTS}
    miss = {k: v for k, v in miss.items() if v}
    if miss:
        print("DAG-ORC READ BLOCKED (nothing written): required slots are not complete:", file=sys.stderr)
        print(json.dumps(miss, indent=1), file=sys.stderr)
        return 4
    b_partial = {f"s{s}": slots["B"][s] for s in SEEDS if slots["B"][s]["state"] == "PARTIAL"}
    if b_partial:
        print("DAG-ORC READ BLOCKED (nothing written): B is half-present:", json.dumps(b_partial), file=sys.stderr)
        return 4
    ctl = DR.Ctl()
    ctl.add("STATIC", True, "staged readers, Read A's banked inputs, PREREG text, RF0 pin")
    roll = rollout_files(man)
    pr, rf0, det = k_p0(a.rf0, roll)
    ctl.add("K-P0", not pr, pr or {"RF0_sha256": RF0_SHA256[:16] + "...", **det})
    ctl.run("K-BUILD", lambda: k_build(man.get("build")))
    ctl.run("K-RC", lambda: rc_evidence_ok(rc_evidence))
    try:
        rig = Rig(a)
        res = read_core(rig, ctl, man, "read", rf0 or {}, floors_dir=os.path.dirname(os.path.abspath(out_path)))
    except DR.Refusal as e:
        print("DAG-ORC READ REFUSED (nothing written):", file=sys.stderr)
        for m_ in e.msgs:
            print("  -", m_, file=sys.stderr)
        return 2
    except (OSError, KeyError, ValueError) as e:
        print(f"DAG-ORC READ INPUT ERROR (nothing written): REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except RowNotComputed as e:
        print(f"DAG-ORC READ REFUSED (nothing written): a Read-A row is NOT COMPUTED: {e}", file=sys.stderr)
        return 2
    if ctl.failed():
        print("DAG-ORC READ REFUSED (nothing written): failed controls:", file=sys.stderr)
        for k, v in ctl.failed().items():
            print("  -", k, str(v["detail"])[:300], file=sys.stderr)
        return 2
    try:
        comp = compose_all(res, True)
    except RowNotComputed as e:
        print(f"DAG-ORC READ REFUSED (nothing written): a Read-A row is NOT COMPUTED: {e}", file=sys.stderr)
        return 2
    casc = cascade(comp["A"]["decision"], comp["B"]["decision"] if "B" in comp else None)
    out = {"OUTCOME": comp[casc["final_arm"]]["decision"], "cascade": casc, "controls_PASS": ctl.flags(), "A": comp["A"]["decision"], "B": comp["B"]["decision"] if "B" in comp else None,
           "rows": {x: {f"s{s}": {k: v for k, v in comp[x]["per_seed"][s].items()} for s in SEEDS} for x in comp}, "floors": res["floors"],
           "read_a": {k: {f"s{s}": trim(v[s]) for s in SEEDS} for k, v in res["tabs"].items()},
           "four_families": {x: families_block(res, x) for x in res["xs"]},
           "t0_four_families": {"status": "NOT RUN", "work_item": "orc_families.py pointed at A / CTRL (section 5): the T0 LONGITUDINAL / LATERAL / TACTICAL rows and distance keeping at T0; a build item, never a silent omission"},
           "beside_not_built": {"M-LAUNCH": "X - CTRL of the mean (v_own - v_GT) at ticks 1-2 in the GT-launch windows, from the tick dumps: NOT BUILT", "anti_echo_block": "longitudinal_claim_admissible, hold-v0, copy detector (RESULT_V8F_ORECOVER 3.1): NOT BUILT",
                                "M2_and_L2_launch_subset": "NOT BUILT"},
           "perturbation_counts": {f"{arm}_s{s}": {k: (DR.find_key(res["recs"][arm][s]["train"]["sources"], k) or [(None, None)])[0][1] for k in ("ego_perturb_n", "ego_perturb_eligible", "ego_perturb_nav_kept")}
                                   for arm in res["xs"] + ["REP", "CTRL"] for s in SEEDS},
           "k_build": man.get("build"), "tier": {"T1": TIER_T1, "T0": TIER_T0}, "estimator": ESTIMATOR,
           "strategic_family": "NOT PAUSED for v8F (PI ruling 2026-10-17)", "provenance": provenance(a, t_start, manifest_path)}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"OUTCOME": out["OUTCOME"]["OUTCOME"], "arm": out["OUTCOME"]["arm"], "adopts": out["OUTCOME"]["adopts"], "committed_next": out["OUTCOME"]["committed_next"], "cascade": casc}, indent=1, ensure_ascii=False))
    print(f"wrote {out_path}  sha256 {sha256_file(out_path)}")
    return 3 if out["OUTCOME"]["OUTCOME"] == "UNPLACED" else 0


def rollout_files(man):
    """Every file of A's / REP's / B's dev-box rollouts (dumps, tick files, run jsons, both lanes): K-P0 requires the frozen RF0 to predate all of them."""
    out = []
    for arm in ("A", "REP", "B"):
        for s in SEEDS:
            e = ((man.get("arms") or {}).get(arm) or {}).get(str(s)) or {}
            for lane in ("t1", "t1_logged"):
                t = e.get(lane) or {}
                out += glob.glob(os.path.join(t.get("dump_dir") or "", "ep*.npz")) + [t.get("json") or "", t.get("ticks") or ""]
    return out


def rc_evidence_ok(path):
    """K-RC: the evidence file of a PASSing `--rc-dag2` run exists AND was produced by THIS reader (its recorded reader_sha256 equals this file's)."""
    if not os.path.isfile(path):
        return False, f"{path} is missing: run `dag_orc_read.py --rc-dag2` first"
    d = json.load(open(path, encoding="utf-8"))
    me = sha256_file(os.path.abspath(__file__))
    bad = []
    if d.get("reader_sha256") != me:
        bad.append(f"RC evidence was produced by reader {str(d.get('reader_sha256'))[:12]}..., this is {me[:12]}...")
    if d.get("differences") != 0 or d.get("PASS") is not True:
        bad.append(f"RC evidence: differences {d.get('differences')}, PASS {d.get('PASS')}")
    return not bad, bad or {"evidence": path, "differences": 0, "reader_sha256": me[:16] + "..."}


def rc_manifest():
    """K-RC's slots: A := DAG2, CTRL := DAG2-0 (v8f-dagger2 / v8f-dagger2-t1), DAG := DAG (v8f-dagger / v8f-dagger-t1)."""
    m2 = json.load(open(f"{ART}/v8f-dagger2-t1/runs.json", encoding="utf-8"))
    m1 = json.load(open(f"{ART}/v8f-dagger-t1/runs.json", encoding="utf-8"))
    return {"seeds": [0, 1], "arms": {"A": m2["arms"]["DAG"], "CTRL": m2["arms"]["DAG0"], "DAG": m1["arms"]["DAG"]}}


def main_rc(a, out_path):
    """K-RC (section 6): this reader's analysis on DAG2 / DAG2-0 (slots A := DAG2, CTRL := DAG2-0), floors set to 0, adoption off, reproduces the banked DAG2_READ.json's decisive rows with 0 differences:
    the three Read-A tables (DAG2 - P10, DAG2 - DAG2-0 without STRATEGIC, DAG - P10), R-TLAT / R-TLON / R-ADE / R-T0, the crawl rows (R-CRAWL / R-MOVE per row), and the newly-worse rows."""
    t_start = time.time()
    D2, DR = mods()
    bad = static_problems(a)
    if bad:
        print("K-RC REFUSED:", bad, file=sys.stderr)
        return 2
    banked = json.load(open(f"{ART}/v8f-dagger2-t1/read/DAG2_READ.json", encoding="utf-8"))
    ctl = DR.Ctl()
    rig = Rig(a)
    man = rc_manifest()
    res = read_core(rig, ctl, man, "rc", None)
    diffs, detail = [], {}
    for key, pair in (("DAG2_minus_P10", "A_P10"), ("DAG2_minus_DAG20", "A_CTRL"), ("DAG_minus_P10", "DAG_P10")):
        for s in SEEDS:
            mine = {l: r for l, r in res["tabs"][pair][s].items() if not (key == "DAG2_minus_DAG20" and l in SPOS_LABELS)}
            ref = {l: r for l, r in banked["read_a"][key][f"s{s}"].items() if not (key == "DAG2_minus_DAG20" and l in SPOS_LABELS)}
            d = rc_diffs(mine, ref)
            detail[f"{key}.s{s}"] = {"rows": len(ref), "differences": d}
            diffs += [f"{key}.s{s}: {x}" for x in d]
    for s in SEEDS:
        g = banked["rows"][f"s{s}"]
        for nm in ("R-TLAT", "R-TLON"):
            mine, ref = res["g"]["A"][s][nm], g[nm]
            d = [f"{nm}.{k}: {mine[k]} vs {ref[k]}" for k in ("delta", "lo", "hi") if abs(float(mine[k]) - float(ref[k])) > 1e-9] + ([f"{nm}.separated"] if bool(mine["separated"]) != bool(ref["separated"]) else [])
            detail[f"{nm}.s{s}"] = d
            diffs += [f"s{s} {x}" for x in d]
        mine, ref = res["g"]["A"][s]["R-ADE_row"], g["R-ADE"]
        d = [f"R-ADE.{k}: {mine[k]} vs {ref[k]}" for k in ("delta", "lo", "hi") if abs(float(mine[k]) - float(ref[k])) > 1e-9]
        diffs += [f"s{s} {x}" for x in d]
        detail[f"R-ADE.s{s}"] = d
        mine, ref = res["g"]["A"][s]["T0_point"], g["R-T0"]["point"]
        d = [f"R-T0.point: {mine} vs {ref}"] if abs(float(mine) - float(ref)) > 1e-12 else []
        diffs += [f"s{s} {x}" for x in d]
        detail[f"R-T0.s{s}"] = d
        # crawl rows and newly-worse rows, composed with the floors at 0
        t = {"X_P10": res["tabs"]["A_P10"][s], "X_CTRL": res["tabs"]["A_CTRL"][s], "X_DAG": res["tabs"]["A_DAG"][s], "DAG_P10": res["tabs"]["DAG_P10"][s]}
        cs = compose_seed(s, t, res["g"]["A"][s], {l: 0.0 for l in LABELS}, {l: 0.0 for l in LABELS})
        bc = g["crawl"]
        d = ([f"R-CRAWL_rows: {cs['R-CRAWL_rows']} vs {bc['R-CRAWL_rows']}"] if cs["R-CRAWL_rows"] != bc["R-CRAWL_rows"] else []) + \
            ([f"R-MOVE_rows (floors 0: clears == separated-better): {cs['R-MOVE_rows']} vs {bc['R-MOVE_rows']}"] if cs["R-MOVE_rows"] != bc["R-MOVE_rows"] else [])
        nw_mine = sorted(l for l in cs["domain"] if sep_worse(l, t["X_P10"][l]))
        nw_ref = sorted(g["newly_worse"]["violations"])
        d += [f"newly-worse rows {nw_mine} vs banked {nw_ref}"] if nw_mine != nw_ref else []
        diffs += [f"s{s} {x}" for x in d]
        detail[f"crawl+newly_worse.s{s}"] = d
        detail[f"domain_n.s{s}"] = len(cs["domain"])
    # the S-POS definitions agree: per-window difference of contrasts == difference of the runs' banked per-run points (both seeds, all arcs): the P-0 RF0 uses the latter
    spos_check = {}
    for s in SEEDS:
        pa, pc = res["recs"]["A"][s]["pack"].extra[SPOS_KEY], res["recs"]["CTRL"][s]["pack"].extra[SPOS_KEY]
        for lab in SPOS_LABELS:
            win = res["tabs"]["A_CTRL"][s][lab]["delta"]
            pts = float(pa[lab]["delta"]) - float(pc[lab]["delta"])
            spos_check[f"s{s}.{lab}"] = {"per_window_difference_of_contrasts": win, "difference_of_per_run_points": round(pts, 4), "equal_at_4dp": abs(win - pts) <= 1.5e-4}
    out = {"K-RC": "this reader's analysis on DAG2 / DAG2-0, floors 0, adoption off, vs the banked DAG2_READ.json", "PASS": not diffs and not ctl.failed(), "differences": len(diffs), "difference_list": diffs[:20], "detail": detail,
           "controls_failed": {k: str(v["detail"])[:200] for k, v in ctl.failed().items()}, "controls_n": len(ctl.items),
           "spos_definition_check": spos_check, "reader_sha256": sha256_file(os.path.abspath(__file__)), "banked_sha256": sha256_file(f"{ART}/v8f-dagger2-t1/read/DAG2_READ.json"),
           "wall_s": round(time.time() - t_start, 1)}
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: out[k] for k in ("PASS", "differences", "difference_list", "controls_failed", "controls_n")}, indent=1))
    print(f"wrote {out_path}  sha256 {sha256_file(out_path)}")
    return 0 if out["PASS"] else 1



def main_smoke(a, out_path):
    """--smoke-standins: exercise the NON-rc branches of read_core / compose_all / cascade on REAL data with stand-in slots (A := DAG2, B := DAG2, REP := DAG2-0, CTRL := DAG2-0, DAG := DAG).
    The run-record controls (K-PAIR, K-REACH, argv, ...) are EXPECTED to fail on stand-ins; the point is that every code path runs.  Nothing is an outcome: the result is labelled SMOKE and is not a read."""
    t_start = time.time()
    D2, DR = mods()
    bad = static_problems(a)
    if bad:
        print("SMOKE REFUSED:", bad, file=sys.stderr)
        return 2
    man = rc_manifest()
    man["arms"]["REP"], man["arms"]["B"] = man["arms"]["CTRL"], man["arms"]["A"]
    pr, rf0, _ = k_p0(a.rf0, [], check_inputs=False)
    ctl = DR.Ctl()
    rig = Rig(a)
    fdir = os.path.join(os.path.dirname(os.path.abspath(out_path)), "smoke_floors")
    res = read_core(rig, ctl, man, "read", rf0, floors_dir=fdir)
    comp = compose_all(res, not ctl.failed())
    casc = cascade(comp["A"]["decision"], comp["B"]["decision"])
    out = {"SMOKE": "stand-in slots (A := DAG2, B := DAG2, REP := DAG2-0): NOT A READ", "controls_failed": {k: str(v["detail"])[:160] for k, v in ctl.failed().items()}, "controls_n": len(ctl.items),
           "decisions": {x: {"OUTCOME": comp[x]["decision"]["OUTCOME"], "flags": comp[x]["decision"]["flags"]} for x in comp}, "cascade": casc, "floors_RF_max": max(res["floors"]["RF"].values()),
           "bars": res["floors"]["bar"], "RF_file": res["floors"].get("RF_file"), "families_keys": list(families_block(res, "A")["s0"]), "wall_s": round(time.time() - t_start, 1)}
    with open(out_path, "w", encoding="utf-8", newline=chr(10)) as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: out[k] for k in ("SMOKE", "controls_failed", "controls_n", "cascade", "floors_RF_max")}, indent=1))
    print(f"wrote {out_path}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--static-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rc-dag2", action="store_true", help="K-RC: the analysis on DAG2 / DAG2-0, floors 0, adoption off, against the banked DAG2_READ.json")
    ap.add_argument("--smoke-standins", action="store_true", help="exercise the read path on stand-in slots (NOT a read; controls expected to fail)")
    ap.add_argument("--manifest")
    ap.add_argument("--out")
    ap.add_argument("--rf0", default=f"{HERE}/P0_RF0.json")
    ap.add_argument("--prereg", default=f"{HERE}/PREREG_V8F_DAG_ORC.md")
    ap.add_argument("--rc-evidence", default=f"{HERE}/RC_DAG2.json", help="K-RC's evidence file (must carry this reader's sha256)")
    ap.add_argument("--work-dir")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    return ap


def main(argv=None):
    ns = build_parser().parse_args(argv)
    a = default_args(prereg=ns.prereg, rf0=ns.rf0, work_dir=ns.work_dir, n_boot=ns.n_boot)
    if ns.static_check:
        return main_static(a)
    if ns.dry_run:
        return main_dry(a, ns.manifest, ns.out)
    if ns.smoke_standins:
        return main_smoke(a, ns.out or f"{HERE}/SMOKE.json")
    if ns.rc_dag2:
        return main_rc(a, ns.out or f"{HERE}/RC_DAG2.json")
    if not (ns.manifest and ns.out):
        print("usage: --manifest runs.json --out DAG_ORC_READ.json  (or --static-check / --dry-run / --rc-dag2)", file=sys.stderr)
        return 2
    return main_read(a, ns.manifest, ns.out, ns.rc_evidence)


if __name__ == "__main__":
    sys.exit(main())
