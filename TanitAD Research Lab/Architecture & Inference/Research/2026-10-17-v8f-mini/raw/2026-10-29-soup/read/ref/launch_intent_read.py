#!/usr/bin/env python3
"""launch_intent_read.py -- the dev-box T1-kin read of PREREG_V8F_LAUNCH_INTENT (DRAFT v2; claim row H-V8F-LAUNCH-INTENT-1), the reader of the Master Mind's redesign `MM_REDESIGN_2026-10-28.md`.
BUILT AND SELF-TESTED; NOT RUN ON LAUNCH-INTENT OUTPUTS (none exist).

THE DESIGN IT READS (one change of the bar, one added arm per seed).  At each seed s there are two same-seed one-lever pairs:
    the main pair       A_s    - CTRL_s   (CTRL_s = DAG-OACC_s, banked)
    the replicate pair  A_rep,s - REP_s   (REP_s banked from DAG-ORC r2, under the REP licence)
plus P10 (the floor planner, the `fl` arm of every dump).  The bar is 2 * max(F_r, RF0_r) (F = 0.1207 m on ADE6, 0 elsewhere; RF0 = the frozen P0_RF0.json).  RF = max over seeds of |REP - DAG-OACC| is
computed from the REP pair and reported BESIDE every row; it is not a bar term.  Each per-seed predicate holds only when it holds on BOTH pairs of that seed (PREREG section 10); the outcome table, its
precedence and its enumeration over the per-seed booleans are DAG-ORC's (524,288 cells, each matching exactly one row: `tools/enumerate_cells_li.py`).

NOTHING IS RE-DERIVED.  The Read-A rows, the TAC / DK rows and the T1-HEADS pipeline are the DAG-ORC reader's own `Rig` (imported unchanged from `--orc-dir`, its `ref/` pins intact), as the SPEEDSUB-T1 reader
does.  This file adds only the PREREG's composition, in two layers:

  PURE LAYER (no heavy import; every function above `# ---- heavy` is unit-tested and mutation-tested)
     the bar, clears / fires / domain, the per-pair composition, the per-seed AND, the outcome table as ONE function, the dose cascade, NEXT-LAUNCH routing, Read A on adoption, the RF floors (beside), the
     slow-draw flag, the run-record controls K-PAIR / K-REGISTERED / K-INIT / K-SHARE / K-MASK / K-BUILD / K-REP-LICENCE / K-ZERO, the four families, the PREREG pin.
  HEAVY LAYER (`Heavy`, `main_read`, `main_dry`)
     the DAG-ORC `Rig` applied to the eight runs (A, A_rep, CTRL, REP at both seeds) and P10; the same wiring is exercised end to end on a FAKE heavy layer by the tests.

    python launch_intent_read.py --static-check --prereg PREREG_V8F_LAUNCH_INTENT.md
    python launch_intent_read.py --dry-run --out dry_run.json                           the banked dag-orc dumps standing in for the pairs (CTRL = DAG-OACC, REP = REP); reproduces DAG-ORC's banked tables
    python launch_intent_read.py --manifest runs.json --prereg PREREG.md --out LAUNCH_INTENT_READ.json    THE READ (rc 2 REFUSED | rc 0 outcome written | rc 3 UNPLACED | rc 4 slots missing)

Manifest (runs.json): {"seeds": [0, 1], "arms": {"A"|"Arep"|"CTRL"|"REP"|("B"|"Brep"): {"0": ENTRY, "1": ENTRY}}, "build": K-BUILD record}.  ENTRY = {ckpt_tail, train_dir, token, t1: {tag, ticks,
dump_dir, json}, t1_logged: {..}, t0: {json, npz}, t0_fresh: {..}} exactly as the DAG-ORC manifest writes them (CTRL and REP are banked: t1 + t0 + train records; the new arms add the logged lane and the fresh T0).
"""
from __future__ import annotations

import argparse
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
import numpy as np  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:       # noqa: BLE001
        pass

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts"
ORC_DIR = f"{ART}/dagorc_rebase/read"

# ---------------------------------------------------------------------------------------------------------------- PREREG_V8F_LAUNCH_INTENT (committed numbers)
PREREG_ID = "Project Steering/PREREG_V8F_LAUNCH_INTENT.md (DRAFT v2, MM_REDESIGN_2026-10-28)"
CLAIM_ROW = "H-V8F-LAUNCH-INTENT-1"
RF0_SHA256 = "1608d70b21b6b6a45136a6d96828f70b5bbabd9c0ceb84db136927593c79ad0f"        # P0_RF0.json, frozen 2026-10-27 (PREREG_V8F_DAG_ORC section 3)
SEEDS = (0, 1)
F_ADE6 = 0.1207                  # F_r for ADE6 (F_M2b, MODEL_REGISTRY section 17, H-V8F-SEED-1); every other row: 0
HALF_F = 0.0604                  # +F/2: R-T0 and R-KEEP, inclusive, on the point
BAR_FACTOR = 2.0                 # bar_r = 2 * max(F_r, RF0_r)
W_ARM = {"A": "1.58926", "B": "0.726175", "HI": "4.17852"}          # --launch-intent-w, as the argv prints it
SHARE_TARGET = {"A": 1.0 / 3.0, "B": 1.0 / 4.0, "HI": 1.0 / 2.0}
SHARE_TOL = 0.02
F_LAUNCH = 49618 / 306566        # f, the GT-launch fraction of the TRAIN windows (tools/measure_f.py, f_measure.json)
MASK_TOL = 0.02
WINDOW_LIST_SHA = "e144adcd9b6273f2b44ad7dfa162931b90e159f7b3911e0893b25777b2ae8168"      # train_window_list_sha256 of both banked DAG-OACC runs
EXCLUDED_SHA = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"
REP_SEED = {0: 10, 1: 11}
STEPS, LR, WARMUP, PGT, SWAP_P, SWAP_TOL = 3000, 1e-4, 100, 0.2, 0.1, 0.02
W_ACC = "49.6969"
M2B_CKPT_SHA = {0: "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", 1: "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
OACC_BANK_SHA = {0: "197a68bb540be5a7d157942e9157e1c8e60ca60d82b1ca4fb4fcb68edccfaa8f", 1: "36c7aea964dafe9423253aa479c2fc6cdfc93848395b593315cba812f5f04d4d"}
OACC_CKPT_SHA = {0: "67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f", 1: "90b01474819082a6d32a9000d63a4954e0074916140abdbbcb67b353f7ac3333"}
CKPT_PREFIX = {"CTRL": {0: "67265ec1", 1: "90b01474"}, "REP": {0: "b03fd6ef", 1: "77853e56"}}
CTRL_BUILD_COMMIT, REP_BUILD_COMMIT = "aeb54468", "8c24823f"       # the commits DAG-OACC and DAG-ORC r2's REP trained at
BUILD_BASE_COMMIT = CTRL_BUILD_COMMIT
BUILD_FILES = ("stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/models/v8f_mini.py")
# PREREG Amendment 1 (A1-2026-10-28), K-BUILD: v8f_mini_data.py at the launch commit is NOT byte-identical to aeb54468 (DAG-ORC's compose hunks, e7ea20d / 8c24823f, sit between them).
# Where a pinned file differs from the base, its launch sha256 must equal the pin: the data module of DAG-ORC r2's build 8c24823f, the commit REP trained at. Its hunks still need the general rule.
LAUNCH_PINNED_FILES = {"stack/tanitad/data/v8f_mini_data.py": "c117e1cf5be08b9739f02e2730ce555be5a707a3183ded6481256e419f751752"}
N_BOOT, N_WINDOWS, N_EPISODES = 2000, 2399, 141
LEVER = "--launch-intent-w"
ROLES_NEW = ("A", "Arep", "B", "Brep")
ROLES_BANKED = ("CTRL", "REP")
REQUIRED_ROLES = ("A", "Arep", "CTRL", "REP")
ARM_OF_ROLE = {"A": "A", "Arep": "A", "B": "B", "Brep": "B"}          # the dose arm a role belongs to
CONTROL_OF = {"A": "CTRL", "Arep": "REP", "B": "CTRL", "Brep": "REP"}
PAIR_OF = {"A": "main", "Arep": "rep", "B": "main", "Brep": "rep"}
PAIRS = ("main", "rep")

LABELS = ("ADE6", "speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG", "heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE",
          "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)", "TAC-LAT", "TAC-LON", "S-POS@30", "S-POS@50", "S-POS@80", "M1 crawl share")          # Read A's 19 rows
DIRECTION = {l: "lower" for l in LABELS}
DIRECTION.update({"TAC-LAT": "higher", "TAC-LON": "higher"})
FAMILY = {"ADE6": "PRIMARY", "speed MAE": "LONGITUDINAL", "along-track MAE": "LONGITUDINAL", "accel MAE": "LONGITUDINAL", "DK-VIOL": "LONGITUDINAL", "DK-HW": "LONGITUDINAL",
          "DK-TG": "LONGITUDINAL", "heading MAE": "LATERAL", "yaw-rate MAE": "LATERAL", "curvature MAE": "LATERAL", "cross-track MAE": "LATERAL", "L1 lateral-accel MAE": "LATERAL",
          "L2 curvature MAE (GT>=5.2)": "LATERAL", "TAC-LAT": "TACTICAL", "TAC-LON": "TACTICAL", "S-POS@30": "STRATEGIC", "S-POS@50": "STRATEGIC", "S-POS@80": "STRATEGIC",
          "M1 crawl share": "crawl"}
CRAWL_SET = ("yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")          # the rows separated-worse for DAG-OACC - P10 at both seeds (Amendment 1); the same for both pairs
GUARDS = ("R-TLAT", "R-TLON", "R-ADE", "R-T0", "R-NEW")
FLAG_KEYS = ("R-CRAWL", "R-MOVE", "R-KEEP", "R-NOBASE")
TWO_F = 0.2415                   # PREREG_V8F_DAG_REREAD section 2 D1, as printed
TIER_T1 = "T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan; plain arms). Never 'closed loop'."
TIER_T0 = "T0 (teacher-forced own-code, eval-141, ages cycle: box 2's standard eval). Never 'driving performance'."
ESTIMATOR = ("taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0, 141 episodes / 2,399 windows, per seed and per pair, paired, on the same windows and matched cells. It answers 'would another "
             "draw of EPISODES say this?'. The training-run question is answered by the replicate pair at each seed (a lever effect is credited at a seed only when both pairs show it) and by the two seeds; "
             "the planner is deterministic, so no inference-seed question arises.")

CONS_NAMES = ("lat_theta_deg", "turn_t_start_s", "turn_t_end_s", "turn_d_start_m", "turn_len_m", "turn_dyaw_deg", "turn_r_arc_m", "turn_exit_x_m", "turn_exit_y_m", "turn_exit_psi_deg",
              "lc_t_cross_s", "lc_d_cross_m", "lon_v_target_ms", "lon_t_reach_s", "lon_d_reach_m", "lon_t_start_s", "stop_x_m", "stop_y_m", "lead_gap_m", "lead_tg_s", "lead_gap_min_m",
              "lead_tg_min_s", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")      # rep_s0_diag/cons_names_ea5e3d3b.py
SPEED_CELLS = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
SLOW_SHIFT = 0.5                 # m/s: a slow-T draw has all five cells separated by MORE than this

# The tables of the PREREG that this reader codes, verbatim (markdown as printed); prereg_problems() compares each with the PREREG file.
PREREG_OUTCOME_ROWS = (
    "| **REFUSED** | any control fails | fix and re-read; no row is reported |",
    "| **HARMS** | controls pass; a guard (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW) fails at ≥ 1 seed | **A:** not adopted; read **B** under this table (train B first, with its own PI go). **B:** not adopted; the launch weight cannot be dosed under the guards; **NEXT-LAUNCH** (§11) |",
    "| **FIXED / ADOPTED** | guards held; R-MOVE and R-CRAWL at both seeds; R-KEEP and R-NOBASE at both seeds | adopt X as the base. Re-run Read A on X: DOMINANT there is the dominance claim |",
    "| **FIXED / NOT ADOPTED** | guards held; R-MOVE and R-CRAWL at both seeds; R-KEEP or R-NOBASE fails at ≥ 1 seed | the lever is attributed but X cannot replace DAG-OACC. **A:** B (the lower dose) under this table, with its own PI go. **B:** NEXT-LAUNCH |",
    "| **PARTIAL / ADOPTED** | guards held; R-MOVE at both seeds; R-CRAWL fails at ≥ 1 seed; R-KEEP and R-NOBASE at both seeds | adopt X. The crawl rows that still fail go to **NEXT-LAUNCH on X's states** |",
    "| **PARTIAL / NOT ADOPTED** | guards held; R-MOVE at both seeds; R-CRAWL fails at ≥ 1 seed; R-KEEP or R-NOBASE fails at ≥ 1 seed | as FIXED / NOT ADOPTED |",
    "| **PARTIAL / ONE SEED** | guards held; R-MOVE at exactly one seed | not adopted, and never called an effect of the lever (`H-ESTIM-SEED-1`). **NEXT-LAUNCH on DAG-OACC states** |",
    "| **NO EFFECT** | guards held; R-MOVE at neither seed | the along-track weight on launch samples is not the crawl's lever at this dose. **NEXT-LAUNCH on DAG-OACC states** |",
)
PREREG_PREDICATE_ROWS = (
    "| R-MOVE | every crawl row clears in A_s − CTRL_s **and** in A_rep,s − REP_s |",
    "| R-CRAWL | A_s − P10 **and** A_rep,s − P10 are each not separated-worse on every crawl row |",
    "| R-TLAT, R-TLON, R-ADE | not separated-worse in either pair |",
    "| R-T0 | the T0 ADE6 point is ≤ +F/2 = 0.0604 m in both pairs |",
    "| R-NEW | no row fires in either pair (fire as in §6: each pair's own control in place of DAG-OACC, and the §6 bar) |",
    "| R-KEEP | the T1-kin ADE6 point is ≤ +F/2 in both pairs |",
    "| R-NOBASE | holds in both pairs |",
)
PREREG_BAR_ROW = "| **bar_r** | **2 × max(F_r, RF0_r)**, one value per row, applied at both seeds and in both pairs, inclusive. A row whose bar is 0 clears on separation alone |"


def strip_md(t):
    return re.sub(r"\s+", " ", t.replace("**", "").replace("`", "")).strip()


def split_row(ln):
    cells = [c.strip() for c in ln.strip().strip("|").split("|")]
    return cells


OUTCOME_TABLE = tuple((strip_md(split_row(r)[0]), strip_md(split_row(r)[1]), strip_md(split_row(r)[2])) for r in PREREG_OUTCOME_ROWS)      # (outcome, condition, committed next)
OUTCOME_NAMES = tuple(r[0] for r in OUTCOME_TABLE)
ADOPTING = ("FIXED / ADOPTED", "PARTIAL / ADOPTED")
SENDS_TO_B = ("HARMS", "FIXED / NOT ADOPTED", "PARTIAL / NOT ADOPTED")          # the A rows that send the programme to B
UNPLACED_TEXT = ("UNPLACED: this reading matches no row (or more than one row) of the PREREG's section-10 table. enumerate_cells_li.py proved the table complete and exclusive over its 524,288 "
                 "cells, so this path is unreachable through real cells: a reader / PREREG gap, not a result. No outcome word is invented; the PI / Master Mind decides, WITHOUT looking at which way the numbers fall.")


# ---- pure ---------------------------------------------------------------------------------------------------------------------------------------------------------
class RowNotComputed(Exception):
    pass


class Ctl:
    """The controls. ``add`` records PASS / FAIL with its detail; a control that raises is a FAIL carrying the exception, never a skip."""

    def __init__(self):
        self.items = {}

    def add(self, cid, ok, detail=None):
        self.items[cid] = {"PASS": bool(ok), "detail": detail}
        return bool(ok)

    def run(self, cid, fn):
        try:
            ok, detail = fn()
        except Exception as e:      # noqa: BLE001  -- a control that cannot be evaluated has not passed
            ok, detail = False, f"{type(e).__name__}: {e}"
        return self.add(cid, ok, detail)

    def failed(self):
        return {k: v for k, v in self.items.items() if not v["PASS"]}

    def flags(self):
        return {k: v["PASS"] for k, v in self.items.items()}


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


def bar_for(label, rf0):
    """bar_r = 2 * max(F_r, RF0_r); F_r = F_ADE6 for ADE6, 0 for every other row.  RF (the cross-run spread) is NOT a term: it is reported beside.  One value per row, both seeds, both pairs."""
    return BAR_FACTOR * max(F_ADE6 if label == "ADE6" else 0.0, abs(rf0[label]))


def bar_table(rf0):
    return {l: bar_for(l, rf0) for l in LABELS}


def clears(label, row, bar):
    """R-MOVE, per row: X - Y is separated-better AND the improvement is >= bar_r (inclusive)."""
    return bool(sep_better(label, row) and -worsening(label, row) >= bar)


def fires(label, x_p10, y_p10, x_y, bar):
    """R-NEW, per row of the pair's domain: (1) X - P10 separated-worse, (2) Y - P10 NOT separated-worse (the row is in the domain), (3) X - Y separated-worse with the worsening >= bar_r (inclusive).
    Y is the pair's own control: DAG-OACC for the main pair, REP for the replicate pair."""
    return bool(sep_worse(label, x_p10) and not sep_worse(label, y_p10) and sep_worse(label, x_y) and worsening(label, x_y) >= bar)


def domain(y_p10):
    """D of a pair: every Read-A row not separated-worse for Y - P10 (no exempt rows)."""
    return [l for l in LABELS if not sep_worse(l, y_p10[l])]


def f1_worse(row):
    """macro-F1, higher is better: separated-worse = separated and negative."""
    return bool(row["separated"] and float(row["delta"]) < 0)


def compose_pair(t, g, bars, crawl=CRAWL_SET):
    """One pair's predicates.  t = {"X_P10", "X_Y", "Y_P10"}: {Read-A label: row}; g = {"R-TLAT": F1 row, "R-TLON": F1 row, "T0_point": float, "KEEP_point": float}; bars = {label: bar_r}.
    -> flags + detail."""
    for k_ in ("X_P10", "X_Y", "Y_P10"):                  # every row of all three tables must be computed: a short-circuited `and` must not hide a missing row
        for l_ in LABELS:
            _row(l_, t[k_].get(l_))
    crawl_rows = {l: not sep_worse(l, t["X_P10"][l]) for l in crawl}
    move_rows = {l: clears(l, t["X_Y"][l], bars[l]) for l in crawl}
    d_ = domain(t["Y_P10"])
    fire_rows = {l: fires(l, t["X_P10"][l], t["Y_P10"][l], t["X_Y"][l], bars[l]) for l in d_}
    nobase_rows = {l: bool(sep_worse(l, t["X_P10"][l]) and sep_worse(l, t["X_Y"][l]) and worsening(l, t["X_Y"][l]) >= bars[l]) for l in d_}
    guards = {"R-TLAT": not f1_worse(g["R-TLAT"]), "R-TLON": not f1_worse(g["R-TLON"]), "R-ADE": not sep_worse("ADE6", t["X_Y"]["ADE6"]),
              "R-T0": bool(g["T0_point"] <= HALF_F), "R-NEW": not any(fire_rows.values())}
    flags = {"R-CRAWL": all(crawl_rows.values()), "R-MOVE": all(move_rows.values()), "R-KEEP": bool(g["KEEP_point"] <= HALF_F), "R-NOBASE": not any(nobase_rows.values())}
    return {"flags": flags, "guards": guards, "crawl_set": list(crawl), "R-CRAWL_rows": crawl_rows, "R-MOVE_rows": move_rows, "domain": d_, "R-NEW_fires": fire_rows,
            "R-NOBASE_violations": nobase_rows, "T0_point": g["T0_point"], "KEEP_point": g["KEEP_point"]}


def compose_seed(main, rep):
    """PREREG section 10: a seed's predicate holds only when it holds on BOTH pairs of that seed."""
    return {"flags": {k: bool(main["flags"][k] and rep["flags"][k]) for k in FLAG_KEYS}, "guards": {g: bool(main["guards"][g] and rep["guards"][g]) for g in GUARDS},
            "pairs": {"main": main, "rep": rep}}


def outcome_rows(ctrl, held, rc, rm, keep, nobase):
    """Section 10, each row as its own literal predicate (DAG-ORC's, unchanged).  ``held[g][s]`` (g in GUARDS), ``rc[s]`` R-CRAWL, ``rm[s]`` R-MOVE, ``keep[s]`` R-KEEP, ``nobase[s]`` R-NOBASE: booleans per seed.
    Precedence: REFUSED first, then HARMS; with the guards held R-MOVE partitions the rest (both seeds / one seed / neither), R-CRAWL splits FIXED from PARTIAL, adoption is last."""
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


def decide_li(ctrl, held, rc, rm, keep, nobase):
    """The outcome table as ONE function.  -> {"OUTCOME", "matching_rows_in_table_order", "adopts", "flags"}.  Exactly one row must match; otherwise UNPLACED (defensive; unreachable through real cells)."""
    r = outcome_rows(ctrl, held, rc, rm, keep, nobase)
    matching = [n for n in OUTCOME_NAMES if r.get(n)]
    out = matching[0] if len(matching) == 1 else "UNPLACED"
    flags = {"controls_pass": bool(ctrl), "guards_held_per_seed": {f"s{s}": {g: bool(held[g][s]) for g in GUARDS} for s in SEEDS},
             "R-CRAWL": {f"s{s}": bool(rc[s]) for s in SEEDS}, "R-MOVE": {f"s{s}": bool(rm[s]) for s in SEEDS},
             "R-KEEP": {f"s{s}": bool(keep[s]) for s in SEEDS}, "R-NOBASE": {f"s{s}": bool(nobase[s]) for s in SEEDS}}
    return {"OUTCOME": out, "matching_rows_in_table_order": matching, "adopts": out in ADOPTING, "flags": flags}


def decide_arm(ctrl_ok, per_seed):
    """per_seed[s] = compose_seed(...)."""
    held = {g: {s: per_seed[s]["guards"][g] for s in SEEDS} for g in GUARDS}
    f = lambda k: {s: per_seed[s]["flags"][k] for s in SEEDS}        # noqa: E731
    return decide_li(ctrl_ok, held, f("R-CRAWL"), f("R-MOVE"), f("R-KEEP"), f("R-NOBASE"))


def table_row(name):
    return next((c, n) for nm, c, n in OUTCOME_TABLE if nm == name)


def next_for_arm(name, arm):
    """The committed next of row ``name`` for arm ``arm`` ("A" or "B").  A cell of the form 'A: ... B: ...' is split; 'as <ROW>' takes that row's text."""
    cond, nxt = table_row(name)
    m = re.fullmatch(r"as ([A-Z /]+)", nxt)
    if m:
        return next_for_arm(m.group(1).strip(), arm)
    if "B:" in nxt and nxt.startswith("A:"):
        i = nxt.index("B:")
        return nxt[2:i].strip() if arm == "A" else nxt[i + 2:].strip()
    if "A:" in nxt and "B:" in nxt:
        i, j = nxt.index("A:"), nxt.index("B:")
        return nxt[i + 2:j].strip() if arm == "A" else nxt[j + 2:].strip()
    return nxt


def outcome_block(d, arm):
    out = d["OUTCOME"]
    if out == "UNPLACED":
        return {**d, "arm": arm, "condition": None, "committed_next": None, "note": UNPLACED_TEXT}
    cond, _ = table_row(out)
    return {**d, "arm": arm, "condition": cond, "committed_next": next_for_arm(out, arm)}


def cascade(a_block, b_block):
    """Section 10, the dose cascade.  A is read first.  B decides only after A's HARMS or NOT ADOPTED; otherwise B is a dose-response row beside that decides nothing and never promotes B over A.
    -> {"final_arm", "final", "b_status"}."""
    if a_block["OUTCOME"] not in SENDS_TO_B:
        return {"final_arm": "A", "final": a_block["OUTCOME"], "b_status": ("not read" if b_block is None else "beside: dose-response row, decides nothing, never pooled with A")}
    if b_block is None:
        return {"final_arm": "A", "final": a_block["OUTCOME"], "b_status": "PENDING: train B (4 runs, share 1/4) through its gate, with its own PI go, and read it under this same table"}
    return {"final_arm": "B", "final": b_block["OUTCOME"], "b_status": "decisive after A's HARMS or NOT ADOPTED (labelled with its dose; own seeds; never pooled with A)"}


def next_launch(mlaunch_moved):
    """PREREG section 11: M-LAUNCH moved in both pairs at both seeds -> LOOP-STATE; not moved -> LAUNCH-INTENT-HI; not computed (None) -> undecided (the reader does not build M-LAUNCH)."""
    if mlaunch_moved is None:
        return "UNDECIDED: M-LAUNCH is not computed by this reader (NOT BUILT); route NEXT-LAUNCH by hand from the tick dumps"
    if mlaunch_moved:
        return "LOOP-STATE"
    return "LAUNCH-INTENT-HI"


def read_a(rows_by_seed):
    """Read A (PREREG_V8F_DAG_REREAD section 5) on X - P10 at both seeds, as the SPEEDSUB-T1 reader applies it.  D1: ADE6 separated-better with point <= -2F; D2: no row separated-worse; D3: each of the
    four families (and PRIMARY) has a separated-better row; crawl rows pass iff neither L2 nor M1 is separated-worse at any seed.  DOMINANT needs D1-D3 at both seeds AND the crawl rows."""
    per = {}
    for s in SEEDS:
        rows = rows_by_seed[s]
        d1 = bool(sep_better("ADE6", rows["ADE6"]) and float(rows["ADE6"]["delta"]) <= -TWO_F)
        worse = [l for l in LABELS if sep_worse(l, rows[l])]
        fams = {f: any(sep_better(l, rows[l]) for l in LABELS if FAMILY[l] == f) for f in ("PRIMARY", "LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")}
        per[f"s{s}"] = {"D1": d1, "D2": not worse, "D2_worse_rows": worse, "D3": all(fams.values()), "D3_families": fams, "M1_sep_worse": sep_worse("M1 crawl share", rows["M1 crawl share"]),
                        "L2_sep_worse": sep_worse("L2 curvature MAE (GT>=5.2)", rows["L2 curvature MAE (GT>=5.2)"])}
    d1 = all(per[f"s{s}"]["D1"] for s in SEEDS)
    d2 = all(per[f"s{s}"]["D2"] for s in SEEDS)
    d3 = all(per[f"s{s}"]["D3"] for s in SEEDS)
    crawl_pass = not any(per[f"s{s}"]["M1_sep_worse"] or per[f"s{s}"]["L2_sep_worse"] for s in SEEDS)
    if not d1:
        label = "NO MARGIN"
    elif not (d2 and d3):
        label = "MARGIN, NOT DOMINANT"
    elif not crawl_pass:
        label = "DOMINANT, CRAWL OPEN"
    else:
        label = "DOMINANT"
    return {"outcome": label, "D1_both": d1, "D2_both": d2, "D3_both": d3, "crawl_rows_pass": crawl_pass, "per_seed": per}


def floors_from(tables):
    """RF_r = max over seeds of |d_r,s(REP - DAG-OACC)|; tables[s] = {label: row}.  Beside every row; not a bar term."""
    return {l: max(abs(float(_row(l, tables[s][l])["delta"])) for s in SEEDS) for l in LABELS}


def is_exact_zero(rows):
    """K-ZERO / identity: every computed row reads exactly +0 [0, 0]; a table with no computed row is not a pass."""
    comp = [r for r in rows.values() if not (r is None or r.get("status") == "NOT COMPUTED")]
    return bool(comp) and all(r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 for r in comp)


def families_report(rows_by_contrast):
    """Section 8: group the Read-A rows of each contrast by family, with the tier and estimator stamps; STRATEGIC is NOT PAUSED.  ``rows_by_contrast`` = {contrast name: {label: row}} for one seed and pair."""
    out = {}
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
        labs = [l for l in LABELS if FAMILY[l] == fam]
        out[fam] = {"tier": TIER_T1, "estimator": "paired episode-cluster bootstrap (B=2000, seed 0)",
                    "rows": {l: {cn: ({k: rows[l].get(k) for k in ("delta", "lo", "hi", "separated")} if rows.get(l) and rows[l].get("status") != "NOT COMPUTED" else "NOT COMPUTED") for cn, rows in rows_by_contrast.items()} for l in labs}}
    out["STRATEGIC"]["paused"] = False
    out["STRATEGIC"]["ruling"] = "NOT paused for v8F (PI ruling 2026-10-17)"
    out["LONGITUDINAL"]["note"] = "the lever acts on along-track pressure: read first. The anti-echo block is NOT BUILT (the DAG-ORC reader carries none)."
    out["TACTICAL"]["note"] = "T1-kin goal-head F1 and head accuracy are NOT COMPUTABLE here (the paired tool reads eval dumps, not tick dumps); TAC-LAT / TAC-LON are the plan-derived agreements; R-TLAT / R-TLON are the guards."
    return out


# -- the slow-draw flag (beside; decides nothing).  The method of rep_s0_diag/rep_s0_speedcells.py.
def cell_boot(d, eid, rng, n_boot=N_BOOT):
    """(mean, lo, hi) of the per-window paired difference ``d``: episode-cluster bootstrap, the pooled ratio of resampled sums to resampled counts, 2.5 / 97.5 percentiles (rep_s0_speedcells.py's ``boot``)."""
    d = np.asarray(d)                      # the dtype of the T0 npz (float32), as rep_s0_speedcells.py's boot() reads it: a float64 cast moves the 8th digit and the banked table is not reproduced bit for bit
    ueid = np.unique(eid)
    idx = {e: np.where(eid == e)[0] for e in ueid}
    m = np.isfinite(d)
    s = np.array([np.nansum(d[idx[e]]) for e in ueid])
    n = np.array([np.isfinite(d[idx[e]]).sum() for e in ueid])
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ueid), len(ueid))
        bs.append(s[pick].sum() / max(n[pick].sum(), 1))
    return float(np.nanmean(d[m])), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def speed_cells(cons_x, cons_y, eid, rng=None, n_boot=N_BOOT):
    """The T0 speed-cell shift of arm X against its pair's control Y: the five cells of ``cons_pred`` (column order CONS_NAMES), per-window paired, bootstrapped.  -> {cell: (mean, lo, hi)}."""
    rng = np.random.default_rng(0) if rng is None else rng
    cx, cy = np.asarray(cons_x), np.asarray(cons_y)
    if cx.shape != cy.shape or cx.shape[1] != len(CONS_NAMES) or len(eid) != len(cx):
        raise ValueError(f"cons_pred shapes {cx.shape} / {cy.shape}, eid {len(eid)}: not the {len(CONS_NAMES)}-column T0 constraint output on one window list")
    return {n: cell_boot(cx[:, CONS_NAMES.index(n)] - cy[:, CONS_NAMES.index(n)], eid, rng, n_boot) for n in CONS_NAMES if n in SPEED_CELLS}


def slow_draw(cells, thr=SLOW_SHIFT):
    """SLOW-T iff all five speed cells are separated downward (CI upper end < 0) and each shift exceeds ``thr`` in magnitude (strictly).  The mirror (all five separated upward by more than ``thr``) is FAST-T.
    Beside; decides nothing.  -> {"label", "cells"}."""
    if set(cells) != set(SPEED_CELLS):
        raise ValueError(f"the slow-draw flag needs exactly the five cells {SPEED_CELLS}, got {sorted(cells)}")
    slow = all(cells[n][2] < 0.0 and cells[n][0] < -thr for n in SPEED_CELLS)
    fast = all(cells[n][1] > 0.0 and cells[n][0] > thr for n in SPEED_CELLS)
    return {"label": "SLOW-T" if slow else ("FAST-T" if fast else None), "threshold_m_s": thr, "cells": {n: {"mean": cells[n][0], "lo": cells[n][1], "hi": cells[n][2]} for n in SPEED_CELLS}}


# -- run-record controls (pure on the records dag_read.train_records returns)
def parse_flags(argv):
    """argv -> {flag: [values-of-occurrence-1, ...]}; EVERY value after a flag is kept (DAG-ORC's own parse)."""
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
    def tok(x):
        try:
            return repr(float(x))
        except ValueError:
            return str(x).replace("\\", "/")
    return [[tok(x) for x in occ] for occ in vs]


def flag_diff(fa, fb, ignore=("--out",)):
    """Flags (parse_flags dicts) whose numerically normalised values differ, ignoring ``ignore``."""
    ka = {k: _norm(v) for k, v in fa.items() if k not in ignore}
    kb = {k: _norm(v) for k, v in fb.items() if k not in ignore}
    return sorted(k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k))


def one_value(flags, name):
    occ = flags.get(name)
    if not occ or len(occ) != 1 or len(occ[0]) != 1:
        return False, None
    return True, occ[0][0]


def run_seed_of(role, seed):
    return REP_SEED[seed] if role in ("Arep", "Brep", "REP") else seed


def lever_w(role):
    return W_ARM[ARM_OF_ROLE[role]] if role in ARM_OF_ROLE else None


def k_pair(role, seed, flags, ref_flags, a_flags=None):
    """K-PAIR (PREREG section 9), one per pair.  A_s: DAG-OACC_s's argv except --out and --launch-intent-w 1.58926.  A_rep,s: REP_s's argv except --out and --launch-intent-w 1.58926.  REP_s: DAG-OACC_s's
    argv except --out and --seed (10 / 11).  B_s / B_rep,s: the same against CTRL_s / REP_s with W_B, and equal to A_s / A_rep,s's argv except --out and the W.  ``ref_flags`` = the pair's control (CTRL_s for
    A, B and REP; REP_s for A_rep and B_rep); ``a_flags`` = the A arm's argv for a B role.  -> (ok, problems | description)."""
    bad = []
    if role in ROLES_NEW:
        d = flag_diff(flags, ref_flags)
        if d != [LEVER]:
            bad.append(f"{role}_s{seed}'s argv differs from its pair control's in {d}, expected exactly ['{LEVER}']")
        ok, v = one_value(flags, LEVER)
        if not ok or v != lever_w(role):
            bad.append(f"{LEVER} {v!r} on {role}_s{seed}, registered {lever_w(role)}")
        if role in ("B", "Brep") and a_flags is not None:
            d2 = flag_diff(flags, a_flags)
            if d2 != [LEVER]:
                bad.append(f"{role}_s{seed}'s argv differs from A's in {d2}, expected exactly ['{LEVER}']")
        return not bad, bad or f"argv({role}) = argv(its pair control) + {LEVER} {lever_w(role)}"
    if role == "REP":
        want = {k: v for k, v in ref_flags.items()}
        want["--seed"] = [[str(REP_SEED[seed])]]
        d = flag_diff(flags, want)
        return not d, ([f"REP_s{seed}'s argv differs from DAG-OACC_s{seed}'s with --seed {REP_SEED[seed]} in {d}"] if d else f"argv(REP) = argv(DAG-OACC) with --seed {REP_SEED[seed]}")
    return False, [f"unknown role {role!r}"]


def k_registered(role, seed, flags):
    """The fine-tune is NAMED on the argv (a default would not be a test): steps, lr, warmup, p-gt, --ego-perturb-p 0, --dagger-p 0.1, --dagger-seed s, --w-acc 49.6969, the bank sha256 = DAG-OACC's, --arm M2b,
    --seed (s, or 10 + s on the replicate roles), --init-ckpt M2b_s, the lever W exactly once on A / A_rep / B / B_rep and absent on CTRL / REP, and no --ego-perturb-with-dagger."""
    bad = []
    ok_w, v_w = one_value(flags, "--w-acc")
    if not ok_w or v_w != W_ACC:
        bad.append(f"--w-acc {v_w!r}, registered {W_ACC} (DAG-OACC's, verbatim)")
    ok_b, v_b = one_value(flags, "--dagger-bank-sha256")
    if not ok_b or v_b != OACC_BANK_SHA[seed]:
        bad.append(f"--dagger-bank-sha256 {v_b!r} is not DAG-OACC's {OACC_BANK_SHA[seed][:16]}...")
    ok_d, v_d = one_value(flags, "--dagger-seed")
    if not ok_d or v_d != str(seed):
        bad.append(f"--dagger-seed {v_d!r}, registered {seed} (DAG-OACC's)")
    if "--ego-perturb-with-dagger" in flags:
        bad.append("--ego-perturb-with-dagger is named: LAUNCH-INTENT does not compose the perturbation")
    for name, w in (("--steps", STEPS), ("--lr", LR), ("--warmup", WARMUP), ("--p-gt-start", PGT), ("--p-gt-end", PGT), ("--ego-perturb-p", 0.0), ("--dagger-p", SWAP_P), ("--seed", run_seed_of(role, seed))):
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
    if not ok or not str(v).replace("\\", "/").endswith(f"/M2b_s{seed}/ckpt_last.pt"):
        bad.append(f"--init-ckpt {v!r} is not .../M2b_s{seed}/ckpt_last.pt (the init is M2b_s{seed}, DAG-OACC's)")
    if role in ROLES_NEW:
        ok, v = one_value(flags, LEVER)
        if not ok or v != lever_w(role):
            bad.append(f"{LEVER} {v!r}, registered {lever_w(role)} on {role}")
    elif LEVER in flags:
        bad.append(f"{LEVER} is named on {role}: it must be absent")
    return not bad, bad or "argv names the registered fine-tune"


def k_init(seed, init_values):
    """K-INIT: every recorded init-checkpoint sha256 equals M2b_s's, as recorded in DAG-OACC's launch-gate token.  ``init_values`` = [(where, sha)]."""
    if not init_values:
        return False, "no record of the init checkpoint's sha256: the init is unproven"
    bad = [f"{w} = {str(v)[:16]}" for w, v in init_values if str(v) != M2B_CKPT_SHA[seed]]
    return not bad, ("init sha256 differs from the registered init of s%d (%s...): " % (seed, M2B_CKPT_SHA[seed][:16]) + "; ".join(bad)) if bad else {"init_sha256": M2B_CKPT_SHA[seed][:16] + "...", "sources": [w for w, _ in init_values]}


def _count(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def share_of(n, eligible, w):
    return ((1.0 + w) * n) / ((1.0 + w) * n + (eligible - n))


def k_share(arm, summary):
    """K-SHARE (replaces K-REACH), per trained arm (``arm`` in A, B, HI): from the arm's summary.json, launch_weight_share RECOMPUTED as (1 + W) n / ((1 + W) n + (eligible - n)) from launch_n, launch_eligible and
    the lever W lies in the target +- 0.02 (A 1/3, B 1/4, HI 1/2, inclusive), launch_rows == launch_eligible, and the recorded share agrees with the recomputed one."""
    bad, det = [], {}
    n, e, rows = summary.get("launch_n"), summary.get("launch_eligible"), summary.get("launch_rows")
    w = (summary.get("levers") or {}).get("launch_intent_w", summary.get("launch_intent_w"))
    if not (_count(n) and _count(e) and _count(rows) and e > 0 and 0 <= n <= e):
        return False, [f"launch_n {n!r} / launch_eligible {e!r} / launch_rows {rows!r}: not a count pair with 0 <= n <= eligible, eligible > 0"]
    if w is None or abs(float(w) - float(W_ARM[arm])) > 1e-12:
        bad.append(f"levers.launch_intent_w {w!r} is not arm {arm}'s {W_ARM[arm]}")
        return False, bad
    sh = share_of(n, e, float(w))
    det.update({"recomputed_share": sh, "target": SHARE_TARGET[arm], "launch_n": n, "launch_eligible": e})
    if not (SHARE_TARGET[arm] - SHARE_TOL - 1e-12 <= sh <= SHARE_TARGET[arm] + SHARE_TOL + 1e-12):
        bad.append(f"arm {arm}'s launch weight share {sh:.4f} outside {SHARE_TARGET[arm]:.4f} +- {SHARE_TOL}")
    if rows != e:
        bad.append(f"launch_rows {rows} != launch_eligible {e}")
    rec = summary.get("launch_weight_share")
    if not isinstance(rec, (int, float)) or abs(float(rec) - sh) > 1e-9:
        bad.append(f"the recorded launch_weight_share {rec!r} is not the recomputed {sh:.6f}")
    return not bad, bad or det


def k_share_control(summary):
    """K-SHARE for CTRL and REP: both were trained before the flag existed.  summary.json carries no launch_* key, and levers.launch_intent_w is absent or exactly 0.0."""
    bad = [f"summary.json carries {k!r}" for k in summary if str(k).startswith("launch")]
    levers = summary.get("levers") or {}
    if "launch_intent_w" in levers and levers["launch_intent_w"] != 0.0:
        bad.append(f"levers.launch_intent_w = {levers['launch_intent_w']!r}, must be absent or exactly 0.0")
    if levers.get("launch_n") not in (None, 0) or levers.get("launch_eligible") not in (None, 0):
        bad.append("levers carries a non-zero launch count")
    return not bad, bad or "no launch key; launch_intent_w absent or 0.0"


def k_mask(summary):
    """K-MASK, per trained arm: the training window list is DAG-OACC's (sha256) and launch_n / launch_eligible lies within +- 0.02 of f = 49,618 / 306,566 = 0.1619."""
    bad, det = [], {}
    if summary.get("train_window_list_sha256") != WINDOW_LIST_SHA:
        bad.append(f"train_window_list_sha256 {str(summary.get('train_window_list_sha256'))[:16]} is not DAG-OACC's {WINDOW_LIST_SHA[:16]}")
    n, e = summary.get("launch_n"), summary.get("launch_eligible")
    if _count(n) and _count(e) and e > 0:
        r = n / e
        det["launch_fraction"] = r
        if not (F_LAUNCH - MASK_TOL - 1e-12 <= r <= F_LAUNCH + MASK_TOL + 1e-12):
            bad.append(f"launch_n / launch_eligible = {r:.4f} outside f = {F_LAUNCH:.4f} +- {MASK_TOL}")
    else:
        bad.append(f"launch_n {n!r} / launch_eligible {e!r}: no fraction")
    return not bad, bad or det


def k_build(rec):
    """K-BUILD: the launch commit's v8f_mini_train.py, v8f_mini_data.py and v8f_mini.py are byte-identical to aeb54468's (the commit DAG-OACC trained at), or every changed hunk sits behind a default-off flag whose
    bit-identity test passes on the launch commit; where v8f_mini_data.py differs, its launch sha256 is DAG-ORC r2's (PREREG Amendment 1).  ``rec`` = {"launch_commit", "base_commit", "files": {path: {"sha256_launch", "sha256_base"}}, "changed": [{file, hunk, behind_flag, default_off, bit_identity_test}]}."""
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
            pin = LAUNCH_PINNED_FILES.get(f)
            if pin and fr["sha256_launch"] != pin:
                bad.append(f"{f} differs from {BUILD_BASE_COMMIT} and its launch sha256 is not {pin[:12]}... (DAG-ORC r2's build, PREREG Amendment 1)")
            hunks = [h for h in (rec.get("changed") or []) if h.get("file") == f]
            if not hunks:
                bad.append(f"{f} differs from {BUILD_BASE_COMMIT} and no changed hunk is listed")
            for h in hunks:
                if h.get("default_off") is not True or not h.get("behind_flag") or h.get("bit_identity_test") != "PASS":
                    bad.append(f"{f}: hunk {h.get('hunk')!r} is not behind a default-off flag with a PASSing bit-identity test")
    return not bad, bad or {"record": rec}


def k_rep_licence(rep, oacc, seed):
    """K-REP-LICENCE (PREREG section 4), the conditions that live in the run records.  ``rep`` / ``oacc`` = {"summary": summary.json, "token": launch-gate token, "flags": parse_flags(argv)} of REP_s and DAG-OACC_s.
    (1) REP finished and its run record reads: swap share 0.1 +- 0.02, ego exactly 0, composition flag absent, ego_perturb_stream 0; (2) the token has the same init_ckpt_sha256 and bank_sha256 as DAG-OACC's, the
    summary the same excluded_sha256 and train_window_list_sha256, the argv the same --w-acc; (4) the build term: REP's token commit is the DAG-ORC r2 commit and DAG-OACC's is aeb54468."""
    bad = []
    sm, tk = rep["summary"], rep["token"]
    lev = sm.get("levers") or {}
    if sm.get("done") is not True:
        bad.append("REP summary.json done is not true")
    dn, de = sm.get("dagger_n"), sm.get("dagger_eligible")
    if not (_count(dn) and _count(de) and de > 0 and 0 <= dn <= de):
        bad.append(f"REP dagger_n {dn!r} / dagger_eligible {de!r}: not a count pair")
    elif not (SWAP_P - SWAP_TOL - 1e-12 <= dn / de <= SWAP_P + SWAP_TOL + 1e-12):
        bad.append(f"REP swapped fraction {dn}/{de} = {dn / de:.4f} outside {SWAP_P} +- {SWAP_TOL}")
    if sm.get("ego_perturb_n") != 0:
        bad.append(f"REP ego_perturb_n = {sm.get('ego_perturb_n')!r}, must be exactly 0")
    if lev.get("ego_perturb_with_dagger") not in (False, None) or sm.get("ego_perturb_with_dagger") not in (False, None):
        bad.append("REP carries the composition flag")
    if lev.get("ego_perturb_stream") not in (0, None):
        bad.append(f"REP levers.ego_perturb_stream = {lev.get('ego_perturb_stream')!r}, must be 0")
    for k in ("init_ckpt_sha256", "bank_sha256"):
        if tk.get(k) != oacc["token"].get(k) or not tk.get(k):
            bad.append(f"token {k} {str(tk.get(k))[:16]} differs from DAG-OACC's {str(oacc['token'].get(k))[:16]}")
    if tk.get("init_ckpt_sha256") != M2B_CKPT_SHA[seed]:
        bad.append(f"token init_ckpt_sha256 {str(tk.get('init_ckpt_sha256'))[:16]} is not M2b_s{seed}'s")
    if tk.get("bank_sha256") != OACC_BANK_SHA[seed]:
        bad.append(f"token bank_sha256 {str(tk.get('bank_sha256'))[:16]} is not DAG-OACC's")
    for k in ("excluded_sha256", "train_window_list_sha256"):
        if sm.get(k) != oacc["summary"].get(k) or not sm.get(k):
            bad.append(f"summary {k} {str(sm.get(k))[:16]} differs from DAG-OACC's {str(oacc['summary'].get(k))[:16]}")
    if tk.get("excluded_sha256") != oacc["token"].get("excluded_sha256") or not tk.get("excluded_sha256"):
        bad.append(f"token excluded_sha256 {str(tk.get('excluded_sha256'))[:16]} differs from DAG-OACC's {str(oacc['token'].get('excluded_sha256'))[:16]}")
    try:
        w_ok = float(tk.get("w_acc")) == float(W_ACC) == float(oacc["token"].get("w_acc"))
    except (TypeError, ValueError):
        w_ok = False
    if not w_ok:
        bad.append(f"token w_acc {tk.get('w_acc')!r} / DAG-OACC's {oacc['token'].get('w_acc')!r} is not {W_ACC}")
    ok, w = one_value(rep["flags"], "--w-acc")
    if not ok or w != W_ACC:
        bad.append(f"REP --w-acc {w!r}, registered {W_ACC}")
    rc, oc = str((tk.get("binding") or {}).get("commit") or ""), str((oacc["token"].get("binding") or {}).get("commit") or "")
    if not rc.startswith(REP_BUILD_COMMIT):
        bad.append(f"REP token commit {rc[:8]!r} is not the DAG-ORC r2 commit {REP_BUILD_COMMIT}")
    if not oc.startswith(CTRL_BUILD_COMMIT):
        bad.append(f"DAG-OACC token commit {oc[:8]!r} is not {CTRL_BUILD_COMMIT}")
    return not bad, bad or {"swap_ratio": dn / de, "init": tk.get("init_ckpt_sha256")[:16], "bank": tk.get("bank_sha256")[:16], "commits": [oc[:8], rc[:8]]}


# -- the PREREG pin
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def prereg_problems(path, pin=None):
    """The PREREG carries, verbatim, the outcome table, the per-seed predicate table and the bar row this reader codes (markdown as printed); with ``pin`` its LF-normalised sha256 equals it."""
    raw = open(path, "rb").read().replace(b"\r\n", b"\n")
    txt = raw.decode("utf-8")
    bad = []
    if pin and hashlib.sha256(raw).hexdigest() != pin:
        bad.append(f"{path} has sha256 {hashlib.sha256(raw).hexdigest()[:12]}..., the pin is {pin[:12]}...")
    lines = [ln.strip() for ln in txt.split("\n")]
    for what, rows in (("outcome table", PREREG_OUTCOME_ROWS), ("per-seed predicate table", PREREG_PREDICATE_ROWS), ("bar row", (PREREG_BAR_ROW,))):
        for r in rows:
            if r not in lines:
                bad.append(f"the PREREG does not carry the {what} row: {r[:110]!r}")
    return bad


# -- discovery (pure filesystem; no run output is parsed)
def slot_needs(role, e):
    """The files one slot entry must name and have on disk.  CTRL and REP are banked (t1 lane, box 2's T0 eval, train records); a new arm adds the logged lane and the fresh T0 (K4)."""
    need = []
    td = e.get("train_dir")
    need += [("train_dir", td)] + [(f"train_dir/{f}", os.path.join(td, f) if td else None) for f in ("argv.json", "summary.json", "selfcheck.json")]
    need.append(("token", e.get("token")))
    keys = ("t0",) if role in ROLES_BANKED else ("t0", "t0_fresh")
    for k in keys:
        for f in ("json", "npz"):
            need.append((f"{k}.{f}", (e.get(k) or {}).get(f)))
    for k in (("t1",) if role in ROLES_BANKED else ("t1", "t1_logged")):
        t = e.get(k) or {}
        need += [(f"{k}.json", t.get("json")), (f"{k}.ticks", t.get("ticks")), (f"{k}.dump_dir", t.get("dump_dir"))]
    return need


def discover(manifest, roles=("A", "Arep", "CTRL", "REP", "B", "Brep"), seeds=SEEDS):
    """-> {role: {seed: {"state": PRESENT | MISSING | PARTIAL, "missing": [what, ...]}}}."""
    out = {}
    for role in roles:
        out[role] = {}
        for s in seeds:
            e = ((manifest.get("arms") or {}).get(role) or {}).get(str(s))
            if not e:
                out[role][s] = {"state": "MISSING", "missing": ["no manifest entry"]}
                continue
            miss = []
            for what, p in slot_needs(role, e):
                if not p or not os.path.exists(p):
                    miss.append(f"{what}: {p!r}")
                elif what.endswith("dump_dir") and len(glob.glob(os.path.join(p, "ep*.npz"))) < N_EPISODES:
                    miss.append(f"{what}: {p!r} holds {len(glob.glob(os.path.join(p, 'ep*.npz')))} of {N_EPISODES} episodes")
            out[role][s] = {"state": "PRESENT" if not miss else "PARTIAL", "missing": miss}
    return out


def load_train_records(e):
    d = e["train_dir"]
    r = {"dir": d}
    for f in ("argv.json", "summary.json", "selfcheck.json"):
        r[f] = json.load(open(os.path.join(d, f), encoding="utf-8"))
    r["argv"] = list(r["argv.json"].get("argv", []))
    r["flags"] = parse_flags(r["argv"])
    r["token"] = json.load(open(e["token"], encoding="utf-8"))
    return r


def find_init_values(tr):
    """The recorded init-checkpoint sha256 values of a run: summary / selfcheck levers (bare or under levers. / built_levers.)."""
    out = []
    for name in ("summary.json", "selfcheck.json"):
        d = tr.get(name) or {}
        for pre in ("", "levers.", "built_levers."):
            cur = d
            for part in (pre + "init_ckpt.sha256").split("."):
                cur = cur.get(part) if isinstance(cur, dict) else None
            if cur is not None:
                out.append((f"{name}:{pre}init_ckpt.sha256", cur))
    return out


def run_record_controls(ctl, trs, standin=False):
    """The run-record controls on the runs' train records.  ``trs[role][seed]`` = load_train_records(...).  K-INIT on every run; on the new arms K-REGISTERED, K-PAIR (against the pair's control, and a B role against
    its A role), K-SHARE and K-MASK; on CTRL and REP K-REGISTERED, K-SHARE (the control form) and, for REP, K-PAIR against CTRL; K-REP-LICENCE.  ``standin`` (the dry run: the banked dag-orc runs stand in for
    the new arms and carry no lever) skips the lever-specific checks of the new roles only, and says so."""
    for s in SEEDS:
        for role in trs:
            tr = trs[role][s]
            tag = f"{role}_s{s}"
            ctl.run(f"K-INIT:{tag}", lambda tr=tr, s=s: k_init(s, find_init_values(tr)))
            if role in ROLES_NEW:
                if standin:
                    ctl.add(f"K-PAIR:{tag}", True, "n/a: stand-in slot (the banked dag-orc runs carry no lever)")
                    continue
                ctl.run(f"K-REGISTERED:{tag}", lambda role=role, s=s, tr=tr: k_registered(role, s, tr["flags"]))
                a_role = {"B": "A", "Brep": "Arep"}.get(role)
                a_flags = trs[a_role][s]["flags"] if a_role in trs else None
                ctl.run(f"K-PAIR:{tag}", lambda role=role, s=s, tr=tr, a_flags=a_flags: k_pair(role, s, tr["flags"], trs[CONTROL_OF[role]][s]["flags"], a_flags))
                ctl.run(f"K-SHARE:{tag}", lambda role=role, tr=tr: k_share(ARM_OF_ROLE[role], tr["summary.json"]))
                ctl.run(f"K-MASK:{tag}", lambda tr=tr: k_mask(tr["summary.json"]))
            else:
                ctl.run(f"K-REGISTERED:{tag}", lambda role=role, s=s, tr=tr: k_registered(role, s, tr["flags"]))
                ctl.run(f"K-SHARE:{tag}", lambda tr=tr: k_share_control(tr["summary.json"]))
                if role == "REP":
                    ctl.run(f"K-PAIR:{tag}", lambda s=s, tr=tr: k_pair("REP", s, tr["flags"], trs["CTRL"][s]["flags"]))
    if "REP" in trs and "CTRL" in trs:
        for s in SEEDS:
            r, o = trs["REP"][s], trs["CTRL"][s]
            ctl.run(f"K-REP-LICENCE:s{s}", lambda r=r, o=o, s=s: k_rep_licence({"summary": r["summary.json"], "token": r["token"], "flags": r["flags"]}, {"summary": o["summary.json"], "token": o["token"], "flags": o["flags"]}, s))


# ---- heavy --------------------------------------------------------------------------------------------------------------------------------------------------------
def norm_path(p):
    return str(p).replace("\\", "/").rstrip("/")


def tail2(p):
    parts = norm_path(p).split("/")
    return "/".join(parts[-2:])


class Heavy:
    """The DAG-ORC `Rig` applied to the runs.  Built once; every table goes through the Rig's own `pair_table` / `A.table` unchanged.  Interface used by `read_core` (and implemented by the tests' fake):
    static_problems(), rf0(rollout_files), load_slot(ctl, role, seed, entry, standin), prepare(ctl), contrast(x, y), guard_rows(ctl, x, y, s, tab_xy), speed(x, y, s), finish(ctl)."""

    def __init__(self, orc_dir=ORC_DIR, work_dir=None, n_boot=N_BOOT):
        sys.path.insert(0, orc_dir)
        import dag_orc_read as O
        self.O = O
        self.a = O.default_args(work_dir=work_dir or f"{HERE}/_work", n_boot=n_boot)
        self.rig = O.Rig(self.a)
        self.DR, self.A, self.T, self.K = self.rig.DR, self.rig.A, self.rig.T, self.rig.K
        self.n_boot = n_boot
        self.recs, self.dumps, self.cache, self.k1_items = {}, {}, {}, []

    def static_problems(self):
        return self.O.static_problems(self.a)

    def rf0(self, rollout_files):
        return self.O.k_p0(self.a.rf0, rollout_files)

    def load_slot(self, ctl, role, seed, e, standin=False):
        """One run through the readers' own controls (summary, gate token, T1 pipeline, T1 config, K4 for a new arm, T0 arrays) -> rec, stored."""
        DR, rig, np_ = self.DR, self.rig, np
        tag, nb = f"{role}_s{seed}", self.n_boot
        rec = {"role": role, "seed": seed, "tag": tag, "entry": e}
        tr = DR.train_records(e, "init_ckpt.sha256")
        rec["train"] = tr
        run_seed = seed if (standin and role == "Arep") else run_seed_of(role, seed)          # the dry run's replicate-pair stand-in is DAG-ORC's A at its own seed
        ctl.run(f"SUMMARY:{tag}", lambda: DR.g_summary(tag, run_seed, tr))
        ctl.run(f"K-GATE:{tag}", lambda: DR.g_token(tag, tr, DR.read_token(e.get("token"))))
        ckpt_tail = e.get("ckpt_tail") or f"{os.path.basename(norm_path(e['train_dir']))}/ckpt_last.pt"
        rec["ckpt_tail"] = ckpt_tail
        t1 = e["t1"]
        rec["t1json"] = DR.jload(DR.need(t1["json"], f"{tag} t1v8f json"))
        rec["dump"] = DR.load_dump(t1["dump_dir"])
        rec["pack"] = pk = rig.backend.analyze_t1(t1, t1.get("tag") or tag, seed)
        for k, (ok, det) in pk.controls.items():
            ctl.add(f"{k}:{tag}", ok, det)
        ctl.run(f"T1-CONFIG:{tag}", lambda: DR.g_t1cfg(tag, rec["t1json"], ckpt_tail, "causal", pk.meta))
        if role in ROLES_NEW:
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
            t0fj = DR.jload(DR.need(t0f["json"], "fresh T0 json"))
            ctl.add(f"T0F-CKPT:{tag}", DR.tail2(str(t0fj.get("ckpt", ""))) == ckpt_tail, f"fresh T0 eval json ckpt {t0fj.get('ckpt')!r} vs .../{ckpt_tail}")
        t0e = e.get("t0")
        if not t0e:
            raise DR.Refusal(f"{tag}: manifest has no 't0' (box 2's standard T0 eval)")
        rec["t0json"], rec["t0arr"], t0_ok = DR.t0_arrays(t0e, ckpt_tail)
        ctl.add(f"T0-CKPT:{tag}", t0_ok, f"T0 eval json ckpt {rec['t0json'].get('ckpt')!r} vs .../{ckpt_tail}")
        with np_.load(DR.need(t0e["npz"], "T0 eval npz")) as z:
            rec["t0cons"] = (np_.asarray(z["cons_pred"]), np_.asarray(z["eid"]))
        d = rec["dump"]
        ctl.add(f"DUMP-WINDOWS:{tag}", bool(len(d["ws"]) == pk.N and np_.array_equal(d["clip_index"], pk.ci) and np_.array_equal(d["ws"], pk.t0)), {"dump_n": int(len(d["ws"])), "ticks_n": int(pk.N)})
        ctl.add(f"GRID-COUNTS:{tag}", bool(pk.N == N_WINDOWS and len(np_.unique(d["eid"])) == N_EPISODES), {"windows": int(pk.N), "episodes": int(len(np_.unique(d["eid"])))})
        js = (rec["t1json"].get("paired_decision_grade") or {}).get("paired_model_minus_floor_t1", {}).get("ade_m")

        def r2():
            if js is None:
                return False, "the run json carries no paired_model_minus_floor_t1.ade_m to reproduce"
            A_, B_ = self.K.comps(d["cl"], d["g"])["ade6_m"], self.K.comps(d["fl"], d["g"])["ade6_m"]
            mk = np_.isfinite(A_) & np_.isfinite(B_)
            r = self.T.PB(A_[mk], B_[mk], d["eid"][mk], n_boot=nb, seed=0)
            return abs(r["delta"] - float(js["delta"])) <= 1e-6, {"reader_delta": r["delta"], "analysis_delta": js["delta"]}
        ctl.run(f"R2-ADE6:{tag}", r2)
        self.recs.setdefault(role, {})[seed] = rec
        return rec

    def prepare(self, ctl):
        """Window and floor-planner equality of every run with CTRL's dump (C4, C3); the per-seed Read-A context; P10 := CTRL's `fl`."""
        np_, A = np, self.A
        for s in SEEDS:
            self.dumps[s] = {role: A.dump(per[s]["entry"]["t1"]["dump_dir"]) for role, per in self.recs.items()}
            ref = self.dumps[s]["CTRL"]
            for role, d in self.dumps[s].items():
                if role == "CTRL":
                    continue
                same = all(np_.array_equal(d[k], ref[k]) for k in ("clip_index", "ws", "eid", "g", "v0"))
                ctl.add(f"C4-WINDOWS:{role}-vs-CTRL:s{s}", bool(same and len(d["eid"]) == N_WINDOWS and len(np_.unique(d["eid"])) == N_EPISODES), "clip_index, ws, eid, g, v0 bit-equal; 2399 windows / 141 episodes")
                ctl.add(f"C3-FLOOR:{role}-vs-CTRL:s{s}", bool(np_.array_equal(d["fl"], ref["fl"])), "P10's tracks read no model: bit-equal")
            self.dumps[s]["P10"] = self.dumps[s]["CTRL"]
            self.cache[s] = {"cx": A.tacdk_context(self.dumps[s]["CTRL"]), "P": {}}

    def contrast(self, x, y):
        """Read A's 19 rows for X - Y at both seeds.  STRATEGIC: for P10 the within-run (S - CTRV) contrast of X; otherwise the difference of the two runs' CTRV contrasts on the same windows."""
        cj, tj = {}, {}
        for s in SEEDS:
            cj[s], tj[s] = self.rig.pair_table(self.dumps[s], self.cache[s], x, y, self.dumps[s]["CTRL"])
        within = y == "P10"
        dag = {"beside": {"per_seed": {f"s{s}": {"per_run_t1heads": {x: self.recs[x][s]["pack"].extra}} for s in SEEDS}}} if within else {}
        t = self.A.table(cj, tj, dag, x if within else None)
        if not within:
            for s in SEEDS:
                t[s] = self.O.with_spos(t[s], self.rig.spos_pair(self.recs[x][s]["pack"].ticks_path, self.recs[y][s]["pack"].ticks_path, self.recs[x][s]["pack"].ci))
        return t

    def guard_rows(self, ctl, x, y, s, tab_xy):
        """R-TLAT / R-TLON (T-head macro-F1, matched cells), the R-ADE link, R-T0 (T0 own-code ADE6 point) and R-KEEP's point, for X - Y at seed s."""
        DR, T, K, np_, nb = self.DR, self.T, self.K, np, self.n_boot
        rx, ry = self.recs[x][s], self.recs[y][s]
        g = {}
        for hd, nm in (("lat", "R-TLAT"), ("lon", "R-TLON")):
            r, (a_, b_, eid2, red) = DR.f1_pair_row(T, rx["pack"], ry["pack"], hd, nb, matched=True)
            g[nm] = r
            self.k1_items.append((f"{nm} {x}-{y} s{s}", a_, eid2, red))
        dx, dy = rx["dump"], ry["dump"]
        ade_link = DR.err_pair_row(T, f"R-ADE T1-kin ADE6 ({x} - {y}) s{s}", K.comps(dx["cl"], dx["g"])["ade6_m"], K.comps(dy["cl"], dy["g"])["ade6_m"], dx["eid"], nb)
        ctl.add(f"ADE-LINK:{x}-{y}:s{s}", abs(float(tab_xy[s]["ADE6"]["delta"]) - float(ade_link["delta"])) <= 1e-9, {"readA_rows_block": tab_xy[s]["ADE6"]["delta"], "layer_G_R-ADE": ade_link["delta"]})
        t0 = DR.err_pair_row(T, f"R-T0 T0 own-code ADE6 ({x} - {y}) s{s}", rx["t0arr"]["ade6"], ry["t0arr"]["ade6"], rx["t0arr"]["eid"], nb)
        ctl.add(f"T0-WINDOWS:{x}-{y}:s{s}", bool(np_.array_equal(rx["t0arr"]["eid"], ry["t0arr"]["eid"]) and len(rx["t0arr"]["ade6"]) == N_WINDOWS), "T0 eid identical window for window")
        ra = tab_xy[s]["ADE6"]
        g["T0_point"], g["KEEP_point"] = t0["point"], float(ra["mean_arm"]) - float(ra["mean_ref"])
        g["R-T0_row"], g["R-ADE_row"] = t0, ade_link
        a6 = K.comps(dx["cl"], dx["g"])["ade6_m"]
        mk = np_.isfinite(a6)
        self.k1_items += [(f"R-ADE {x} s{s}", a6[mk], dx["eid"][mk], "mean"), (f"R-T0 {x} s{s}", rx["t0arr"]["ade6"], rx["t0arr"]["eid"], "mean")]
        return g

    def speed(self, x, y, s, rng=None):
        (cx, ex), (cy, ey) = self.recs[x][s]["t0cons"], self.recs[y][s]["t0cons"]
        if not np.array_equal(ex, ey):
            raise ValueError(f"T0 eid differs between {x} and {y} at s{s}")
        return speed_cells(cx, cy, ex, rng, self.n_boot)

    def finish(self, ctl):
        ok1, det1 = self.DR.k1_own(self.T, self.k1_items, self.n_boot)
        ctl.add("K1-LI-ROWS", ok1, det1)


def trim(rows):
    return {l: ({k: r.get(k) for k in ("delta", "lo", "hi", "separated", "n", "mean_arm", "mean_ref", "family", "direction", "separated_better", "separated_worse", "kind")} if r.get("status") != "NOT COMPUTED" else r)
            for l, r in rows.items() if not str(l).startswith("_")}


def pair_roles(x):
    """The two pairs of dose arm x ('A' or 'B'): main (x, CTRL), replicate (x + 'rep', REP)."""
    return {"main": (x, "CTRL"), "rep": (x + "rep", "REP")}


def read_core(H, ctl, man, rf0, standin=False, bars=None, speed_rng=None):
    """The whole analysis.  Slots: A, Arep, CTRL, REP (+ B, Brep when both are present).  -> dict of tables, G rows, floors, slow-draw flags, per-arm compose and decisions (nothing is written here).
    ``bars`` overrides the v2 bars (the dry run's K-RC reproduction of DAG-ORC's banked composition passes DAG-ORC's own bars)."""
    xs = ["A"] + (["B"] if all(((man["arms"].get("B") or {}).get(str(s))) and ((man["arms"].get("Brep") or {}).get(str(s))) for s in SEEDS) else [])
    roles = [r for x in xs for r in (x, x + "rep")] + ["CTRL", "REP"]
    trs = {}
    for role in roles:
        for s in SEEDS:
            e = (man["arms"].get(role) or {}).get(str(s))
            if not e:
                raise ValueError(f"manifest has no {role} seed {s}")
            trs.setdefault(role, {})[s] = load_train_records(e)
    run_record_controls(ctl, trs, standin=standin)
    for role in roles:
        for s in SEEDS:
            H.load_slot(ctl, role, s, man["arms"][role][str(s)], standin)
    H.prepare(ctl)
    bars = bars or bar_table(rf0)
    tabs, ident = {}, {}
    # RF first (beside): REP - DAG-OACC, and the structural zeros
    ident["REP"] = H.contrast("REP", "REP")
    ctl.add("K-ZERO:REP-REP", all(is_exact_zero(ident["REP"][s]) for s in SEEDS), "REP - REP reads exactly 0.0 with CI [0, 0] on every row")
    ctl.add("K-ZERO:RF-of-REP-as-CTRL", all(v == 0.0 for v in floors_from({s: ident["REP"][s] for s in SEEDS}).values()), "feeding REP's dump as CTRL makes RF exactly 0.0 on every row")
    tabs["REP_CTRL"] = H.contrast("REP", "CTRL")
    rf = floors_from({s: tabs["REP_CTRL"][s] for s in SEEDS})
    tabs["CTRL_P10"], tabs["REP_P10"] = H.contrast("CTRL", "P10"), H.contrast("REP", "P10")
    for role in roles:
        if role not in ident:
            ident[role] = H.contrast(role, role)
        for s in SEEDS:
            ctl.add(f"K-ZERO:{role}-{role}:s{s}", is_exact_zero(ident[role][s]), "(X, X) reads +0 [0, 0] on every Read-A row (comps, TAC / DK, S-POS)")
    res = {"tabs": tabs, "floors": {"RF0": rf0, "RF": rf, "F": {l: (F_ADE6 if l == "ADE6" else 0.0) for l in LABELS}, "bar": bars}, "xs": xs, "g": {}, "slow": {}, "per_arm": {}}
    for x in xs:
        res["per_arm"][x] = {}
        for pair, (px, py) in pair_roles(x).items():
            tabs[f"{px}_P10"] = H.contrast(px, "P10")
            tabs[f"{px}_{py}"] = H.contrast(px, py)
            res["g"][(x, pair)] = {s: H.guard_rows(ctl, px, py, s, tabs[f"{px}_{py}"]) for s in SEEDS}
            res["slow"][(x, pair)] = {}
            for s in SEEDS:
                res["slow"][(x, pair)][s] = slow_draw(H.speed(px, py, s) if speed_rng is None else H.speed(px, py, s, speed_rng))
    H.finish(ctl)
    res["ctl_ok_after_load"] = not ctl.failed()
    return res


def compose_all(res, ctl_ok):
    """Per dose arm x: compose_pair on both pairs at both seeds, the per-seed AND, the arm's decision, its outcome block.  Then the caller runs the cascade."""
    out = {}
    bars = res["floors"]["bar"]
    for x in res["xs"]:
        per_seed = {}
        for s in SEEDS:
            pairs = {}
            for pair, (px, py) in pair_roles(x).items():
                t = {"X_P10": res["tabs"][f"{px}_P10"][s], "X_Y": res["tabs"][f"{px}_{py}"][s], "Y_P10": res["tabs"][f"{py}_P10"][s]}
                pairs[pair] = compose_pair(t, res["g"][(x, pair)][s], bars)
            per_seed[s] = compose_seed(pairs["main"], pairs["rep"])
        out[x] = {"per_seed": per_seed, "decision": outcome_block(decide_arm(ctl_ok, per_seed), x)}
    return out


def families_block(res, x):
    return {f"s{s}": {pair: families_report({f"{px} - {py}": res["tabs"][f"{px}_{py}"][s], f"{px} - P10": res["tabs"][f"{px}_P10"][s], f"{py} - P10": res["tabs"][f"{py}_P10"][s]})
                      for pair, (px, py) in pair_roles(x).items()} for s in SEEDS}


def beside_flags(res, x):
    """RF and the bar per row (beside), the slow-draw flag per trained arm and seed."""
    return {"RF_beside_every_row": {l: {"RF": res["floors"]["RF"][l], "RF0": res["floors"]["RF0"][l], "bar": res["floors"]["bar"][l]} for l in LABELS},
            "slow_draw": {f"{pair}:{pxy[0]}-{pxy[1]}:s{s}": res["slow"][(x, pair)][s] for pair, pxy in pair_roles(x).items() for s in SEEDS}}


def provenance(a, t_start, manifest_path):
    return {"prereg": PREREG_ID, "rf0_sha256": RF0_SHA256, "claim_row": CLAIM_ROW, "reader_sha256": sha256_file(os.path.abspath(__file__)), "orc_reader": f"{a.orc_dir}/dag_orc_read.py",
            "manifest_sha256": sha256_file(manifest_path) if manifest_path and os.path.isfile(manifest_path) else None, "wall_s": round(time.time() - t_start, 1),
            "evidence_class": "MEASURED (ours; artifact paths in the manifest)"}


NOT_BUILT = {"M-LAUNCH": "X - Y of the mean (v_own - v_GT) at ticks 1-2 in the GT-launch windows, from the tick dumps, in each pair: NOT BUILT (routes NEXT-LAUNCH by hand)",
             "anti_echo_block": "longitudinal_claim_admissible, hold-v0, copy detector (RESULT_V8F_ORECOVER 3.1): NOT BUILT",
             "M2_and_L2_launch_subset": "along-track MAE, speed MAE and L2 on the GT-launch subset; M2: NOT BUILT",
             "t0_four_families": "orc_families.py pointed at the arms and their controls (the T0 LONGITUDINAL / LATERAL / TACTICAL rows and distance keeping at T0): NOT RUN, a build item, never a silent omission"}


def rollout_files(man):
    """Every file of the new arms' dev-box rollouts (dumps, tick files, run jsons, both lanes): K-P0 requires the frozen RF0 to predate all of them."""
    out = []
    for role in ROLES_NEW:
        for s in SEEDS:
            e = ((man.get("arms") or {}).get(role) or {}).get(str(s)) or {}
            for lane in ("t1", "t1_logged"):
                t = e.get(lane) or {}
                out += glob.glob(os.path.join(t.get("dump_dir") or "", "ep*.npz")) + [t.get("json") or "", t.get("ticks") or ""]
    return out


def main_read(a):
    """THE READ.  REFUSED (rc 2) writes only the failed controls (no row); a missing required slot is rc 4 (nothing written); otherwise the outcome and every row are written (rc 0, or 3 for UNPLACED)."""
    t_start = time.time()
    man = json.load(open(a.manifest, encoding="utf-8"))
    slots = discover(man)
    miss = {r: {f"s{s}": slots[r][s] for s in SEEDS if slots[r][s]["state"] != "PRESENT"} for r in REQUIRED_ROLES}
    miss = {k: v for k, v in miss.items() if v}
    if miss:
        print("LAUNCH-INTENT READ BLOCKED (nothing written): required slots are not complete:", file=sys.stderr)
        print(json.dumps(miss, indent=1), file=sys.stderr)
        return 4
    b_slots = {f"{r}_s{s}": slots[r][s] for r in ("B", "Brep") for s in SEEDS}
    if any(v["state"] != "MISSING" for v in b_slots.values()) and not all(v["state"] == "PRESENT" for v in b_slots.values()):
        print("LAUNCH-INTENT READ BLOCKED (nothing written): B / B_rep are half-present (all four slots B_0, B_1, B_rep,0, B_rep,1 or none):", json.dumps({k: v for k, v in b_slots.items() if v["state"] != "PRESENT"}), file=sys.stderr)
        return 4
    bad = prereg_problems(a.prereg, a.prereg_sha or None) if a.prereg else ["--prereg is required"]
    if bad:
        print("LAUNCH-INTENT READ REFUSED (nothing written): the PREREG pin:", *bad, sep="\n  ", file=sys.stderr)
        return 2
    H = Heavy(a.orc_dir, a.work_dir, a.n_boot)
    ctl = Ctl()
    sp = H.static_problems()
    ctl.add("STATIC", not sp, sp or "staged readers, Read A's banked inputs, DAG-ORC's PREREG text, RF0 pin")
    pr, rf0, det = H.rf0(rollout_files(man))
    ctl.add("K-P0", not pr, pr or {"RF0_sha256": RF0_SHA256[:16] + "...", **det})
    ctl.run("K-BUILD", lambda: k_build(man.get("build")))
    if not rf0:                                    # K-P0 failed: there is no RF0 to build the bars from; the REFUSED record still carries every control read so far
        return write_result(a, t_start, ctl, {}, {})
    try:
        res = read_core(H, ctl, man, rf0 or {})
        comp = compose_all(res, not ctl.failed())
    except (RowNotComputed, OSError, KeyError, ValueError) as e:
        print(f"LAUNCH-INTENT READ REFUSED (nothing written): {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return write_result(a, t_start, ctl, res, comp)


def write_result(a, t_start, ctl, res, comp):
    if ctl.failed():
        print("LAUNCH-INTENT READ REFUSED: failed controls:", file=sys.stderr)
        for k, v in ctl.failed().items():
            print("  -", k, str(v["detail"])[:300], file=sys.stderr)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"OUTCOME": "REFUSED", "committed_next": table_row("REFUSED")[1], "controls_PASS": ctl.flags(), "failed": {k: v["detail"] for k, v in ctl.failed().items()},
                       "note": "no row is reported (PREREG section 10)", "provenance": provenance(a, t_start, a.manifest)}, f, indent=1, default=str)
        return 2
    casc = cascade(comp["A"]["decision"], comp["B"]["decision"] if "B" in comp else None)
    final = comp[casc["final_arm"]]["decision"]
    out = {"OUTCOME": final, "cascade": casc, "controls_PASS": ctl.flags(), "A": comp["A"]["decision"], "B": comp["B"]["decision"] if "B" in comp else None,
           "rows": {x: {f"s{s}": comp[x]["per_seed"][s] for s in SEEDS} for x in comp}, "floors": res["floors"],
           "beside": {x: beside_flags(res, x) for x in res["xs"]},
           "read_a_on_adoption": ({x: {"main": read_a(res["tabs"][f"{x}_P10"]), "rep_beside": read_a(res["tabs"][f"{x}rep_P10"])} for x in res["xs"] if comp[x]["decision"]["adopts"]} or "not applicable: no arm adopts"),
           "next_launch": next_launch(None) if final["OUTCOME"] in ("PARTIAL / ONE SEED", "NO EFFECT", "PARTIAL / ADOPTED") or (final["arm"] == "B" and final["OUTCOME"] in SENDS_TO_B) else None,
           "read_a": {k: {f"s{s}": trim(v[s]) for s in SEEDS} for k, v in res["tabs"].items()},
           "four_families": {x: families_block(res, x) for x in res["xs"]}, "not_built": NOT_BUILT, "tier": {"T1": TIER_T1, "T0": TIER_T0}, "estimator": ESTIMATOR,
           "strategic_family": "NOT PAUSED for v8F (PI ruling 2026-10-17)", "provenance": provenance(a, t_start, a.manifest)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"OUTCOME": final["OUTCOME"], "arm": final["arm"], "adopts": final["adopts"], "committed_next": final["committed_next"], "cascade": casc}, indent=1, ensure_ascii=False))
    print(f"ZZLAUNCH-INTENT-READ-DONE {final['OUTCOME']} -> {a.out}", flush=True)
    return 3 if final["OUTCOME"] == "UNPLACED" else 0


# ---- the dry run: the banked dag-orc dumps standing in for the pairs ------------------------------------------------------------------------------------------------
DAGORC_OUT = "F:/TanitAD-scratch/dagorc_t1_out"
BANKED_ORC_READ, BANKED_ORC_READ_SHA = f"{DAGORC_OUT}/read/DAGORC_READ.json", "70a7a3a46e520311ade0cf2e1c71203010a25185ef5eb82b9d34b988b9548112"
BANKED_RF, BANKED_RF_SHA = f"{DAGORC_OUT}/read/RF_REP.json", "64c9444d486939be8dbbf61ef67028c35aa56b13c5d4bdbbc71d6e17e6fc497a"
OACC_READ_JSON, OACC_READ_SHA = f"{ART}/v8f-dagoacc-t1/read/DAGOACC_READ.json", "9e6bd4f6940fb3e868d4d8aede50cf47c082a3833f014c34bb87ff201c99a894"
BANKED_SPEEDCELLS, BANKED_SPEEDCELLS_SHA = f"{HERE}/ref/rep_s0_speedcells.json", "9788df70043ca3e5a4bdf902dd7eb16205348c28dd38b4533f310d643e95188f"
PAIR_KEYS = (("A_CTRL", "A_CTRL"), ("A_P10", "A_P10"), ("CTRL_P10", "CTRL_P10"), ("REP_CTRL", "REP_CTRL"))      # (mine, DAG-ORC's banked read_a key)
COMPOSE_KEYS = ("flags", "guards", "R-CRAWL_rows", "R-MOVE_rows", "domain", "R-NEW_fires", "R-NOBASE_violations", "T0_point", "KEEP_point")


def standin_manifest():
    """The dag-orc slots stand in for the pairs: A := DAG-ORC's A (main pair AND, as a plumbing stand-in, the replicate pair's arm), CTRL := DAG-OACC, REP := REP.  The replicate pair
    (DAG-ORC's A against REP) is NOT a one-lever pair and has no banked table: it exercises the machinery only."""
    m = json.load(open(f"{DAGORC_OUT}/runs.json", encoding="utf-8"))
    return {"seeds": [0, 1], "arms": {"A": m["arms"]["A"], "Arep": m["arms"]["A"], "CTRL": m["arms"]["CTRL"], "REP": m["arms"]["REP"]}, "build": None}


def same_value(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def row_diffs(mine, banked):
    """Every (label, key) at which two trimmed Read-A tables differ, by exact equality; labels or keys on one side only count."""
    bad = []
    for lab in sorted(set(mine) | set(banked)):
        ra, rb = mine.get(lab), banked.get(lab)
        if ra is None or rb is None:
            bad.append([lab, "<row>", ra is None, rb is None])
            continue
        for k in sorted(set(ra) | set(rb)):
            if not same_value(ra.get(k), rb.get(k)):
                bad.append([lab, k, ra.get(k), rb.get(k)])
    return bad


def max_abs_delta(mine, banked):
    return max([abs(float(mine[l][k]) - float(banked[l][k])) for l in banked if l in mine and mine[l].get("delta") is not None for k in ("delta", "lo", "hi")] or [0.0])


def main_dry(a):
    """The banked stand-ins through the whole read (PREREG section 9, K-RC): with the dag-orc dumps standing in for the pairs (CTRL = DAG-OACC, REP = REP) the reader reproduces DAG-ORC's banked A - CTRL,
    A - P10, DAG-OACC - P10 and REP - CTRL tables, RF (RF_REP.json) and the bars with 0 differences; its composition, handed DAG-ORC's own bars, reproduces the banked per-seed predicates of the main pair; every
    (X, X) contrast reads exactly 0.0; DAG-OACC - P10 equals DAGOACC_READ.json; the slow-draw method reproduces rep_s0_speedcells.json and flags REP s0 alone; the REP licence reads PASS on the real REP runs.
    No LAUNCH-INTENT run is read: the outcome printed under 'standin_outcome_v2_bars' is the machinery's output on stand-ins, not a result."""
    t_start = time.time()
    pins = {BANKED_ORC_READ: BANKED_ORC_READ_SHA, BANKED_RF: BANKED_RF_SHA, OACC_READ_JSON: OACC_READ_SHA, BANKED_SPEEDCELLS: BANKED_SPEEDCELLS_SHA}
    ctl = Ctl()
    for p, want in pins.items():
        got = sha256_file(p) if os.path.isfile(p) else None
        ctl.add(f"PIN:{os.path.basename(p)}", got == want, {"path": p, "sha256": got, "pin": want})
    if ctl.failed():
        print("LAUNCH-INTENT DRY RUN STOPPED: a banked input is not the pinned file:", json.dumps({k: v["detail"] for k, v in ctl.failed().items()}), file=sys.stderr)
        return 2
    banked = json.load(open(BANKED_ORC_READ, encoding="utf-8"))
    rf_banked = json.load(open(BANKED_RF, encoding="utf-8"))["RF"]
    oacc = json.load(open(OACC_READ_JSON, encoding="utf-8"))
    sc_banked = json.load(open(BANKED_SPEEDCELLS, encoding="utf-8"))
    H = Heavy(a.orc_dir, a.work_dir, a.n_boot)
    sp = H.static_problems()
    ctl.add("STATIC", not sp, sp or "staged readers, Read A's banked inputs, DAG-ORC's PREREG text, RF0 pin")
    pr, rf0, det = H.rf0([])
    ctl.add("K-P0", not pr, pr or {"RF0_sha256": RF0_SHA256[:16] + "...", **det})
    man = standin_manifest()
    res = read_core(H, ctl, man, rf0 or {}, standin=True, bars=banked["floors"]["bar"])
    out = {"DRY_RUN": True, "note": "the banked dag-orc dumps stand in for the pairs; no LAUNCH-INTENT run exists or is read", "standin_map": {"A": "DAG-ORC A", "Arep": "DAG-ORC A (plumbing stand-in)", "CTRL": "DAG-OACC", "REP": "REP (DAG-ORC r2)"}}
    # 1. the banked tables, row by row, key by key
    tabs_cmp = {}
    for mine_k, bank_k in PAIR_KEYS:
        for s in SEEDS:
            mine, bk = trim(res["tabs"][mine_k][s]), banked["read_a"][bank_k][f"s{s}"]
            d = row_diffs(mine, bk)
            tabs_cmp[f"{bank_k}:s{s}"] = {"n_rows": len(bk), "n_differences": len(d), "max_abs_delta_lo_hi": max_abs_delta(mine, bk), "differences": d[:8]}
    out["banked_tables_vs_reader"] = tabs_cmp
    ctl.add("K-RC:DAG-ORC-tables", all(v["n_differences"] == 0 for v in tabs_cmp.values()), {k: v["n_differences"] for k, v in tabs_cmp.items()})
    # 2. RF and the banked bars
    rf_ok = res["floors"]["RF"] == rf_banked == banked["floors"]["RF"]
    out["RF"] = {"reader_equals_RF_REP_json": res["floors"]["RF"] == rf_banked, "reader_equals_DAGORC_READ_floors_RF": res["floors"]["RF"] == banked["floors"]["RF"]}
    ctl.add("K-RC:RF", rf_ok, out["RF"])
    bars_banked_form = {l: 2.0 * max(F_ADE6 if l == "ADE6" else 0.0, abs(rf0[l]), abs(res["floors"]["RF"][l])) for l in LABELS}
    ctl.add("K-RC:v1-bars", bars_banked_form == banked["floors"]["bar"], "2 * max(F, RF0, RF) from the reader's own RF equals DAG-ORC's banked bars")
    # 3. the main pair's composition, handed DAG-ORC's own bars, against the banked per-seed predicates
    comp_old = compose_all(res, not ctl.failed())
    cmp_c = {}
    for s in SEEDS:
        mine, bk = comp_old["A"]["per_seed"][s]["pairs"]["main"], banked["rows"]["A"][f"s{s}"]
        cmp_c[f"s{s}"] = {k: (mine[k] == bk[k]) for k in COMPOSE_KEYS}
    out["main_pair_composition_with_DAG_ORC_bars_vs_banked_per_seed_predicates"] = cmp_c
    ctl.add("K-RC:composition", all(all(v.values()) for v in cmp_c.values()), cmp_c)
    # 4. the v2 bar on the same stand-ins (the machinery's output, not a result)
    res2 = {**res, "floors": {**res["floors"], "bar": bar_table(rf0)}}
    comp2 = compose_all(res2, not ctl.failed())
    out["standin_outcome_v2_bars"] = {"NOTE": "NOT A LAUNCH-INTENT RESULT: DAG-ORC's A on stand-in pairs", "OUTCOME": comp2["A"]["decision"]["OUTCOME"], "flags": comp2["A"]["decision"]["flags"],
                                      "crawl_bars_v2": {l: bar_table(rf0)[l] for l in CRAWL_SET}}
    # 5. DAG-OACC - P10 against DAGOACC_READ.json (K-RC leg 2)
    leg2 = {}
    for s in SEEDS:
        bt = oacc["rows"][f"s{s}"]["READ_A"]["table_DAGOACC_minus_P10"]
        mine = res["tabs"]["CTRL_P10"][s]
        leg2[f"s{s}"] = {"n_rows": len(LABELS), "max_abs_delta_lo_hi": max_abs_delta(mine, bt), "separated_flags_equal": all(bool(mine[l]["separated"]) == bool(bt[l]["separated"]) for l in LABELS if l in bt)}
    out["DAG_OACC_minus_P10_vs_DAGOACC_READ"] = leg2
    ctl.add("K-RC:DAGOACC_READ", all(v["max_abs_delta_lo_hi"] == 0.0 and v["separated_flags_equal"] for v in leg2.values()), leg2)
    # 6. the slow-draw method on the banked T0 evals, in the banked rng order
    rng = np.random.default_rng(0)
    cells, labels = {}, {}
    for s in SEEDS:
        for role in ("REP", "A"):
            c = H.speed(role, "CTRL", s, rng)
            cells[f"{role}_s{s}-CTRL_s{s}"] = c
            labels[f"{role}_s{s}"] = slow_draw(c)["label"]
    sc_diff = []
    for n in SPEED_CELLS:
        for k, v in cells.items():
            if not all(same_value(float(x), float(y)) for x, y in zip(v[n], sc_banked["cells"][n][k])):
                sc_diff.append([n, k, list(v[n]), sc_banked["cells"][n][k]])
    out["slow_draw_method_vs_rep_s0_speedcells_json"] = {"n_cells_compared": len(cells) * len(SPEED_CELLS), "n_differences": len(sc_diff), "differences": sc_diff[:6], "labels": labels}
    ctl.add("K-RC:slow-draw-method", not sc_diff, {"n_differences": len(sc_diff)})
    ctl.add("SLOW-DRAW:REP_s0-alone", labels == {"REP_s0": "SLOW-T", "A_s0": None, "REP_s1": None, "A_s1": None}, labels)
    # 7. the structural zeros and the REP licence, from the controls
    out["K_ZERO"] = {k: v["PASS"] for k, v in ctl.items.items() if k.startswith("K-ZERO")}
    out["K_REP_LICENCE"] = {k: {"PASS": v["PASS"], "detail": v["detail"]} for k, v in ctl.items.items() if k.startswith("K-REP-LICENCE")}
    out["controls_PASS"] = ctl.flags()
    out["n_controls"], out["failed_controls"] = len(ctl.items), {k: str(v["detail"])[:300] for k, v in ctl.failed().items()}
    out["wall_s"] = round(time.time() - t_start, 1)
    out["reader_sha256"] = sha256_file(os.path.abspath(__file__))
    out["pins"] = {os.path.basename(p): w for p, w in pins.items()}
    ok = not ctl.failed()
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("ZZLAUNCH-INTENT-READ-DRY", "OK" if ok else "NOT-OK", json.dumps({"n_controls": out["n_controls"], "failed": out["failed_controls"], "tables_differences": {k: v["n_differences"] for k, v in tabs_cmp.items()},
                                                                         "wall_s": out["wall_s"]}, default=str)[:1500], flush=True)
    return 0 if ok else 1


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--static-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--out", default="LAUNCH_INTENT_READ.json")
    ap.add_argument("--prereg", default="")
    ap.add_argument("--prereg-sha", default="")
    ap.add_argument("--orc-dir", default=ORC_DIR)
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.static_check:
        bad = prereg_problems(a.prereg, a.prereg_sha or None) if a.prereg else ["--prereg is required"]
        print(json.dumps({"problems": bad, "rf0_sha256": RF0_SHA256}, indent=1))
        print("LAUNCH-INTENT-READ-STATIC-OK" if not bad else "LAUNCH-INTENT-READ-STATIC-REFUSED")
        return 0 if not bad else 2
    if a.dry_run:
        return main_dry(a)
    if not a.manifest:
        raise SystemExit("--manifest is required (or --static-check / --dry-run)")
    return main_read(a)


if __name__ == "__main__":
    sys.exit(main())
