#!/usr/bin/env python3
"""lfo_read.py -- the T1-kin read of arm LFO075 (PREREG_LEVELFLOOR_A4.md, sha256 8bfbc386...): the LF075 floor with the level estimate v_hat replaced by the LOGGED route speed v_log (planner C0, lead cap from the LLR heads, kappa 0.75).
EVERY RESULT HERE CARRIES THE LABEL CEILING: the level input is the human's logged speed, a privileged channel that is NOT admissible at inference. No outcome of this read is a driving result, a capability claim or a counted
DOMINANT; it ranks levers and nothing else.

NOT a text clone of lf_read.py: this module LOADS lf_read.py BY SHA-PINNED PATH (which itself loads llr_read.py by pinned path) and calls its machinery unchanged: tick_instruments, reach_control, logged_speed, m1_ticks, the
pinned base arrays, and llr_read's rule_P / rule_G / rule_F / rule_D / family_table (the CL card's section 4 with CL := LFO). What is new here: the roles (RILFO075 reindexed, RILFO075C reindexed + kbox; no logged scene), the
A4 outcome labels (CEILING-CLEARS / CEILING-HARMS / CEILING-SHORT / F-UNREACHED), the attribution RILFO - RILF075 on all 19 rows (reported, never deciding), the composite RILFOC - P10 read A reported as the CEILING of D (never
counted), the oracle instruments, and the tool-versus-card tick controls for the logged-speed input.

Roles (tag = name + "_s{s}"; planner = C0_s{s}, presence / gap heads = LLR_s{s}; flags = the banked RIC0 / RIC0C flags + the floor flags + `--floor-vhat-source logged`):
  NEW (this package's 4 rollouts, F:/TanitAD-scratch/motion_c/lfo_read/out)   RILFO{075} reindexed | RILFO075C reindexed + kbox
  REUSED (read only, sha256-pinned)   C0, C0C, C0REP (recover_next/read/raw/arrays)   RIC0, RIC0C (motion_c/ri_read/raw/arrays)   RILF075, RILF075C (motion_c/lf_read/raw/arrays: the heads-source arms, the attribution base). P10 is the b_ part of every array.
Rules (deciding scene `reindexed`; the A3 rules with LF := LFO, section 4 of the A4 card):
  P_O: (RIC0 - RILFO)[M1] / (RIC0 - P10)[M1] >= 0.5 at both seeds, with RILFO - RIC0 on M1 separated-better at both.      G_O: on RILFO - RIC0, ADE6 point <= +0.0604 m and no non-crawl row of the 19 separated-worse, at both seeds.
  F_O: |RILFO - RIC0|[M1] > |C0REP - C0|[M1] at each seed (reported beside the label).   Attribution RILFO - RILF075 on all 19 rows (reported).   The composite RILFOC - P10 read A: the CEILING of D, never counted.
Outcome (A4 section 5), exactly one: CEILING-CLEARS (P_O and G_O at both seeds) | CEILING-HARMS (P_O holds, G_O fails) | CEILING-SHORT (P_O fails at either seed) | F-UNREACHED (reach control first: F binds on < 1 % of the M1 ticks at either seed).
Tier T1-kin. Estimator: paired episode-cluster bootstrap B = 2000, seed 0 (gapdecomp.decompose) over the 141 eval episodes / 2,399 windows: answers another draw of EPISODES only; the planner, F and the oracle input are deterministic.

usage: python -I lfo_read.py --mode production [--out-root F:/TanitAD-scratch/motion_c/lfo_read/out] [--workers 4]   (controls first; a failed control REFUSES the read, exit 3)
       python -I lfo_read.py --mode ka-lf | ka-null                                                               (known answers on EXISTING arrays, no rollout; raw/KA_READER_<mode>.json)
         ka-lf:   the banked RILF075 / RILF075C arrays fed in the LFO slot reproduce LF075's P / G / F numbers and rows exactly and read attribution 0.0 on every row.
         ka-null: the banked RIC0 / RIC0C arrays fed in the LFO slot read share 0.0 and every contrast row exactly 0.0."""
import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import re
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
MC = f"{ART}/dagland/motion_c"
PKG = f"{MC}/lfo_read"                                                  # the package (fixed path: a mutated copy of this file still finds its siblings)
LF_PK = f"{MC}/lf_read"
LF_READ_PY = f"{LF_PK}/code/lf_read.py"
LF_READ_SHA = "862e4e6c357bc084c971eceebd3ead1576d2218b1296eb1b10799f76710050e8"
LF_READ_JSON = f"{LF_PK}/raw/LF_READ.json"
LF_READ_JSON_SHA = "8d7850bfa65424ff1e385bab50c5551c0021cec3b413d9d0964cc8261d06b894"      # the LF read's result: the reference of the ka-lf known answer
LF_ARR = f"{LF_PK}/raw/arrays"
LF_OUT = "F:/TanitAD-scratch/motion_c/lf_read/out"                      # the LF rollouts' tick dumps (instruments beside)
CARD = f"{MC}/a4_prereg/PREREG_LEVELFLOOR_A4.md"
CARD_FROZEN = f"{MC}/a4_prereg/FROZEN_SHA.txt"
CARD_SHA = "8bfbc386083ce9b78f4938f3c44cb8772cbb17f3e6beb2ee96a018fcb7066721"
A3_CARD = f"{MC}/a3_prereg/PREREG_LEVELFLOOR_A3.md"
A3_CARD_SHA = "977a785e54df0b7e13bb8f81639d88eb52bb64092e5e1657e70b6e09fde2d405"
LFO_BUILD_PINS = os.environ.get("LFO_BUILD_PINS", f"{MC}/lfo_build/PINS.txt")      # key `patched_tool`: the LFO tool's sha256
RDB = f"{ART}/v8f-recover"
OUT_ROOT = "F:/TanitAD-scratch/motion_c/lfo_read/out"
WORK_DIR = "F:/TanitAD-scratch/motion_c/lfo_read/work"
SEEDS = (0, 1)
KAPPA = 0.75
KAPPA_FLAG = "0.75"
BASE_SMOOTH = "sg72"
LF_ARR_SHA = {"RILF075_s0": "cc00f4069ed0f0829bbf0315f06fc10b0623184ae6a59f88f42e970da991dc7a", "RILF075_s1": "15cfef21b5a590667f7b27d45966c0e214e8a69c64fb4aa9a6d0b2b862360659",
              "RILF075C_s0": "1c3da3c9b1e2d5c860ed28792d3d1a719f9981882a3842a3bec0f02d3f78d17f", "RILF075C_s1": "d2783f310abbbe8e2fa9842073238a22241360b9c2ab0db64b83926109fb7f49"}
GENERIC = {"RILFO": ("reindexed", "off"), "RILFOC": ("reindexed", "kbox")}
PRIORITY = ("RILFO", "RILFOC")
LABELS = ("CEILING-CLEARS", "CEILING-HARMS", "CEILING-SHORT", "F-UNREACHED")
M1 = "M1 crawl share"
CEILING_NOTE = ("CEILING: the level input is the human's logged route speed, a privileged channel that is NOT admissible at inference. This is not a driving result, not a capability claim and not a counted DOMINANT; "
                "it ranks levers (A4 card).")
LR = rr = SR = GD = LFR = PCO = None
DT, BAND_LO, REACH_BAR = 0.2, 3.0, 0.01


# ------------------------------------------------------------------------------------------------ loading
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def init(lf_read_path=None):
    """Load lf_read.py BY FILE PATH (sha256 checked first; ``lf_read_path`` only for the tests' own copies), run its init() (which loads llr_read.py by pinned path and the RECOVER-NEXT reader), load lfo_precheck.py by path."""
    global LFR, LR, rr, SR, GD, PCO
    path = lf_read_path or LF_READ_PY
    got = sha256_file(path)
    if path == LF_READ_PY and got != LF_READ_SHA:
        raise SystemExit(f"REFUSED: {path} is {got}, not the pinned {LF_READ_SHA}")
    LFR = _load("lf_read_for_lfo", path)
    LFR.init()
    LR, rr, SR, GD = LFR.LR, LFR.rr, LFR.SR, LFR.GD
    PCO = _load("lfo_precheck_for_read", f"{PKG}/code/lfo_precheck.py")
    assert LR.M1 == M1 and tuple(LR.SEEDS) == SEEDS
    return LR


def name_of(role):
    """The rollout / array name of a role: RILFO -> RILFO075, RILFOC -> RILFO075C."""
    return "RILFO075" + ("C" if role.endswith("C") else "")


def rollouts():
    """The 4 rollouts in priority order: [(tag, role, seed, scene, feasible)] (bare arm first, then the composite; seed 0 then seed 1)."""
    return [(f"{name_of(r)}_s{s}", r, s, GENERIC[r][0], GENERIC[r][1]) for r in PRIORITY for s in SEEDS]


def floor_ckpt(s):
    return f"{RDB}/M2b_s{s}_recft_LLR/ckpt_last.pt"


def expected_flags(role, s):
    scene, feas = GENERIC[role]
    return (f"--smooth {BASE_SMOOTH} --scene {scene}" + (f" --feasible {feas}" if feas != "off" else "") +
            f" --level-floor on --floor-kappa {KAPPA_FLAG} --floor-ckpt {floor_ckpt(s)} --floor-vhat-source logged")


# ------------------------------------------------------------------------------------------------ pure layer 1: tick instruments
def oracle_block(t, kappa=KAPPA):
    """The instruments of the logged-speed input from one LFO tick dump (``t`` = dict of arrays with the LF fields, ``lf_vhat_heads``, ``gt``, ``gt_ok``, ``v0``):
    the share of ticks with no complete logged route (no floor there), that the dumped input equals the reader's own logged speed, v_log - v_f against (1 - kappa) * v_log (a CHECK by construction), and the heads' error on this arm's own frames."""
    v_log, ok = LFR.logged_speed(t["gt"], t["gt_ok"])
    vh, vf, cap = np.asarray(t["lf_vhat"], np.float64), np.asarray(t["lf_vf"], np.float64), np.asarray(t["lf_cap"], bool)
    n = len(vh)
    band = ok & np.isfinite(v_log) & (v_log >= BAND_LO)
    unc = band & ~cap & np.isfinite(vf)
    both = np.isfinite(vh) & np.isfinite(v_log)
    out = {"n_ticks": int(n), "n_no_route_ticks": int((~ok).sum()), "share_no_route_ticks": float((~ok).mean()) if n else None, "n_band_ticks": int(band.sum()),
           "lf_vhat_equals_logged_speed_max_abs_mps": None if not both.any() else float(np.abs(vh[both] - v_log[both]).max()),
           "lf_vhat_nan_exactly_where_no_route": bool(np.array_equal(np.isnan(vh), ~ok)),
           "shortfall_check_max_abs_mps": None if not unc.any() else float(np.abs((v_log[unc] - vf[unc]) - (1.0 - kappa) * v_log[unc]).max()),
           "mean_shortfall_vlog_minus_vf_uncapped_mps": None if not unc.any() else float((v_log[unc] - vf[unc]).mean())}
    if "lf_vhat_heads" in t:
        vhh = np.asarray(t["lf_vhat_heads"], np.float64)
        sel = np.isfinite(vhh) & np.isfinite(v_log)
        e = vhh[sel] - v_log[sel]
        out["heads_estimate_on_this_arms_frames"] = {"n": int(sel.sum()), "mae_mps": None if not sel.any() else float(np.abs(e).mean()), "share_heads_exceed_logged": None if not sel.any() else float((e > 0).mean()),
                                                      "note": "beside, never deciding: what the floor heads would have said on the oracle arm's own frames"}
    return out


def tick_instruments(t):
    """The A3 instruments (LFR.tick_instruments: bind overall and inside the M1 ticks, cap, lift where bound incl. the negative chord lift of A3_RULING_2, v_f against the logged speed) plus the oracle block."""
    r = LFR.tick_instruments(t)
    r["oracle"] = oracle_block(t)
    r["v_f_vs_logged_note"] = "here v_f = kappa * v_log wherever the lead cap did not bind, so v_log - v_f reads (1 - kappa) * v_log by construction: a check, not a result (A4 section 3)"
    return r


def tick_controls(t, kappa=KAPPA):
    """Tool-versus-card consistency of one LFO tick dump (A4 card sections 2-3): a violation means the tool does not implement the frozen operator with the logged-speed input."""
    need = LFR.LF_FIELDS + ("v0", "gt", "gt_ok", "lf_vhat_heads")
    if any(k not in t for k in need):
        return {"fields_present": False}
    f = lambda k: np.asarray(t[k], np.float64)
    bind, cap = np.asarray(t["lf_bind"], bool), np.asarray(t["lf_cap"], bool)
    vf, vh, pi, lift = f("lf_vf"), f("lf_vhat"), f("lf_pi"), f("lf_lift")
    v_log, ok = LFR.logged_speed(t["gt"], t["gt_ok"])
    fin = np.isfinite(vf) & np.isfinite(vh) & (vh >= BAND_LO)
    lo = np.isfinite(vh) & (vh < BAND_LO)
    nr = ~np.isfinite(vh)
    both = np.isfinite(vh) & np.isfinite(v_log)
    return {"fields_present": True, "n_ticks_is_3x2399": bool(len(vf) == LFR.N_TICKS),
            "v_hat_is_the_logged_speed_of_the_dumped_gt": bool(both.any() and np.abs(vh[both] - v_log[both]).max() <= 1e-5 and np.array_equal(np.isnan(vh), ~ok)),
            "no_floor_where_no_complete_route": bool(np.isnan(vf[nr]).all() and not bind[nr].any()),
            "pi_finite_every_tick": bool(np.isfinite(pi).all()), "pi_in_unit_interval": bool(((pi >= 0) & (pi <= 1)).all()),
            "no_bind_below_the_3_mps_band": bool(not (bind & lo).any()),
            "v_f_between_0_and_kappa_v_hat": bool(((vf[fin] >= -1e-6) & (vf[fin] <= kappa * vh[fin] + 1e-3)).all()),
            "v_f_is_kappa_v_log_where_the_cap_did_not_bind": bool((np.abs(vf[fin & ~cap] - kappa * vh[fin & ~cap]) <= 1e-3).all()),
            "cap_only_where_presence_at_least_half": bool((pi[cap] >= 0.5 - 1e-6).all()),
            "lift_zero_where_unbound": bool((np.abs(np.nan_to_num(lift[~bind])) <= 1e-6).all())}      # A3_RULING_2: a negative CHORD lift where F binds is a reported result, not a control


# ------------------------------------------------------------------------------------------------ pure layer 2: the A4 card's outcomes (section 5)
def failing_rows(G):
    """The rows that name the harm when G_O fails: G's separated-worse non-crawl rows (and ADE6 when its point is over the guard), over both seeds."""
    rows = set()
    for p in G["per_seed"].values():
        rows |= set(p["non_crawl_separated_worse"])
        if not p["ade6_within_guard"]:
            rows.add("ADE6")
    return sorted(rows)


def decide(reach, P=None, G=None, m1_moved=False):
    """A4 card section 5. Precedence: F-UNREACHED (reach control first, no rule read) > CEILING-SHORT (P_O fails at either seed) > CEILING-HARMS (P_O holds, G_O fails) > CEILING-CLEARS (P_O and G_O hold at both seeds).
    ``reach`` None = not evaluated (known-answer runs only). Returns {label, next, estimator_levers_signalled, rows, notes}."""
    if reach is not None and not reach["PASS"]:
        return {"label": "F-UNREACHED", "next": "the floor does not bind where the crawl is even with an exact level: the instrument is checked first; no rule is read (A4 card section 5)", "estimator_levers_signalled": False, "rows": [], "notes": []}
    notes = [f"M1 moved (RILFO - RIC0 separated-better at >= 1 seed): {bool(m1_moved)}"]
    if not P["PASS"]:
        return {"label": "CEILING-SHORT", "estimator_levers_signalled": False, "rows": [], "notes": notes,
                "next": "an exact level at kappa 0.75 does not close half the crawl, so a floor on progress is not the form that closes M1 at this dose; the estimator levers are NOT signalled for this form; "
                        "what remains is reported to the PI: T2 re-perception (option (c)), or the operator at kappa 1 in its own addendum"}
    if not G["PASS"]:
        rows = failing_rows(G)
        return {"label": "CEILING-HARMS", "estimator_levers_signalled": False, "rows": rows, "notes": notes,
                "next": "even an exact level harms, so the harm is the operator's and no estimator removes it; the failing rows name it (" + (", ".join(rows) or "none") + "); the estimator levers are NOT signalled; "
                        "the next arm changes F, not its input: an operator variant in its own addendum, written after the failing rows are read and before its rollouts"}
    return {"label": "CEILING-CLEARS", "estimator_levers_signalled": True, "rows": [], "notes": notes,
            "next": "the operator works given a good level, so the estimator is the MEASURED bottleneck and its value is the ceiling read here; this goes to the PI as the evidence for the estimator levers (the A40 motion-aware encoder, "
                    "or a stronger estimator on more data); the next CPU arm is a degradation sweep (addendum A5): the oracle input with a priori multiplicative noise, which sets the estimator accuracy at which P and G stop holding"}


# ------------------------------------------------------------------------------------------------ the read (arrays)
def _rows(ar, eid, cls, x, y, n_boot, full=None):
    return rr.rows_of(rr.Aof(ar[x]), rr.Aof(ar[y]), eid, cls, n_boot, full)


def _vs_p10(ar, eid, cls, x, n_boot):
    return rr.rows_of(rr.Aof(ar[x]), rr.Bof(ar[x]), eid, cls, n_boot)[0]


def read_ceiling(arrays, n_boot=2000, reach=None, cache=None, log=print):
    """``arrays`` = {seed: {role: array dict}} with RILFO RILFOC (this arm's rollouts), RILF RILFC (the banked LF075 arms = the attribution base) and C0 C0C C0REP RIC0 RIC0C. ``reach`` = reach_control() of RILFO075 (None only in a
    known-answer run). -> {kappa, label_set, reach, seeds, rules, outcome, m1_moved, ceiling}. A failed reach control returns WITHOUT reading a rule."""
    res = {"kappa": KAPPA, "reach": reach, "ceiling": CEILING_NOTE}
    if reach is not None and not reach["PASS"]:
        res.update({"seeds": None, "rules": None, "m1_moved": False, "outcome": decide(reach)})
        return res
    cache = {} if cache is None else cache
    res["seeds"] = {}
    cells, xy, rows_ri, ra_c, eff, rep = {}, {}, {}, {}, {}, {}
    for s in SEEDS:
        k, ar = f"s{s}", arrays[s]
        eid = ar["C0"]["eid"]
        for role, z in ar.items():
            assert np.array_equal(z["eid"], eid), f"s{s} {role}: windows differ from C0's"
        cls, _ = GD.classify(ar["C0"]["v0"], ar["C0"]["g"])

        def base(key, fn):
            if (s, key) not in cache:
                cache[(s, key)] = fn()
            return cache[(s, key)]
        b = {}
        b["RILFO_minus_RIC0"], b["M1_per_class_split_RILFO_minus_RIC0"] = _rows(ar, eid, cls, "RILFO", "RIC0", n_boot, M1)
        b["RIC0_minus_P10"] = base("RIC0_P10", lambda: _vs_p10(ar, eid, cls, "RIC0", n_boot))
        b["RILFO_minus_P10"], b["RILFOC_minus_P10"] = _vs_p10(ar, eid, cls, "RILFO", n_boot), _vs_p10(ar, eid, cls, "RILFOC", n_boot)
        b["RIC0C_minus_P10"] = base("RIC0C_P10", lambda: _vs_p10(ar, eid, cls, "RIC0C", n_boot))
        b["RILFOC_minus_RIC0C"] = _rows(ar, eid, cls, "RILFOC", "RIC0C", n_boot)[0]
        b["RILFO_minus_RILF075"] = _rows(ar, eid, cls, "RILFO", "RILF", n_boot)[0]                    # ATTRIBUTION: the level estimate's error at a fixed operator, dose, planner and lead cap
        b["RILFOC_minus_RILF075C"] = _rows(ar, eid, cls, "RILFOC", "RILFC", n_boot)[0]                # beside: the same on the composite
        b["RILF075_minus_RIC0_beside"] = _rows(ar, eid, cls, "RILF", "RIC0", n_boot)[0]               # beside: the heads-source arm's own contrast (the LF read's), same rows
        b["C0REP_minus_C0"] = base("C0REP_C0", lambda: _rows(ar, eid, cls, "C0REP", "C0", n_boot)[0])
        cells[k] = rr.m1_cell(b["RILFO_minus_RIC0"][M1], b["RIC0_minus_P10"][M1])
        xy[k], rows_ri[k] = SR.read_xy(GD, b["RILFO_minus_RIC0"]), b["RILFO_minus_RIC0"]
        ra_c[k] = rr.read_a_block(SR, GD, b["RILFOC_minus_P10"])
        eff[k], rep[k] = b["RILFO_minus_RIC0"][M1]["delta"], b["C0REP_minus_C0"][M1]["delta"]
        b["read_a_RILFOC_minus_P10"] = ra_c[k]
        b["read_a_RIC0C_base_beside"] = rr.read_a_block(SR, GD, b["RIC0C_minus_P10"])
        b["read_xy_RILFO_minus_RIC0"] = xy[k]
        b["m1_cell_reindexed"] = cells[k]
        b["family_tables"] = {c: LR.family_table(b[c]) for c in ("RILFO_minus_RIC0", "RILFOC_minus_P10", "RILFO_minus_RILF075")}
        b["attribution_rows_changed_vs_RILF075"] = {"RILFO": [lab for lab, r in b["RILFO_minus_RILF075"].items() if r["delta"] != 0.0], "RILFO_separated": [lab for lab, r in b["RILFO_minus_RILF075"].items() if r["separated"]]}
        b["floor_max_abs_dev_vs_C0_m"] = {role: float(max(np.nanmax(np.abs(np.asarray(u, float) - np.asarray(v, float))) for u, v in zip(rr.Bof(ar[role]), rr.Bof(ar["C0"])))) for role in ar if role != "C0"}
        res["seeds"][k] = b
        log(k, "M1 RILFO-RIC0 %+.4f [%+.4f, %+.4f] sep %s | share %s | ADE6 RILFO-RIC0 %+.4f | RILFO-RILF075 M1 %+.4f | RILFOC read A (ceiling of D) %s" % (
            b["RILFO_minus_RIC0"][M1]["delta"], b["RILFO_minus_RIC0"][M1]["lo"], b["RILFO_minus_RIC0"][M1]["hi"], b["RILFO_minus_RIC0"][M1]["separated"], cells[k]["share_closed"],
            b["RILFO_minus_RIC0"]["ADE6"]["delta"], b["RILFO_minus_RILF075"][M1]["delta"], ra_c[k]["DOMINANT"]))
    P, G = LR.rule_P(cells), LR.rule_G(xy, rows_ri)
    F = LR.rule_F(eff, rep)
    Dr = LR.rule_D(ra_c, P["PASS"], G["PASS"])
    ceiling_D = {"ceiling_of_D_read_a_both_seeds": Dr["read_a_both_seeds"], "ceiling_of_D_with_P_O_and_G_O": Dr["DOMINANT_COUNTED"], "three_family_form_reported_beside": Dr["three_family_form_reported_beside"], "by_seed": Dr["by_seed"],
                 "counted": False, "note": "the CEILING of D under an exact level (A4 section 4): reported, NEVER counted as a DOMINANT"}
    moved = bool(any(c["better"] for c in cells.values()))
    out = decide(reach, P, G, moved)
    if out["label"] == "CEILING-CLEARS" and not F["PASS"]:
        out["notes"].append("F_O fails: |RILFO - RIC0| on M1 does not exceed the C0REP replicate floor at every seed (reported beside the label; the training-variance reading is the Master Mind's)")
    res.update({"rules": {"P_O": P, "G_O": G, "F_O": F, "ceiling_of_D": ceiling_D, "deciding_scene": "reindexed", "m1_moved_at_any_seed": moved,
                          "m1_moved_definition": "RILFO - RIC0 on M1 separated-better at >= 1 seed (A3_RULING_1 item 3)"}, "m1_moved": moved, "outcome": out})
    return res


# ------------------------------------------------------------------------------------------------ files and controls
def reused_arrays(s):
    """The pinned base arrays (LFR.reused_arrays: C0 C0C C0REP RIC0 RIC0C) plus the heads-source arrays RILF075 / RILF075C as roles RILF / RILFC."""
    z, ctl = LFR.reused_arrays(s)
    for role, tag in (("RILF", f"RILF075_s{s}"), ("RILFC", f"RILF075C_s{s}")):
        p = f"{LF_ARR}/{tag}.npz"
        got = sha256_file(p)
        ctl[tag] = {"path": p, "sha256": got, "pinned": LF_ARR_SHA[tag], "PASS": got == LF_ARR_SHA[tag]}
        z[role] = rr.read_arrays(p)
    return z, ctl


def parse_pins(path=None):
    return LFR.parse_pins(path or LFO_BUILD_PINS)


def frozen_card_sha():
    for line in open(CARD_FROZEN, encoding="utf-8"):
        t = line.split()
        if len(t) >= 2 and t[0] == os.path.basename(CARD):
            return t[1]
    return None


def run_controls(out_root, raw, pins_path=None):
    """Production controls before any row: per rollout the done marker (flags exactly the banked base's + the floor flags + `--floor-vhat-source logged`, tool sha256 = lfo_build `patched_tool`, planner checkpoint = C0's), the run header
    (scene, feasible, smooth, 2399 windows, yaw, CEILING labelling), the floor keys of the output JSON (lfo_precheck.check_rollout), the tick dump's fields and their consistency with the card; one planner checkpoint per seed; the
    pre-check file; the reused arrays; the cards and the pinned readers."""
    ctl, ckpts = {}, {}
    try:
        tool_pin = parse_pins(pins_path).get("patched_tool")
    except Exception:
        tool_pin = None
    ctl["patched_tool_pin"] = {"PASS": bool(tool_pin and re.fullmatch(r"[0-9a-f]{64}", tool_pin)), "patched_tool": tool_pin, "pins_file": pins_path or LFO_BUILD_PINS}
    for tag, role, s, scene, feas in rollouts():
        c = {"PASS": False}
        try:
            d = json.load(open(f"{out_root}/done_{tag}.txt", encoding="utf-8"))
            h = rr.run_header(out_root, tag)
            fe = (h.get("feasible") or {}).get("mode", "off") if isinstance(h.get("feasible"), dict) else ("off" if not h.get("feasible") else str(h.get("feasible")))
            sm = h["smooth"]["mode"] if isinstance(h.get("smooth"), dict) else h.get("smooth")
            fc, fdet = PCO.check_rollout(out_root, tag)
            t = np.load(f"{out_root}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True)
            tc = tick_controls({k: t[k] for k in t.files})
            c = {"flags_exactly_the_banked_base_plus_floor_plus_logged_source": d["flags"] == expected_flags(role, s), "tool_sha256_is_patched_tool": bool(tool_pin and d["tool_sha256"] == tool_pin),
                 "ckpt_is_C0": d["ckpt"].replace("\\", "/").endswith(f"/M2b_s{s}_recft_C0/ckpt_last.pt"), "header_scene": h["scene"] == scene, "header_feasible": fe == feas, "header_smooth": sm == BASE_SMOOTH,
                 "header_n_windows_2399": int(h["n_windows"]) == 2399, "header_kin_yaw": h["kin_mode"] == "yaw", "ckpt_sha256": d["ckpt_sha256"], "threads": d.get("threads")}
            c.update({f"floor_{k}": v for k, v in fc.items()})
            c.update({f"ticks_{k}": v for k, v in tc.items()})
            ckpts.setdefault(s, set()).add(d["ckpt_sha256"])
            c["PASS"] = all(v for k, v in c.items() if isinstance(v, bool))
        except Exception as e:      # a missing marker / header / tick file is a failed control, named
            c["error"] = f"{type(e).__name__}: {e}"
        ctl[f"rollout_{tag}"] = c
    ctl["one_checkpoint_per_seed"] = {"PASS": bool(len(ckpts) == len(SEEDS) and all(len(v) == 1 and next(iter(v)) == PCO.C0_SHA[s] for s, v in ckpts.items())), "ckpt_sha256": {f"s{s}": sorted(v) for s, v in ckpts.items()}}
    try:
        pc = json.load(open(f"{raw}/LFO_PRECHECK.json", encoding="utf-8"))
        ctl["LFO_PRECHECK"] = {"PASS": bool(pc["PASS"] is True and all(pc["seeds"][f"s{s}"]["C0"]["ckpt_sha256"] == PCO.C0_SHA[s] and pc["seeds"][f"s{s}"]["LLR"]["ckpt_sha256"] == PCO.llr_sha_of(s) for s in SEEDS))}
    except Exception as e:
        ctl["LFO_PRECHECK"] = {"PASS": False, "error": f"{type(e).__name__}: {e}"}
    got, fz = sha256_file(CARD), frozen_card_sha()
    ctl["card_a4"] = {"PASS": bool(got == CARD_SHA == fz), "sha256": got, "pinned": CARD_SHA, "frozen_sha_file": fz}
    got = sha256_file(A3_CARD)
    ctl["card_a3"] = {"PASS": bool(got == A3_CARD_SHA), "sha256": got, "pinned": A3_CARD_SHA}
    got = LR.card_rules_sha()
    ctl["cl_card_rules"] = {"PASS": got == LR.CARD_RULES_SHA, "sha256": got, "pinned": LR.CARD_RULES_SHA}
    ctl["llr_read_py"] = {"PASS": sha256_file(LR.__file__.replace("\\", "/")) == LFR.LLR_READ_SHA, "pinned": LFR.LLR_READ_SHA}
    ctl["lf_read_py"] = {"PASS": sha256_file(LFR.__file__.replace("\\", "/")) == LF_READ_SHA, "pinned": LF_READ_SHA}
    return ctl


def extract_new(out_root, arr_dir, workers):
    jobs = [(tag, s) for tag, _r, s, _sc, _f in rollouts()]
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(rr.extract, tag, s, out_root, arr_dir): tag for tag, s in jobs}
        for f in cf.as_completed(futs):
            f.result()
    return {tag: f"{arr_dir}/{tag}.npz" for tag, _ in jobs}


def ka_arrays(mode, s):
    """Known-answer arrays: the roles RILFO RILFOC filled from EXISTING arrays, RILF RILFC being the banked LF075 arrays in both modes.
    ka-lf: the banked RILF075 / RILF075C arrays IN THE LFO SLOT (the arm IS the heads-source arm). ka-null: the arm IS its base (RIC0, RIC0C)."""
    z, ctl = reused_arrays(s)
    z["RILFO"], z["RILFOC"] = (z["RILF"], z["RILFC"]) if mode == "ka-lf" else (z["RIC0"], z["RIC0C"])
    return z, ctl


def _same_row(a, b):
    return all(float(a[q]) == float(b[q]) for q in ("delta", "lo", "hi")) and bool(a["separated"]) == bool(b["separated"])


def ka_check(mode, res):
    """Expected answers. ka-lf: from LF_READ.json dose 075 (every contrast row, the M1 cells, rules P / G / F and the read A of the composite equal), attribution 0.0 on every row, label CEILING-SHORT (LF075's P fails).
    ka-null: from first principles (every contrast row exactly 0.0, shares 0.0 or undefined, the composite's read A = the banked RIC0C's)."""
    chk = {}
    ref = json.load(open(LF_READ_JSON, encoding="utf-8"))["doses"]["075"]
    for s in SEEDS:
        k, b = f"s{s}", res["seeds"][f"s{s}"]
        rs = ref["seeds"][k]
        chk[f"{k}_attribution_RILFO_minus_RILF075_all_19_rows_exactly_zero_in_ka_lf"] = (mode != "ka-lf") or all(x["delta"] == 0.0 and x["lo"] == 0.0 and x["hi"] == 0.0 and not x["separated"] for x in b["RILFO_minus_RILF075"].values())
        chk[f"{k}_attribution_composite_all_19_rows_exactly_zero_in_ka_lf"] = (mode != "ka-lf") or all(x["delta"] == 0.0 and x["lo"] == 0.0 and x["hi"] == 0.0 and not x["separated"] for x in b["RILFOC_minus_RILF075C"].values())
        chk[f"{k}_19_rows_in_every_contrast"] = all(len(b[c]) == 19 for c in ("RILFO_minus_RIC0", "RILFO_minus_P10", "RILFOC_minus_P10", "RILFOC_minus_RIC0C", "RILFO_minus_RILF075", "RILFOC_minus_RILF075C"))
        if mode == "ka-lf":
            for mine, old in (("RILFO_minus_RIC0", "RILF_minus_RIC0"), ("RILFO_minus_P10", "RILF_minus_P10"), ("RILFOC_minus_P10", "RILFC_minus_P10"), ("RILFOC_minus_RIC0C", "RILFC_minus_RIC0C")):
                chk[f"{k}_{mine}_equals_LF_READ[{old}]"] = all(_same_row(b[mine][lab], rs[old][lab]) for lab in rs[old])
            chk[f"{k}_m1_cell_equals_LF_READ"] = b["m1_cell_reindexed"] == rs["m1_cell_reindexed"]
            chk[f"{k}_read_a_of_the_composite_equals_LF_READ"] = b["read_a_RILFOC_minus_P10"]["DOMINANT"] == rs["read_a_RILFC_minus_P10"]["DOMINANT"] and b["read_a_RILFOC_minus_P10"]["failing_terms"] == rs["read_a_RILFC_minus_P10"]["failing_terms"]
        else:
            for c in ("RILFO_minus_RIC0", "RILFOC_minus_RIC0C"):
                chk[f"{k}_{c}_all_rows_exactly_zero"] = all(x["delta"] == 0.0 and not x["separated"] for x in b[c].values())
            chk[f"{k}_share_zero_or_undefined"] = b["m1_cell_reindexed"]["share_closed"] in (0.0, None)
            chk[f"{k}_RILFOC_minus_P10_equals_RIC0C_minus_P10"] = all(_same_row(b["RILFOC_minus_P10"][lab], b["RIC0C_minus_P10"][lab]) for lab in b["RIC0C_minus_P10"])
    R = res["rules"]
    if mode == "ka-lf":
        Rr = ref["rules"]
        chk["rule_P_O_equals_LF_READ_P"] = R["P_O"]["PASS"] == Rr["P"]["PASS"] and all(R["P_O"]["per_seed"][k] == Rr["P"]["per_seed"][k] for k in ("s0", "s1"))
        chk["rule_G_O_equals_LF_READ_G"] = R["G_O"]["PASS"] == Rr["G"]["PASS"] and all(R["G_O"]["per_seed"][k] == Rr["G"]["per_seed"][k] for k in ("s0", "s1"))
        chk["rule_F_O_equals_LF_READ_F"] = R["F_O"]["PASS"] == Rr["F"]["PASS"] and all(R["F_O"]["per_seed"][k] == Rr["F"]["per_seed"][k] for k in ("s0", "s1"))
        chk["ceiling_of_D_equals_LF_READ_D"] = R["ceiling_of_D"]["ceiling_of_D_with_P_O_and_G_O"] == Rr["D"]["DOMINANT_COUNTED"] and all(R["ceiling_of_D"]["by_seed"][k] == Rr["D"]["by_seed"][k] for k in ("s0", "s1"))
        chk["label_is_CEILING_SHORT_because_LF075_P_fails"] = res["outcome"]["label"] == "CEILING-SHORT" and (not Rr["P"]["PASS"])
    else:
        chk["P_fails_F_fails_G_holds"] = bool((not R["P_O"]["PASS"]) and (not R["F_O"]["PASS"]) and R["G_O"]["PASS"])
        chk["label_is_CEILING_SHORT"] = res["outcome"]["label"] == "CEILING-SHORT"
    chk["ceiling_of_D_is_never_counted"] = R["ceiling_of_D"]["counted"] is False
    return chk


# ------------------------------------------------------------------------------------------------ RESULTS.md
def fmt(r):
    return "%+.4f [%+.4f, %+.4f]%s" % (r["delta"], r["lo"], r["hi"], " *" if r["separated"] else "")


def render_md(res):
    r = res["arm"]
    banner = [] if res.get("mode", "production") == "production" else [f"**KNOWN-ANSWER RUN `{res['mode']}` on EXISTING arrays (no LFO rollout): a test of this reader, not a read of arm LFO075.**", ""]
    sh = lambda c: "None" if c is None else "%.3f" % (float(c) + 0.0)
    L = banner + [f"# LFO read: arm LFO075 (LF075 with the level estimate replaced by the LOGGED speed), CEILING, T1-kin, generated {res.get('clock', '')}", "",
                  f"**Outcome: {r['outcome']['label']} (CEILING diagnostic).**", "", f"**Headline: {res['headline']}**", ""]
    L += [f"Tier: {res.get('tier', '')}. Estimator: {res.get('estimator', '')}. `*` = interval excludes 0. STRATEGIC: {res['strategic_family']}", "", "`logged` is not read (A4 section 3: the level input is the human's speed by construction, so the LEAK-BORNE rule has no question left; disclosed). `hold_now` is not read.", ""]
    L += [f"Next: {r['outcome']['next']}.", "" if not r["outcome"]["notes"] else "Notes: " + "; ".join(r["outcome"]["notes"]) + ".", ""]
    if r["reach"] is not None:
        L += ["Reach control (F binds on >= 1 % of the replans inside the M1 crawl ticks, each seed): " + "; ".join(f"{k} {sh(v['bind_rate_in_m1_ticks'])} of {v['n_m1_ticks']} ticks" for k, v in r["reach"]["per_seed"].items()) + f" -> **{r['reach']['PASS']}**", ""]
    if r["rules"] is None:
        L += ["No rule is read (F-UNREACHED).", ""]
    else:
        R = r["rules"]
        P, G, F, D = R["P_O"], R["G_O"], R["F_O"], R["ceiling_of_D"]
        L += ["| rule | s0 | s1 | holds |", "|---|---|---|---|",
              "| P_O: M1 separated-better and share >= 0.5 | " + " | ".join(f"sep {P['per_seed'][k]['separated_better']}, share {sh(P['per_seed'][k]['share_closed'])}" for k in ("s0", "s1")) + f" | **{P['PASS']}** |",
              "| G_O: ADE6 <= +0.0604 m, no non-crawl row separated-worse | " + " | ".join(f"ADE6 {G['per_seed'][k]['ade6_delta']:+.4f}, worse: {', '.join(G['per_seed'][k]['non_crawl_separated_worse']) or 'none'}" for k in ("s0", "s1")) + f" | **{G['PASS']}** |",
              "| F_O (beside the label): abs(RILFO-RIC0)[M1] > abs(C0REP-C0)[M1] | " + " | ".join(f"{F['per_seed'][k]['abs_R_minus_C0']:.4f} vs {F['per_seed'][k]['abs_C0rep_minus_C0']:.4f}" for k in ("s0", "s1")) + f" | **{F['PASS']}** |",
              "| CEILING of D (never counted): RILFOC - P10 read A (D1, D2, D3 four families) | " + " | ".join(f"D1 {D['by_seed'][k]['D1']}, D2 {D['by_seed'][k]['D2']}, D3 {D['by_seed'][k]['D3']}" for k in ("s0", "s1")) + f" | read A both seeds {D['ceiling_of_D_read_a_both_seeds']} (counted: False) |",
              "", f"Beside: three-family form of the ceiling of D {D['three_family_form_reported_beside']}; M1 moved (RILFO - RIC0 separated-better at >= 1 seed) {R['m1_moved_at_any_seed']}.", "",
              "Ceiling-of-D failing terms: " + "; ".join(f"{k}: " + ("; ".join(D["by_seed"][k]["failing_terms"]) or "none") for k in ("s0", "s1")), "", "Failing rows (G_O, whatever the label): " + (", ".join(failing_rows(G)) or "none"), ""]
        for c, title in (("RILFO_minus_RIC0", "RILFO - RIC0 (reindexed, the deciding contrast)"), ("RILFO_minus_RILF075", "ATTRIBUTION: RILFO - RILF075 (the level estimate's error at a fixed operator, dose, planner and lead cap; reported, never deciding)"),
                         ("RILFOC_minus_P10", "RILFOC - P10 (reindexed composite, the CEILING of D, never counted)"), ("RILFOC_minus_RILF075C", "attribution on the composite: RILFOC - RILF075C (beside)"),
                         ("RILF075_minus_RIC0_beside", "beside: RILF075 - RIC0 (the heads-source arm's own contrast, the LF read's)")):
            L += ["", f"### 19 rows: {title}", "", "| row | family | s0 | s1 |", "|---|---|---|---|"]
            for lab in GD.LABELS:
                L.append(f"| {lab} | {SR.FAMILY[lab]} | " + " | ".join(fmt(r["seeds"][k][c][lab]) for k in ("s0", "s1")) + " |")
    if res.get("instruments"):
        L += ["", "## Instruments (from the tick dumps; beside, never deciding)", "", "| rollout | bind overall | bind in M1 ticks (n) | cap overall | cap in M1 ticks | mean lift where bound, m/s | negative chord lift: n, share, min m/s | no-route ticks | v_log - v_f: check vs (1-kappa) v_log, max abs m/s | heads' MAE on these frames, m/s (share heads > logged) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for tag, v in res["instruments"].items():
            lw, o = v["lift_where_bound_overall"], v["oracle"]
            hh = o.get("heads_estimate_on_this_arms_frames") or {}
            L.append(f"| {tag} | {sh(v['bind_rate_overall']['rate'])} | {sh(v['bind_rate_in_m1_ticks']['rate'])} ({v['bind_rate_in_m1_ticks']['n']}) | {sh(v['cap_rate_overall']['rate'])} | {sh(v['cap_rate_in_m1_ticks']['rate'])} | "
                     f"{'None' if lw['mean_lift_mps'] is None else '%.3f' % lw['mean_lift_mps']} | {lw['n_negative_chord_lift']}, {sh(lw['share_negative_chord_lift'])}, {'None' if lw['min_lift_mps'] is None else '%.3f' % lw['min_lift_mps']} | "
                     f"{o['n_no_route_ticks']} ({sh(o['share_no_route_ticks'])}) | {'None' if o['shortfall_check_max_abs_mps'] is None else '%.2e' % o['shortfall_check_max_abs_mps']} | "
                     f"{'None' if hh.get('mae_mps') is None else '%.3f' % hh['mae_mps']} ({sh(hh.get('share_heads_exceed_logged'))}) |")
        if res.get("instruments_lf075_beside"):
            L += ["", "Beside, the heads-source arm LF075 (the LF read's rollouts, same instruments): ", "", "| rollout | bind overall | bind in M1 ticks (n) | cap overall | mean lift where bound, m/s | v_f vs logged: MAE m/s, share v_f > logged (n) |", "|---|---|---|---|---|---|"]
            for tag, v in res["instruments_lf075_beside"].items():
                vf = v["vf_vs_logged_overall"]
                L.append(f"| {tag} | {sh(v['bind_rate_overall']['rate'])} | {sh(v['bind_rate_in_m1_ticks']['rate'])} ({v['bind_rate_in_m1_ticks']['n']}) | {sh(v['cap_rate_overall']['rate'])} | "
                         f"{'None' if v['lift_where_bound_overall']['mean_lift_mps'] is None else '%.3f' % v['lift_where_bound_overall']['mean_lift_mps']} | {'None' if vf['mae_mps'] is None else '%.3f' % vf['mae_mps']}, {sh(vf['share_vf_exceeds_logged'])} ({vf['n']}) |")
    L += ["", "## Controls", "", "| control | PASS |", "|---|---|"] + [f"| {k} | {v.get('PASS')} |" for k, v in res.get("controls", {}).items()]
    L += ["", "## Variance named", ""] + [f"* {k}: {v}" for k, v in res["variance_questions"].items()]
    return "\n".join(x for x in L if x is not None) + "\n"


def headline(arm):
    """One line: the label, the P_O shares and the G_O ADE6 deltas, and the CEILING statement."""
    o = arm["outcome"]
    if o["label"] == "F-UNREACHED":
        return "F-UNREACHED (CEILING diagnostic): the floor does not bind where the crawl is even with an exact level; the instrument is checked first and no rule is read. " + CEILING_NOTE
    R = arm["rules"]
    sh = [R["P_O"]["per_seed"][k]["share_closed"] for k in ("s0", "s1")]
    ad = [R["G_O"]["per_seed"][k]["ade6_delta"] for k in ("s0", "s1")]
    f3 = lambda x: "undefined" if x is None else "%.3f" % x
    return (f"{o['label']} (CEILING diagnostic): P_O {'holds' if R['P_O']['PASS'] else 'fails'} (share closed s0 {f3(sh[0])}, s1 {f3(sh[1])}; bar 0.5), "
            f"G_O {'holds' if R['G_O']['PASS'] else 'fails'} (ADE6 s0 {ad[0]:+.4f} m, s1 {ad[1]:+.4f} m; guard +0.0604 m). " + CEILING_NOTE)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("production", "ka-lf", "ka-null"), default="production")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--work-dir", default=WORK_DIR)
    ap.add_argument("--raw", default=f"{PK}/raw")
    ap.add_argument("--arr-dir", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--results-md", default=f"{PK}/RESULTS.md")
    a = ap.parse_args(argv)
    init()
    rr.PK = PK
    rr.WORK = f"{a.work_dir}/rig"
    os.makedirs(a.raw, exist_ok=True)
    arr_dir = a.arr_dir or f"{a.raw}/arrays"
    res = {"label": "CEILING", "card": "PREREG_LEVELFLOOR_A4.md (arm LFO075; the rules by reference to PREREG_LEVELFLOOR_A3.md section 4 and PREREG_LEVELHEAD_CL.md sections 4-5, LF := LFO)", "card_sha256": sha256_file(CARD),
           "reader": os.path.abspath(__file__).replace("\\", "/"), "reader_sha256": sha256_file(__file__), "lf_read_sha256": sha256_file(LF_READ_PY), "mode": a.mode,
           "tier": "T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2); CEILING: privileged level input",
           "estimator": f"paired episode-cluster bootstrap B={a.n_boot} seed 0 (gapdecomp.decompose); answers another draw of EPISODES only", "clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "strategic_family": rr.STRATEGIC_NOTE, "ceiling_note": CEILING_NOTE,
           "variance_questions": {"episodes": "answered by the paired episode-cluster bootstrap (every interval here)",
                                  "training_run": "O is C0's own checkpoint in every contrast with C0 / RIC0, so O's training variance is absent from it; the lead cap comes from one LLR run per seed (not measured); rule F_O reads against the C0REP floor",
                                  "inference": "the planner, F and the oracle input are deterministic: no inference seed applies (the oracle input carries no draw)"}}
    base, rctl, inst, ticks = {}, {}, {}, {}
    if a.mode == "production":
        ctl = run_controls(a.out_root, a.raw)
        for s in SEEDS:
            z, c = reused_arrays(s)
            base[s], rctl = z, {**rctl, **c}
        ctl["reused_arrays"] = {"PASS": all(v["PASS"] for v in rctl.values()), "files": rctl}
        res["controls"] = ctl
        bad = [k for k, v in ctl.items() if not v.get("PASS")]
        if bad:
            res["refused"] = f"controls failed: {bad}"
            json.dump(res, open(f"{a.raw}/LFO_READ.json", "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
            print("REFUSED:", res["refused"])
            return 3
        for tag, _r, _s, _sc, _f in rollouts():
            t = np.load(f"{a.out_root}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True)
            inst[tag] = tick_instruments({k: t[k] for k in t.files})
        res["instruments"] = inst
        beside = {}
        for s in SEEDS:                                                  # the heads-source arm's own instruments, from the LF read's tick dumps (beside; absent = None)
            p = f"{LF_OUT}/RILF075_s{s}/ticks/ticks_RILF075_s{s}.npz"
            if os.path.isfile(p):
                t = np.load(p, allow_pickle=True)
                beside[f"RILF075_s{s}"] = LFR.tick_instruments({k: t[k] for k in t.files})
        res["instruments_lf075_beside"] = beside or None
        paths = extract_new(a.out_root, arr_dir, a.workers)
        out_json = f"{a.raw}/LFO_READ.json"
    else:
        for s in SEEDS:
            base[s], c = ka_arrays(a.mode, s)
            rctl.update(c)
        res["controls"] = {"reused_arrays": {"PASS": all(v["PASS"] for v in rctl.values()), "files": rctl}}
        if not res["controls"]["reused_arrays"]["PASS"]:
            print("REFUSED: a reused array differs from its pinned sha256")
            return 3
        out_json = f"{a.raw}/KA_READER_{a.mode}.json"
    arrays = {}
    for s in SEEDS:
        arrays[s] = dict(base[s])
        if a.mode == "production":
            for role in PRIORITY:
                arrays[s][role] = rr.read_arrays(paths[f"{name_of(role)}_s{s}"])
    reach = LFR.reach_control({f"s{s}": inst[f"{name_of('RILFO')}_s{s}"] for s in SEEDS}) if a.mode == "production" else None
    res["arm"] = read_ceiling(arrays, a.n_boot, reach)
    res["headline"] = headline(res["arm"])
    if a.mode != "production":
        chk = ka_check(a.mode, res["arm"])
        res["ka_checks"], res["KA_PASS"] = chk, bool(all(chk.values()))
        print(f"KA {a.mode}: {sum(chk.values())}/{len(chk)} checks hold; label {res['arm']['outcome']['label']}")
        for k, v in chk.items():
            if not v:
                print("  RED", k)
    json.dump(res, open(out_json, "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
    if a.mode == "production":
        open(a.results_md, "w", encoding="utf-8", newline="\n").write(render_md(res))
        print("HEADLINE", res["headline"])
    else:
        open(f"{a.raw}/KA_READER_{a.mode}_RESULTS.md", "w", encoding="utf-8", newline="\n").write(render_md(res))
    return 0 if res.get("KA_PASS", True) else 2


if __name__ == "__main__":
    sys.exit(main())
