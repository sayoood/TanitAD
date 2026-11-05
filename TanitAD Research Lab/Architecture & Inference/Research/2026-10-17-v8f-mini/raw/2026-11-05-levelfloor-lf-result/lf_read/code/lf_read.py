#!/usr/bin/env python3
"""lf_read.py -- the T1-kin read of arms LF050 and LF075 (PREREG_LEVELFLOOR_A3.md, sha256 977a785e...: the trained vision heads of LLR read OUTSIDE O as a floor on the plan's along-track progress; planner = C0).

NOT a text clone of llr_read.py: the LF read is per DOSE (two doses, each its own P, G, F, D, LEAK-BORNE and outcome), so this module LOADS llr_read.py BY SHA-PINNED PATH and calls its rule functions (rule_P, rule_G, rule_F,
rule_D, rule_leak, family_table, init; the CL card's section 4 with CL := LF) and re-uses their arithmetic unchanged.  What is new here: the roles (RILF{d}, RILF{d}C reindexed; LF{d}, LF{d}C logged), the per-dose loop, the
tick instruments (bind, cap, lift, v_f against the logged speed), the reach control (F-UNREACHED), the A3 card's outcome labels (section 5), the headline with the multiplicity disclosure, and the floor-key controls.

Roles per dose d in {050, 075} (kappa 0.50 / 0.75) and seed s (tag = name + "_s{s}"; planner = C0_s{s}, floor heads = LLR_s{s}; flags = the BANKED base rollouts' flags + the three floor flags):
  NEW (this package's 16 rollouts, F:/TanitAD-scratch/motion_c/lf_read/out)   RILF{d} reindexed | RILF{d}C reindexed + kbox | LF{d} logged | LF{d}C logged + kbox
  REUSED (read only, sha256-pinned)   C0, C0C, C0REP (recover_next/read/raw/arrays)   RIC0, RIC0C (motion_c/ri_read/raw/arrays); P10 is the b_ part of every array.   No new base rollout.
Rules (per dose, deciding scene `reindexed`):  P | G | F | D | LEAK-BORNE as llr_read.py implements them with CL := LF.  Reach control first: F must bind on >= 1 % of the replans inside the M1 crawl ticks at EACH seed
(RILF{d}), else F-UNREACHED and NO rule is read for that dose.  hold_now is not read (A3 card section 3, disclosed).
Tier T1-kin.  Estimator: paired episode-cluster bootstrap B = 2000, seed 0 (gapdecomp.decompose) over the 141 eval episodes / 2,399 windows: answers another draw of EPISODES only; the planner and F are deterministic.

usage: python -I lf_read.py --mode production [--out-root F:/TanitAD-scratch/motion_c/lf_read/out] [--workers 6]   (controls first; a failed control REFUSES the read, exit 3)
       python -I lf_read.py --mode ka-null | ka-llr                                                               (known answers on EXISTING arrays, no rollout; raw/KA_READER_<mode>.json)
"""
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
DL = f"{ART}/dagland"
MC = f"{DL}/motion_c"
PKG = f"{MC}/lf_read"                                                   # the package (fixed path: a mutated copy of this file still finds its siblings)
LLR_PK = f"{MC}/llr_read"
LLR_READ_PY = f"{LLR_PK}/code/llr_read.py"
LLR_READ_SHA = "4d6274607d0cec1dc825c973d6e7c6178dbb898ea2e8f9c7819c4f0e634d2e85"
LLR_READ_JSON = f"{LLR_PK}/raw/LLR_READ.json"
LLR_READ_JSON_SHA = "fb2d8d4613ecfe194f899a2c2e809c37cdad582e42d3f069fbe3d5fa0cdc0604"      # the LLR read's result: the reference of the ka-llr known answer
LLR_ARR = f"{LLR_PK}/raw/arrays"
REF_ARR = f"{DL}/recover_next/read/raw/arrays"
RI_ARR = f"{MC}/ri_read/raw/arrays"
CARD = f"{MC}/a3_prereg/PREREG_LEVELFLOOR_A3.md"
CARD_FROZEN = f"{MC}/a3_prereg/FROZEN_SHA.txt"
CARD_SHA = "977a785e54df0b7e13bb8f81639d88eb52bb64092e5e1657e70b6e09fde2d405"
LF_BUILD_PINS = os.environ.get("LF_BUILD_PINS", f"{MC}/lf_build/PINS.txt")      # key `patched_tool`: the floor operator's tool sha256 (built by the tool agent)
RDB = f"{ART}/v8f-recover"
OUT_ROOT = "F:/TanitAD-scratch/motion_c/lf_read/out"
WORK_DIR = "F:/TanitAD-scratch/motion_c/lf_read/work"
SEEDS = (0, 1)
DOSES = ("050", "075")
KAPPA = {"050": 0.5, "075": 0.75}
KAPPA_FLAG = {"050": "0.5", "075": "0.75"}
M1 = "M1 crawl share"
BASE_SMOOTH = "sg72"             # the banked C0 / C0C / RIC0 / RIC0C rollouts' flags carry `--smooth sg72` (their done markers); the bare LF arm copies them, so LF - RIC0 isolates the floor
BASE_ROLES = ("C0", "C0C", "C0REP", "RIC0", "RIC0C")
BASE_DIR = {"C0": REF_ARR, "C0C": REF_ARR, "C0REP": REF_ARR, "RIC0": RI_ARR, "RIC0C": RI_ARR}
BASE_SHA = {"C0_s0": "5b04b8f92c710444bb1517d802d0b9a57b8134c41f4304d635417585cbc163ab", "C0_s1": "8280c3f47d90de9f68432bc1a97fe584b87d1e0fee94ac5a00c465dfc71d7ef9",
            "C0C_s0": "265c804427eb66d6b541236697e941c184e487c8a31c03ee894a377d6141c8dc", "C0C_s1": "33e78abcf17519b425f81be7a8091b23595f5fae9d825e32e6977b1ab972b791",
            "C0REP_s0": "f6b6336fd1d6421f910f78e3bde6f9c236a03643c2ac845ef25eda7feb2b4a7b", "C0REP_s1": "7ee047c07ca387a76453ef23f7e53f67376ca9d30fbb98d368bded6663cc7b80",
            "RIC0_s0": "31e2ad8ddb4f6e8e6bfe6ee780f47b7c0adcba1cc1f9047071eb64939042a244", "RIC0_s1": "3b9b814d7b285cb05fcefbdfeb21c71eab01d728383ba1d737da84f1b322601d",
            "RIC0C_s0": "f88c7a80aec9d3711cd616c9b45aa064109368e6a16d806c5e7bb65af275efab", "RIC0C_s1": "7ef6cc2a30600d11c822a2b609b4aea40eb5f20c1f1b72759e6290dc5973050f"}
LLR_ARR_SHA = {"LLR_s0": "6d73a673d3473bb0701e955837a395cf0705db73e233ee0ad903a19529e3a869", "LLR_s1": "09c70c3c86fd8a57615270158f36b39b3fca63480c69f090fd2ef7b8a74ce98a",
               "LLRC_s0": "273c1634452e96039a7966020780cace14d484a6169d2053214cb5a614f2173e", "LLRC_s1": "70034787aace4771bc1157a02637d974ba5b7643b75403f8c65b4c1988e5136a",
               "RILLR_s0": "3c493f019bb283e285b30f5c110b8f642d5acf834e0f86f55807b08be78a29f9", "RILLR_s1": "812897cb6683bef4d1575ee74ceb913c9352c4155f8a54b1d4b23cd26e5737b3",
               "RILLRC_s0": "821474d52d49767dc316cfcd67be3b295ca46487bd9f413775f53b26cde3df29", "RILLRC_s1": "e6a79360dcc2ea5c0a17c93cb6f75e4120a2da150d245220d37bc1335d7c28c4"}      # the LLR read's arrays: the ka-llr known answer
# generic role -> (scene, feasible); the rollout name carries the dose: RILF050C = reindexed + kbox at kappa 0.50
GENERIC = {"RILF": ("reindexed", "off"), "RILFC": ("reindexed", "kbox"), "LF": ("logged", "off"), "LFC": ("logged", "kbox")}
PRIORITY = (("RILF", "050"), ("RILFC", "050"), ("RILF", "075"), ("RILFC", "075"), ("LF", "050"), ("LFC", "050"), ("LF", "075"), ("LFC", "075"))      # A3 card section 3 / the brief: deciding scene first
FOUR_FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
LF_FIELDS = ("lf_bind", "lf_cap", "lf_vf", "lf_vhat", "lf_pi", "lf_gap", "lf_lift")
N_TICKS = 3 * 2399                  # replan_steps 10 over the 30-step horizon: 3 replans per window (the LLR read's tick files carry 7,197 rows)
DT, VMIN, BAND_LO, CRAWL_RATIO, REACH_BAR = 0.2, 5.2, 3.0, 0.5, 0.01          # VMIN = rr_lib.VMIN (M1's own threshold); BAND_LO / CRAWL_RATIO = the card's 3 m/s band and the 0.5 crawl ratio
DK_ROWS, LONG_ROWS = ("DK-VIOL", "DK-HW", "DK-TG"), ("ADE6", "speed MAE", "along-track MAE")
DISCLOSURE = "Two doses were read (kappa 0.50 and kappa 0.75): a counted DOMINANT names its dose and says that two were read (A3 card section 4, multiplicity)."
LR = rr = SR = GD = PC = None


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


def init(llr_path=None):
    """Load llr_read.py BY FILE PATH (sha256 checked first; ``llr_path`` only for the tests' own copies), run its init() (the RECOVER-NEXT reader and its pinned instruments), load lf_precheck.py by path."""
    global LR, rr, SR, GD, PC
    path = llr_path or LLR_READ_PY
    got = sha256_file(path)
    if path == LLR_READ_PY and got != LLR_READ_SHA:
        raise SystemExit(f"REFUSED: {path} is {got}, not the pinned {LLR_READ_SHA}")
    LR = _load("llr_read_for_lf", path)
    LR.init()
    rr, SR, GD = LR.rr, LR.SR, LR.GD
    PC = _load("lf_precheck_for_read", f"{PKG}/code/lf_precheck.py")
    assert LR.M1 == M1 and tuple(LR.SEEDS) == SEEDS
    return LR


def name_of(base, dose):
    """The rollout / array name of a generic role at a dose: RILF + 050 -> RILF050, RILFC + 075 -> RILF075C."""
    return ("RI" if base.startswith("RI") else "") + "LF" + dose + ("C" if base.endswith("C") else "")


def rollouts():
    """The 16 rollouts in priority order: [(tag, base role, dose, seed, scene, feasible)]."""
    return [(f"{name_of(b, d)}_s{s}", b, d, s, GENERIC[b][0], GENERIC[b][1]) for b, d in PRIORITY for s in SEEDS]


def floor_ckpt(s):
    return f"{RDB}/M2b_s{s}_recft_LLR/ckpt_last.pt"


def c0_ckpt(s):
    return f"{RDB}/M2b_s{s}_recft_C0/ckpt_last.pt"


def expected_flags(base, dose, s):
    scene, feas = GENERIC[base]
    return f"--smooth {BASE_SMOOTH} --scene {scene}" + (f" --feasible {feas}" if feas != "off" else "") + f" --level-floor on --floor-kappa {KAPPA_FLAG[dose]} --floor-ckpt {floor_ckpt(s)}"


# ------------------------------------------------------------------------------------------------ pure layer 1: tick instruments (no file access)
def logged_speed(gt, gt_ok):
    """The logged route's own speed at the tick, over its first second: the path length of knots 1..6 (five segments, offset-free: the executed pose may differ from the logged one) / 1.0 s.  -> (v_log [N], ok [N] bool)."""
    gt = np.asarray(gt, np.float64)
    v = np.linalg.norm(np.diff(gt[:, :6], axis=1), axis=-1).sum(1) / (5 * DT)
    ok = np.asarray(gt_ok, bool) & np.isfinite(v)
    return np.where(ok, v, np.nan), ok


def m1_ticks(v0, v_log, ok):
    """The M1 crawl tick: M1's own pair definition (rr_lib.m1_window: the logged pair speed >= 5.2 m/s while the own speed is < 5.2 m/s) applied to the tick's (executed measured speed, logged route speed) pair."""
    return np.asarray(ok, bool) & (np.nan_to_num(v_log, nan=-1.0) >= VMIN) & (np.asarray(v0, np.float64) < VMIN)


def card_crawl_ticks(v0, v_log, ok):
    """Beside: the CL / A3 card's crawl band: the executed speed below half the logged speed while the logged speed is >= 3 m/s."""
    vl = np.nan_to_num(v_log, nan=-1.0)
    return np.asarray(ok, bool) & (vl >= BAND_LO) & (np.asarray(v0, np.float64) < CRAWL_RATIO * vl)


def _rate(x, m):
    n = int(np.sum(m))
    return {"n": n, "rate": None if n == 0 else float(np.mean(np.asarray(x, float)[m]))}


def tick_instruments(t):
    """The A3 card section 3 instruments from one rollout's tick dump (``t`` = dict of arrays with the LF fields): bind rate overall and inside the M1 crawl ticks, cap rate, mean lift where bound, v_f against the logged speed."""
    need = LF_FIELDS + ("v0", "gt", "gt_ok")
    miss = [k for k in need if k not in t]
    if miss:
        raise KeyError(f"tick dump lacks {miss}")
    n = len(t["v0"])
    bind, cap = np.asarray(t["lf_bind"], bool), np.asarray(t["lf_cap"], bool)
    vf, vh, lift = (np.asarray(t[k], np.float64) for k in ("lf_vf", "lf_vhat", "lf_lift"))
    v_log, ok = logged_speed(t["gt"], t["gt_ok"])
    m1, card = m1_ticks(t["v0"], v_log, ok), card_crawl_ticks(t["v0"], v_log, ok)
    allt = np.ones(n, bool)
    fd = np.isfinite(vf) & np.isfinite(vh) & (vh >= BAND_LO) & ok

    def lift_where_bound(m):
        sel = bind & m & np.isfinite(lift)
        neg = sel & (lift < -1e-6)                                     # A3_RULING_2: points moved forward along a bending path are joined by shorter chords
        return {"n_bound": int(sel.sum()), "mean_lift_mps": None if not sel.any() else float(lift[sel].mean()),
                "n_negative_chord_lift": int(neg.sum()), "share_negative_chord_lift": None if not sel.any() else float(neg.sum() / sel.sum()),
                "min_lift_mps": None if not sel.any() else float(lift[sel].min()), "q01_lift_mps": None if not sel.any() else float(np.quantile(lift[sel], 0.01))}

    def vf_block(m):
        sel = fd & m
        if not sel.any():
            return {"n": 0, "mae_mps": None, "share_vf_exceeds_logged": None}
        e = vf[sel] - v_log[sel]
        return {"n": int(sel.sum()), "mae_mps": float(np.abs(e).mean()), "share_vf_exceeds_logged": float((e > 0).mean())}
    return {"n_ticks": int(n), "n_m1_ticks": int(m1.sum()), "n_card_crawl_ticks": int(card.sum()), "n_logged_speed_valid": int(ok.sum()),
            "bind_rate_overall": _rate(bind, allt), "bind_rate_in_m1_ticks": _rate(bind, m1), "bind_rate_in_card_crawl_ticks_beside": _rate(bind, card),
            "cap_rate_overall": _rate(cap, allt), "cap_rate_in_m1_ticks": _rate(cap, m1),
            "lift_where_bound_overall": lift_where_bound(allt), "lift_where_bound_in_m1_ticks": lift_where_bound(m1),
            "vf_vs_logged_overall": vf_block(allt), "vf_vs_logged_in_m1_ticks": vf_block(m1),
            "definitions": {"m1_tick": "logged route speed (first second of the tick's gt, path length / 1 s) >= 5.2 m/s and the executed measured speed v0 < 5.2 m/s (rr_lib.m1_window's pair thresholds on the tick pair)",
                            "card_crawl_tick_beside": "v0 < 0.5 x logged route speed and logged route speed >= 3 m/s", "lift": "lf_lift where lf_bind (the tool's 0-3 s plan speed lift)",
                            "v_f_vs_logged": "ticks with a floor (v_hat >= 3 m/s, finite v_f) and a valid logged speed: mean abs(v_f - v_log) and the share where v_f > v_log"}}


def tick_controls(t, kappa):
    """Tool-versus-card consistency of one tick dump (A3 card section 2): a violation means the tool does not implement the frozen operator."""
    f = lambda k: np.asarray(t[k], np.float64)
    if any(k not in t for k in LF_FIELDS + ("v0", "gt", "gt_ok")):
        return {"fields_present": False}
    bind, cap = np.asarray(t["lf_bind"], bool), np.asarray(t["lf_cap"], bool)
    vf, vh, pi, lift = f("lf_vf"), f("lf_vhat"), f("lf_pi"), f("lf_lift")
    fin = np.isfinite(vf) & np.isfinite(vh) & (vh >= BAND_LO)
    lo = np.isfinite(vh) & (vh < BAND_LO)
    return {"fields_present": True, "n_ticks_is_3x2399": bool(len(vf) == N_TICKS),
            "v_hat_and_pi_finite_every_tick": bool(np.isfinite(vh).all() and np.isfinite(pi).all()),
            "pi_in_unit_interval": bool(((pi >= 0) & (pi <= 1)).all()),
            "no_bind_below_the_3_mps_band": bool(not (bind & lo).any()),
            "v_f_between_0_and_kappa_v_hat": bool(((vf[fin] >= -1e-6) & (vf[fin] <= kappa * vh[fin] + 1e-3)).all()),
            "cap_only_where_presence_at_least_half": bool((pi[cap] >= 0.5 - 1e-6).all()),
            "lift_zero_where_unbound": bool((np.abs(np.nan_to_num(lift[~bind])) <= 1e-6).all())}      # A3_RULING_2: a negative CHORD lift where F binds is a reported result (tick_instruments), not a control


def reach_control(inst_by_seed):
    """A3 card section 3: F must bind on >= 1 % of the replans inside the M1 crawl ticks at EACH seed, else F-UNREACHED.  ``inst_by_seed`` = {seed key: tick_instruments of RILF{d}}.  A seed with no M1 tick cannot evidence reach."""
    per = {}
    for k, v in inst_by_seed.items():
        b = v["bind_rate_in_m1_ticks"]
        per[k] = {"bind_rate_in_m1_ticks": b["rate"], "n_m1_ticks": b["n"], "reached": bool(b["n"] > 0 and b["rate"] is not None and b["rate"] >= REACH_BAR)}
    return {"per_seed": per, "bar": REACH_BAR, "PASS": bool(len(per) == len(SEEDS) and all(p["reached"] for p in per.values()))}


# ------------------------------------------------------------------------------------------------ pure layer 2: the A3 card's outcomes (section 5)
def harm_rows(G, D):
    """The rows that name the harm when G or D fails: G's separated-worse non-crawl rows (and ADE6 when its point is over the guard) and D's separated-worse rows, over both seeds."""
    rows = set()
    for p in G["per_seed"].values():
        rows |= set(p["non_crawl_separated_worse"])
        if not p["ade6_within_guard"]:
            rows.add("ADE6")
    for v in D["by_seed"].values():
        rows |= set(v["separated_worse_rows"])
    return sorted(rows)


def next_for_harm(dose, rows):
    """A3 card section 5, 'P holds, G or D fails': failures in ADE6 / speed MAE / along-track MAE -> the dose midway to the lower dose (kappa 0.375 below 0.50); failures only in the DK rows -> the same dose with a 2.0 s time gap."""
    if set(rows) & set(LONG_ROWS):
        mid = "kappa 0.375 (below 0.50)" if dose == "050" else "kappa 0.625 (midway to 0.50)"
        return f"next arm: {mid}, zero training, its own addendum (failures in {', '.join(sorted(set(rows) & set(LONG_ROWS)))})"
    if rows and set(rows) <= set(DK_ROWS):
        return f"next arm: the same dose (kappa {KAPPA[dose]:.2f}) with a 2.0 s time gap, zero training, its own addendum (failures only in the DK rows: {', '.join(rows)})"
    return "the card names no next arm for these failing rows (" + (", ".join(rows) if rows else "none in a row: D1 / D3 terms") + "): the Master Mind names it, reported"


def decide_dose(dose, reach, P=None, G=None, D=None, leak=None, m1_moved=False):
    """A3 card sections 4-5 for ONE dose.  Precedence: F-UNREACHED (no rule read) > LEAK-BORNE (whatever else) > D-HOLDS > P-HOLDS-G-OR-D-FAILS > P-FAILS-M1-MOVED / P-FAILS-M1-NOT-MOVED.
    ``reach`` None = not evaluated (known-answer runs only).  Returns {label, underlying, next, counts_as_result, harm_rows, notes}."""
    if reach is not None and not reach["PASS"]:
        return {"label": "F-UNREACHED", "underlying": None, "next": "the floor does not bind where the crawl is: the instrument is checked first; no rule is read for this dose (A3 card section 5)", "counts_as_result": False, "harm_rows": [], "notes": []}
    notes, rows = [], []
    if D["DOMINANT_COUNTED"]:
        under, nxt = "D-HOLDS", "counted DOMINANT: signal the PI for scale (data and size), with the video evidence"
    elif P["PASS"]:
        under = "P-HOLDS-G-OR-D-FAILS"
        rows = harm_rows(G, D)
        nxt = next_for_harm(dose, rows)
        notes.append("G fails" if not G["PASS"] else "G holds, D fails")
    else:
        under = "P-FAILS-M1-MOVED" if m1_moved else "P-FAILS-M1-NOT-MOVED"
        nxt = ("the floor binds but the head's estimate is too weak to lift the crawl: the level estimator is the bottleneck (levers: motion from vision, option (c) T2 re-perception / the A40, or a stronger estimator on more data)" if m1_moved
               else "P fails and M1 is not moved: the floor does not bind where the crawl is; the instrument is checked first")
    if leak["LEAK_BORNE"]:
        return {"label": "LEAK-BORNE", "underlying": under, "next": "the camera-clock effect, never a result; next as for 'P fails' (" + ("estimator bottleneck" if m1_moved else "instrument first") + ")", "counts_as_result": False, "harm_rows": rows, "notes": notes}
    return {"label": under, "underlying": under, "next": nxt, "counts_as_result": True, "harm_rows": rows, "notes": notes}


def joint_reading(per_dose):
    """The headline across the two doses.  ``per_dose`` = {dose: {"outcome": decide_dose result, "m1_moved": bool}}.  D-HOLDS at a dose = counted DOMINANT, naming the dose; always the multiplicity disclosure."""
    lab = {d: v["outcome"]["label"] for d, v in per_dose.items()}
    dom = [d for d in DOSES if d in lab and lab[d] == "D-HOLDS"]
    ph = [d for d in DOSES if d in lab and lab[d] == "P-HOLDS-G-OR-D-FAILS"]
    lines = [f"dose {d} (kappa {KAPPA[d]:.2f}): {lab[d]}" + (f" (underlying {per_dose[d]['outcome']['underlying']})" if lab[d] == "LEAK-BORNE" else "") + f" -> {per_dose[d]['outcome']['next']}" for d in DOSES if d in lab]
    if dom:
        branch = "COUNTED-DOMINANT"
        head = "COUNTED DOMINANT at dose " + " and ".join(f"{d} (kappa {KAPPA[d]:.2f})" for d in dom) + ". Signal the PI for scale (data and size), with the video evidence. " + DISCLOSURE
    elif ph:
        branch = "P-HOLDS-G-OR-D-FAILS"
        head = "P holds at dose " + " and ".join(f"{d}" for d in ph) + ", G or D fails: the failing rows name the harm (below); each next arm gets its own addendum and costs zero training. " + DISCLOSURE
    elif any(lab[d] == "F-UNREACHED" for d in lab):
        branch = "F-UNREACHED"
        head = "F-UNREACHED at dose " + " and ".join(d for d in lab if lab[d] == "F-UNREACHED") + ": the floor does not bind where the crawl is; the instrument is checked first and no rule is read at that dose. " + DISCLOSURE
    elif any(per_dose[d]["m1_moved"] for d in per_dose):
        branch = "P-FAILS-ESTIMATOR-BOTTLENECK"
        head = ("P fails at both doses and M1 is separated-better at dose " + " and ".join(d for d in per_dose if per_dose[d]["m1_moved"]) + ": the floor binds but the head's estimate is too weak to lift the crawl; the level estimator is the bottleneck. "
                "Its levers are motion from vision (option (c): T2 re-perception, the A40) or a stronger estimator trained on more data, reported to the PI with the binding and v_f-error instruments. " + DISCLOSURE)
    else:
        branch = "P-FAILS-M1-NOT-MOVED"
        head = "P fails at both doses and M1 is not moved: the floor does not bind where the crawl is; the instrument is checked first. " + DISCLOSURE
    if any(lab[d] == "LEAK-BORNE" for d in lab):
        head += " LEAK-BORNE at dose " + " and ".join(d for d in lab if lab[d] == "LEAK-BORNE") + ": the camera-clock effect, never a result."
    return {"branch": branch, "headline": head, "per_dose_lines": lines, "doses_read": list(DOSES), "counted_dominant_doses": dom, "p_holds_doses": ph}


# ------------------------------------------------------------------------------------------------ the per-dose read (arrays)
def _rows(ar, eid, cls, x, y, n_boot, full=None):
    return rr.rows_of(rr.Aof(ar[x]), rr.Aof(ar[y]), eid, cls, n_boot, full)


def _vs_p10(ar, eid, cls, x, n_boot):
    return rr.rows_of(rr.Aof(ar[x]), rr.Bof(ar[x]), eid, cls, n_boot)[0]


def read_dose(dose, arrays, n_boot=2000, reach=None, cache=None, log=print):
    """``arrays`` = {seed: {generic role: array dict}} with RILF RILFC LF LFC (this dose's rollouts) and C0 C0C C0REP RIC0 RIC0C.  ``reach`` = reach_control() of RILF{dose} (None only in a known-answer run).  ``cache`` shares the
    dose-independent base contrasts between the two doses.  -> {dose, kappa, reach, seeds, rules, outcome, m1_moved}; a failed reach control returns WITHOUT reading a rule."""
    res = {"dose": dose, "kappa": KAPPA[dose], "reach": reach}
    if reach is not None and not reach["PASS"]:
        res.update({"seeds": None, "rules": None, "m1_moved": False, "outcome": decide_dose(dose, reach)})
        return res
    cache = {} if cache is None else cache
    res["seeds"] = {}
    cells_ri, cells_lg, xy_ri, rows_ri, ra_c, eff_ri, eff_lg, rep = {}, {}, {}, {}, {}, {}, {}, {}
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
        b["RILF_minus_RIC0"], b["M1_per_class_split_RILF_minus_RIC0"] = _rows(ar, eid, cls, "RILF", "RIC0", n_boot, M1)
        b["RIC0_minus_P10"] = base("RIC0_P10", lambda: _vs_p10(ar, eid, cls, "RIC0", n_boot))
        b["RILF_minus_P10"], b["RILFC_minus_P10"] = _vs_p10(ar, eid, cls, "RILF", n_boot), _vs_p10(ar, eid, cls, "RILFC", n_boot)
        b["RIC0C_minus_P10"] = base("RIC0C_P10", lambda: _vs_p10(ar, eid, cls, "RIC0C", n_boot))
        b["RILFC_minus_RIC0C"] = _rows(ar, eid, cls, "RILFC", "RIC0C", n_boot)[0]
        b["LF_minus_C0"], b["M1_per_class_split_LF_minus_C0"] = _rows(ar, eid, cls, "LF", "C0", n_boot, M1)
        b["C0_minus_P10"] = base("C0_P10", lambda: _vs_p10(ar, eid, cls, "C0", n_boot))
        b["LF_minus_P10"], b["LFC_minus_P10"] = _vs_p10(ar, eid, cls, "LF", n_boot), _vs_p10(ar, eid, cls, "LFC", n_boot)
        b["C0C_minus_P10"] = base("C0C_P10", lambda: _vs_p10(ar, eid, cls, "C0C", n_boot))
        b["LFC_minus_C0C"] = _rows(ar, eid, cls, "LFC", "C0C", n_boot)[0]
        b["C0REP_minus_C0"] = base("C0REP_C0", lambda: _rows(ar, eid, cls, "C0REP", "C0", n_boot)[0])
        cells_ri[k] = rr.m1_cell(b["RILF_minus_RIC0"][M1], b["RIC0_minus_P10"][M1])
        cells_lg[k] = rr.m1_cell(b["LF_minus_C0"][M1], b["C0_minus_P10"][M1])
        xy_ri[k], rows_ri[k] = SR.read_xy(GD, b["RILF_minus_RIC0"]), b["RILF_minus_RIC0"]
        ra_c[k] = rr.read_a_block(SR, GD, b["RILFC_minus_P10"])
        eff_ri[k], eff_lg[k], rep[k] = b["RILF_minus_RIC0"][M1]["delta"], b["LF_minus_C0"][M1]["delta"], b["C0REP_minus_C0"][M1]["delta"]
        b["read_a_RILFC_minus_P10"], b["read_a_LFC_minus_P10_logged_beside"] = ra_c[k], rr.read_a_block(SR, GD, b["LFC_minus_P10"])
        b["read_a_RIC0C_base_beside"], b["read_a_C0C_base_beside"] = rr.read_a_block(SR, GD, b["RIC0C_minus_P10"]), rr.read_a_block(SR, GD, b["C0C_minus_P10"])
        b["read_xy_RILF_minus_RIC0"], b["read_xy_LF_minus_C0_logged_beside"] = xy_ri[k], SR.read_xy(GD, b["LF_minus_C0"])
        b["m1_cell_reindexed"], b["m1_cell_logged"] = cells_ri[k], cells_lg[k]
        b["family_tables"] = {c: LR.family_table(b[c]) for c in ("RILF_minus_RIC0", "LF_minus_C0", "RILFC_minus_P10", "LFC_minus_P10")}
        b["floor_max_abs_dev_vs_C0_m"] = {role: float(max(np.nanmax(np.abs(np.asarray(u, float) - np.asarray(v, float))) for u, v in zip(rr.Bof(ar[role]), rr.Bof(ar["C0"])))) for role in ar if role != "C0"}
        res["seeds"][k] = b
        log(f"dose {dose}", k, "M1 RILF-RIC0 %+.4f [%+.4f, %+.4f] sep %s | share %s | logged share %s | ADE6 RILF-RIC0 %+.4f | RILFC read A DOMINANT %s" % (
            b["RILF_minus_RIC0"][M1]["delta"], b["RILF_minus_RIC0"][M1]["lo"], b["RILF_minus_RIC0"][M1]["hi"], b["RILF_minus_RIC0"][M1]["separated"], cells_ri[k]["share_closed"],
            cells_lg[k]["share_closed"], b["RILF_minus_RIC0"]["ADE6"]["delta"], ra_c[k]["DOMINANT"]))
    P, G = LR.rule_P(cells_ri), LR.rule_G(xy_ri, rows_ri)
    F = LR.rule_F(eff_ri, rep, eff_lg)
    leak = LR.rule_leak(cells_lg, cells_ri)
    D = LR.rule_D(ra_c, P["PASS"], G["PASS"])
    moved = bool(any(c["better"] for c in cells_ri.values()))
    out = decide_dose(dose, reach, P, G, D, leak, moved)
    if out["underlying"] in ("D-HOLDS", "P-HOLDS-G-OR-D-FAILS") and not F["PASS"]:
        out["notes"].append("F fails: |RILF - RIC0| on M1 does not exceed the C0REP replicate floor at every seed (the card's D does not require F; the training-variance reading is the Master Mind's)")
    res.update({"rules": {"P": P, "G": G, "F": F, "D": D, "LEAK_BORNE": leak, "P_logged_form_beside": LR.rule_P(cells_lg), "deciding_scene": "reindexed", "m1_moved_at_any_seed": moved,
                          "m1_moved_definition": "RILF - RIC0 on M1 separated-better at >= 1 seed (the CL card's reading (d), as llr_read.py's attribution)"}, "m1_moved": moved, "outcome": out})
    return res


# ------------------------------------------------------------------------------------------------ files and controls
def reused_arrays(s):
    z, ctl = {}, {}
    for role in BASE_ROLES:
        p = f"{BASE_DIR[role]}/{role}_s{s}.npz"
        got = sha256_file(p)
        ctl[f"{role}_s{s}"] = {"path": p, "sha256": got, "pinned": BASE_SHA[f"{role}_s{s}"], "PASS": got == BASE_SHA[f"{role}_s{s}"]}
        z[role] = rr.read_arrays(p)
    return z, ctl


def parse_pins(path=None):
    d = {}
    for line in open(path or LF_BUILD_PINS, encoding="utf-8"):
        m = re.search(r"[0-9a-f]{64}", line)
        if m and not line.lstrip().startswith("#"):
            d.setdefault(re.split(r"[\s:=]+", line.strip())[0], m.group(0))
    return d


def frozen_card_sha():
    for line in open(CARD_FROZEN, encoding="utf-8"):
        t = line.split()
        if len(t) >= 2 and t[0] == os.path.basename(CARD):
            return t[1]
    return None


def run_controls(out_root, raw, pins_path=None):
    """Production controls before any row: per rollout the done marker (flags exactly the banked base's + the floor flags, tool sha256 = lf_build `patched_tool`, planner checkpoint = C0's), the run header, the floor keys of the output
    JSON (lf_precheck.check_rollout), the tick dump's LF fields and their consistency with the card; one planner checkpoint per seed; the pre-check file; the reused arrays; the cards."""
    ctl, ckpts = {}, {}
    try:
        tool_pin = parse_pins(pins_path).get("patched_tool")
    except Exception:
        tool_pin = None
    ctl["patched_tool_pin"] = {"PASS": bool(tool_pin and re.fullmatch(r"[0-9a-f]{64}", tool_pin)), "patched_tool": tool_pin, "pins_file": pins_path or LF_BUILD_PINS}
    for tag, base, dose, s, scene, feas in rollouts():
        c = {"PASS": False}
        try:
            d = json.load(open(f"{out_root}/done_{tag}.txt", encoding="utf-8"))
            h = rr.run_header(out_root, tag)
            fe = (h.get("feasible") or {}).get("mode", "off") if isinstance(h.get("feasible"), dict) else ("off" if not h.get("feasible") else str(h.get("feasible")))
            sm = h["smooth"]["mode"] if isinstance(h.get("smooth"), dict) else h.get("smooth")
            fc, fdet = PC.check_rollout(out_root, tag)
            t = np.load(f"{out_root}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True)
            tc = tick_controls({k: t[k] for k in t.files}, KAPPA[dose])
            c = {"flags_exactly_the_banked_base_plus_floor": d["flags"] == expected_flags(base, dose, s), "tool_sha256_is_patched_tool": bool(tool_pin and d["tool_sha256"] == tool_pin),
                 "ckpt_is_C0": d["ckpt"].replace("\\", "/").endswith(f"/M2b_s{s}_recft_C0/ckpt_last.pt"), "header_scene": h["scene"] == scene, "header_feasible": fe == feas, "header_smooth": sm == BASE_SMOOTH,
                 "header_n_windows_2399": int(h["n_windows"]) == 2399, "header_kin_yaw": h["kin_mode"] == "yaw", "ckpt_sha256": d["ckpt_sha256"], "threads": d.get("threads")}
            c.update({f"floor_{k}": v for k, v in fc.items()})
            c.update({f"ticks_{k}": v for k, v in tc.items()})
            ckpts.setdefault(s, set()).add(d["ckpt_sha256"])
            c["PASS"] = all(v for k, v in c.items() if isinstance(v, bool))
        except Exception as e:      # a missing marker / header / tick file is a failed control, named
            c["error"] = f"{type(e).__name__}: {e}"
        ctl[f"rollout_{tag}"] = c
    ctl["one_checkpoint_per_seed"] = {"PASS": bool(len(ckpts) == len(SEEDS) and all(len(v) == 1 and next(iter(v)) == PC.C0_SHA[s] for s, v in ckpts.items())), "ckpt_sha256": {f"s{s}": sorted(v) for s, v in ckpts.items()}}
    try:
        pc = json.load(open(f"{raw}/LF_PRECHECK.json", encoding="utf-8"))
        ctl["LF_PRECHECK"] = {"PASS": bool(pc["PASS"] is True and all(pc["seeds"][f"s{s}"]["C0"]["ckpt_sha256"] == PC.C0_SHA[s] and pc["seeds"][f"s{s}"]["LLR"]["ckpt_sha256"] == PC.llr_sha_of(s) for s in SEEDS))}
    except Exception as e:
        ctl["LF_PRECHECK"] = {"PASS": False, "error": f"{type(e).__name__}: {e}"}
    got, fz = sha256_file(CARD), frozen_card_sha()
    ctl["card_a3"] = {"PASS": bool(got == CARD_SHA == fz), "sha256": got, "pinned": CARD_SHA, "frozen_sha_file": fz}
    got = LR.card_rules_sha()
    ctl["cl_card_rules"] = {"PASS": got == LR.CARD_RULES_SHA, "sha256": got, "pinned": LR.CARD_RULES_SHA}
    ctl["llr_read_py"] = {"PASS": sha256_file(LR.__file__.replace("\\", "/")) == LLR_READ_SHA, "pinned": LLR_READ_SHA}
    return ctl


def extract_new(out_root, arr_dir, workers):
    jobs = [(tag, s) for tag, _b, _d, s, _sc, _f in rollouts()]
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(rr.extract, tag, s, out_root, arr_dir): tag for tag, s in jobs}
        for f in cf.as_completed(futs):
            f.result()
    return {tag: f"{arr_dir}/{tag}.npz" for tag, _ in jobs}


def ka_arrays(mode, s):
    """Known-answer arrays: the generic roles RILF RILFC LF LFC filled from EXISTING arrays.  ka-null: the arm IS its base (RIC0, RIC0C, C0, C0C).  ka-llr: the LLR read's banked arrays (RILLR, RILLRC, LLR, LLRC) as LF's."""
    z, ctl = reused_arrays(s)
    if mode == "ka-null":
        z.update({"RILF": z["RIC0"], "RILFC": z["RIC0C"], "LF": z["C0"], "LFC": z["C0C"]})
    else:
        for r, c in (("RILF", "RILLR"), ("RILFC", "RILLRC"), ("LF", "LLR"), ("LFC", "LLRC")):
            p = f"{LLR_ARR}/{c}_s{s}.npz"
            got = sha256_file(p)
            ctl[f"{c}_s{s}_fed_as_{r}"] = {"path": p, "sha256": got, "pinned": LLR_ARR_SHA[f"{c}_s{s}"], "PASS": got == LLR_ARR_SHA[f"{c}_s{s}"]}
            z[r] = rr.read_arrays(p)
    return z, ctl


def _same_row(a, b):
    return all(float(a[q]) == float(b[q]) for q in ("delta", "lo", "hi")) and bool(a["separated"]) == bool(b["separated"])


def ka_check(mode, dres):
    """Expected answers: ka-null from first principles (every contrast row exactly 0.0, shares 0.0 or undefined, RILFC read A = the banked RIC0C's); ka-llr from LLR_READ.json (every LLR contrast row equal, every rule equal)."""
    chk = {}
    ref = json.load(open(LLR_READ_JSON, encoding="utf-8"))
    for dose, r in dres.items():
        for s in SEEDS:
            k, b = f"s{s}", r["seeds"][f"s{s}"]
            if mode == "ka-null":
                for c in ("RILF_minus_RIC0", "LF_minus_C0", "RILFC_minus_RIC0C", "LFC_minus_C0C"):
                    chk[f"d{dose}_{k}_{c}_all_rows_exactly_zero"] = all(x["delta"] == 0.0 and not x["separated"] for x in b[c].values())
                chk[f"d{dose}_{k}_shares_zero_or_undefined"] = all(c["share_closed"] in (0.0, None) for c in (b["m1_cell_reindexed"], b["m1_cell_logged"]))
                chk[f"d{dose}_{k}_RILFC_minus_P10_equals_RIC0C_minus_P10"] = all(_same_row(b["RILFC_minus_P10"][lab], b["RIC0C_minus_P10"][lab]) for lab in b["RIC0C_minus_P10"])
            else:
                for mine, old in (("RILF_minus_RIC0", "RILLR_minus_RIC0"), ("LF_minus_C0", "LLR_minus_C0"), ("RILFC_minus_P10", "RILLRC_minus_P10"), ("LFC_minus_P10", "LLRC_minus_P10"),
                                  ("RILFC_minus_RIC0C", "RILLRC_minus_RIC0C"), ("LFC_minus_C0C", "LLRC_minus_C0C"), ("RILF_minus_P10", "RILLR_minus_P10"), ("LF_minus_P10", "LLR_minus_P10")):
                    chk[f"d{dose}_{k}_{mine}_equals_LLR_READ[{old}]"] = all(_same_row(b[mine][lab], ref["seeds"][k][old][lab]) for lab in ref["seeds"][k][old])
                chk[f"d{dose}_{k}_m1_cells_equal"] = b["m1_cell_reindexed"] == ref["seeds"][k]["m1_cell_reindexed"] and b["m1_cell_logged"] == ref["seeds"][k]["m1_cell_logged"]
        R, Rr = r["rules"], ref["rules"]
        if mode == "ka-null":
            chk[f"d{dose}_P_fails_F_fails_G_holds_not_leak"] = bool((not R["P"]["PASS"]) and (not R["F"]["PASS"]) and R["G"]["PASS"] and not R["LEAK_BORNE"]["LEAK_BORNE"])
            chk[f"d{dose}_label_is_P_fails_M1_not_moved"] = r["outcome"]["label"] == "P-FAILS-M1-NOT-MOVED"
        else:
            chk[f"d{dose}_rule_P_equals_LLR"] = R["P"]["PASS"] == Rr["P"]["PASS"] and all(R["P"]["per_seed"][k] == Rr["P"]["per_seed"][k] for k in ("s0", "s1"))
            chk[f"d{dose}_rule_G_F_equal_LLR"] = R["G"]["PASS"] == Rr["G"]["PASS"] and R["F"]["PASS"] == Rr["F"]["PASS"] and R["F"]["per_seed"] == Rr["F"]["per_seed"]
            chk[f"d{dose}_rule_D_leak_equal_LLR"] = R["D"]["DOMINANT_COUNTED"] == Rr["D"]["DOMINANT_COUNTED"] and R["LEAK_BORNE"]["LEAK_BORNE"] == Rr["LEAK_BORNE"]["LEAK_BORNE"] and R["LEAK_BORNE"]["per_seed"] == Rr["LEAK_BORNE"]["per_seed"]
            chk[f"d{dose}_label_is_LLRs_branch_mapped"] = (r["outcome"]["label"] == "P-FAILS-M1-MOVED") == (ref["outcome"]["label"] == "P-FAILS-FAMILY-EXHAUSTED" and any(ref["seeds"][k]["m1_cell_reindexed"]["better"] for k in ("s0", "s1")))
    chk["both_doses_identical_in_a_known_answer"] = json.dumps(dres["050"]["rules"], sort_keys=True, default=rr._jd) == json.dumps(dres["075"]["rules"], sort_keys=True, default=rr._jd)
    return chk


# ------------------------------------------------------------------------------------------------ RESULTS.md
def fmt(r):
    return "%+.4f [%+.4f, %+.4f]%s" % (r["delta"], r["lo"], r["hi"], " *" if r["separated"] else "")


def render_md(res):
    J = res["joint"]
    banner = [] if res.get("mode", "production") == "production" else [f"**KNOWN-ANSWER RUN `{res['mode']}` on EXISTING arrays (no LF rollout): a test of this reader, not a read of arms LF050 / LF075.**", ""]
    L = banner + [f"# LF read: arms LF050 and LF075 (the LLR heads as a floor on O's along-track progress, planner C0), T1-kin, generated {res.get('clock', '')}", "", f"**Headline: {J['headline']}**", ""]
    L += [f"* {x}" for x in J["per_dose_lines"]]
    L += ["", f"Tier: {res.get('tier', '')}. Estimator: {res.get('estimator', '')}. `*` = interval excludes 0. STRATEGIC: {res['strategic_family']}", "", "`hold_now` is not read (a frozen frame makes v_hat constant: no information about a frame-reading floor; A3 card section 3).", ""]
    sh = lambda c: "None" if c is None else "%.3f" % (float(c) + 0.0)
    for d in DOSES:
        r = res["doses"][d]
        L += ["", f"## Dose {d} (kappa {KAPPA[d]:.2f}): **{r['outcome']['label']}**" + (f" (underlying {r['outcome']['underlying']})" if r["outcome"]["label"] == "LEAK-BORNE" else ""), "", f"Next: {r['outcome']['next']}.",
              "" if not r["outcome"]["notes"] else "Notes: " + "; ".join(r["outcome"]["notes"]) + ".", ""]
        if r["reach"] is not None:
            L += ["Reach control (F binds on >= 1 % of the replans inside the M1 crawl ticks, each seed): " + "; ".join(f"{k} {sh(v['bind_rate_in_m1_ticks'])} of {v['n_m1_ticks']} ticks" for k, v in r["reach"]["per_seed"].items()) + f" -> **{r['reach']['PASS']}**", ""]
        if r["rules"] is None:
            L += ["No rule is read for this dose (F-UNREACHED).", ""]
            continue
        R = r["rules"]
        P, G, F, D, LK = R["P"], R["G"], R["F"], R["D"], R["LEAK_BORNE"]
        L += ["| rule | s0 | s1 | holds |", "|---|---|---|---|",
              "| P: M1 separated-better and share >= 0.5 | " + " | ".join(f"sep {P['per_seed'][k]['separated_better']}, share {sh(P['per_seed'][k]['share_closed'])}" for k in ("s0", "s1")) + f" | **{P['PASS']}** |",
              "| G: ADE6 <= +0.0604 m, no non-crawl row separated-worse | " + " | ".join(f"ADE6 {G['per_seed'][k]['ade6_delta']:+.4f}, worse: {', '.join(G['per_seed'][k]['non_crawl_separated_worse']) or 'none'}" for k in ("s0", "s1")) + f" | **{G['PASS']}** |",
              "| F: abs(RILF-RIC0)[M1] > abs(C0REP-C0)[M1] | " + " | ".join(f"{F['per_seed'][k]['abs_R_minus_C0']:.4f} vs {F['per_seed'][k]['abs_C0rep_minus_C0']:.4f}" for k in ("s0", "s1")) + f" | **{F['PASS']}** |",
              "| LEAK-BORNE: logged share - reindexed share > half the logged share | " + " | ".join(f"logged {sh(LK['per_seed'][k]['logged_share'])}, reindexed {sh(LK['per_seed'][k]['reindexed_share'])}" for k in ("s0", "s1")) + f" | **{LK['LEAK_BORNE']}** |",
              "| D: RILFC - P10 read A (D1, D2, D3 four families) + P + G | " + " | ".join(f"D1 {D['by_seed'][k]['D1']}, D2 {D['by_seed'][k]['D2']}, D3 {D['by_seed'][k]['D3']}" for k in ("s0", "s1")) + f" | **{D['DOMINANT_COUNTED']}** |",
              "", f"Beside: P on the logged scene {R['P_logged_form_beside']['PASS']}; D three-family form {D['three_family_form_reported_beside']}; M1 moved (RILF - RIC0 separated-better at >= 1 seed) {R['m1_moved_at_any_seed']}.", "",
              "D failing terms: " + "; ".join(f"{k}: " + ("; ".join(D["by_seed"][k]["failing_terms"]) or "none") for k in ("s0", "s1")), "", "Harm rows: " + (", ".join(r["outcome"]["harm_rows"]) or "none"), ""]
        for c, title in (("RILF_minus_RIC0", "RILF - RIC0 (reindexed, the deciding contrast)"), ("RILFC_minus_P10", "RILFC - P10 (reindexed composite, D)"), ("LF_minus_C0", "LF - C0 (logged, LEAK-BORNE only)")):
            L += ["", f"### 19 rows, dose {d}: {title}", "", "| row | family | s0 | s1 |", "|---|---|---|---|"]
            for lab in GD.LABELS:
                L.append(f"| {lab} | {SR.FAMILY[lab]} | " + " | ".join(fmt(r["seeds"][k][c][lab]) for k in ("s0", "s1")) + " |")
    if res.get("instruments"):
        L += ["", "## Instruments (from the tick dumps; beside, never deciding)", "", "| rollout | bind overall | bind in M1 ticks (n) | cap overall | cap in M1 ticks | mean lift where bound, m/s | v_f vs logged: MAE m/s, share v_f > logged |", "|---|---|---|---|---|---|---|"]
        for tag, v in res["instruments"].items():
            vf = v["vf_vs_logged_overall"]
            L.append(f"| {tag} | {sh(v['bind_rate_overall']['rate'])} | {sh(v['bind_rate_in_m1_ticks']['rate'])} ({v['bind_rate_in_m1_ticks']['n']}) | {sh(v['cap_rate_overall']['rate'])} | {sh(v['cap_rate_in_m1_ticks']['rate'])} | "
                     f"{'None' if v['lift_where_bound_overall']['mean_lift_mps'] is None else '%.3f' % v['lift_where_bound_overall']['mean_lift_mps']} | "
                     f"{'None' if vf['mae_mps'] is None else '%.3f' % vf['mae_mps']}, {sh(vf['share_vf_exceeds_logged'])} (n {vf['n']}) |")
    L += ["", "## Controls", "", "| control | PASS |", "|---|---|"] + [f"| {k} | {v.get('PASS')} |" for k, v in res.get("controls", {}).items()]
    L += ["", "## Variance named", ""] + [f"* {k}: {v}" for k, v in res["variance_questions"].items()]
    return "\n".join(x for x in L if x is not None) + "\n"


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("production", "ka-null", "ka-llr"), default="production")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--work-dir", default=WORK_DIR)
    ap.add_argument("--raw", default=f"{PK}/raw")
    ap.add_argument("--arr-dir", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--results-md", default=f"{PK}/RESULTS.md")
    a = ap.parse_args(argv)
    init()
    rr.PK = PK
    rr.WORK = f"{a.work_dir}/rig"
    os.makedirs(a.raw, exist_ok=True)
    arr_dir = a.arr_dir or f"{a.raw}/arrays"
    res = {"card": "PREREG_LEVELFLOOR_A3.md (arms LF050 / LF075; the rules by reference to PREREG_LEVELHEAD_CL.md sections 4-5, CL := LF)", "card_sha256": sha256_file(CARD), "reader": os.path.abspath(__file__).replace("\\", "/"),
           "reader_sha256": sha256_file(__file__), "llr_read_sha256": sha256_file(LLR_READ_PY), "mode": a.mode, "tier": "T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2)",
           "estimator": f"paired episode-cluster bootstrap B={a.n_boot} seed 0 (gapdecomp.decompose); answers another draw of EPISODES only", "clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "strategic_family": rr.STRATEGIC_NOTE, "doses_read": list(DOSES), "disclosure": DISCLOSURE,
           "variance_questions": {"episodes": "answered by the paired episode-cluster bootstrap (every interval here)",
                                  "training_run": "O is C0's own checkpoint in both arms of every LF - C0 contrast, so O's training variance is absent from it; the heads come from one LLR run per seed (not measured); rule F reads against the C0REP floor",
                                  "inference": "the planner and F are deterministic: no inference seed applies"}}
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
            json.dump(res, open(f"{a.raw}/LF_READ.json", "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
            print("REFUSED:", res["refused"])
            return 3
        for tag, _b, _d, s, _sc, _f in rollouts():
            t = np.load(f"{a.out_root}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True)
            inst[tag] = tick_instruments({k: t[k] for k in t.files})
        res["instruments"] = inst
        paths = extract_new(a.out_root, arr_dir, a.workers)
        out_json = f"{a.raw}/LF_READ.json"
    else:
        for s in SEEDS:
            base[s], c = ka_arrays(a.mode, s)
            rctl.update(c)
        res["controls"] = {"reused_arrays": {"PASS": all(v["PASS"] for v in rctl.values()), "files": rctl}}
        if not res["controls"]["reused_arrays"]["PASS"]:
            print("REFUSED: a reused array differs from its pinned sha256")
            return 3
        out_json = f"{a.raw}/KA_READER_{a.mode}.json"
    cache, doses = {}, {}
    for d in DOSES:
        arrays = {}
        for s in SEEDS:
            arrays[s] = dict(base[s])
            if a.mode == "production":
                for b_, _ in (("RILF", 0), ("RILFC", 0), ("LF", 0), ("LFC", 0)):
                    arrays[s][b_] = rr.read_arrays(paths[f"{name_of(b_, d)}_s{s}"])
        reach = reach_control({f"s{s}": inst[f"{name_of('RILF', d)}_s{s}"] for s in SEEDS}) if a.mode == "production" else None
        doses[d] = read_dose(d, arrays, a.n_boot, reach, cache)
    res["doses"] = doses
    res["joint"] = joint_reading({d: {"outcome": doses[d]["outcome"], "m1_moved": doses[d]["m1_moved"]} for d in DOSES})
    if a.mode != "production":
        chk = ka_check(a.mode, doses)
        res["ka_checks"], res["KA_PASS"] = chk, bool(all(chk.values()))
        print(f"KA {a.mode}: {sum(chk.values())}/{len(chk)} checks hold; labels {[doses[d]['outcome']['label'] for d in DOSES]}")
        for k, v in chk.items():
            if not v:
                print("  RED", k)
    json.dump(res, open(out_json, "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
    if a.mode == "production":
        open(a.results_md, "w", encoding="utf-8", newline="\n").write(render_md(res))
        print("HEADLINE", res["joint"]["headline"])
    else:
        open(f"{a.raw}/KA_READER_{a.mode}_RESULTS.md", "w", encoding="utf-8", newline="\n").write(render_md(res))
    return 0 if res.get("KA_PASS", True) else 2


if __name__ == "__main__":
    sys.exit(main())
