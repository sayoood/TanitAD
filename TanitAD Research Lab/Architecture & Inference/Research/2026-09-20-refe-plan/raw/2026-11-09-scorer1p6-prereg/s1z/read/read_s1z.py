#!/usr/bin/env python3
"""read_s1z.py -- the read tool of PREREG_S1Z_P6 (frozen 2026-11-09, sha256 8c7b6b69...efa69) sections 3 to 5. CPU only. It judges ONE arm named S1Z against C0 and P6.

    python read_s1z.py read --c0 DIR --c0r DIR --s1z DIR --out OUT [--smoke-standin] [--control-verdict scorer1p6_read.json]
    python read_s1z.py emit-changed --c0 DIR --c0r DIR --s1z DIR --out OUT       # seam + token json for the harness, for S1Z picks that have no score yet

REUSE, NOT A FORK: the Batch 1 reader read_scorer1p6.py (eval/) does the gates (G0-E2, G0-E3, G0-PICK, G0-P6CSV), the score pool, P1 on E3 with the paired log-cluster bootstrap
(10,000 draws, seed 20260927), F = |C0r - C0|, G1, G2, G3, goal status, the family tables, the Shapley split and the top-K curve. S1Z is handed to it under the name S1E (its
arm slot) and relabelled S1Z in this output. What this file ADDS (card section 4 beyond Batch 1): M2 (failure share), G4 (CRUISE slow-pick share), the S1Z form of the bars
(G3 against P6 only, adoption = P1 and G1 to G4), the Z1 to Z6 / Unnamed outcome map, the K ledger and the Bonferroni interval. Strategic is PAUSED (PI 2026-09-19), never a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = Path(__file__).resolve().parent
P6BASE = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base"
EVAL_DIR = os.environ.get("S1Z_EVAL_DIR", P6BASE + "/eval")
SEL_DIR = P6BASE + "/selectors"
for _p in (EVAL_DIR,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import read_scorer1p6 as R  # noqa: E402  (the Batch 1 reader, imported unchanged)

CARD = P6BASE + "/PREREG_S1Z_P6.md"
CARD_SHA256 = "8c7b6b697b94ccf1a4781a727ed28c10ed8b11878c33eeb2d016a507b36efa69"
P6_RESULT = "E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/p6_read/result/P6_RESULT.md"
SMOKE_ROOT = "F:/tanitad_scratch/s1zread/smoke"
DEFAULT_LEDGER = SEL_DIR + "/ledger_K.json"
DEFAULT_E2_TOKENS = P6BASE + "/e2/e2_tokens.json"
DEFAULT_E2_TABLE = P6BASE + "/e2/e2_table_p6final.npz"

# ------------------------------------------------------------------------------------------------ bars: constants, each with the card line it is copied from (a test compares them to the card text)
B_DRAWS_BAR, SEED_BAR = 10000, 20260927      # PREREG_S1Z_P6.md line 56: "10,000 draws, seed 20260927, 95 % percentile"
FLOOR_FACTOR = 2.0                           # line 60 (P1): "|S1Z - C0| > 2*F"
M2_CI_HI_BAR = 0.0                           # line 61 (M2): "S1Z - P6: CI-hi < 0, and S1Z - C0: CI-hi < 0"
G1_BAR = -0.5                                # line 62 (G1): "each point >= -0.5 against P6"
G2_BAR = 0.5                                 # line 63 (G2): "CI-hi of delta regret against P6 < +0.5 points"
OVERSPEED_RATIO = 1.15                       # line 64 (G3): "plan v4/v0 > 1.15"
G3_BAR = 0.10                                # line 64 (G3): "CI-hi of the difference against P6 < +0.10"
G4_BAR = 0.10                                # line 65 (G4): "CI-hi of the difference against P6 < +0.10"
SLOW_RATIO = 0.85                            # line 65 (G4): "same definition as P6_RESULT.md" -> selc6_analyze.py:42 SLOW_RATIO = 0.85 (see HANDOFF.md)
GOAL = 94.55                                 # line 66 (goal status): "GOAL MET if the point >= 94.55"
BARS = {"bootstrap_draws": B_DRAWS_BAR, "bootstrap_seed": SEED_BAR, "P1_floor_factor_2F": FLOOR_FACTOR, "M2_ci_hi_lt": M2_CI_HI_BAR, "G1_each_point_ge": G1_BAR,
        "G2_ci_hi_lt": G2_BAR, "overspeed_ratio": OVERSPEED_RATIO, "G3_ci_hi_lt": G3_BAR, "G4_ci_hi_lt": G4_BAR, "slow_ratio": SLOW_RATIO, "goal": GOAL}

# the next levers, copied from card section 5 (lines 77 to 83)
NEXT = {
    "Z1": "train S1Z seed 1 as the replicate, then the closed-loop read of PREREG_SCORER1P6 section 6",
    "Z2": ("the Shapley split on S1Z on E2 (zero GPU) names the largest head left; the next lever targets it. Stacking S1Z's logits with an adopted Batch 1 arm or a selector is a counted "
           "candidate under its own card"),
    "Z3_G4": "the same loss with speed-matched pairs only (under its own card)",
    "Z3_G1": "the failing head is removed from L_twin (under its own card)",
    "Z3_other": ("the card names a lever only for a failed G4 or G1; a failed G2 or G3 matches Z3 HARM but has no committed lever: labelled so, and the Shapley split on E2 names the next "
                 "lever (card Unnamed rule)"),
    "Z4": ("the scorer cannot learn the twin distinction from its current inputs at this dose. Next: explicit relative features between proposals, under its own card, and then the "
           "1,600-update dose"),
    "Z5": "NOT PROVEN. Next: S1Z seed 1 and a second C0 replicate",
    "Z6": "the failures fell and the gain was spent elsewhere. Next: the Shapley split on E2",
    "Unnamed": "matches none of Z1 to Z6 (or an input is undefined): labelled as such; the Shapley split on E2 names the next lever",
}
Z_NAMES = {"Z1": "GOAL MET", "Z2": "GAIN, BELOW GOAL", "Z3": "HARM", "Z4": "NULL", "Z5": "SWAMPED", "Z6": "MECHANISM ONLY", "Unnamed": "UNNAMED"}
Z_ORDER = ("Z1", "Z2", "Z3", "Z5", "Z6", "Z4")      # primary label: most specific first (as Batch 1 A7)

TIER = dict(R.TIER)
ESTIMATOR = R.ESTIMATOR
EVIDENCE = R.EVIDENCE
STRATEGIC = R.STRATEGIC
S1Z_AMBIGUITIES = [
    "S1 M2 failure = the pick's PDMS == 0 (the harness score, an exact product with a zero factor) OR its TTC sub-score == 0, on all 12,146 E3 tokens, from the same score pool as P1. "
    "Share is a fraction (0 to 1); the diffs S1Z - P6 and S1Z - C0 use the same paired log-cluster bootstrap draws as P1; the bar CI-hi < 0 is strict with no slack (not a decimal bar).",
    "S2 G4 reuses the Batch 1 slow-pick definition (CRUISE tokens of E2 on the 0.5 s grid, pick's plan v4/v0 < 0.85; Batch 1 A1/A2), on E2's UNIFORM token set. P6_RESULT.md's 0.224 is a "
    "C3 stratum-weighted share on 20-step plans, so it is not the same number; the DEFINITION is shared, the value is read on E2. G4 uses the EPS slack of Batch 1 A12 (as G3).",
    "S3 G3 is read against P6 only (card line 64 has no 'and against C0' clause, unlike Batch 1's S1E). Adoption = P1 and G1 to G4.",
    "S4 P1 = both lines of the card row: S1Z vs C0 (CI-lo > 0 and |S1Z - C0| > 2F) AND S1Z vs P6 (CI-lo > 0). F = |C0r - C0| on PDMS x100 on E3 (Batch 1 A5); C0r is REQUIRED here.",
    "S5 Z-outcomes overlap as Batch 1's O-outcomes did (A7): every match is listed; the primary label follows Z1, Z2, Z3, Z5, Z6, Z4, Unnamed. Z3 = (P1 or M2 clears) and a guard fails. "
    "Z5 = S1Z vs C0 has CI-lo > 0 but |diff| <= 2F. Z6 = M2 holds, P1 fails, and NO guard failed ('no guard caused it'). Unnamed is reached only when an input is undefined "
    "(every defined combination of P1, M2, guards and the 2F test lands in Z1 to Z6).",
    "S6 Z3 lever: G4 failed -> speed-matched pairs; G1 failed -> remove the failing head(s) from L_twin; G2 or G3 failed alone -> no committed lever (stated, Shapley split on E2).",
    "S7 The Bonferroni interval is printed for the PDMS differences S1Z - P6 and S1Z - C0 at the ledger's running K (selcore.pct_pair). It is for best-of-K statements only; P1 is the arm's own test.",
]
NOT_COMPUTED = {
    "head_within_token_AUC_E2": "not computed: the card gives no pooling rule for a within-token AUC (per-token AUC then mean? tokens with one class?); computing one would invent the estimand",
    "near_twin_AUC_E2": "not computed: needs the section 1 pair construction on raw poses (ADE <= 1.0 m among valid proposals, hard negatives of arm B's rule) from the E2 table's proposals; the table carries truth, not the pair list",
}


# ------------------------------------------------------------------------------------------------ small helpers
def sha256_file(p):
    return R.sha256_file(p)


def free_ram_gb():
    import ctypes

    class MEM(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    m = MEM()
    m.dwLength = ctypes.sizeof(MEM)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def wait_free_ram(min_gb=3.0, timeout_s=1800):
    t0 = time.time()
    while True:
        try:
            f = free_ram_gb()
        except Exception:     # noqa: BLE001 -- non-Windows: no gate
            return None
        if f >= min_gb:
            return f
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"REFUSING: free RAM {f:.2f} GB < {min_gb} GB after {timeout_s}s")
        time.sleep(15)


def tag(d, estimator, tier, n, artifact):
    """every reported block carries estimator, tier, n and artifact path"""
    d = dict(d)
    d.update(estimator=estimator, tier=tier, n=n, artifact=artifact)
    return d


# ------------------------------------------------------------------------------------------------ pure bar and outcome functions (tested at exact boundaries)
def m2_clears(hi_vs_p6, hi_vs_c0):
    """M2: S1Z - P6 CI-hi < 0 AND S1Z - C0 CI-hi < 0 (strict, no slack)"""
    if hi_vs_p6 is None or hi_vs_c0 is None:
        return None
    return bool(hi_vs_p6 < M2_CI_HI_BAR and hi_vs_c0 < M2_CI_HI_BAR)


def g4_ok(hi):
    """G4: CI-hi of (S1Z - P6) slow-pick share < +0.10 (EPS slack in the arm's disfavour, as G3 in Batch 1)"""
    return R.lt(hi, G4_BAR)


def swamped_vs_c0(lo, diff, floor):
    """Z5: a CI-lo > 0 inside 2F (the P1 test against C0 with the floor rule failing)"""
    if floor is None or lo is None or diff is None:
        return None
    return bool(lo > 0 and not abs(diff) > FLOOR_FACTOR * floor)


def decide_outcome(st):
    """st: p1 (bool|None), m2 (bool|None), guards {"G1": bool, "G2": bool, "G3": bool, "G4": bool}, g1_heads_failed [names], swamped (bool|None), point (float|None).
    -> dict(matches, primary, name, next_levers, adopted)"""
    p1, m2, sw, gd = st.get("p1"), st.get("m2"), st.get("swamped"), st.get("guards") or {}
    if p1 is None or m2 is None or sw is None or any(v is None for v in gd.values()) or not gd:
        return {"matches": ["Unnamed"], "primary": "Unnamed", "name": Z_NAMES["Unnamed"], "adopted": False,
                "next_levers": {"Unnamed": NEXT["Unnamed"]}, "why": "an input is undefined (F, M2 or a guard): the card's Unnamed rule"}
    failed = [k for k in ("G1", "G2", "G3", "G4") if not gd[k]]
    guards_ok = not failed
    adopted = bool(p1 and guards_ok)
    pt = st.get("point")
    m = []
    if adopted and pt is not None and R.ge(pt, GOAL):
        m.append("Z1")
    if adopted and not (pt is not None and R.ge(pt, GOAL)):
        m.append("Z2")
    if (p1 or m2) and failed:
        m.append("Z3")
    if sw:
        m.append("Z5")
    if m2 and not p1 and guards_ok:
        m.append("Z6")
    if not p1 and not m2:
        m.append("Z4")
    prim = next((z for z in Z_ORDER if z in m), "Unnamed")
    levers = {}
    for z in (m or ["Unnamed"]):
        if z == "Z3":
            parts = []
            if "G4" in failed:
                parts.append("G4 failed: " + NEXT["Z3_G4"])
            if "G1" in failed:
                heads = st.get("g1_heads_failed") or []
                parts.append("G1 failed (" + ", ".join(heads) + "): " + NEXT["Z3_G1"])
            if not parts:
                parts.append(NEXT["Z3_other"])
            levers[z] = "; ".join(parts)
        else:
            levers[z] = NEXT[z]
    return {"matches": m or ["Unnamed"], "primary": prim, "name": Z_NAMES[prim], "adopted": adopted, "failed_guards": failed, "next_levers": levers}


# ------------------------------------------------------------------------------------------------ inputs
def make_inputs(a):
    """C0, C0r, S1Z arm dirs -> read_scorer1p6.Inputs with arms keyed C0, C0r, S1Z"""
    ns = argparse.Namespace(p6_proposals=a.p6_proposals, p6_csv=a.p6_csv, e2_tokens=a.e2_tokens, e2_table=a.e2_table, export=a.export, world_dump=a.world_dump,
                            arm=[f"C0={a.c0}", f"C0r={a.c0r}", f"S1Z={a.s1z}"], ens3_csv=None, c3_tokens=a.c3_tokens, no_tactical=a.no_tactical)
    return R.inputs_from_files(ns)


def _as_s1e_slot(inp):
    """the Batch 1 reader has fixed arm slots (C0, C0r, S1E, S1): S1Z occupies the S1E slot for its calls; nothing else is changed"""
    arms = {"C0": inp.arms["C0"], "C0r": inp.arms["C0r"], "S1E": inp.arms["S1Z"]}
    return R.Inputs(inp.world, inp.p6_logits, inp.p6_proposals, inp.p6_rows, inp.e2_tokens, inp.e2_table, arms, inp.p6_pick_banked, inp.p6_fingerprint, inp.c3_strata,
                    inp.tactical, inp.provenance, inp.p6_token)


def _bool(x):
    return None if x is None else bool(x)


# ------------------------------------------------------------------------------------------------ the read
def s1z_read(inp, smoke=False, ledger_path=None, ledger_key_extra="", control=None):
    """inp.arms must hold C0, C0r and S1Z. -> result dict; status READ, READ STOPPED (a gate failed, no verdict) or INCOMPLETE (scores missing)."""
    t0 = time.time()
    res = {"tool": "read_s1z.py", "card": {"path": CARD, "sha256_expected": CARD_SHA256, "sha256_actual": sha256_file(CARD) if os.path.exists(CARD) else None},
           "tier": TIER, "evidence_class": EVIDENCE, "estimator": ESTIMATOR, "bars": BARS, "strategic": STRATEGIC, "ambiguities_batch1": R.AMBIGUITIES, "ambiguities_s1z": S1Z_AMBIGUITIES,
           "arm": "S1Z" + (" (SMOKE STAND-IN: the arm dir is S1E's)" if smoke else ""), "inputs": inp.provenance}
    if smoke:
        res["stamp"] = "SMOKE_NOT_A_RESULT"
    if res["card"]["sha256_actual"] != CARD_SHA256:
        res.update(status="READ STOPPED", why="the card's sha256 differs from the frozen one; the bars may have moved")
        return res
    for need in ("C0", "C0r", "S1Z"):
        if need not in inp.arms:
            res.update(status="READ STOPPED", why=f"arm {need} is required (C0 and C0r give P1's zero point and the floor F; S1Z is the arm)")
            return res
    res["code_sha256"] = {"read_s1z.py": sha256_file(HERE / "read_s1z.py"), "read_scorer1p6.py": sha256_file(Path(EVAL_DIR) / "read_scorer1p6.py")}
    inp2 = _as_s1e_slot(inp)
    base = R.run_read(inp2, need_ens3=False)
    res["gates"] = base.get("gates", {})
    if base.get("status") != "READ":
        res.update(status=base.get("status"), why=base.get("why"), missing_tokens=base.get("missing_tokens"))
        return res

    W, N = inp2.world, len(inp2.world.token)
    ep = base["endpoints"]
    e = ep["S1E"]
    verdict1 = base["verdict"]
    F = verdict1["F_pdms_E3"]
    arms = {k: v for k, v in inp2.arms.items() if not k.startswith("_")}

    # ---- picks and the score pool (the Batch 1 pool class and recipe; run_read does not expose its arrays)
    p6_pick = R.pick_logits(inp2.p6_logits)
    L3, pk3 = {}, {"P6": p6_pick}
    for nm, a in arms.items():
        L3[nm] = np.asarray(a.e3_logits)[a._ix]
        pk3[nm] = R.pick_logits(L3[nm])
    pool = R.Pool(N)
    pool.add_rows(inp2.p6_rows, W, p6_pick, "p6 csv")
    tpos = {t: i for i, t in enumerate(inp2.e2_table["token"])}
    for t in inp2.e2_tokens:
        i, ti = W.index[t], tpos[t]
        for k in range(R.M):
            pool.put(i, k, inp2.e2_table["pdms"][ti, k], inp2.e2_table["sub"][ti, k], "e2 table")
    for nm, a in arms.items():
        pool.add_rows(a.rows, W, pk3[nm], "csv " + nm)
    if pool.conflicts or len(pool.missing(pk3["S1E"])) or len(pool.missing(pk3["C0"])):
        res.update(status="READ STOPPED", why="score pool inconsistent on the second build")
        return res
    r = np.arange(N)
    pd3 = {nm: pool.pdms[r, pk3[nm]] for nm in pk3}
    ttc3 = {nm: pool.sub[r, pk3[nm], R.ALL_HEADS["TTC"]] for nm in pk3}
    fail3 = {nm: ((pd3[nm] <= 0.0) | (ttc3[nm] <= 0.0)).astype(np.float64) for nm in pk3}
    B3 = R.BootSet(W.log)
    ctl = abs(float(pd3["S1E"].mean() * 100.0) - e["E3_pdms"]["point"])
    res["internal_check_pool_pdms_vs_run_read_abs_diff"] = ctl
    if ctl > 1e-9:
        res.update(status="READ STOPPED", why=f"the rebuilt score pool disagrees with run_read's PDMS ({ctl})")
        return res

    art3 = "S1Z e3_logits.npz + e3_changed_scores.csv; P6 refe_p6final.csv; E3 = navtest 12,146 tokens"
    # ---- M2
    def lvl(nm):
        pt, rr, n = B3.stat(fail3[nm])
        return R.rec(pt, rr, n)

    pt_p, rr_p, n_p = B3.diff(fail3["S1E"], fail3["P6"])
    pt_c, rr_c, n_c = B3.diff(fail3["S1E"], fail3["C0"])
    d_p6, d_c0 = R.rec(pt_p, rr_p, n_p), R.rec(pt_c, rr_c, n_c)
    m2 = m2_clears(d_p6["hi"], d_c0["hi"])
    m2_blk = tag({"definition": "share of E3 tokens whose pick has PDMS 0 or TTC sub-score 0", "S1Z_share": lvl("S1E"), "P6_share": lvl("P6"), "C0_share": lvl("C0"),
                  "S1Z_minus_P6": d_p6, "S1Z_minus_C0": d_c0, "bar": "CI-hi < 0 for both differences", "S1Z_minus_P6_clears": bool(d_p6["hi"] < M2_CI_HI_BAR),
                  "S1Z_minus_C0_clears": bool(d_c0["hi"] < M2_CI_HI_BAR), "clears": m2,
                  "n_pick_pdms_zero": {nm: int((pd3[nm] <= 0).sum()) for nm in ("S1E", "P6", "C0")}, "n_pick_ttc_zero": {nm: int((ttc3[nm] <= 0).sum()) for nm in ("S1E", "P6", "C0")}},
                 "paired log-cluster bootstrap, 10,000 draws, seed 20260927 (same draws as P1)", TIER["E3"], N, art3)
    m2_blk["n_pick_pdms_zero"] = {("S1Z" if k == "S1E" else k): v for k, v in m2_blk["n_pick_pdms_zero"].items()}
    m2_blk["n_pick_ttc_zero"] = {("S1Z" if k == "S1E" else k): v for k, v in m2_blk["n_pick_ttc_zero"].items()}

    # ---- E2 quantities for G4 (the Batch 1 recipe: arm e2 logits if given, else its E3 rows)
    wi2 = np.array([W.index[t] for t in inp2.e2_tokens])
    n2 = len(wi2)
    a_s = arms["S1E"]
    if a_s.e2_logits is not None:
        j, err = R._align(a_s.e2_token, list(inp2.e2_tokens), "S1Z e2 logits")
        if err:
            res.update(status="READ STOPPED", why=err)
            return res
        L2s = np.asarray(a_s.e2_logits)[j]
    else:
        L2s = L3["S1E"][wi2]
    pk2 = {"S1E": R.pick_logits(L2s), "P6": p6_pick[wi2]}
    ratio2 = R.plan_v4(inp2.p6_proposals[wi2]) / np.maximum(W.v0[wi2], 1e-6)[:, None]
    cru2 = W.cruise()[wi2]
    r2 = np.arange(n2)
    slow = {nm: (ratio2[r2, pk2[nm]] < SLOW_RATIO).astype(np.float64) for nm in pk2}
    B2 = R.BootSet(W.log[wi2])
    pt, rr, n = B2.diff(slow["S1E"], slow["P6"], cru2)
    g4d = R.rec(pt, rr, n)
    g4 = g4_ok(g4d["hi"])
    slow_s, slow_p = R.fin(B2.stat(slow["S1E"], cru2)[0]), R.fin(B2.stat(slow["P6"], cru2)[0])
    chk = abs(slow_s - e["slow_pick_share_cruise"]) if slow_s is not None and e.get("slow_pick_share_cruise") is not None else None
    g4_blk = tag({"definition": "CRUISE tokens of E2: share whose pick's plan v4/v0 < 0.85 (Batch 1 definition; P6_RESULT.md's slow pick; see HANDOFF.md)", "S1Z_share": slow_s, "P6_share": slow_p,
                  "S1Z_minus_P6": g4d, "bar": "CI-hi of the difference against P6 < +0.10", "ok": g4, "n_cruise_tokens_E2": int(cru2.sum()),
                  "internal_check_vs_run_read_slow_share_abs_diff": chk}, "paired log-cluster bootstrap over E2's logs, 10,000 draws, seed 20260927", TIER["E2"], int(cru2.sum()),
                 "S1Z e3_logits.npz rows of the 2,000 E2 tokens; P6 proposals; E2 table " + str(inp2.provenance.get("e2_table", ["?"])[0]))
    if chk is not None and chk > 1e-12:
        res.update(status="READ STOPPED", why=f"slow share disagrees with run_read ({chk})")
        return res

    # ---- P1, G1, G2, G3, goal (all from the Batch 1 reader's S1E-slot numbers; same code, same draws)
    p1_c0 = bool(e["P1_vs_C0"]["clears"])
    p1_p6 = bool(e["P1_vs_P6"]["clears_ci_lo_gt_0"])
    p1 = bool(p1_c0 and p1_p6)
    g1d = {s: e["E3_subscores_vs_P6"][s] for s in ("TTC", "NC", "DAC")}
    g1 = {s: R.g1_ok(g1d[s]["point"]) for s in g1d}
    g2d = e["G2_delta_regret_vs_P6"]
    g2 = R.g2_ok(g2d["hi"])
    g3d = e["G3_overspeed_diff_vs_P6"]
    g3 = R.g3_ok(g3d["hi"])
    guards = {"G1": all(g1.values()), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4)}
    swamped = swamped_vs_c0(e["P1_vs_C0"]["lo"], e["P1_vs_C0"]["point"], F)
    goal = e["goal_status"]

    # ---- Bonferroni interval at K
    K, kinfo = None, {"ledger": None}
    pt, rr, n = B3.diff(pd3["S1E"] * 100.0, pd3["P6"] * 100.0)
    rr_p6 = rr
    pt, rr, n = B3.diff(pd3["S1E"] * 100.0, pd3["C0"] * 100.0)
    rr_c0 = rr
    if ledger_path:
        if SEL_DIR not in sys.path:
            sys.path.insert(0, SEL_DIR)
        import selcore
        led = selcore.Ledger(ledger_path)
        sha_logits = inp.provenance["arms"]["S1Z"][1] if "S1Z" in (inp.provenance.get("arms") or {}) else "no-provenance"
        key = f"{'SMOKE:' if smoke else ''}S1Z:{sha_logits}:E3{ledger_key_extra}"
        added = led.add(key, nominee="S1Z", family="S1Z", set="E3", smoke=bool(smoke))
        K = led.K
        lo_p, hi_p = selcore.interval(rr_p6[np.isfinite(rr_p6)], K)
        lo_c, hi_c = selcore.interval(rr_c0[np.isfinite(rr_c0)], K)
        kinfo = {"ledger": ledger_path, "K": K, "this_read_added": bool(added), "key": key,
                 "bonferroni_pdms_S1Z_minus_P6": {"point": float(e["E3_pdms_vs_P6"]["point"]), "lo": lo_p, "hi": hi_p},
                 "bonferroni_pdms_S1Z_minus_C0": {"point": float(e["P1_vs_C0"]["point"]), "lo": lo_c, "hi": hi_c},
                 "note": "K counts every navtest read in the programme's ledger (selectors/ledger_K.json); Batch 1 reads (C0, C0r, S1E) are in it only if they were written there (UNVERIFIED); best-of-K statements use these intervals, P1 is the arm's own test"}
    else:
        kinfo = {"ledger": None, "K": None, "note": "no ledger path given: K NOT COUNTED"}

    # ---- endpoint blocks with provenance
    p1_blk = tag({"PDMS_x100_E3": e["E3_pdms"], "S1Z_vs_C0": {**e["P1_vs_C0"], "bar": "CI-lo > 0 and |diff| > 2F", "F": F, "2F": None if F is None else 2 * F},
                  "S1Z_vs_P6": {**e["E3_pdms_vs_P6"], "bar": "CI-lo > 0", "clears": p1_p6}, "clears": p1,
                  "F_definition": "|C0r - C0| on PDMS x100, E3 (Batch 1 replicate; F is training variance, the bootstrap is blind to it)"},
                 "paired log-cluster bootstrap, 10,000 draws, seed 20260927", TIER["E3"], N, art3)
    g_blk = {
        "G1": tag({"per_head_points_vs_P6": g1d, "ok_each": g1, "bar": "each point >= -0.5", "ok": guards["G1"], "failed_heads": [s for s, v in g1.items() if not v]},
                  "paired log-cluster bootstrap (CIs shown; the bar reads the point)", TIER["E3"], N, art3),
        "G2": tag({"delta_regret_vs_P6": g2d, "bar": "CI-hi < +0.5 points", "ok": guards["G2"]}, "paired log-cluster bootstrap over E2's logs", TIER["E2"], n2, "E2 table of P6's 64 proposals"),
        "G3": tag({"overspeed_diff_vs_P6": g3d, "bar": "CI-hi < +0.10 (against P6 only)", "ok": guards["G3"]}, "paired log-cluster bootstrap over E2's logs, CRUISE tokens", TIER["E2"],
                  g3d.get("n"), "E2 table; plan v4/v0 > 1.15"),
        "G4": g4_blk}

    # ---- family clause (Batch 1 tables, arm S1Z)
    fam1 = base["family_clause"]["per_arm_vs_P6"]["S1E"]
    adv = fam1["adverse_components"]
    family = tag({"STRATEGIC": STRATEGIC, "components_vs_P6": fam1["components"], "adverse_components": adv,
                  "verdict_if_adopted": "ADOPT" if not adv else "PDMS GAIN WITH COST",
                  "rule": "none adverse is ADOPT; one or more adverse is PDMS GAIN WITH COST, every adverse component named; never described as the better driver"},
                 "paired log-cluster bootstrap per component (arm - P6), no multiplicity correction (card silent); adverse iff the CI lies entirely on the worse side", TIER["E3"], N, art3)

    st = dict(p1=p1, m2=m2, guards=guards, g1_heads_failed=g_blk["G1"]["failed_heads"], swamped=swamped, point=e["E3_pdms"]["point"])
    out = decide_outcome(st)
    out["inputs"] = {"P1_vs_C0": p1_c0, "P1_vs_P6": p1_p6, "P1": p1, "M2": m2, "guards": guards, "swamped": swamped, "point": st["point"], "goal_status": goal}
    adopted = out["adopted"]

    reported = base["reported_not_deciding"].get("S1E", {})
    rep = tag({"EP_subscore_vs_P6": e["E3_subscores_vs_P6"]["EP"], "other_E3_subscores_vs_P6": {s: e["E3_subscores_vs_P6"][s] for s in ("NC", "DAC", "TTC", "C", "DDC")},
               "shapley_split_E2": reported.get("shapley_E2"), "topK_oracle_curve_E2_pdms_x100": reported.get("topK_oracle_E2_pdms_x100"),
               "each_head_dRho_E2_CRUISE": reported.get("each_head_dRho"), "EP_dRho_E2_CRUISE": reported.get("EP_dRho"), "not_computed": NOT_COMPUTED},
              "see each item (E3 sub-scores: paired log-cluster bootstrap; Shapley / top-K: point estimates)", "E3 sub-scores; E2 for the rest", None, art3)

    res.update(status="READ", gates=base["gates"], P1=p1_blk, M2=m2_blk, guards=g_blk, goal_status=tag({"PDMS_x100_E3": e["E3_pdms"], "status": goal, "bar": "GOAL MET if point >= 94.55; GOAL NOT EXCLUDED if CI-hi >= 94.55; else BELOW GOAL"},
                                                                                                      "paired log-cluster bootstrap", TIER["E3"], N, art3),
               adoption={"adopted": adopted, "rule": "S1Z if it clears P1 and G1 to G4", "P1": p1, "guards": guards}, family_clause=family, outcome=out,
               reported_not_deciding=rep, K_ledger=kinfo, verdict_label=(("ADOPT" if not adv else "PDMS GAIN WITH COST") if adopted else "NO ADOPTION"),
               E3_pick_changed_vs_P6=e["E3_pick_changed_vs_P6"], seconds=round(time.time() - t0, 1))
    res["batch1_endpoints_S1Z_slot"] = {k: e[k] for k in ("E3_pdms", "E3_pdms_vs_P6", "P1_vs_C0", "E3_subscores_vs_P6", "G2_delta_regret_vs_P6", "G3_overspeed_diff_vs_P6", "goal_status",
                                                         "slow_pick_share_cruise")}
    if control:
        res["control"] = compare_control(res, control)
    return res


# ------------------------------------------------------------------------------------------------ control: S1E stand-in numbers equal the Batch 1 verdict's exactly
CONTROL_FIELDS = ("E3_pdms", "E3_pdms_vs_P6", "P1_vs_C0", "E3_subscores_vs_P6", "G2_delta_regret_vs_P6", "G3_overspeed_diff_vs_P6", "goal_status", "slow_pick_share_cruise")


def _flat(o, path=""):
    if isinstance(o, dict):
        for k in sorted(o):
            yield from _flat(o[k], f"{path}/{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from _flat(v, f"{path}[{i}]")
    else:
        yield path, o


def compare_control(res, verdict_path):
    """P1, G1 to G3 of the stand-in against scorer1p6_read.json's S1E endpoints: every leaf must be exactly equal (after the same json round trip)"""
    ref = json.load(open(verdict_path, encoding="utf-8"))["endpoints"]["S1E"]
    mine = json.loads(json.dumps(res["batch1_endpoints_S1Z_slot"], default=R.jdefault))
    n, bad = 0, []
    for f in CONTROL_FIELDS:
        a, b = dict(_flat(mine[f], f)), dict(_flat(ref[f], f))
        for k in sorted(set(a) | set(b)):
            n += 1
            if a.get(k, "MISSING") != b.get(k, "MISSING"):
                bad.append((k, a.get(k, "MISSING"), b.get(k, "MISSING")))
    # the bar decisions too
    dec = {"P1_clears(batch1)": ref.get("P1_clears")}
    mine_p1 = res["P1"]["clears"]
    if mine_p1 != ref.get("P1_clears"):
        bad.append(("P1.clears", mine_p1, ref.get("P1_clears")))
    for s in ("TTC", "NC", "DAC"):
        if res["guards"]["G1"]["ok_each"][s] != ref["guards"]["G1_each_head_ge_-0.5"][s]:
            bad.append((f"G1.{s}", res["guards"]["G1"]["ok_each"][s], ref["guards"]["G1_each_head_ge_-0.5"][s]))
    if res["guards"]["G2"]["ok"] != ref["guards"]["G2_ci_hi_lt_0.5"]:
        bad.append(("G2", res["guards"]["G2"]["ok"], ref["guards"]["G2_ci_hi_lt_0.5"]))
    if res["guards"]["G3"]["ok"] != ref["guards"]["G3_vs_P6_ci_hi_lt_0.10"]:
        bad.append(("G3", res["guards"]["G3"]["ok"], ref["guards"]["G3_vs_P6_ci_hi_lt_0.10"]))
    return {"verdict_path": verdict_path, "n_leaves_compared": n, "n_mismatch": len(bad), "mismatches": bad[:10], "EQUAL": len(bad) == 0, "decisions_compared": dec}


# ------------------------------------------------------------------------------------------------ markdown
def to_markdown(res):
    sm = res.get("stamp") == "SMOKE_NOT_A_RESULT"
    L = []
    if sm:
        L += ["**SMOKE_NOT_A_RESULT** -- the arm below is S1E standing in for S1Z; no number here is a result and none may be quoted as one.", ""]
    L += ["# S1Z read", "", f"**Status: {res.get('status')}**" + (" (SMOKE_NOT_A_RESULT)" if sm else ""), "",
          f"- Card: `{res['card']['path']}` sha256 `{res['card']['sha256_actual']}` (frozen `{res['card']['sha256_expected']}`)", f"- Tier: E3 = {TIER['E3']}; E2 = {TIER['E2']}.",
          f"- Evidence class: {EVIDENCE}.", f"- Estimator: {ESTIMATOR}", ""]
    L += ["## Gates", ""]
    for k, g in (res.get("gates") or {}).items():
        L.append(f"- {k}: **{'PASS' if g['PASS'] else 'FAIL'}** " + ", ".join(f"{a}={b}" for a, b in g.items() if a not in ("PASS", "name")))
    if res.get("status") != "READ":
        L += ["", f"**{res.get('status')}: {res.get('why')}**", "", "No verdict is written."]
        return "\n".join(L) + "\n"
    ci = R.ci_s
    o = res["outcome"]
    L += ["", "## Verdict", "", f"- Outcome: **{o['primary']} {o['name']}** (all matches: {', '.join(o['matches'])})", f"- Next lever: {o['next_levers'][o['primary']]}",
          f"- Adopted: **{res['adoption']['adopted']}**; verdict: **{res['verdict_label']}**" + (f"; adverse: {', '.join(res['family_clause']['adverse_components'])}" if res["family_clause"]["adverse_components"] else ""),
          f"- Goal status: **{res['goal_status']['status']}**", f"- K ledger: **K = {res['K_ledger'].get('K')}** ({res['K_ledger'].get('note')})", ""]
    p = res["P1"]
    L += ["## Bars", "", "| id | value | bar | result |", "|---|---|---|---|",
          f"| P1 PDMS x100 E3 | S1Z {R.lvl_s(p['PDMS_x100_E3'])} | vs C0: CI-lo>0 and abs>2F (F={R.fmt(p['S1Z_vs_C0']['F'])}); vs P6: CI-lo>0 | vs C0 {ci(p['S1Z_vs_C0'])} {p['S1Z_vs_C0']['clears']}; vs P6 {ci(p['S1Z_vs_P6'])} {p['S1Z_vs_P6']['clears']}; **P1 {p['clears']}** |"]
    m = res["M2"]
    L.append(f"| M2 failure share E3 | S1Z {m['S1Z_share']['point']:.4f}, P6 {m['P6_share']['point']:.4f}, C0 {m['C0_share']['point']:.4f} | CI-hi<0 vs P6 and vs C0 | vs P6 {ci(m['S1Z_minus_P6'])}; vs C0 {ci(m['S1Z_minus_C0'])}; **M2 {m['clears']}** |")
    g = res["guards"]
    g1txt = ", ".join("%s %+.3f" % (s_, v_["point"]) for s_, v_ in g["G1"]["per_head_points_vs_P6"].items())
    L.append(f"| G1 TTC/NC/DAC pts E3 | {g1txt} | each >= -0.5 | **{g['G1']['ok']}** |")
    L.append(f"| G2 delta regret E2 | {ci(g['G2']['delta_regret_vs_P6'])} | CI-hi<+0.5 | **{g['G2']['ok']}** |")
    L.append(f"| G3 overspeed E2 | {ci(g['G3']['overspeed_diff_vs_P6'])} | CI-hi<+0.10 vs P6 | **{g['G3']['ok']}** |")
    L.append(f"| G4 slow-pick E2 | {ci(g['G4']['S1Z_minus_P6'])} (S1Z {R.fmt(g['G4']['S1Z_share'])}, P6 {R.fmt(g['G4']['P6_share'])}) | CI-hi<+0.10 vs P6 | **{g['G4']['ok']}** |")
    L += ["", "## Family clause (arm - P6, paired CIs)", "", f"- adverse components: {res['family_clause']['adverse_components'] or 'none'}", f"- STRATEGIC: {STRATEGIC}", ""]
    r = res["reported_not_deciding"]
    L += ["## Reported, not deciding", "", f"- EP sub-score vs P6 (E3): {ci(r['EP_subscore_vs_P6'])}", f"- Shapley split E2: {json.dumps(r['shapley_split_E2'], default=R.jdefault)}",
          f"- top-K oracle curve E2: {r['topK_oracle_curve_E2_pdms_x100']}"] + [f"- {k}: {v}" for k, v in r["not_computed"].items()]
    if "control" in res:
        c = res["control"]
        L += ["", "## Control (S1E stand-in against the Batch 1 verdict)", "", f"- leaves compared {c['n_leaves_compared']}, mismatches {c['n_mismatch']}: **{'EQUAL' if c['EQUAL'] else 'NOT EQUAL'}**"]
    L += ["", "## Ambiguities (S1Z)", ""] + [f"- {a}" for a in res["ambiguities_s1z"]]
    return "\n".join(L) + "\n"


def write_outputs(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    jp, mp = os.path.join(out_dir, "s1z_read.json"), os.path.join(out_dir, "s1z_read.md")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=1, default=R.jdefault)
    open(mp, "w", encoding="utf-8", newline="\n").write(to_markdown(res))
    return jp, mp


# ------------------------------------------------------------------------------------------------ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["read", "emit-changed"])
    ap.add_argument("--c0", required=True)
    ap.add_argument("--c0r", required=True)
    ap.add_argument("--s1z", required=True, help="arm dir: e3_logits.npz [e2_logits.npz] [e3_changed_scores.csv]")
    ap.add_argument("--p6-proposals", default=R.DEFAULT_P6_PROPOSALS)
    ap.add_argument("--p6-csv", default=R.DEFAULT_P6_CSV)
    ap.add_argument("--e2-tokens", default=DEFAULT_E2_TOKENS)
    ap.add_argument("--e2-table", default=DEFAULT_E2_TABLE)
    ap.add_argument("--export", default=R.DEFAULT_EXPORT)
    ap.add_argument("--world-dump", default=None)
    ap.add_argument("--c3-tokens", default=None)
    ap.add_argument("--no-tactical", action="store_true")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER, help="the programme's K ledger (selectors/ledger_K.json); a successful real read adds its key")
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--smoke-standin", action="store_true", help="--s1z is S1E's dir standing in for S1Z; outputs only under " + SMOKE_ROOT + ", all stamped SMOKE_NOT_A_RESULT")
    ap.add_argument("--control-verdict", default=None, help="Batch 1 scorer1p6_read.json: the stand-in's P1 and G1 to G3 numbers must equal its S1E numbers exactly")
    ap.add_argument("--min-free-ram-gb", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    out = os.path.abspath(a.out).replace("\\", "/")
    if a.smoke_standin and not out.startswith(SMOKE_ROOT):
        raise SystemExit(f"REFUSING: --smoke-standin writes only under {SMOKE_ROOT}")
    if not a.smoke_standin and out.startswith(SMOKE_ROOT):
        raise SystemExit("REFUSING: a real read must not write under the smoke folder")
    wait_free_ram(a.min_free_ram_gb)
    inp = make_inputs(a)
    if a.cmd == "emit-changed":
        os.makedirs(a.out, exist_ok=True)
        res = R.emit_changed(inp, ["S1Z"], a.out, ens3=False)
        print(json.dumps(res, indent=1, default=R.jdefault))
        return 0
    ledger = None if a.no_ledger else (os.path.join(a.out, "smoke_ledger_K.json") if a.smoke_standin else a.ledger)
    res = s1z_read(inp, smoke=a.smoke_standin, ledger_path=ledger, control=a.control_verdict)
    jp, mp = write_outputs(res, a.out)
    print(json.dumps({"status": res.get("status"), "why": res.get("why"), "outcome": res.get("outcome", {}).get("primary"), "K": res.get("K_ledger", {}).get("K"),
                      "control": res.get("control", {}).get("EQUAL"), "stamp": res.get("stamp"), "json": jp, "md": mp}, indent=1, default=R.jdefault))
    return 0 if res.get("status") == "READ" else 2


if __name__ == "__main__":
    sys.exit(main())
