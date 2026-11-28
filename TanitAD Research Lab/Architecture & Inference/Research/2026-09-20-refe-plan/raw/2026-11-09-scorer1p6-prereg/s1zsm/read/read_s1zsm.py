#!/usr/bin/env python3
"""read_s1zsm.py -- the read tool of PREREG_S1ZSM_P6 (frozen 2026-11-10, sha256 ac75b289...d6b8) sections 3 to 5. CPU only. It judges ONE arm named S1ZSM against C0, P6 and S1Z.

    python read_s1zsm.py read --c0 DIR --c0r DIR --s1z DIR --s1zsm DIR --out OUT [--rep-read rep_read.json] [--run-log pair_weight.json|run.jsonl] [--no-ledger]
    python read_s1zsm.py emit-changed --c0 DIR --c0r DIR --s1z DIR --s1zsm DIR --out OUT   # seam + token json for the harness, for S1ZSM picks that have no score yet

REUSE, NOT A FORK: the S1Z reader read_s1z.py (which itself imports the Batch 1 reader read_scorer1p6.py unchanged) does the gates (G0-E2, G0-E3, G0-PICK, G0-P6CSV), P1 (vs C0 with
the 2F rule, vs P6), M2's failure-share draws, G1 to G4, goal status, the family clause, the Shapley split and the top-K curve. The S1ZSM arm is handed to s1z_read in its S1Z slot and
relabelled S1ZSM in this output. What this file ADDS (card sections 4 and 5 beyond S1Z): the S1ZSM form of M2 (S1ZSM - P6 only, no C0 clause), M3 (EP sub-score x100 on E3, S1ZSM - S1Z,
paired, CI-lo > 0), the SM1..SM6/Unnamed outcome map, the reported-not-deciding items against S1Z, the run's per-head pair shares, the replicate read's s_p and t*s_p, the card check
against the SM card, and the K ledger write. Strategic is PAUSED (PI 2026-09-19), never a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = Path(__file__).resolve().parent
P6BASE = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base"
S1Z_READ_DIR = os.environ.get("S1ZSM_S1Z_READ_DIR", P6BASE + "/s1z/read")
SEL_DIR = P6BASE + "/selectors"
if S1Z_READ_DIR not in sys.path:
    sys.path.insert(0, S1Z_READ_DIR)
import read_s1z as S  # noqa: E402  (the S1Z reader, imported unchanged; it puts the Batch 1 eval dir on sys.path)

R = S.R

CARD = P6BASE + "/PREREG_S1ZSM_P6.md"
CARD_SHA256 = "ac75b2891e794de800102a6d5a0f0353e8092e5f976f31b543ac7700811ad6b8"
SMOKE_ROOT = "F:/tanitad_scratch/s1zsmread/smoke"
DEFAULT_LEDGER = SEL_DIR + "/ledger_K.json"
EP_COL = R.SUBN.index("EP")

# ------------------------------------------------------------------------------------------------ bars (a test compares each to the card text)
M2_CI_HI_BAR = 0.0          # M2 row: "S1ZSM - P6: CI-hi < 0"
M3_CI_LO_BAR = 0.0          # M3 row: "S1ZSM - S1Z: CI-lo > 0"
BARS = dict(S.BARS)
BARS.pop("M2_ci_hi_lt", None)
BARS.update({"M2_ci_hi_lt_vs_P6_only": M2_CI_HI_BAR, "M3_ci_lo_gt_vs_S1Z": M3_CI_LO_BAR})

# the next levers, copied from card section 5
NEXT = {
    "SM1": "S1ZSM seed 1 as the replicate, then the closed-loop read of PREREG_SCORER1P6 section 6",
    "SM2": "S1ZSM seed 1 under the R1/R2 rule. Then stack it with S1E's loss (REP-C), or the 1,600-update dose (PI check)",
    "SM3": "S1ZSM seed 1, read under the R1/R2 rule",
    "SM4": ("the speed direction was S1Z's EP leak and it is closed, but the gain stays inside noise at this dose. Next: the 1,600-update dose (PI check)"),
    "SM5": ("speed direction is not the EP leak. Next, zero GPU: the Shapley split on E2, and the per-head slow-pick share. Then TTC removed from L_twin under its own card"),
    "SM6": ("speed-neutral pairs cannot teach the heads the twin distinction, so S1Z's failure gain was carried by speed. Next: explicit relative features between proposals (lead gap, "
            "relative speed), under their own card (PREREG_S1Z_P6 Z4)"),
    "Unnamed": "matches none of SM1 to SM6 (or an input is undefined): labelled as such; the Shapley split on E2 names the next lever",
}
SM_NAMES = {"SM1": "GOAL MET", "SM2": "ADOPTED, BELOW GOAL", "SM3": "SWAMPED", "SM4": "MECHANISM, NO PDMS", "SM5": "EP NOT RECOVERED", "SM6": "FAILURES BACK", "Unnamed": "UNNAMED"}
SM_ORDER = ("SM1", "SM2", "SM3", "SM4", "SM5", "SM6")      # primary label: the card's order

SM_AMBIGUITIES = [
    "T1 P1, G1 to G4, goal status, family clause, Shapley split and top-K are read_s1z.s1z_read's own, with the S1ZSM arm in its S1Z slot (nothing re-implemented). The S1Z reader's M2 also "
    "computes a vs-C0 clause; that clause is NOT used here (card M2 row: S1ZSM - P6 only).",
    "T2 M2 = the failure-share difference S1ZSM - P6 on E3 (pick PDMS 0 or TTC 0; same paired draws as P1), CI-hi < 0 strict, no slack.",
    "T3 M3 = EP sub-score x100 of the pick on E3 (harness sub-score 'ego_progress' x100, as G1's sub-scores), S1ZSM - S1Z, paired log-cluster bootstrap on the same draws, CI-lo > 0 strict. "
    "S1Z's picks are read from its pass dir (logits + e3_changed_scores.csv) through the same score pool; S1Z must pass G0-E3 as an arm and must have every changed pick scored.",
    "T4 SM labels overlap as Batch 1's O-outcomes did: every match is listed; the primary label follows SM1, SM2, SM3, SM4, SM5, SM6, Unnamed (the card's order). SM1 = adopted and point >= 94.55; "
    "SM2 = adopted and point < 94.55; SM3 = S1ZSM vs C0 has CI-lo > 0 but |diff| <= 2F; SM4 = M2 and M3 hold, P1 fails, not SM3; SM5 = M2 holds, M3 fails; SM6 = M2 fails. "
    "Unnamed is reached when no label matches (for example P1 and M2 and M3 hold but a guard fails and the gain is not swamped) or when an input is undefined.",
    "T5 SM5 and SM6 are not conditioned on P1 (the card gives them only the M2/M3 conditions), so SM5 or SM6 can be listed next to SM1/SM2/SM3. The primary label still follows the card order.",
    "T6 The pair shares of the run (reported only) are read from --run-log: a pair_weight.json, or a JSON/JSONL log containing an object with 'sets_with_pair_share'. The card does not fix the file; "
    "if none is given or none is found the item reads 'not given'.",
    "T7 --rep-read: the S1Z-style rep_read.json (read_rep.py) key seed.{s_p, df, t_0.975}; t*s_p is printed next to 2F. Missing file or key: 'not given'.",
    "T8 The K ledger key is S1ZSM:<sha256 of the arm's e3_logits.npz>:E3 (family S1ZSM); the Bonferroni intervals are for best-of-K statements only, P1 is the arm's own test.",
    "T9 G0 for the reference S1Z: its proposals must be bit-identical to P6's (G0-E3, the same gate every arm passes). The 'logits bit-identical to the banked picks on unchanged tokens' check of the "
    "brief is read as that gate plus G0-PICK (P6 logits pick == banked pick); read_s1z has no further logit-level gate.",
]


# ------------------------------------------------------------------------------------------------ pure bar and outcome functions
def m2_clears(hi_vs_p6):
    """M2: S1ZSM - P6 CI-hi < 0 (strict)"""
    return None if hi_vs_p6 is None else bool(hi_vs_p6 < M2_CI_HI_BAR)


def m3_clears(lo_vs_s1z):
    """M3: EP S1ZSM - S1Z CI-lo > 0 (strict)"""
    return None if lo_vs_s1z is None else bool(lo_vs_s1z > M3_CI_LO_BAR)


def decide_outcome(st):
    """st: p1, m2, m3, swamped (bool|None), guards {"G1".."G4": bool}, g1_heads_failed [names], point (float|None).
    -> dict(matches, primary, name, next_levers, adopted)"""
    p1, m2, m3, sw, gd = st.get("p1"), st.get("m2"), st.get("m3"), st.get("swamped"), st.get("guards") or {}
    if any(v is None for v in (p1, m2, m3, sw)) or not gd or any(v is None for v in gd.values()):
        return {"matches": ["Unnamed"], "primary": "Unnamed", "name": SM_NAMES["Unnamed"], "adopted": False, "failed_guards": [], "next_levers": {"Unnamed": NEXT["Unnamed"]},
                "why": "an input is undefined (F, M2, M3 or a guard): the card's Unnamed rule"}
    failed = [k for k in ("G1", "G2", "G3", "G4") if not gd[k]]
    adopted = bool(p1 and not failed)
    pt = st.get("point")
    goal = bool(pt is not None and R.ge(pt, S.GOAL))
    m = []
    if adopted and goal:
        m.append("SM1")
    if adopted and not goal:
        m.append("SM2")
    if sw:
        m.append("SM3")
    if m2 and m3 and not p1 and not sw:
        m.append("SM4")
    if m2 and not m3:
        m.append("SM5")
    if not m2:
        m.append("SM6")
    prim = next((z for z in SM_ORDER if z in m), "Unnamed")
    out = {"matches": m or ["Unnamed"], "primary": prim, "name": SM_NAMES[prim], "adopted": adopted, "failed_guards": failed,
           "next_levers": {z: NEXT[z] for z in (m or ["Unnamed"])}}
    if failed:
        out["note_failed_guards"] = ("guards failed: " + ", ".join(failed) + (" (heads " + ", ".join(st.get("g1_heads_failed") or []) + ")" if "G1" in failed else "") +
                                     "; the card names no lever for a failed guard; the Shapley split on E2 names the next lever (Unnamed rule)")
    return out


# ------------------------------------------------------------------------------------------------ helpers
def sha256_file(p):
    return R.sha256_file(p)


_REL = re.compile(r"(?<![A-Za-z0-9])S1Z(?![A-Za-z0-9])")


def relabel(o):
    """S1Z -> S1ZSM in keys and strings of the blocks that s1z_read produced for the S1ZSM arm in its S1Z slot"""
    if isinstance(o, dict):
        return {(_REL.sub("S1ZSM", k) if isinstance(k, str) else k): relabel(v) for k, v in o.items()}
    if isinstance(o, list):
        return [relabel(v) for v in o]
    if isinstance(o, tuple):
        return tuple(relabel(v) for v in o)
    if isinstance(o, str):
        return _REL.sub("S1ZSM", o)
    return o


def _pub(d):
    return json.loads(json.dumps(d, default=R.jdefault))


def make_inputs(a):
    """C0, C0r, S1Z (reference) and S1ZSM (the arm) dirs -> read_scorer1p6.Inputs with arms keyed C0, C0r, S1Z, S1ZSM"""
    ns = argparse.Namespace(p6_proposals=a.p6_proposals, p6_csv=a.p6_csv, e2_tokens=a.e2_tokens, e2_table=a.e2_table, export=a.export, world_dump=a.world_dump,
                            arm=[f"C0={a.c0}", f"C0r={a.c0r}", f"S1Z={a.s1z}", f"S1ZSM={a.s1zsm}"], ens3_csv=None, c3_tokens=a.c3_tokens, no_tactical=a.no_tactical)
    return R.inputs_from_files(ns)


def slot_inputs(inp, mapping):
    arms = {slot: inp.arms[run] for slot, run in mapping.items()}
    return R.Inputs(inp.world, inp.p6_logits, inp.p6_proposals, inp.p6_rows, inp.e2_tokens, inp.e2_table, arms, inp.p6_pick_banked, inp.p6_fingerprint, inp.c3_strata,
                    inp.tactical, inp.provenance, inp.p6_token)


def read_pair_shares(path):
    """per-head pair shares of the run: pair_weight.json, or the last object with 'sets_with_pair_share' in a JSON / JSONL log"""
    if not path:
        return {"given": False}
    if not os.path.exists(path):
        return {"given": True, "path": path, "found": False, "why": "file does not exist"}
    found = None

    def walk(o):
        nonlocal found
        if isinstance(o, dict):
            if isinstance(o.get("sets_with_pair_share"), dict):
                found = o["sets_with_pair_share"]
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    txt = open(path, encoding="utf-8", errors="replace").read()
    try:
        walk(json.loads(txt))
    except ValueError:
        for ln in txt.splitlines():
            ln = ln.strip()
            if ln.startswith("{") and "sets_with_pair_share" in ln:
                try:
                    walk(json.loads(ln))
                except ValueError:
                    pass
    return {"given": True, "path": path, "found": found is not None, "sets_with_pair_share": found, "note": "reported, not deciding (card section 4); 0 on any head was a launch blocker (G18)"}


def read_rep(path, F):
    if not path:
        return {"given": False}
    try:
        d = json.load(open(path, encoding="utf-8"))
        sd = d["seed"]
        s_p, t = float(sd["s_p"]), float(sd["t_0.975"])
    except Exception as ex:   # noqa: BLE001
        return {"given": True, "path": path, "ok": False, "why": f"{type(ex).__name__}: {ex}"}
    return {"given": True, "path": path, "ok": True, "s_p": s_p, "df": sd.get("df"), "t_0.975": t, "t_times_s_p": t * s_p, "2F": None if F is None else 2 * F,
            "note": "reported next to 2F, not deciding"}


# ------------------------------------------------------------------------------------------------ the read
def s1zsm_read(inp, smoke=False, ledger_path=None, ledger_key_extra="", card_path=CARD, run_log=None, rep_read=None, control_s1z_read=None):
    """inp.arms must hold C0, C0r, S1Z (the reference) and S1ZSM (the arm). -> result dict; status READ, READ STOPPED (card/gate failed) or INCOMPLETE (scores missing)."""
    t0 = time.time()
    sha_act = sha256_file(card_path) if os.path.exists(card_path) else None
    res = {"tool": "read_s1zsm.py", "card": {"path": card_path, "sha256_expected": CARD_SHA256, "sha256_actual": sha_act}, "tier": S.TIER, "evidence_class": S.EVIDENCE,
           "estimator": S.ESTIMATOR, "bars": BARS, "strategic": S.STRATEGIC, "ambiguities_s1zsm": SM_AMBIGUITIES, "ambiguities_s1z": S.S1Z_AMBIGUITIES, "ambiguities_batch1": R.AMBIGUITIES,
           "arm": "S1ZSM" + (" (SMOKE STAND-IN)" if smoke else ""), "inputs": inp.provenance}
    if smoke:
        res["stamp"] = "SMOKE_NOT_A_RESULT"
    if sha_act != CARD_SHA256:
        res.update(status="READ STOPPED", why="REFUSED: the card's sha256 differs from the frozen one; the bars may have moved")
        return res
    for need in ("C0", "C0r", "S1Z", "S1ZSM"):
        if need not in inp.arms:
            res.update(status="READ STOPPED", why=f"arm {need} is required (C0 and C0r give P1's zero point and F; S1Z is M3's reference; S1ZSM is the arm)")
            return res
    res["code_sha256"] = {"read_s1zsm.py": sha256_file(HERE / "read_s1zsm.py"), "read_s1z.py": sha256_file(Path(S1Z_READ_DIR) / "read_s1z.py"),
                          "read_scorer1p6.py": sha256_file(Path(S.EVAL_DIR) / "read_scorer1p6.py")}

    # ---- everything the S1Z reader decides, with S1ZSM in its S1Z slot (no ledger here: the SM key is written below)
    base = S.s1z_read(slot_inputs(inp, {"C0": "C0", "C0r": "C0r", "S1Z": "S1ZSM"}), smoke=smoke, ledger_path=None)
    res["gates"] = base.get("gates", {})
    if base.get("status") != "READ":
        res.update(status=base.get("status"), why=base.get("why"), missing_tokens=base.get("missing_tokens"))
        return res

    W, N = inp.world, len(inp.world.token)
    # ---- the reference S1Z: its own G0-E3 gate, then picks and scores from the same pool recipe as s1z_read
    ref = inp.arms["S1Z"]
    g = R.gate_e3(ref, W, inp.p6_proposals)
    ref._ix = g.pop("_ix", None)
    res["gates"]["G0-E3[S1Z reference]"] = g
    if not g["PASS"]:
        res.update(status="READ STOPPED", why="a gate failed: G0-E3 of the reference arm S1Z; no verdict is written")
        return res
    arm = inp.arms["S1ZSM"]
    p6_pick = R.pick_logits(inp.p6_logits)
    L3 = {"S1ZSM": np.asarray(arm.e3_logits)[arm._ix], "S1Z": np.asarray(ref.e3_logits)[ref._ix]}
    pk3 = {"P6": p6_pick, "S1ZSM": R.pick_logits(L3["S1ZSM"]), "S1Z": R.pick_logits(L3["S1Z"])}
    pool = R.Pool(N)
    pool.add_rows(inp.p6_rows, W, p6_pick, "p6 csv")
    tpos = {t: i for i, t in enumerate(inp.e2_table["token"])}
    for t in inp.e2_tokens:
        i, ti = W.index[t], tpos[t]
        for k in range(R.M):
            pool.put(i, k, inp.e2_table["pdms"][ti, k], inp.e2_table["sub"][ti, k], "e2 table")
    c0 = inp.arms["C0"]
    pk_c0 = R.pick_logits(np.asarray(c0.e3_logits)[c0._ix])
    nbad = pool.add_rows(arm.rows, W, pk3["S1ZSM"], "csv S1ZSM") + pool.add_rows(ref.rows, W, pk3["S1Z"], "csv S1Z") + pool.add_rows(c0.rows, W, pk_c0, "csv C0")
    if pool.conflicts or nbad:
        res.update(status="READ STOPPED", why=f"score pool: {len(pool.conflicts)} conflicting scores, {nbad} invalid harness rows")
        return res
    short = {nm: int(len(pool.missing(pk3[nm]))) for nm in ("S1ZSM", "S1Z") if len(pool.missing(pk3[nm]))}
    if short:
        res.update(status="INCOMPLETE", why="harness scores missing for changed picks: " + json.dumps(short) + "; run emit-changed, score with the harness, add e3_changed_scores.csv")
        return res
    r = np.arange(N)
    pd3 = {nm: pool.pdms[r, pk3[nm]] for nm in pk3}
    sb3 = {nm: pool.sub[r, pk3[nm]] * 100.0 for nm in pk3}
    ttc3 = {nm: pool.sub[r, pk3[nm], R.ALL_HEADS["TTC"]] for nm in pk3}
    fail3 = {nm: ((pd3[nm] <= 0.0) | (ttc3[nm] <= 0.0)).astype(np.float64) for nm in pk3}
    B3 = R.BootSet(W.log)
    art3 = "S1ZSM and S1Z e3_logits.npz + e3_changed_scores.csv; P6 refe_p6final.csv; E3 = navtest 12,146 tokens"
    est3 = "paired log-cluster bootstrap, 10,000 draws, seed 20260927 (same draws as P1)"

    # ---- cross-checks against s1z_read's own numbers (same pool recipe, so these must agree to the last bit)
    pt_chk = abs(float(pd3["S1ZSM"].mean() * 100.0) - base["P1"]["PDMS_x100_E3"]["point"])
    pt_ep, rr_ep, n_ep = B3.diff(sb3["S1ZSM"][:, EP_COL], sb3["P6"][:, EP_COL])
    ep_vs_p6 = R.rec(pt_ep, rr_ep, n_ep)
    ep_chk = max(abs(ep_vs_p6[k] - base["reported_not_deciding"]["EP_subscore_vs_P6"][k]) for k in ("point", "lo", "hi"))
    res["internal_checks"] = {"pdms_point_vs_s1z_read_abs_diff": pt_chk, "EP_vs_P6_vs_s1z_read_max_abs_diff": ep_chk}
    if pt_chk > 1e-9 or ep_chk > 1e-12:
        res.update(status="READ STOPPED", why=f"the rebuilt pool disagrees with s1z_read ({pt_chk}, {ep_chk})")
        return res

    # ---- M2 (vs P6 only) and M3
    m2_p6 = base["M2"]["S1Z_minus_P6"]
    m2 = m2_clears(m2_p6["hi"])
    pt3, rr3, n3 = B3.diff(sb3["S1ZSM"][:, EP_COL], sb3["S1Z"][:, EP_COL])
    m3_d = R.rec(pt3, rr3, n3)
    m3 = m3_clears(m3_d["lo"])
    m2_blk = S.tag({"definition": "share of E3 tokens whose pick has PDMS 0 or TTC sub-score 0", "S1ZSM_share": base["M2"]["S1Z_share"], "P6_share": base["M2"]["P6_share"],
                    "S1ZSM_minus_P6": m2_p6, "bar": "S1ZSM - P6: CI-hi < 0 (no C0 clause)", "clears": m2,
                    "n_pick_pdms_zero": base["M2"]["n_pick_pdms_zero"], "n_pick_ttc_zero": base["M2"]["n_pick_ttc_zero"]}, est3, S.TIER["E3"], N, art3)
    m3_blk = S.tag({"definition": "EP (ego_progress) sub-score x100 of the pick, E3", "S1ZSM_EP_mean_x100": float(sb3["S1ZSM"][:, EP_COL].mean()), "S1Z_EP_mean_x100": float(sb3["S1Z"][:, EP_COL].mean()),
                    "P6_EP_mean_x100": float(sb3["P6"][:, EP_COL].mean()), "S1ZSM_minus_S1Z": m3_d, "bar": "S1ZSM - S1Z: CI-lo > 0", "clears": m3}, est3, S.TIER["E3"], N, art3)

    # ---- reported against S1Z: PDMS, failure share (E3), slow-pick share (E2 CRUISE)
    pt, rr, n = B3.diff(pd3["S1ZSM"] * 100.0, pd3["S1Z"] * 100.0)
    d_pdms = R.rec(pt, rr, n)
    pt, rr, n = B3.diff(fail3["S1ZSM"], fail3["S1Z"])
    d_fail = R.rec(pt, rr, n)
    wi2 = np.array([W.index[t] for t in inp.e2_tokens])
    n2 = len(wi2)

    def l2(a_, L3a, what):
        if a_.e2_logits is not None:
            j, err = R._align(a_.e2_token, list(inp.e2_tokens), what + " e2 logits")
            if err:
                raise SystemExit("REFUSING: " + err)
            return np.asarray(a_.e2_logits)[j]
        return L3a[wi2]

    pk2 = {"S1ZSM": R.pick_logits(l2(arm, L3["S1ZSM"], "S1ZSM")), "S1Z": R.pick_logits(l2(ref, L3["S1Z"], "S1Z"))}
    ratio2 = R.plan_v4(inp.p6_proposals[wi2]) / np.maximum(W.v0[wi2], 1e-6)[:, None]
    cru2 = W.cruise()[wi2]
    r2 = np.arange(n2)
    slow = {nm: (ratio2[r2, pk2[nm]] < S.SLOW_RATIO).astype(np.float64) for nm in pk2}
    B2 = R.BootSet(W.log[wi2])
    pt, rr, n = B2.diff(slow["S1ZSM"], slow["S1Z"], cru2)
    d_slow = R.rec(pt, rr, n)
    rep = base["reported_not_deciding"]
    F = base["P1"]["S1Z_vs_C0"]["F"]
    reported = S.tag({
        "S1ZSM_minus_S1Z_PDMS_x100_E3": d_pdms, "S1ZSM_minus_S1Z_failure_share_E3": d_fail,
        "S1ZSM_minus_S1Z_slow_pick_share_E2_CRUISE": {**d_slow, "S1ZSM_share": R.fin(B2.stat(slow["S1ZSM"], cru2)[0]), "S1Z_share": R.fin(B2.stat(slow["S1Z"], cru2)[0]), "n_cruise_tokens": int(cru2.sum())},
        "EP_subscore_vs_P6": rep["EP_subscore_vs_P6"], "other_E3_subscores_vs_P6": rep["other_E3_subscores_vs_P6"], "shapley_split_E2": rep["shapley_split_E2"],
        "topK_oracle_curve_E2_pdms_x100": rep["topK_oracle_curve_E2_pdms_x100"], "each_head_dRho_E2_CRUISE": rep["each_head_dRho_E2_CRUISE"], "EP_dRho_E2_CRUISE": rep["EP_dRho_E2_CRUISE"],
        "pair_shares_of_the_run": read_pair_shares(run_log), "seed_sd_next_to_2F": read_rep(rep_read, F), "not_computed": rep["not_computed"]},
        "E3 diffs: paired log-cluster bootstrap (same draws); slow share: paired over E2's logs, CRUISE; Shapley / top-K: point estimates", "E3 for PDMS/failure/EP; E2 for the rest", None,
        art3)

    # ---- outcome
    b = base
    p1 = bool(b["P1"]["clears"])
    guards = {k: bool(v["ok"]) for k, v in b["guards"].items()}
    swamped = b["outcome"]["inputs"]["swamped"]
    point = b["P1"]["PDMS_x100_E3"]["point"]
    st = dict(p1=p1, m2=m2, m3=m3, swamped=swamped, guards=guards, g1_heads_failed=b["guards"]["G1"]["failed_heads"], point=point)
    out = decide_outcome(st)
    out["inputs"] = {"P1_vs_C0": b["outcome"]["inputs"]["P1_vs_C0"], "P1_vs_P6": b["outcome"]["inputs"]["P1_vs_P6"], "P1": p1, "M2": m2, "M3": m3, "guards": guards,
                     "swamped": swamped, "point": point, "goal_status": b["goal_status"]["status"]}
    adopted = out["adopted"]
    adv = b["family_clause"]["adverse_components"]

    # ---- K ledger
    if ledger_path:
        if SEL_DIR not in sys.path:
            sys.path.insert(0, SEL_DIR)
        import selcore
        pt, rr_p6, _ = B3.diff(pd3["S1ZSM"] * 100.0, pd3["P6"] * 100.0)
        pt, rr_c0, _ = B3.diff(pd3["S1ZSM"] * 100.0, _c0_pdms(inp, pool, W, p6_pick) * 100.0)
        led = selcore.Ledger(ledger_path)
        sha_logits = (inp.provenance.get("arms") or {}).get("S1ZSM", [None, "no-provenance"])[1]
        key = f"{'SMOKE:' if smoke else ''}S1ZSM:{sha_logits}:E3{ledger_key_extra}"
        added = led.add(key, nominee="S1ZSM", family="S1ZSM", set="E3", smoke=bool(smoke))
        K = led.K
        lo_p, hi_p = selcore.interval(rr_p6[np.isfinite(rr_p6)], K)
        lo_c, hi_c = selcore.interval(rr_c0[np.isfinite(rr_c0)], K)
        kinfo = {"ledger": ledger_path, "K": K, "this_read_added": bool(added), "key": key,
                 "bonferroni_pdms_S1ZSM_minus_P6": {"point": float(b["P1"]["S1Z_vs_P6"]["point"]), "lo": lo_p, "hi": hi_p},
                 "bonferroni_pdms_S1ZSM_minus_C0": {"point": float(b["P1"]["S1Z_vs_C0"]["point"]), "lo": lo_c, "hi": hi_c},
                 "note": "K counts every navtest read in the programme's ledger; best-of-K statements use these intervals, P1 is the arm's own test"}
    else:
        kinfo = {"ledger": None, "K": None, "note": "no ledger written (--no-ledger): K NOT COUNTED"}

    p1_blk = relabel(b["P1"])
    res.update(status="READ", P1=p1_blk, M2=m2_blk, M3=m3_blk, guards=relabel(b["guards"]), goal_status=relabel(b["goal_status"]),
               adoption={"adopted": adopted, "rule": "S1ZSM if it clears P1 and G1 to G4", "P1": p1, "guards": guards}, family_clause=relabel(b["family_clause"]), outcome=out,
               reported_not_deciding=reported, K_ledger=kinfo, verdict_label=(("ADOPT" if not adv else "PDMS GAIN WITH COST") if adopted else "NO ADOPTION"),
               E3_pick_changed_vs_P6=b["E3_pick_changed_vs_P6"], E3_pick_changed_vs_S1Z=int((pk3["S1ZSM"] != pk3["S1Z"]).sum()), seconds=round(time.time() - t0, 1))
    res["batch1_endpoints_S1ZSM_slot"] = relabel(b["batch1_endpoints_S1Z_slot"])
    if control_s1z_read:
        res["control"] = compare_control(b, m2_blk, m3_blk, control_s1z_read)
    return res


def _c0_pdms(inp, pool, W, p6_pick):
    """C0's E3 pick PDMS (for the Bonferroni interval against C0), from the same pool"""
    a = inp.arms["C0"]
    pk = R.pick_logits(np.asarray(a.e3_logits)[a._ix])
    assert not len(pool.missing(pk)), "C0 picks must be scored"
    return pool.pdms[np.arange(len(pk)), pk]


# ------------------------------------------------------------------------------------------------ control: S1Z as the arm equals the banked S1Z read exactly
def _flat(o, path=""):
    if isinstance(o, dict):
        for k in sorted(o):
            yield from _flat(o[k], f"{path}/{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from _flat(v, f"{path}[{i}]")
    else:
        yield path, o


def compare_control(base, m2_blk, m3_blk, ref_path):
    """S1Z's own pass dir fed as the arm: every P1, M2-vs-P6, G1 to G4, EP-vs-P6 and goal number must equal s1z_read.json leaf for leaf; M3 (S1Z - S1Z) must be exactly 0.0, CI [0.0, 0.0]"""
    ref = json.load(open(ref_path, encoding="utf-8"))
    mine = _pub({"P1": {k: v for k, v in base["P1"].items() if k not in ("estimator", "tier", "n", "artifact")}, "goal": base["goal_status"]["status"],
                 "guards": {g: {k: v for k, v in base["guards"][g].items() if k not in ("estimator", "tier", "n", "artifact")} for g in ("G1", "G2", "G3", "G4")},
                 "M2_vs_P6": base["M2"]["S1Z_minus_P6"], "M2_shares": [base["M2"]["S1Z_share"], base["M2"]["P6_share"]], "EP_vs_P6": base["reported_not_deciding"]["EP_subscore_vs_P6"]})
    theirs = _pub({"P1": {k: v for k, v in ref["P1"].items() if k not in ("estimator", "tier", "n", "artifact")}, "goal": ref["goal_status"]["status"],
                   "guards": {g: {k: v for k, v in ref["guards"][g].items() if k not in ("estimator", "tier", "n", "artifact")} for g in ("G1", "G2", "G3", "G4")},
                   "M2_vs_P6": ref["M2"]["S1Z_minus_P6"], "M2_shares": [ref["M2"]["S1Z_share"], ref["M2"]["P6_share"]], "EP_vs_P6": ref["reported_not_deciding"]["EP_subscore_vs_P6"]})
    a, b = dict(_flat(mine)), dict(_flat(theirs))
    bad = [(k, a.get(k, "MISSING"), b.get(k, "MISSING")) for k in sorted(set(a) | set(b)) if a.get(k, "MISSING") != b.get(k, "MISSING")]
    d = m3_blk["S1ZSM_minus_S1Z"]
    m3_zero = bool(d["point"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0)
    return {"reference": ref_path, "reference_sha256": sha256_file(ref_path), "n_leaves_compared": len(set(a) | set(b)), "n_mismatch": len(bad), "mismatches": bad[:10],
            "EQUAL": len(bad) == 0, "M3_S1Z_minus_S1Z": d, "M3_exactly_zero_CI_zero": m3_zero, "PASS": bool(len(bad) == 0 and m3_zero)}


# ------------------------------------------------------------------------------------------------ markdown
def to_markdown(res):
    sm = res.get("stamp") == "SMOKE_NOT_A_RESULT"
    ctl = res.get("control")
    L = []
    if sm:
        L += ["**SMOKE_NOT_A_RESULT** -- the arm below is a stand-in; no number here is a result.", ""]
    if ctl:
        L += ["**CONTROL RUN: S1Z's own pass dir is fed as the arm. The outcome below is the control's, not an S1ZSM result.**", ""]
    L += ["# S1ZSM read", "", f"**Status: {res.get('status')}**", "",
          f"- Card: `{res['card']['path']}` sha256 `{res['card']['sha256_actual']}` (frozen `{res['card']['sha256_expected']}`)", f"- Tier: E3 = {S.TIER['E3']}; E2 = {S.TIER['E2']}.",
          f"- Evidence class: {S.EVIDENCE}.", f"- Estimator: {S.ESTIMATOR}", "", "## Gates", ""]
    for k, g in (res.get("gates") or {}).items():
        L.append(f"- {k}: **{'PASS' if g['PASS'] else 'FAIL'}** " + ", ".join(f"{a}={b}" for a, b in g.items() if a not in ("PASS", "name")))
    if res.get("status") != "READ":
        L += ["", f"**{res.get('status')}: {res.get('why')}**", "", "No verdict is written."]
        return "\n".join(L) + "\n"
    ci = R.ci_s
    o = res["outcome"]
    L += ["", "## Verdict", "", f"- Outcome: **{o['primary']} {o['name']}** (all matches: {', '.join(o['matches'])})"]
    for z in o["matches"]:
        L.append(f"  - {z}: {o['next_levers'][z]}")
    if o.get("note_failed_guards"):
        L.append(f"- {o['note_failed_guards']}")
    L += [f"- Adopted: **{res['adoption']['adopted']}**; verdict: **{res['verdict_label']}**" + (f"; adverse: {', '.join(res['family_clause']['adverse_components'])}" if res["family_clause"]["adverse_components"] else ""),
          f"- Goal status: **{res['goal_status']['status']}**", f"- K ledger: **K = {res['K_ledger'].get('K')}** ({res['K_ledger'].get('note')})", ""]
    p = res["P1"]
    L += ["## Bars", "", "| id | value | bar | result |", "|---|---|---|---|",
          f"| P1 PDMS x100 E3 | S1ZSM {R.lvl_s(p['PDMS_x100_E3'])} | vs C0: CI-lo>0 and abs>2F (F={R.fmt(p['S1ZSM_vs_C0']['F'])}); vs P6: CI-lo>0 | vs C0 {ci(p['S1ZSM_vs_C0'])} {p['S1ZSM_vs_C0']['clears']}; vs P6 {ci(p['S1ZSM_vs_P6'])} {p['S1ZSM_vs_P6']['clears']}; **P1 {p['clears']}** |"]
    m = res["M2"]
    L.append(f"| M2 failure share E3 | S1ZSM {m['S1ZSM_share']['point']:.4f}, P6 {m['P6_share']['point']:.4f} | S1ZSM-P6 CI-hi<0 | {ci(m['S1ZSM_minus_P6'])}; **M2 {m['clears']}** |")
    m3 = res["M3"]
    L.append(f"| M3 EP x100 E3 | S1ZSM {m3['S1ZSM_EP_mean_x100']:.4f}, S1Z {m3['S1Z_EP_mean_x100']:.4f}, P6 {m3['P6_EP_mean_x100']:.4f} | S1ZSM-S1Z CI-lo>0 | {ci(m3['S1ZSM_minus_S1Z'])}; **M3 {m3['clears']}** |")
    g = res["guards"]
    L.append("| G1 TTC/NC/DAC pts E3 | " + ", ".join("%s %+.3f" % (s_, v_["point"]) for s_, v_ in g["G1"]["per_head_points_vs_P6"].items()) + f" | each >= -0.5 | **{g['G1']['ok']}** |")
    L.append(f"| G2 delta regret E2 | {ci(g['G2']['delta_regret_vs_P6'])} | CI-hi<+0.5 | **{g['G2']['ok']}** |")
    L.append(f"| G3 overspeed E2 | {ci(g['G3']['overspeed_diff_vs_P6'])} | CI-hi<+0.10 vs P6 | **{g['G3']['ok']}** |")
    L.append(f"| G4 slow-pick E2 | {ci(g['G4']['S1ZSM_minus_P6'])} (S1ZSM {R.fmt(g['G4']['S1ZSM_share'])}, P6 {R.fmt(g['G4']['P6_share'])}) | CI-hi<+0.10 vs P6 | **{g['G4']['ok']}** |")
    rp = res["reported_not_deciding"]
    L += ["", "## Family clause (arm - P6, paired CIs)", "", f"- adverse components: {res['family_clause']['adverse_components'] or 'none'}", f"- STRATEGIC: {S.STRATEGIC}", "",
          "## Reported, not deciding", "", f"- S1ZSM - S1Z PDMS x100 E3: {ci(rp['S1ZSM_minus_S1Z_PDMS_x100_E3'])}", f"- S1ZSM - S1Z failure share E3: {ci(rp['S1ZSM_minus_S1Z_failure_share_E3'])}",
          f"- S1ZSM - S1Z slow-pick share E2 CRUISE: {ci(rp['S1ZSM_minus_S1Z_slow_pick_share_E2_CRUISE'])}", f"- EP sub-score vs P6 (E3): {ci(rp['EP_subscore_vs_P6'])}",
          f"- Shapley split E2: {json.dumps(rp['shapley_split_E2'], default=R.jdefault)}", f"- top-K oracle curve E2: {rp['topK_oracle_curve_E2_pdms_x100']}",
          f"- pair shares of the run: {json.dumps(rp['pair_shares_of_the_run'], default=R.jdefault)}", f"- seed SD next to 2F: {json.dumps(rp['seed_sd_next_to_2F'], default=R.jdefault)}"]
    if ctl:
        L += ["", "## Control (S1Z as the arm against the banked s1z_read.json)", "",
              f"- leaves compared {ctl['n_leaves_compared']}, mismatches {ctl['n_mismatch']}: **{'EQUAL' if ctl['EQUAL'] else 'NOT EQUAL'}**",
              f"- M3 (S1Z - S1Z): {ctl['M3_S1Z_minus_S1Z']} exactly zero with CI [0, 0]: **{ctl['M3_exactly_zero_CI_zero']}**", f"- control PASS: **{ctl['PASS']}**"]
    L += ["", "## Ambiguities (S1ZSM)", ""] + [f"- {a}" for a in res["ambiguities_s1zsm"]]
    return "\n".join(L) + "\n"


def write_outputs(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    jp, mp = os.path.join(out_dir, "s1zsm_read.json"), os.path.join(out_dir, "s1zsm_read.md")
    json.dump(res, open(jp, "w", encoding="utf-8", newline="\n"), indent=1, default=R.jdefault)
    open(mp, "w", encoding="utf-8", newline="\n").write(to_markdown(res))
    return jp, mp


# ------------------------------------------------------------------------------------------------ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["read", "emit-changed"])
    ap.add_argument("--c0", required=True)
    ap.add_argument("--c0r", required=True)
    ap.add_argument("--s1z", required=True, help="the S1Z pass dir (M3's reference)")
    ap.add_argument("--s1zsm", required=True, help="arm dir: e3_logits.npz [e2_logits.npz] [e3_changed_scores.csv]")
    ap.add_argument("--p6-proposals", default=R.DEFAULT_P6_PROPOSALS)
    ap.add_argument("--p6-csv", default=R.DEFAULT_P6_CSV)
    ap.add_argument("--e2-tokens", default=S.DEFAULT_E2_TOKENS)
    ap.add_argument("--e2-table", default=S.DEFAULT_E2_TABLE)
    ap.add_argument("--export", default=R.DEFAULT_EXPORT)
    ap.add_argument("--world-dump", default=None)
    ap.add_argument("--c3-tokens", default=None)
    ap.add_argument("--no-tactical", action="store_true")
    ap.add_argument("--card", default=CARD, help="the frozen card (sha256 checked)")
    ap.add_argument("--run-log", default=None, help="pair_weight.json or a run log with 'sets_with_pair_share' (reported only)")
    ap.add_argument("--rep-read", default=None, help="rep_read.json of read_rep.py (s_p, t*s_p next to 2F; reported only)")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--control-s1z-read", default=None, help="CONTROL: --s1zsm is S1Z's own dir; compare with the banked s1z_read.json (implies the control stamp; never writes the ledger)")
    ap.add_argument("--min-free-ram-gb", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    wait = S.wait_free_ram
    wait(a.min_free_ram_gb)
    inp = make_inputs(a)
    if a.cmd == "emit-changed":
        os.makedirs(a.out, exist_ok=True)
        inp2 = slot_inputs(inp, {"C0": "C0", "C0r": "C0r", "S1ZSM": "S1ZSM"})
        res = R.emit_changed(inp2, ["S1ZSM"], a.out, ens3=False)
        print(json.dumps(res, indent=1, default=R.jdefault))
        return 0
    ledger = None if (a.no_ledger or a.control_s1z_read) else a.ledger
    res = s1zsm_read(inp, ledger_path=ledger, card_path=a.card, run_log=a.run_log, rep_read=a.rep_read, control_s1z_read=a.control_s1z_read)
    jp, mp = write_outputs(res, a.out)
    print(json.dumps({"status": res.get("status"), "why": res.get("why"), "outcome": res.get("outcome", {}).get("primary"), "matches": res.get("outcome", {}).get("matches"),
                      "K": res.get("K_ledger", {}).get("K"), "control": res.get("control", {}).get("PASS"), "json": jp, "md": mp}, indent=1, default=R.jdefault))
    return 0 if res.get("status") == "READ" else 2


if __name__ == "__main__":
    sys.exit(main())
