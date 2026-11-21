#!/usr/bin/env python3
"""read_div.py -- the read tool of PREREG_DIV_P6 (frozen 2026-11-10, sha256 e5fb2f8e...f6bacf). CPU only, no GPU, no training.

    python read_div.py read --out OUT [--member NAME=DIR ...] [--ledger PATH | --no-ledger] [--min-free-ram-gb 3]        # the real read (ten members, K + 5)
    python read_div.py read --control c1 --out OUT        # PESS-8 rebuilt from the eight banked members == the banked PESS read (sha and PDMS), reads only an arm already read
    python read_div.py read --control c2 --out OUT        # PESS over the single member S1ZSM == the banked S1ZSM PDMS, reads only an arm already read
    (control c3 = the synthetic unit tests in test_read_div.py: every outcome branch and the card-sha refusal)

Arms (card section 2), all with read_pess.py's rules (min on NC, DAC, TTC; mean on EP, C, DDC; float32) and the full truth-table lookup:
    PESS-DV   = PESS over {DV0, DV1}                          PESS-SEED = PESS over {C0, C0r}  (the matched control)
    PESS-10   = PESS over the eight PESS-read members + DV0 + DV1 (member ORDER is part of the combined-logit sha256: float32 mean)
    PESS-8    = PESS over the eight members (the banked PESS arm; its combined-logit sha256 must equal the banked one, else READ STOPPED)
    DV0, DV1  = the single members; C0, C0r single members as references (already-read arms, never ledger keys)
Bars (card section 3), E3 = full navtest, PDMS x100, paired log-cluster bootstrap (R.BootSet, 10,000 draws, seed 20260927):
    Q1 PESS-DV - PESS-SEED CI-lo > 0;  Q2 PESS-10 - P6 CI-lo > 0 and d > 0.4971, AND PESS-10 - PESS-8 CI-lo > 0;  M zero-pick share PESS-DV - PESS-SEED CI-hi < 0;
    G1 to G4 as PREREG_PESS_P6 for PESS-10 vs P6.  Outcomes DV1 / DV2 / DV3 / Unnamed.  K + 5 (DV0, DV1, PESS-DV, PESS-SEED, PESS-10; key ARM:sha256(combined logits):E3).

REUSE, NOT A FORK: read_pess.py (combine, load_member + G0 gates through pess_read, guards_vs_p6, family_vs_p6, Truth, wait_free_ram) and, through it, read_scorer1p6.py and
selectors/selcore.py are imported unchanged.  pess_read is called once with arms=() to load, gate and align every member; the E3 BootSet (built on the 12,146-token log array,
NOT the later E2 one), the world, P6's proposals and the E2 quantities are captured from that call (the read_pess_a1.py trick).
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
PESS_DIR = P6BASE + "/pess/read"
if PESS_DIR not in sys.path:
    sys.path.insert(0, PESS_DIR)
import read_pess as RP  # noqa: E402  (unchanged)

R = RP.R
CARD = P6BASE + "/PREREG_DIV_P6.md"
CARD_SHA256 = "e5fb2f8e4123fcfe3a599b0cc71537bab187716c6a5d3e46735ce094bacf6fac"
LEDGER = RP.DEFAULT_LEDGER
PESS_BANKED_JSON = PESS_DIR + "/real/pess_read.json"
A = "F:/tanitad_scratch/p6arms/"
P6_DIR_DEFAULT = A + "pass/P6_control"
MEMBER_DIRS = {"C0": A + "pass/C0", "C0r": A + "pass/C0r", "S1E": A + "pass/S1E",
               "C0r2": A + "pass_rep_c0r2_s1e1_s1z1/C0r2", "S1E1": A + "pass_rep_c0r2_s1e1_s1z1/S1E1", "S1Z1": A + "pass_rep_c0r2_s1e1_s1z1/S1Z1",
               "S1Z": A + "pass_s1z/S1Z", "S1ZSM": A + "pass_s1zsm/S1ZSM", "DV0": A + "pass_div/DV0", "DV1": A + "pass_div/DV1"}
EIGHT = ["C0", "C0r", "S1E", "C0r2", "S1E1", "S1Z1", "S1Z", "S1ZSM"]      # the order of the banked PESS read (pess_read.json members); order matters for the float32 mean
TEN = EIGHT + ["DV0", "DV1"]
# arm -> member names (order as listed).  Every arm is PESS over its members.
ARMS = {"DV0": ["DV0"], "DV1": ["DV1"], "PESS-DV": ["DV0", "DV1"], "PESS-SEED": ["C0", "C0r"], "PESS-10": TEN}
REF_ARMS = {"PESS-8": EIGHT, "C0": ["C0"], "C0r": ["C0r"]}       # already-read arms, reported as references, never ledger keys
LEDGER_ARMS = ("DV0", "DV1", "PESS-DV", "PESS-SEED", "PESS-10")  # K + 5
BANKED_PESS8_PDMS_X100 = 91.47901718634304                      # pess_read.json arms.PESS.PDMS_x100_E3.point
BANKED_S1ZSM_PDMS_X100 = 91.29707529608928                      # s1zsm_read.json P1.PDMS_x100_E3.point
BANKED_PESS8_SHA256_PREFIX = "8912f1d04ab3e22571397d53580ddafebc3beb78e1e27d34fe21db2"   # card text prints "8912f1d0..."; the full sha is read from PESS_BANKED_JSON
CONTROL_MEMBERS = {"c1": EIGHT, "c2": ["S1ZSM"]}
CONTROL_ARMS = {"c1": {"PESS-8": EIGHT}, "c2": {"S1ZSM": ["S1ZSM"]}}   # the ONLY arms a control may score: both already read; PESS-SEED etc. are new navtest reads

P1_FLOOR = RP.P1_FLOOR
STRATEGIC = R.STRATEGIC

BARS = {"Q1": "PESS-DV - PESS-SEED (PDMS x100): CI-lo > 0",
        "Q2": f"PESS-10 - P6: CI-lo > 0 and d > {P1_FLOOR}; AND PESS-10 - PESS-8: CI-lo > 0",
        "M": "zero-pick share (pick PDMS exactly 0), PESS-DV - PESS-SEED: CI-hi < 0",
        "G1-G4": "as PREREG_PESS_P6 for PESS-10 vs P6 (G1 TTC/NC/DAC points >= -0.5; G2 regret CI-hi < +0.5; G3 overspeed CI-hi < +0.10; G4 slow CI-hi < +0.10)"}

DV_NAMES = {"DV1": "ADOPT", "DV2": "DIVERSITY HELPS, TOO SMALL", "DV3": "NO DIVERSITY GAIN", "Unnamed": "UNNAMED"}
DV_ORDER = ("DV1", "DV2", "DV3")
DV_NEXT = {
    "DV1": "PESS-10 becomes P6's selector. Next: four quarter-split members under their own card",
    "DV2": "data diversity is real but the ensemble stays below the bar. Next: Decision A (the vision drivable-area check) carries the line; more members only if A stalls",
    "DV3": "the shared error sits in P6's frozen trunk, not in the fine-tune data. The ensemble line is closed; Decision A",
    "Unnamed": "matches none of DV1 to DV3: labelled as such",
}

AMBIGUITIES = [
    "D1 Every arm is PESS (read_pess.combine, rule PESS) over the named members; DV0 and DV1 as single arms are bit-for-bit the member's own logits (one member: min = mean). "
    "Member ORDER is part of the combined-logit sha256 (float32 mean); PESS-8 uses the banked order C0, C0r, S1E, C0r2, S1E1, S1Z1, S1Z, S1ZSM and PESS-10 appends DV0, DV1.",
    "D2 Q2 has two parts, both required: PESS-10 - P6 with CI-lo > 0 and the POINT d strictly above the printed constant 0.4971 (as P1 of the PESS card), and PESS-10 - PESS-8 with CI-lo > 0. "
    "'Q2 fails' = not both parts.",
    "D3 Outcomes, card section 4: DV1 = Q2 (both parts) and G1 to G4; DV2 = Q1 holds and Q2 fails; DV3 = Q1 fails. M is a reported bar and is not a condition of any label. "
    "Labels may overlap (Q1 fails and Q2 and the guards hold matches DV1 and DV3): every match is listed, the primary follows DV1, DV2, DV3, Unnamed. "
    "Q1 holds, Q2 holds, a guard fails matches none -> Unnamed. Any undefined input -> Unnamed.",
    "D4 M is the share of E3 tokens whose pick has PDMS exactly 0.0, difference PESS-DV - PESS-SEED, over the same paired log-cluster draws as Q1; CI-hi < 0 strict.",
    "D5 G1 to G4 are read_pess.guards_vs_p6 (Batch 1 / S1Z definitions) for PESS-10 vs P6; E2 quantities use PESS-10's E3 combined logits restricted to the E2 tokens. Guards are also printed for the other arms, not deciding.",
    "D6 Scoring = lookup in the full truth table at the pick (valid and finite at every pick or the read stops). G0 gates of read_pess run for P6 and every member (G0-E3: proposals bit-identical to P6's). "
    "A further gate G0-PESS8 requires PESS-8 rebuilt here to equal the banked PESS read (combined-logit sha256 and PDMS x100) exactly; otherwise READ STOPPED and no verdict is written.",
    "D7 Ledger keys <ARM>:<sha256 of the combined float32 [12146,64,6] logits>:E3 for DV0, DV1, PESS-DV, PESS-SEED, PESS-10 (K + 5). PESS-8, C0 and C0r are already-read arms and are not keyed. "
    "A key that is already in the ledger is not added again (selcore.Ledger.add), so K + 5 holds only if all five are new. Controls never write the ledger.",
    "D8 Reported, not deciding (card section 3): DV0 and DV1 each vs P6 and vs C0 / C0r (paired); the failure share and the four metric families per arm vs P6 (STRATEGIC PAUSED). "
    "The trainer held-out monitor is read from the chain's training log, not by this tool.",
]


def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def card_ok(card_path=CARD):
    act = RP.sha256_file(card_path) if os.path.exists(card_path) else None
    return act == CARD_SHA256, act


# ------------------------------------------------------------------------------------------------ pure decisions
def q1_clears(lo):
    return None if lo is None else bool(lo > 0)


def q2a_clears(point, lo):
    return None if point is None or lo is None else bool(lo > 0 and point > P1_FLOOR)


def q2b_clears(lo):
    return None if lo is None else bool(lo > 0)


def m_clears(hi):
    return None if hi is None else bool(hi < 0)


def decide_outcome(q1, q2a, q2b, guards):
    """q1, q2a, q2b: bool|None; guards {"G1".."G4": bool|None}. -> dict(matches, primary, name, adopted, q2, failed_guards, next_levers)"""
    if any(v is None for v in (q1, q2a, q2b)) or not guards or any(v is None for v in guards.values()):
        return {"matches": ["Unnamed"], "primary": "Unnamed", "name": DV_NAMES["Unnamed"], "adopted": False, "q2": None, "failed_guards": [],
                "next_levers": {"Unnamed": DV_NEXT["Unnamed"]}, "why": "an input is undefined: the card's Unnamed rule"}
    q2 = bool(q2a and q2b)
    failed = [k for k in ("G1", "G2", "G3", "G4") if not guards[k]]
    mt = []
    if q2 and not failed:
        mt.append("DV1")
    if q1 and not q2:
        mt.append("DV2")
    if not q1:
        mt.append("DV3")
    prim = next((z for z in DV_ORDER if z in mt), "Unnamed")
    return {"matches": mt or ["Unnamed"], "primary": prim, "name": DV_NAMES[prim], "adopted": "DV1" in mt, "q2": q2, "failed_guards": failed,
            "next_levers": {z: DV_NEXT[z] for z in (mt or ["Unnamed"])}}


def bars_from_arrays(B3, pdms, zero_share_inputs=None):
    """pdms: {arm: [N] PDMS in 0..1 at the pick} for PESS-DV, PESS-SEED, PESS-10, PESS-8, P6. -> dict of Q1, Q2a, Q2b, M records with their verdict booleans.  Pure given B3."""
    d = lambda x, y: R.rec(*B3.diff(x, y))      # noqa: E731
    q1 = d(pdms["PESS-DV"] * 100.0, pdms["PESS-SEED"] * 100.0)
    q2a = d(pdms["PESS-10"] * 100.0, pdms["P6"] * 100.0)
    q2b = d(pdms["PESS-10"] * 100.0, pdms["PESS-8"] * 100.0)
    zr = lambda nm: (pdms[nm] == 0.0).astype(np.float64)    # noqa: E731
    m = d(zr("PESS-DV"), zr("PESS-SEED"))
    return {"Q1": {**q1, "clears": q1_clears(q1["lo"])}, "Q2a_vs_P6": {**q2a, "floor": P1_FLOOR, "clears": q2a_clears(q2a["point"], q2a["lo"])},
            "Q2b_vs_PESS8": {**q2b, "clears": q2b_clears(q2b["lo"])},
            "M": {**m, "PESS-DV_share": R.fin(zr("PESS-DV").mean()), "PESS-SEED_share": R.fin(zr("PESS-SEED").mean()), "clears": m_clears(m["hi"]),
                  "n_pick_pdms_zero_PESS-DV": int((pdms["PESS-DV"] == 0.0).sum()), "n_pick_pdms_zero_PESS-SEED": int((pdms["PESS-SEED"] == 0.0).sum())}}


# ------------------------------------------------------------------------------------------------ the load (through pess_read, unchanged)
def load_all(members, p6_dir, truth_path, **kw):
    """members: list of (name, dir). Runs read_pess.pess_read with arms=() (loading + every G0 gate), capturing the E3 BootSet and the world. -> (res, cap)"""
    cap = {}
    _BS, _E2 = R.BootSet, RP.e2_quantities

    def BS(*x, **k):
        b = _BS(*x, **k)
        lg = x[0] if x else k["log"]
        if len(lg) == 12146 and "B3" not in cap:           # the E3 set, not the 2,000-token E2 one
            cap["B3"] = b
        return b

    def E2(W, truth, e2_tokens, e2_table, p6_props):
        out = _E2(W, truth, e2_tokens, e2_table, p6_props)
        cap.update(W=W, E=out, p6_props=p6_props)
        return out
    R.BootSet, RP.e2_quantities = BS, E2
    sha0, card0 = RP.CARD_SHA256, RP.CARD
    RP.CARD_SHA256, RP.CARD = CARD_SHA256, kw.get("card_path", CARD)    # pess_read's own card check now tests THIS card's sha256
    try:
        res = RP.pess_read(members, p6_dir, truth_path=truth_path, arms=(), ledger_path=None, **kw)
    finally:
        R.BootSet, RP.e2_quantities = _BS, _E2
        RP.CARD_SHA256, RP.CARD = sha0, card0
    return res, cap


def combine_arm(logits_by_name, names):
    return RP.combine([logits_by_name[n] for n in names], "PESS")


def score_arm(truth, comb):
    pick = R.pick_logits(comb)
    d, s = truth.lookup(pick)
    return pick, d, s * 100.0


def check_pess8(comb8, pdms8):
    """G0-PESS8: the rebuilt PESS-8 equals the banked PESS read, bit for bit and to float equality"""
    banked = json.load(open(PESS_BANKED_JSON, encoding="utf-8"))
    want_sha = banked["arm_logits_sha256"]["PESS"]
    want_pdms = banked["arms"]["PESS"]["PDMS_x100_E3"]["point"]
    got_sha = sha256_array(comb8)
    return {"name": "G0-PESS8", "banked_json": PESS_BANKED_JSON, "banked_json_sha256": RP.sha256_file(PESS_BANKED_JSON), "banked_members": banked["members"], "members_equal_banked_order": banked["members"] == EIGHT,
            "banked_sha256": want_sha, "rebuilt_sha256": got_sha, "sha256_equal": bool(got_sha == want_sha), "sha256_has_card_prefix": bool(want_sha.startswith(BANKED_PESS8_SHA256_PREFIX)),
            "banked_PDMS_x100": want_pdms, "rebuilt_PDMS_x100": float(pdms8), "PDMS_equal_exactly": bool(float(pdms8) == want_pdms),
            "PASS": bool(got_sha == want_sha and float(pdms8) == want_pdms and banked["members"] == EIGHT and want_sha.startswith(BANKED_PESS8_SHA256_PREFIX))}


# ------------------------------------------------------------------------------------------------ the read
def div_read(members, p6_dir, truth_path=RP.DEFAULT_TABLE, ledger_path=None, card_path=CARD, control=None, **kw):
    t0 = time.time()
    ok, act = card_ok(card_path)
    res = {"tool": "read_div.py", "card": {"path": card_path, "sha256_expected": CARD_SHA256, "sha256_actual": act}, "tier": R.TIER, "evidence_class": R.EVIDENCE,
           "estimator": R.ESTIMATOR + " Training variance is not resampled: ensembles of fixed members are read as fixed scorers. Inference is deterministic.",
           "bars": BARS, "strategic": STRATEGIC, "ambiguities": AMBIGUITIES, "members": [n for n, _ in members], "n_members": len(members), "control": control}
    if not ok:
        res.update(status="READ STOPPED", why="REFUSED: the card's sha256 differs from the frozen one; the bars may have moved")
        return res
    names = [n for n, _ in members]
    want = CONTROL_MEMBERS[control] if control else TEN
    if names != want:
        res.update(status="READ STOPPED", why=f"REFUSED: members must be exactly {want} in this order, got {names}")
        return res
    res["code_sha256"] = {"read_div.py": RP.sha256_file(HERE / "read_div.py"), "read_pess.py": RP.sha256_file(Path(PESS_DIR) / "read_pess.py"),
                          "read_scorer1p6.py": RP.sha256_file(Path(RP.EVAL_DIR) / "read_scorer1p6.py")}
    base, cap = load_all(members, p6_dir, truth_path, card_path=card_path, **kw)
    res["gates"] = base.get("gates", {})
    res["member_e3_logits_sha256"] = base.get("member_e3_logits_sha256", {})
    if base.get("status") != "READ":
        res.update(status="READ STOPPED", why=base.get("why"))
        return res
    truth, mem, pick6 = base["_truth"], dict(zip(names, base["_mem_logits"])), base["_pick"]["P6"]
    B3 = cap["B3"]
    sb6 = truth.lookup(pick6)[1] * 100.0
    d6 = base["_pdms_by_arm"]["P6"]
    spec = dict(CONTROL_ARMS[control]) if control else {**ARMS, **REF_ARMS}
    comb, pick, pdms, sub = {}, {"P6": pick6}, {"P6": d6}, {"P6": sb6}
    for a, ns in spec.items():
        comb[a] = combine_arm(mem, ns)
        pick[a], pdms[a], sub[a] = score_arm(truth, comb[a])
    res["arm_members"] = {a: spec[a] for a in spec}
    res["arm_logits_sha256"] = {a: sha256_array(comb[a]) for a in comb}
    res["arm_pick_changed_vs_P6"] = {a: int((pick[a] != pick6).sum()) for a in comb}
    lvl = {a: R.rec(*B3.stat(pdms[a] * 100.0)) for a in comb}
    if control:                                                    # controls: the level of an already-read arm and the equality checks, nothing else
        res["arm_PDMS_x100_E3"] = lvl
        res["_comb"], res["_mem"], res["_pdms"] = comb, mem, pdms
        res.update(status="READ", K_ledger={"ledger": None, "K": None, "note": "no ledger written in a control: K NOT COUNTED"}, seconds=round(time.time() - t0, 1))
        return res
    g8 = check_pess8(comb["PESS-8"], lvl["PESS-8"]["point"])
    res["gates"][g8["name"]] = g8
    if not g8["PASS"]:
        res.update(status="READ STOPPED", why="a gate failed: G0-PESS8 (the rebuilt PESS-8 differs from the banked PESS read); no verdict is written")
        return res
    W, E, p6_props = cap["W"], cap["E"], cap["p6_props"]
    zero = {a: (pdms[a] == 0.0).astype(np.float64) for a in pdms}
    fail = {a: ((pdms[a] <= 0.0) | (sub[a][:, R.ALL_HEADS["TTC"]] <= 0.0)).astype(np.float64) for a in pdms}
    arms_out, rr_p6 = {}, {}
    for a in comb:
        pt, rr, n = B3.diff(pdms[a] * 100.0, pdms["P6"] * 100.0)
        rr_p6[a] = rr
        o = {"members": spec[a], "PDMS_x100_E3": lvl[a], "minus_P6": R.rec(pt, rr, n), "zero_pick_share": R.fin(zero[a].mean())}
        if a in REF_ARMS and a == "PESS-8":
            arms_out[a] = o
            continue
        if a in ("C0", "C0r"):
            arms_out[a] = o
            continue
        pt, rr, n = B3.diff(sub[a][:, RP.EP_COL], sub["P6"][:, RP.EP_COL])
        o["EP_x100_E3"] = {"arm_mean": float(sub[a][:, RP.EP_COL].mean()), "P6_mean": float(sub["P6"][:, RP.EP_COL].mean()), "minus_P6": R.rec(pt, rr, n)}
        pt, rr, n = B3.diff(fail[a], fail["P6"])
        o["failure_share_minus_P6_reported"] = R.rec(pt, rr, n)
        o["G"] = RP.guards_vs_p6(E, pick[a], pick6, sub[a], sb6, B3)
        o["family_vs_P6"], o["adverse_components"] = RP.family_vs_p6(W, p6_props, truth.sub, pick[a], pick6, B3, E)
        arms_out[a] = o
    res["arms"] = arms_out
    # ---- reported, not deciding: each DV member vs C0 / C0r
    pair = {}
    for a, ref in (("DV0", "C0"), ("DV1", "C0r")):
        pair[f"{a}_minus_{ref}"] = R.rec(*B3.diff(pdms[a] * 100.0, pdms[ref] * 100.0))
    res["reported_pairs"] = pair
    # ---- bars and outcome
    bars = bars_from_arrays(B3, pdms)
    res["bars_read"] = bars
    g = arms_out["PESS-10"]["G"]["ok"]
    res["outcome"] = decide_outcome(bars["Q1"]["clears"], bars["Q2a_vs_P6"]["clears"], bars["Q2b_vs_PESS8"]["clears"], g)
    res["outcome"]["inputs"] = {"Q1": bars["Q1"]["clears"], "Q2a": bars["Q2a_vs_P6"]["clears"], "Q2b": bars["Q2b_vs_PESS8"]["clears"], "M_reported": bars["M"]["clears"], "guards": g,
                                "point_PESS-10": lvl["PESS-10"]["point"]}
    res["goal_status_PESS-10"] = R.goal_status(lvl["PESS-10"]["point"], lvl["PESS-10"]["hi"])
    # ---- ledger (K + 5) and Bonferroni intervals
    kinfo = {"ledger": None, "K": None, "note": "no ledger written: K NOT COUNTED"}
    if ledger_path:
        if RP.SEL_DIR not in sys.path:
            sys.path.insert(0, RP.SEL_DIR)
        import selcore
        led = selcore.Ledger(ledger_path)
        K_before = led.K
        keys, added = {}, {}
        for a in LEDGER_ARMS:
            keys[a] = f"{a}:{res['arm_logits_sha256'][a]}:E3"
            added[a] = bool(led.add(keys[a], nominee=a, family="DIV", set="E3", smoke=False))
        bon = {}
        for a in LEDGER_ARMS:
            lo, hi = selcore.interval(rr_p6[a][np.isfinite(rr_p6[a])], led.K)
            bon[a] = {"point": arms_out[a]["minus_P6"]["point"], "lo": lo, "hi": hi}
        kinfo = {"ledger": ledger_path, "K_before": K_before, "K": led.K, "keys": keys, "this_read_added": added, "bonferroni_pdms_minus_P6": bon,
                 "note": "K counts every navtest read in the programme's ledger; expected K_before + 5; best-of-K statements use these intervals; Q1, Q2, M are the card's own tests"}
    res.update(status="READ", K_ledger=kinfo, seconds=round(time.time() - t0, 1))
    return res


# ------------------------------------------------------------------------------------------------ controls
def control_result(control, res):
    """c1 and c2: equality checks against banked numbers; both read only arms that are already read"""
    banked_json = json.load(open(PESS_BANKED_JSON, encoding="utf-8"))
    if control == "c1":
        pd = res["arm_PDMS_x100_E3"]["PESS-8"]["point"]
        got = res["arm_logits_sha256"]["PESS-8"]
        out = {"control": "c1", "arm": "PESS-8 rebuilt from the eight members", "banked_path": PESS_BANKED_JSON, "banked_json_sha256": RP.sha256_file(PESS_BANKED_JSON),
               "banked_sha256": banked_json["arm_logits_sha256"]["PESS"], "rebuilt_sha256": got, "sha256_equal": bool(got == banked_json["arm_logits_sha256"]["PESS"]),
               "banked_PDMS_x100": banked_json["arms"]["PESS"]["PDMS_x100_E3"]["point"], "literal_PDMS_x100": BANKED_PESS8_PDMS_X100, "rebuilt_PDMS_x100": pd,
               "PDMS_equal_exactly_banked": bool(pd == banked_json["arms"]["PESS"]["PDMS_x100_E3"]["point"]), "PDMS_equal_exactly_literal": bool(pd == BANKED_PESS8_PDMS_X100),
               "members": res["members"], "arms_scored": list(res["arm_logits_sha256"])}
        out["PASS"] = bool(out["sha256_equal"] and out["PDMS_equal_exactly_banked"] and out["PDMS_equal_exactly_literal"] and out["arms_scored"] == ["PESS-8"]
                           and banked_json["arm_logits_sha256"]["PESS"].startswith(BANKED_PESS8_SHA256_PREFIX))
        return out
    pd = res["arm_PDMS_x100_E3"]["S1ZSM"]["point"]
    L = res["_mem"]["S1ZSM"]
    s1 = json.load(open(RP.S1ZSM_BANKED_JSON, encoding="utf-8"))["P1"]["PDMS_x100_E3"]["point"]
    out = {"control": "c2", "arm": "PESS over the single member S1ZSM", "banked_path": RP.S1ZSM_BANKED_JSON, "banked_json_sha256": RP.sha256_file(RP.S1ZSM_BANKED_JSON), "banked_PDMS_x100": s1,
           "literal_PDMS_x100": BANKED_S1ZSM_PDMS_X100, "rebuilt_PDMS_x100": pd, "PDMS_equal_exactly_banked": bool(pd == s1), "PDMS_equal_exactly_literal": bool(pd == BANKED_S1ZSM_PDMS_X100),
           "arm_logits_equal_member_bits": bool(R.bits_equal(res["_comb"]["S1ZSM"], L)), "members": res["members"], "arms_scored": list(res["arm_logits_sha256"])}
    out["PASS"] = bool(out["PDMS_equal_exactly_banked"] and out["PDMS_equal_exactly_literal"] and out["arm_logits_equal_member_bits"] and out["arms_scored"] == ["S1ZSM"])
    return out


# ------------------------------------------------------------------------------------------------ markdown / outputs
def to_markdown(res):
    ci = R.ci_s
    L = ["# DIV read" + (f" (CONTROL {res['control']})" if res.get("control") else ""), "", f"**Status: {res.get('status')}**", ""]
    if res.get("control"):
        L += ["**CONTROL RUN: not a DIV result. It reads only an arm already read; the ledger is never written.**", ""]
    L += [f"- Card: `{res['card']['path']}` sha256 `{res['card']['sha256_actual']}` (frozen `{res['card']['sha256_expected']}`)", f"- Tier: {R.TIER['E3']}", f"- Evidence class: {R.EVIDENCE}.",
          f"- Estimator: {res['estimator']}", f"- Members ({res['n_members']}): {', '.join(res['members'])}", "", "## Gates", ""]
    for k, g in (res.get("gates") or {}).items():
        L.append(f"- {k}: **{'PASS' if g['PASS'] else 'FAIL'}** " + ", ".join(f"{a}={b}" for a, b in g.items() if a not in ("PASS", "name")))
    if res.get("status") != "READ":
        return "\n".join(L + ["", f"**{res.get('status')}: {res.get('why')}**", "", "No verdict is written."]) + "\n"
    if res.get("control_result"):
        L += ["", "## Control result", "", "```", json.dumps(res["control_result"], indent=1, default=R.jdefault), "```"]
        return "\n".join(L) + "\n"
    o = res["outcome"]
    L += ["", "## Verdict", "", f"- Outcome: **{o['primary']} {o['name']}** (all matches: {', '.join(o['matches'])})"] + [f"  - {z}: {o['next_levers'][z]}" for z in o["matches"]]
    L += [f"- Inputs: {json.dumps(o['inputs'], default=R.jdefault)}", f"- Goal status of PESS-10: **{res['goal_status_PESS-10']}**"]
    b = res["bars_read"]
    L += ["", "## Bars", "", "| bar | estimate | clears |", "|---|---|---|", f"| Q1 PESS-DV - PESS-SEED | {ci(b['Q1'])} | {b['Q1']['clears']} |",
          f"| Q2a PESS-10 - P6 (floor {P1_FLOOR}) | {ci(b['Q2a_vs_P6'])} | {b['Q2a_vs_P6']['clears']} |", f"| Q2b PESS-10 - PESS-8 | {ci(b['Q2b_vs_PESS8'])} | {b['Q2b_vs_PESS8']['clears']} |",
          f"| M zero-pick share PESS-DV - PESS-SEED | {ci(b['M'])} (DV {b['M']['PESS-DV_share']:.5f}, SEED {b['M']['PESS-SEED_share']:.5f}) | {b['M']['clears']} |"]
    k = res["K_ledger"]
    L += ["", f"- K ledger: K = {k.get('K')} (before {k.get('K_before')}); {k.get('note')}", "", "## Arms (E3, PDMS x100; paired log-cluster bootstrap)", "",
          "| arm | members | PDMS | minus P6 | EP x100 minus P6 | zero-pick share | picks changed vs P6 | G1 | G2 | G3 | G4 | logits sha256 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, q in res["arms"].items():
        g = q.get("G")
        ep = ci(q["EP_x100_E3"]["minus_P6"]) if "EP_x100_E3" in q else "-"
        gs = " | ".join(str(g["ok"][x]) for x in ("G1", "G2", "G3", "G4")) if g else "- | - | - | -"
        L.append(f"| {a} | {','.join(q['members'])} | {R.lvl_s(q['PDMS_x100_E3'])} | {ci(q['minus_P6'])} | {ep} | {q['zero_pick_share']:.5f} | {res['arm_pick_changed_vs_P6'][a]} | {gs} | `{res['arm_logits_sha256'][a][:16]}` |")
    L += ["", "## Reported pairs", ""] + [f"- {k2}: {ci(v)}" for k2, v in res["reported_pairs"].items()]
    L += ["", "## Four metric families vs P6 (adverse components per arm)", ""] + [f"- {a}: {q['adverse_components'] or 'none'}" for a, q in res["arms"].items() if "adverse_components" in q]
    L += [f"- STRATEGIC: {STRATEGIC}", "", "## Ambiguities", ""] + [f"- {x}" for x in res["ambiguities"]]
    return "\n".join(L) + "\n"


def write_outputs(res, out_dir, stem="div_read"):
    os.makedirs(out_dir, exist_ok=True)
    pub = {k: v for k, v in res.items() if not k.startswith("_")}
    jp, mp = os.path.join(out_dir, stem + ".json"), os.path.join(out_dir, stem + ".md")
    with open(jp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(pub, f, indent=1, default=R.jdefault)
    with open(mp, "w", encoding="utf-8", newline="\n") as f:
        f.write(to_markdown(pub))
    return jp, mp


# ------------------------------------------------------------------------------------------------ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["read"])
    ap.add_argument("--member", action="append", default=[], help="NAME=DIR override (repeatable); defaults are MEMBER_DIRS")
    ap.add_argument("--p6", default=P6_DIR_DEFAULT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default=RP.DEFAULT_TABLE)
    ap.add_argument("--card", default=CARD, help="the frozen card (sha256 checked)")
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--control", choices=["c1", "c2"], default=None, help="control run; never writes the ledger; scores only an already-read arm")
    ap.add_argument("--min-free-ram-gb", type=float, default=3.0)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    ok, act = card_ok(a.card)
    if not ok:
        print(f"REFUSED: card sha256 {act} != frozen {CARD_SHA256}")
        return 2
    dirs = dict(MEMBER_DIRS)
    for n, d in RP.parse_members(a.member):
        dirs[n] = d
    names = CONTROL_MEMBERS[a.control] if a.control else TEN
    members = [(n, dirs[n]) for n in names]
    missing = [(n, d) for n, d in members if not os.path.isfile(os.path.join(d, "e3_logits.npz"))]
    if missing:
        print(f"REFUSED: e3_logits.npz missing for {missing}")
        return 2
    RP.wait_free_ram(a.min_free_ram_gb)
    os.makedirs(a.out, exist_ok=True)
    ledger = None if (a.no_ledger or a.control) else a.ledger
    res = div_read(members, a.p6, truth_path=a.table, ledger_path=ledger, card_path=a.card, control=a.control)
    if res.get("status") == "READ" and a.control:
        res["control_result"] = control_result(a.control, res)
    jp, mp = write_outputs(res, a.out, "div_read" + (f"_control_{a.control}" if a.control else ""))
    summ = {"status": res.get("status"), "why": res.get("why"), "control": a.control, "control_PASS": (res.get("control_result") or {}).get("PASS"), "json": jp, "md": mp}
    if res.get("status") == "READ" and not a.control:
        summ.update(outcome=res["outcome"]["primary"], K=res["K_ledger"].get("K"))
    print(json.dumps(summ, indent=1, default=R.jdefault))
    if a.control:
        return 0 if (res.get("control_result") or {}).get("PASS") else 1
    return 0 if res.get("status") == "READ" else 2


if __name__ == "__main__":
    sys.exit(main())
