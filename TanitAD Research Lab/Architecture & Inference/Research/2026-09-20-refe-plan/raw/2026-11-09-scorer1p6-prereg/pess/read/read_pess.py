#!/usr/bin/env python3
"""read_pess.py -- the read tool of PREREG_PESS_P6 (frozen 2026-11-10, sha256 185de406...f42f380). CPU only, no GPU, no training.

    python read_pess.py read --member C0=DIR ... --member S1ZSM=DIR --p6 DIR --out OUT [--ledger PATH | --no-ledger] [--min-free-ram-gb 3]
    python read_pess.py read --control c1|c2|c3 --member ... --p6 DIR --out OUT       # controls; never write the ledger

What it does: combines the E3 logits of P6's scorer-only fine-tunes ("members") into three arms (head order NC, DAC, EP, TTC, C, DDC = 0..5)
    ENS9 = mean logit on every head
    PESS = MIN member logit on NC, DAC, TTC and mean on EP, C, DDC            (primary)
    LCB1 = mean - 1 x SD (ddof 0) on NC, DAC, TTC and mean elsewhere          (secondary, never adopted under the card)
picks with the unchanged read_scorer1p6.pick_logits (torch float32), scores each pick by LOOKUP in the full navtest truth table (64 proposals x 12,146 tokens;
the changed_from_table.py route), and reads bars P1, P2, M, G1 to G4, outcomes PE1 to PE4 / Unnamed, the four metric families vs P6 (STRATEGIC PAUSED),
EP x100 per arm and the K-ledger keys. The reader refuses to run if the card's sha256 differs from the frozen one.

REUSE, NOT A FORK: read_scorer1p6.py (pick_logits, gate_e2, gate_e3, BootSet, rec, g1_ok/g2_ok/g3_ok, geometry/tactical components, read_harness_csv, world_from_export) and,
for the ledger, selectors/selcore.py are imported unchanged.
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
EVAL_DIR = os.environ.get("PESS_EVAL_DIR", P6BASE + "/eval")
SEL_DIR = P6BASE + "/selectors"
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)
import read_scorer1p6 as R  # noqa: E402  (the Batch 1 reader, imported unchanged)

CARD = P6BASE + "/PREREG_PESS_P6.md"
CARD_SHA256 = "185de4067ede99ff9aca0c0f559e5a62f0989872aef07b8b7c4bd8657d42f380"
DEFAULT_TABLE = "F:/tanitad_scratch/selq2read/table_p6final_c.npz"
DEFAULT_E2_TOKENS = P6BASE + "/e2/e2_tokens.json"
DEFAULT_E2_TABLE = P6BASE + "/e2/e2_table_p6final.npz"
DEFAULT_LEDGER = SEL_LEDGER = SEL_DIR + "/ledger_K.json"
S1ZSM_BANKED_JSON = "F:/tanitad_scratch/s1zsmread/real/s1zsm_read.json"
S1ZSM_CSV = "F:/tanitad_scratch/p6arms/pass_s1zsm/S1ZSM/e3_changed_scores.csv"
ENS3_BANKED_JSON = "F:/tanitad_scratch/p6arms/read/verdict_ens3/scorer1p6_read.json"
ENS3_BANKED_SHA256 = "4cbe86d77ef9a6d3fadc9386cf223d3ca3cfb3137252a17c9018dac8aa5a0bfd"

HEAD = {"NC": 0, "DAC": 1, "EP": 2, "TTC": 3, "C": 4, "DDC": 5}
PESSIMISTIC_HEADS = (HEAD["NC"], HEAD["DAC"], HEAD["TTC"])          # min / mean - SD on these, mean elsewhere
EP_COL = HEAD["EP"]

# ------------------------------------------------------------------------------------------------ bars (each compared by a test to the card text)
P1_FLOOR = 0.4971            # card P1: d > t x s_p = 0.4971 (s_p 0.1791, df 4, t 2.776)
P1_S_P, P1_T = 0.1791, 2.776
G4_BAR = 0.10                # card G4: CI-hi < +0.10 (read_s1z.G4_BAR)
SLOW_RATIO = 0.85            # Batch 1 definition (read_s1z.SLOW_RATIO)
BARS = {"P1": f"PESS - P6: CI-lo > 0 and d > t*s_p = {P1_FLOOR}", "P2": "PESS - ENS9: CI-lo > 0", "M": "zero-pick share (pick PDMS exactly 0), PESS - ENS9: CI-hi < 0",
        "G1": "TTC, NC, DAC sub-scores x100 of the pick, PESS - P6: each point >= -0.5", "G2": "regret on E2, PESS - P6: CI-hi < +0.5",
        "G3": "overspeed share on E2 CRUISE, PESS - P6: CI-hi < +0.10", "G4": "slow-pick share on E2 CRUISE, PESS - P6: CI-hi < +0.10"}

PE_NAMES = {"PE1": "ADOPT", "PE2": "MECHANISM, TOO SMALL", "PE3": "MOVES ZERO PICKS, NOT PDMS", "PE4": "SHARED BLIND SPOT", "Unnamed": "UNNAMED"}
PE_ORDER = ("PE1", "PE2", "PE3", "PE4")
PE_NEXT = {
    "PE1": "PESS becomes P6's selector. Next: goal status vs 94.55; then the same rule with more diverse members (fine-tunes on disjoint navtrain halves; pod time, PI check)",
    "PE2": "member diversity (data-split fine-tunes, a PI check before pod time) under its own card",
    "PE3": "read EP and TTC of the changed tokens; next: pessimism on NC and DAC only (LCB1's result, read here, decides between min and 1 SD)",
    "PE4": ("the perception lever: a check of each proposal against an estimate of drivable area and agents built from richer features than the scorer reads (architecture change, the PI's "
            "decision), with the zero-GPU E1 evidence of the card as its design input"),
    "Unnamed": "matches none of PE1 to PE4: labelled as such",
}

AMBIGUITIES = [
    "T1 P1 d is the POINT difference PESS - P6 in PDMS x100 (E3), compared strictly with the constant 0.4971 printed in the card (not recomputed as 2.776 x 0.1791 = 0.49718); CI-lo > 0 strict. "
    "The earlier readers' 2F rule is not used: the card fixes the floor.",
    "T2 M is the share of E3 tokens whose pick has PDMS exactly 0.0 (card: 'pick PDMS exactly 0'), NOT the S1ZSM reader's M2 (PDMS 0 or TTC 0). The failure share (PDMS 0 or TTC 0) is also printed, as reported-not-deciding. "
    "CI over the same paired log-cluster draws as P1.",
    "T3 G1 to G4 are the Batch 1 / S1Z definitions applied to PESS - P6: G1 sub-score points x100 from the table's sub-scores at the pick (each >= -0.5, EPS slack of read_scorer1p6.g1_ok); G2 regret on the 2,000 E2 tokens "
    "(best valid proposal PDMS x100 minus the pick's); G3 overspeed (plan v4/v0 > 1.15) share on E2 CRUISE, CI-hi < +0.10; G4 slow (v4/v0 < 0.85) share on E2 CRUISE, CI-hi < +0.10 (EPS slack of read_s1z.g4_ok). "
    "E2 quantities use the combined E3 logits restricted to the E2 tokens (the E2 logit files of members are not read; a member's e2 logits could differ from its E3 rows by float drift).",
    "T4 PE labels overlap and every match is listed; the primary label follows PE1, PE2, PE3, PE4, Unnamed. PE1 = P1 and P2 and M and G1..G4; PE2 = P2 holds and P1 fails (M and the guards are not conditions of PE2); "
    "PE3 = M holds and P2 fails; PE4 = neither P2 nor M holds. P1 holds, P2 holds, M fails (or a guard fails) matches none -> Unnamed.",
    "T5 Scoring = lookup in the full truth table at the pick; the table's valid flag must be True and the value finite at every pick (else the read stops). A gate checks that the table reproduces P6's csv scores "
    "on 100 % of tokens (G0-TABLE) and the E2 table on the E2 tokens; a check on S1ZSM's own picks vs its e3_changed_scores.csv runs in control c1.",
    "T6 Combine in float32: mean = np.mean(axis=0, dtype=float32) (the ENS3 convention of read_scorer1p6); min = np.min; SD = ddof-0 population SD computed in float64 from the float32 member logits, "
    "LCB1 = mean32 - SD cast to float32. With one member min = mean and SD = 0 exactly, so every arm is that member's own logits bit for bit (control c1 asserts it).",
    "T7 The ledger key is <ARM>:<sha256 of the arm's combined float32 [12146,64,6] C-contiguous logits>:E3 for each of PESS, LCB1, ENS9 (K + 3). The arm name stays ENS9 even if only eight members are present; "
    "the members used are listed in the output and the sha differs by member set.",
    "T8 Members are loaded as given by --member; the card's nine are C0, C0r, C0r2, S1E, S1E1, S1Z, S1Z1, S1ZSM, S1ZD. If S1ZD is not given the read states it uses n members. "
    "Each member must pass G0-E3 (proposals bit-identical to P6's) and its token set must equal the table's.",
    "T9 Family clause: the Batch 1 family tables of the arm vs P6 (paired CIs) incl. EP x100; STRATEGIC printed as PAUSED with the PI ruling. EP x100 vs P6 is reported for every arm.",
    "T10 The Bonferroni intervals at the ledger's running K are printed per arm for best-of-K statements; P1 and P2 are PESS's own tests.",
]


# ------------------------------------------------------------------------------------------------ small helpers
def sha256_file(p):
    return R.sha256_file(p)


def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _pub(d):
    return json.loads(json.dumps(d, default=R.jdefault))


def card_ok(card_path=CARD):
    act = sha256_file(card_path) if os.path.exists(card_path) else None
    return act == CARD_SHA256, act


# ------------------------------------------------------------------------------------------------ the combine rules (pure)
def _stack(members):
    arrs = [np.asarray(m, np.float32) for m in members]
    if not arrs or any(a.shape != arrs[0].shape for a in arrs):
        raise ValueError("members must be a non-empty list of equally shaped logit arrays")
    return arrs


def combine(members, rule):
    """members: list of [N,64,6] float32 logits -> combined float32 [N,64,6]. rule in ENS9, PESS, PESS_MAX (control C3 only), LCB1."""
    arrs = _stack(members)
    S = np.stack(arrs, axis=0)
    mean = np.mean(S, axis=0, dtype=np.float32)
    if rule == "ENS9":
        return mean
    out = mean.copy()
    cols = list(PESSIMISTIC_HEADS)
    if rule == "PESS":
        out[..., cols] = np.min(S[..., cols], axis=0)
    elif rule == "PESS_MAX":
        out[..., cols] = np.max(S[..., cols], axis=0)
    elif rule == "LCB1":
        sd = np.std(S[..., cols].astype(np.float64), axis=0, ddof=0)
        out[..., cols] = (mean[..., cols].astype(np.float64) - 1.0 * sd).astype(np.float32)
    else:
        raise ValueError("unknown rule " + rule)
    return np.ascontiguousarray(out, dtype=np.float32)


# ------------------------------------------------------------------------------------------------ outcome (pure)
def p1_clears(point, lo):
    return None if point is None or lo is None else bool(lo > 0 and point > P1_FLOOR)


def p2_clears(lo):
    return None if lo is None else bool(lo > 0)


def m_clears(hi):
    return None if hi is None else bool(hi < 0)


def decide_outcome(p1, p2, m, guards):
    """p1, p2, m: bool|None; guards {"G1".."G4": bool|None}. -> dict(matches, primary, name, adopted, failed_guards, next_levers)"""
    if any(v is None for v in (p1, p2, m)) or not guards or any(v is None for v in guards.values()):
        return {"matches": ["Unnamed"], "primary": "Unnamed", "name": PE_NAMES["Unnamed"], "adopted": False, "failed_guards": [], "next_levers": {"Unnamed": PE_NEXT["Unnamed"]},
                "why": "an input is undefined: the card's Unnamed rule"}
    failed = [k for k in ("G1", "G2", "G3", "G4") if not guards[k]]
    mt = []
    if p1 and p2 and m and not failed:
        mt.append("PE1")
    if p2 and not p1:
        mt.append("PE2")
    if m and not p2:
        mt.append("PE3")
    if not p2 and not m:
        mt.append("PE4")
    prim = next((z for z in PE_ORDER if z in mt), "Unnamed")
    return {"matches": mt or ["Unnamed"], "primary": prim, "name": PE_NAMES[prim], "adopted": "PE1" in mt, "failed_guards": failed,
            "next_levers": {z: PE_NEXT[z] for z in (mt or ["Unnamed"])}}


# ------------------------------------------------------------------------------------------------ loading
def parse_members(specs):
    out = []
    for s in specs:
        if "=" not in s:
            raise SystemExit(f"--member wants NAME=DIR, got {s!r}")
        n, d = s.split("=", 1)
        if n in [o[0] for o in out]:
            raise SystemExit(f"duplicate member name {n}")
        out.append((n, d))
    return out


def free_ram_gb():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    s = MS()
    s.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ap / 2 ** 30


def wait_free_ram(min_gb=3.0, timeout_s=1800):
    t0 = time.time()
    while True:
        try:
            f = free_ram_gb()
        except Exception:   # noqa: BLE001 -- non-Windows: no gate
            return None
        if f >= min_gb:
            return f
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"REFUSING: free RAM {f:.2f} GB < {min_gb} GB after {timeout_s}s")
        time.sleep(15)


class Truth:
    """the full navtest truth table: token, log, pdms [N,64], sub [N,64,6], valid, pick, logits"""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as T:
            self.token = np.array([str(t) for t in T["token"]])
            self.pdms, self.sub, self.valid = T["pdms"].astype(np.float64), T["sub"].astype(np.float64), T["valid"].astype(bool)
            self.pick, self.logits = T["pick"].astype(np.int64), T["logits"]
        self.path = path
        self.N = len(self.token)

    def lookup(self, pk):
        r = np.arange(self.N)
        v = self.valid[r, pk]
        d = self.pdms[r, pk]
        if not (v.all() and np.isfinite(d).all()):
            raise SystemExit(f"REFUSING: the table has no valid finite score at {int((~(v & np.isfinite(d))).sum())} picks")
        return d, self.sub[r, pk]


def load_member(name, d, truth, p6_props, world_tokens):
    """-> (aligned logits float32 [N,64,6], G0-E3 gate dict, sha256 of the e3_logits.npz file)"""
    f = os.path.join(d, "e3_logits.npz")
    Z = np.load(f, allow_pickle=False)
    pk = "proposals" if "proposals" in Z.files else "props8"
    arm = R.Arm(name, Z["token"], Z["logits"], Z[pk])
    g = R.gate_e3(arm, _W(world_tokens), p6_props)
    ix = g.pop("_ix", None)
    if not g["PASS"]:
        return None, g, sha256_file(f)
    return np.ascontiguousarray(np.asarray(arm.e3_logits)[ix], dtype=np.float32), g, sha256_file(f)


class _W:
    """minimal world stand-in for R.gate_e3 (it reads .token only)"""

    def __init__(self, token):
        self.token = np.asarray(token)


# ------------------------------------------------------------------------------------------------ label-free quantities (control C3)
def labelfree_compare(members, arm_a="PESS", arm_b="PESS_MAX"):
    a, b = combine(members, arm_a), combine(members, arm_b)
    pa, pb = R.pick_logits(a), R.pick_logits(b)
    return {"arm_a": arm_a, "arm_b": arm_b, "n_tokens": int(len(pa)), "n_pick_differs": int((pa != pb).sum()), "share_pick_differs": float((pa != pb).mean()),
            "sha256_a": sha256_array(a), "sha256_b": sha256_array(b), "different_arms": bool(sha256_array(a) != sha256_array(b))}


# ------------------------------------------------------------------------------------------------ the read
def e2_quantities(W, truth, e2_tokens, e2_table, p6_props):
    wi2 = np.array([W.index[t] for t in e2_tokens])
    tpos = {t: i for i, t in enumerate(e2_table["token"])}
    ti = np.array([tpos[t] for t in e2_tokens])
    T_pd, T_val = e2_table["pdms"][ti], e2_table["valid"][ti]
    best2 = np.where(T_val, T_pd, -1.0).max(1) * 100.0
    ratio2 = R.plan_v4(p6_props[wi2]) / np.maximum(W.v0[wi2], 1e-6)[:, None]
    cru2 = W.cruise()[wi2]
    return dict(wi2=wi2, T_pd=T_pd, best2=best2, ratio2=ratio2, cru2=cru2, B2=R.BootSet(W.log[wi2]), r2=np.arange(len(wi2)))


def guards_vs_p6(E, pk_arm, pk_p6, sb_arm, sb_p6, B3):
    """G1..G4 of an arm against P6. pk_*: E3 picks [N]. -> dict(blocks, ok)"""
    r2, wi2, B2, cru2 = E["r2"], E["wi2"], E["B2"], E["cru2"]
    out = {}
    g1 = {}
    for s in ("TTC", "NC", "DAC"):
        j = R.ALL_HEADS[s]
        pt, rr, n = B3.diff(sb_arm[:, j], sb_p6[:, j])
        g1[s] = {**R.rec(pt, rr, n), "arm_mean_x100": float(sb_arm[:, j].mean()), "P6_mean_x100": float(sb_p6[:, j].mean()), "ok": R.g1_ok(pt)}
    out["G1"] = {"per_head_points_vs_P6": g1, "bar": "each point >= -0.5", "ok": bool(all(v["ok"] for v in g1.values())), "failed_heads": [s for s, v in g1.items() if not v["ok"]]}
    a2, p2 = pk_arm[wi2], pk_p6[wi2]
    reg = lambda pk: E["best2"] - E["T_pd"][r2, pk] * 100.0          # noqa: E731
    pt, rr, n = B2.diff(reg(a2), reg(p2))
    d = R.rec(pt, rr, n)
    out["G2"] = {"delta_regret_vs_P6": {**d, "arm_regret": float(reg(a2).mean()), "P6_regret": float(reg(p2).mean())}, "bar": "CI-hi < +0.5 points", "ok": bool(R.g2_ok(d["hi"]))}
    ov = lambda pk: (E["ratio2"][r2, pk] > R.OVERSPEED).astype(float)   # noqa: E731
    pt, rr, n = B2.diff(ov(a2), ov(p2), cru2)
    d = R.rec(pt, rr, n)
    out["G3"] = {"overspeed_diff_vs_P6": {**d, "arm_share": R.fin(B2.stat(ov(a2), cru2)[0]), "P6_share": R.fin(B2.stat(ov(p2), cru2)[0])}, "bar": "CI-hi < +0.10", "ok": bool(R.g3_ok(d["hi"]))}
    sl = lambda pk: (E["ratio2"][r2, pk] < SLOW_RATIO).astype(float)    # noqa: E731
    pt, rr, n = B2.diff(sl(a2), sl(p2), cru2)
    d = R.rec(pt, rr, n)
    out["G4"] = {"slow_diff_vs_P6": {**d, "arm_share": R.fin(B2.stat(sl(a2), cru2)[0]), "P6_share": R.fin(B2.stat(sl(p2), cru2)[0])}, "bar": "CI-hi < +0.10", "n_cruise_tokens_E2": int(cru2.sum()),
                 "ok": bool(R.lt(d["hi"], G4_BAR))}
    out["ok"] = {k: out[k]["ok"] for k in ("G1", "G2", "G3", "G4")}
    return out


def family_vs_p6(W, props, truth_sub, pk_arm, pk_p6, B3, E, tactical=True):
    """the Batch 1 family table of an arm against P6 (paired CIs). -> (rows, adverse list)"""
    N = len(pk_arm)
    r = np.arange(N)

    def comps(pk):
        c = {}
        for j, s in enumerate(R.SUBN):
            fam_name = {"EP": "longitudinal", "TTC": "longitudinal", "DAC": "lateral"}.get(s, "pdms-other")
            c[s] = (truth_sub[r, pk, j] * 100.0, "higher", fam_name)
        pose = props[r, pk]
        c.update(R.geometry_components(pose, W.human8))
        if tactical and W.cmd is not None:
            c.update(R.tactical_components(pose, W.human8, W.cmd))
        return c

    a, b = comps(pk_arm), comps(pk_p6)
    rows, adv = {}, []
    for k, v in a.items():
        if not isinstance(v, tuple):
            rows[k] = {"status": v if isinstance(v, str) else str(v)}
            continue
        vals, dr, fm = v
        pt, rr_, n = B3.diff(vals, b[k][0])
        rr = R.rec(pt, rr_, n)
        rr.update(direction=dr, family=fm, arm_mean=R.fin(np.nanmean(vals)), P6_mean=R.fin(np.nanmean(b[k][0])))
        rr["adverse"] = R.adverse(dr, rr["lo"], rr["hi"]) if fm in ("longitudinal", "lateral") else False
        rows[k] = rr
        if rr["adverse"]:
            adv.append(k)
    rows["_distance_keeping"] = "UNAVAILABLE at this tier (no lead-agent block in NAVSIM open loop); read in card section 6 of PREREG_SCORER1P6"
    return rows, adv


def pess_read(members, p6_dir, truth_path=DEFAULT_TABLE, arms=("ENS9", "PESS", "LCB1"), e2_tokens_path=DEFAULT_E2_TOKENS, e2_table_path=DEFAULT_E2_TABLE, ledger_path=None,
              card_path=CARD, control=None, p6_proposals=R.DEFAULT_P6_PROPOSALS, p6_csv=R.DEFAULT_P6_CSV, export=R.DEFAULT_EXPORT):
    """members: list of (name, dir). arms: which of ENS9/PESS/LCB1 to score (controls restrict it). -> result dict (status READ | READ STOPPED)."""
    t0 = time.time()
    ok, act = card_ok(card_path)
    res = {"tool": "read_pess.py", "card": {"path": card_path, "sha256_expected": CARD_SHA256, "sha256_actual": act}, "tier": R.TIER, "evidence_class": R.EVIDENCE,
           "estimator": R.ESTIMATOR + " Training variance is not resampled: ensembles of fixed members are read as fixed scorers. Inference is deterministic.", "bars": BARS,
           "strategic": R.STRATEGIC, "ambiguities": AMBIGUITIES, "members": [n for n, _ in members], "n_members": len(members), "control": control, "arms_scored": list(arms)}
    if not ok:
        res.update(status="READ STOPPED", why="REFUSED: the card's sha256 differs from the frozen one; the bars may have moved")
        return res
    res["code_sha256"] = {"read_pess.py": sha256_file(HERE / "read_pess.py") if (HERE / "read_pess.py").exists() else None, "read_scorer1p6.py": sha256_file(Path(EVAL_DIR) / "read_scorer1p6.py")}
    truth = Truth(truth_path)
    N = truth.N
    P6 = np.load(p6_proposals, allow_pickle=False)
    key = "proposals" if "proposals" in P6.files else "props8"
    ixp, err = R._align(P6["token"], list(truth.token), "p6 proposals")
    if err:
        res.update(status="READ STOPPED", why=err)
        return res
    p6_props = P6[key][ixp]
    p6_banked = P6["pick"][ixp].astype(np.int64) if "pick" in P6.files else None
    gates = {}
    # ---- P6: G0-PICK (rule on the dir's logits == banked pick), the dir's logits == the table's, G0-P6CSV (table reproduces the P6 csv on 100 %)
    Zp = np.load(os.path.join(p6_dir, "e3_logits.npz"), allow_pickle=False)
    jp, err = R._align(Zp["token"], list(truth.token), "P6 dir logits")
    if err:
        res.update(status="READ STOPPED", why=err)
        return res
    p6_logits = np.asarray(Zp["logits"])[jp]
    p6_pick = R.pick_logits(p6_logits)
    gates["G0-PICK"] = {"name": "G0-PICK", "n_equal_to_banked_pick": (int((p6_pick == p6_banked).sum()) if p6_banked is not None else None), "n": N,
                        "PASS": bool(p6_banked is None or (p6_pick == p6_banked).all())}
    gates["G0-P6LOGITS"] = {"name": "G0-P6LOGITS", "P6 dir logits bit-identical to the table's logits": R.bits_equal(p6_logits, truth.logits), "PASS": bool(R.bits_equal(p6_logits, truth.logits))}
    rows = R.read_harness_csv(p6_csv)
    absent = [t for t in truth.token if t not in rows]
    d6, sb6_raw = truth.lookup(p6_pick)
    csvs = np.array([rows[t][1] for t in truth.token if t in rows])
    n_eq = int((csvs == d6[[i for i, t in enumerate(truth.token) if t in rows]]).sum()) if not absent else 0
    gates["G0-TABLE"] = {"name": "G0-TABLE", "n_tokens_absent_in_p6_csv": len(absent), "n_pdms_equal_to_p6_csv": n_eq, "PASS": bool(not absent and n_eq == N)}
    world = R.world_from_export(export, [str(t) for t in truth.token])
    W = world
    e2_tokens = json.load(open(e2_tokens_path, encoding="utf-8"))["tokens"]
    T2 = np.load(e2_table_path, allow_pickle=False)
    e2_table = dict(token=np.array([str(t) for t in T2["token"]]), pdms=T2["pdms"].astype(np.float64), sub=T2["sub"].astype(np.float64), valid=T2["valid"].astype(bool),
                    red=(T2["red_pick_pdms"] if "red_pick_pdms" in T2.files else None))
    gates["G0-E2"] = R.gate_e2(e2_table, e2_tokens, W, rows, p6_logits, p6_pick)
    tposf = {t: i for i, t in enumerate(truth.token)}
    wi = np.array([tposf[t] for t in e2_tokens])
    tpos2 = {t: i for i, t in enumerate(e2_table["token"])}
    ti2 = np.array([tpos2[t] for t in e2_tokens])
    gates["G0-E2-vs-TABLE"] = {"name": "G0-E2-vs-TABLE", "E2 table pdms bit-identical to the full table on the E2 tokens": bool((e2_table["pdms"][ti2] == truth.pdms[wi]).all()),
                               "PASS": bool((e2_table["pdms"][ti2] == truth.pdms[wi]).all() and (e2_table["sub"][ti2] == truth.sub[wi]).all())}
    # ---- members
    mem_logits, mem_sha = [], {}
    for n, d in members:
        L, g, sha = load_member(n, d, truth, p6_props, truth.token)
        gates[g["name"]] = g
        mem_sha[n] = sha
        if L is not None:
            mem_logits.append(L)
    res["gates"] = gates
    res["member_e3_logits_sha256"] = mem_sha
    if not all(g["PASS"] for g in gates.values()):
        res.update(status="READ STOPPED", why="a gate failed: " + ", ".join(k for k, g in gates.items() if not g["PASS"]) + "; no verdict is written")
        return res
    # ---- arms
    sb6 = sb6_raw * 100.0
    B3 = R.BootSet(W.log)
    E = e2_quantities(W, truth, e2_tokens, e2_table, p6_props)
    comb, pick, pdms, sub = {}, {}, {"P6": d6}, {"P6": sb6}
    pick["P6"] = p6_pick
    for a in arms:
        comb[a] = combine(mem_logits, a)
        pick[a] = R.pick_logits(comb[a])
        pdms[a], s = truth.lookup(pick[a])
        sub[a] = s * 100.0
    res["arm_logits_sha256"] = {a: sha256_array(comb[a]) for a in arms}
    res["arm_pick_changed_vs_P6"] = {a: int((pick[a] != p6_pick).sum()) for a in arms}
    res["arm_pick_changed_vs_ENS9"] = {a: int((pick[a] != pick["ENS9"]).sum()) for a in arms if a != "ENS9"} if "ENS9" in arms else {}
    arms_out = {}
    zero = {nm: (pdms[nm] == 0.0).astype(np.float64) for nm in pdms}
    fail = {nm: ((pdms[nm] <= 0.0) | (sub[nm][:, R.ALL_HEADS["TTC"]] <= 0.0)).astype(np.float64) for nm in pdms}
    for a in arms:
        pt, rr, n = B3.stat(pdms[a] * 100.0)
        lvl = R.rec(pt, rr, n)
        pt, rr, n = B3.diff(pdms[a] * 100.0, pdms["P6"] * 100.0)
        vs_p6 = R.rec(pt, rr, n)
        rr_p6 = rr
        o = {"PDMS_x100_E3": lvl, "minus_P6": vs_p6, "P1_vs_P6": {**vs_p6, "floor": P1_FLOOR, "clears": p1_clears(vs_p6["point"], vs_p6["lo"])}}
        pt, rr, n = B3.diff(sub[a][:, EP_COL], sub["P6"][:, EP_COL])
        o["EP_x100_E3"] = {"arm_mean": float(sub[a][:, EP_COL].mean()), "P6_mean": float(sub["P6"][:, EP_COL].mean()), "minus_P6": R.rec(pt, rr, n)}
        pt, rr, n = B3.diff(fail[a], fail["P6"])
        o["failure_share_minus_P6_reported"] = R.rec(pt, rr, n)
        o["zero_pick_share"] = R.fin(zero[a].mean())
        o["G"] = guards_vs_p6(E, pick[a], p6_pick, sub[a], sb6, B3)
        o["family_vs_P6"], o["adverse_components"] = family_vs_p6(W, p6_props, truth.sub, pick[a], p6_pick, B3, E)
        if a != "ENS9" and "ENS9" in arms:
            pt, rr, n = B3.diff(pdms[a] * 100.0, pdms["ENS9"] * 100.0)
            d2 = R.rec(pt, rr, n)
            pt, rr, n = B3.diff(zero[a], zero["ENS9"])
            dm = R.rec(pt, rr, n)
            o["P2_vs_ENS9"] = {**d2, "clears": p2_clears(d2["lo"])}
            o["M_zero_pick_share_minus_ENS9"] = {**dm, "arm_share": R.fin(zero[a].mean()), "ENS9_share": R.fin(zero["ENS9"].mean()), "clears": m_clears(dm["hi"]),
                                                 "n_pick_pdms_zero": int((pdms[a] == 0.0).sum()), "n_pick_pdms_zero_ENS9": int((pdms["ENS9"] == 0.0).sum())}
        o["_rr_vs_p6"] = rr_p6
        arms_out[a] = o
    # ---- outcome (PESS is the card's arm; LCB1 is reported with the same bars and cannot be adopted under the card)
    if "PESS" in arms and "ENS9" in arms:
        o = arms_out["PESS"]
        p1, p2, m = o["P1_vs_P6"]["clears"], o["P2_vs_ENS9"]["clears"], o["M_zero_pick_share_minus_ENS9"]["clears"]
        res["outcome"] = decide_outcome(p1, p2, m, o["G"]["ok"])
        res["outcome"]["inputs"] = {"P1": p1, "P2": p2, "M": m, "guards": o["G"]["ok"], "point_PESS": o["PDMS_x100_E3"]["point"]}
        res["goal_status_PESS"] = R.goal_status(o["PDMS_x100_E3"]["point"], o["PDMS_x100_E3"]["hi"])
        if "LCB1" in arms_out:
            l = arms_out["LCB1"]
            res["LCB1_reported_not_adopted"] = {"P1": l["P1_vs_P6"]["clears"], "P2": l["P2_vs_ENS9"]["clears"], "M": l["M_zero_pick_share_minus_ENS9"]["clears"], "guards": l["G"]["ok"],
                                                "note": "read with the same bars; can only name the next lever (card section 2)"}
    # ---- ledger and Bonferroni intervals (never in a control)
    kinfo = {"ledger": None, "K": None, "note": "no ledger written: K NOT COUNTED"}
    if ledger_path and not control:
        if SEL_DIR not in sys.path:
            sys.path.insert(0, SEL_DIR)
        import selcore
        led = selcore.Ledger(ledger_path)
        keys, added = {}, {}
        for a in arms:
            k = f"{a}:{res['arm_logits_sha256'][a]}:E3"
            keys[a] = k
            added[a] = bool(led.add(k, nominee=a, family="PESS", set="E3", smoke=False))
        K = led.K
        bon = {}
        for a in arms:
            lo, hi = selcore.interval(arms_out[a]["_rr_vs_p6"][np.isfinite(arms_out[a]["_rr_vs_p6"])], K)
            bon[a] = {"point": arms_out[a]["minus_P6"]["point"], "lo": lo, "hi": hi}
        kinfo = {"ledger": ledger_path, "K": K, "keys": keys, "this_read_added": added, "bonferroni_pdms_minus_P6": bon,
                 "note": "K counts every navtest read in the programme's ledger; best-of-K statements use these intervals; P1 and P2 are PESS's own tests"}
    for o in arms_out.values():
        o.pop("_rr_vs_p6", None)
    res.update(status="READ", arms=arms_out, K_ledger=kinfo, seconds=round(time.time() - t0, 1))
    res["_pdms_by_arm"] = {a: pdms[a] for a in pdms}      # stripped before writing; used by controls
    res["_pick"] = pick
    res["_comb"] = comb
    res["_truth"] = truth
    res["_mem_logits"] = mem_logits
    return res


# ------------------------------------------------------------------------------------------------ controls
def control_c1(members, res):
    """members = {S1ZSM} only: every arm collapses to the member's own logits; PDMS x100 equals the banked 91.297 to float equality; the table gives S1ZSM's csv scores exactly"""
    out = {"control": "c1"}
    bank = json.load(open(S1ZSM_BANKED_JSON, encoding="utf-8"))["P1"]["PDMS_x100_E3"]["point"]
    out["banked_PDMS_x100_E3"] = bank
    out["banked_path"] = S1ZSM_BANKED_JSON
    out["banked_json_sha256"] = sha256_file(S1ZSM_BANKED_JSON)
    L = res["_mem_logits"][0]
    out["n_members"] = len(res["_mem_logits"])
    out["arm_logits_equal_member_bits"] = {a: R.bits_equal(res["_comb"][a], L) for a in res["_comb"]}
    out["arm_PDMS_x100_E3"] = {a: float(res["_pdms_by_arm"][a].mean() * 100.0) for a in res["_comb"]}
    out["arm_PDMS_equals_banked_exactly"] = {a: bool(out["arm_PDMS_x100_E3"][a] == bank) for a in out["arm_PDMS_x100_E3"]}
    # table vs S1ZSM's own harness csv at S1ZSM's picks
    rows = R.read_harness_csv(S1ZSM_CSV)
    truth = res["_truth"]
    pos = {t: i for i, t in enumerate(truth.token)}
    pk = res["_pick"]["ENS9"]
    bad, n = [], 0
    for t, (v, s, sb) in rows.items():
        i = pos[t]
        k = int(pk[i])
        n += 1
        if not (v == bool(truth.valid[i, k]) and s == truth.pdms[i, k] and np.array_equal(sb, truth.sub[i, k])):
            bad.append(t)
    changed = int((pk != R.pick_logits(truth.logits)).sum())
    out["table_vs_s1zsm_csv"] = {"csv": S1ZSM_CSV, "csv_sha256": sha256_file(S1ZSM_CSV), "n_csv_rows": n, "n_changed_picks_vs_P6": changed, "n_rows_not_equal": len(bad), "rows_cover_every_changed_pick": bool(n == changed)}
    out["PASS"] = bool(out["n_members"] == 1 and all(out["arm_logits_equal_member_bits"].values()) and all(out["arm_PDMS_equals_banked_exactly"].values()) and not bad and n == changed)
    return out


def control_c2(res):
    """members = {C0, C0r, S1E}, arm ENS9 only: ENS9 - P6 equals the banked ENS3 verdict's point to 1e-4"""
    d = json.load(open(ENS3_BANKED_JSON, encoding="utf-8"))["ENS3"]
    a = res["arms"]["ENS9"]
    out = {"control": "c2", "banked_path": ENS3_BANKED_JSON, "banked_sha256_actual": sha256_file(ENS3_BANKED_JSON), "banked_sha256_noted": ENS3_BANKED_SHA256,
           "banked_diff_vs_P6": d["diff_vs_base"], "banked_PDMS_x100_E3": d["pdms_e3"]["point"], "mine_diff_vs_P6": a["minus_P6"], "mine_PDMS_x100_E3": a["PDMS_x100_E3"]["point"],
           "abs_diff_point": abs(a["minus_P6"]["point"] - d["diff_vs_base"]["point"]), "abs_diff_lo": abs(a["minus_P6"]["lo"] - d["diff_vs_base"]["lo"]),
           "abs_diff_hi": abs(a["minus_P6"]["hi"] - d["diff_vs_base"]["hi"]), "n_changed_picks_vs_P6_banked": d["n_changed_picks_vs_P6"],
           "n_changed_picks_vs_P6_mine": res["arm_pick_changed_vs_P6"]["ENS9"], "arms_scored": res["arms_scored"]}
    out["PASS"] = bool(out["abs_diff_point"] < 1e-4 and res["arms_scored"] == ["ENS9"] and out["banked_sha256_actual"] == ENS3_BANKED_SHA256)
    return out


def control_c3(members_logits):
    """label-free: PESS with MAX instead of MIN on the members available. No table, no scores."""
    c = labelfree_compare(members_logits, "PESS", "PESS_MAX")
    c.update(control="c3", n_members=len(members_logits), required_min_share=0.10)
    c["PASS"] = bool(c["share_pick_differs"] >= 0.10 and c["different_arms"])
    c["labelled_as_different_arm"] = "PESS_MAX (deliberate regression) is labelled a different arm: its combined-logit sha256 differs from PESS's"
    return c


# ------------------------------------------------------------------------------------------------ markdown / outputs
def to_markdown(res):
    L = []
    ci = R.ci_s
    L += ["# PESS read" + (f" (CONTROL {res['control']})" if res.get("control") else ""), "", f"**Status: {res.get('status')}**", ""]
    if res.get("control"):
        L += ["**CONTROL RUN: not a PESS result. The numbers below are the control's; the ledger is never written.**", ""]
    L += [f"- Card: `{res['card']['path']}` sha256 `{res['card']['sha256_actual']}` (frozen `{res['card']['sha256_expected']}`)", f"- Tier: {R.TIER['E3']}", f"- Evidence class: {R.EVIDENCE}.",
          f"- Estimator: {res['estimator']}", f"- Members ({res['n_members']}): {', '.join(res['members'])}", f"- Arms scored: {', '.join(res.get('arms_scored', []))}", "", "## Gates", ""]
    for k, g in (res.get("gates") or {}).items():
        L.append(f"- {k}: **{'PASS' if g['PASS'] else 'FAIL'}** " + ", ".join(f"{a}={b}" for a, b in g.items() if a not in ("PASS", "name")))
    if res.get("status") != "READ":
        L += ["", f"**{res.get('status')}: {res.get('why')}**", "", "No verdict is written."]
        return "\n".join(L) + "\n"
    if res.get("outcome"):
        o = res["outcome"]
        L += ["", "## Verdict", "", f"- Outcome: **{o['primary']} {o['name']}** (all matches: {', '.join(o['matches'])})"]
        L += [f"  - {z}: {o['next_levers'][z]}" for z in o["matches"]]
        L += [f"- Inputs: {json.dumps(o['inputs'], default=R.jdefault)}", f"- Goal status of PESS: **{res.get('goal_status_PESS')}**"]
    L += ["", f"- K ledger: K = {res['K_ledger'].get('K')} ({res['K_ledger'].get('note')})", "", "## Per arm vs P6 (E3, PDMS x100; paired log-cluster bootstrap)", "",
          "| arm | PDMS | minus P6 | P1 clears | EP x100 minus P6 | zero-pick share | picks changed vs P6 | logits sha256 |", "|---|---|---|---|---|---|---|---|"]
    for a, o in res["arms"].items():
        L.append(f"| {a} | {R.lvl_s(o['PDMS_x100_E3'])} | {ci(o['minus_P6'])} | {o['P1_vs_P6']['clears']} | {ci(o['EP_x100_E3']['minus_P6'])} | {o['zero_pick_share']:.5f} | "
                 f"{res['arm_pick_changed_vs_P6'][a]} | `{res['arm_logits_sha256'][a][:16]}` |")
    L += ["", "## P2, M and guards", "", "| arm | P2 (arm - ENS9) | M zero-pick share (arm - ENS9) | G1 | G2 | G3 | G4 |", "|---|---|---|---|---|---|---|"]
    for a, o in res["arms"].items():
        p2 = o.get("P2_vs_ENS9")
        m = o.get("M_zero_pick_share_minus_ENS9")
        g = o["G"]
        L.append(f"| {a} | {ci(p2) + ' ' + str(p2['clears']) if p2 else '-'} | {ci(m) + ' ' + str(m['clears']) if m else '-'} | "
                 + " | ".join(f"{g[k]['ok']}" for k in ("G1", "G2", "G3", "G4")) + " |")
    L += ["", "## Four metric families vs P6 (per arm: adverse components)", ""]
    for a, o in res["arms"].items():
        L.append(f"- {a}: adverse components: {o['adverse_components'] or 'none'}")
    L += [f"- STRATEGIC: {R.STRATEGIC}", "", "## Ambiguities", ""] + [f"- {x}" for x in res["ambiguities"]]
    return "\n".join(L) + "\n"


def write_outputs(res, out_dir, stem="pess_read"):
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
    ap.add_argument("--member", action="append", default=[], help="NAME=DIR (repeatable); DIR holds e3_logits.npz")
    ap.add_argument("--p6", required=True, help="P6 pass dir holding e3_logits.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--e2-tokens", default=DEFAULT_E2_TOKENS)
    ap.add_argument("--e2-table", default=DEFAULT_E2_TABLE)
    ap.add_argument("--p6-proposals", default=R.DEFAULT_P6_PROPOSALS)
    ap.add_argument("--p6-csv", default=R.DEFAULT_P6_CSV)
    ap.add_argument("--export", default=R.DEFAULT_EXPORT)
    ap.add_argument("--card", default=CARD, help="the frozen card (sha256 checked)")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--control", choices=["c1", "c2", "c3"], default=None, help="control run; never writes the ledger")
    ap.add_argument("--min-free-ram-gb", type=float, default=3.0)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    ok, act = card_ok(a.card)
    if not ok:
        print(f"REFUSED: card sha256 {act} != frozen {CARD_SHA256}")
        return 2
    members = parse_members(a.member)
    if not members:
        print("REFUSED: no --member given")
        return 2
    if a.control == "c1" and [n for n, _ in members] != ["S1ZSM"]:
        print("REFUSED: control c1 takes exactly one member, S1ZSM")
        return 2
    wait_free_ram(a.min_free_ram_gb)
    os.makedirs(a.out, exist_ok=True)
    if a.control == "c3":
        # label-free: logits only, no truth table, no score, no K key
        mem = []
        for n, d in members:
            Z = np.load(os.path.join(d, "e3_logits.npz"), allow_pickle=False)
            mem.append(np.ascontiguousarray(Z["logits"], dtype=np.float32))
        c = control_c3(mem)
        c.update(members=[n for n, _ in members], card_sha256=act, note="label-free control: no truth table, no PDMS, no ledger")
        with open(os.path.join(a.out, "control_c3.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(c, f, indent=1)
        print(json.dumps(c, indent=1))
        return 0 if c["PASS"] else 1
    arms = ("ENS9",) if a.control == "c2" else ("ENS9", "PESS", "LCB1")
    ledger = None if (a.no_ledger or a.control) else a.ledger
    res = pess_read(members, a.p6, truth_path=a.table, arms=arms, e2_tokens_path=a.e2_tokens, e2_table_path=a.e2_table, ledger_path=ledger, card_path=a.card, control=a.control,
                    p6_proposals=a.p6_proposals, p6_csv=a.p6_csv, export=a.export)
    if res.get("status") == "READ" and a.control == "c1":
        res["control_result"] = control_c1(members, res)
    elif res.get("status") == "READ" and a.control == "c2":
        res["control_result"] = control_c2(res)
    jp, mp = write_outputs(res, a.out, "pess_read" + (f"_control_{a.control}" if a.control else ""))
    summ = {"status": res.get("status"), "why": res.get("why"), "control": a.control, "control_PASS": (res.get("control_result") or {}).get("PASS"), "json": jp, "md": mp}
    if res.get("status") == "READ" and not a.control:
        summ.update(outcome=res.get("outcome", {}).get("primary"), K=res["K_ledger"].get("K"))
    print(json.dumps(summ, indent=1, default=R.jdefault))
    if a.control:
        return 0 if (res.get("control_result") or {}).get("PASS") else 1
    return 0 if res.get("status") == "READ" else 2


if __name__ == "__main__":
    sys.exit(main())
