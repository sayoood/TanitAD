#!/usr/bin/env python3
"""read_scorer1p6.py -- the read tool of PREREG_SCORER1P6 (frozen 2026-11-09) sections 3 to 5. CPU only. It needs only the arms' outputs.

    python read_scorer1p6.py read   --p6-proposals P --p6-csv C --e2-tokens J --e2-table T --arm C0=DIR --arm C0r=DIR --arm S1E=DIR [--arm S1=DIR] --out OUT
    python read_scorer1p6.py emit-changed ... --arm NAME=DIR|ENS3 --out OUT     # seams + token json for the harness for tokens whose pick has no score yet
    python read_scorer1p6.py selfcheck --out OUT                                 # real data: P6 as a fake arm against itself (C3 stand-in, see HANDOFF.md)

WHAT IT DOES (card sections): G0-E2/G0-E3/G0-PICK gates; P1 (PDMS on E3, paired log-cluster bootstrap, floor F = |C0r - C0|); M1 (three-head within-token Spearman gap on E2 CRUISE,
average ranks); G1 to G3; goal status; the Adopted-arm rule; the family clause (ADOPT or PDMS GAIN WITH COST, each adverse component named with its paired CI); outcomes O1 to O6
or Unnamed; ENS3 AFTER the verdict. STRATEGIC is reported PAUSED with the PI ruling, never as a pass. A failed gate STOPS the read: no verdict is written.

REUSE (imported, not rewritten): selc6_analyze.Boot / per_token_spearman / probs / cruise rule (tie-correct average ranks, sel_common.log_boot_idx draws); sel_common.pick_from_logits;
taniteval.four_families._seq_geometry (family components); command_adherence.plan_classes (command adherence); the p6_ceiling Shapley recipe.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = Path(__file__).resolve().parent
ART = "E:/Projects/TanitAD-artifacts"
P6READ = ART + "/refe_behaviour_20261107/p6_read"
EVALDIR = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval"
TANITEVAL = "E:/Projects/TanitAD/taniteval"
DEFAULT_P6_PROPOSALS = "F:/TanitAD_p6read_work/pass2/proposals_p6final.npz"
DEFAULT_P6_CSV = "E:/Projects/TanitAD/data/refe_navtest/score/refe_p6final/refe_p6final.csv"
DEFAULT_EXPORT = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
DEFAULT_C3 = ART + "/refe_ego_probe_20261106/tokens_all.json"
PREREG = ART + "/scorer_gym_20261107/p6base/PREREG_SCORER1P6.md"

if P6READ not in sys.path:
    sys.path.insert(0, P6READ)
import selc6_analyze as SA  # noqa: E402  (REUSED: Boot, per_token_spearman, probs, ci; imports sel_common as sc)

sc = SA.sc

# ------------------------------------------------------------------------------------------------ frozen constants (card sections 2 to 5)
B_DRAWS, SEED = SA.B_DRAWS, SA.SEED                    # 10,000 and 20260927
GOAL = 94.55
G1_BAR = -0.5                                          # points x100, each of TTC, NC, DAC, against P6
G2_BAR = 0.5                                           # CI-hi of delta regret (points) must be < this
G3_BAR = 0.10                                          # CI-hi of the overspeed-share difference must be < this
OVERSPEED, SLOW = 1.15, 0.85
M1_FRAC = 0.5
M1_HEADS = {"TTC": 3, "NC": 0, "DAC": 1}               # column order NC DAC EP TTC C DDC
ALL_HEADS = {"NC": 0, "DAC": 1, "EP": 2, "TTC": 3}
SUBN = ("NC", "DAC", "EP", "TTC", "C", "DDC")
CSV_SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
EPS = 1e-9                                             # float slack on exact-boundary comparisons (card bars are decimal; documented in HANDOFF)
M = 64
DT = 0.5
ARM_ORDER = ("C0", "C0r", "S1E", "S1")
TOPK = (1, 2, 4, 8, 16, 32, 64)

ESTIMATOR = ("paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, over the evaluation set's own logs (sel_common.log_boot_idx draws via selc6_analyze.Boot). "
             "It answers 'another draw of logs' only. Training variance is read from the replicate floor F = |C0r - C0|; inference is deterministic (PREREG_P6 bit-identity check).")
TIER = {"E3": "T1-class open loop: NAVSIM v1.1 PDMS of the arm's own pick on the full navtest (self-action open loop, not closed loop)",
        "E2": "selection tier: PDM truth of P6's own 64 proposals on E2, the arm judged by which proposal it picks",
        "E1": "descriptive, C3 stratum-weighted, never decisive"}
EVIDENCE = "MEASURED (ours; every input artifact and its sha256 is listed under 'inputs')"
STRATEGIC = ("PAUSED -- PI ruling Sayed 2026-09-19, PI_DECISION_QUEUE ITEM 25: \"the strategic layer will be only switched off temporarily until we proved that both tactical and operative layers are "
             "driving with high quality. So the 4 layer rule is just temporarily paused and concerns only the strategic part.\" Not a pass; not omitted; lifting it is the PI's call.")

AMBIGUITIES = [
    "A1 plan speed v4: the arms' outputs carry the executed plan as 8 NAVSIM poses at 0.5 s, not the 20-step 0.2 s trajectory the C3 read used. v4 is therefore the chord of the LAST step (|p[7]-p[6]|/0.5) "
    "for plans and for the human future alike (same estimator on both sides). Overspeed (v4/v0 > 1.15), slow share (< 0.85) and the Spearman ratio v4/v0 use it. Not tuned: it is the only plan representation in the outputs.",
    "A2 CRUISE: selc6_analyze.cruise_formula applied to the export's own fields on the 0.5 s grid: human future finite, v0 = |ego_velocity| > 5, |ego_acceleration_x| < 0.3, |v4_gt - v0| < 1.0 (v4_gt the human last-step chord). "
    "E2/E3 tokens have no 20-step ego-probe record, so tokens_cruise.json cannot be used there.",
    "A3 M1 token set per head: a token enters head h only if its TRUE labels vary AND every arm's predicted head is defined on it (a common set, so every arm difference is paired); n per head is reported. Mean over TTC, NC, DAC of the per-head means.",
    "A4 M1 bar read as written: S1E - C0 >= +0.5 * |dRho_C0| (dRho_C0 = C0's own mean-over-heads gap), CI-lo of S1E - C0 > 0, and |S1E - C0| > 2 * |dRho_C0r - dRho_C0|. The M1 floor is the same C0r-C0 pair as P1's.",
    "A5 F is |C0r - C0| on the point estimates (PDMS x100 on E3 for P1). If C0r is absent F is undefined and no bar that needs F is cleared (conservative).",
    "A6 'clears P1 against P6' for S1E means its own P1 (vs C0 with the 2F rule AND vs P6 CI-lo > 0). For O4 'no arm clears P1 against P6' is read as: neither C0 nor S1E has CI-lo > 0 and |diff| > 2F against P6.",
    "A7 Outcome overlap: O1..O6 are not mutually exclusive as written. Every match is listed (matches); the primary label follows the fixed order O1, O2, O3, O5, O6, O4, Unnamed (most specific first; O5 NOT PROVEN before O4/O6). O3 (S1E harm) is evaluated whether or not C0 is adopted.",
    "A8 Family clause covers LONGITUDINAL (EP, TTC, target-speed accuracy, speed MAE, along-track MAE and final error, accel MAE, overspeed share) and LATERAL (DAC, heading, curvature, yaw-rate, cross-track MAE and final error) components, "
    "adverse iff the paired CI of (arm - P6) lies entirely on the worse side. No multiplicity correction is applied (card silent). NC, C, DDC are reported with CIs but sit in G1/PDMS, not in the clause. TACTICAL components are reported with paired CIs and do not enter the clause (the clause text names longitudinal and lateral).",
    "A9 Per-token family components are masked means over the horizon steps (the four_families masks); a token enters a component's bootstrap only where the component is defined for both arm and P6. Distance-keeping is UNAVAILABLE at this tier (card).",
    "A10 E2 and E3 use the arm's logits with the planner's navsim_v1 rule WITHOUT the lane factor (6 outputs): pick = argmax of p_NC*p_DAC*(5 p_EP + 5 p_TTC + 2 p_C)/12 in torch float32, first index on ties. All 64 proposals are treated as valid (E2 table validity is gated).",
    "A11 Unchanged tokens carry P6's banked csv value; a changed token's value is looked up as the score of the SAME proposal index, from P6's csv, the E2 table, or any arm's harness csv (proposals are bit-identical across arms by G0-E3, so a (token, k) score is arm-independent; conflicts stop the read).",
    "A12 Boundary comparisons use a 1e-9 slack in the arm's disfavour only where the card bar is a decimal (G1 -0.5 points, G2/G3 strict '<', goal 94.55): see 'bar' helpers; CI-lo > 0 and |diff| > 2F are strict with no slack.",
]
NOT_IMPLEMENTED = [
    "E1 (C3, 1,058 tokens) is reported as a stratum-weighted point only when --c3-tokens is given; no interval (descriptive).",
    "Closed-loop read (card section 6) and the S1 / GPU arms are outside this tool.",
    "ENS3 has no replicate floor; stated with its result.",
]


# ------------------------------------------------------------------------------------------------ small helpers
def sha256_file(p):
    return SA.sha256_file(str(p))


def jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def fin(x):
    return None if x is None or not np.isfinite(x) else float(x)


def ge(x, bar):
    """x >= bar with EPS slack in the arm's disfavour only (bar helpers, A12)"""
    return bool(x >= bar - EPS)


def lt(x, bar):
    return bool(x < bar - EPS)


def pick_logits(L):
    """navsim_v1 rule on 6 outputs, torch float32, first index on ties (sel_common.pick_from_logits, lane=False)"""
    return sc.pick_from_logits(np.asarray(L, np.float32), "torch", lane=False).astype(np.int64)


def agg_scores(L):
    return sc.agg_torch(np.asarray(L, np.float32), lane=False)


def plan_v4(poses8):
    p = np.asarray(poses8, np.float64)[..., :2]
    return np.linalg.norm(p[..., 7, :] - p[..., 6, :], axis=-1) / DT


def read_harness_csv(p):
    rows = {}
    with open(p, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("token") in (None, "average", ""):
                continue
            rows[r["token"]] = (r["valid"] == "True", float(r["score"]), np.array([float(r[c]) for c in CSV_SUB]))
    return rows


# ------------------------------------------------------------------------------------------------ the bootstrap set
class BootSet:
    """One shared [B, n_logs] resample of log ids for every statistic on a token set (selc6_analyze.Boot)."""

    def __init__(self, log):
        self.log = np.asarray(log)
        self.B = SA.Boot(self.log)
        self.n_logs = self.B.nl

    def stat(self, vals, mask=None):
        v = np.asarray(vals, np.float64)
        m = np.isfinite(v) if mask is None else (np.asarray(mask, bool) & np.isfinite(v))
        s, c = self.B.parts(v, m)
        S, C = s[self.B.draws].sum(1), c[self.B.draws].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = S / C
        pt = float(s.sum() / c.sum()) if c.sum() > 0 else float("nan")
        return pt, r, int(m.sum())

    def diff(self, a, b, mask=None):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return self.stat(a - b, mask)


def rec(pt, r, n=None):
    r = np.asarray(r)
    r = r[np.isfinite(r)]
    out = {"point": fin(pt), "lo": float(np.percentile(r, 2.5)) if r.size else None, "hi": float(np.percentile(r, 97.5)) if r.size else None}
    if n is not None:
        out["n"] = int(n)
    return out


# ------------------------------------------------------------------------------------------------ world / inputs
class World:
    """Per-token facts of the evaluation set: token, log, human future (8 poses, 0.5 s), v0, a0, command (or None)."""

    def __init__(self, token, log, human8, v0, a0, cmd=None, fingerprint=None):
        self.token = np.array([str(t) for t in token])
        self.log = np.array([str(x) for x in log])
        self.human8 = np.asarray(human8, np.float64)
        self.v0 = np.asarray(v0, np.float64)
        self.a0 = np.asarray(a0, np.float64)
        self.cmd = None if cmd is None else np.asarray(cmd, np.int64)
        self.fingerprint = None if fingerprint is None else np.array([str(x) for x in fingerprint])
        self.index = {t: i for i, t in enumerate(self.token)}

    def cruise(self):
        """selc6_analyze.cruise_formula on the 0.5 s grid (A2)"""
        v4_gt = plan_v4(self.human8)
        ok = np.isfinite(self.human8).all(axis=(1, 2))
        return ok & (self.v0 > 5.0) & (np.abs(self.a0) < 0.3) & (np.abs(v4_gt - self.v0) < 1.0)


def world_from_export(path, tokens=None):
    exp = json.load(gzip.open(path, "rt", encoding="utf-8"))["tokens"]
    toks = sorted(exp) if tokens is None else list(tokens)
    log, hum, v0, a0, cmd, fp = [], [], [], [], [], []
    for t in toks:
        r = exp[t]
        es = r["ego_statuses"][-1]
        log.append(r["log_name"])
        hum.append(r["human_future_poses"])
        v0.append(math.hypot(*es["ego_velocity"]))
        a0.append(es["ego_acceleration"][0])
        cmd.append(int(np.argmax(es["driving_command"])))
        fp.append(r.get("fingerprint", ""))
    return World(toks, log, hum, v0, a0, cmd, fp)


def world_from_dump(dump_npz):
    """C3 stand-in world from F:/.../dump_p6.npz: human8, ego v0 = ego[:,6], ax0 = ego[:,2], log. No command (adherence UNAVAILABLE)."""
    Z = np.load(dump_npz, allow_pickle=False)
    return World(Z["token"], Z["log"], Z["human8"], Z["ego"][:, 6], Z["ego"][:, 2], None, Z["fingerprint"])


class Inputs:
    """Everything the read needs, in memory (the file loaders build it; tests build it directly)."""

    def __init__(self, world, p6_logits, p6_proposals, p6_rows, e2_tokens, e2_table, arms, p6_pick_banked=None, p6_fingerprint=None,
                 c3_strata=None, tactical=True, provenance=None, p6_token=None):
        self.world, self.p6_logits, self.p6_proposals = world, p6_logits, p6_proposals
        self.p6_rows = p6_rows                    # token -> (valid, score, subs6)
        self.e2_tokens, self.e2_table, self.arms = list(e2_tokens), e2_table, arms
        self.p6_pick_banked, self.p6_fingerprint = p6_pick_banked, p6_fingerprint
        self.c3_strata, self.tactical = c3_strata, tactical
        self.provenance = provenance or {}
        self.p6_token = p6_token


class Arm:
    def __init__(self, name, e3_token, e3_logits, e3_proposals, e2_token=None, e2_logits=None, rows=None):
        self.name, self.e3_token, self.e3_logits, self.e3_proposals = name, e3_token, e3_logits, e3_proposals
        self.e2_token, self.e2_logits = e2_token, e2_logits
        self.rows = rows or {}                    # token -> (valid, score, subs6): the arm's harness csv of tokens whose pick differs from P6's


def _align(tokens, world_tokens, what):
    idx = {str(t): i for i, t in enumerate(tokens)}
    miss = [t for t in world_tokens if t not in idx]
    extra = len(idx) - (len(world_tokens) - len(miss))
    if miss or extra:
        return None, f"{what}: {len(miss)} evaluation tokens absent, {extra} tokens not in the evaluation set"
    return np.array([idx[t] for t in world_tokens]), None


def inputs_from_files(a):
    P6 = np.load(a.p6_proposals, allow_pickle=False)
    world = world_from_export(a.export, [str(t) for t in P6["token"]]) if not a.world_dump else world_from_dump(a.world_dump)
    props_key = "proposals" if "proposals" in P6.files else "props8"
    ix, err = _align(P6["token"], list(world.token), "p6 proposals")
    if err:
        raise SystemExit("REFUSING: " + err)
    p6 = dict(logits=P6["logits"][ix], props=P6[props_key][ix], pick=(P6["pick"][ix].astype(np.int64) if "pick" in P6.files else None),
              fp=(P6["fingerprint"][ix] if "fingerprint" in P6.files else None))
    p6_rows = read_harness_csv(a.p6_csv)
    e2 = json.load(open(a.e2_tokens, encoding="utf-8"))
    T = np.load(a.e2_table, allow_pickle=False)
    table = dict(token=np.array([str(t) for t in T["token"]]), pdms=T["pdms"].astype(np.float64), sub=T["sub"].astype(np.float64), valid=T["valid"].astype(bool),
                 red=(T["red_pick_pdms"] if "red_pick_pdms" in T.files else None))
    arms = {}
    for spec in a.arm:
        name, d = spec.split("=", 1)
        if name == "ENS3":
            raise SystemExit("ENS3 is built by the tool from C0, C0r and S1E; give its scored changed tokens as --ens3-csv")
        Z = np.load(os.path.join(d, "e3_logits.npz"), allow_pickle=False)
        pk = "proposals" if "proposals" in Z.files else "props8"
        e2p = os.path.join(d, "e2_logits.npz")
        Z2 = np.load(e2p, allow_pickle=False) if os.path.exists(e2p) else None
        cp = os.path.join(d, "e3_changed_scores.csv")
        arms[name] = Arm(name, Z["token"], Z["logits"], Z[pk], (Z2["token"] if Z2 is not None else None), (Z2["logits"] if Z2 is not None else None),
                         read_harness_csv(cp) if os.path.exists(cp) else {})
    if getattr(a, "ens3_csv", None):
        arms["_ENS3_rows"] = Arm("_ENS3_rows", np.array([]), np.zeros((0, M, 6)), np.zeros((0, M, 8, 3), np.float32), rows=read_harness_csv(a.ens3_csv))
    strata = None
    if a.c3_tokens:
        d = json.load(open(a.c3_tokens, encoding="utf-8"))
        strata = {t: d["stratum"][t] for t in d["tokens"]}
    prov = {"p6_proposals": (a.p6_proposals, sha256_file(a.p6_proposals)), "p6_csv": (a.p6_csv, sha256_file(a.p6_csv)),
            "e2_tokens": (a.e2_tokens, sha256_file(a.e2_tokens)), "e2_table": (a.e2_table, sha256_file(a.e2_table)),
            "export": (a.export, sha256_file(a.export)) if not a.world_dump else ("(world from dump)", a.world_dump),
            "arms": {n: [str(Path(x).resolve()) for x in (Path(s.split('=', 1)[1]) / "e3_logits.npz",)] + [sha256_file(Path(s.split('=', 1)[1]) / "e3_logits.npz")]
                     for s in a.arm for n in [s.split('=', 1)[0]]}}
    return Inputs(world, p6["logits"], p6["props"], p6_rows, e2["tokens"], table, arms, p6["pick"], p6["fp"], strata, not a.no_tactical, prov,
                  p6_token=np.array([str(t) for t in P6["token"]])[ix])


# ------------------------------------------------------------------------------------------------ gates
def bits_equal(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return a.tobytes() == b.tobytes()


def gate_e2(table, e2_tokens, world, p6_rows, p6_logits, p6_pick_e3, extra_red=True):
    """G0-E2: the E2 truth table reproduces P6's picks and per-token PDMS (and sub-scores) of refe_p6final.csv on 100 % of its tokens.
    RED control (when the table carries a shifted-pick column): the shifted column must differ from the csv on >= 1 token."""
    g = {"name": "G0-E2"}
    tpos = {t: i for i, t in enumerate(table["token"])}
    miss = [t for t in e2_tokens if t not in tpos]
    notw = [t for t in e2_tokens if t not in world.index]
    g["n_e2"], g["n_missing_in_table"], g["n_missing_in_world"] = len(e2_tokens), len(miss), len(notw)
    if miss or notw or len(set(e2_tokens)) != len(e2_tokens):
        g["PASS"], g["why"] = False, "token sets differ or duplicates"
        return g
    ti = np.array([tpos[t] for t in e2_tokens])
    wi = np.array([world.index[t] for t in e2_tokens])
    pk = p6_pick_e3[wi]
    pdms, sub, val = table["pdms"][ti], table["sub"][ti], table["valid"][ti]
    r = np.arange(len(ti))
    csvs = [p6_rows.get(t) for t in e2_tokens]
    if any(c is None for c in csvs):
        g["PASS"], g["why"] = False, "tokens absent from the P6 csv"
        return g
    cs = np.array([c[1] for c in csvs])
    csub = np.stack([c[2] for c in csvs])
    mine = pdms[r, pk]
    g["n_pdms_bit_equal"] = int((mine == cs).sum())
    g["n_sub_bit_equal"] = int((sub[r, pk] == csub).all(axis=1).sum())
    g["max_abs_pdms_diff"] = float(np.abs(mine - cs).max())
    f = sub[..., 0] * sub[..., 1] * (5 * sub[..., 2] + 5 * sub[..., 3] + 2 * sub[..., 4]) / 12
    g["formula_max_abs_diff"] = float(np.abs(f - pdms).max())
    g["all_finite_valid"] = bool(np.isfinite(pdms).all() and np.isfinite(sub).all() and val.all())
    g["pick_is_argmax_of_p6_logits"] = int((pick_logits(p6_logits[wi]) == pk).sum())
    ok = (g["n_pdms_bit_equal"] == len(ti) and g["n_sub_bit_equal"] == len(ti) and g["all_finite_valid"] and g["formula_max_abs_diff"] <= 1e-9
          and g["pick_is_argmax_of_p6_logits"] == len(ti))
    if table.get("red") is not None:
        red = np.asarray(table["red"])[ti]
        g["RED_shifted_pick_n_differ"] = int((red != cs).sum())
        ok = ok and g["RED_shifted_pick_n_differ"] >= 1
    else:
        g["RED_shifted_pick_n_differ"] = "NOT CARRIED by the table (control not evaluated)"
    g["PASS"] = bool(ok)
    return g


def gate_e3(arm, world, p6_props):
    """G0-E3: the arm's 64 proposals are bit-identical to P6's on every token."""
    g = {"name": f"G0-E3[{arm.name}]"}
    ix, err = _align(arm.e3_token, list(world.token), arm.name)
    if err:
        g["PASS"], g["why"] = False, err
        return g
    P = np.asarray(arm.e3_proposals)[ix]
    g["n"] = int(len(ix))
    if P.shape != p6_props.shape or P.dtype != p6_props.dtype:
        g["PASS"], g["why"] = False, f"shape/dtype {P.shape}/{P.dtype} != P6 {p6_props.shape}/{p6_props.dtype}"
        return g
    ne = (np.ascontiguousarray(P).view(np.uint8).reshape(P.shape[0], -1) != np.ascontiguousarray(p6_props).view(np.uint8).reshape(P.shape[0], -1)).any(axis=1)
    g["n_tokens_not_bit_identical"] = int(ne.sum())
    g["PASS"] = bool(ne.sum() == 0)
    g["_ix"] = ix
    return g


# ------------------------------------------------------------------------------------------------ the score pool
class Pool:
    """(token index, proposal k) -> harness score + sub-scores. Proposals are bit-identical across arms, so an entry is arm-independent; a conflict stops the read."""

    def __init__(self, n):
        self.known = np.zeros((n, M), bool)
        self.pdms = np.full((n, M), np.nan)
        self.sub = np.full((n, M, 6), np.nan)
        self.conflicts = []

    def put(self, i, k, score, sub, src):
        if self.known[i, k]:
            if not (self.pdms[i, k] == score and np.array_equal(self.sub[i, k], sub)):
                self.conflicts.append((int(i), int(k), src))
            return
        self.known[i, k], self.pdms[i, k], self.sub[i, k] = True, score, sub

    def add_rows(self, rows, world, pick, src):
        n_bad = 0
        for t, (v, s, sb) in rows.items():
            i = world.index.get(t)
            if i is None:
                continue
            if not v:
                n_bad += 1
                continue
            self.put(i, int(pick[i]), s, sb, src)
        return n_bad

    def missing(self, pick):
        r = np.arange(len(pick))
        return np.where(~self.known[r, pick])[0]


# ------------------------------------------------------------------------------------------------ family components
_FF = None


def load_ff():
    global _FF
    if _FF is None:
        if TANITEVAL not in sys.path:
            sys.path.append(TANITEVAL)
        from taniteval import four_families as ff
        _FF = ff
    return _FF


def geometry_components(pred8, human8):
    """per-token family components (A9), pred vs human on the 0.5 s grid; -> {name: (values [N] float64 NaN where undefined, direction, family)}"""
    import torch
    ff = load_ff()
    p = torch.as_tensor(np.ascontiguousarray(np.asarray(pred8)[..., :2])).float()
    g = torch.as_tensor(np.ascontiguousarray(np.asarray(human8)[..., :2])).float()
    P, G = ff._seq_geometry(p, DT), ff._seq_geometry(g, DT)

    def mm(x, m):
        m = m.float()
        c = m.sum(1)
        return ((x * m).sum(1) / c.clamp(min=1)).masked_fill(c == 0, float("nan")).double().numpy()

    ones = torch.ones_like(P["speed"], dtype=torch.bool)
    sp = (P["speed"] - G["speed"]).abs()
    al = P["along"] - G["along"]
    ct = P["cross"] - G["cross"]
    both, bp = P["valid"] & G["valid"], P["pair_valid"] & G["pair_valid"]
    dh = P["heading"] - G["heading"]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi
    L, T = "longitudinal", "lateral"
    out = {"speed_mae_mps": (mm(sp, ones), "lower", L)}
    for b in ff.TARGET_SPEED_BANDS_MPS:
        out[f"target_speed_within_{b}mps"] = (mm((sp <= b).float(), ones), "higher", L)
    out["along_mae_m"] = (mm(al.abs(), ones), "lower", L)
    out["along_final_abs_m"] = (al[:, -1].abs().double().numpy(), "lower", L)
    out["accel_mae_mps2"] = (mm((P["accel"] - G["accel"]).abs(), torch.ones_like(P["accel"], dtype=torch.bool)), "lower", L)
    out["heading_mae_deg"] = (mm(dh.abs() * 180.0 / math.pi, both), "lower", T)
    out["yaw_rate_mae_degps"] = (mm((P["yaw_rate"] - G["yaw_rate"]).abs() * 180.0 / math.pi, bp), "lower", T)
    out["curvature_mae_1pm"] = (mm((P["curvature"] - G["curvature"]).abs(), bp), "lower", T)
    out["cross_mae_m"] = (mm(ct.abs(), ones), "lower", T)
    out["cross_final_abs_m"] = (ct[:, -1].abs().double().numpy(), "lower", T)
    return out


def tactical_components(pred8, human8, cmd):
    """decision accuracy (lateral / longitudinal, plan class == human class) and command adherence (all / turn-active); canonical labeller via command_adherence"""
    try:
        if EVALDIR not in sys.path:
            sys.path.insert(0, EVALDIR)
        import command_adherence as CA
        import torch
        RT, FF = CA.labeller()

        def classes(poses):
            w = torch.as_tensor(np.ascontiguousarray(np.asarray(poses)[..., :2])).float()
            dyaw, dv, v0, v1, _ = FF.maneuver_kinematics(w, DT)
            lat, lon = RT.factor_from_kinematics(dyaw, dv, v0, v1)
            return lat.numpy(), lon.numpy()

        lp, np_ = classes(pred8)
        lg, ng = classes(human8)
        out = {"lateral_decision_acc": ((lp == lg).astype(float), "higher", "tactical"), "longitudinal_decision_acc": ((np_ == ng).astype(float), "higher", "tactical")}
        if cmd is not None:
            pc, _ = CA.plan_classes(np.asarray(pred8))
            ok = (pc == cmd).astype(float)
            out["command_adherence_all"] = (np.where(cmd < 3, ok, np.nan), "higher", "tactical")
            out["command_adherence_turn_active"] = (np.where((cmd == 0) | (cmd == 2), ok, np.nan), "higher", "tactical")
        else:
            out["_command_adherence"] = "UNAVAILABLE: no driving_command in this world"
        return out
    except BaseException as e:      # noqa: BLE001 -- UNAVAILABLE with its reason, never a pass
        return {"_tactical": f"UNAVAILABLE: {type(e).__name__}: {e}"}


# ------------------------------------------------------------------------------------------------ pure decision functions (tested at exact boundaries)
def p1_clears(lo, diff, floor):
    """CI-lo > 0 and |diff| > 2F (strict). floor None (no replicate) -> not cleared (A5)."""
    if floor is None or lo is None or diff is None:
        return False
    return bool(lo > 0 and abs(diff) > 2.0 * floor)


def m1_clears(diff, lo, dro_c0, floor_m1):
    """S1E - C0 >= +0.5 * |dRho_C0| and CI-lo > 0 and |S1E - C0| > 2 * floor"""
    if floor_m1 is None or lo is None or diff is None or dro_c0 is None:
        return False
    return bool(ge(diff, M1_FRAC * abs(dro_c0)) and lo > 0 and abs(diff) > 2.0 * floor_m1)


def g1_ok(delta_points):
    return bool(ge(delta_points, G1_BAR))


def g2_ok(hi):
    return lt(hi, G2_BAR)


def g3_ok(hi):
    return lt(hi, G3_BAR)


def goal_status(point, hi):
    if point is None:
        return "UNDEFINED"
    if ge(point, GOAL):
        return "GOAL MET"
    if hi is not None and ge(hi, GOAL):
        return "GOAL NOT EXCLUDED"
    return "BELOW GOAL"


def adverse(direction, lo, hi):
    """paired CI of (arm - P6) entirely on the worse side"""
    if lo is None or hi is None:
        return False
    return bool(lo > 0) if direction == "lower" else bool(hi < 0)


def decide_outcome(st):
    """st: adopted ('C0'|'S1E'|None), adopted_point, s1e_p1 (bool), s1e_m1 (bool), s1e_guards_ok (bool|None if S1E absent), c0_vs_p6 (bool), s1e_vs_p6 (bool),
    swamped (bool), m1_holds (bool), guard_caused (bool). -> dict(matches, primary, next)"""
    nxt = {"O1": "confirm on a replicate (C0r exists for C0; train S1E seed 1 for S1E), then the closed-loop read of card section 6",
           "O2": "recompute the Shapley split on the adopted arm on E2 (zero GPU), then run the lever for its largest head; ENS3 is read first (no training)",
           "O3": "run S1 (no EP) to isolate the EP term; if G3 is the failed guard and S1 clears G3, the next lever is the EP term with a pre-registered overspeed penalty",
           "O4": "this recipe does not move the selection: a four-fold larger dose on C0 (1,600 updates), then privileged-teacher distillation (SCORER_GYM_DESIGN.md section 2.7)",
           "O5": "NOT PROVEN: S1E seed 1 and a second C0 replicate",
           "O6": "the safety heads moved and the picks did not gain: the Shapley split on S1E and C0 on E2, then the lever for the largest head",
           "Unnamed": "matches none of O1 to O6: labelled as such; the Shapley split on E2 names the next lever"}
    names = {"O1": "GOAL MET", "O2": "GAIN, BELOW GOAL", "O3": "HARM", "O4": "NULL", "O5": "SWAMPED", "O6": "MECHANISM ONLY", "Unnamed": "UNNAMED"}
    adopted = st.get("adopted")
    m = []
    if adopted and st["adopted_point"] is not None and ge(st["adopted_point"], GOAL):
        m.append("O1")
    if adopted and not ge(st["adopted_point"], GOAL):
        m.append("O2")
    if st.get("s1e_guards_ok") is False and (st.get("s1e_p1") or st.get("s1e_m1")):
        m.append("O3")
    if st.get("swamped"):
        m.append("O5")
    if (not adopted) and st.get("m1_holds") and not st.get("guard_caused"):
        m.append("O6")
    if (not adopted) and not st.get("c0_vs_p6") and not st.get("s1e_vs_p6") and not st.get("m1_holds"):
        m.append("O4")
    order = ["O1", "O2", "O3", "O5", "O6", "O4"]
    prim = next((o for o in order if o in m), "Unnamed")
    return {"matches": m or ["Unnamed"], "primary": prim, "name": names[prim], "next_levers": {o: nxt[o] for o in (m or ["Unnamed"])}}


# ------------------------------------------------------------------------------------------------ the read
def head_rhos(P, sub, ratio, cm):
    """within-token Spearman of each head against v4/v0, average ranks (selc6_analyze.per_token_spearman, method 'tie')"""
    return {h: SA.per_token_spearman(P[..., j], ratio, cm, "tie") for h, j in ALL_HEADS.items()}, \
           {h: SA.per_token_spearman(sub[..., j], ratio, cm, "tie") for h, j in ALL_HEADS.items()}


def shapley(logits, sub, pdms, valid, heads=(0, 1, 2, 3)):
    """exact Shapley over {NC, DAC, EP, TTC} of the E2 PDMS x100 of the pick when the coalition's heads are replaced by their truth (the p6_ceiling recipe)"""
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64)))
    r = np.arange(len(p))

    def val(c):
        x = p.copy()
        for j in c:
            x[..., j] = sub[..., j]
        a = x[..., 0] * x[..., 1] * (5 * x[..., 2] + 5 * x[..., 3] + 2 * x[..., 4]) / 12
        a = np.where(valid, a, -1e9)
        return float(100 * pdms[r, a.argmax(1)].mean())

    vals = {frozenset(c): val(c) for k in range(len(heads) + 1) for c in itertools.combinations(heads, k)}
    n = len(heads)
    sh = {}
    for i in heads:
        tot = 0.0
        oth = [j for j in heads if j != i]
        for k in range(n):
            for c in itertools.combinations(oth, k):
                tot += math.factorial(k) * math.factorial(n - 1 - k) / math.factorial(n) * (vals[frozenset(c) | {i}] - vals[frozenset(c)])
        sh[SUBN[i]] = tot
    return {"v_empty": vals[frozenset()], "v_all": vals[frozenset(heads)], "shapley_points": sh, "sum": float(sum(sh.values()))}


def run_read(inp, swap_labels=False, need_ens3=True):
    """-> result dict. status in {READ STOPPED (a gate failed), INCOMPLETE (scores missing), READ}. swap_labels = the deliberate-regression arm (pred/true swapped in M1)."""
    t0 = time.time()
    W, N = inp.world, len(inp.world.token)
    res = {"prereg": {"path": PREREG, "sha256": sha256_file(PREREG) if os.path.exists(PREREG) else None}, "tier": TIER, "evidence_class": EVIDENCE, "estimator": ESTIMATOR,
           "strategic": STRATEGIC, "ambiguities_applied": AMBIGUITIES, "not_implemented": NOT_IMPLEMENTED, "inputs": inp.provenance, "gates": {}, "n_e3": N}
    code = {}
    for nm, p in (("selc6_analyze.py", P6READ + "/selc6_analyze.py"), ("selc6_score.py", P6READ + "/selc6_score.py"), ("sel_common.py", ART + "/refe_selection_probe_20261107/sel_common.py"),
                  ("read_scorer1p6.py", str(HERE / "read_scorer1p6.py"))):
        code[nm] = sha256_file(p) if os.path.exists(p) else None
    res["code_sha256"] = code
    arms = {k: v for k, v in inp.arms.items() if not k.startswith("_")}
    if "C0" not in arms:
        res.update(status="READ STOPPED", why="C0 is required (the zero point of P1 and M1)")
        return res

    # ---- P6 reference on E3
    p6_pick = pick_logits(inp.p6_logits)
    gp = {"name": "G0-PICK"}
    if inp.p6_pick_banked is not None:
        gp["n_equal_to_banked_pick"], gp["n"] = int((p6_pick == inp.p6_pick_banked).sum()), N
        gp["PASS"] = bool(gp["n_equal_to_banked_pick"] == N)
    else:
        gp["PASS"], gp["note"] = True, "no banked pick in the P6 file; the rule is not cross-checked"
    res["gates"]["G0-PICK"] = gp
    miss_csv = [t for t in W.token if t not in inp.p6_rows]
    gc = {"name": "G0-P6CSV", "n_tokens_absent": len(miss_csv), "n_invalid_rows": int(sum(1 for t in W.token if t in inp.p6_rows and not inp.p6_rows[t][0]))}
    gc["PASS"] = bool(gc["n_tokens_absent"] == 0 and gc["n_invalid_rows"] == 0)
    res["gates"]["G0-P6CSV"] = gc
    ge2 = gate_e2(inp.e2_table, inp.e2_tokens, W, inp.p6_rows, inp.p6_logits, p6_pick) if gc["PASS"] else {"name": "G0-E2", "PASS": False, "why": "P6 csv gate failed"}
    res["gates"]["G0-E2"] = ge2
    for nm, a in arms.items():
        g = gate_e3(a, W, inp.p6_proposals)
        ix = g.pop("_ix", None)
        a._ix = ix
        res["gates"][g["name"]] = g
    if not all(g["PASS"] for g in res["gates"].values()):
        res.update(status="READ STOPPED", why="a gate failed: " + ", ".join(k for k, g in res["gates"].items() if not g["PASS"]) + "; no verdict is written")
        return res

    # ---- aligned logits / picks (E3, E2)
    wi2 = np.array([W.index[t] for t in inp.e2_tokens])
    n2 = len(wi2)
    L3, L2, pk3, pk2 = {"P6": inp.p6_logits}, {"P6": inp.p6_logits[wi2]}, {"P6": p6_pick}, {"P6": p6_pick[wi2]}
    for nm, a in arms.items():
        L3[nm] = np.asarray(a.e3_logits)[a._ix]
        if a.e2_logits is not None:
            j, err = _align(a.e2_token, list(inp.e2_tokens), nm + " e2 logits")
            if err:
                res.update(status="READ STOPPED", why=err)
                return res
            L2[nm] = np.asarray(a.e2_logits)[j]
        else:
            L2[nm] = L3[nm][wi2]
        pk3[nm], pk2[nm] = pick_logits(L3[nm]), pick_logits(L2[nm])
    if need_ens3 and all(k in arms for k in ("C0", "C0r", "S1E")):
        L3["ENS3"] = np.mean([L3[k].astype(np.float32) for k in ("C0", "C0r", "S1E")], axis=0, dtype=np.float32)
        L2["ENS3"] = np.mean([L2[k].astype(np.float32) for k in ("C0", "C0r", "S1E")], axis=0, dtype=np.float32)
        pk3["ENS3"], pk2["ENS3"] = pick_logits(L3["ENS3"]), pick_logits(L2["ENS3"])

    # ---- the score pool
    pool = Pool(N)
    pool.add_rows(inp.p6_rows, W, p6_pick, "p6 csv")
    tpos = {t: i for i, t in enumerate(inp.e2_table["token"])}
    for r, t in enumerate(inp.e2_tokens):
        i, ti = W.index[t], tpos[t]
        for k in range(M):
            pool.put(i, k, inp.e2_table["pdms"][ti, k], inp.e2_table["sub"][ti, k], "e2 table")
    nbad = 0
    for nm, a in inp.arms.items():
        base = nm.lstrip("_")
        pk = pk3.get("ENS3") if nm == "_ENS3_rows" else pk3.get(nm)
        if pk is not None:
            nbad += pool.add_rows(a.rows, W, pk, "csv " + nm)
    if pool.conflicts or nbad:
        res.update(status="READ STOPPED", why=f"score pool: {len(pool.conflicts)} conflicting scores for the same (token, proposal), {nbad} invalid harness rows; examples {pool.conflicts[:3]}")
        return res
    need = {nm: pool.missing(pk3[nm]) for nm in pk3 if nm != "ENS3" or need_ens3}
    ens_need = need.pop("ENS3", np.array([], int)) if "ENS3" in need else np.array([], int)
    short = {nm: int(len(v)) for nm, v in need.items() if len(v)}
    if short:
        res.update(status="INCOMPLETE", why="harness scores missing for changed picks: " + json.dumps(short) + "; run emit-changed, score with the harness, add e3_changed_scores.csv",
                   missing_tokens={nm: [str(W.token[i]) for i in v[:5]] for nm, v in need.items() if len(v)})
        return res

    def e3_arrays(nm):
        r = np.arange(N)
        return pool.pdms[r, pk3[nm]] * 100.0, pool.sub[r, pk3[nm]] * 100.0

    # ---- bootstrap sets
    B3 = BootSet(W.log)
    B2 = BootSet(W.log[wi2])
    pd3, sb3 = {}, {}
    for nm in ["P6"] + [k for k in pk3 if k != "ENS3"]:
        pd3[nm], sb3[nm] = e3_arrays(nm)
    # E2 arrays
    tab = inp.e2_table
    ti = np.array([tpos[t] for t in inp.e2_tokens])
    T_pd, T_sub, T_val = tab["pdms"][ti], tab["sub"][ti], tab["valid"][ti]
    r2 = np.arange(n2)
    ratio2 = plan_v4(inp.p6_proposals[wi2]) / np.maximum(W.v0[wi2], 1e-6)[:, None]
    cru2 = W.cruise()[wi2]
    Pm = {nm: SA.probs(L2[nm]) for nm in L2}
    cm = cru2[:, None] & T_val
    rho_true = {h: SA.per_token_spearman(T_sub[..., j], ratio2, cm, "tie") for h, j in ALL_HEADS.items()}
    rho_pred = {nm: {h: SA.per_token_spearman(Pm[nm][..., j], ratio2, cm, "tie") for h, j in ALL_HEADS.items()} for nm in L2}

    def gap(nm, h, swap=False):
        a, b = rho_pred[nm][h], rho_true[h]
        return (b - a) if swap else (a - b)

    def head_set(h, names):
        ok = np.isfinite(rho_true[h])
        for n_ in names:
            ok &= np.isfinite(rho_pred[n_][h])
        return ok

    core = [n_ for n_ in ["P6"] + list(arms)]
    sets = {h: head_set(h, core) for h in ALL_HEADS}
    res["m1_token_sets"] = {h: {"n_cruise_tokens": int(cru2.sum()), "n_head_entered": int((sets[h] & cru2).sum()),
                                "n_true_defined": int((np.isfinite(rho_true[h]) & cru2).sum())} for h in ALL_HEADS}

    def m1_level(nm):
        pts, rs, per = [], [], {}
        for h in M1_HEADS:
            v = gap(nm, h, swap_labels)
            pt, r, n = B2.stat(v, cru2 & sets[h])
            pts.append(pt)
            rs.append(r)
            per[h] = {**rec(pt, r, n), "pred_mean": fin(B2.stat(rho_pred[nm][h], cru2 & sets[h])[0]), "true_mean": fin(B2.stat(rho_true[h], cru2 & sets[h])[0])}
        r = np.mean(rs, axis=0)
        return float(np.mean(pts)), r, per

    def m1_diff(nm, base):
        pts, rs = [], []
        for h in M1_HEADS:
            v = gap(nm, h, swap_labels) - gap(base, h, swap_labels)
            pt, r, n = B2.stat(v, cru2 & sets[h])
            pts.append(pt)
            rs.append(r)
        return float(np.mean(pts)), np.mean(rs, axis=0)

    # ---- per-arm endpoints
    def e2_vec(nm):
        return T_pd[r2, pk2[nm]] * 100.0

    best2 = np.where(T_val, T_pd, -1.0).max(1) * 100.0
    regret = {nm: best2 - e2_vec(nm) for nm in pk2}
    over = {nm: (ratio2[r2, pk2[nm]] > OVERSPEED).astype(float) for nm in pk2}
    slow = {nm: (ratio2[r2, pk2[nm]] < SLOW).astype(float) for nm in pk2}
    random2 = np.where(T_val, T_pd, np.nan)
    random2 = np.nanmean(random2, axis=1) * 100.0

    def level3(nm):
        pt, r, n = B3.stat(pd3[nm])
        return rec(pt, r, n)

    def vs3(nm, base, vec=None):
        v = pd3 if vec is None else vec
        pt, r, n = B3.diff(v[nm], v[base])
        return rec(pt, r, n)

    F_pdms = abs(float(pd3["C0r"].mean() - pd3["C0"].mean())) if "C0r" in arms else None
    m1_c0_pt, m1_c0_r, m1_c0_per = m1_level("C0")
    F_m1 = None
    if "C0r" in arms:
        pt, r = m1_diff("C0r", "C0")
        F_m1 = abs(pt)
    endpoints = {"P6": {"E3_pdms": level3("P6"), "E3_pick_changed_vs_P6": 0}}
    arm_names = [k for k in ARM_ORDER if k in arms]
    for nm in arm_names:
        e = {"E3_pdms": level3(nm), "E3_pdms_vs_P6": vs3(nm, "P6"), "E3_pick_changed_vs_P6": int((pk3[nm] != p6_pick).sum())}
        e["goal_status"] = goal_status(e["E3_pdms"]["point"], e["E3_pdms"]["hi"])
        e["E3_subscores_vs_P6"] = {}
        for j, s in enumerate(SUBN):
            pt, r, n = B3.diff(sb3[nm][:, j], sb3["P6"][:, j])
            e["E3_subscores_vs_P6"][s] = {**rec(pt, r, n), "arm_mean_x100": float(sb3[nm][:, j].mean()), "P6_mean_x100": float(sb3["P6"][:, j].mean())}
        pt, r, n = B2.diff(regret[nm], regret["P6"])
        e["G2_delta_regret_vs_P6"] = {**rec(pt, r, n), "arm_regret": float(regret[nm].mean()), "P6_regret": float(regret["P6"].mean())}
        pt, r, n = B2.diff(over[nm], over["P6"], cru2)
        e["G3_overspeed_diff_vs_P6"] = {**rec(pt, r, n), "arm_share": fin(B2.stat(over[nm], cru2)[0]), "P6_share": fin(B2.stat(over["P6"], cru2)[0])}
        e["slow_pick_share_cruise"] = fin(B2.stat(slow[nm], cru2)[0])
        mp, mr, per = m1_level(nm)
        e["M1_dRho_level"] = {**rec(mp, mr), "per_head": per}
        e["E2_pdms"] = {**rec(*B2.stat(e2_vec(nm))[:2])}
        endpoints[nm] = e
    p6m = m1_level("P6")
    endpoints["P6"]["M1_dRho_level"] = {**rec(p6m[0], p6m[1]), "per_head": p6m[2]}
    endpoints["P6"]["E2_pdms"] = {**rec(*B2.stat(e2_vec("P6"))[:2])}
    endpoints["P6"]["slow_pick_share_cruise"] = fin(B2.stat(slow["P6"], cru2)[0])
    endpoints["P6"]["G3_overspeed_share"] = fin(B2.stat(over["P6"], cru2)[0])

    # ---- P1 / M1 / guards
    def guards(nm, also_vs=None):
        e = endpoints[nm]
        g1 = {s: g1_ok(e["E3_subscores_vs_P6"][s]["point"]) for s in ("TTC", "NC", "DAC")}
        g2 = g2_ok(e["G2_delta_regret_vs_P6"]["hi"])
        g3p6 = g3_ok(e["G3_overspeed_diff_vs_P6"]["hi"])
        out = {"G1_each_head_ge_-0.5": g1, "G2_ci_hi_lt_0.5": g2, "G3_vs_P6_ci_hi_lt_0.10": g3p6}
        ok = all(g1.values()) and g2 and g3p6
        if also_vs:
            pt, r, n = B2.diff(over[nm], over[also_vs], cru2)
            gd = {**rec(pt, r, n)}
            out[f"G3_vs_{also_vs}_ci_hi_lt_0.10"] = {"ci": gd, "ok": g3_ok(gd["hi"])}
            ok = ok and out[f"G3_vs_{also_vs}_ci_hi_lt_0.10"]["ok"]
        out["all_guards_ok"] = bool(ok)
        return out

    verdict = {"F_pdms_E3": fin(F_pdms), "F_M1": fin(F_m1), "dRho_C0": fin(m1_c0_pt)}
    c0 = endpoints["C0"]
    c0["P1_vs_P6"] = {**c0["E3_pdms_vs_P6"], "clears": p1_clears(c0["E3_pdms_vs_P6"]["lo"], c0["E3_pdms_vs_P6"]["point"], F_pdms)}
    c0["guards"] = guards("C0")
    c0["P1_clears_and_guards"] = bool(c0["P1_vs_P6"]["clears"] and c0["guards"]["all_guards_ok"])
    if "C0r" in endpoints:
        endpoints["C0r"]["P1_vs_P6"] = {**endpoints["C0r"]["E3_pdms_vs_P6"], "note": "replicate, reported"}
        endpoints["C0r"]["guards"] = guards("C0r")
    s1e_p1 = s1e_m1 = s1e_vs_p6 = s1e_guards = None
    if "S1E" in endpoints:
        e = endpoints["S1E"]
        pt, r, n = B3.diff(pd3["S1E"], pd3["C0"])
        e["P1_vs_C0"] = {**rec(pt, r, n), "clears": p1_clears(rec(pt, r, n)["lo"], pt, F_pdms)}
        e["P1_vs_P6"] = {**e["E3_pdms_vs_P6"], "clears_ci_lo_gt_0": bool(e["E3_pdms_vs_P6"]["lo"] > 0)}
        s1e_vs_p6 = bool(p1_clears(e["E3_pdms_vs_P6"]["lo"], e["E3_pdms_vs_P6"]["point"], F_pdms))
        e["P1_clears"] = bool(e["P1_vs_C0"]["clears"] and e["P1_vs_P6"]["clears_ci_lo_gt_0"])
        dp, dr = m1_diff("S1E", "C0")
        e["M1_vs_C0"] = {**rec(dp, dr), "bar_0.5_x_abs_dRho_C0": M1_FRAC * abs(m1_c0_pt), "floor_abs_C0r_minus_C0": fin(F_m1)}
        e["M1_holds"] = m1_clears(dp, e["M1_vs_C0"]["lo"], m1_c0_pt, F_m1)
        e["guards"] = guards("S1E", also_vs="C0")
        e["P1_clears_and_guards"] = bool(e["P1_clears"] and e["guards"]["all_guards_ok"])
        s1e_p1, s1e_m1, s1e_guards = e["P1_clears"], e["M1_holds"], e["guards"]["all_guards_ok"]
    if "S1" in endpoints:
        e = endpoints["S1"]
        pt, r, n = B3.diff(pd3["S1"], pd3["C0"])
        e["P1_vs_C0"] = {**rec(pt, r, n), "clears": p1_clears(rec(pt, r, n)["lo"], pt, F_pdms)}
        e["P1_vs_P6"] = e["E3_pdms_vs_P6"]
        dp, dr = m1_diff("S1", "C0")
        e["M1_vs_C0"] = {**rec(dp, dr)}
        e["M1_holds"] = m1_clears(dp, e["M1_vs_C0"]["lo"], m1_c0_pt, F_m1)
        e["guards"] = guards("S1", also_vs="C0")
        e["note"] = "S1 is the conditional follow-up arm: reported with the same endpoints, it takes no part in the Adopted-arm rule"

    # ---- adopted arm
    adopted = None
    if s1e_p1 and s1e_guards:
        adopted = "S1E"
    elif c0["P1_clears_and_guards"]:
        adopted = "C0"
    verdict["adopted_arm"] = adopted
    verdict["adopted_rule"] = "S1E if it clears P1 against C0 (and P6) and G1 to G3; otherwise C0 if it clears P1 against P6 and G1 to G3; otherwise none"
    verdict["adopted_pdms_e3"] = endpoints[adopted]["E3_pdms"] if adopted else None
    verdict["goal_status_adopted"] = endpoints[adopted]["goal_status"] if adopted else None
    verdict["goal_status_all_arms"] = {nm: endpoints[nm]["goal_status"] for nm in arm_names}

    # ---- family clause
    fam = {"STRATEGIC": STRATEGIC}
    comp_cache = {}

    def comps_of(nm):
        if nm in comp_cache:
            return comp_cache[nm]
        pose = inp.p6_proposals[np.arange(N), pk3[nm]]
        c = {}
        r = np.arange(N)
        for j, s in enumerate(SUBN):
            fam_name = {"EP": "longitudinal", "TTC": "longitudinal", "DAC": "lateral"}.get(s, "pdms-other")
            c[s] = (pool.sub[r, pk3[nm], j] * 100.0, "higher", fam_name)
        c.update(geometry_components(pose, W.human8))
        if inp.tactical:
            c.update(tactical_components(pose, W.human8, W.cmd))
        comp_cache[nm] = c
        return c

    def fam_table(nm):
        a, b = comps_of(nm), comps_of("P6")
        rows, adv = {}, []
        for k, v in a.items():
            if not isinstance(v, tuple):
                rows[k] = {"status": v if isinstance(v, str) else str(v)}
                continue
            vals, dr, fm = v
            pt, r, n = B3.diff(vals, b[k][0])
            rr = rec(pt, r, n)
            rr.update(direction=dr, family=fm, arm_mean=fin(np.nanmean(vals)), P6_mean=fin(np.nanmean(b[k][0])))
            rr["adverse"] = adverse(dr, rr["lo"], rr["hi"]) if fm in ("longitudinal", "lateral") else False
            rows[k] = rr
            if rr["adverse"]:
                adv.append(k)
        ov = endpoints[nm]["G3_overspeed_diff_vs_P6"]
        rows["overspeed_share_cruise_E2"] = {**{q: ov[q] for q in ("point", "lo", "hi", "n")}, "direction": "lower", "family": "longitudinal", "arm_mean": ov["arm_share"],
                                             "P6_mean": ov["P6_share"], "adverse": adverse("lower", ov["lo"], ov["hi"])}
        if rows["overspeed_share_cruise_E2"]["adverse"]:
            adv.append("overspeed_share_cruise_E2")
        sel = {"actual_pdms_E2": endpoints[nm]["E2_pdms"], "random_pdms_E2": fin(random2.mean()), "best_of_64_pdms_E2": fin(best2.mean())}
        rows["_selection_quality_E2"] = sel
        rows["_distance_keeping"] = "UNAVAILABLE at this tier (no lead-agent block in NAVSIM open loop); read in card section 6"
        return rows, adv

    fam["per_arm_vs_P6"] = {}
    for nm in arm_names:
        rows, adv = fam_table(nm)
        fam["per_arm_vs_P6"][nm] = {"components": rows, "adverse_components": adv}
    if adopted:
        adv = fam["per_arm_vs_P6"][adopted]["adverse_components"]
        verdict["verdict"] = "ADOPT" if not adv else "PDMS GAIN WITH COST"
        verdict["adverse_components"] = {k: {q: fam["per_arm_vs_P6"][adopted]["components"][k][q] for q in ("point", "lo", "hi", "direction", "n")} for k in adv}
        if adv:
            verdict["cost_note"] = ("Most of these components are distances from the human driver, which PDMS does not score. A PDMS GAIN WITH COST arm may be carried forward and used for the goal status, "
                                    "and is never described as the better driver.")
    else:
        verdict["verdict"] = "NO ADOPTION"
        verdict["adverse_components"] = {}
    res["family_clause"] = fam

    # ---- outcome
    swamped = False
    for e_, key in ((c0, "E3_pdms_vs_P6"),):
        if F_pdms is not None and e_[key]["lo"] is not None and e_[key]["lo"] > 0 and not abs(e_[key]["point"]) > 2 * F_pdms:
            swamped = True
    if "S1E" in endpoints and F_pdms is not None:
        d = endpoints["S1E"]["P1_vs_C0"]
        if d["lo"] is not None and d["lo"] > 0 and not abs(d["point"]) > 2 * F_pdms:
            swamped = True
        m = endpoints["S1E"]["M1_vs_C0"]
        if F_m1 is not None and m["lo"] is not None and m["lo"] > 0 and not abs(m["point"]) > 2 * F_m1:
            swamped = True
    guard_caused = bool((c0["P1_vs_P6"]["clears"] and not c0["guards"]["all_guards_ok"]) or (s1e_p1 and s1e_guards is False))
    st = dict(adopted=adopted, adopted_point=(endpoints[adopted]["E3_pdms"]["point"] if adopted else None), s1e_p1=bool(s1e_p1), s1e_m1=bool(s1e_m1),
              s1e_guards_ok=s1e_guards, c0_vs_p6=bool(c0["P1_vs_P6"]["clears"]), s1e_vs_p6=bool(s1e_vs_p6), swamped=swamped, m1_holds=bool(s1e_m1), guard_caused=guard_caused)
    verdict["outcome"] = decide_outcome(st)
    verdict["outcome_inputs"] = st
    verdict["verdict_written_before_ENS3"] = True
    res["endpoints"] = endpoints
    res["verdict"] = verdict
    res["m1_token_n_per_head"] = {h: int((sets[h] & cru2).sum()) for h in M1_HEADS}

    # ---- reported, not deciding: Shapley split and top-K oracle on E2
    rep = {}
    for nm in ([adopted] if adopted else []) + [k for k in arm_names if k != adopted]:
        sh = shapley(L2[nm], T_sub, T_pd, T_val)
        agg = agg_scores(L2[nm])
        order = np.argsort(-agg, axis=1, kind="stable")
        topk = {}
        for K in TOPK:
            sel_ = np.take_along_axis(np.where(T_val, T_pd, -1.0), order[:, :K], axis=1).max(1)
            topk[str(K)] = float(100 * sel_.mean())
        rep[nm] = {"shapley_E2": sh, "topK_oracle_E2_pdms_x100": topk, "EP_dRho": {"gap_level": rec(*B2.stat(gap(nm, "EP", swap_labels), cru2 & sets["EP"])[:2])},
                   "each_head_dRho": {h: rec(*B2.stat(gap(nm, h, swap_labels), cru2 & sets[h])[:2]) for h in ALL_HEADS}}
    res["reported_not_deciding"] = rep

    # ---- tactical (reported with paired CIs inside the family tables); E1 point
    if inp.c3_strata:
        zero = np.array([inp.c3_strata.get(t) == "zero" for t in W.token])
        c3m = np.array([t in inp.c3_strata for t in W.token])
        rnd = c3m & ~zero
        zr = c3m & zero
        e1 = {}
        for nm in ["P6"] + arm_names:
            e1[nm] = float(SA.W_ZERO * pd3[nm][zr].mean() + SA.W_RANDOM * pd3[nm][rnd].mean()) if zr.any() and rnd.any() else None
        res["E1_descriptive_weighted_pdms"] = {"values": e1, "n_c3_tokens_in_E3": int(c3m.sum()), "tier": TIER["E1"]}

    # ---- ENS3, after the verdict, cannot change it
    ens = {"read_after_verdict": True, "replicate_floor": "none: ENS3 has no replicate arm, so no 2F rule can be applied to it (stated per card section 5)"}
    if "ENS3" not in pk3:
        ens["status"] = "NOT READ: needs C0, C0r and S1E"
    elif len(ens_need):
        ens["status"] = f"NOT READ: harness scores missing for {len(ens_need)} ENS3 changed picks (run emit-changed --arm ENS3, score, pass --ens3-csv)"
    else:
        pd3["ENS3"], sb3["ENS3"] = e3_arrays("ENS3")
        base = adopted or "P6"
        pt, r, n = B3.diff(pd3["ENS3"], pd3[base])
        d = rec(pt, r, n)
        g1 = {}
        for s in ("TTC", "NC", "DAC"):
            p_, r_, n_ = B3.diff(sb3["ENS3"][:, SUBN.index(s)], sb3["P6"][:, SUBN.index(s)])
            g1[s] = {**rec(p_, r_, n_), "ok": g1_ok(p_)}
        p_, r_, n_ = B2.diff(regret["ENS3"], regret["P6"])
        g2 = {**rec(p_, r_, n_), "ok": g2_ok(rec(p_, r_, n_)["hi"])}
        p_, r_, n_ = B2.diff(over["ENS3"], over["P6"], cru2)
        g3 = {"vs_P6": {**rec(p_, r_, n_)}, "ok_vs_P6": g3_ok(rec(p_, r_, n_)["hi"])}
        if adopted:
            p_, r_, n_ = B2.diff(over["ENS3"], over[adopted], cru2)
            g3["vs_adopted"] = {**rec(p_, r_, n_)}
            g3["ok_vs_adopted"] = g3_ok(rec(p_, r_, n_)["hi"])
        guards_ok = all(v["ok"] for v in g1.values()) and g2["ok"] and g3["ok_vs_P6"] and g3.get("ok_vs_adopted", True)
        ens.update(status="READ", against=base, pdms_e3=level3_of(B3, pd3["ENS3"]), diff_vs_base=d, G1=g1, G2=g2, G3=g3,
                   pdms_e3_goal_status=goal_status(float(pd3["ENS3"].mean()), level3_of(B3, pd3["ENS3"])["hi"]),
                   bar_ci_lo_gt_0=bool(d["lo"] is not None and d["lo"] > 0), guards_ok=bool(guards_ok),
                   BAR_CLEARED=bool(d["lo"] is not None and d["lo"] > 0 and guards_ok), n_changed_picks_vs_P6=int((pk3["ENS3"] != p6_pick).sum()))
    res["ENS3"] = ens
    res["status"] = "READ"
    res["seconds"] = round(time.time() - t0, 1)
    return res


def level3_of(B, vec):
    pt, r, n = B.stat(vec)
    return rec(pt, r, n)


# ------------------------------------------------------------------------------------------------ emit-changed (seams for the harness)
def emit_changed(inp, names, out_dir, ens3=True):
    W, N = inp.world, len(inp.world.token)
    p6_pick = pick_logits(inp.p6_logits)
    arms = {k: v for k, v in inp.arms.items() if not k.startswith("_")}
    pool = Pool(N)
    pool.add_rows(inp.p6_rows, W, p6_pick, "p6")
    tpos = {t: i for i, t in enumerate(inp.e2_table["token"])}
    for t in inp.e2_tokens:
        for k in range(M):
            pool.put(W.index[t], k, inp.e2_table["pdms"][tpos[t], k], inp.e2_table["sub"][tpos[t], k], "e2")
    L, picks = {}, {}
    for nm, a in arms.items():
        ix, err = _align(a.e3_token, list(W.token), nm)
        if err:
            raise SystemExit(err)
        L[nm] = np.asarray(a.e3_logits)[ix]
        picks[nm] = pick_logits(L[nm])
        pool.add_rows(a.rows, W, picks[nm], nm)
    if "ENS3" in names and all(k in L for k in ("C0", "C0r", "S1E")):
        picks["ENS3"] = pick_logits(np.mean([L[k].astype(np.float32) for k in ("C0", "C0r", "S1E")], axis=0, dtype=np.float32))
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for nm in names:
        if nm not in picks:
            out[nm] = "no logits"
            continue
        idx = pool.missing(picks[nm])
        if not len(idx):
            out[nm] = {"n_needed": 0}
            continue
        poses = inp.p6_proposals[idx, picks[nm][idx]].astype(np.float32)
        fp = inp.p6_fingerprint[idx] if inp.p6_fingerprint is not None else W.fingerprint[idx]
        toks = [str(W.token[i]) for i in idx]
        sp = os.path.join(out_dir, f"seam_{nm}.npz")
        np.savez(sp, token=np.array(toks), fingerprint=np.array([str(x) for x in fp]), poses=poses, sampling=np.array([8, 0.5]), arm=np.array(f"scorer1p6_{nm}"))
        tj = os.path.join(out_dir, f"tokens_{nm}.json")
        json.dump({"tokens": sorted(toks), "token_log": {t: str(W.log[W.index[t]]) for t in toks}}, open(tj, "w", encoding="utf-8"))
        out[nm] = {"n_needed": int(len(idx)), "seam": sp, "tokens": tj,
                   "harness": f"cd <eval/harness>; python score_e.py --label refe_p6s1_{nm} --seam {sp} --tokens {tj} --out <score dir>  (then copy the csv to <arm dir>/e3_changed_scores.csv)"}
    json.dump(out, open(os.path.join(out_dir, "needs_scoring.json"), "w", encoding="utf-8"), indent=1)
    return out


# ------------------------------------------------------------------------------------------------ markdown report
def fmt(x, d=4):
    return "n/a" if x is None else (f"{x:+.{d}f}" if isinstance(x, float) else str(x))


def ci_s(r, d=4):
    if not r or r.get("point") is None:
        return "n/a"
    return f"{r['point']:+.{d}f} [{r['lo']:+.{d}f}, {r['hi']:+.{d}f}]" if r.get("lo") is not None else f"{r['point']:+.{d}f}"


def lvl_s(r, d=3):
    return "n/a" if not r or r.get("point") is None else f"{r['point']:.{d}f} [{r['lo']:.{d}f}, {r['hi']:.{d}f}]"


def to_markdown(res):
    L = ["# SCORER1P6 read", "", f"**Status: {res.get('status')}**", ""]
    L += [f"- Tier: E3 = {TIER['E3']}; E2 = {TIER['E2']}.", f"- Evidence class: {EVIDENCE}.", f"- Estimator: {ESTIMATOR}", f"- Card: `{res['prereg']['path']}` sha256 `{res['prereg']['sha256']}`", ""]
    L += ["## Gates", ""]
    for k, g in res["gates"].items():
        L.append(f"- {k}: **{'PASS' if g['PASS'] else 'FAIL'}** " + ", ".join(f"{a}={b}" for a, b in g.items() if a not in ("PASS", "name")))
    if res.get("status") != "READ":
        L += ["", f"**{res.get('status')}: {res.get('why')}**", "", "No verdict is written."]
        return "\n".join(L) + "\n"
    v = res["verdict"]
    o = v["outcome"]
    L += ["", "## Verdict (written before ENS3)", "", f"- Adopted arm: **{v['adopted_arm']}**; verdict: **{v['verdict']}**" + (f"; adverse: {', '.join(v['adverse_components'])}" if v["adverse_components"] else ""),
          f"- Goal status (adopted): {v['goal_status_adopted']}; per arm: {v['goal_status_all_arms']}", f"- Outcome: **{o['primary']} {o['name']}** (all matches: {', '.join(o['matches'])})",
          f"- Next lever: {o['next_levers'][o['primary']]}", f"- F (PDMS, E3) = {fmt(v['F_pdms_E3'])}; F (M1) = {fmt(v['F_M1'])}; dRho_C0 = {fmt(v['dRho_C0'])}", ""]
    L += ["## Endpoints", "", "| arm | PDMS E3 | vs P6 | changed picks | goal |", "|---|---|---|---|---|"]
    ep = res["endpoints"]
    L.append(f"| P6 | {lvl_s(ep['P6']['E3_pdms'])} | - | 0 | |")
    for nm, e in ep.items():
        if nm == "P6":
            continue
        L.append(f"| {nm} | {lvl_s(e['E3_pdms'])} | {ci_s(e['E3_pdms_vs_P6'])} | {e['E3_pick_changed_vs_P6']} | {e['goal_status']} |")
    L += ["", "| arm | M1 dRho level | M1 S1E-C0 style diff | G1 TTC/NC/DAC pts | G2 d-regret | G3 overspeed diff | guards |", "|---|---|---|---|---|---|---|"]
    for nm, e in ep.items():
        if nm == "P6":
            continue
        g1 = e["E3_subscores_vs_P6"]
        L.append(f"| {nm} | {ci_s(e['M1_dRho_level'])} | {ci_s(e.get('M1_vs_C0'))} | {g1['TTC']['point']:+.3f}/{g1['NC']['point']:+.3f}/{g1['DAC']['point']:+.3f} | {ci_s(e['G2_delta_regret_vs_P6'])} | "
                 f"{ci_s(e['G3_overspeed_diff_vs_P6'])} | {e.get('guards', {}).get('all_guards_ok')} |")
    L += ["", f"M1 n per head (CRUISE tokens entering): {res['m1_token_n_per_head']}", "", "## Family clause (arm - P6, paired CIs)", ""]
    for nm, f in res["family_clause"]["per_arm_vs_P6"].items():
        L.append(f"- {nm}: adverse = {f['adverse_components'] or 'none'}")
    L += ["", f"- STRATEGIC: {STRATEGIC}", ""]
    ens = res["ENS3"]
    L += ["## ENS3 (read after the verdict; cannot change it)", "", f"- {ens.get('status')}" + (f"; vs {ens['against']}: {ci_s(ens['diff_vs_base'])}; bar cleared: {ens['BAR_CLEARED']}" if ens.get("status") == "READ" else ""),
          f"- {ens['replicate_floor']}", "", "## Ambiguities applied", ""] + [f"- {a}" for a in res["ambiguities_applied"]] + ["", "## Not implemented / not verified", ""] + [f"- {a}" for a in res["not_implemented"]]
    return "\n".join(L) + "\n"


def write_outputs(res, out_dir, stem="scorer1p6_read"):
    os.makedirs(out_dir, exist_ok=True)
    jp, mp = os.path.join(out_dir, stem + ".json"), os.path.join(out_dir, stem + ".md")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=1, default=jdefault)
    open(mp, "w", encoding="utf-8").write(to_markdown(res))
    return jp, mp


# ------------------------------------------------------------------------------------------------ self-check on real data (C3 stand-in)
def inputs_selfcheck(csv_path=DEFAULT_P6_CSV):
    """P6 as the fake arm (C0 = C0r = S1E = P6's own logits) on real data. World/E2/E3 = the 1,058 C3 tokens of dump_p6 / table_p6 (STAND-IN: the real E2 table and P6's full-navtest
    proposals do not exist yet); the P6 baseline csv is the real 12,146-token csv."""
    dump, table = "F:/TanitAD_p6read_work/run/dump/dump_p6.npz", "F:/TanitAD_p6read_work/run/score/table_p6.npz"
    Z, T = np.load(dump, allow_pickle=False), np.load(table, allow_pickle=False)
    world = world_from_dump(dump)
    tt = {str(t): i for i, t in enumerate(T["token"])}
    assert [str(t) for t in T["token"]] == [str(t) for t in Z["token"]]
    tab = dict(token=np.array([str(t) for t in T["token"]]), pdms=T["pdms"], sub=T["sub"], valid=T["valid"], red=T["red_pick_pdms"])
    logits, props = Z["logits"], Z["props8"]
    mk = lambda n: Arm(n, Z["token"], logits, props)    # noqa: E731
    arms = {n: mk(n) for n in ("C0", "C0r", "S1E")}
    return Inputs(world, logits, props, read_harness_csv(csv_path), list(world.token), tab, arms, Z["pick"].astype(np.int64), Z["fingerprint"], None, True,
                  {"selfcheck": "C3 stand-in", "dump": (dump, sha256_file(dump)), "table": (table, sha256_file(table)), "p6_csv": (csv_path, sha256_file(csv_path))}, p6_token=Z["token"])


def zero_difference_report(res):
    """walk the result: every arm-vs-P6 and arm-vs-arm difference (point, lo, hi) must be exactly 0.0 when the arms are P6 itself"""
    bad, n = [], 0

    def walk(o, path):
        nonlocal n
        if isinstance(o, dict):
            if {"point", "lo", "hi"} <= set(o) and any(s in path for s in ("_vs_P6", "vs_C0", "diff_vs", "M1_vs", "G2_delta", "G3_overspeed_diff", "G3_vs")) and "level" not in path:
                n += 1
                for q in ("point", "lo", "hi"):
                    if o[q] is not None and o[q] != 0.0:
                        bad.append((path, q, o[q]))
                if o["point"] is None:
                    bad.append((path, "point", None))
            for k, v in o.items():
                if str(k).startswith("_") and "family" in path:
                    continue                      # levels (selection quality), not differences
                walk(v, path + "/" + str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, path + f"[{i}]")

    walk(res.get("endpoints", {}), "endpoints")
    walk(res.get("family_clause", {}), "family")
    return {"n_difference_records": n, "n_nonzero_or_missing": len(bad), "examples": bad[:8]}


# ------------------------------------------------------------------------------------------------ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["read", "emit-changed", "selfcheck"])
    ap.add_argument("--p6-proposals", default=DEFAULT_P6_PROPOSALS)
    ap.add_argument("--p6-csv", default=DEFAULT_P6_CSV)
    ap.add_argument("--e2-tokens")
    ap.add_argument("--e2-table")
    ap.add_argument("--export", default=DEFAULT_EXPORT)
    ap.add_argument("--world-dump", default=None, help="stand-in world from a dump_p6.npz instead of the export")
    ap.add_argument("--arm", action="append", default=[], help="NAME=DIR (C0, C0r, S1E, S1); DIR holds e3_logits.npz [e2_logits.npz] [e3_changed_scores.csv]; for emit-changed ENS3 is also allowed as NAME only")
    ap.add_argument("--ens3-csv", default=None, help="harness csv of ENS3's changed picks")
    ap.add_argument("--c3-tokens", default=None, help="tokens_all.json (C3 strata) for the descriptive E1 row")
    ap.add_argument("--no-tactical", action="store_true")
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "selfcheck":
        inp = inputs_selfcheck(a.p6_csv)
        res = run_read(inp)
        res["selfcheck_zero_difference"] = zero_difference_report(res)
        jp, mp = write_outputs(res, a.out, "selfcheck_read")
        print(json.dumps({"status": res.get("status"), "zero_difference": res["selfcheck_zero_difference"], "adopted": res.get("verdict", {}).get("adopted_arm"),
                          "outcome": res.get("verdict", {}).get("outcome", {}).get("primary"), "json": jp}, indent=1, default=jdefault))
        return 0 if res.get("status") == "READ" and res["selfcheck_zero_difference"]["n_nonzero_or_missing"] == 0 else 1
    names = []
    specs = []
    for s in a.arm:
        if "=" in s:
            specs.append(s)
        else:
            names.append(s)
    a.arm = specs
    if not (a.e2_tokens and a.e2_table):
        raise SystemExit("--e2-tokens and --e2-table are required")
    inp = inputs_from_files(a)
    if a.cmd == "emit-changed":
        out = emit_changed(inp, [s.split("=", 1)[0] for s in specs] + names, a.out)
        print(json.dumps(out, indent=1, default=jdefault))
        return 0
    res = run_read(inp)
    jp, mp = write_outputs(res, a.out)
    print(json.dumps({"status": res.get("status"), "why": res.get("why"), "verdict": res.get("verdict", {}).get("verdict"), "outcome": res.get("verdict", {}).get("outcome", {}).get("primary"),
                      "json": jp, "md": mp}, indent=1, default=jdefault))
    return 0 if res.get("status") == "READ" else 2


if __name__ == "__main__":
    sys.exit(main())
