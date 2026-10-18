#!/usr/bin/env python3
"""safeobj_outcome: bars G, L, E and the outcome table O1-O7 of PREREG_SAFEOBJ.md (sections 5-6) WITH ADDENDUM 1 (PREREG_SAFEOBJ.addendum1.md, R1-R4) applied MECHANICALLY.
IMPL_PLAN step 8, second half.  Stdlib + numpy-free (imports eval/safeobj_read.py, never edits it).

    python eval/safeobj_outcome.py --arm R0d=<RLS1>/arms/R0d [--arm R1d=...] [--arm R1o=...] [--arm R0=<banked R0 arm dir>]
                                   [--navtest-e <navtest_E.json>] [--dose <dose.json>] [--addendum <PREREG_SAFEOBJ.addendum1.md>] [--prereg <PREREG_SAFEOBJ.md>] [--out <safeobj_outcome.json>]

It writes ONE file (`--out`, default `<parent of the first constrained arm dir>/safeobj_outcome.json`) and prints `ZZSAFEOBJ_BAR <bar> <verdict> <why>` and `ZZSAFEOBJ_OUTCOME <arm> <outcome> [flags] <why>`.

THE TOOL REFUSES TO RUN (rc 4, `ZZSAFEOBJ_REFUSED`, no record written) when the addendum is missing or its sha256 is not the registered one: it applies the registered rules or none.

WHAT EACH BAR READS (every number in the record names its source file and that file's sha256):
  S   safeobj_read.read_arm(arm_dir), re-run here and required to EQUAL the arm's `safeobj_S.json` (record_sha256): a stale or hand-edited S file reads NOT_READ.
  G   (R1) `<arm>/rlval_B_vs_ownA.json` (rl_val_eval.py --scores A=..._A.npz B=..._B.npz --base A: the paired B - own-A read WITH its interval) cross-checked against `<arm>/rlval_vs_A0.json`, and
      dose.json's EXACT Pareto vertices (pinned by sha256).  PASS iff the PDMS lower 95 % bound is > 0 AND the PDMS point >= G(t), t = the TTC point delta, G(t) = max(+0.60, F(t)), F = the linear
      interpolation of dose.json's alpha 0.25 / 0.375 / 0.5 points at full precision, flat beyond the last vertex.  >= as written.  No dose.json with the pinned sha256 -> NOT_READ.
  L   (R4) point TTC and PDMS B - own-A at u1000 of R0 (banked), R0d, R1d, R1o, each from `<arm>/rlval_vs_A0.json`.  DECIDED ON TTC ONLY; PDMS is reported with its yardstick.  PENDING until R1o is given.
  E   (R4) `--navtest-e` (the navtest T1 contract below).  Pooled contrast = the mean of the two pairs' contrasts, interval from a JOINT resample of the 136 logs; clause 4 PER PAIR (four bounds);
      seed floor = the larger of the RL-val training-seed floor (PDMS, from L's inputs) and the navtest seed floor |R0d - R1d|, then the I-rep.  PENDING until a file is given.  ADOPT / REFUTED / NOT_PROVEN.
  O   (R2, R3) a FIXED PRECEDENCE, the first stage that matches is the outcome:  O6 > O7 > O4/O5 > O1/O2/O3.
        O6   lambda_TTC = lambda_NC = 0.0 at every logged update (wins over everything).
        O7   any kill whose recorded rule is not K3 (a `K-MON` kill whose recorded reason is the K3 rule counts as K3), or "g3 alone" (S fails and no g5 breach at any check).
        O4/O5 (breached = a K3 kill, or S fails with at least one g5 breach):  exactly one of O4, O5 -> that row;  both -> O4+O5;  neither -> O4 flagged SLACK.
              window = the 100 logged updates ending AT the first breached check (a K3 kill: the first of the two consecutive breaches); violation = each term's SIGNED window mean of
              `con_g_ttc` / `con_g_nc`, held iff both <= 0.001; post-ramp share and "lambda reached lambda_max" are counted THROUGH the first breached check.
        O1/O2/O3  S holds, no kill: O1 if G holds, O2 if the PDMS lower bound > 0 and the point < G(t), O3 if the lower bound <= 0.
      Facts that cannot be read are three-valued (None) and an unknown that could change the answer reads NOT_READ.
  The record lists, under `prereg_ambiguities`, ONLY the readings the addendum does not settle.

FAIL CLOSED.  A missing, unparsable, inconsistent or unpinned input reads NOT_READ for that bar.  PENDING is ONLY for an input that does not exist yet by design (no `--navtest-e`; no R1o arm; an arm
directory with neither a `done` event nor a KILLED.json).  PENDING is never PASS.  Exit code: 0 = every constrained arm got a determinate outcome (O1-O7 or O4+O5), 2 = otherwise,
3 = usage, 4 = refused (the addendum is not the registered file).

navtest_E.json (the contract the IMPL step 9 adapter must write; no producer exists yet), all values x100:
  {"tier": "T1", "estimator": "...10000 resamples...seed 20260927", "n_tokens": 12146, "n_logs": 136,
   "pairs": {"R0d": {SUB: {"mean": m, "ci95": [lo, hi]} for SUB in PDMS, NC, DAC, EP, TTC, C, DDC}, "R1d": {...}},
   "pooled": {"PDMS": {"mean": m, "ci95": [lo, hi]}, "resample": "joint"}, "floors": {"irep_abs_delta": b}}
  pooled.PDMS.mean must be the mean of the two pairs' PDMS means; "resample": "joint" declares one index set per draw applied to both pairs.  A `floors.seed_floor_abs` is IGNORED (the tool derives the floors).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import safeobj_read as SR  # noqa: E402  (the bar-S reader: imported, never edited)

TOOL = "safeobj_outcome/2"
OUT_NAME = "safeobj_outcome.json"
BA_NAME = "rlval_B_vs_ownA.json"
RLVAL_NAME = "rlval_vs_A0.json"
S_NAME = SR.OUT_NAME

# ============================================================ the registered files (PREREG_SAFEOBJ.md, its addendum 1, dose.json); pinned against their text by eval/test_safeobj_outcome.py
PREREG_SHA256 = "b404c2fb7b2f724775a1dda5a9de06c9778505a3374e6467060ae7c7a9e04bcb"
ADDENDUM_NAME = "PREREG_SAFEOBJ.addendum1.md"
ADDENDUM_SHA256 = "3c6a25c14d4266230281b03d92d8201d21dbaa0ab27d9699f934af69a96761c4"       # registered before any data (PREREG_SAFEOBJ.addendum1.sha256)
DOSE_SHA256 = "735a6d909bc0ef41b87421bed225d3b206b0b3bd886a5e0a9776088c2e8d9d5c"            # addendum R1: "dose.json, sha256 735a6d909bc0ef41..."
DOSE_ALPHAS = ("0.25", "0.375", "0.5")                                                      # addendum R1: key full[alpha].{TTC,PDMS}.mean (x100)
N_BOOT, SEED = 10000, 20260927                                                              # section 4: the estimator
RLVAL_SETS, RLVAL_LOGS = 3000, 75                                                           # addendum R1: "RL-val (3,000 sets / 75 logs)"
NAVTEST_TOKENS, NAVTEST_LOGS = 12146, 136                                                   # section 4 / addendum R4: "the same 12,146 tokens", "the 136 navtest logs"
# --- bar G (section 5 + addendum R1): PDMS lower bound > 0 AND point >= G(t); G(t) = max(+0.60, F(t)); F = dose.json's exact vertices, flat beyond the last
G_FLOOR = 0.60                                                                              # a stated constant (52 % of R0's +1.15), NOT a dose.json number: exact
G_POINTS_PRINTED = ((-0.03, 0.60), (-0.27, 0.71), (-0.47, 0.84))                            # the 2-decimal RENDERINGS printed in section 5: NEVER used for the bar (addendum R1); a cross-check only
G_LOWER_BOUND_MIN = 0.0                                                                     # "the PDMS lower 95 % bound is > 0"
# --- section 6, O4 / O5 / O6 + addendum R3
O4_VIOLATION_MAX = 0.001                                                                    # "the in-sample violation ... is <= 0.001"
O4_WINDOW_UPDATES = 100
O4_LAM_POS_SHARE_MIN = 0.5                                                                  # "lambda_TTC or lambda_NC > 0 on >= 50 % of post-ramp updates"
LOG_EVERY = 25                                                                              # the trainer writes one `train` row per 25 updates (scorer_finetune.py: `if upd % 25 == 0`)
LAM_MAX = 10.0                                                                              # PREREG section 2: lambda_max
REGISTERED_EPO_CON = {"mode": "g5", "comps": ["ttc", "nc"], "cols": [3, 0], "eps": 0.0, "lr": 4.0, "max": 10.0, "lam0": 0.0, "ref": "armA_detached"}      # PREREG section 2 (the data event's BUILT facts)
REGISTERED_CON_TEMP = 0.1                                                                   # PREREG section 2: the constraint's policy is `epo_policy` at T = 0.1 ("a second temperature would be a second lever")
REGISTERED_RAMP = 200                                                                       # --epo-ramp default; R0's banked argv does not name it (PREREG_RL_S1 section 10: "the ramp (200 of 500 updates)")
# addendum R2: the fixed precedence.  The stages are evaluated IN THIS ORDER and the first that matches is the outcome.
PRECEDENCE = ("O6", "O7", "O45", "O123")
# --- addendum R2 / A-O9: what a kill's RECORDED REASON says.  The real monitor's K3 text (rls1_checks.judge): "g3/g5 bound breached at <tag> and <tag>".
K3_REASON_RX = re.compile(r"g3/g5 bound breached at (update\d+) and (update\d+)")
# --- bar L (section 5)
L_FLOOR_MULT = 2.0                                                                          # "> 2 x F_seed"
# --- bar E (section 5)
E_EACH_MIN, E_MEAN_MIN = 0.25, 0.5
E_REFUTED_UPPER = 0.25
E_FLOOR_MULT = 2.0
E_POOL_TOL = 1e-9                                                                           # pooled.mean vs the mean of the two pairs' means
SUBS = ("NC", "DAC", "EP", "TTC", "C", "DDC")
CON_COLS = ("con_lam_ttc", "con_lam_nc", "con_g_ttc", "con_g_nc", "con_cap_hits", "con_lam_pos_updates", "con_lam_max_ttc", "con_lam_max_nc")
CONSTRAINED = ("R0d", "R1d")
ARM_NAMES = ("R0d", "R1d", "R1o", "R0")
DETERMINATE = ("O1", "O2", "O3", "O4", "O5", "O6", "O7", "O4+O5")
TOL = 1e-9                                                                                  # float agreement between two files that carry the same quantity

# ============================================================ what the addendum SETTLES (traceability) and what it does NOT (listed in every record; NEVER resolved here)
ADDENDUM_SETTLES = {
    "A-G1": "R1: dose.json's exact vertices (sha256-pinned), the +0.60 floor exact, >=", "A-G2": "R1: kept as written (the own-A vs C0-A disclosure is NOT computed here: see A-G2b)",
    "A-G3": "R1: the u1000 B - own-A PDMS contrast, two-sided 95 % percentile interval of the section 4 estimator; t = that read's TTC point delta",
    "A-O1": "R2.1: O6 wins over every other row", "A-O2": "R2.2: 'g3 alone' = S fails and no g5 breach at any check; it is O7, not O4/O5", "A-O3": "R2.3: both -> O4+O5; neither -> O4 flagged SLACK",
    "A-O4": "R3: the SIGNED window mean of the logged per-update g, both terms <= 0.001", "A-O5": "R3: the 100 logged updates before the first breached check (a K3 kill: the first of the two consecutive breaches); ramp overlap flagged",
    "A-O6": "R3: post-ramp share counted through the first breached check", "A-O7": "R3: lambda_max reached at any update through the first breached check",
    "A-O8": "R3: O6 through the last logged update, the unlogged tail disclosed", "A-O9": "R2.2: any kill whose recorded rule is not K3 is O7; a K-MON kill applying the K3 rule counts as K3 (see A-O9b)",
    "A-O10": "R4: R1d O1 while R0d is not reads NOT_APPLICABLE", "A-L1": "R4: L is decided on TTC only; PDMS reported beside its yardstick", "A-L2": "R4: as written",
    "A-E1": "R4: pooled contrast = the mean of the pairs' contrasts, joint resample of the 136 logs (see A-E1b)", "A-E2": "R4: clause 4 per pair, TTC and NC upper bounds (four bounds)",
    "A-E3": "R4: seed floor = the larger of the RL-val training-seed floor and the navtest seed floor (see A-E3b)", "A-E4": "R4: 'the mean contrast' = the mean of the two pairs' navtest PDMS contrasts",
}
PREREG_AMBIGUITIES = [
    {"id": "A-E3b", "where": "addendum R4 / section 5 E clause 3 / section 7", "text": "the addendum takes 'the larger of section 7's RL-val training-seed floor and the navtest seed floor', but section 7's training-seed row names TWO quantities "
     "(|R0d - R1d| constrained, |R0 - R1o| unconstrained) and does not say PDMS or TTC.  Read here: PDMS (E's unit), the larger of the two terms = bar L's PDMS F_seed; both terms are in the record.  "
     "Without R0 and R1o (RL-val reads) the floor cannot be built and clause 3 is unread: E reads NOT_READ unless REFUTED (which needs no floor)."},
    {"id": "A-G2b", "where": "addendum R1 A-G2", "text": "'the own-A vs C0-A difference is reported beside G as a disclosure': this tool is not given C0-A's u1000 dumps and does NOT compute it.  The G record carries "
     "disclosures.A-G2 = NOT_COMPUTED_BY_THIS_TOOL; the owner reports it beside G (section 7's 'same run again' floor, precedent R0-A vs C0-A PDMS +0.01 [-0.13, +0.14])."},
    {"id": "A-O9b", "where": "addendum R2.2", "text": "'a K-MON kill applying the K3 rule counts as K3, judged by its recorded reason': the addendum gives no text for that reason.  Read here: the real monitor's K3 text "
     "'g3/g5 bound breached at <tag> and <tag>' (rls1_checks.judge).  Any other recorded reason (the chain writes 'kill rules unenforceable: monitor died rc N' / 'not READY within Ns') is not the K3 rule, so the kill is O7."},
    {"id": "A-G3b", "where": "addendum R1 A-G3", "text": "'(3,000 sets / 75 logs)' is parenthetical: the addendum does not say a different pool is a refusal.  The G record carries the pool counts and `pool_matches_registered`; a mismatch is a note on the "
     "arm, not a NOT_READ."},
    {"id": "A-E1b", "where": "addendum R4 A-E1", "text": "the JOINT-resample pooled interval is the producer's (IMPL step 9); this tool cannot re-derive it from per-token data.  It checks the declaration (`pooled.resample == 'joint'`), "
     "`n_tokens == 12146`, `n_logs == 136` and that pooled.PDMS.mean is the mean of the two pairs' PDMS means; the interval itself is taken from the file."},
    {"id": "A-L1b", "where": "addendum R4 A-L1", "text": "'the same 2 x F_seed yardstick printed beside' the PDMS cost: F_seed is recomputed on the PDMS contrasts (its own unit); the TTC F_seed that decides L is printed beside it.  Neither moves the verdict."},
]


# ============================================================ small helpers
class AddendumRefused(Exception):
    """the addendum is missing or is not the registered file: the tool applies the registered rules or none."""


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def and3(*xs):
    """Kleene AND over True / False / None (None = unknown)."""
    if any(x is False for x in xs):
        return False
    return None if any(x is None for x in xs) else True


def or3(*xs):
    if any(x is True for x in xs):
        return True
    return None if any(x is None for x in xs) else False


def not3(x):
    return None if x is None else (not x)


class Ctx:
    """the sha256 of every file a number comes from (computed once, recorded once)."""

    def __init__(self):
        self.inputs = {}

    def sha(self, path):
        path = os.path.abspath(path)
        if path not in self.inputs:
            try:
                self.inputs[path] = SR.sha256_file(path)
            except OSError:
                self.inputs[path] = None
        return self.inputs[path]

    def N(self, value, path):
        """a number WITH its provenance."""
        return {"value": value, "source": os.path.abspath(path), "sha256": self.sha(path)}


def _jsonl(path):
    """-> (rows, bad_lines, last_line_bad).  A half-written FINAL line (a SIGKILLed trainer) is tolerated and reported."""
    rows, bad, last_bad = [], 0, False
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [ln for ln in f.read().split("\n") if ln.strip()]
    for i, ln in enumerate(lines):
        try:
            r = json.loads(ln)
        except ValueError:
            bad += 1
            last_bad = i == len(lines) - 1
            continue
        if isinstance(r, dict):
            rows.append(r)
        else:
            bad += 1
    return rows, bad, last_bad


# ============================================================ the addendum: the tool refuses to run without the registered file
def check_addendum(path):
    """-> the addendum record.  Raises AddendumRefused when the file is not given, not there, or not the registered bytes."""
    if not path:
        raise AddendumRefused("no addendum file given (--addendum) and none beside the prereg")
    if not os.path.isfile(path):
        raise AddendumRefused(f"the addendum {path} does not exist")
    sha = SR.sha256_file(path)
    if sha != ADDENDUM_SHA256:
        raise AddendumRefused(f"the addendum {path} has sha256 {sha[:12]}..., not the registered {ADDENDUM_SHA256[:12]}...: these are not the registered rules")
    return {"name": ADDENDUM_NAME, "path": os.path.abspath(path), "sha256_registered": ADDENDUM_SHA256, "sha256_checked": sha}


# ============================================================ bar G: dose.json's exact frontier
def frontier(t, points, floor=G_FLOOR):
    """G(t) = max(floor, F(t)); F = the linear interpolation of `points` (TTC delta descending, PDMS ascending), flat beyond the last vertex and flat at the first vertex's PDMS above it.
    At a vertex the vertex's own PDMS is returned (no float round trip)."""
    xs = [p[0] for p in points]
    if t >= xs[0]:
        f = points[0][1]
    elif t <= xs[-1]:
        f = points[-1][1]
    else:
        f = None
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            if x1 <= t <= x0:
                f = y1 if t == x1 else (y0 if t == x0 else y0 + (t - x0) * (y1 - y0) / (x1 - x0))
                break
    return max(floor, f)


def dose_points(dose):
    """dose.json's exact Pareto vertices for alpha 0.25 / 0.375 / 0.5 (full-set reads, full precision) -> ((ttc, pdms), ...) or None."""
    try:
        full = dose["full"]
        pts = tuple((float(full[a]["TTC"]["mean"]), float(full[a]["PDMS"]["mean"])) for a in DOSE_ALPHAS)
    except (KeyError, TypeError, ValueError):
        return None
    return pts if all(math.isfinite(v) for p in pts for v in p) else None


def load_dose(path, ctx):
    """-> {"ok", "points", "problems", "source", "printed_match"}.  Never raises.  The file must have the pinned sha256 BEFORE a number is taken from it."""
    out = {"ok": False, "points": None, "problems": [], "source": None, "printed_match": None}
    if not path:
        out["problems"].append("no dose.json given: bar G's frontier has no vertices (addendum R1)")
        return out
    if not os.path.isfile(path):
        out["problems"].append(f"dose.json {path} not found: bar G's frontier has no vertices (addendum R1)")
        return out
    sha = ctx.sha(path)
    if sha != DOSE_SHA256:
        out["problems"].append(f"dose.json {path} has sha256 {str(sha)[:12]}..., not the pinned {DOSE_SHA256[:12]}...: its vertices are not the registered ones")
        return out
    try:
        pts = dose_points(_load(path))
    except (OSError, ValueError):
        out["problems"].append(f"dose.json {path} does not parse")
        return out
    if pts is None:
        out["problems"].append(f"dose.json {path} has no full-set alpha {' / '.join(DOSE_ALPHAS)} points")
        return out
    if not all(a[0] > b[0] and a[1] < b[1] for a, b in zip(pts[:-1], pts[1:])):
        out["problems"].append("dose.json's vertices are not a Pareto frontier (TTC strictly falling, PDMS strictly rising)")
        return out
    out.update({"ok": True, "points": pts, "printed_match": [tuple(round(v, 2) for v in p) for p in pts] == [tuple(p) for p in G_POINTS_PRINTED],
                "source": {"file": os.path.abspath(path), "sha256": sha, "pinned_sha256": DOSE_SHA256, "keys": "full[alpha].{TTC,PDMS}.mean (x100)", "alphas": list(DOSE_ALPHAS)}})
    return out


def read_ba(arm_dir, ctx):
    """-> (info, problems).  info = {"pdms_mean", "pdms_lo", "pdms_hi", "ttc_mean"} (x100, B - own A, u1000), each a provenance record, + "read" (the file and its sha256) + "pool".  Never raises."""
    problems = []
    p = os.path.join(arm_dir, BA_NAME)
    pr = os.path.join(arm_dir, RLVAL_NAME)
    if not os.path.exists(p):
        return None, [f"{BA_NAME} is missing: run  rl_val_eval.py --scores A=<arm>/scores/scores_update1000_A.npz B=<arm>/scores/scores_update1000_B.npz --base A ... --out {BA_NAME}  (OUTCOME_BUILD.md has the command)"]
    try:
        d = _load(p)
        ctx.sha(p)
        if not isinstance(d, dict):
            return None, [f"{BA_NAME} is not an object"]
        if d.get("tool") != "rl_val_eval/1" or d.get("rule") != "navsim_v1_lane" or d.get("base") != "A" or set(d.get("arms") or {}) != {"A", "B"}:
            problems.append(f"{BA_NAME} is not an rl_val_eval read of arms A and B with base A under rule navsim_v1_lane")
        est = str(d.get("estimator", ""))
        if f"{N_BOOT} resamples" not in est or f"seed {SEED}" not in est:
            problems.append(f"{BA_NAME}: the estimator is not the registered one ({N_BOOT} resamples, seed {SEED}): {est[:90]!r}")
        q = d["paired_vs_base"]["B"]["quantities"]
        pd, tt = q["PDMS"], q["TTC"]
        lo, hi = float(pd["ci95"][0]), float(pd["ci95"][1])
        vals = (float(pd["mean"]), lo, hi, float(tt["mean"]))
        if not all(math.isfinite(v) for v in vals) or not lo - TOL <= vals[0] <= hi + TOL:
            problems.append(f"{BA_NAME}: a non-finite number or a PDMS point outside its own interval")
        # tie to the chain's own u1000 read: same dumps, same sets, same means
        if not os.path.exists(pr):
            problems.append(f"{RLVAL_NAME} is missing: {BA_NAME} cannot be tied to the arm's own dumps")
        else:
            r = _load(pr)
            ctx.sha(pr)
            h_ba, h_r = (d.get("inputs") or {}).get("scores_sha256") or {}, (r.get("inputs") or {}).get("scores_sha256") or {}
            if not (h_ba.get("A") and h_ba.get("A") == h_r.get("CTRL") and h_ba.get("B") == h_r.get("R")):
                problems.append(f"{BA_NAME} was read from dumps that are NOT the ones {RLVAL_NAME} read as CTRL / R (scores_sha256 differ): B may not be compared with its own A")
            a = r["arms"]
            ttc_r = float(a["R"]["subscores_x100"]["TTC"]) - float(a["CTRL"]["subscores_x100"]["TTC"])
            pdms_r = float(a["R"]["PDMS_x100"]) - float(a["CTRL"]["PDMS_x100"])
            if abs(ttc_r - vals[3]) > TOL or abs(pdms_r - vals[0]) > TOL:
                problems.append(f"{BA_NAME} and {RLVAL_NAME} disagree on the u1000 B - A point deltas (TTC {vals[3]!r} vs {ttc_r!r}, PDMS {vals[0]!r} vs {pdms_r!r})")
            if (d.get("eligibility") or {}).get("sets_used") != (r.get("eligibility") or {}).get("sets_used"):
                problems.append(f"{BA_NAME} and {RLVAL_NAME} read different numbers of sets")
        for name, f in (("A", "scores_update1000_A.npz"), ("B", "scores_update1000_B.npz")):          # when the dumps are on disk, they must be the ones read
            fp = os.path.join(arm_dir, "scores", f)
            if os.path.exists(fp):
                if ((d.get("inputs") or {}).get("scores_sha256") or {}).get(name) != ctx.sha(fp):
                    problems.append(f"{BA_NAME} read {name} from a file that is NOT this arm's scores/{f} (sha256 differs)")
        pool = {"sets_used": (d.get("eligibility") or {}).get("sets_used"), "n_logs_B": ((d.get("arms") or {}).get("B") or {}).get("n_logs")}
        pool["pool_matches_registered"] = bool(pool["sets_used"] == RLVAL_SETS and pool["n_logs_B"] == RLVAL_LOGS)
        pool["registered"] = {"sets": RLVAL_SETS, "logs": RLVAL_LOGS}
    except (KeyError, TypeError, ValueError, IndexError, OSError, AttributeError) as e:
        return None, problems + [f"{BA_NAME} could not be interpreted: {type(e).__name__}: {str(e)[:120]}"]
    if problems:
        return None, problems
    return {"pdms_mean": ctx.N(vals[0], p), "pdms_lo": ctx.N(lo, p), "pdms_hi": ctx.N(hi, p), "ttc_mean": ctx.N(vals[3], p),
            "read": {"file": os.path.abspath(p), "sha256": ctx.sha(p), "scores_sha256": (d.get("inputs") or {}).get("scores_sha256")}, "pool": pool}, []


def bar_G(ba, problems, dose, ctx):
    """-> the G record.  ba from read_ba (None = unreadable); dose from load_dose (None / not ok = no frontier = NOT_READ)."""
    if dose is None or not dose.get("ok"):
        return {"verdict": "NOT_READ", "problems": list(problems) + list((dose or {}).get("problems") or ["no dose.json"])}
    if ba is None:
        return {"verdict": "NOT_READ", "problems": problems}
    t, pt, lo = ba["ttc_mean"]["value"], ba["pdms_mean"]["value"], ba["pdms_lo"]["value"]
    bar = frontier(t, dose["points"])
    lb_pos, pt_ok = lo > G_LOWER_BOUND_MIN, pt >= bar
    src = dose["source"]["file"]
    rec = {"verdict": "PASS" if (lb_pos and pt_ok) else "FAIL", "lower_bound_gt_0": lb_pos, "point_ge_bar": pt_ok, "bar_at_t": bar, "margin_point_minus_bar": pt - bar,
           "ttc_point_delta_x100": ba["ttc_mean"], "pdms_point_delta_x100": ba["pdms_mean"], "pdms_ci95_lower": ba["pdms_lo"], "pdms_ci95_upper": ba["pdms_hi"],
           "vertices": [{"alpha": a, "ttc_x100": ctx.N(p[0], src), "pdms_x100": ctx.N(p[1], src)} for a, p in zip(DOSE_ALPHAS, dose["points"])],
           "vertices_source": dose["source"], "printed_vertices_round_match": dose["printed_match"], "floor": G_FLOOR, "read": ba.get("read"), "pool": ba.get("pool"),
           "disclosures": {"A-G2": "NOT_COMPUTED_BY_THIS_TOOL: the own-A vs C0-A difference needs C0-A's dumps (see prereg_ambiguities A-G2b)"}, "problems": list(problems)}
    return rec


# ============================================================ the trainer's own log: lambda trajectory, in-sample violation, cap hits
def _registered_view(blk):
    """PICKCON (raw/2026-10-18-so-pickcon): `EpoDual.built_facts()` gained a `temp` key -- the T the constraint's softmax RUNS at.  The registered R0d / R1d logs were written by trainers that predate it
    (the banked R0d `sft.jsonl` carries the eight-key block), so a block WITHOUT the key is the legacy shape and is compared exactly as it always was.  A block WITH the key is compared as the registered
    block ONLY when temp is exactly the registered T = 0.1 (PREREG section 2: a second temperature would be a second lever); any other value -- another T, None, a bool, a string -- leaves the key in,
    the block then differs from the registered one, and the arm is refused (`not the registered`)."""
    if isinstance(blk, dict) and "temp" in blk:
        t = blk["temp"]
        if isinstance(t, (int, float)) and not isinstance(t, bool) and t == REGISTERED_CON_TEMP:
            return {k: v for k, v in blk.items() if k != "temp"}
    return blk          # PICKCON-LEGACY: the pre-PICKCON shape (no `temp`), or a T that is not the registered one


def read_log(path, c, ctx, registered=None, registered_ramp=REGISTERED_RAMP):
    """-> facts dict (never raises).  Facts: lam_zero (O6, through the last logged update), and, only when `c` (the update of the first breached check) is given, lam_pos_share_ok (post-ramp share
    THROUGH c), cap_hit (lambda_max reached through c) and violation_ok (the 100 logged updates ending at c); every one True / False / None (None = unreadable), with the numbers and their source."""
    registered = REGISTERED_EPO_CON if registered is None else registered
    out = {"readable": False, "problems": [], "notes": [], "lam_zero": None, "lam_pos_share_ok": None, "cap_hit": None, "violation_ok": None, "numbers": {}, "window_overlaps_ramp": None,
           "columns_read": {"ttc": "con_g_ttc", "nc": "con_g_nc"}}
    pb = out["problems"]
    if not os.path.exists(path):
        pb.append("sft.jsonl is missing")
        return out
    try:
        ctx.sha(path)
        rows, bad, last_bad = _jsonl(path)
        if bad:
            (out["notes"] if (bad == 1 and last_bad) else pb).append(f"{bad} unparsable line(s) in sft.jsonl" + (" (only the final, half-written one: tolerated)" if (bad == 1 and last_bad) else ""))
        data = next((r for r in rows if r.get("event") == "data"), None)
        if data is None:
            pb.append("no `data` event: the BUILT facts of the run are unknown")
            return out
        data = dict(data, epo_con=_registered_view(data.get("epo_con")))          # PICKCON: a `temp` of exactly the registered T is the same block as the legacy one; anything else stays and is refused below
        if data.get("epo_con") != registered:
            pb.append(f"the data event's BUILT epo_con block {data.get('epo_con')!r} is not the registered {registered!r}")
        ramp = (data.get("epo") or {}).get("ramp")
        if ramp != registered_ramp or isinstance(ramp, bool):
            pb.append(f"the BUILT --epo-ramp is {ramp!r}, not {registered_ramp}")
        trains = [r for r in rows if r.get("event") == "train"]
        if not trains:
            pb.append("no `train` row")
            return out
        ups = [r.get("update") for r in trains]
        if ups != list(range(LOG_EVERY, LOG_EVERY * len(trains) + 1, LOG_EVERY)):
            pb.append(f"the `train` rows are not the contiguous {LOG_EVERY}-update series from {LOG_EVERY} (updates {ups[:4]}...{ups[-2:]})")
        for r in trains:
            miss = [k for k in CON_COLS if not _num(r.get(k))]
            if miss:
                pb.append(f"train row at update {r.get('update')!r} lacks finite {miss}")
                break
        if pb:
            return out
        by_u = {r["update"]: r for r in trains}
        last = trains[-1]
        U = last["update"]
        lam_max_cfg = registered["max"]
        # internal consistency of the cumulative columns
        for a, b in zip(trains[:-1], trains[1:]):
            for k in ("con_cap_hits", "con_lam_pos_updates", "con_lam_max_ttc", "con_lam_max_nc"):
                if b[k] < a[k]:
                    pb.append(f"the cumulative column {k} DECREASES between updates {a['update']} and {b['update']}")
        for r in trains:
            for cc in ("ttc", "nc"):
                if not 0.0 <= r[f"con_lam_{cc}"] <= lam_max_cfg + TOL or r[f"con_lam_max_{cc}"] + TOL < r[f"con_lam_{cc}"]:
                    pb.append(f"lambda_{cc} outside [0, lambda_max] or above its own running maximum at update {r['update']}")
                    break
            if (r["con_cap_hits"] > 0) != (r["con_lam_max_ttc"] >= lam_max_cfg or r["con_lam_max_nc"] >= lam_max_cfg):
                pb.append(f"cap_hits and lam_seen_max disagree about reaching lambda_max at update {r['update']}")
                break
        done = next((r for r in reversed(rows) if r.get("event") == "done"), None)
        if done is not None and done.get("updates") == U:
            ec = done.get("epo_con") or {}
            if (ec.get("pos_updates"), ec.get("cap_hits"), (ec.get("lam_seen_max") or {}).get("ttc"), (ec.get("lam_seen_max") or {}).get("nc")) != (
                    last["con_lam_pos_updates"], last["con_cap_hits"], last["con_lam_max_ttc"], last["con_lam_max_nc"]):
                pb.append("the `done` event's dual state disagrees with the last `train` row")
        elif done is not None:
            out["notes"].append(f"the `done` event is at update {done.get('updates')!r}, the last train row at {U}: the cross-check is skipped")
        if pb:
            return out
        out["readable"] = True
        N = lambda v: ctx.N(v, path)          # noqa: E731
        out["numbers"].update({"last_logged_update": N(U), "ramp": N(ramp), "cap_hits_last": N(last["con_cap_hits"]), "lam_pos_updates_last": N(last["con_lam_pos_updates"]),
                               "lam_seen_max_ttc_last": N(last["con_lam_max_ttc"]), "lam_seen_max_nc_last": N(last["con_lam_max_nc"])})
        # ---- O6 (addendum R3 / A-O8): lambda == 0 at every update THROUGH THE LAST LOGGED UPDATE; the unlogged tail is disclosed
        out["lam_zero"] = bool(last["con_lam_pos_updates"] == 0 and last["con_lam_max_ttc"] == 0.0 and last["con_lam_max_nc"] == 0.0)
        if done is None:
            out["notes"].append(f"no `done` event (a killed arm): the updates after the last `train` row (update {U}; fewer than {LOG_EVERY}) are unlogged, O6 is read through update {U}")
        elif _num(done.get("updates")) and done["updates"] > U:
            out["notes"].append(f"{int(done['updates']) - U} update(s) after the last `train` row (update {U}) are unlogged: O6 is read through update {U}")
        # ---- everything below is read THROUGH THE FIRST BREACHED CHECK `c`, and only for a breached arm (addendum R3)
        if c is None:
            return out
        rc = by_u.get(c)
        if rc is None:
            out["notes"].append(f"no `train` row at the first breached check (update {c}): the post-ramp share, lambda_max and the window violation are unreadable")
            return out
        out["numbers"].update({"first_breached_update": N(c), "cap_hits_at_first_breach": N(rc["con_cap_hits"]),
                               "lam_seen_max_ttc_at_first_breach": N(rc["con_lam_max_ttc"]), "lam_seen_max_nc_at_first_breach": N(rc["con_lam_max_nc"])})
        # lambda reached lambda_max at any update from 0 through c
        out["cap_hit"] = bool(rc["con_cap_hits"] > 0 or rc["con_lam_max_ttc"] >= lam_max_cfg or rc["con_lam_max_nc"] >= lam_max_cfg)
        # lambda > 0 on >= 50 % of the post-ramp updates (lam_pg = 1: update numbers ramp+1 .. c), counted from the end of the ramp through c
        if c > ramp and ramp in by_u:
            m, k = c - ramp, rc["con_lam_pos_updates"] - by_u[ramp]["con_lam_pos_updates"]
            out["lam_pos_share_ok"] = bool(k >= O4_LAM_POS_SHARE_MIN * m)
            out["numbers"].update({"post_ramp_updates": N(m), "post_ramp_updates_with_lambda_gt_0": N(k), "post_ramp_share": N(k / m)})
        else:
            out["notes"].append(f"no `train` row at the ramp end (update {ramp}) or the first breached check (update {c}) is not past it: the post-ramp share of lambda > 0 is unreadable")
        # the in-sample violation: the SIGNED mean of the per-update g over the 100 logged updates ending at c, per term
        need = [c - O4_WINDOW_UPDATES + LOG_EVERY * i for i in range(1, O4_WINDOW_UPDATES // LOG_EVERY + 1)]
        out["window_overlaps_ramp"] = bool(c - O4_WINDOW_UPDATES + 1 <= ramp)
        if out["window_overlaps_ramp"]:
            out["notes"].append(f"the window (updates {c - O4_WINDOW_UPDATES + 1}..{c}) overlaps the lambda_pg ramp (updates 1..{ramp}) by {ramp - (c - O4_WINDOW_UPDATES)} update(s); the window does not move (addendum R3)")
        if all(u in by_u for u in need):
            g = {t: math.fsum(by_u[u][f"con_g_{t}"] for u in need) / len(need) for t in ("ttc", "nc")}
            out["numbers"].update({"window_end_update": N(c), "window_g_ttc_mean": N(g["ttc"]), "window_g_nc_mean": N(g["nc"])})
            out["violation_ok"] = bool(max(g["ttc"], g["nc"]) <= O4_VIOLATION_MAX)
        else:
            out["notes"].append(f"the four `train` rows {need} that make the 100-update window ending at update {c} are not all present: the in-sample violation is unreadable")
    except (KeyError, TypeError, ValueError, OSError) as e:
        out["readable"] = False
        pb.append(f"sft.jsonl could not be interpreted: {type(e).__name__}: {str(e)[:120]}")
    return out


# ============================================================ the arm: S, the kills, the breach table, G, the log, the table
def _kind(s):
    if "g3" in s:
        return "g3"
    if "g5" in s or " B-A " in s:
        return "g5"
    return "other"


def breach_info(arm_dir, S, ctx):
    """-> (first_tag or None, kinds {g3, g5} or None when nothing is located, per_tag).  Replay rows first (the S record's `checks`), then the live monitor's rows, then the numeric u1000 read."""
    per_tag = {}
    for row in S.get("checks") or []:
        if isinstance(row, dict) and row.get("present") and row.get("breaches"):
            per_tag.setdefault(row["tag"], []).extend(row["breaches"])
    mp = os.path.join(arm_dir, "kill_checks.jsonl")
    if os.path.exists(mp):
        try:
            ctx.sha(mp)
            for r in SR._rows(mp):
                if isinstance(r, dict) and r.get("tag") in SR.CHECK_TAGS and r["tag"] not in per_tag:
                    b, _ = SR.judge_check(r)
                    if b:
                        per_tag[r["tag"]] = list(b)
        except (OSError, KeyError, TypeError, AttributeError):
            pass
    for s in S.get("breaches") or []:
        if isinstance(s, str) and s.startswith("u1000 "):
            per_tag.setdefault("update1000", []).append(s)
    first = next((t for t in SR.CHECK_TAGS if t in per_tag), None)
    kinds = {_kind(s) for v in per_tag.values() for s in v} - {"other"} if per_tag else None
    return first, kinds, per_tag


def kill_pair_first(per_tag):
    """a K3 kill needs TWO CONSECUTIVE breached checks (rls1_checks.judge); the monitor kills at the first such pair.  -> the FIRST of that pair (addendum R3), or None."""
    return next((a for a, b in zip(SR.CHECK_TAGS[:-1], SR.CHECK_TAGS[1:]) if a in per_tag and b in per_tag), None)


def kill_rule(code, why):
    """the rule a kill APPLIED, judged by its recorded REASON, not by its code (addendum R2.2 / A-O9): a reason that is the K3 rule is K3 whatever the code (a `K-MON` record applying K3); otherwise the code."""
    if isinstance(why, str) and K3_REASON_RX.search(why):
        return "K3"
    return str(code)


def read_kills(arm_dir, S, ctx):
    """-> (kills [{source, code, why, rule}], unknown).  unknown = a KILLED.json that exists and cannot be read."""
    kills, unknown = [], False
    kp = os.path.join(arm_dir, "KILLED.json")
    if os.path.exists(kp):
        try:
            k = _load(kp)
            ctx.sha(kp)
            if isinstance(k, dict) and k.get("code") is not None:
                kills.append({"source": "KILLED.json", "code": str(k["code"]), "why": k.get("why"), "rule": kill_rule(k["code"], k.get("why"))})
            else:
                unknown = True
        except (OSError, ValueError):
            unknown = True
    if S.get("killed") == "?":
        unknown = True
    rp = os.path.join(arm_dir, "kill_replay.json")
    if os.path.exists(rp):
        try:
            ctx.sha(rp)
            rep = _load(rp)
            if isinstance(rep, dict) and rep.get("kill"):
                kills.append({"source": "kill_replay.json", "code": str(rep["kill"]), "why": rep.get("why"), "rule": kill_rule(rep["kill"], rep.get("why"))})
        except (OSError, ValueError, AttributeError):
            pass
    return kills, unknown


# ---- the outcome table with addendum R2's fixed precedence.  PURE: the whole table of section 6 is here, and only here.
def _raw_rows(f):
    """the rows of section 6 evaluated on their own (three-valued).  `breached` = a K3 kill, or S fails with at least one g5 breach."""
    return {
        "O1": and3(f["s_pass"], f["g_lb_pos"], f["g_point_ok"]),
        "O2": and3(f["s_pass"], f["g_lb_pos"], not3(f["g_point_ok"])),
        "O3": and3(f["s_pass"], not3(f["g_lb_pos"])),
        "O4": and3(f["breached"], f["violation_ok"], f["lam_pos_share_ok"]),
        "O5": and3(f["breached"], or3(f["cap_hit"], not3(f["violation_ok"]))),
        "O6": f["lam_zero"],
        "O7": or3(f["other_kill"], f["g3_alone"]),
    }


def _stage_o6(f, raw):
    v = raw["O6"]
    return ("unknown", "O6") if v is None else (("match", "O6", ()) if v else ("pass",))


def _stage_o7(f, raw):
    v = raw["O7"]
    return ("unknown", "O7") if v is None else (("match", "O7", ()) if v else ("pass",))


def _stage_o45(f, raw):
    if f["breached"] is None:
        return ("unknown", "breached")
    if not f["breached"]:
        return ("pass",)
    o4, o5 = raw["O4"], raw["O5"]
    if o4 is True and o5 is True:
        return ("match", "O4+O5", ())                       # held in-sample, lambda > 0 on >= 50 %, AND lambda reached lambda_max: both rows (addendum R2.3)
    if o4 is True and o5 is False:
        return ("match", "O4", ())
    if o4 is False and o5 is True:
        return ("match", "O5", ())
    if o4 is False and o5 is False:
        return ("match", "O4", ("SLACK",))                  # held in-sample, lambda > 0 on < 50 %, never at lambda_max: O4's meaning with the dual mostly idle (addendum R2.3)
    return ("unknown", "O4/O5")


def _stage_o123(f, raw):
    if f["breached"] is True:
        return ("pass",)
    rows = [raw["O1"], raw["O2"], raw["O3"]]
    hit = [k for k, v in zip(("O1", "O2", "O3"), rows) if v is True]
    if len(hit) == 1:
        return ("match", hit[0], ())
    if any(v is None for v in rows):
        return ("unknown", "O1/O2/O3")
    return ("inconsistent", "O1/O2/O3" if not hit else "+".join(hit))


STAGES = {"O6": _stage_o6, "O7": _stage_o7, "O45": _stage_o45, "O123": _stage_o123}


def classify(f):
    """f = the three-valued facts -> {"rows" (the raw table), "precedence", "decided_at", "matched_raw", "shadowed", "unknown", "outcome", "flags", "why"}."""
    raw = _raw_rows(f)
    matched = [k for k, v in raw.items() if v is True]
    unk = [k for k, v in raw.items() if v is None]
    base = {"rows": raw, "precedence": list(PRECEDENCE), "matched_raw": matched, "unknown": unk}
    for stage in PRECEDENCE:
        r = STAGES[stage](f, raw)
        if r[0] == "pass":
            continue
        if r[0] == "match":
            winners = set(r[1].split("+"))
            return {**base, "decided_at": stage, "outcome": r[1], "flags": list(r[2]), "shadowed": [k for k in matched if k not in winners],
                    "why": f"precedence O6 > O7 > O4/O5 > O1/O2/O3: stage {stage} matches" + (f"; flags {list(r[2])}" if r[2] else "")}
        if r[0] == "unknown":
            return {**base, "decided_at": stage, "outcome": "NOT_READ", "flags": [], "shadowed": [], "why": f"stage {stage}: {r[1]} cannot be decided from the readable inputs (an unknown that could win)"}
        return {**base, "decided_at": stage, "outcome": "NO_ROW", "flags": [], "shadowed": [], "why": f"the facts are inconsistent: no row of {r[1]} matches although the arm is not breached (a TOOL defect, not a gap in the prereg)"}
    return {**base, "decided_at": None, "outcome": "NO_ROW", "flags": [], "shadowed": [], "why": "no stage matched (a TOOL defect)"}


def read_constrained(name, arm_dir, ctx, dose):
    """one constrained arm (R0d / R1d) -> its record.  Never raises.  `dose` = load_dose's dict."""
    rec = {"arm": name, "dir": os.path.abspath(arm_dir), "addendum": ADDENDUM_NAME, "addendum_sha256": ADDENDUM_SHA256, "state": None, "S": None, "G": None, "outcome": None, "problems": [], "notes": []}
    try:
        return _read_constrained(rec, name, arm_dir, ctx, dose)
    except Exception as e:                                                                                                   # noqa: BLE001
        rec["outcome"] = "NOT_READ"
        rec["problems"].append(f"the tool could not interpret this arm directory: {type(e).__name__}: {str(e)[:160]}")
        return rec


def _read_constrained(rec, name, arm_dir, ctx, dose):
    pb = rec["problems"]
    if not os.path.isdir(arm_dir):
        rec["state"], rec["outcome"] = "missing", "NOT_READ"
        pb.append(f"no such directory: {arm_dir}")
        return rec
    sft = os.path.join(arm_dir, "sft.jsonl")
    killed_file = os.path.exists(os.path.join(arm_dir, "KILLED.json"))
    has_done = False
    if os.path.exists(sft):
        try:
            has_done = any(r.get("event") == "done" for r in _jsonl(sft)[0])
        except OSError:
            pass
    has_reads = any(os.path.exists(os.path.join(arm_dir, f)) for f in (RLVAL_NAME, "kill_replay.json", S_NAME))
    rec["state"] = "killed" if killed_file else ("complete" if has_done else ("unclear" if has_reads else "incomplete"))
    if rec["state"] == "incomplete":
        rec["outcome"] = "PENDING"
        pb.append("the run has neither a `done` event in sft.jsonl nor a KILLED.json, and the chain has read nothing: it has not finished (PENDING, not a read)")
        return rec
    if rec["state"] == "unclear":
        rec["outcome"] = "NOT_READ"
        pb.append("the chain's reads exist but sft.jsonl has no `done` event and there is no KILLED.json: the run's end is unknown (not PENDING: something has been read)")
        return rec
    # ---- S: fresh read, and the file the chain acted on must be that very record
    S = SR.read_arm(arm_dir)
    sp = os.path.join(arm_dir, S_NAME)
    s_problem = None
    if not os.path.exists(sp):
        s_problem = f"{S_NAME} is missing: run eval/safeobj_read.py first"
    else:
        try:
            fileS = _load(sp)
            ctx.sha(sp)
            body = json.dumps({k: v for k, v in fileS.items() if k != "record_sha256"}, sort_keys=True, separators=(",", ":"))
            if fileS.get("record_sha256") != hashlib.sha256(body.encode("utf-8")).hexdigest():
                s_problem = f"{S_NAME} is not self-consistent (hand-edited?)"
            elif fileS.get("record_sha256") != S["record_sha256"]:
                s_problem = f"{S_NAME} is stale: it is not what safeobj_read.py reads from this directory now"
        except (OSError, ValueError, AttributeError, TypeError):
            s_problem = f"{S_NAME} does not parse"
    rec["S"] = {"verdict": "NOT_READ" if s_problem else S["S"], "reader_verdict": S["S"], "reader_record_sha256": S["record_sha256"], "reader_breaches": S["breaches"], "reader_problems": S["problems"],
                "file": os.path.abspath(sp), "file_sha256": ctx.sha(sp)}
    if s_problem:
        pb.append(s_problem)
        rec["outcome"] = "NOT_READ"
        return rec
    # ---- the kills (addendum R2.2): each judged by its recorded reason
    kills, kill_unknown = read_kills(arm_dir, S, ctx)
    rules = {k["rule"] for k in kills}
    rec["kills"] = kills
    rec["kill_codes"] = sorted({k["code"] for k in kills})
    rec["kill_rules"] = sorted(rules)
    k3_kill = "K3" in rules
    other_kill = True if (rules - {"K3"}) else (None if kill_unknown else False)          # any kill whose recorded rule is not K3
    first_tag, kinds, per_tag = breach_info(arm_dir, S, ctx)
    sv = S["S"]
    s_fails = True if (sv == "FAIL" or k3_kill) else (False if (sv == "PASS" and not kills and not kill_unknown) else None)
    s_pass = True if (sv == "PASS" and not kills and not kill_unknown) else (False if (sv == "FAIL" or kills) else None)
    # "g3 alone" = S fails and every recorded breach at every check is a g3 breach (no g5 breach anywhere)
    if s_fails is True:
        g3_alone = (("g3" in kinds) and ("g5" not in kinds)) if kinds else None
    else:
        g3_alone = False if s_fails is False else None
    # breached = a K3 kill, or S fails with at least one g5 breach
    if k3_kill:
        breached = True
    elif sv == "FAIL":
        breached = ("g5" in kinds) if kinds else None
    elif s_pass:
        breached = False
    else:
        breached = None
    # the first breached check: a K3 kill -> the first of the two consecutive breaches; otherwise the first breached check
    c_tag = (kill_pair_first(per_tag) if k3_kill else first_tag) if breached else None
    c_upd = int(c_tag[len("update"):]) if c_tag else None
    rec["first_breached_check"], rec["breach_kinds"] = c_tag, (sorted(kinds) if kinds is not None else None)
    rec["first_check_with_any_breach"] = first_tag
    if breached and c_tag is None:
        rec["notes"].append("the first breached check could not be located from the replay / monitor rows: the window, the post-ramp share and lambda_max are unreadable")
    # ---- G (needs u1000: a killed arm has none)
    if rec["state"] == "killed":
        rec["G"] = {"verdict": "NOT_READ", "problems": ["the arm was killed before u1000: the B - own-A read at u1000 cannot exist"]}
    else:
        ba, ba_pb = read_ba(arm_dir, ctx)
        rec["G"] = bar_G(ba, ba_pb, dose, ctx)
        if rec["G"].get("pool") and not rec["G"]["pool"]["pool_matches_registered"]:
            rec["notes"].append(f"A-G3b: the B - own-A read covers {rec['G']['pool']['sets_used']} sets / {rec['G']['pool']['n_logs_B']} logs, not the registered RL-val {RLVAL_SETS} / {RLVAL_LOGS}")
    g = rec["G"]
    lb_pos = g.get("lower_bound_gt_0") if g["verdict"] != "NOT_READ" else None
    pt_ok = g.get("point_ge_bar") if g["verdict"] != "NOT_READ" else None
    # ---- the trainer's log
    lg = read_log(sft, c_upd, ctx)
    rec["log"] = {k: lg[k] for k in ("readable", "problems", "notes", "numbers", "lam_zero", "lam_pos_share_ok", "cap_hit", "violation_ok", "window_overlaps_ramp", "columns_read")}
    facts = {"s_pass": s_pass, "g_lb_pos": lb_pos, "g_point_ok": pt_ok, "breached": breached, "violation_ok": lg["violation_ok"], "lam_pos_share_ok": lg["lam_pos_share_ok"], "cap_hit": lg["cap_hit"],
             "lam_zero": lg["lam_zero"], "other_kill": other_kill, "g3_alone": g3_alone}
    rec["facts"] = {k: v for k, v in facts.items()}
    dec = classify(facts)
    rec["table"] = dec
    rec["outcome"] = dec["outcome"]
    rec["outcome_flags"] = dec["flags"]
    if dec["outcome"] == "NO_ROW":
        pb.append(f"NO_ROW: {dec['why']}")
    if dec["outcome"] == "NOT_READ":                                     # say WHY: the unreadable inputs behind the unknown
        pb.extend(f"log: {p}" for p in lg["problems"])
        if dec["decided_at"] == "O123":
            pb.extend(f"G: {p}" for p in g.get("problems", []))
    if "SLACK" in dec["flags"]:
        rec["notes"].append("O4 flagged SLACK (addendum R2.3): held in-sample, lambda > 0 on < 50 % of the post-ramp updates through the first breach, never at lambda_max: the dual was mostly idle")
    if dec["shadowed"]:
        rec["notes"].append(f"rows {dec['shadowed']} also match their own text but lose to the precedence (decided at {dec['decided_at']})")
    return rec


# ============================================================ bar L and the controls
def read_control(name, arm_dir, ctx):
    """point u1000 B - own-A TTC and PDMS (x100) of one arm, from its rlval_vs_A0.json.  -> (info or None, problems)."""
    if not os.path.isdir(arm_dir):
        return None, [f"{name}: no such directory: {arm_dir}"]
    p = os.path.join(arm_dir, RLVAL_NAME)
    if not os.path.exists(p):
        return None, [f"{name}: {RLVAL_NAME} is missing"]
    try:
        r = _load(p)
        ctx.sha(p)
        a = r["arms"]
        ttc = float(a["R"]["subscores_x100"]["TTC"]) - float(a["CTRL"]["subscores_x100"]["TTC"])
        pdms = float(a["R"]["PDMS_x100"]) - float(a["CTRL"]["PDMS_x100"])
        if not (math.isfinite(ttc) and math.isfinite(pdms)):
            return None, [f"{name}: a non-finite number in {RLVAL_NAME}"]
        pb = []
        for f_ in ("scores_update1000_A.npz", "scores_update1000_B.npz"):
            fp = os.path.join(arm_dir, "scores", f_)
            if os.path.exists(fp) and ((r.get("inputs") or {}).get("scores_sha256") or {}).get("CTRL" if f_.endswith("A.npz") else "R") != ctx.sha(fp):
                pb.append(f"{name}: {RLVAL_NAME} read {f_} from a file that is not this arm's own (sha256 differs)")
        if pb:
            return None, pb
        return {"ttc": ctx.N(ttc, p), "pdms": ctx.N(pdms, p)}, []
    except (KeyError, TypeError, ValueError, OSError, AttributeError) as e:
        return None, [f"{name}: {RLVAL_NAME} could not be interpreted: {type(e).__name__}: {str(e)[:100]}"]


def bar_L(arm_dirs, ctx):
    """-> the L record.  PENDING iff R1o was not given.  Needs R0 (banked), R0d, R1d, R1o.  DECIDED ON TTC ONLY (addendum R4 / A-L1): the PDMS line is reported, with its own yardstick."""
    if "R1o" not in arm_dirs:
        return {"verdict": "PENDING", "why": "R1o has not run (not given): the unconstrained seed-1 control of the lever claim does not exist yet"}
    info, pb = {}, []
    for n in ("R0", "R0d", "R1d", "R1o"):
        if n not in arm_dirs:
            pb.append(f"{n}: not given (the banked R0 arm directory is --arm R0=...)" if n == "R0" else f"{n}: not given")
            continue
        i, p = read_control(n, arm_dirs[n], ctx)
        pb += p
        if i is not None:
            info[n] = i
    if pb or len(info) != 4:
        return {"verdict": "NOT_READ", "problems": pb}
    out = {"inputs": {n: {"ttc_B_minus_A_x100": info[n]["ttc"], "pdms_B_minus_A_x100": info[n]["pdms"]} for n in info}, "decided_on": "TTC only (addendum R4 / A-L1)"}
    for q in ("ttc", "pdms"):
        v = {n: info[n][q]["value"] for n in info}
        eff0, eff1 = v["R0d"] - v["R0"], v["R1d"] - v["R1o"]
        f_seed = max(abs(v["R0"] - v["R1o"]), abs(v["R0d"] - v["R1d"]))
        out[q] = {"effect_seed0": eff0, "effect_seed1": eff1, "F_seed": f_seed, "floor_2xF_seed": L_FLOOR_MULT * f_seed,
                  "seed_terms": {"abs_R0d_minus_R1d": abs(v["R0d"] - v["R1d"]), "abs_R0_minus_R1o": abs(v["R0"] - v["R1o"])}}
    t = out["ttc"]
    t["seed0_ok"] = bool(t["effect_seed0"] > 0 and t["effect_seed0"] > L_FLOOR_MULT * t["F_seed"])
    t["seed1_ok"] = bool(t["effect_seed1"] > 0 and t["effect_seed1"] > L_FLOOR_MULT * t["F_seed"])
    out["verdict"] = "PASS" if (t["seed0_ok"] and t["seed1_ok"]) else "FAIL"
    p_ = out["pdms"]
    p_["cost_report_not_a_bar"] = {"seed0_abs_effect_gt_2xF": bool(abs(p_["effect_seed0"]) > p_["floor_2xF_seed"]), "seed1_abs_effect_gt_2xF": bool(abs(p_["effect_seed1"]) > p_["floor_2xF_seed"]),
                                   "yardstick_2xF_seed_pdms": p_["floor_2xF_seed"], "yardstick_2xF_seed_ttc_printed_beside": t["floor_2xF_seed"], "note": "PREREG A-L1: reported, not a bar"}
    return out


# ============================================================ bar E
def _ci(x):
    return isinstance(x, dict) and _num(x.get("mean")) and isinstance(x.get("ci95"), list) and len(x["ci95"]) == 2 and all(_num(v) for v in x["ci95"]) and x["ci95"][0] <= x["ci95"][1]


def rlval_seed_floor(L):
    """the RL-val training-seed floor at E (addendum R4 / A-E3): section 7's two terms on the PDMS B - A contrast, the larger.  -> (floor record or None, why-not)."""
    if not isinstance(L, dict) or "pdms" not in L:
        return None, f"the RL-val seed floor needs R0, R0d, R1d and R1o RL-val reads (bar L is {(L or {}).get('verdict')})"
    p = L["pdms"]
    return {"value": p["F_seed"], "terms": p["seed_terms"]}, None


def bar_E(path, ctx, rlval_floor=None, rlval_floor_why=None):
    """-> the E record.  PENDING iff no navtest file was given.  rlval_floor = {"value", "terms"} from rlval_seed_floor (None = clause 3 cannot be read)."""
    if path is None:
        return {"verdict": "PENDING", "why": "no navtest read exists yet (IMPL step 9, the navtest families adapter, is not built; the T1 reads of the final u1000 B and A of R0d / R1d have not run)"}
    if not os.path.exists(path):
        return {"verdict": "NOT_READ", "problems": [f"--navtest-e {path}: no such file"]}
    try:
        d = _load(path)
        ctx.sha(path)
        pb = []
        if not isinstance(d, dict) or d.get("tier") != "T1":
            pb.append("the navtest file is not an object with tier T1")
        est = str(d.get("estimator", "")) if isinstance(d, dict) else ""
        if f"{N_BOOT} resamples" not in est or f"seed {SEED}" not in est:
            pb.append(f"the estimator is not the registered one ({N_BOOT} resamples, seed {SEED})")
        if not isinstance(d, dict) or d.get("n_tokens") != NAVTEST_TOKENS or d.get("n_logs") != NAVTEST_LOGS:
            pb.append(f"n_tokens / n_logs must be {NAVTEST_TOKENS} / {NAVTEST_LOGS} (the same navtest tokens for both pairs)")
        pairs = d.get("pairs") or {}
        if set(pairs) != set(CONSTRAINED):
            pb.append(f"pairs must be exactly {list(CONSTRAINED)}, got {sorted(pairs)}")
        else:
            for n in CONSTRAINED:
                miss = [s for s in ("PDMS",) + SUBS if not _ci(pairs[n].get(s))]
                if miss:
                    pb.append(f"pair {n}: sub-score(s) {miss} missing or malformed (clause 5: every sub-score reported, EP included)")
        if not _ci((d.get("pooled") or {}).get("PDMS")):
            pb.append("pooled.PDMS missing or malformed")
        elif (d.get("pooled") or {}).get("resample") != "joint":
            pb.append("pooled.resample must be 'joint': the pooled interval resamples the 136 navtest logs JOINTLY for both pairs (addendum R4 / A-E1)")
        elif not pb and abs(d["pooled"]["PDMS"]["mean"] - (pairs["R0d"]["PDMS"]["mean"] + pairs["R1d"]["PDMS"]["mean"]) / 2.0) > E_POOL_TOL:
            pb.append("pooled.PDMS.mean is not the mean of the two pairs' PDMS means (addendum R4: the pooled contrast IS that mean)")
        fl = d.get("floors") or {}
        if not (_num(fl.get("irep_abs_delta")) and fl["irep_abs_delta"] >= 0):
            pb.append("floors.irep_abs_delta missing or negative")
        if pb:
            return {"verdict": "NOT_READ", "problems": pb}
        N = lambda v: ctx.N(v, path)          # noqa: E731
        pd = {n: pairs[n]["PDMS"] for n in CONSTRAINED}
        pooled = d["pooled"]["PDMS"]
        mean_contrast = (pd["R0d"]["mean"] + pd["R1d"]["mean"]) / 2.0
        navtest_seed = abs(pd["R0d"]["mean"] - pd["R1d"]["mean"])
        floor_terms = [navtest_seed, fl["irep_abs_delta"]] + ([rlval_floor["value"]] if rlval_floor is not None else [])
        floor = max(floor_terms) if rlval_floor is not None else None
        c1 = bool(all(pd[n]["mean"] >= E_EACH_MIN for n in CONSTRAINED) and mean_contrast >= E_MEAN_MIN)
        c2 = bool(pooled["ci95"][0] > 0)
        c3 = bool(mean_contrast > E_FLOOR_MULT * floor) if floor is not None else None
        c4 = bool(all(pairs[n][s]["ci95"][1] >= 0 for n in CONSTRAINED for s in ("TTC", "NC")))          # PER PAIR: four upper bounds (addendum R4 / A-E2)
        refuted = bool(pooled["ci95"][1] < E_REFUTED_UPPER)
        adopt = and3(c1, c2, c3, c4)
        if adopt is True and refuted:
            return {"verdict": "NOT_READ", "problems": ["ADOPT and REFUTED both hold: the file is internally inconsistent"]}
        if adopt is True:
            verdict = "ADOPT"
        elif refuted:
            verdict = "REFUTED"
        elif adopt is False:
            verdict = "NOT_PROVEN"
        else:
            verdict = "NOT_READ"
        rec = {"verdict": verdict,
               "clauses": {"1_each_ge_0.25_and_mean_ge_0.5": c1, "2_pooled_lower_gt_0": c2, "3_mean_contrast_gt_2x_max_floor": c3, "4_ttc_nc_upper_ge_0_each_pair": c4, "5_all_subscores_reported": True},
               "numbers": {"R0d_pdms": N(pd["R0d"]["mean"]), "R1d_pdms": N(pd["R1d"]["mean"]), "mean_contrast": N(mean_contrast), "pooled_pdms_lower": N(pooled["ci95"][0]), "pooled_pdms_upper": N(pooled["ci95"][1]),
                           "navtest_seed_floor": N(navtest_seed), "irep_abs_delta": N(fl["irep_abs_delta"])},
               "floor": {"navtest_seed_floor": navtest_seed, "irep_abs_delta": fl["irep_abs_delta"], "rlval_seed_floor": rlval_floor, "max_of_all": floor},
               "refuted_iff_pooled_upper_lt_0.25": refuted, "problems": []}
        if "seed_floor_abs" in fl:
            rec["problems"].append("floors.seed_floor_abs in the file is IGNORED: the tool derives the seed floors (addendum R4 / A-E3)")
        if verdict == "NOT_READ":
            rec["problems"].append(f"clause 3 is unread: {rlval_floor_why or 'the RL-val seed floor is unavailable'} (A-E3b)")
        return rec
    except (KeyError, TypeError, ValueError, OSError) as e:
        return {"verdict": "NOT_READ", "problems": [f"the navtest file could not be interpreted: {type(e).__name__}: {str(e)[:100]}"]}


# ============================================================ the whole read
def run(arm_dirs, navtest_e=None, dose_path=None, prereg_path=None, addendum_path=None):
    """arm_dirs = {name: dir}.  -> the record.  Raises AddendumRefused (before reading anything) when the addendum is not the registered file."""
    add = check_addendum(addendum_path)
    ctx = Ctx()
    ctx.sha(add["path"])
    rec = {"tool": TOOL, "addendum": add["name"], "addendum_sha256": add["sha256_checked"], "addendum_file": add,
           "prereg_sha256_registered": PREREG_SHA256, "prereg_sha256_checked": None, "arms": {}, "L": None, "E": None, "joint": None, "problems": [],
           "prereg_ambiguities": PREREG_AMBIGUITIES, "settled_by_addendum": ADDENDUM_SETTLES,
           "registered": {"G_floor": G_FLOOR, "G_vertices_source": "dose.json (sha256 " + DOSE_SHA256 + ")", "G_points_printed_not_used": [list(p) for p in G_POINTS_PRINTED], "G_lower_bound_min": G_LOWER_BOUND_MIN,
                          "precedence": list(PRECEDENCE), "O4_violation_max": O4_VIOLATION_MAX, "O4_window_updates": O4_WINDOW_UPDATES, "O4_lam_pos_share_min": O4_LAM_POS_SHARE_MIN, "lambda_max": LAM_MAX,
                          "ramp": REGISTERED_RAMP, "L_floor_mult": L_FLOOR_MULT, "E_each_min": E_EACH_MIN, "E_mean_min": E_MEAN_MIN, "E_refuted_upper": E_REFUTED_UPPER, "E_floor_mult": E_FLOOR_MULT,
                          "n_boot": N_BOOT, "seed": SEED}}
    if prereg_path and os.path.exists(prereg_path):
        sha = SR.sha256_file(prereg_path)
        rec["prereg_sha256_checked"] = sha
        if sha != PREREG_SHA256:
            rec["problems"].append(f"the prereg file {prereg_path} has sha256 {sha[:12]}..., not the registered {PREREG_SHA256[:12]}...: this tool's constants may not be the registered text")
    dose = load_dose(dose_path, ctx)
    rec["problems"] += [f"dose: {p}" for p in dose["problems"]]
    for n in CONSTRAINED:
        if n in arm_dirs:
            a = read_constrained(n, arm_dirs[n], ctx, dose)
            rec["arms"][n] = a
            rec["problems"] += [f"{n}: {p}" for p in a["problems"]]
    rec["L"] = bar_L(arm_dirs, ctx)
    floor, why = rlval_seed_floor(rec["L"])
    rec["E"] = bar_E(navtest_e, ctx, floor, why)
    for b in ("L", "E"):
        rec["problems"] += [f"{b}: {p}" for p in rec[b].get("problems", [])]
    # ---- joint: section 6's last paragraph (A-O10, addendum R4)
    o0, o1 = (rec["arms"].get(n, {}).get("outcome") for n in CONSTRAINED)
    if "R0d" not in rec["arms"] or "R1d" not in rec["arms"]:
        j = {"replication": "PENDING", "why": "both constrained arms are needed (R1d runs only after R0d passes S)"}
    elif o0 == "O1" and o1 == "O1":
        j = {"replication": "REPLICATED_O1_BOTH", "why": "R0d and R1d are both O1: Stage 2 proceeds to R1o, bar L, navtest bar E"}
    elif o0 == "O1":
        j = {"replication": "NOT_REPLICATED", "why": f"R0d is O1 and R1d is {o1}: no lever claim, R1o does not run, the next step is O2 (i)"}
    elif o1 == "O1":
        j = {"replication": "NOT_APPLICABLE", "why": f"R0d is {o0}; R1d is O1 (outside the protocol, addendum R4 / A-O10): no lever claim"}
    else:
        j = {"replication": "NOT_APPLICABLE", "why": f"R0d is {o0}, R1d is {o1}"}
    rec["joint"] = j
    rec["inputs_sha256"] = dict(sorted(ctx.inputs.items()))
    body = json.dumps({k: v for k, v in rec.items() if k != "record_sha256"}, sort_keys=True, separators=(",", ":"), default=str)
    rec["record_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return rec


def write_record(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)
    return path


def default_extras():
    pk = os.path.dirname(HERE)
    d = os.path.join(pk, "raw", "2026-10-14-rls1b-attribution", "dose.json")
    p = os.path.join(pk, "raw", "2026-10-17-safety-objective", "PREREG_SAFEOBJ.md")
    a = os.path.join(pk, "raw", "2026-10-17-safety-objective", ADDENDUM_NAME)
    return (d if os.path.exists(d) else None), (p if os.path.exists(p) else None), (a if os.path.exists(a) else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=DIR", help="R0d / R1d (constrained), R1o (control), R0 (banked unconstrained)")
    ap.add_argument("--navtest-e", default=None)
    ap.add_argument("--dose", default=None)
    ap.add_argument("--addendum", default=None)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    dirs = {}
    for spec in a.arm:
        n, _, d = spec.partition("=")
        if n not in ARM_NAMES or not d or n in dirs:
            print(f"usage: --arm NAME=DIR with NAME in {ARM_NAMES}, each at most once (got {spec!r})", flush=True)
            return 3
        dirs[n] = d
    if not any(n in dirs for n in CONSTRAINED):
        print("usage: at least one constrained arm (--arm R0d=... or --arm R1d=...) is required", flush=True)
        return 3
    d0, p0, a0 = default_extras()
    try:
        rec = run(dirs, a.navtest_e, a.dose or d0, a.prereg or p0, a.addendum or a0)
    except AddendumRefused as e:
        print(f"ZZSAFEOBJ_REFUSED {e}", flush=True)
        return 4
    first = dirs.get("R0d") or dirs.get("R1d")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(first)), OUT_NAME)
    write_record(out, rec)
    for n, r in rec["arms"].items():
        why = (r.get("table") or {}).get("why") or "; ".join(r["problems"])[:300]
        flags = f"[{','.join(r['outcome_flags'])}] " if r.get("outcome_flags") else ""
        print(f"ZZSAFEOBJ_OUTCOME {n} {r['outcome']} {flags}{why[:300]}", flush=True)
        if r.get("G"):
            print(f"ZZSAFEOBJ_BAR G[{n}] {r['G']['verdict']} {('margin ' + format(r['G']['margin_point_minus_bar'], '.6f')) if 'margin_point_minus_bar' in r['G'] else '; '.join(r['G'].get('problems', []))[:200]}", flush=True)
        if r.get("S"):
            print(f"ZZSAFEOBJ_BAR S[{n}] {r['S']['verdict']}", flush=True)
    for b in ("L", "E"):
        print(f"ZZSAFEOBJ_BAR {b} {rec[b]['verdict']} {rec[b].get('why') or '; '.join(rec[b].get('problems', []))[:200]}", flush=True)
    print(f"ZZSAFEOBJ_JOINT {rec['joint']['replication']} {rec['joint']['why']}", flush=True)
    print(f"ZZSAFEOBJ_FILE {out} record_sha256={rec['record_sha256']} addendum_sha256={rec['addendum_sha256']}", flush=True)
    done = all(rec["arms"][n]["outcome"] in DETERMINATE for n in rec["arms"])
    return 0 if done else 2


if __name__ == "__main__":
    sys.exit(main())
