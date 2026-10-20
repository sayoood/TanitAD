#!/usr/bin/env python3
"""navsim_bars8 -- the statistics / bars layer of the refcv8 NavSim read (TANITAD VENV, CPU only, no GPU, no download).

Turns a FINISHED ``run_navsim_refcv8.py`` output directory (``scores_navtest/<label>/<label>.csv|.counts.json|_rows.jsonl``,
``bridge_navtest/seam_<arm>.npz|.manifest.json``, ``INDEX.json``) into the registered verdicts of SPEC_REFCV8 6.2 (BAR-R8-N1..N4)
under A-R8-NS1 + AMENDMENT 1 (GOALS_AND_CLAIMS).  ``NAVSIM_BARS_<tag>.json`` + ``.md`` are written; nothing is committed.

    python code/navsim_bars8.py --run-dir <out/step48852_ema> --tag step48852_ema --out-dir <dir> \
        [--navhard-summary summary_navhard.json] [--warmup-summary summary_warmup.json] [--ship-r8-passed 0|1]

WHAT IT DOES NOT DO (and says so in every output): no leaderboard submission, no download, no GPU.

ESTIMATOR (the one the refcv7 NavSim reads used -- IMPORTED, not re-implemented): ``taniteval/adapters/navsim_ci.py``
``log_cluster_bootstrap`` / ``paired_log_cluster_bootstrap`` = ``navsim_log_cluster_bootstrap`` /
``paired_navsim_log_cluster_bootstrap``.  CLUSTER UNIT = the OpenScene LOG (``log_name``; 136 logs on navtest), NEVER the scene
token (consecutive scenes share 13 of 14 frames).  Coarser sensitivity unit ``nuplan_drive`` (44 drives) is computed beside and a
paired result is ``separated`` only when BOTH levels exclude 0 (SPEC 5 conjunction).  B = 2,000, seed 0, percentile 95 %.
It answers ONE question: "would another draw of LOGS give this?"  It is BLIND to training variance (H-ESTIM-SEED-1) and to
inference variance (the planner samples): the inference floor is read from R8_LEGAL vs R8_LEGAL_s1 and a difference below it is
not an effect; training variance is UNTESTED (one training seed), so a separated interval is necessary, never sufficient.

WHAT EACH BAR NEEDS AS INPUT
  N1  navtest, FULL 12,146 tokens: R8_LEGAL (+ R8_LEGAL_s1, the floor) and the banked STOP per-token CSV.
  N4  navtest: R8_LEGAL (+ s1) and the refcv7-50,400 LEGAL per-token CSV (Thor backend: MIXED-BACKEND is stated).
  N2  navhard official two-stage EPDMS: a ``summary_navhard.json`` in the parse7 schema with R8 arm names (arms R8_LEGAL,
      R8_LEGAL_s1, CV_official, STOP_zero, ECHO_ha0_ext; paired_intervals; pairs) -- the two-stage per-key contributions need
      the weight column the CSV drops, so they cannot be rebuilt here.  Absent => UNAVAILABLE, with the reason.
  N3  warmup S2-EPDMS-u: a ``summary_warmup.json`` (same schema, arm R8_BLIND, seed_floor_S2_EPDMS_u).  R8_BLIND MUST FAIL N3 or
      the whole panel is VOID (SPEC 6.2); an absent R8_BLIND leaves N3 UNAVAILABLE, never PASS.
  R8_RT carries a bar ONLY if ``--ship-r8-passed 1`` (SHIP-R8 on the per-frame feed) AND its table is not the clip-fed one;
  R8_PRIV NEVER carries a bar (diagnostic, PRIV - LEGAL with a paired CI).

Evidence class of every number written: MEASURED (the arm CSVs of the run) unless a row says INHERITED (banked floors, the
refcv7 row).  Tier: NavSim open-loop benchmark (T1-family), zero-shot PhysicalAI-AV -> nuPlan cameras, non-parity, NEVER closed loop.

Deliberate-regression hooks (``NAVSIM_BARS8_MUTATION``) exist so ``tests/test_mutation_bars8.py`` can prove the literal tests go
red; ``main`` REFUSES to score while one is set.
"""
from __future__ import annotations

import argparse
import copy
import glob
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MUTATION = os.environ.get("NAVSIM_BARS8_MUTATION", "")
MUTATIONS = ("swap_arm_labels", "drop_seed_floor_gate", "speed_edges_closed_right", "positional_join",
             "rt_bar_without_ship", "priv_carries_bar", "drop_drive_conjunction", "n4_floor_1x", "n2_log_only",
             "n2n3_seed0_only")

# ---- registered constants (SPEC_REFCV8 6.2, A-R8-NS1, AMENDMENT 1) -------------------------------------------------------------
BAR_ARM, FLOOR_ARM, RT_ARM, PRIV_ARM, BLIND_ARM, PRIOR_ARM = "R8_LEGAL", "R8_LEGAL_s1", "R8_RT", "R8_PRIV", "R8_BLIND", "PRIOR_ha0p"
R7_ARM = "REFCV7_50400_LEGAL"            # refcv7-50,400 through the LEGAL inputs (the R7_VMAXOFF arm), BAR-R8-N4's baseline
REF_ARMS = ("STOP", "CV", "HUMAN", R7_ARM)
N_FULL_NAVTEST, N_LOGS_NAVTEST, FINAL_STEP = 12146, 136, 48852
SPEED_EDGES_MPS = (2.0, 8.33, 13.89)     # AMENDMENT 1, literal: stopped/creep | <=30 km/h | 30-50 km/h | >50 km/h
SPEED_BANDS = ("[0,2)", "[2,8.33)", "[8.33,13.89)", ">=13.89")
EP_CLAUSE_M = 5.0                        # the "<= 5 m EP == 1" clause: max compliant progress <= 5 m  =>  EP is set to 1
SG_MAP = "sg-one-north"                  # Singapore (left-hand traffic)
CLIPFED_TABLE_PREFIX = "b208c9c1"        # the clip-fed RETIME table: NEVER admissible on the per-frame feed (AMENDMENT 1 b)
TERM_COLS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "EP": "ego_progress",
             "TTC": "time_to_collision_within_bound", "C": "comfort", "DDC": "driving_direction_compliance", "PDMS": "score"}
METRICS = ("NC", "DAC", "TTC", "EP", "C", "DDC", "PDMS")       # report order; PDMS never alone
SUBS = ("NC", "DAC", "TTC", "EP", "C")                          # DDC enters PDMS v1.1 at weight 0 (MEASURED below)
BANKED_X100 = {"STOP": 61.8202, "CV": 20.6517, "HUMAN": 94.5514, R7_ARM: 70.2048}   # INHERITED (plan 5.2 / registry 15.4.1)
TIER = ("NavSim open-loop benchmark (T1-family), zero-shot PhysicalAI-AV -> nuPlan cameras, non-parity; NEVER closed loop. "
        "NOT leaderboard-submitted (PI 2026-10-04).")
DEFAULT_W3 = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw"
DEFAULT_R7 = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/"
              "step50400/thor_scores/navtest/r7thor_s50400_R7_VMAXOFF/r7thor_s50400_R7_VMAXOFF.csv")
DEFAULT_INPUTS = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
_CENSUS_REL = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/raw/navtest_city_census.json"
CENSUS_CANDIDATES = (f"D:/Projects/TanitAD/{_CENSUS_REL}", f"C:/lgs/Lfb47a58376/{_CENSUS_REL}",
                     f"C:/Users/Admin/qland/landed_hold_r8navsim2/stage/{_CENSUS_REL}")
DEFAULT_CENSUS = ""        # "" = the first existing of CENSUS_CANDIDATES (the file is landed in the repo, raw/navtest_city_census.json)
CI_CANDIDATES = ("C:/lgs/Lfb47a58376/taniteval/adapters/navsim_ci.py", "D:/Projects/TanitAD/taniteval/adapters/navsim_ci.py")
DEFAULT_CFG = {"n_boot": 2000, "seed": 0, "n_full": N_FULL_NAVTEST, "final_step": FINAL_STEP, "ship_r8_passed": False,
               "sub_intervals": True, "strata_pairs": True}


class BarsError(RuntimeError):
    """Refuses rather than guesses; every message names the defect it prevents."""


# ---- the estimator (imported; pinned) --------------------------------------------------------------------------------------------
_CI = None


def load_ci(path: str = ""):
    """Import ``navsim_ci`` (the refcv7 NavSim estimator) and PIN its identity: a silently different estimator would be
    'precise about the wrong thing'."""
    global _CI
    if _CI is not None and not path:
        return _CI
    for p in [path or os.environ.get("NAVSIM_CI_PATH", "")] + list(CI_CANDIDATES):
        if p and os.path.exists(p):
            spec = importlib.util.spec_from_file_location("navsim_ci_bars8", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["navsim_ci_bars8"] = mod
            spec.loader.exec_module(mod)
            pins = (mod.ESTIMATOR, mod.PAIRED_ESTIMATOR, mod.CLUSTER_UNIT, mod.SENSITIVITY_UNIT, mod.N_BOOT, mod.MIN_CLUSTERS)
            if pins != ("navsim_log_cluster_bootstrap", "paired_navsim_log_cluster_bootstrap", "log_name", "nuplan_drive", 2000, 8):
                raise BarsError(f"navsim_ci at {p} is not the refcv7 estimator: {pins}")
            mod.__file_sha256__ = hashlib.sha256(open(p, "rb").read()).hexdigest()
            _CI = mod
            return mod
    raise BarsError("navsim_ci.py (taniteval/adapters) not found; set NAVSIM_CI_PATH")


# ---- strata ----------------------------------------------------------------------------------------------------------------------
def speed_band(v0: float) -> str:
    """AMENDMENT 1 edges, RIGHT-OPEN: [0,2) [2,8.33) [8.33,13.89) >=13.89 m/s (the literal registered numbers)."""
    v = float(v0)
    if not (v == v) or v < 0.0:
        return "invalid"
    closed = MUTATION == "speed_edges_closed_right"
    for edge, name in zip(SPEED_EDGES_MPS, SPEED_BANDS):
        if (v <= edge) if closed else (v < edge):
            return name
    return SPEED_BANDS[-1]


def command_key(onehot) -> str:
    """NavSim one-hot (left, forward, right, unknown) -> key.  Anything that is not a one-hot is 'malformed' (never guessed)."""
    try:
        v = [float(x) for x in onehot]
    except (TypeError, ValueError):
        return "malformed"
    if len(v) != 4 or sorted(v) != [0.0, 0.0, 0.0, 1.0]:
        return "malformed"
    return ("left", "forward", "right", "unknown")[v.index(1.0)]


def command_stratum(onehot) -> str:
    """Registered split: straight (forward) vs turn (left | right).  'unknown' is its own level (0 tokens on navtest, MEASURED)."""
    k = command_key(onehot)
    return {"forward": "straight", "left": "turn", "right": "turn"}.get(k, k)


def ep_clause_fired(max_compliant_progress_m: float) -> bool:
    """The devkit sets EP = 1 when the best compliant progress is <= 5 m (inclusive)."""
    return float(max_compliant_progress_m) <= EP_CLAUSE_M


def city_stratum(map_name: str) -> str:
    return "Singapore" if str(map_name) == SG_MAP else "US"


PARTITIONS = {"city": ("Singapore", "US"), "command": ("straight", "turn", "unknown", "malformed"),
              "t0_speed_band": SPEED_BANDS + ("invalid",), "ep_clause": ("fired", "not_fired")}


def strata_of(tok_info: pd.DataFrame) -> dict:
    """partition -> Series(level) over tok_info's index.  tok_info columns: map_name, command (one-hot), v0, mcp (max compliant m)."""
    out = {"city": tok_info["map_name"].map(city_stratum),
           "command": tok_info["command"].map(command_stratum),
           "t0_speed_band": tok_info["v0"].map(speed_band)}
    out["ep_clause"] = tok_info["mcp"].map(lambda m: "fired" if ep_clause_fired(m) else "not_fired")
    return out


# ---- reading the arms --------------------------------------------------------------------------------------------------------------
def pdms_recompute(df: pd.DataFrame) -> np.ndarray:
    """PDMS v1.1 from the sub-scores, INDEPENDENT of the CSV's ``score`` column: NC * DAC * (5 TTC + 5 EP + 2 C) / 12, DDC at
    weight 0.  MEASURED 2026-10-20 to reproduce ``score`` to <= 3.3e-16 on STOP / CV / HUMAN / refcv7-LEGAL (12,146 rows each)."""
    return (df["NC"].to_numpy(float) * df["DAC"].to_numpy(float)
            * (5.0 * df["TTC"].to_numpy(float) + 5.0 * df["EP"].to_numpy(float) + 2.0 * df["C"].to_numpy(float)) / 12.0)


def read_scores_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "pdm_score" in df.columns:
        raise BarsError(f"{path}: carries pdm_score (a two-stage frame) -- refusing a navtest read of it")
    df = df[df["token"] != "average"].copy()
    if not df["token"].is_unique:
        raise BarsError(f"{path}: duplicate token rows")
    ok = df["valid"].astype(str).isin(("True", "true", "1"))
    df = df[ok].set_index("token")
    out = pd.DataFrame({k: df[c].astype(float) for k, c in TERM_COLS.items()})
    out.attrs["n_invalid_dropped"] = int((~ok).sum())
    return out[list(METRICS)]


def check_counts_guard(csv_path: str, counts: dict, arm: str) -> None:
    """A CSV beside a FAILED count guard is PARTIAL work that reads like a result (MEASURED 2026-09-26: 1,464 of 12,146 rows ->
    'PDMS 64.14').  Assert on the artifact's own guard, never on an exit code."""
    bad = []
    if counts.get("status") != "PASS":
        bad.append(f"status={counts.get('status')!r}")
    for k in ("log_failed", "C3_missing", "C3_unexpected"):
        if counts.get(k, 0) not in (0, None):
            bad.append(f"{k}={counts.get(k)}")
    if bad:
        raise BarsError(f"{arm}: scorer count guard not clean ({', '.join(bad)}) for {csv_path} -- refused, no verdict")


def discover_arms(run_dir: str, split: str = "navtest") -> dict:
    """arm -> paths, found through each ``counts.json``'s own ``arm`` field (SEAM:...seam_<ARM>.npz), not the directory name."""
    found = {}
    for cj in sorted(glob.glob(os.path.join(run_dir, f"scores_{split}", "*", "*.counts.json"))):
        c = json.load(open(cj, encoding="utf-8"))
        m = re.search(r"seam_(.+?)\.npz", str(c.get("arm", "")))
        if not m:
            continue
        arm, stem = m.group(1), cj[: -len(".counts.json")]
        if arm in found:
            raise BarsError(f"two scorer outputs claim arm {arm}: {found[arm]['counts']} and {cj}")
        found[arm] = {"counts": cj, "counts_doc": c, "csv": stem + ".csv", "rows": stem + "_rows.jsonl",
                      "seam": os.path.join(run_dir, f"bridge_{split}", f"seam_{arm}.npz"),
                      "manifest": os.path.join(run_dir, f"bridge_{split}", f"seam_{arm}.manifest.json")}
    return found


def read_rows_jsonl(path: str) -> dict:
    out = {}
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                out[str(r["token"])] = r
    return out


def stop_rate_from_poses(poses: np.ndarray) -> dict:
    """Fraction of plans whose 4 s endpoint is < 1 m from the origin (the gaming loophole BAR-W3-M1 names: a stop-rate column is
    MANDATORY next to every PDMS)."""
    p = np.asarray(poses, np.float64)
    end = np.hypot(p[:, -1, 0], p[:, -1, 1])
    return {"n": int(len(end)), "frac_endpoint_lt_1m": float((end < 1.0).mean()) if len(end) else None,
            "median_4s_distance_m": float(np.median(end)) if len(end) else None}


def stop_rate_seam(npz_path: str, toks) -> dict:
    if not os.path.exists(npz_path):
        return {"status": "UNAVAILABLE", "reason": "seam npz absent"}
    z = np.load(npz_path, allow_pickle=False)
    pos = {str(t): i for i, t in enumerate(z["token"])}
    src = [str(s) for s in z["source"]]
    idx = [pos[t] for t in toks if t in pos and src[pos[t]] != "cv_standin"]
    out = stop_rate_from_poses(np.asarray(z["poses"])[idx])
    out["n_cv_standin_excluded"] = int(sum(1 for t in toks if t in pos and src[pos[t]] == "cv_standin"))
    return out


def load_inputs(path: str) -> pd.DataFrame:
    """token -> log_name, map_name, command (one-hot at t0), v0 = |velocity| at t0 (the last ego status), human / cv plans."""
    doc = json.load(gzip.open(path, "rt", encoding="utf-8"))
    rec = {}
    for t, r in doc["tokens"].items():
        es = r["ego_statuses"][-1]
        v = es["ego_velocity"]
        rec[t] = {"log": r["log_name"], "map_name": r["map_name"], "command": list(es["driving_command"]),
                  "v0": float(math.hypot(float(v[0]), float(v[1]))),
                  "human_end": float(math.hypot(*np.asarray(r["human_future_poses"], float)[-1, :2])),
                  "cv_end": float(math.hypot(*np.asarray(r["cv_poses"], float)[-1, :2]))}
    return pd.DataFrame.from_dict(rec, orient="index")


# ---- estimator wrappers ------------------------------------------------------------------------------------------------------------
def _vec(df: pd.DataFrame, toks, col: str) -> np.ndarray:
    """x100 per-token vector keyed BY TOKEN.  (Mutation: positional -- the join defect that pairs the wrong scenes.)"""
    if MUTATION == "positional_join":
        return 100.0 * df[col].to_numpy(float)[: len(toks)]
    return 100.0 * df.loc[list(toks), col].to_numpy(float)


def interval_single(x100: np.ndarray, clusters, cfg) -> dict:
    ci = load_ci()
    r = ci.log_cluster_bootstrap(x100, clusters, aggregation=ci.AGG_SINGLE_STAGE, n_boot=cfg["n_boot"], seed=cfg["seed"],
                                 official_value=float(np.mean(x100)))
    if r.get("status") != "OK":
        return {"status": "UNAVAILABLE", "reason": r.get("reason"), "n_clusters": r.get("n_clusters")}
    return {"status": "OK", "point": r["point"], "lo": r["lo"], "hi": r["hi"], "n_clusters": r["n_clusters"],
            "drive_sensitivity": {k: v for k, v in r.get("sensitivity_nuplan_drive", {}).items() if k in ("status", "lo", "hi", "n_clusters", "n")}}


def compact_pair(iv: dict) -> dict:
    """Shrink the estimator's paired record; ``positive`` = the interval excludes 0 ABOVE at the log level AND (when the drive
    level can rule) at the drive level -- the SPEC 5 conjunction.  CANNOT-RULE at drive level is stated, never silently passed."""
    if iv.get("status") != "OK":
        return {"status": "UNAVAILABLE", "reason": iv.get("reason"), "n_clusters": iv.get("n_clusters")}
    dr = iv.get("sensitivity_nuplan_drive", {}) or {}
    drive_ok = dr.get("status") == "OK"
    log_pos, log_neg = iv["lo"] > 0, iv["hi"] < 0
    if MUTATION == "drop_drive_conjunction":
        drive_pos, drive_neg = True, True
    else:
        drive_pos = (dr["lo"] > 0) if drive_ok else True
        drive_neg = (dr["hi"] < 0) if drive_ok else True
    return {"status": "OK", "delta": iv["delta"], "lo": iv["lo"], "hi": iv["hi"], "n_clusters": iv["n_clusters"],
            "separated_log_name": bool(iv.get("separated_log_name", log_pos or log_neg)),
            "drive_level": ({"status": "OK", "n_clusters": dr["n_clusters"], "lo": dr["lo"], "hi": dr["hi"]} if drive_ok
                            else {"status": dr.get("status", "CANNOT-RULE"), "reason": dr.get("reason")}),
            "positive": bool(log_pos and drive_pos), "negative": bool(log_neg and drive_neg),
            "separated_scope": iv.get("separated_scope")}


def pair_record(a_df, b_df, toks, clusters, cfg, col: str = "PDMS") -> dict:
    """a - b on the IDENTICAL tokens, the SAME resampled logs every draw (paired); x100 units for PDMS-like columns."""
    ci = load_ci()
    a, b = _vec(a_df, toks, col), _vec(b_df, toks, col)
    dd = a - b
    rec = {"delta_x100": round(float(dd.mean()), 4), "wins": int((dd > 1e-10).sum()), "ties": int((np.abs(dd) <= 1e-10).sum()),
           "losses": int((dd < -1e-10).sum()), "n": int(len(dd))}
    iv = ci.paired_log_cluster_bootstrap(a, b, clusters, aggregation=ci.AGG_SINGLE_STAGE, n_boot=cfg["n_boot"], seed=cfg["seed"],
                                         official_a=float(a.mean()), official_b=float(b.mean()))
    rec["interval"] = compact_pair(iv)
    return rec


def arm_record(df, toks, clusters, cfg) -> dict:
    d = df.loc[list(toks)]
    rec = {m: round(100.0 * float(d[m].mean()), 4) for m in METRICS}
    rec["n"] = len(toks)
    rec["zeroed_by_NC_or_DAC"] = int(((d["NC"] == 0) | (d["DAC"] == 0)).sum())
    rec["pdms_recompute_max_abs"] = float(np.abs(pdms_recompute(d) - d["PDMS"].to_numpy(float)).max())
    cols = (SUBS + ("PDMS",)) if cfg["sub_intervals"] else ("PDMS",)
    rec["intervals_x100"] = {m: interval_single(_vec(df, toks, m), clusters, cfg) for m in cols}
    return rec


# ---- verdict rules (bars7's rule, parameterised: FAILED unless margin > 0 and separated; NOT PROVEN inside 2x the seed floor) ------
def verdict_rule(margin, sep, floor, *, full_ok: bool = True, why_no_sep: str = "") -> str:
    if margin is None or (isinstance(margin, float) and math.isnan(margin)):
        return "UNAVAILABLE"
    if not full_ok:
        return "NOT_EVALUATED -- SUBSET (the bar needs the FULL split)"
    if margin <= 0:
        return "FAILED"
    if sep is None:
        return f"UNAVAILABLE (no admissible paired interval: {why_no_sep})"
    if not sep:
        return "FAILED"
    if MUTATION == "drop_seed_floor_gate":
        return "PASS"
    if floor is None:
        return "NOT PROVEN (inference-seed floor not measured: R8_LEGAL_s1 absent)"
    if margin <= 2.0 * abs(floor):
        return "NOT PROVEN (margin inside 2x the inference-seed floor)"
    return "PASS"


def verdict_rule_n4(margin, sep, floor, *, full_ok: bool = True, why_no_sep: str = "") -> str:
    """N4: delta > 0 and separated, AND the SPEC's general NOT PROVEN rule (SPEC_REFCV8 / SPEC_REFCV7 3): a margin within 2x the
    inference floor reads NOT PROVEN, never PASS.  A-R8-NS1 AMENDMENT 2 (Master Mind, 2026-10-20, before any 48,852 NavSim
    number): N4's own wording carries no multiplier, so the general rule governs; the builder's 1x gate is overruled.  The
    deliberate-regression arm ``n4_floor_1x`` restores the 1x gate and must go red."""
    if margin is None or (isinstance(margin, float) and math.isnan(margin)):
        return "UNAVAILABLE"
    if not full_ok:
        return "NOT_EVALUATED -- SUBSET (the bar needs the FULL split)"
    if margin <= 0:
        return "FAILED"
    if sep is None:
        return f"UNAVAILABLE (no admissible paired interval: {why_no_sep})"
    if not sep:
        return "FAILED"
    if MUTATION == "drop_seed_floor_gate":
        return "PASS"
    if floor is None:
        return "NOT PROVEN (inference-seed floor not measured: R8_LEGAL_s1 absent)"
    mult = 1.0 if MUTATION == "n4_floor_1x" else 2.0
    if margin <= mult * abs(floor):
        return "NOT PROVEN (margin inside 2x the inference-seed floor)"
    return "PASS"


_RANK = {"PASS": 2, "NOT PROVEN": 1, "FAILED": 0}


def _cat(v: str) -> str:
    for k in ("PASS", "NOT PROVEN", "FAILED"):
        if v.startswith(k):
            return k
    return v


def combine_seeds(v0: str, v1: str | None) -> str:
    """SPEC 6.2 header: 'Bars (LEGAL row, both inference seeds)'.  The registered carrying arm is R8_LEGAL (seed 0); the s1 mirror
    must agree, else the bar is NOT PROVEN (never PASS on one lucky draw)."""
    if v1 is None or _cat(v0) not in _RANK or _cat(v1) not in _RANK:
        return v0
    if _cat(v0) == _cat(v1):
        return v0
    return f"NOT PROVEN (inference seeds disagree: seed0={_cat(v0)}, seed1={_cat(v1)})"


def combine_seeds_strict(v0: str, v1: str | None) -> str:
    """N2 / N3 (A-R8-NS1 AMENDMENT 3): as ``combine_seeds``, but a seed-0 PASS whose seed-1 mirror is absent or not
    evaluable reads NOT PROVEN -- 'both inference seeds' (SPEC 6.2) is never satisfied by one draw."""
    if MUTATION == "n2n3_seed0_only":
        return v0
    if _cat(v0) == "PASS" and (v1 is None or _cat(v1) not in _RANK):
        return f"NOT PROVEN (seed-1 mirror not evaluable: {v1 or 'absent'})"
    return combine_seeds(v0, v1)


def _sep(pr: dict | None):
    """tri-state: True / False / None(unavailable)."""
    iv = (pr or {}).get("interval") or {}
    if iv.get("status") != "OK":
        return None, iv.get("reason") or "pair missing"
    return bool(iv["positive"]), ""


# ---- strata ------------------------------------------------------------------------------------------------------------------------
def strata_report(arms: dict, toks, tok_info: pd.DataFrame, clusters, cfg, floor_global) -> dict:
    labs = strata_of(tok_info.loc[list(toks)])
    cl = pd.Series(list(clusters), index=list(toks))
    show = [a for a in (BAR_ARM, FLOOR_ARM, RT_ARM, PRIV_ARM, "STOP", R7_ARM) if a in arms]
    pairs = [(BAR_ARM, "STOP"), (BAR_ARM, R7_ARM), (PRIV_ARM, BAR_ARM), (BAR_ARM, FLOOR_ARM)]
    out = {"partition_sums_ok": True, "partitions": {}}
    for part, levels in PARTITIONS.items():
        blk, tot = {}, 0
        for lv in levels:
            ts = [t for t in toks if labs[part][t] == lv]
            tot += len(ts)
            if not ts:
                blk[lv] = {"n": 0}
                continue
            sub_cl = [cl[t] for t in ts]
            rec = {"n": len(ts), "n_logs": len(set(sub_cl)),
                   "arms_x100": {a: {m: round(100.0 * float(arms[a].loc[ts, m].mean()), 4) for m in METRICS} for a in show},
                   "pairs": {}}
            fl_s = None
            if BAR_ARM in arms and FLOOR_ARM in arms:
                fl_s = abs(float(100.0 * (arms[BAR_ARM].loc[ts, "PDMS"].mean() - arms[FLOOR_ARM].loc[ts, "PDMS"].mean())))
            rec["seed_floor_x100_in_stratum"] = None if fl_s is None else round(fl_s, 4)
            used = None if fl_s is None else max(fl_s, floor_global or 0.0)
            if cfg["strata_pairs"]:
                for x, y in pairs:
                    if x in arms and y in arms:
                        pr = pair_record(arms[x], arms[y], ts, sub_cl, cfg)
                        if used is not None and (x, y) != (BAR_ARM, FLOOR_ARM):
                            pr["floor_used_x100"] = round(used, 4)
                            pr["exceeds_seed_floor"] = bool(abs(pr["delta_x100"]) > used)
                            iv = pr["interval"]
                            # an EFFECT needs both: outside the inference floor AND separated at log and drive level
                            pr["is_effect"] = bool(pr["exceeds_seed_floor"] and iv.get("status") == "OK"
                                                   and (iv["positive"] or iv["negative"]))
                        rec["pairs"][f"{x}__minus__{y}"] = pr
            blk[lv] = rec
        out["partitions"][part] = blk
        if tot != len(list(toks)):
            out["partition_sums_ok"] = False
            out.setdefault("partition_sum_errors", []).append(f"{part}: {tot} != {len(list(toks))}")
    return out


# ---- bars --------------------------------------------------------------------------------------------------------------------------
def eligibility(meta: dict, cfg: dict) -> list:
    why = []
    if "VALIDATION" in str(meta.get("label", "")).upper():
        why.append("the run's own label says VALIDATION ONLY")
    if meta.get("step") != cfg["final_step"]:
        why.append(f"checkpoint step {meta.get('step')} != {cfg['final_step']}")
    if meta.get("weights") != "ema":
        why.append(f"weights {meta.get('weights')!r} != 'ema' (the bars read EMA; raw is beside, PI 2026-10-16)")
    return why


def classify(meta: dict, cfg: dict) -> str:
    """RESULT = EMA step 48,852 and the run says nothing against it.  BESIDE = the final checkpoint's RAW weights (reported, no bar).
    Anything else (an earlier step, a validation label, an unknown step) is VALIDATION ONLY."""
    if "VALIDATION" in str(meta.get("label", "")).upper() or meta.get("step") != cfg["final_step"]:
        return "VALIDATION"
    return "RESULT" if meta.get("weights") == "ema" else "BESIDE"


def gate_text(meta: dict, cfg: dict):
    k = classify(meta, cfg)
    return {"RESULT": None, "VALIDATION": "NOT_EVALUATED -- VALIDATION ONLY",
            "BESIDE": "NOT_EVALUATED -- BESIDE (raw weights; the bars read the EMA weights, PI 2026-10-16)"}[k]


def navtest_bars(arms: dict, toks, pairs: dict, floor, cfg: dict, meta: dict) -> dict:
    """BAR-R8-N1 / N4 (navtest), plus the R8_RT / R8_PRIV rules."""
    full = len(list(toks)) == cfg["n_full"]
    gate = gate_text(meta, cfg)

    def pr(x, y):
        return pairs.get(f"{x}__minus__{y}")

    def one(arm, mirror, base, rule, bar_id, statement, floor_note):
        if arm not in arms or base not in arms:
            return {"bar": bar_id, "statement": statement, "verdict": f"UNAVAILABLE (arm or baseline missing: {arm}, {base})"}
        per, arith = {}, {}
        for a in (arm, mirror):
            if a is None or a not in arms or pr(a, base) is None:
                continue
            p = pr(a, base)
            sep, why = _sep(p)
            margin = p["delta_x100"]
            per[a] = rule(margin, sep, floor, full_ok=full, why_no_sep=why)
            arith[a] = {"PDMS_x100": round(100.0 * float(arms[a].loc[list(toks), "PDMS"].mean()), 4), "margin_x100": margin,
                        "paired_interval": p["interval"], "wins_ties_losses": [p["wins"], p["ties"], p["losses"]]}
        head = per.get(arm, "UNAVAILABLE")
        verdict = combine_seeds(head, per.get(mirror)) if mirror else head
        return {"bar": bar_id, "statement": statement, "baseline": base, "carrying_arm": arm, "mirror_arm": mirror,
                "verdict": gate or verdict, "verdict_if_eligible_ARITHMETIC_ONLY": verdict if gate else None,
                "per_arm_verdict": per, "arithmetic": arith,
                "seed_floor_x100": floor, "seed_floor_rule": floor_note, "n_tokens": len(list(toks)), "full_split": full}

    n1 = one(BAR_ARM, FLOOR_ARM, "STOP", verdict_rule, "BAR-R8-N1",
             "navtest PDMS(R8_LEGAL) > STOP, paired interval excluding 0 (log_name AND nuplan_drive), margin > 2x the full-split "
             "inference floor |PDMS(R8_LEGAL) - PDMS(R8_LEGAL_s1)|", "margin must exceed 2x the floor")
    n4 = one(BAR_ARM, FLOOR_ARM, R7_ARM, verdict_rule_n4, "BAR-R8-N4",
             "navtest PDMS(R8_LEGAL) - PDMS(refcv7-50,400 LEGAL) > 0, separated; margin > 2x the full-split inference floor "
             "(the SPEC's general NOT PROVEN rule, A-R8-NS1 AMENDMENT 2)", "margin must exceed 2x the floor (A-R8-NS1 AMENDMENT 2)")
    if BAR_ARM in arms and pr(BAR_ARM, R7_ARM) and floor is not None:
        n4["margin_exceeds_1x_floor_BESIDE"] = bool(pr(BAR_ARM, R7_ARM)["delta_x100"] > abs(floor))
    n4["caveats"] = ["MIXED-BACKEND: refcv7-50,400 LEGAL was scored on the Thor backend (INHERITED, MODEL_REGISTRY 15.4.1); "
                     "the refcv8 backend is stated in the arm table -- comparable only under RULING_BACKEND_POLICY.",
                     "ONE training seed per model: a separated paired interval is NECESSARY, NOT SUFFICIENT (H-ESTIM-SEED-1); "
                     "the interval answers 'another draw of LOGS?' only."]
    n1["caveats"] = ["ONE training seed: separated is necessary, not sufficient (H-ESTIM-SEED-1)."]
    out = {"BAR-R8-N1": n1, "BAR-R8-N4": n4}
    # ---- R8_RT: a bar only when SHIP-R8 passed on the per-frame feed AND its table is not the clip-fed one
    rt_ok = (cfg["ship_r8_passed"] or MUTATION == "rt_bar_without_ship") and RT_ARM in arms
    sha = str(meta.get("rt_table_sha256") or "")
    if not rt_ok:
        out["R8_RT"] = {"verdict": "NO_BAR (SHIP-R8 has not passed on the per-frame feed; R8_RT is reported beside, "
                                   "D-R8-SHIP-RETIME / D-NS8-3)" if RT_ARM in arms else "NO_BAR (R8_RT not in this run)"}
    elif not sha or sha.startswith(CLIPFED_TABLE_PREFIX):
        out["R8_RT"] = {"verdict": f"REFUSED (R8_RT table sha256 {sha or 'absent'!r}: the clip-fed table b208c9c1... is never "
                                   "admissible; AMENDMENT 1 b needs the PER-FRAME re-fitted table)"}
    else:
        out["R8_RT"] = {"table_sha256": sha,
                        "BAR-R8-N1[R8_RT]": one(RT_ARM, None, "STOP", verdict_rule, "BAR-R8-N1[R8_RT]",
                                                "N1 with R8_RT as the carrying arm", "floor borrowed from R8_LEGAL vs R8_LEGAL_s1 "
                                                "(R8_RT is derived from R8_LEGAL's rows; no R8_RT_s1 exists)"),
                        "BAR-R8-N4[R8_RT]": one(RT_ARM, None, R7_ARM, verdict_rule_n4, "BAR-R8-N4[R8_RT]",
                                                "N4 with R8_RT as the carrying arm", "floor borrowed (see N1[R8_RT])")}
    # ---- R8_PRIV: NEVER a bar
    if MUTATION == "priv_carries_bar" and PRIV_ARM in arms:
        out["R8_PRIV"] = {"BAR-R8-N1[R8_PRIV]": one(PRIV_ARM, None, "STOP", verdict_rule, "BAR-R8-N1[R8_PRIV]", "mutated", "")}
    else:
        out["R8_PRIV"] = {"verdict": "NO_BAR (diagnostic only, never a bar, never submitted)",
                          "PRIV_minus_LEGAL": pr(PRIV_ARM, BAR_ARM)}
    return out


def navhard_bar(summary: dict | None, meta: dict, cfg: dict) -> dict:
    """BAR-R8-N2 from a parse7-schema ``summary_navhard.json`` with R8 arm names (official two-stage EPDMS).

    A-R8-NS1 AMENDMENT 3 (Master Mind, 2026-10-20, before any 48,852 NavSim number): the paired interval vs STOP must exclude 0
    ABOVE at log_name AND nuplan_drive (SPEC_REFCV8 l.651-652 -- the same ``compact_pair`` conjunction N1 uses; the builder's
    log-level-only ``lo > 0`` is overruled), and the bar is read at BOTH inference seeds (SPEC 6.2 header): R8_LEGAL carries it,
    R8_LEGAL_s1 is its mirror with its own paired interval ``R8_LEGAL_s1__minus__STOP_zero``; ``combine_seeds_strict``."""
    stmt = ("official two-stage EPDMS(R8_LEGAL) > max(STOP, CV, ECHO); paired interval vs STOP excludes 0 (log_name AND "
            "nuplan_drive); margin > 2x the seed floor; at both inference seeds (A-R8-NS1 AMENDMENT 3)")
    if not summary:
        return {"bar": "BAR-R8-N2", "statement": stmt, "verdict": "UNAVAILABLE (summary_navhard.json not supplied: needs official "
                "two-stage EPDMS for R8_LEGAL, R8_LEGAL_s1, CV_official, STOP_zero, ECHO_ha0_ext and the paired interval vs STOP_zero)"}
    arms = summary.get("arms", {})
    need = (BAR_ARM, "CV_official", "STOP_zero", "ECHO_ha0_ext")
    o = {k: (arms.get(k) or {}).get("official_two_stage_EPDMS") for k in need}
    if any(not isinstance(v, (int, float)) for v in o.values()):
        return {"bar": "BAR-R8-N2", "statement": stmt, "verdict": f"UNAVAILABLE (official two-stage EPDMS not defined for all of {need}: {o})"}
    floor_ref = max(o["CV_official"], o["STOP_zero"], o["ECHO_ha0_ext"])
    margin = o[BAR_ARM] - floor_ref
    sf = ((summary.get("pairs") or {}).get(f"{BAR_ARM}__minus__{FLOOR_ARM}") or {}).get("official_two_stage_EPDMS_delta")
    pis = summary.get("paired_intervals") or {}
    per, used = {}, {}
    for a in (BAR_ARM, FLOOR_ARM):
        val = (arms.get(a) or {}).get("official_two_stage_EPDMS")
        if not isinstance(val, (int, float)):
            per[a] = f"UNAVAILABLE (official two-stage EPDMS of {a} not defined)"
            continue
        pi = pis.get(f"{a}__minus__STOP_zero") or {}
        if pi.get("status") != "OK":
            sep, why = None, str(pi.get("reason") or pi.get("status") or f"{a}__minus__STOP_zero absent from the summary")
        else:
            cp = compact_pair(pi)
            sep, why = ((cp["lo"] > 0) if MUTATION == "n2_log_only" else cp["positive"]), ""
            used[a] = cp
        per[a] = verdict_rule(val - floor_ref, sep, sf, why_no_sep=why)
    v = combine_seeds_strict(per[BAR_ARM], per.get(FLOOR_ARM))
    gate = gate_text(meta, cfg)
    return {"bar": "BAR-R8-N2", "statement": stmt, "values": o, "margin": margin, "paired_vs_STOP": used.get(BAR_ARM),
            "paired_vs_STOP_s1": used.get(FLOOR_ARM), "per_arm_verdict": per, "seed_floor": sf,
            "verdict": gate or v, "verdict_if_eligible_ARITHMETIC_ONLY": v if gate else None,
            "note": "units as the summary (EPDMS fraction); the interval is the summary's own paired log-cluster record, read "
                    "through compact_pair (log_name AND nuplan_drive)"}


def warmup_bar(summary: dict | None, meta: dict, cfg: dict) -> dict:
    """BAR-R8-N3 + its deliberate-regression control: R8_BLIND must FAIL N3 or the panel is VOID."""
    stmt = "warmup S2-EPDMS-u(R8_LEGAL) > max(CV, STOP, ECHO), margin > 2x the seed floor; R8_BLIND must FAIL it (else the panel is VOID)"
    if not summary:
        return {"bar": "BAR-R8-N3", "statement": stmt, "verdict": "UNAVAILABLE (summary_warmup.json not supplied)"}
    arms = summary.get("arms", {})

    def u(k):
        return ((arms.get(k) or {}).get("S2_EPDMS_u") or {}).get("value")

    need = ("CV_official", "STOP_zero", "ECHO_ha0_ext")
    if any(u(k) is None for k in (BAR_ARM,) + need):
        return {"bar": "BAR-R8-N3", "statement": stmt, "verdict": f"UNAVAILABLE (S2_EPDMS_u missing for {[k for k in (BAR_ARM,) + need if u(k) is None]})"}
    floor_ref = max(u(k) for k in need)
    sf = summary.get("seed_floor_S2_EPDMS_u")
    # A-R8-NS1 AMENDMENT 3: both inference seeds (SPEC 6.2 header); warmup never carries an interval (7 logs < 8)
    v0 = verdict_rule(u(BAR_ARM) - floor_ref, True, sf)
    v1 = (f"UNAVAILABLE (S2_EPDMS_u of {FLOOR_ARM} missing)" if u(FLOOR_ARM) is None
          else verdict_rule(u(FLOOR_ARM) - floor_ref, True, sf))
    v = combine_seeds_strict(v0, v1)
    blind = None if u(BLIND_ARM) is None else verdict_rule(u(BLIND_ARM) - floor_ref, True, sf)
    if blind is None:
        panel = "UNAVAILABLE (control R8_BLIND absent: the panel cannot be certified, so N3 cannot PASS)"
    elif _cat(blind) == "PASS":
        panel = "VOID (R8_BLIND PASSED N3: the panel cannot see the frames, SPEC_REFCV8 6.2)"
    else:
        panel = "OK (R8_BLIND fails N3 as it must)"
    final = v
    if panel.startswith("VOID"):
        final = "VOID (control R8_BLIND passed)"
    elif panel.startswith("UNAVAILABLE") and _cat(v) == "PASS":
        final = "UNAVAILABLE (R8_BLIND control absent)"
    gate = gate_text(meta, cfg)
    return {"bar": "BAR-R8-N3", "statement": stmt, "values": {k: u(k) for k in (BAR_ARM, FLOOR_ARM, BLIND_ARM) + need},
            "per_arm_verdict": {BAR_ARM: v0, FLOOR_ARM: v1},
            "margin": u(BAR_ARM) - floor_ref, "seed_floor": sf, "blind_control_verdict": blind, "panel": panel,
            "interval": "UNAVAILABLE (warmup has 7 logs < 8: it never carries an interval)",
            "verdict": gate or final, "verdict_if_eligible_ARITHMETIC_ONLY": final if gate else None}


# ---- the evaluation of one navtest read --------------------------------------------------------------------------------------------
def evaluate_navtest(arms: dict, tok_info: pd.DataFrame, meta: dict, cfg: dict | None = None, *, extras: dict | None = None,
                     navhard_summary: dict | None = None, warmup_summary: dict | None = None) -> dict:
    """Pure function over in-memory frames (the tests drive it with synthetic rows).  ``arms``: name -> DataFrame (index token,
    columns METRICS).  ``tok_info``: index token, columns log, map_name, command, v0, mcp.  ``meta``: label, step, weights, ..."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    if MUTATION == "swap_arm_labels" and BAR_ARM in arms and "STOP" in arms:
        arms = dict(arms)
        arms[BAR_ARM], arms["STOP"] = arms["STOP"], arms[BAR_ARM]
    ci = load_ci()
    if BAR_ARM not in arms:
        raise BarsError(f"{BAR_ARM} is the bar arm and is absent")
    toks = sorted(arms[BAR_ARM].index)
    usable, refused = {}, {}
    for name, df in arms.items():
        miss = set(toks) - set(df.index)
        if miss:
            refused[name] = f"lacks {len(miss)} of the {len(toks)} bar-arm tokens -- not paired"
        else:
            usable[name] = df
    clusters = [str(tok_info.loc[t, "log"]) for t in toks]
    out = {"tier": TIER, "n_tokens": len(toks), "n_logs": len(set(clusters)),
           "estimator": {"name": ci.ESTIMATOR, "paired": ci.PAIRED_ESTIMATOR, "cluster_unit": ci.CLUSTER_UNIT,
                         "sensitivity_unit": ci.SENSITIVITY_UNIT, "aggregation": ci.AGG_SINGLE_STAGE, "n_boot": cfg["n_boot"],
                         "seed": cfg["seed"], "alpha": 0.05, "module": os.path.abspath(ci.__file__),
                         "module_sha256": getattr(ci, "__file_sha256__", None), "question_answered": ci.QUESTION_ANSWERED,
                         "separated_means": "interval excludes 0 at the log_name level AND at the nuplan_drive level (SPEC 5)"},
           "arms": {}, "refused_pairing": refused, "pairs": {}}
    for name, df in usable.items():
        rec = arm_record(df, toks, clusters, cfg)
        rec["role"] = ("BAR ARM" if name == BAR_ARM else "inference-seed floor arm" if name == FLOOR_ARM else
                       "beside; carries a bar only if SHIP-R8 passed" if name == RT_ARM else
                       "DIAGNOSTIC ONLY (never a bar)" if name == PRIV_ARM else
                       "reference (banked floor, INHERITED)" if name in REF_ARMS else "model-free reference (this run)"
                       if name == PRIOR_ARM else "extra arm (no bar)")
        if extras and name in extras:
            rec["stop_rate"] = extras[name].get("stop_rate")
            rec["backend"] = extras[name].get("backend")
        out["arms"][name] = rec
    model_arms = [a for a in usable if a not in REF_ARMS]
    plan = []
    for x in model_arms:
        for y in REF_ARMS + (PRIOR_ARM,):
            if y in usable and y != x:
                plan.append((x, y))
    for x, y in ((BAR_ARM, FLOOR_ARM), (PRIV_ARM, BAR_ARM), (RT_ARM, BAR_ARM)):
        if x in usable and y in usable and (x, y) not in plan:
            plan.append((x, y))
    for x, y in plan:
        out["pairs"][f"{x}__minus__{y}"] = pair_record(usable[x], usable[y], toks, clusters, cfg)
    floor = None
    sf = out["pairs"].get(f"{BAR_ARM}__minus__{FLOOR_ARM}")
    if sf is not None:
        floor = abs(sf["delta_x100"])
    out["inference_seed_floor"] = {"x100": floor, "pair": f"{BAR_ARM}__minus__{FLOOR_ARM}", "interval": (sf or {}).get("interval"),
                                   "rule": "a difference below this floor is not an effect (one pair of inference draws: a lower "
                                           "bound on the noise, not an estimate of it)"}
    out["strata"] = strata_report(usable, toks, tok_info, clusters, cfg, floor)
    fired = out["strata"]["partitions"]["ep_clause"]["fired"].get("n", 0)
    out["ep_clause_fired"] = {"n": fired, "fraction": round(fired / len(toks), 4),
                              "rule": f"max compliant progress <= {EP_CLAUSE_M} m => EP is set to 1 (the oracle's progress, model-independent)"}
    out["bars"] = navtest_bars(usable, toks, out["pairs"], floor, cfg, meta)
    out["bars"]["BAR-R8-N2"] = navhard_bar(navhard_summary, meta, cfg)
    out["bars"]["BAR-R8-N3"] = warmup_bar(warmup_summary, meta, cfg)
    return out


# ---- loading a real run ------------------------------------------------------------------------------------------------------------
def load_run(run_dir: str, *, inputs: str, floors_dir: str, r7_csv: str, census: str, split: str = "navtest") -> tuple:
    found = discover_arms(run_dir, split)
    if BAR_ARM not in found:
        raise BarsError(f"{BAR_ARM} not found under {run_dir}/scores_{split} (arms found: {sorted(found)})")
    idx = {}
    ip = os.path.join(run_dir, "INDEX.json")
    if os.path.exists(ip):
        idx = json.load(open(ip, encoding="utf-8"))
    meta = {"label": idx.get("label", "UNKNOWN (no INDEX.json)"), "step": idx.get("step"), "weights": idx.get("weights"),
            "ckpt_md5": idx.get("md5"), "run_dir": os.path.abspath(run_dir)}
    arms, extras, controls = {}, {}, {}
    for name, p in found.items():
        check_counts_guard(p["csv"], p["counts_doc"], name)
        df = read_scores_csv(p["csv"])
        if len(df) != int(p["counts_doc"].get("n_expected", len(df))):
            raise BarsError(f"{name}: {len(df)} valid rows != n_expected {p['counts_doc'].get('n_expected')}")
        arms[name] = df
        man = json.load(open(p["manifest"], encoding="utf-8")) if os.path.exists(p["manifest"]) else {}
        extras[name] = {"backend": {"device": man.get("device"), "precision": man.get("precision"), "partial": man.get("partial"),
                                    "gpu_lock_token": man.get("gpu_lock_token")}}
        if name == RT_ARM:
            meta["rt_table_sha256"] = (man.get("retime") or {}).get("table_sha256")
    toks = sorted(arms[BAR_ARM].index)
    for name, p in found.items():
        extras[name]["stop_rate"] = stop_rate_seam(p["seam"], toks)
    ti = load_inputs(inputs)
    missing = [t for t in toks if t not in ti.index]
    if missing:
        raise BarsError(f"{len(missing)} bar-arm tokens absent from the inputs file {inputs}")
    # city census cross-check (two independent derivations of 'Singapore'): inputs map_name vs the log-pickle census
    census = census or next((c for c in CENSUS_CANDIDATES if os.path.exists(c)), "")
    if census and os.path.exists(census):
        cen = json.load(open(census, encoding="utf-8"))
        n_sg = int((ti["map_name"] == SG_MAP).sum())
        controls["city_census"] = {"status": "OK" if (n_sg == cen["tokens_by_city"].get(SG_MAP) and len(ti) == cen["n_tokens"]) else "MISMATCH",
                                   "sg_tokens_inputs": n_sg, "sg_tokens_census": cen["tokens_by_city"].get(SG_MAP),
                                   "n_tokens_inputs": int(len(ti)), "n_tokens_census": cen["n_tokens"]}
        if controls["city_census"]["status"] != "OK":
            raise BarsError(f"city stratum refused: {controls['city_census']}")
    else:
        controls["city_census"] = {"status": "UNCHECKED", "reason": "raw/navtest_city_census.json not found (pass --city-census)"}
    # the EP clause: the oracle's best compliant progress is MODEL-INDEPENDENT -> read it from the bar arm's scorer rows, check
    # it against every other arm's rows and the banked STOP rows
    mcp, src = {}, {}
    rows_by_arm = {n: read_rows_jsonl(p["rows"]) for n, p in found.items()}
    stop_rows = read_rows_jsonl(os.path.join(floors_dir, "STOP_navtest", "STOP_navtest_rows.jsonl"))
    for t in toks:
        r = rows_by_arm[BAR_ARM].get(t) or stop_rows.get(t)
        if r is None:
            raise BarsError(f"token {t}: no max_compliant_progress_m in {BAR_ARM} rows or the STOP rows -- EP-clause stratum undefined")
        mcp[t] = float(r["max_compliant_progress_m"])
    dif = [abs(float(r["max_compliant_progress_m"]) - mcp[t]) for n, rows in rows_by_arm.items() for t, r in rows.items() if t in mcp]
    dif += [abs(float(stop_rows[t]["max_compliant_progress_m"]) - mcp[t]) for t in toks if t in stop_rows]
    controls["ep_clause_source_consistent"] = {"max_abs_diff_m": (max(dif) if dif else None), "n_compared": len(dif),
                                               "status": "OK" if (not dif or max(dif) < 1e-6) else "MISMATCH"}
    v0d = [abs(float(r["v0_mps"]) - float(ti.loc[t, "v0"])) for t, r in rows_by_arm[BAR_ARM].items() if t in ti.index and "v0_mps" in r]
    controls["v0_inputs_vs_scorer_rows"] = {"max_abs_diff_mps": (max(v0d) if v0d else None), "n": len(v0d),
                                            "status": "OK" if (not v0d or max(v0d) < 1e-4) else "MISMATCH"}
    ti = ti.loc[toks].copy()
    ti["mcp"] = pd.Series(mcp)
    # references (banked, INHERITED) restricted to the bar arm's tokens
    for ref in ("STOP", "CV", "HUMAN"):
        pth = os.path.join(floors_dir, f"{ref}_navtest", f"{ref}_navtest.csv")
        arms[ref] = read_scores_csv(pth)
    if r7_csv and os.path.exists(r7_csv):
        arms[R7_ARM] = read_scores_csv(r7_csv)
    extras["STOP"] = {"stop_rate": {"status": "BY CONSTRUCTION", "note": "the zero plan"}}
    for ref, col in (("CV", "cv_end"), ("HUMAN", "human_end")):      # the inputs' own 4 s plans (endpoint distance only)
        extras[ref] = {"stop_rate": stop_rate_from_poses(np.stack([[[0, 0, 0], [ti.loc[t, col], 0, 0]] for t in toks]))}
    # banked-value controls: only meaningful on the full split
    full = len(toks) == N_FULL_NAVTEST
    for ref, banked in BANKED_X100.items():
        if ref in arms:
            got = round(100.0 * float(arms[ref]["PDMS"].mean()), 4) if full else None
            controls[f"banked_{ref}"] = ({"status": "OK" if abs(got - banked) < 6e-5 else "MISMATCH", "read": got, "banked": banked}
                                         if full else {"status": "SKIPPED (subset)", "banked": banked})
    return arms, ti, meta, extras, controls


# ---- rendering ---------------------------------------------------------------------------------------------------------------------
def _f(x, nd=4):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def _iv(rec):
    if not rec or rec.get("status") != "OK":
        n = (rec or {}).get("n_clusters")
        return f"UNAVAILABLE ({n} logs < 8: cannot rule)" if (n is not None and n < 8) else             "UNAVAILABLE" + (f" ({str(rec.get('reason'))[:70]})" if rec and rec.get("reason") else "")
    return f"[{_f(rec['lo'], 2)}, {_f(rec['hi'], 2)}]"


def render_md(res: dict, tag: str) -> str:
    L = []
    lab = res["label"]
    L += [f"# NAVSIM_BARS_{tag}", "", f"**{lab}**", "",
          f"Tier: {res['tier']}", f"Run: `{res['meta'].get('run_dir')}` step {res['meta'].get('step')} weights {res['meta'].get('weights')}; "
          f"n = {res['n_tokens']} tokens / {res['n_logs']} logs.", ""]
    if res.get("validation_only"):
        L += [f"> **VALIDATION ONLY -- n = {res['n_tokens']}, NO VERDICT.** " + "; ".join(res["ineligible_because"]) +
              ". Numbers exercise the path; they are never refcv8's NavSim performance.", ""]
    e = res["estimator"]
    L += ["## Estimator", f"`{e['name']}` / `{e['paired']}` (IMPORTED from `{e['module']}`, sha256 `{str(e['module_sha256'])[:16]}`); "
          f"cluster unit **{e['cluster_unit']}** (sensitivity unit {e['sensitivity_unit']}), B = {e['n_boot']}, seed {e['seed']}, "
          f"percentile 95 %. Answers: *another draw of LOGS?* only. Blind to training variance (one training seed) and to inference "
          f"variance, which is read from R8_LEGAL vs R8_LEGAL_s1 (below). `separated` = {e['separated_means']}.", ""]
    L += ["## Per-arm (x100; NC / DAC / TTC / EP / C and PDMS -- never PDMS alone)", "",
          "| arm | role | n | NC | DAC | TTC | EP | C | DDC | PDMS | PDMS 95 % (log bootstrap) | zeroed NC/DAC | stop-rate |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in res["arms"].items():
        sr = (r.get("stop_rate") or {}).get("frac_endpoint_lt_1m")
        L.append(f"| {n} | {r['role']} | {r['n']} | " + " | ".join(_f(r[m], 2) for m in METRICS[:-1]) + f" | **{_f(r['PDMS'])}** | "
                 f"{_iv(r['intervals_x100'].get('PDMS'))} | {r['zeroed_by_NC_or_DAC']} | {_f(sr, 3) if sr is not None else (r.get('stop_rate') or {}).get('status', '-')} |")
    L += ["", "## Paired differences (a - b, x100 PDMS; same tokens, same resampled logs)", "",
          "| pair | delta | 95 % | log-level | drive-level | positive (both) | W/T/L |", "|---|---|---|---|---|---|---|"]
    for k, p in res["pairs"].items():
        iv = p["interval"]
        dl = iv.get("drive_level", {}) if iv.get("status") == "OK" else {}
        L.append(f"| {k} | {_f(p['delta_x100'])} | {_iv(iv)} | {'sep' if iv.get('separated_log_name') else 'no'} | "
                 f"{_iv(dl) if dl.get('status') == 'OK' else dl.get('status', '-')} | {iv.get('positive', '-')} | {p['wins']}/{p['ties']}/{p['losses']} |")
    sf = res["inference_seed_floor"]
    L += ["", f"**Inference-seed floor** (R8_LEGAL vs R8_LEGAL_s1) = {_f(sf['x100'])} PDMS x100. {sf['rule']}.", ""]
    L += ["## Bars", "", "| bar | verdict | margin (x100) | floor | note |", "|---|---|---|---|---|"]
    for k in ("BAR-R8-N1", "BAR-R8-N2", "BAR-R8-N3", "BAR-R8-N4"):
        b = res["bars"][k]
        m = (b.get("arithmetic", {}).get(BAR_ARM, {}) or {}).get("margin_x100", b.get("margin"))
        L.append(f"| {k} | **{b['verdict']}** | {_f(m)} | {_f(b.get('seed_floor_x100', b.get('seed_floor')))} | "
                 f"{b.get('panel', '') or ('; '.join(b.get('caveats', [])))[:160]} |")
    L += ["", f"R8_RT: {res['bars']['R8_RT'].get('verdict', 'carries a bar (see JSON)')}", f"R8_PRIV: {res['bars']['R8_PRIV'].get('verdict')}", ""]
    L += ["## Strata (PDMS x100; delta vs STOP / refcv7-LEGAL; bold = an EFFECT: outside the inference floor AND separated at log and drive level)", ""]
    for part, blk in res["strata"]["partitions"].items():
        L += [f"### {part}", "", "| level | n | logs | R8_LEGAL PDMS | STOP | " + R7_ARM + " | LEGAL-STOP | LEGAL-R7 | PRIV-LEGAL | s0-s1 floor |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for lv, r in blk.items():
            if not r.get("n"):
                L.append(f"| {lv} | 0 | - | - | - | - | - | - | - | - |")
                continue
            a = r["arms_x100"]
            def d(k):
                p = r["pairs"].get(k)
                if not p:
                    return "-"
                s = _f(p["delta_x100"], 2) + (" " + _iv(p["interval"]) if p["interval"].get("status") == "OK" else " (no interval)")
                return f"**{s}**" if p.get("is_effect") else s
            L.append(f"| {lv} | {r['n']} | {r['n_logs']} | {_f(a.get(BAR_ARM, {}).get('PDMS'), 2)} | {_f(a.get('STOP', {}).get('PDMS'), 2)} | "
                     f"{_f(a.get(R7_ARM, {}).get('PDMS'), 2)} | {d(f'{BAR_ARM}__minus__STOP')} | {d(f'{BAR_ARM}__minus__{R7_ARM}')} | "
                     f"{d(f'{PRIV_ARM}__minus__{BAR_ARM}')} | {_f(r.get('seed_floor_x100_in_stratum'), 2)} |")
        L.append("")
    L += ["## Controls", "```", json.dumps(res["controls"], indent=1, default=str), "```", "",
          "## Inputs each bar needs", "- N1: R8_LEGAL + R8_LEGAL_s1 + banked STOP csv, FULL 12,146 tokens, EMA step 48,852.",
          "- N4: as N1 + the refcv7-50,400 LEGAL csv (Thor backend: MIXED-BACKEND stated).",
          "- N2: `summary_navhard.json` (parse7 schema, R8 arm names, official two-stage EPDMS) -- not rebuilt here (the CSV drops the stage-2 weights).",
          "- N3: `summary_warmup.json` incl. R8_BLIND (must FAIL, else the panel is VOID).",
          "- R8_RT: `--ship-r8-passed 1` AND the per-frame table (never b208c9c1...). R8_PRIV: never a bar.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--inputs", default=DEFAULT_INPUTS)
    ap.add_argument("--floors-dir", default=DEFAULT_W3)
    ap.add_argument("--r7-legal-csv", default=DEFAULT_R7)
    ap.add_argument("--city-census", default=DEFAULT_CENSUS)
    ap.add_argument("--navhard-summary", default="")
    ap.add_argument("--warmup-summary", default="")
    ap.add_argument("--ship-r8-passed", type=int, default=0, choices=(0, 1))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sub-intervals", action="store_true")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if MUTATION:
        raise SystemExit(f"NAVSIM_BARS8_MUTATION={MUTATION!r} is set: a deliberate-regression environment never scores")
    cfg = {**DEFAULT_CFG, "n_boot": a.n_boot, "seed": a.seed, "ship_r8_passed": bool(a.ship_r8_passed), "sub_intervals": not a.no_sub_intervals}
    arms, ti, meta, extras, controls = load_run(a.run_dir, inputs=a.inputs, floors_dir=a.floors_dir, r7_csv=a.r7_legal_csv, census=a.city_census)
    nh = json.load(open(a.navhard_summary, encoding="utf-8")) if a.navhard_summary else None
    wu = json.load(open(a.warmup_summary, encoding="utf-8")) if a.warmup_summary else None
    res = evaluate_navtest(arms, ti, meta, cfg, extras=extras, navhard_summary=nh, warmup_summary=wu)
    why, kind = eligibility(meta, cfg), classify(meta, cfg)
    res["meta"], res["controls"] = meta, controls
    res["validation_only"] = kind == "VALIDATION"
    res["ineligible_because"] = why
    res["label"] = {"VALIDATION": "VALIDATION ONLY (not refcv8's NavSim score): " + "; ".join(why),
                    "BESIDE": "BESIDE (raw weights of the final checkpoint; the bars read EMA): no bar verdict",
                    "RESULT": "RESULT (EMA 48,852, the bar read)"}[kind]
    res["ship_r8_passed_flag"] = bool(a.ship_r8_passed)
    for k, v in controls.items():
        if isinstance(v, dict) and v.get("status") == "MISMATCH":
            res["label"] += f"  [CONTROL MISMATCH: {k}]"
    os.makedirs(a.out_dir, exist_ok=True)
    jp, mp = (os.path.join(a.out_dir, f"NAVSIM_BARS_{a.tag}.{e}") for e in ("json", "md"))
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=1, default=str)
    open(mp, "w", encoding="utf-8").write(render_md(res, a.tag))
    print(res["label"])
    for k in ("BAR-R8-N1", "BAR-R8-N2", "BAR-R8-N3", "BAR-R8-N4"):
        print(f"{k}: {res['bars'][k]['verdict']}")
    print(jp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
