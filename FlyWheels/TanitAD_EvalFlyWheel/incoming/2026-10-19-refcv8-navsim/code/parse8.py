#!/usr/bin/env python3
"""parse8 -- official NavSim CSVs of a refcv8 run -> ``summary_navhard.json`` / ``summary_warmup.json`` (TANITAD VENV, CPU only).

PORT of ``FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/code/parse7.py``.  The STATISTICS ARE
UNCHANGED (same official ``score`` column, same ``s2_group_uniform`` imported from E2, same ``navsim_ci`` log-cluster bootstrap
with B = 2,000 / seed 0 / 95 % percentile, clusters = the mapping key's ``log_name``, same paired variant, same stop fraction,
same seed-floor rule).  What changed is ONLY (1) the arm names, (2) the paths, (3) fail-loud handling of a missing arm, (4) the
consumer contract.  The consumer is ``navsim_bars8.navhard_bar`` (BAR-R8-N2) and ``navsim_bars8.warmup_bar`` (BAR-R8-N3).

    python code/parse8.py --split navhard_two_stage --scores <run>/scores_navhard --bridge <run>/bridge_navhard \\
        --inputs <navhard navsim_agent_inputs.json> --label RESULT --csv-suffix __navhard_two_stage --out summary_navhard.json
    python code/parse8.py --split warmup_two_stage  --scores <run>/scores_warmup  --bridge <run>/bridge_warmup \\
        --inputs <warmup navsim_agent_inputs.json> --label RESULT --out summary_warmup.json

ARM-NAME MAP  R7 -> R8  (parse7 -> parse8; the on-disk file stem a refcv8 run writes is the R8 name):

    R7_A1         -> R8_LEGAL       the BAR arm (LEGAL inputs, inference seed 0)
    R7_A1_s1      -> R8_LEGAL_s1    the inference-seed floor arm (seed 1)
    R7_BLIND      -> R8_BLIND       warmup deliberate regression; MUST FAIL N3 or the panel is VOID
    R7_A1NT       -> R8_NT          optional (not a default arm)
    R7_NAVOFF     -> R8_NAVOFF      optional (not a default arm)
    (new)            R8_PRIV        optional DIAGNOSTIC (navtest default; never a bar)
    (new)            R8_RT          optional, derived from R8_LEGAL's rows only with --retime-table
    PRIOR_ha0p    -> PRIOR_ha0p     unchanged (derived, model-free)
    CV_official, STOP_zero, ECHO_ha0_ext   unchanged (model-free floors, read from ``--floors``)
    DROPPED       R7_VMAXOFF, R7_FILTOFF, R7_CEILDECL_d   (R7-recipe arms; refcv8 has no max-speed input or ceiling filter to ablate)

OUTPUT CONTRACT (what navsim_bars8 reads, nothing else is load-bearing):
    navhard: arms[R8_LEGAL|CV_official|STOP_zero|ECHO_ha0_ext].official_two_stage_EPDMS (float);
             paired_intervals["R8_LEGAL__minus__STOP_zero"] {status, lo, ...};
             pairs["R8_LEGAL__minus__R8_LEGAL_s1"].official_two_stage_EPDMS_delta  (the inference-seed floor);
             A-R8-NS1 AMENDMENT 3 (bars hold on BOTH inference seeds): the seed-1 mirror rows
             pairs / paired_intervals["R8_LEGAL_s1__minus__STOP_zero" | "__minus__CV_official" | "__minus__ECHO_ha0_ext"], same
             function and record shape as the R8_LEGAL rows (``separated``, ``separated_log_name``, ``sensitivity_nuplan_drive``);
             UNAVAILABLE with a reason when R8_LEGAL_s1 is absent.  On warmup (7 logs < 8) the mirror rows are ``pairs`` only --
             point deltas and W/T/L, no interval, exactly like the R8_LEGAL rows there.
    warmup:  arms[R8_LEGAL|R8_BLIND|CV_official|STOP_zero|ECHO_ha0_ext].S2_EPDMS_u.value (float);
             seed_floor_S2_EPDMS_u (float, or null = UNAVAILABLE with ``seed_floor_S2_EPDMS_u_status``)

FAIL LOUD: an EXPECTED arm with no score CSV (or a CSV beside a failed count guard) reads
``{"status": "UNAVAILABLE", "reason": ...}`` -- never a fabricated value, never a silent omission -- and every pair / interval
that needs it reads UNAVAILABLE too.  parse7 silently skipped such arms.

WHAT THIS DOES NOT FIX (inherited from parse7, stated so nobody reads more into it): the navhard interval answers ONE question,
"would another draw of LOGS give this?"; it is BLIND to training variance (H-ESTIM-SEED-1) and to inference variance (the seed
floor is read separately from R8_LEGAL vs R8_LEGAL_s1).  The warmup split has 7 logs < 8, so it NEVER carries an interval.

Deliberate-regression hooks (``PARSE8_MUTATION``) exist so ``tests/test_mutation_parse8.py`` can prove the literal tests go red;
``main`` REFUSES to write while one is set.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
MUTATION = os.environ.get("PARSE8_MUTATION", "")
MUTATIONS = ("swap_seed_arms", "drop_blind_arm", "skip_count_guard", "silent_skip_missing", "ignore_cv_standin",
             "s1_pair_reuses_s0")

SUB = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
       "DDC": "driving_direction_compliance", "TLC": "traffic_light_compliance",
       "EP": "ego_progress", "TTC": "time_to_collision_within_bound",
       "LK": "lane_keeping", "HC": "history_comfort", "EC": "two_frame_extended_comfort"}
BAR_ARM, FLOOR_ARM, BLIND_ARM, PRIOR_ARM = "R8_LEGAL", "R8_LEGAL_s1", "R8_BLIND", "PRIOR_ha0p"
FLOORS = ("CV_official", "STOP_zero", "ECHO_ha0_ext")
MODEL_ARMS = ("R8_LEGAL", "R8_LEGAL_s1", "R8_BLIND", "R8_NT", "R8_NAVOFF", "R8_PRIV", "R8_RT", "PRIOR_ha0p")
#: arms a refcv8 run MUST have produced for each split (A-R8-NS1: EMA = R8_LEGAL + R8_LEGAL_s1 on every split, R8_BLIND on
#: warmup; PRIOR_ha0p is always derived by run_navsim_refcv8.py).  Anything else is optional and is listed under arms_not_run.
SPLIT_EXPECTED = {
    "navhard_two_stage": ("R8_LEGAL", "R8_LEGAL_s1", "PRIOR_ha0p") + FLOORS,
    "warmup_two_stage": ("R8_LEGAL", "R8_LEGAL_s1", "R8_BLIND", "PRIOR_ha0p") + FLOORS,
}
PAIRS = [("R8_LEGAL", "CV_official", "LEGAL vs CV"),
         ("R8_LEGAL", "STOP_zero", "LEGAL vs STOP (the real floor)"),
         ("R8_LEGAL", "ECHO_ha0_ext", "LEGAL vs the echo of its own ego inputs"),
         ("R8_LEGAL", "R8_LEGAL_s1", "SEED FLOOR: LEGAL vs the same arm at inference seed 1"),
         # SPEC_REFCV8 6.2 / A-R8-NS1 AMENDMENT 3: the bars hold on BOTH inference seeds -> the seed-1 mirror of the three LEGAL rows,
         # through the SAME code path (pair record, paired log-cluster bootstrap incl. separated / separated_log_name / nuplan_drive).
         ("R8_LEGAL_s1", "CV_official", "seed-1 mirror: LEGAL_s1 vs CV"),
         ("R8_LEGAL_s1", "STOP_zero", "seed-1 mirror: LEGAL_s1 vs STOP (the real floor)"),
         ("R8_LEGAL_s1", "ECHO_ha0_ext", "seed-1 mirror: LEGAL_s1 vs the echo of its own ego inputs"),
         ("R8_LEGAL", "R8_BLIND", "vision contribution (LEGAL - frames-blind)"),
         ("R8_LEGAL", "R8_NAVOFF", "nav contribution (LEGAL - nav withheld)"),
         ("R8_LEGAL", "R8_NT", "time construction (ST - NT)"),
         ("R8_PRIV", "R8_LEGAL", "DIAGNOSTIC (never a bar): PRIV - LEGAL, the price of the privileged speed input"),
         ("R8_RT", "R8_LEGAL", "R8_RT - LEGAL: the re-timing stage on LEGAL's own rows"),
         ("R8_LEGAL", "PRIOR_ha0p", "the residual: LEGAL - its own prior P alone"),
         ("PRIOR_ha0p", "STOP_zero", "the prior vs doing nothing"),
         ("PRIOR_ha0p", "CV_official", "the prior vs CV"),
         ("PRIOR_ha0p", "ECHO_ha0_ext", "the prior (pose-derived) vs the echo (EgoStatus a, ay)"),
         ("STOP_zero", "CV_official", "STOP vs CV (floor ordering)")]
#: the R7 -> R8 on-disk names, used ONLY by the control (``--control-r7-names``): logical R8 arm -> the R7 file stem on disk.
CONTROL_R7_ALIAS = {"R8_LEGAL": "R7_A1", "R8_LEGAL_s1": "R7_A1_s1", "R8_BLIND": "R7_BLIND", "R8_NT": "R7_A1NT",
                    "R8_NAVOFF": "R7_NAVOFF"}

_INC = "FlyWheels/TanitAD_EvalFlyWheel/incoming"
_ROOTS = ("C:/lgs/Lfb47a58376", "D:/Projects/TanitAD", "E:/Projects/TanitAD")
E2_CANDIDATES = tuple(f"{r}/{_INC}/2026-09-19-navsim-refcv4b-bridge" for r in _ROOTS)
CI_CANDIDATES = ("C:/lgs/Lfb47a58376/taniteval/adapters/navsim_ci.py", "D:/Projects/TanitAD/taniteval/adapters/navsim_ci.py")
FLOORS_CANDIDATES = tuple(f"{r}/{_INC}/2026-09-28-refcv7-standard-tests/navsim/raw/floors" for r in _ROOTS)


class ParseError(RuntimeError):
    """Refuses rather than guesses; every message names the defect it prevents."""


def unavailable(reason: str, **extra) -> dict:
    return {"status": "UNAVAILABLE", "reason": reason, **extra}


# ---- module loading (E2's s2_group_uniform and the pinned navsim_ci estimator) ------------------------------------------------------
def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_e2(path: str = ""):
    for d in ([path] if path else []) + [os.environ.get("PARSE8_E2_DIR", "")] + list(E2_CANDIDATES) + \
            [os.path.join(PKG, "..", "2026-09-19-navsim-refcv4b-bridge")]:
        p = os.path.join(d, "code", "parse_scores.py") if d else ""
        if p and os.path.exists(p):
            return _load_mod("e2_parse_scores8", p)
    raise ParseError("E2 parse_scores.py (2026-09-19-navsim-refcv4b-bridge/code) not found; set PARSE8_E2_DIR")


def load_ci(path: str = ""):
    """Import ``navsim_ci`` (the refcv7 NavSim estimator) and PIN its identity (the same pins navsim_bars8 asserts): a silently
    different estimator would be 'precise about the wrong thing'."""
    for p in [path or os.environ.get("PARSE8_NAVSIM_CI", "")] + list(CI_CANDIDATES):
        if p and os.path.exists(p):
            mod = _load_mod("navsim_ci_parse8", p)
            pins = (mod.ESTIMATOR, mod.PAIRED_ESTIMATOR, mod.CLUSTER_UNIT, mod.SENSITIVITY_UNIT, mod.N_BOOT, mod.MIN_CLUSTERS)
            if pins != ("navsim_log_cluster_bootstrap", "paired_navsim_log_cluster_bootstrap", "log_name", "nuplan_drive", 2000, 8):
                raise ParseError(f"navsim_ci at {p} is not the refcv7 estimator: {pins}")
            mod.__file_sha256__ = hashlib.sha256(open(p, "rb").read()).hexdigest()
            return mod
    raise ParseError("navsim_ci.py (taniteval/adapters) not found; set PARSE8_NAVSIM_CI")


# ---- parse7's readers (unchanged) ---------------------------------------------------------------------------------------------------
def load_csv(path: str, stage_of: dict, arm: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "pdm_score" in df.columns:
        raise SystemExit(f"{path}: carries pdm_score — refusing (read `score`)")
    tok = df[~df.token.str.startswith("extended_pdm_score")].copy()
    tok["stage"] = tok.token.map(stage_of)
    if tok.stage.isna().any():
        raise SystemExit(f"{arm}: {int(tok.stage.isna().sum())} tokens not in the export")
    for k, c in SUB.items():
        tok[k] = np.where(tok.stage == 1, tok.get(f"{c}_stage_one"), tok.get(f"{c}_stage_two"))
    return tok


def official_rows(path: str) -> dict:
    df = pd.read_csv(path)
    s = df[df.token.str.startswith("extended_pdm_score")].set_index("token")
    return {k: float(s.loc[k, "score"]) for k in s.index}


def stop_fraction(seam_path: str, stage_of: dict) -> dict:
    z = np.load(seam_path, allow_pickle=False)
    src = [str(x) for x in z["source"]]
    toks = [str(x) for x in z["token"]]
    p = np.asarray(z["poses"], dtype=np.float64)
    keep = [i for i, (t, s) in enumerate(zip(toks, src)) if s != "cv_standin" and stage_of[t] == 2]
    if not keep:
        return {"status": "UNAVAILABLE", "reason": "no own stage-2 rows"}
    end = np.hypot(p[keep, -1, 0], p[keep, -1, 1])
    return {"n": len(keep), "frac_endpoint_lt_1m": float((end < 1.0).mean()),
            "median_4s_distance_m": float(np.median(end))}


def _jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _disk(arm: str, alias: dict | None) -> str:
    """The on-disk file stem of a logical R8 arm.  ``swap_seed_arms`` (mutation) trades the two seed arms' files."""
    a = arm
    if MUTATION == "swap_seed_arms":
        a = {BAR_ARM: FLOOR_ARM, FLOOR_ARM: BAR_ARM}.get(a, a)
    return (alias or {}).get(a, a)


def _src(x: str, y: str) -> str:
    """The arm whose ROWS feed the pair (x, y).  ``s1_pair_reuses_s0`` (mutation) silently feeds the seed-1 mirror rows from seed 0."""
    if MUTATION == "s1_pair_reuses_s0" and x == FLOOR_ARM and y in FLOORS:
        return BAR_ARM
    return x


def _model_info(bridge: str, alias: dict | None):
    """(step, ckpt_md5) read from the BAR arm's seam manifest; (None, None) if absent."""
    try:
        m = _jload(os.path.join(bridge, f"seam_{_disk(BAR_ARM, alias)}.manifest.json"))
        mod = m.get("model") or {}
        return mod.get("step"), mod.get("ckpt_md5")
    except Exception:                                                     # noqa: BLE001
        return None, None


def stamp(step, md5) -> str:
    """No recipe text is restated here (refcv7's stamp described refcv7; inventing refcv8's would be INHERITED-unverified)."""
    try:
        int(step)
    except (TypeError, ValueError):
        return "UNKNOWN step -- the BAR arm's seam manifest was not readable"
    return (f"model as trained: step={int(step)} ckpt_md5={md5} (read from the seam manifest; arm names are refcv8's; "
            f"the recipe is NOT restated here -- MODEL_REGISTRY.md)")


def find_floors(split: str, csv_suffix: str, given: str = "") -> str:
    if given:
        return given
    for root in (os.path.join(PKG, "raw", "floors"),) + FLOORS_CANDIDATES:
        d = os.path.join(root, split)
        if os.path.exists(os.path.join(d, f"score_STOP_zero{csv_suffix}.csv")):
            return d
    return ""


# ---- the summary --------------------------------------------------------------------------------------------------------------------
def build_summary(split: str, scores: str, bridge: str, inputs: str, label: str, *, floors: str = "", csv_suffix: str = "",
                  alias: dict | None = None, ps=None, ci=None) -> dict:
    if split not in SPLIT_EXPECTED:
        raise ParseError(f"unknown split {split!r}; expected one of {sorted(SPLIT_EXPECTED)}")
    expected = SPLIT_EXPECTED[split]
    ps = ps or load_e2()
    doc = _jload(inputs)
    stage_of = {t: r["stage"] for t, r in doc["tokens"].items()}
    log_of = {t: r["log_name"] for t, r in doc["tokens"].items()}
    mapping = doc["reactive_all_mapping"]
    step, md5 = _model_info(bridge, alias)
    floor_dir = floors or scores
    out = {"_label": label, "split": split, "model_as_trained": stamp(step, md5), "checkpoint_step": step,
           "_primary": "S2-EPDMS-u (E2 SPEC §2) + the official two-stage EPDMS where the arm's stage-1 rows are its own",
           "arms": {}, "pairs": {}, "arms_refused": {}, "arms_not_run": []}
    out["_parse8"] = {"port_of": "parse7.py", "expected_arms": list(expected), "alias": dict(alias or {}),
                      "scores_dir": scores, "floors_dir": floor_dir, "bridge_dir": bridge, "inputs": inputs,
                      "csv_suffix": csv_suffix, "mutation": MUTATION or None}
    fp = os.path.join(floor_dir, "FLOORS_PROVENANCE.json")
    if os.path.exists(fp):
        out["_parse8"]["floors_provenance_sha256"] = hashlib.sha256(open(fp, "rb").read()).hexdigest()
    frames = {}
    for arm in MODEL_ARMS + FLOORS:
        is_model = arm in MODEL_ARMS
        d = scores if is_model else floor_dir
        disk = _disk(arm, alias)
        p = os.path.join(d, f"score_{disk}{csv_suffix}.csv")
        absent = (not os.path.exists(p)) or (MUTATION == "drop_blind_arm" and arm == BLIND_ARM)
        if absent:
            if arm in expected and MUTATION != "silent_skip_missing":
                out["arms"][arm] = unavailable(f"score CSV not found: {p}")
            elif arm not in expected:
                out["arms_not_run"].append(arm)
            continue
        cnt = os.path.join(d, f"score_{disk}{csv_suffix}.counts.json")
        if is_model:
            st = _jload(cnt).get("status") if os.path.exists(cnt) else "NO_COUNTS"
            if st != "PASS" and MUTATION != "skip_count_guard":
                # ⛔ a CSV beside a FAILED count guard is partial work (MEASURED 2026-09-26 on navtest)
                out["arms_refused"][arm] = f"count guard {st} ({cnt})"
                out["arms"][arm] = unavailable(f"count guard {st} ({cnt}): a CSV beside a failed guard is partial work")
                continue
        df = load_csv(p, stage_of, arm)
        frames[arm] = df
        s2, s1 = df[df.stage == 2], df[df.stage == 1]
        seam = os.path.join(bridge, f"seam_{disk}.npz")
        own_s1 = True
        if is_model:
            z = np.load(seam, allow_pickle=False)
            own_s1 = "cv_standin" not in {str(x) for x in z["source"]}
            if MUTATION == "ignore_cv_standin":
                own_s1 = True
        rows = official_rows(p)
        rec = {"n_stage2": int(len(s2)), "n_stage1": int(len(s1)),
               "S2_EPDMS_u": ps.s2_group_uniform(df, mapping),
               "S2_scene_mean": float(s2.score.mean()),
               "S2_submetric_means": {k: float(s2[k].mean()) for k in SUB},
               "S2_multiplier_zero_rates": {k: float((s2[k] == 0).mean()) for k in ("NC", "DAC", "DDC", "TLC")},
               "S1_scene_mean": float(s1.score.mean()) if len(s1) else None,
               "stage1_rows_own": own_s1}
        if own_s1:
            rec["official_two_stage_EPDMS"] = rows.get("extended_pdm_score_combined")
            rec["official_rows"] = rows
        else:
            rec["official_two_stage_EPDMS"] = {
                "status": "UNAVAILABLE", "reason": "stage-1 rows are the declared CV stand-in "
                "(no original stage-1 frames for this split on this box)", "n": int(len(s1))}
            rec["official_rows_HYBRID"] = rows
        if is_model and os.path.exists(seam):
            rec["stop_fraction"] = stop_fraction(seam, stage_of)
        elif arm == "STOP_zero":
            rec["stop_fraction"] = {"frac_endpoint_lt_1m": 1.0, "note": "by construction"}
        # THE DEVICE ON EVERY SCORE (Master Mind, 2026-09-24): read from the seam's own manifest
        man = os.path.join(bridge, f"seam_{disk}.manifest.json")
        if is_model:
            m = _jload(man) if os.path.exists(man) else {}
            rec["device"] = "/".join(m.get("device", ["UNKNOWN"])) + " " + "/".join(m.get("precision", ["UNKNOWN"]))
        else:
            rec["device"] = "model-free (devkit scorer, CPU)"
        out["arms"][arm] = rec

    # ---- paired per-scene deltas (stage 2), S2-EPDMS-u deltas ------------------------------------------------------------------- #
    def _usable(arm):
        return arm in frames

    for x, y, what in PAIRS:
        key = f"{x}__minus__{y}"
        xs = _src(x, y)
        if not (_usable(xs) and _usable(y)):
            if x in expected and y in expected and MUTATION != "silent_skip_missing":
                gone = [a for a in (xs, y) if not _usable(a)]
                out["pairs"][key] = unavailable(f"arm(s) unavailable: {gone}", what=what)
            continue
        a2 = frames[xs][frames[xs].stage == 2].set_index("token")
        b2 = frames[y][frames[y].stage == 2].set_index("token")
        common = sorted(set(a2.index) & set(b2.index))
        d = (a2.loc[common, "score"] - b2.loc[common, "score"]).to_numpy()
        ux = out["arms"][xs]["S2_EPDMS_u"].get("value")
        uy = out["arms"][y]["S2_EPDMS_u"].get("value")
        rec = {"what": what, "n_scenes": len(common),
               "S2_EPDMS_u_delta": None if ux is None or uy is None else ux - uy,
               "scene_mean_delta": float(d.mean()), "wins": int((d > 1e-12).sum()),
               "ties": int((np.abs(d) <= 1e-12).sum()), "losses": int((d < -1e-12).sum()),
               "submetric_mean_deltas": {k: float(a2.loc[common, k].mean() - b2.loc[common, k].mean()) for k in SUB}}
        ox = out["arms"][xs].get("official_two_stage_EPDMS")
        oy = out["arms"][y].get("official_two_stage_EPDMS")
        if isinstance(ox, float) and isinstance(oy, float):
            rec["official_two_stage_EPDMS_delta"] = ox - oy
        out["pairs"][key] = rec
    # ---- the seed floor and the lever reads against it ------------------------------------------------------------------------- #
    sf = out["pairs"].get(f"{BAR_ARM}__minus__{FLOOR_ARM}")
    if sf and sf.get("S2_EPDMS_u_delta") is not None:
        floor = abs(sf["S2_EPDMS_u_delta"])
        out["seed_floor_S2_EPDMS_u"] = floor
        for k, v in out["pairs"].items():
            if k.startswith(f"{BAR_ARM}__minus__") and k != f"{BAR_ARM}__minus__{FLOOR_ARM}" \
                    and v.get("S2_EPDMS_u_delta") is not None:
                v["exceeds_seed_floor"] = abs(v["S2_EPDMS_u_delta"]) > floor
    else:
        # null, not a dict: navsim_bars8 does ``abs(floor)`` on this key, and a null reads "floor not measured" there
        out["seed_floor_S2_EPDMS_u"] = None
        out["seed_floor_S2_EPDMS_u_status"] = unavailable(
            f"{BAR_ARM} or {FLOOR_ARM} unavailable on this split -- the inference-seed floor is NOT measured")
    # ---- navhard: the settled log-cluster bootstrap on the official two-stage aggregate ----------------------------------------- #
    if split == "navhard_two_stage":
        ci = ci or load_ci()
        out["estimator"] = {"name": ci.ESTIMATOR, "unit": ci.CLUSTER_UNIT, "B": ci.N_BOOT, "seed": 0,
                            "aggregation": ci.AGG_TWO_STAGE,
                            "question": "another draw of LOGS? (blind to training and inference variance; the seed floor is "
                                        "read separately)"}
        out["estimator"]["module"] = os.path.abspath(ci.__file__)
        out["estimator"]["module_sha256"] = getattr(ci, "__file_sha256__", None)
        out["intervals"], out["paired_intervals"] = {}, {}
        contribs = {}
        clusters = [log_of[str(m[0])] for m in mapping]
        for arm in list(MODEL_ARMS) + list(FLOORS):
            if arm not in frames:
                if arm in expected and MUTATION != "silent_skip_missing":
                    out["intervals"][arm] = unavailable(f"arm unavailable: {out['arms'].get(arm, {}).get('reason')}")
                continue
            if not out["arms"][arm].get("stage1_rows_own"):
                if MUTATION != "silent_skip_missing":
                    out["intervals"][arm] = unavailable("stage-1 rows are the declared CV stand-in: no real two-stage EPDMS")
                continue
            d = scores if arm in MODEL_ARMS else floor_dir
            tag = f"{_disk(arm, alias)}{csv_suffix}"
            frame = os.path.join(d, f"score_{tag}_wrapper", f"{tag}_final_scores_frame.csv")
            try:
                rws, _t2l = ci.rows_from_score_frame(frame)
                contrib = ci.two_stage_key_contributions(rws, mapping, "score")
                contribs[arm] = contrib
                out["intervals"][arm] = ci.log_cluster_bootstrap(
                    contrib, clusters, aggregation=ci.AGG_TWO_STAGE,
                    official_value=out["arms"][arm]["official_two_stage_EPDMS"])
            except Exception as e:                                       # noqa: BLE001
                out["intervals"][arm] = {"status": "ERROR", "reason": repr(e)[:300], "frame": frame}
        for x, y, what in PAIRS:
            key = f"{x}__minus__{y}"
            xs = _src(x, y)
            if xs in contribs and y in contribs:
                try:
                    out["paired_intervals"][key] = ci.paired_log_cluster_bootstrap(
                        contribs[xs], contribs[y], clusters, aggregation=ci.AGG_TWO_STAGE,
                        official_a=out["arms"][xs]["official_two_stage_EPDMS"],
                        official_b=out["arms"][y]["official_two_stage_EPDMS"])
                except Exception as e:                                   # noqa: BLE001
                    out["paired_intervals"][key] = {"status": "ERROR", "reason": repr(e)[:300]}
            elif x in expected and y in expected and MUTATION != "silent_skip_missing":
                gone = [a for a in (xs, y) if a not in contribs]
                out["paired_intervals"][key] = unavailable(f"no per-key contributions for {gone} (arm unavailable or "
                                                           f"stage-1 not own)", what=what)
    else:
        out["estimator"] = {"status": "UNAVAILABLE",
                            "reason": "warmup_two_stage has 7 log clusters < the 8-cluster floor", "n": 7}
    return out


def write_summary(out: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o)
                  else (o.tolist() if hasattr(o, "tolist") else str(o)))


def _f4(v):
    return "n/a" if not isinstance(v, (int, float)) else f"{v:+.4f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="refcv8 NavSim summary parser (port of parse7.py)")
    ap.add_argument("--split", required=True, choices=sorted(SPLIT_EXPECTED))
    ap.add_argument("--scores", required=True, help="dir with score_<arm>.csv for the model arms (<run>/scores_<split>)")
    ap.add_argument("--floors", default="", help="dir with the floors' CSVs (default: the first existing refcv7 raw/floors/<split>)")
    ap.add_argument("--bridge", required=True, help="dir with seam_<arm>.npz (<run>/bridge_<split>)")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv-suffix", default="", help="__navhard_two_stage for navhard, empty for warmup (as run_navsim_refcv8 names them)")
    ap.add_argument("--alias", action="append", default=[], metavar="R8ARM=DISKSTEM",
                    help="read logical arm R8ARM from files named DISKSTEM (repeatable / comma-separated); the CONTROL uses this")
    ap.add_argument("--control-r7-names", action="store_true",
                    help="CONTROL: read refcv7's banked R7_* files as the mapped R8 arms (R7_A1->R8_LEGAL, R7_A1_s1->R8_LEGAL_s1, ...)")
    a = ap.parse_args(argv)
    if MUTATION:
        sys.exit(f"[parse8] PARSE8_MUTATION={MUTATION!r} is set: a deliberate-regression hook -- refusing to write a summary")
    sys.stdout.reconfigure(encoding="utf-8")
    alias = dict(CONTROL_R7_ALIAS) if a.control_r7_names else {}
    for item in a.alias:
        for kv in item.split(","):
            if kv:
                k, _, v = kv.partition("=")
                alias[k.strip()] = v.strip()
    floors = find_floors(a.split, a.csv_suffix, a.floors)
    if not floors:
        print("[parse8] WARNING: no floors dir found (STOP_zero/CV_official/ECHO_ha0_ext) -- those arms will read UNAVAILABLE; "
              "pass --floors", file=sys.stderr)
    out = build_summary(a.split, a.scores, a.bridge, a.inputs, a.label, floors=floors, csv_suffix=a.csv_suffix, alias=alias)
    write_summary(out, a.out)
    for arm, rec in out["arms"].items():
        if rec.get("status") == "UNAVAILABLE":
            print(f"{arm:14s} UNAVAILABLE: {rec.get('reason')}")
            continue
        u = rec["S2_EPDMS_u"]
        o = rec["official_two_stage_EPDMS"]
        print(f"{arm:14s} S2u={u.get('value', float('nan')):.4f} official2s="
              f"{o if isinstance(o, (float, type(None))) else 'UNAVAIL'} stop={rec.get('stop_fraction', {}).get('frac_endpoint_lt_1m')}")
    for k, v in out["pairs"].items():
        if v.get("status") == "UNAVAILABLE":
            print(f"{k:34s} UNAVAILABLE: {v.get('reason')}")
            continue
        print(f"{k:34s} dU={_f4(v['S2_EPDMS_u_delta'])} W/T/L={v['wins']}/{v['ties']}/{v['losses']}"
              f" {'(> seed floor)' if v.get('exceeds_seed_floor') else ''}")
    n_un = sum(1 for r in out["arms"].values() if r.get("status") == "UNAVAILABLE")
    print(f"[parse8] wrote {a.out}; expected arms {list(out['_parse8']['expected_arms'])}; UNAVAILABLE arms: {n_un}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
