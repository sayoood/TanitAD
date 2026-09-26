#!/usr/bin/env python3
"""Official NavSim CSVs -> this SPEC's statistics (SPEC §4/§5). TANITAD VENV.

    python code/parse6.py --split warmup_two_stage --scores raw/scores_warmup_s1000 \
        --bridge raw/bridge_warmup_s1000 --label PIPELINE-VALIDATION --out raw/summary_warmup_s1000.json

* reads the official ``score`` column (EPDMS with the two-frame EC injected) — NEVER ``pdm_score``;
* S2-EPDMS-u with E2's OWN ``parse_scores.s2_group_uniform`` (imported);
* the official two-stage row (``extended_pdm_score_combined``) is reported as a real two-stage
  EPDMS ONLY for arms whose stage-1 rows are their own (no ``cv_standin`` in the seam); otherwise
  it is kept under ``official_rows_HYBRID`` and never headlined;
* navhard: the settled ``navsim_log_cluster_bootstrap`` (``taniteval/adapters/navsim_ci.py``,
  IMPORTED) on the official two-stage aggregate, and its paired variant vs each floor;
* the seed floor (R6_A1 vs R6_A1_s1) and the stop fraction (plans whose 4 s endpoint is < 1 m from
  the origin) beside every arm.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
E2 = os.path.abspath(os.path.join(PKG, "..", "..", "2026-09-19-navsim-refcv4b-bridge"))
SUB = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
       "DDC": "driving_direction_compliance", "TLC": "traffic_light_compliance",
       "EP": "ego_progress", "TTC": "time_to_collision_within_bound",
       "LK": "lane_keeping", "HC": "history_comfort", "EC": "two_frame_extended_comfort"}
FLOORS = ("CV_official", "STOP_zero", "ECHO_ha0_ext")
MODEL_ARMS = ("R6_A1", "R6_A1_s1", "R6_A1NT", "R6_BLIND", "R6_NAVOFF", "R6_VMAXOFF")
PAIRS = [("R6_A1", "CV_official", "A1 vs CV"), ("R6_A1", "STOP_zero", "A1 vs STOP (the real floor)"),
         ("R6_A1", "ECHO_ha0_ext", "A1 vs the echo of its own ego inputs"),
         ("R6_A1", "R6_A1_s1", "SEED FLOOR: A1 vs the same arm at inference seed 1"),
         ("R6_A1", "R6_BLIND", "vision contribution (A1 - frames-blind)"),
         ("R6_A1", "R6_NAVOFF", "nav contribution (A1 - nav withheld)"),
         ("R6_A1", "R6_VMAXOFF", "max-speed contribution (A1 - max speed withheld)"),
         ("R6_A1", "R6_A1NT", "time construction (ST - NT)"),
         ("STOP_zero", "CV_official", "STOP vs CV (floor ordering)")]


#: every refcv6 number carries this stamp (Master Mind, 2026-09-26)
MODEL_STAMP = ('"F3 detach-only, F4 on the last layer only" — Master Mind audit 2026-09-26 (GOALS_AND_CLAIMS D-REFCV6-F3-WHITELIST, D-REFCV6-LABEL-CLOCK; landed 9d16c441): the F3 per-stage cascade loss never ran (decoder stages 0-2 frozen at init) and tactical labels are read ~0.37 s early; INHERITED')

def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--scores", required=True, help="dir with score_<arm>.csv for the model arms")
    ap.add_argument("--floors", default="", help="dir with the floors' CSVs (default --scores)")
    ap.add_argument("--bridge", required=True, help="dir with seam_<arm>.npz")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv-suffix", default="", help="e.g. __navhard_two_stage")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    ps = _load_mod("e2_parse_scores6", os.path.join(E2, "code", "parse_scores.py"))
    doc = json.load(open(a.inputs, encoding="utf-8"))
    stage_of = {t: r["stage"] for t, r in doc["tokens"].items()}
    log_of = {t: r["log_name"] for t, r in doc["tokens"].items()}
    mapping = doc["reactive_all_mapping"]
    out = {"_label": a.label, "split": a.split,
           "model_as_trained": MODEL_STAMP,
           "_primary": "S2-EPDMS-u (E2 SPEC §2) + the official two-stage EPDMS where the arm's "
                       "stage-1 rows are its own", "arms": {}, "pairs": {}}
    arms = {}
    floor_dir = a.floors or a.scores
    for arm in MODEL_ARMS + FLOORS:
        d = a.scores if arm in MODEL_ARMS else floor_dir
        p = os.path.join(d, f"score_{arm}{a.csv_suffix}.csv")
        if not os.path.exists(p):
            continue
        df = load_csv(p, stage_of, arm)
        arms[arm] = df
        s2, s1 = df[df.stage == 2], df[df.stage == 1]
        seam = os.path.join(a.bridge, f"seam_{arm}.npz")
        own_s1 = True
        if arm in MODEL_ARMS:
            z = np.load(seam, allow_pickle=False)
            own_s1 = "cv_standin" not in {str(x) for x in z["source"]}
        rows = official_rows(p)
        rec = {"n_stage2": int(len(s2)), "n_stage1": int(len(s1)),
               "S2_EPDMS_u": ps.s2_group_uniform(df, mapping),
               "S2_scene_mean": float(s2.score.mean()),
               "S2_submetric_means": {k: float(s2[k].mean()) for k in SUB},
               "S2_multiplier_zero_rates": {k: float((s2[k] == 0).mean())
                                            for k in ("NC", "DAC", "DDC", "TLC")},
               "S1_scene_mean": float(s1.score.mean()) if len(s1) else None,
               "stage1_rows_own": own_s1}
        if own_s1:
            rec["official_two_stage_EPDMS"] = rows.get("extended_pdm_score_combined")
            rec["official_rows"] = rows
        else:
            rec["official_two_stage_EPDMS"] = {
                "status": "UNAVAILABLE", "reason": "stage-1 rows are the declared CV stand-in "
                "(no original stage-1 frames for this split on this box)",
                "n": int(len(s1))}
            rec["official_rows_HYBRID"] = rows
        if arm in MODEL_ARMS and os.path.exists(seam):
            rec["stop_fraction"] = stop_fraction(seam, stage_of)
        elif arm == "STOP_zero":
            rec["stop_fraction"] = {"frac_endpoint_lt_1m": 1.0, "note": "by construction"}
        # THE DEVICE ON EVERY SCORE (Master Mind, 2026-09-24): read from the seam's own manifest
        man = os.path.join(a.bridge, f"seam_{arm}.manifest.json")
        if arm in MODEL_ARMS:
            m = json.load(open(man, encoding="utf-8")) if os.path.exists(man) else {}
            rec["device"] = "/".join(m.get("device", ["UNKNOWN"])) + " " + "/".join(
                m.get("precision", ["UNKNOWN"]))
        else:
            rec["device"] = "model-free (devkit scorer, CPU)"
        out["arms"][arm] = rec
    # ---- paired per-scene deltas (stage 2), S2-EPDMS-u deltas ------------------------------ #
    for x, y, what in PAIRS:
        if x not in arms or y not in arms:
            continue
        a2 = arms[x][arms[x].stage == 2].set_index("token")
        b2 = arms[y][arms[y].stage == 2].set_index("token")
        common = sorted(set(a2.index) & set(b2.index))
        d = (a2.loc[common, "score"] - b2.loc[common, "score"]).to_numpy()
        ux = out["arms"][x]["S2_EPDMS_u"].get("value")
        uy = out["arms"][y]["S2_EPDMS_u"].get("value")
        rec = {"what": what, "n_scenes": len(common),
               "S2_EPDMS_u_delta": None if ux is None or uy is None else ux - uy,
               "scene_mean_delta": float(d.mean()), "wins": int((d > 1e-12).sum()),
               "ties": int((np.abs(d) <= 1e-12).sum()), "losses": int((d < -1e-12).sum()),
               "submetric_mean_deltas": {k: float(a2.loc[common, k].mean() - b2.loc[common, k].mean())
                                         for k in SUB}}
        ox = out["arms"][x].get("official_two_stage_EPDMS")
        oy = out["arms"][y].get("official_two_stage_EPDMS")
        if isinstance(ox, float) and isinstance(oy, float):
            rec["official_two_stage_EPDMS_delta"] = ox - oy
        out["pairs"][f"{x}__minus__{y}"] = rec
    # ---- the seed floor and the lever reads against it ---------------------------------- #
    sf = out["pairs"].get("R6_A1__minus__R6_A1_s1")
    if sf and sf["S2_EPDMS_u_delta"] is not None:
        floor = abs(sf["S2_EPDMS_u_delta"])
        out["seed_floor_S2_EPDMS_u"] = floor
        for k, v in out["pairs"].items():
            if k.startswith("R6_A1__minus__") and k != "R6_A1__minus__R6_A1_s1" \
                    and v["S2_EPDMS_u_delta"] is not None:
                v["exceeds_seed_floor"] = abs(v["S2_EPDMS_u_delta"]) > floor
    # ---- navhard: the settled log-cluster bootstrap on the official two-stage aggregate ---- #
    if a.split == "navhard_two_stage":
        ci = _load_mod("navsim_ci6", "D:/Projects/TanitAD/taniteval/adapters/navsim_ci.py")
        out["estimator"] = {"name": ci.ESTIMATOR, "unit": ci.CLUSTER_UNIT, "B": ci.N_BOOT,
                            "seed": 0, "aggregation": ci.AGG_TWO_STAGE,
                            "question": "another draw of LOGS? (blind to training and "
                                        "inference variance; the seed floor is read separately)"}
        out["estimator"]["module"] = os.path.abspath(ci.__file__)
        out["intervals"], out["paired_intervals"] = {}, {}
        contribs = {}
        clusters = [log_of[str(m[0])] for m in mapping]
        for arm in arms:
            if not out["arms"][arm].get("stage1_rows_own"):
                continue
            d = a.scores if arm in MODEL_ARMS else floor_dir
            tag = f"{arm}{a.csv_suffix}"
            frame = os.path.join(d, f"score_{tag}_wrapper", f"{tag}_final_scores_frame.csv")
            try:
                rows, _t2l = ci.rows_from_score_frame(frame)
                contrib = ci.two_stage_key_contributions(rows, mapping, "score")
                contribs[arm] = contrib
                out["intervals"][arm] = ci.log_cluster_bootstrap(
                    contrib, clusters, aggregation=ci.AGG_TWO_STAGE,
                    official_value=out["arms"][arm]["official_two_stage_EPDMS"])
            except Exception as e:                                       # noqa: BLE001
                out["intervals"][arm] = {"status": "ERROR", "reason": repr(e)[:300],
                                         "frame": frame}
        for x, y, _w in PAIRS:
            if x in contribs and y in contribs:
                try:
                    out["paired_intervals"][f"{x}__minus__{y}"] = ci.paired_log_cluster_bootstrap(
                        contribs[x], contribs[y], clusters, aggregation=ci.AGG_TWO_STAGE,
                        official_a=out["arms"][x]["official_two_stage_EPDMS"],
                        official_b=out["arms"][y]["official_two_stage_EPDMS"])
                except Exception as e:                                   # noqa: BLE001
                    out["paired_intervals"][f"{x}__minus__{y}"] = {"status": "ERROR",
                                                                   "reason": repr(e)[:300]}
    else:
        out["estimator"] = {"status": "UNAVAILABLE",
                            "reason": "warmup_two_stage has 7 log clusters < the 8-cluster floor",
                            "n": 7}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o)
                  else (o.tolist() if hasattr(o, "tolist") else str(o)))
    for arm, rec in out["arms"].items():
        u = rec["S2_EPDMS_u"]
        o = rec["official_two_stage_EPDMS"]
        print(f"{arm:14s} S2u={u.get('value', float('nan')):.4f} official2s="
              f"{o if isinstance(o, (float, type(None))) else 'UNAVAIL'} stop="
              f"{rec.get('stop_fraction', {}).get('frac_endpoint_lt_1m')}")
    for k, v in out["pairs"].items():
        print(f"{k:34s} dU={v['S2_EPDMS_u_delta']:+.4f} W/T/L={v['wins']}/{v['ties']}/{v['losses']}"
              f" {'(> seed floor)' if v.get('exceeds_seed_floor') else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
