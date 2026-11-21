#!/usr/bin/env python3
"""Official NavSim CSVs -> the pre-registered statistics (SPEC §2, §4, §5). TANITAD VENV.

Reads ``raw/score_<arm>.csv`` (copied verbatim from the official run) and
``raw/navsim_agent_inputs.json`` (token -> stage, log; the reactive_all_mapping).
Reads the ``score`` column (EPDMS, EC injected) and NEVER ``pdm_score`` (the
official CSV does not even carry it; asserted).

Writes raw/scores_summary.json and prints the tables.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")
SUB = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
       "DDC": "driving_direction_compliance", "TLC": "traffic_light_compliance",
       "EP": "ego_progress", "TTC": "time_to_collision_within_bound",
       "LK": "lane_keeping", "HC": "history_comfort", "EC": "two_frame_extended_comfort"}
CAMERA_ARMS = ("A1_ego_cmd", "A2_vision_pure", "A3_ego_nocmd", "A1NT_ego_cmd_nearest",
               "A2NT_vision_pure_nearest")
PAIRS = [("A1_ego_cmd", "CV_official", "BAR-E2-1: A1 vs CV"),
         ("A1_ego_cmd", "A2_vision_pure", "ego-status + command contribution (A1-A2)"),
         ("A1_ego_cmd", "A3_ego_nocmd", "command contribution (A1-A3)"),
         ("A1_ego_cmd", "A4_blind_ego_cmd", "vision contribution (A1-A4 frames-blind)"),
         ("A1_ego_cmd", "A1NT_ego_cmd_nearest", "time construction ST vs NT (A1-A1NT)"),
         ("A2_vision_pure", "A2NT_vision_pure_nearest", "time construction, vision-pure (A2-A2NT)"),
         ("A3_ego_nocmd", "CV_official", "A3 vs CV"),
         ("A4_blind_ego_cmd", "CV_official", "A4 (blind) vs CV"),
         ("A1_ego_cmd", "ECHO_ha0_ext", "POST-HOC: A1 vs the echo of its own ego inputs"),
         ("A3_ego_nocmd", "ECHO_ha0_ext", "POST-HOC: A3 vs the echo of its own ego inputs"),
         ("ECHO_ha0_ext", "CV_official", "POST-HOC: echo control vs CV"),
         ("A1NT_ego_cmd_nearest", "CV_official", "A1NT vs CV"),
         ("A1_ego_cmd", "STOP_zero", "POST-HOC: A1 vs doing nothing"),
         ("A2_vision_pure", "STOP_zero", "POST-HOC: vision-pure vs doing nothing"),
         ("CV_official", "STOP_zero", "POST-HOC: CV vs doing nothing"),
         ("ECHO_ha0_ext", "STOP_zero", "POST-HOC: echo vs doing nothing"),
         ("A4_blind_ego_cmd", "STOP_zero", "POST-HOC: frames-blind vs doing nothing")]


def load_arm(arm: str, stage_of: dict) -> pd.DataFrame | None:
    p = os.path.join(RAW, f"score_{arm}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if "pdm_score" in df.columns:
        raise SystemExit(f"{p}: carries a pdm_score column — refusing (read `score`)")
    tok = df[~df.token.str.startswith("extended_pdm_score")].copy()
    tok["stage"] = tok.token.map(stage_of)
    if tok.stage.isna().any():
        raise SystemExit(f"{arm}: {int(tok.stage.isna().sum())} tokens not in the export")
    for k, c in SUB.items():
        c1, c2 = f"{c}_stage_one", f"{c}_stage_two"
        tok[k] = np.where(tok.stage == 1, tok.get(c1), tok.get(c2))
    tok["arm"] = arm
    return tok


def s2_group_uniform(df: pd.DataFrame, mapping: list, col: str = "score") -> dict:
    s2 = df[df.stage == 2].set_index("token")[col]
    gm, sizes = [], []
    for orig, prev, pairs in mapping:
        for grp in ([p[0] for p in pairs], [p[1] for p in pairs]):
            v = s2.reindex(grp)
            if v.isna().any():
                return {"status": "UNAVAILABLE", "reason": f"{int(v.isna().sum())} group tokens "
                        f"missing/NaN", "n": int(v.notna().sum())}
            gm.append(float(v.mean()))
            sizes.append(len(grp))
    return {"value": float(np.mean(gm)), "n_groups": len(gm), "group_sizes": sizes,
            "n_scenes": int(sum(sizes))}


def official_rows(arm: str) -> dict:
    df = pd.read_csv(os.path.join(RAW, f"score_{arm}.csv"))
    s = df[df.token.str.startswith("extended_pdm_score")].set_index("token")
    return {k: float(s.loc[k, "score"]) for k in s.index}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="CV_official,A1_ego_cmd,A2_vision_pure,A3_ego_nocmd,"
                                      "A4_blind_ego_cmd,A1NT_ego_cmd_nearest,"
                                      "A2NT_vision_pure_nearest,K4_seam_cv,ECHO_ha0_ext,STOP_zero")
    a = ap.parse_args(argv)
    doc = json.load(open(os.path.join(RAW, "navsim_agent_inputs.json"), encoding="utf-8"))
    stage_of = {t: r["stage"] for t, r in doc["tokens"].items()}
    log_of = {t: r["log_name"] for t, r in doc["tokens"].items()}
    mapping = doc["reactive_all_mapping"]
    arms = {}
    for arm in [x for x in a.arms.split(",") if x]:
        df = load_arm(arm, stage_of)
        if df is not None:
            arms[arm] = df
    out = {"_estimator": {"status": "UNAVAILABLE",
                          "reason": "warmup_two_stage has 7 log groups < the RG-14 floor of 8 "
                                    "— no interval (point estimates + paired per-scene counts only)",
                          "n": 7},
           "_primary": "S2-EPDMS-u (SPEC §2): mean over 16 groups of the uniform mean of the "
                       "official per-scene `score` of the stage-2 tokens", "arms": {}, "pairs": {}}
    for arm, df in arms.items():
        s2 = df[df.stage == 2]
        s1 = df[df.stage == 1]
        rec = {"n_stage2": int(len(s2)), "n_stage1": int(len(s1)),
               "S2_EPDMS_u": s2_group_uniform(df, mapping),
               "S2_scene_mean": float(s2.score.mean()),
               "S2_submetric_means": {k: float(s2[k].mean()) for k in SUB},
               "S2_multiplier_zero_rates": {k: float((s2[k] == 0).mean()) for k in
                                            ("NC", "DAC", "DDC", "TLC")},
               "S2_by_log": {ln: float(g.score.mean()) for ln, g in
                             s2.assign(log=s2.token.map(log_of)).groupby("log")},
               "S1_scene_mean": float(s1.score.mean()) if len(s1) else None,
               "S1_submetric_means": {k: float(s1[k].mean()) for k in SUB} if len(s1) else None,
               "official_summary_rows": official_rows(arm)}
        if arm in CAMERA_ARMS:
            rec["_HYBRID_WARNING"] = ("stage-1 rows = devkit CV stand-in; the official "
                                      "summary rows (and the stage-2 kernel weights) are HYBRID "
                                      "and are NOT this arm's EPDMS")
        out["arms"][arm] = rec
    for x, y, what in PAIRS:
        if x not in arms or y not in arms:
            continue
        a2 = arms[x][arms[x].stage == 2].set_index("token")
        b2 = arms[y][arms[y].stage == 2].set_index("token")
        common = sorted(set(a2.index) & set(b2.index))
        d = (a2.loc[common, "score"] - b2.loc[common, "score"]).to_numpy()
        ux = out["arms"][x]["S2_EPDMS_u"].get("value")
        uy = out["arms"][y]["S2_EPDMS_u"].get("value")
        out["pairs"][f"{x}__minus__{y}"] = {
            "what": what, "n_scenes": len(common),
            "S2_EPDMS_u_delta": (None if ux is None or uy is None else ux - uy),
            "scene_mean_delta": float(d.mean()),
            "wins": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()),
            "losses": int((d < -1e-12).sum()),
            "submetric_mean_deltas": {k: float(a2.loc[common, k].mean() - b2.loc[common, k].mean())
                                      for k in SUB}}
    if "A1_ego_cmd" in out["arms"] and "CV_official" in out["arms"]:
        a1 = out["arms"]["A1_ego_cmd"]["S2_EPDMS_u"].get("value")
        cv = out["arms"]["CV_official"]["S2_EPDMS_u"].get("value")
        out["BAR_E2_1"] = {"A1": a1, "CV": cv,
                           "verdict": ("PASS" if (a1 is not None and cv is not None and a1 > cv)
                                       else "FAIL"),
                           "rule": "S2-EPDMS-u(A1) > S2-EPDMS-u(CV), identical 204 stage-2 tokens"}
    if "K4_seam_cv" in arms and "CV_official" in arms:
        k = arms["K4_seam_cv"].set_index("token").sort_index()
        c = arms["CV_official"].set_index("token").sort_index()
        cols = [col for col in c.columns if col not in ("arm",)]
        same = bool(k.index.equals(c.index)) and all(
            np.array_equal(k[col].to_numpy(), c[col].to_numpy(), equal_nan=True)
            if k[col].dtype.kind in "fi" else (k[col] == c[col]).all() for col in cols)
        out["K4_seam_transparency"] = {"n_tokens": int(len(k)), "identical_all_columns": same}
    with open(os.path.join(RAW, "scores_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=lambda o: None if (isinstance(o, float) and
                                                               math.isnan(o)) else str(o))
    for arm, rec in out["arms"].items():
        u = rec["S2_EPDMS_u"]
        print(f"{arm:28s} S2-EPDMS-u={u.get('value', float('nan')):.4f}  scene-mean="
              f"{rec['S2_scene_mean']:.4f}  " + " ".join(f"{k}={v:.3f}" for k, v in
                                                         rec["S2_submetric_means"].items()))
    for k, v in out["pairs"].items():
        print(f"{k:55s} dU={v['S2_EPDMS_u_delta'] if v['S2_EPDMS_u_delta'] is None else round(v['S2_EPDMS_u_delta'], 4)}"
              f"  W/T/L={v['wins']}/{v['ties']}/{v['losses']}")
    if "BAR_E2_1" in out:
        print("BAR-E2-1:", out["BAR_E2_1"])
    if "K4_seam_transparency" in out:
        print("K4:", out["K4_seam_transparency"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
