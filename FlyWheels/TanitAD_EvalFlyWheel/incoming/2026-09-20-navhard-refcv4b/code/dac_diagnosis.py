#!/usr/bin/env python3
"""Why does refcv4b leave the drivable area? — the RULE-ZERO next experiment, 0 GPU, 0 scoring.

The decomposition named drivable-area compliance (DAC) as the one lever whose single-term repair lifts
A1 above STOP (0.3742 -> 0.4875 vs 0.4702). This asks WHICH KIND of plan fails it.

⭐ THE ATTRIBUTABLE SET. STOP also fails DAC (748 stage-2 scenes) — a parked car can start partly off
the drivable polygon. Those failures are the SCENE's, not the model's. The scenes that measure
refcv4b are the ones where **STOP passes DAC and A1 fails it**: there, A1's MOTION caused the failure.
The comparison set is where **both pass**.

WHAT IS MEASURED, per scene, from the plan the OFFICIAL scorer executed (``raw/A1/A1_hooks.json``,
``agent_poses`` — the scorer's own hook, not the seam), in the ego frame at t0 (x fwd, y left):
  * lateral excursion  = max |y| over the 4 s plan (m)
  * final heading      = |heading(4 s)| (deg)
  * progress           = x(4 s) (m), and the t0 speed v0 from the export (m/s)
  * progress ratio     = x(4 s) / max(v0 * 4 s, 1 m)  — >1 means the plan speeds up, <1 slows
⛔ A DIFFERENCE IN MEDIANS BETWEEN THE TWO SETS IS A DESCRIPTION, NOT A MECHANISM. Both halves of the
reading (lateral vs longitudinal) are reported, with n, and neither is promoted to a cause here.

    python dac_diagnosis.py --run <run> --inputs <export> --out raw/dac_diagnosis.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUMMARY_ROWS = ("extended_pdm_score_stage_one", "extended_pdm_score_stage_two",
                "extended_pdm_score_combined", "average_all_frames")


def dac_stage2(csv: Path) -> dict:
    d = pd.read_csv(csv, index_col=0)
    d = d[~d["token"].isin(SUMMARY_ROWS)]
    d = d[d["ego_progress_stage_one"].isna()]
    return {str(t): float(v) for t, v in zip(d["token"], d["drivable_area_compliance_stage_two"])}


def q(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "p25": round(float(np.percentile(v, 25)), 3),
            "median": round(float(np.median(v)), 3), "p75": round(float(np.percentile(v, 75)), 3),
            "p90": round(float(np.percentile(v, 90)), 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run = Path(a.run)
    doc = json.load(open(a.inputs, encoding="utf-8"))["tokens"]
    dac_a1 = dac_stage2(run / "scores" / "A1.csv")
    dac_st = dac_stage2(run / "scores" / "STOP.csv")
    hooks = json.load(open(run / "raw" / "A1" / "A1_hooks.json", encoding="utf-8")).get("pdm_score_calls", [])
    poses = {c["token"]: np.asarray(c["agent_poses"], dtype=np.float64)
             for c in hooks if c.get("token") in dac_a1 and c.get("agent_poses") is not None}
    sets = {"MODEL_CAUSED (STOP DAC=1, A1 DAC=0)": [t for t in dac_a1 if dac_st.get(t) == 1.0 and dac_a1[t] == 0.0],
            "BOTH_PASS (STOP DAC=1, A1 DAC=1)": [t for t in dac_a1 if dac_st.get(t) == 1.0 and dac_a1[t] == 1.0],
            "SCENE_CAUSED (STOP DAC=0)": [t for t in dac_a1 if dac_st.get(t) == 0.0]}
    out = {"_what": ("per-scene geometry of the plan the OFFICIAL scorer executed (A1_hooks.json agent_poses, "
                     "ego frame at t0), split by who failed drivable-area compliance on stage 2"),
           "run": str(run), "n_stage2_with_poses": len(poses), "sets": {}}
    for name, toks in sets.items():
        toks = [t for t in toks if t in poses]
        lat, head, prog, v0s, ratio = [], [], [], [], []
        for t in toks:
            p = poses[t]
            v0 = math.hypot(*doc[t]["ego_statuses"][-1]["ego_velocity"][:2])
            lat.append(float(np.max(np.abs(p[:, 1]))))
            head.append(abs(math.degrees(float(p[-1, 2]))))
            prog.append(float(p[-1, 0]))
            v0s.append(v0)
            ratio.append(float(p[-1, 0]) / max(v0 * 4.0, 1.0))
        out["sets"][name] = {"n": len(toks), "lateral_excursion_m": q(lat), "final_heading_deg": q(head),
                             "progress_4s_m": q(prog), "v0_mps": q(v0s), "progress_ratio_vs_v0x4s": q(ratio)}
    mc = out["sets"]["MODEL_CAUSED (STOP DAC=1, A1 DAC=0)"]
    bp = out["sets"]["BOTH_PASS (STOP DAC=1, A1 DAC=1)"]
    out["reading"] = {
        "lateral_excursion_median_ratio": (round(mc["lateral_excursion_m"]["median"] / bp["lateral_excursion_m"]["median"], 2)
                                           if bp["lateral_excursion_m"].get("median") else None),
        "final_heading_median_ratio": (round(mc["final_heading_deg"]["median"] / bp["final_heading_deg"]["median"], 2)
                                       if bp["final_heading_deg"].get("median") else None),
        "progress_ratio_median_model_caused": mc["progress_ratio_vs_v0x4s"].get("median"),
        "progress_ratio_median_both_pass": bp["progress_ratio_vs_v0x4s"].get("median"),
        "caveat": ("a difference in medians is a DESCRIPTION of the failing plans, not a mechanism; "
                   "neither reading is promoted to a cause here")}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, s in out["sets"].items():
        print(f"{name:40s} n={s['n']:5d}  lat|y|max med {s['lateral_excursion_m'].get('median')} m  "
              f"|head| med {s['final_heading_deg'].get('median')} deg  x(4s) med {s['progress_4s_m'].get('median')} m  "
              f"v0 med {s['v0_mps'].get('median')}  ratio med {s['progress_ratio_vs_v0x4s'].get('median')}")
    print(json.dumps(out["reading"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
