#!/usr/bin/env python3
"""RE-AGGREGATE a banked PRE-AGGREGATION dump with the DEVKIT'S OWN functions (NavSim venv).

⛔ WHY THIS EXISTS (MEASURED 2026-09-19, EvalFlyWheel): the navhard CV run scored **all 5,912
scenarios in 68 minutes** and then died in aggregation — ``create_scene_aggregators`` raised
(``AssertionError: Invalid interval nan``, caught at ``run_pdm_score.py:382``) and
``calculate_individual_mapping_scores`` then raised ``TypeError: Could not convert [array([nan…])]``
— so the run exited 1 and wrote **no CSV**. The compute was already paid for. Same class as the
``t1_eval`` analysis-time import that destroyed a completed rollout (CLAUDE.md traps preflight):
⇒ **check for the banked dump BEFORE re-running anything.**

The wrapper's ``--dump-preaggregation`` banks the per-token frame the PARENT receives from
``worker_map`` (pool-safe) as CSV **and** pickle; this script re-runs ONLY the aggregation, using
the devkit's own ``create_scene_aggregators`` / ``compute_final_scores`` /
``calculate_individual_mapping_scores`` — never a second implementation of the metric. The output
tail (stage/combined summary rows, column layout) is a VERBATIM copy of
``run_pdm_score.py::main`` (lines 388-462 @0a380a9), marked below, so the CSV is the devkit's own
format; on warmup it is validated to reproduce the devkit's CSV bit-for-bit.

    python reaggregate.py --preagg <…_preaggregation.pkl|csv> --split warmup_two_stage --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.enums import SceneFrameType
from navsim.common.dataclasses import PDMResults
from navsim.planning.script.run_pdm_score import (calculate_individual_mapping_scores,
                                                  compute_final_scores, create_scene_aggregators)
from dataclasses import fields

DEVKIT = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", "C:/Users/Admin/navsim-crun/devkit"))
TTS = DEVKIT / "navsim/planning/script/config/common/train_test_split"
COMMON = DEVKIT / "navsim/planning/script/config/common/default_common.yaml"


def proposal_sampling():
    """The devkit's OWN ``proposal_sampling`` (default_common.yaml) — read, never assumed."""
    cfg = OmegaConf.create(yaml.safe_load(COMMON.read_text(encoding="utf-8"))["proposal_sampling"])
    return instantiate(cfg)


def load_dump(path: Path) -> pd.DataFrame:
    p = Path(path)
    pkl = p.with_suffix(".pkl")
    if pkl.exists():
        df = pd.read_pickle(pkl)
        df.attrs["source"] = str(pkl)
        return df
    df = pd.read_csv(p)
    df.attrs["source"] = str(p)
    df.attrs["csv_only"] = ("⛔ CSV fallback: the array columns (ego_simulated_states) are NOT recoverable from a "
                            "CSV, so the two-frame comfort cannot be recomputed and the aggregation will fail the "
                            "same way it did originally. Use the .pkl.")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preagg", required=True, type=Path)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True, type=Path, help="output DIRECTORY (a devkit-format CSV is written in it)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    pdm_score_df = load_dump(a.preagg)
    rec = {"preagg": str(a.preagg), "source": pdm_score_df.attrs.get("source"), "n_rows": int(len(pdm_score_df)),
           "split": a.split, "csv_only_warning": pdm_score_df.attrs.get("csv_only")}
    if "frame_type" in pdm_score_df.columns and pdm_score_df["frame_type"].dtype == object:
        m = {str(t): t for t in SceneFrameType}
        pdm_score_df["frame_type"] = [m.get(str(v), v) for v in pdm_score_df["frame_type"]]
    split_cfg = yaml.safe_load((TTS / f"{a.split}.yaml").read_text(encoding="utf-8"))
    tokens = set(pdm_score_df["token"])
    all_mappings = {}
    for orig_token, prev_token, two_stage_pairs in split_cfg["reactive_all_mapping"]:
        if prev_token in tokens or orig_token in tokens:
            all_mappings[(orig_token, prev_token)] = [tuple(pair) for pair in two_stage_pairs]
    rec["n_mapping_keys"] = len(all_mappings)

    # ---- VERBATIM from run_pdm_score.py::main (lines 388-462 @0a380a9) -------------------------
    try:
        pdm_score_df = create_scene_aggregators(all_mappings, pdm_score_df, proposal_sampling())
        pdm_score_df = compute_final_scores(pdm_score_df)
        pseudo_closed_loop_valid = True
    except Exception:
        print("----------- Failed to calculate pseudo closed-loop weights or comfort:")
        traceback.print_exc()
        rec["aggregation_exception"] = traceback.format_exc()[-4000:]
        pdm_score_df["weight"] = 1.0
        pseudo_closed_loop_valid = False

    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios

    score_cols = [
        c
        for c in pdm_score_df.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]
    pcl_group_score, pcl_stage1_score, pcl_stage2_score = calculate_individual_mapping_scores(
        pdm_score_df[score_cols + ["token", "weight"]], all_mappings
    )
    for col in score_cols:
        stage_one_mask = pdm_score_df["frame_type"] == SceneFrameType.ORIGINAL
        stage_two_mask = pdm_score_df["frame_type"] == SceneFrameType.SYNTHETIC
        pdm_score_df.loc[stage_one_mask, f"{col}_stage_one"] = pdm_score_df.loc[stage_one_mask, col]
        pdm_score_df.loc[stage_two_mask, f"{col}_stage_two"] = pdm_score_df.loc[stage_two_mask, col]
    pdm_score_df.drop(columns=score_cols, inplace=True)
    pdm_score_df["score"] = pdm_score_df["score_stage_one"].combine_first(pdm_score_df["score_stage_two"])
    pdm_score_df.drop(columns=["score_stage_one", "score_stage_two"], inplace=True)
    stage1_cols = [f"{col}_stage_one" for col in score_cols if col != "score"]
    stage2_cols = [f"{col}_stage_two" for col in score_cols if col != "score"]
    score_cols = stage1_cols + stage2_cols + ["score"]
    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]
    summary_rows = []
    stage1_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage1_row["token"] = "extended_pdm_score_stage_one"
    stage1_row["valid"] = pseudo_closed_loop_valid
    stage1_row["score"] = pcl_stage1_score.get("score", np.nan)
    for col in pcl_stage1_score.index:
        if col not in ["token", "valid", "score"]:
            stage1_row[f"{col}_stage_one"] = pcl_stage1_score[col]
    summary_rows.append(stage1_row)
    stage2_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage2_row["token"] = "extended_pdm_score_stage_two"
    stage2_row["valid"] = pseudo_closed_loop_valid
    stage2_row["score"] = pcl_stage2_score.get("score", np.nan)
    for col in pcl_stage2_score.index:
        if col not in ["token", "valid", "score"]:
            stage2_row[f"{col}_stage_two"] = pcl_stage2_score[col]
    summary_rows.append(stage2_row)
    combined_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    combined_row["token"] = "extended_pdm_score_combined"
    combined_row["valid"] = pseudo_closed_loop_valid
    combined_row["score"] = pcl_group_score["score"]
    for col in pcl_stage1_score.index:
        if col not in ["token", "valid", "score"]:
            combined_row[f"{col}_stage_one"] = pcl_stage1_score[col]
    for col in pcl_stage2_score.index:
        if col not in ["token", "valid", "score"]:
            combined_row[f"{col}_stage_two"] = pcl_stage2_score[col]
    summary_rows.append(combined_row)
    pdm_score_df = pd.concat([pdm_score_df, pd.DataFrame(summary_rows)], ignore_index=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    out_csv = a.out / f"{timestamp}.csv"
    pdm_score_df.to_csv(out_csv)
    # ---- end of the verbatim block --------------------------------------------------------------

    rec.update({"csv": str(out_csv), "pseudo_closed_loop_valid": bool(pseudo_closed_loop_valid),
                "num_successful_scenarios": int(num_sucessful_scenarios),
                "num_failed_scenarios": int(num_failed_scenarios),
                "combined_score": (None if pd.isna(combined_row["score"]) else float(combined_row["score"])),
                "proposal_sampling": {"num_poses": proposal_sampling().num_poses,
                                      "interval_length": proposal_sampling().interval_length,
                                      "source": str(COMMON)}})
    (a.out / "REAGGREGATION.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("csv", "n_rows", "n_mapping_keys", "pseudo_closed_loop_valid",
                                          "combined_score", "num_failed_scenarios")}, indent=1))
    return 0 if pseudo_closed_loop_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
