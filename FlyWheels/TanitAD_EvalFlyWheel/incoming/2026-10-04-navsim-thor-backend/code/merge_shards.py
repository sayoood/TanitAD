#!/usr/bin/env python3
"""Merge K Thor shard outputs of ONE arm into the single-run artifacts the refcv7 parsers read.

navhard (NAVSIM v2 venv, devkit importable -- Thor ``navsim-cpu`` or the dev-box navsim venv):
    python merge_shards.py navhard --frames k0_frame.csv k1_frame.csv ... --order-ref <dev-box CSV of the
        same split> --yaml <…/train_test_split/navhard_two_stage.yaml> --out-csv score_<ARM>__navhard_two_stage.csv \
        --out-frame <ARM>__navhard_two_stage_final_scores_frame.csv
  Each shard holds WHOLE two-stage groups, so every per-token column of its final frame (incl. the
  aggregator's weight / two_frame_extended_comfort / score) is already the full-split value. This
  re-runs the devkit's ``main()`` tail VERBATIM (``run_pdm_score.py`` lines after ``compute_final_scores``:
  stage split, ``calculate_individual_mapping_scores`` over ``reactive_all_mapping``, the three
  ``extended_pdm_score_*`` rows) on the merged frame. Row order = the reference CSV's token order
  (the dev box's log-iteration order), because pandas sums in row order.

navtest (any python):
    python merge_shards.py navtest --csvs k0.csv ... --order-ref <dev-box navtest CSV> --out-csv merged.csv
  v1.1's ``average`` row = ``DataFrame.mean(skipna=True)`` over the token rows in run order; it is
  recomputed with pandas in the reference order.

⚠ CSVs are parsed with float_precision="round_trip": pandas' default fast parser is NOT
round-trip exact and moved 1-ulp on ~14 % of navhard tokens in the first merge (MEASURED, a merge-tool
artefact, not a platform difference).

Every merge refuses on duplicate tokens, a token set different from the reference, or any invalid row.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys


def _ref_order(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return [r["token"] for r in csv.DictReader(fh)
                if not r["token"].startswith(("extended_pdm_score", "average"))]


def merge_navtest(a) -> int:
    import pandas as pd
    parts = [pd.read_csv(p, index_col=0, float_precision="round_trip") for p in a.csvs]
    df = pd.concat([p[p["token"] != "average"] for p in parts], ignore_index=True)
    order = _ref_order(a.order_ref)
    if df["token"].duplicated().any():
        raise SystemExit("duplicate tokens across shards")
    if set(df["token"]) != set(order):
        raise SystemExit(f"token set != reference ({len(set(df['token']) ^ set(order))} differ)")
    if not df["valid"].astype(str).isin(["True"]).all():
        raise SystemExit("invalid rows present -- refusing to merge")
    df["valid"] = True
    df = df.set_index("token").loc[order].reset_index()
    cols = list(pd.read_csv(a.csvs[0], index_col=0, nrows=1, float_precision="round_trip").columns)
    df = df[cols]
    # verbatim v1.1 run_pdm_score.main tail
    average_row = df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = df["valid"].all()
    df.loc[len(df)] = average_row
    df.to_csv(a.out_csv)
    print(json.dumps({"rows": len(df) - 1, "average_score": float(average_row["score"]), "out": a.out_csv}))
    return 0


def merge_navhard(a) -> int:
    from dataclasses import fields
    import numpy as np
    import pandas as pd
    import yaml
    if os.name != "nt":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import thor_compat
        thor_compat.install()
    from navsim.common.dataclasses import PDMResults
    from navsim.common.enums import SceneFrameType
    from navsim.planning.script.run_pdm_score import calculate_individual_mapping_scores

    parts = [pd.read_csv(p, float_precision="round_trip") for p in a.frames]
    pdm_score_df = pd.concat(parts, ignore_index=True)
    if "index" in pdm_score_df.columns:
        pdm_score_df = pdm_score_df.drop(columns=["index"])
    order = _ref_order(a.order_ref)
    if pdm_score_df["token"].duplicated().any():
        raise SystemExit("duplicate tokens across shards")
    if set(pdm_score_df["token"]) != set(order):
        raise SystemExit(f"token set != reference ({len(set(pdm_score_df['token']) ^ set(order))} differ)")
    pdm_score_df["valid"] = pdm_score_df["valid"].astype(str) == "True"
    if not pdm_score_df["valid"].all():
        raise SystemExit("invalid rows present -- refusing to merge")
    pdm_score_df = pdm_score_df.set_index("token").loc[order].reset_index()
    pdm_score_df["frame_type"] = pdm_score_df["frame_type"].astype(int)
    if a.out_frame:
        fr = pdm_score_df.copy()
        fr["frame_type"] = fr["frame_type"].astype(str)
        fr.insert(0, "index", range(len(fr)))
        fr.to_csv(a.out_frame, index=False)
    doc = yaml.safe_load(open(a.yaml, encoding="utf-8"))
    tokens = set(pdm_score_df["token"])
    all_mappings = {}
    for orig_token, prev_token, two_stage_pairs in doc["reactive_all_mapping"]:
        if prev_token in tokens or orig_token in tokens:
            all_mappings[(orig_token, prev_token)] = [tuple(pair) for pair in two_stage_pairs]
    pseudo_closed_loop_valid = True
    # ---- verbatim from navsim/planning/script/run_pdm_score.py main(), after compute_final_scores ----
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
    # ---- end verbatim ----
    pdm_score_df.to_csv(a.out_csv)
    print(json.dumps({"rows": len(pdm_score_df) - 3, "extended_pdm_score_combined": float(pcl_group_score["score"]),
                      "out": a.out_csv}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="split", required=True)
    h = sub.add_parser("navhard")
    h.add_argument("--frames", nargs="+", required=True)
    h.add_argument("--order-ref", required=True)
    h.add_argument("--yaml", required=True)
    h.add_argument("--out-csv", required=True)
    h.add_argument("--out-frame", default="")
    t = sub.add_parser("navtest")
    t.add_argument("--csvs", nargs="+", required=True)
    t.add_argument("--order-ref", required=True)
    t.add_argument("--out-csv", required=True)
    a = ap.parse_args()
    return merge_navhard(a) if a.split == "navhard" else merge_navtest(a)


if __name__ == "__main__":
    raise SystemExit(main())
