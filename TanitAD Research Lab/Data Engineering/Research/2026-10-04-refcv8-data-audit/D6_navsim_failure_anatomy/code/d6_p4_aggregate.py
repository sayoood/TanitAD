"""D6 P4 -- assemble each arm's per-token frame and run the DEVKIT's OWN two-stage aggregation on it.   NAVSIM VENV.

    PYTHONPATH="C:/Users/Admin/navsim-crun/devkit;C:/Users/Admin/navsim-crun/nuplan-devkit" C:/Users/Admin/navsim-crun/venv/Scripts/python.exe \
        d6_p4_aggregate.py --arms G0,G1,G1DER,GORC

Per arm: the per-token rows = the arm's exact re-score where its pick CHANGED (raw/p4_scored/<arm>_s*.jsonl) and the G0 exact re-score
otherwise (raw/p4_scored/G0_s*.jsonl), in the banked final frame's row order, with exactly the columns ``run_pdm_score.run_pdm_score``
builds.  Then ``create_scene_aggregators`` -> ``compute_final_scores`` -> ``calculate_individual_mapping_scores`` (navsim/planning/script/
run_pdm_score.py @0a380a9, imported, not copied) with the export's ``reactive_all_mapping``.
Writes raw/p4_frame_<arm>.csv (the compute_final_scores frame, written exactly as the official wrapper wrote it) and raw/p4_agg.json
(EPDMS + stage aggregates per arm, and the G0 identity control: every column of the re-built frame vs the banked frame, and the G0 EPDMS vs
the banked combined score).  Refuses an arm with a missing or non-OK token.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys
from dataclasses import fields

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d6_rescore  # noqa: E402,F401  (env defaults before the devkit import)
import d6_p4_common as C  # noqa: E402
from navsim.common.dataclasses import PDMResults  # noqa: E402
from navsim.common.enums import SceneFrameType  # noqa: E402
from navsim.planning.script import run_pdm_score as RPS  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402

RAW = C.RAW
PROPOSAL_SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)      # cfg.simulator.proposal_sampling
PDM_COLS = [f.name for f in fields(PDMResults)]


def load_scored(arm: str) -> dict:
    out, bad = {}, []
    for p in sorted(glob.glob(os.path.join(RAW, "p4_scored", f"{arm}_s*.jsonl"))):
        for ln in open(p, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "OK":
                out[r["token"]] = r
            else:
                bad.append(r["token"])
    return out, sorted(set(bad) - set(out))


def to_row_df(r: dict) -> pd.DataFrame:
    """The single-row frame run_pdm_score builds for one token (PDMResults columns, then the runner's added columns, in its order)."""
    ess = np.frombuffer(base64.b64decode(r["ess_b64"]), np.float64).reshape(r["ess_shape"]).copy()
    d = {c: r[c] for c in ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
                           "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
                           "history_comfort", "multiplicative_metrics_prod")}
    df = pd.DataFrame([{**d, "weighted_metrics": np.asarray(r["weighted_metrics"], np.float64),
                        "weighted_metrics_array": np.asarray(r["weighted_metrics_array"], np.float64),
                        "pdm_score": r["pdm_score"]}])
    df["valid"] = True
    df["log_name"] = r["log_name"]
    df["frame_type"] = SceneFrameType(int(r["frame_type"]))
    df["start_time"] = r["start_time"]
    df["endpoint_x"] = r["endpoint_x"]
    df["endpoint_y"] = r["endpoint_y"]
    df["start_point_x"] = r["start_point_x"]
    df["start_point_y"] = r["start_point_y"]
    df["ego_simulated_states"] = [ess]
    df["token"] = r["token"]
    return df


def aggregate(rows: list, mapping) -> tuple:
    pdm_score_df = pd.concat(rows)
    all_mappings = {}
    for orig_token, prev_token, two_stage_pairs in mapping:
        all_mappings[(orig_token, prev_token)] = [tuple(pair) for pair in two_stage_pairs]
    df = RPS.create_scene_aggregators(all_mappings, pdm_score_df, PROPOSAL_SAMPLING)
    df = RPS.compute_final_scores(df)
    final = df.copy()
    score_cols = [c for c in df.columns
                  if ((any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
                      and c != "pdm_score")]
    g, s1, s2 = RPS.calculate_individual_mapping_scores(df[score_cols + ["token", "weight"]], all_mappings)
    return final, {"EPDMS": float(g["score"]), "stage1": {k: float(v) for k, v in s1.items()},
                   "stage2": {k: float(v) for k, v in s2.items()}, "group": {k: float(v) for k, v in g.items()}}


def write_like_wrapper(final: pd.DataFrame, path: str) -> None:
    d = final.copy()
    if "frame_type" in d.columns:
        d["frame_type"] = d["frame_type"].astype(str)
    d.to_csv(path, index=False)


def compare_frames(mine: str, banked: str) -> dict:
    a = pd.read_csv(mine)
    b = pd.read_csv(banked)
    out = {"rows": [len(a), len(b)], "columns_equal": list(a.columns) == list(b.columns), "per_column": {}}
    a = a.set_index("token")
    b = b.set_index("token").loc[a.index]
    worst = 0.0
    for c in b.columns:
        if c not in a.columns:
            out["per_column"][c] = "MISSING"
            worst = float("inf")
            continue
        if pd.api.types.is_numeric_dtype(b[c]) and pd.api.types.is_numeric_dtype(a[c]):
            x, y = a[c].to_numpy(np.float64), b[c].to_numpy(np.float64)
            nan_eq = bool((np.isnan(x) == np.isnan(y)).all())
            d = float(np.nanmax(np.abs(x - y))) if (~np.isnan(x)).any() else 0.0
            out["per_column"][c] = {"max_abs_diff": d, "nan_pattern_equal": nan_eq}
            worst = max(worst, d if nan_eq else float("inf"))
        else:
            eq = bool((a[c].astype(str) == b[c].astype(str)).all())
            out["per_column"][c] = {"all_equal": eq}
            worst = max(worst, 0.0 if eq else float("inf"))
    out["max_abs_diff_any_column"] = worst
    out["token_sets_equal"] = sorted(a.index) == sorted(b.index)
    return out


def csv_roundtrip(x: float) -> float:
    import io
    b = io.StringIO()
    pd.DataFrame({"v": [x]}).to_csv(b, index=False)
    b.seek(0)
    return float(pd.read_csv(b)["v"].iloc[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="G0")
    a = ap.parse_args()
    inp = json.load(open(C.INPUTS, encoding="utf-8"))
    mapping = inp["reactive_all_mapping"]
    banked = C.FRAME.format(arm="R7_A1")
    order = pd.read_csv(banked)["token"].tolist()
    g0, g0_bad = load_scored("G0")
    if len(g0) != C.N_TOK or set(g0) != set(order):
        raise SystemExit(f"G0 rows {len(g0)} (non-OK {len(g0_bad)}), expected {C.N_TOK}: refusing")
    out_path = os.path.join(RAW, "p4_agg.json")
    res = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    for arm in [x for x in a.arms.split(",") if x]:
        if arm == "G0":
            src = g0
            changed = set()
        else:
            picks = json.load(open(os.path.join(RAW, f"p4_picks_{arm}.json"), encoding="utf-8"))["picks"]
            changed = {t for t, r in picks.items() if r["changed"]}
            sc, bad = load_scored(arm)
            miss = sorted(changed - set(sc))
            if miss:
                raise SystemExit(f"{arm}: {len(miss)} changed picks without an OK exact row (non-OK {len(bad)}): refusing")
            pz = np.load(os.path.join(RAW, f"p4_plans_{arm}.npz"), allow_pickle=False)
            psha = {str(t): __import__("hashlib").sha256(np.ascontiguousarray(p).tobytes()).hexdigest()[:16]
                    for t, p in zip(pz["token"], np.asarray(pz["poses"], np.float32))}
            stale = [t for t in changed if sc[t]["plan_sha16"] != psha[t]]
            if stale:
                raise SystemExit(f"{arm}: {len(stale)} scored rows are of a DIFFERENT plan than the current picks: refusing")
            src = {t: (sc[t] if t in changed else g0[t]) for t in order}
        rows = [to_row_df(src[t]) for t in order]
        final, agg = aggregate(rows, mapping)
        fpath = os.path.join(RAW, f"p4_frame_{arm}.csv")
        write_like_wrapper(final, fpath)
        agg["n_changed"] = len(changed)
        agg["frame"] = os.path.relpath(fpath, C.PKGD6)
        if arm == "G0":
            cmp_ = compare_frames(fpath, banked)
            summ = pd.read_csv(C.SUMMARY.format(arm="R7_A1"))
            official = float(summ.loc[summ["token"] == "extended_pdm_score_combined", "score"].iloc[0])
            agg["identity_control"] = {"frame_vs_banked": cmp_, "EPDMS_official_banked": official,
                                       "EPDMS_rebuilt": agg["EPDMS"], "EPDMS_abs_diff": abs(agg["EPDMS"] - official),
                                       # the banked summary is a CSV: compare at the SAME writer's precision (round-trip through to_csv)
                                       "EPDMS_rebuilt_via_csv_writer": csv_roundtrip(agg["EPDMS"]),
                                       "PASS": bool(cmp_["max_abs_diff_any_column"] == 0.0 and cmp_["token_sets_equal"]
                                                    and csv_roundtrip(agg["EPDMS"]) == official)}
        res[arm] = agg
        json.dump(res, open(out_path, "w"), indent=1)
        print(arm, json.dumps({k: agg[k] for k in ("EPDMS", "n_changed")}),
              ("" if arm != "G0" else json.dumps({"PASS": agg["identity_control"]["PASS"],
                                                  "max_abs": agg["identity_control"]["frame_vs_banked"]["max_abs_diff_any_column"]})),
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
