#!/usr/bin/env python3
"""E1 controls C1, C3-C8 (+ P1, the M1 mutation) over the banked devkit outputs. See SPEC.md §4.

Every expectation is derived INDEPENDENTLY of the devkit code that produced the value:
  * the EPDMS formula comes from docs/metrics.md (C4), not from compute_final_scores;
  * counts and the stage-1/stage-2 token sets come from the yaml (C3);
  * the aggregate is recomputed by this file from per-token rows + Gaussian weights that this
    file recomputes from endpoints/start points (C5);
  * C1's values are literals from the metric definitions (binary {0,1}, NC/DDC {0,.5,1}).
Usage: python verify_controls.py --raw <raw dir> --out <controls.json>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TTS = "C:/Users/Admin/navsim-crun/devkit/navsim/planning/script/config/common/train_test_split/"
M8 = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
      "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
      "history_comfort"]
EC = "two_frame_extended_comfort"
SHORT = dict(zip(M8 + [EC], ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]))
W = {"ego_progress": 5.0, "time_to_collision_within_bound": 5.0, "lane_keeping": 2.0, "history_comfort": 2.0, EC: 2.0}
SUMMARY = ["extended_pdm_score_stage_one", "extended_pdm_score_stage_two", "extended_pdm_score_combined"]
SIGMA2 = 0.1                                               # scene_aggregator.py:18


def load_csv(raw: Path, arm: str) -> pd.DataFrame | None:
    c = sorted((raw / arm).glob("devkit_*.csv"))
    if not c:
        return None
    return pd.read_csv(c[-1], index_col=0)


def hooks(raw: Path, arm: str) -> list:
    p = raw / arm / f"{arm}_hooks.json"
    return json.load(open(p))["pdm_score_calls"] if p.exists() else []


def epdms_formula(r: dict, suffix: str = "") -> float:
    """docs/metrics.md: prod(NC,DAC,DDC,TLC) * sum(w*m)/sum(w); EC dropped (/14) when NaN
    (run_pdm_score_one_stage.py:177-189 zeroes EC's weight for a row with no adjacent frame)."""
    g = lambda k: float(r[k + suffix])                                    # noqa: E731
    prod = g("no_at_fault_collisions") * g("drivable_area_compliance") * g("driving_direction_compliance") * g("traffic_light_compliance")
    num = den = 0.0
    for k, w in W.items():
        v = g(k)
        if k == EC and math.isnan(v):
            continue
        num += w * v
        den += w
    return prod * num / den


def c4(df: pd.DataFrame, two_stage: bool) -> dict:
    rows = df[~df["token"].isin(SUMMARY + ["average_all_frames"]) & (df["valid"].astype(str) == "True")]
    worst, n, n_nan_ec = 0.0, 0, 0
    for _, r in rows.iterrows():
        if two_stage:
            suffix = "_stage_one" if not pd.isna(r.get("ego_progress_stage_one")) else "_stage_two"
        else:
            suffix = ""
        n_nan_ec += int(pd.isna(r[EC + suffix]))
        d = abs(epdms_formula(r, suffix) - float(r["score"]))
        worst, n = max(worst, d), n + 1
    return {"n_rows": n, "n_rows_with_nan_EC_dropped_to_14": n_nan_ec, "max_abs_diff": worst,
            "tol": 1e-9, "pass": bool(n > 0 and worst <= 1e-9)}


def per_token(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    cols = [c + suffix for c in M8 + [EC]] + ["score"]
    rows = df[~df["token"].isin(SUMMARY + ["average_all_frames"])].set_index("token")
    rows = rows[[c for c in cols if c in rows.columns]].dropna(how="all", subset=[c + suffix for c in M8])
    rows.columns = [c.replace(suffix, "") if suffix else c for c in rows.columns]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default="warmup_two_stage")
    ap.add_argument("--alias", action="append", default=[], help="ROLE=dir, e.g. A1=N1 (navhard CV arm dir)")
    ap.add_argument("--external-ref", type=float, default=18.5356, help="published/LB CV combined EPDMS x100")
    ap.add_argument("--external-tol", type=float, default=0.01, help="half of the reference's last printed digit")
    ap.add_argument("--external-src", default="HF warmup LB baseline_constant_velocity, INHERITED via NAVSIM_PROTOCOL.md §6.3")
    a = ap.parse_args()
    alias = dict(x.split("=", 1) for x in a.alias)
    sf = yaml.safe_load(open(TTS + f"scene_filter/{a.split}.yaml"))
    tts = yaml.safe_load(open(TTS + f"{a.split}.yaml"))
    S1, S2 = set(sf["tokens"]), set(sf["reactive_synthetic_initial_tokens"])
    mapping = tts["reactive_all_mapping"]
    out = {"split": a.split, "alias": alias,
           "expected": {"n_stage_one": len(S1), "n_stage_two": len(S2), "n_mapping_entries": len(mapping)}}
    A1, R1, A2, A1b, A2b, M1 = (load_csv(a.raw, alias.get(x, x)) for x in ("A1", "R1", "A2", "A1b", "A2b", "M1"))
    a1dir = alias.get("A1", "A1")

    # ---------------- C3 counts (A1) ----------------
    if A1 is not None:
        body = A1[~A1["token"].isin(SUMMARY)]
        s1 = body[~body["ego_progress_stage_one"].isna()]
        s2 = body[~body["ego_progress_stage_two"].isna()]
        out["C3_counts_A1"] = {
            "n_rows": len(body), "n_stage_one_rows": len(s1), "n_stage_two_rows": len(s2),
            "stage_one_tokens_eq_yaml": set(s1["token"]) == S1, "stage_two_tokens_eq_yaml": set(s2["token"]) == S2,
            "n_valid": int((body["valid"].astype(str) == "True").sum()),
            "summary_rows_present": [t for t in SUMMARY if t in set(A1["token"])],
            "summary_valid": {t: str(A1.loc[A1["token"] == t, "valid"].iloc[0]) for t in SUMMARY if t in set(A1["token"])}}
        c = out["C3_counts_A1"]
        c["pass"] = bool(len(s1) == len(S1) and len(s2) == len(S2) and c["stage_one_tokens_eq_yaml"]
                         and c["stage_two_tokens_eq_yaml"] and c["n_valid"] == len(S1) + len(S2)
                         and len(c["summary_rows_present"]) == 3 and all(v == "True" for v in c["summary_valid"].values()))

    # ---------------- C4 per-token identity ----------------
    out["C4_formula"] = {}
    for name, df, two in (("A1", A1, True), ("R1", R1, True), ("A2", A2, True), ("A1b", A1b, False),
                          ("A2b", A2b, False), ("M1", M1, False)):
        if df is not None and "score" in df.columns:
            out["C4_formula"][name] = c4(df, two)

    # ---------------- C5 aggregate identity (A1, from the compute_final_scores dump) ----------------
    fdump = a.raw / a1dir / f"{a1dir}_final_scores_frame.csv"
    if A1 is not None and fdump.exists():
        F = pd.read_csv(fdump).set_index("token")
        cols = M8 + [EC, "score"]
        wmax, s1_rows, s2_groups, comb = 0.0, [], [], []
        for orig, prev, pairs in mapping:
            nows, prevs = [p[0] for p in pairs], [p[1] for p in pairs]
            def wts(first, toks):
                d2 = [(F.loc[first, "endpoint_x"] - F.loc[t, "start_point_x"]) ** 2 +
                      (F.loc[first, "endpoint_y"] - F.loc[t, "start_point_y"]) ** 2 for t in toks]
                w = np.exp(-np.asarray(d2) / (2 * SIGMA2))
                return np.full(len(toks), 1.0 / len(toks)) if (np.isclose(w.sum(), 0.0) or np.isnan(w.sum())) else w / w.sum()
            w_now, w_prev = wts(orig, nows), wts(prev, prevs)
            wmax = max(wmax, float(np.abs(w_now - F.loc[nows, "weight"].to_numpy()).max()),
                       float(np.abs(w_prev - F.loc[prevs, "weight"].to_numpy()).max()))
            g1s2 = (F.loc[nows, cols].to_numpy() * w_now[:, None]).sum(0)
            g2s2 = (F.loc[prevs, cols].to_numpy() * w_prev[:, None]).sum(0)
            g1s1, g2s1 = F.loc[orig, cols].to_numpy(dtype=float), F.loc[prev, cols].to_numpy(dtype=float)
            s1_rows += [g1s1, g2s1]
            s2_groups += [g1s2, g2s2]
            comb.append((g1s1 * g1s2 + g2s1 * g2s2) / 2)
        mine = {"stage_one": np.mean(s1_rows, 0), "stage_two": np.mean(s2_groups, 0), "combined": np.mean(comb, 0)}
        diffs = {}
        for key, tok, suf in (("stage_one", SUMMARY[0], "_stage_one"), ("stage_two", SUMMARY[1], "_stage_two")):
            row = A1[A1["token"] == tok].iloc[0]
            for i, cname in enumerate(cols):
                col = "score" if cname == "score" else cname + suf
                diffs[f"{key}.{SHORT.get(cname, cname)}"] = abs(float(row[col]) - float(mine[key][i]))
        diffs["combined.score"] = abs(float(A1[A1["token"] == SUMMARY[2]].iloc[0]["score"]) - float(mine["combined"][-1]))
        worst = max(diffs.values())
        out["C5_aggregate_A1"] = {"max_abs_weight_diff": wmax, "max_abs_summary_diff": worst,
                                  "recomputed": {k: dict(zip([SHORT.get(c, c) for c in cols], map(float, v)))
                                                 for k, v in mine.items()},
                                  "tol": 1e-9, "pass": bool(wmax <= 1e-9 and worst <= 1e-9)}

    # ---------------- C7 determinism / order invariance (A1 vs R1) ----------------
    if A1 is not None and R1 is not None:
        x, y = A1.set_index("token").sort_index(), R1.set_index("token").sort_index()
        num = [c for c in x.columns if c != "valid"]
        same_index = list(x.index) == list(y.index)
        d = (x[num].astype(float) - y[num].astype(float)).abs()
        both_nan = x[num].isna() & y[num].isna()
        maxd = float(d.where(~both_nan, 0.0).max().max()) if same_index else None
        nan_mismatch = int((x[num].isna() ^ y[num].isna()).sum().sum()) if same_index else None
        o1 = [h["token"] for h in hooks(a.raw, a1dir) if "token" in h]
        o2 = [h["token"] for h in hooks(a.raw, alias.get("R1", "R1")) if "token" in h]
        out["C7_determinism"] = {"same_token_set": same_index, "max_abs_diff": maxd, "nan_pattern_mismatch": nan_mismatch,
                                 "n_calls_A1": len(o1), "n_calls_R1": len(o2),
                                 "evaluation_order_differed": o1 != o2 and sorted(o1) == sorted(o2),
                                 "pass": bool(same_index and maxd == 0.0 and nan_mismatch == 0)}
        out["C7_determinism"]["teeth"] = ("order differed, so the identity also tests order-invariance"
                                          if out["C7_determinism"]["evaluation_order_differed"]
                                          else "⚠️ order did NOT differ — C7 tests rerun determinism only")

    # ---------------- C6 cross-runner (A1 stage one vs A1b) ----------------
    if A1 is not None and A1b is not None:
        t2, t1 = per_token(A1, "_stage_one"), per_token(A1b, "")
        common = sorted(set(t2.index) & set(t1.index))
        d8 = float((t2.loc[common, M8].astype(float) - t1.loc[common, M8].astype(float)).abs().max().max())
        ec_nan_b = [t for t in common if pd.isna(t1.loc[t, EC])]
        ec_ok = [t for t in common if t not in ec_nan_b]
        dec = float((t2.loc[ec_ok, EC].astype(float) - t1.loc[ec_ok, EC].astype(float)).abs().max()) if ec_ok else None
        prevs = {m[1] for m in mapping}
        out["C6_cross_runner"] = {"n_common": len(common), "max_abs_diff_8_metrics": d8,
                                  "n_EC_nan_in_one_stage": len(ec_nan_b), "EC_nan_tokens_are_prev_tokens": set(ec_nan_b) <= prevs,
                                  "max_abs_diff_EC_where_defined": dec,
                                  "pass": bool(len(common) == len(S1) and d8 == 0.0 and (dec in (0.0, None)) and set(ec_nan_b) <= prevs)}

    # ---------------- C1 human filter (A2b) + P1 + M1 teeth ----------------
    if A2b is not None:
        h = per_token(A2b, "")
        vals = {SHORT[m]: sorted(set(map(float, h[m].dropna()))) for m in M8}
        exact1 = {k: vals[k] == [1.0] for k in ("DAC", "TLC", "TTC", "LK")}
        half = {k: set(vals[k]) <= {0.5, 1.0} for k in ("NC", "DDC")}
        c1 = {"n_tokens": len(h), "value_sets": vals, "DAC_TLC_TTC_LK_exactly_1": exact1, "NC_DDC_in_half_one": half}
        c1["pass"] = bool(len(h) == len(S1) and all(exact1.values()) and all(half.values()))
        calls = [c for c in hooks(a.raw, alias.get("A2b", "A2b")) if "token" in c]
        p1 = [float(np.abs(np.asarray(c["agent_poses"]) - np.asarray(c["human_poses"])).max()) for c in calls
              if c.get("human_poses") is not None]
        c1["P1_agent_vs_cache_human_traj"] = {"n": len(p1), "max_abs_pose_diff": (max(p1) if p1 else None),
                                              "pass": bool(p1 and len(p1) == len(S1) and max(p1) < 1e-6)}
        if M1 is not None:
            m = per_token(M1, "")
            zero_cells = {SHORT[k]: sorted(m.index[m[k].astype(float) == 0.0]) for k in M8}
            fired_tokens = sorted({t for v in zero_cells.values() for t in v})
            diffs = []
            for t in h.index:
                for k in M8 + [EC]:
                    a_v, m_v = h.loc[t, k], m.loc[t, k]
                    if pd.isna(a_v) and pd.isna(m_v):
                        continue
                    if a_v != m_v:
                        diffs.append((t, SHORT[k], float(m_v), float(a_v)))
            unexplained = [d for d in diffs if not (d[2] == 0.0 and d[3] == 1.0)]
            c1["M1_mutation_filter_off"] = {
                "zero_cells_per_metric": {k: len(v) for k, v in zero_cells.items()},
                "n_tokens_where_filter_fires": len(fired_tokens), "fired_tokens": fired_tokens,
                "n_cells_changed_by_filter": len(diffs), "n_changed_cells_not_0_to_1": len(unexplained),
                "unexplained": unexplained[:20],
                "vacuous": len(fired_tokens) == 0,
                "pass": bool(len(unexplained) == 0)}
        out["C1_human_filter"] = c1

    # ---------------- C8 external (INHERITED) ----------------
    if A1 is not None:
        comb = float(A1[A1["token"] == SUMMARY[2]].iloc[0]["score"])
        out["C8_external"] = {"local_combined_x100": round(100 * comb, 6), "local_combined_raw": comb,
                              "reference": a.external_ref, "reference_source": a.external_src,
                              "delta": round(100 * comb - a.external_ref, 6), "tolerance": a.external_tol,
                              "reproduced": abs(100 * comb - a.external_ref) <= a.external_tol}

    a.out.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    for k, v in out.items():
        if isinstance(v, dict) and "pass" in v:
            print(k, "PASS" if v["pass"] else "FAIL")
        elif k == "C4_formula":
            print(k, {n: ("PASS" if r["pass"] else "FAIL") for n, r in v.items()})
    print("C8", out.get("C8_external"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
