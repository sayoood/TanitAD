#!/usr/bin/env python3
"""Summarise the twin re-roll probe: per arm and rank, D7 (c) from the UNCHANGED verify, the paired subset, hold stats, rollout cost,
plus the END-POSE diagnostic (is the 4 s pose nearer the twin centreline than the logged one?). Run on the pod from the frozen D7 tree:
    cd /workspace/ops/d7fix/pkg/refe && PYTHONPATH=../code $PY /workspace/ops/reroll_probe/code/analyze_probe.py /workspace/ops/reroll_probe
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import shapely

import online_labeller as OL

R = Path(sys.argv[1])
MC = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/workspace/ops/d7fix/mc_final")


def stats_rows(arm):
    out = {}
    for fp in sorted((R / f"rr_{arm}").glob("s*/reroll_stats.jsonl")):
        for ln in open(fp, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                out[(r["log_name"], r["token"], r["k"])] = r
    return out


def bank_rows(arm):
    out = {}
    for ln in open(R / f"vbank_{arm}" / "targets_rank1.jsonl", encoding="utf-8"):
        r = json.loads(ln)
        out[(r["log_name"], r["token"], int(r["aug"]["value"]))] = r
    return out


def frac(xs):
    return None if not xs else round(sum(map(bool, xs)) / len(xs), 4)


summary = {}
for arm in ("legacy", "held"):
    S, B = stats_rows(arm), bank_rows(arm)
    secs = sorted(s["rollout_s"] for s in S.values() if "rollout_s" in s)
    arm_out = {"n_rows": len(B), "n_stats": len(S), "n_error": sum("error" in s for s in S.values()),
               "rollout_s_median": round(st.median(secs), 3) if secs else None, "rollout_s_mean": round(st.mean(secs), 3) if secs else None,
               "rollout_s_p90": round(secs[int(0.9 * (len(secs) - 1))], 3) if secs else None,
               "traj_vs_bank_median_m": round(st.median(s["traj_vs_bank_max_m"] for s in S.values() if "traj_vs_bank_max_m" in s), 4),
               "goal_max_abs_diff_max": max(s.get("goal_max_abs_diff", 0.0) for s in S.values()),
               "hold_misses_total": sum((s.get("hold") or {}).get("misses", 0) for s in S.values()),
               "ranks": {}}
    for k in (1, 2, 3):
        vp = R / "results" / f"verify_{arm}_k{k}.json"
        if not vp.exists():
            arm_out["ranks"][k] = "verify missing"; continue
        V = json.load(open(vp, encoding="utf-8"))
        tw = [t for t in V["twins"] if "error" not in t]
        hi = [t for t in tw if (t["divergence_m"] or 0) >= 0.5]
        # the paired subset: the twins the ORIGINAL bank's divergence_m selected (>= 0.5), whatever the re-roll's divergence is
        paired = [t for t in tw if (S.get((t["log"], t["token"], k), {}).get("div_bank") or 0) >= 0.5]
        endpose = []
        for t in hi:
            b = B[(t["log"], t["token"], k)]
            tw_p, lg_p = OL.parts_path(MC, t["log"], t["token"], b["aug"]), OL.parts_path(MC, t["log"], t["token"], None)
            twp, lgp = OL.load_parts(tw_p), OL.load_parts(lg_p)
            end = shapely.points(np.asarray(b["traj_world"][-1][:2], dtype=np.float64))
            endpose.append(float(shapely.distance(twp[3].linestring, end)) < float(shapely.distance(lgp[3].linestring, end)))
        s = V["summary"]
        arm_out["ranks"][k] = {
            "n": len(tw), "n_div_ge_0_5": len(hi), "c_traj_closer_to_twin": frac([t["closer_to_twin"] for t in hi]),
            "c_on_bank_selected_subset": {"n": len(paired), "frac": frac([t["closer_to_twin"] for t in paired])},
            "endpose_closer_to_twin_div_ge_0_5": frac(endpose),
            "c_among_start_block_agrees": frac([t["closer_to_twin"] for t in hi if t["start_block_agrees"]]),
            "n_start_block_disagrees_div_ge_0_5": sum(not t["start_block_agrees"] for t in hi),
            "c_among_twin_lane_ge_half_logged": frac([t["closer_to_twin"] for t in hi if t["twin_centerline_len"] >= 0.5 * t["logged_centerline_len"]]),
            "mean_d_traj_to_route_k": s["teacher_traj_mean_dist_to_own_route_k"], "mean_d_traj_to_route_0": s["teacher_traj_mean_dist_to_route_0"],
            "mean_d_twin_hi": s["c_mean_d_twin"], "mean_d_logged_hi": s["c_mean_d_logged"],
            "verify_flags": {"a_ok": s["a_ok"], "b_ok": s["b_ok"], "c_ok": s["c_ok"]},
            "judged": len(hi) >= 5,
        }
    summary[arm] = arm_out

print(json.dumps(summary, indent=1))
(R / "results" / "probe_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
