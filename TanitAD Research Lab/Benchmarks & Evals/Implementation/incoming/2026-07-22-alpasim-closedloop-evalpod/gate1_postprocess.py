#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate-1 on-policy rollout post-processing.

Consumes the CLOSED-LOOP per-step log (`gate1_junc/preds.jsonl`, from refc_driver.py
--log-preds) + each rollout's metrics.parquet + the scene GT trajectory
(ArtifactSceneProvider), and emits the Gate-1 dataset:

  gate1_rollouts.jsonl  one JSON line per (scene, step): the ON-POLICY visited state, the
                        issued plan, the aligned GT reference (nearest-corridor projection +
                        expert lookahead = the RECOVERY SIGNAL), and the plan-vs-executed
                        divergence decomposition.
  gate1_summary.json    per-scene + per-category + overall aggregates, departure-onset
                        characterization, and the plan-vs-executed divergence PATTERN, with
                        parquet cross-validation.

Signals collected (per the Gate-1 brief):
  1. VISITED STATE SEQUENCE  — the closed-loop on-policy poses (x,y,yaw,speed) the planner
     actually reaches (the distribution open-loop training omits).
  2. EXECUTED + ALIGNED GT per step — executed pose vs (a) nearest-corridor GT projection
     (cross-track error + recovery vector in the rig frame) and (b) GT expert lookahead
     (where the recorded expert goes from near this off-policy state) = the recovery target.
  3. PLAN-vs-EXECUTED DIVERGENCE over the rollout — short-horizon (controller tracking),
     long-horizon (plan vision vs reality), plan churn (replan instability), and the
     plan-corridor-vs-executed-corridor comparison that tests "plan on-road, ego departs".

Frames: the driver-logged ego poses and the GT trajectory are BOTH in the scene world frame
(same USDZ, Gate-0 verified) so executed-vs-GT is directly comparable; cross-validated against
the parquet's own dist_to_gt_trajectory / plan_deviation. Nearest-corridor uses GT vertices
(~0.25 m spacing -> +-0.13 m). WITHIN-SIM RELATIVE, ~3.2x-OOD (RUN_RECIPE s13).

Run (alpasim venv, no GPU needed):
  CUDA_VISIBLE_DEVICES="" /workspace/alpa-invest/alpasim/.venv/bin/python \
    /workspace/gate1_postprocess.py
"""
from __future__ import annotations
import glob
import json
import os
from collections import defaultdict

import numpy as np
import polars as pl

SS = ("/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets/"
      "986fec83193b1baf3d5121f09462e248")
LOGDIR = "/workspace/gate1_junc"
PREDS = f"{LOGDIR}/preds.jsonl"
OUT_JSONL = f"{LOGDIR}/gate1_rollouts.jsonl"
OUT_SUM = f"{LOGDIR}/gate1_summary.json"
HOR = [0.5, 1.0, 1.5, 2.0]
XTE_ONSET_M = 1.0        # departure-onset threshold on executed cross-track error

CATEGORY = {  # the 15 junction scenes (scaled_suite_labels.json)
    "00169207": "intersection", "0810968d": "intersection", "41c06176": "intersection",
    "59cb0598": "intersection", "69efe005": "intersection", "780ece49": "intersection",
    "8b04d54e": "intersection",
    "3cc29c99": "roundabout", "471f2484": "roundabout", "6dcd2117": "roundabout",
    "bc843fa0": "roundabout", "fd3a49fa": "roundabout", "adb72a39": "roundabout",
    "c3d4065e": "roundabout", "d3267951": "roundabout",
}

# ---------------------------------------------------------------- GT loading
_gt_cache: dict[str, dict] = {}
_prov = None


def load_gt(scene_full: str):
    global _prov
    if scene_full in _gt_cache:
        return _gt_cache[scene_full]
    if _prov is None:
        from alpasim_runtime.scene_loader import ArtifactSceneProvider
        _prov = ArtifactSceneProvider.from_path(SS, smooth_trajectories=True)
    ds = _prov.get_data_source(f"clipgt-{scene_full}")
    tr = ds.rig.trajectory
    g = {"pos": np.asarray(tr.positions)[:, :2].astype(float),
         "t": np.asarray(tr.timestamps_us).astype(float),
         "yaw": np.asarray(tr.yaws).astype(float)}
    _gt_cache[scene_full] = g
    return g


# ---------------------------------------------------------------- geometry
def world_to_rig(px, py, x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = px - x, py - y
    return float(c * dx + s * dy), float(-s * dx + c * dy)   # (fwd, left)


def nearest_gt(gt, x, y):
    d = np.hypot(gt["pos"][:, 0] - x, gt["pos"][:, 1] - y)
    j = int(np.argmin(d))
    return j, float(d[j])


def gt_at_time(gt, T):
    return (float(np.interp(T, gt["t"], gt["pos"][:, 0])),
            float(np.interp(T, gt["t"], gt["pos"][:, 1])))


# ---------------------------------------------------------------- parquet
def parquet_series(pq):
    df = pl.read_parquet(pq)
    if "valid" in df.columns:
        df = df.filter(pl.col("valid"))
    series, scal = {}, {}
    for name in df["name"].unique().to_list():
        sub = df.filter(pl.col("name") == name).sort("timestamps_us")
        t = sub["timestamps_us"].to_numpy().astype(float)
        v = sub["values"].to_numpy().astype(float)
        if len(v) == 0:
            continue
        series[name] = (t, v)
        agg = sub["time_aggregation"][0]
        scal[name] = float({"max": v.max(), "min": v.min(), "mean": v.mean(),
                            "sum": v.sum(), "first": v[0]}.get(agg, v[-1]))
    return series, scal


def score_rollout(m):
    caf = 1.0 if (m.get("collision_front", 0) >= 0.5 or m.get("collision_lateral", 0) >= 0.5) else 0.0
    off = 1.0 if m.get("offroad", 0) >= 0.5 else 0.0
    passed = (caf == 0.0 and off == 0.0)
    prog = m.get("progress", m.get("progress_rel_to_total", 0.0))
    gtd = m.get("gt_dist_traveled_m", 999.0)
    pscore = 1.0 if gtd < 5.0 else min(max(min(prog, 1.0), 0.0) / 0.8, 1.0)
    return {"collision_at_fault": caf, "collision_any": float(m.get("collision_any", 0.0) >= 0.5),
            "offroad": off, "status": "pass" if passed else "fail",
            "score": round(pscore if passed else 0.0, 4),
            "dist_to_gt_trajectory": round(m.get("dist_to_gt_trajectory", 0.0), 4),
            "plan_deviation": round(m.get("plan_deviation", 0.0), 4),
            "progress": round(prog, 4),
            "min_dist_to_lane_boundary_m": round(m.get("min_distance_to_lane_boundary_m", -1), 4),
            "wrong_lane": round(m.get("wrong_lane", 0.0), 4)}


# ---------------------------------------------------------------- session -> scene
def scene_map():
    m = {}
    for d in glob.glob(f"{LOGDIR}/rollouts/clipgt-*/*/"):
        p = d.rstrip("/").split("/")
        roll, scene_full = p[-1], p[-2].replace("clipgt-", "")
        pq = os.path.join(d, "metrics.parquet")
        m[roll] = {"scene8": scene_full[:8], "scene_full": scene_full,
                   "rollout_dir": d.rstrip("/"), "parquet": pq if os.path.exists(pq) else None}
    return m


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else None


def sround(x, n=4):
    return None if x is None else round(x, n)


# ---------------------------------------------------------------- main
def main():
    smap = scene_map()
    # index preds by session
    by_sess = defaultdict(list)
    n_bad = 0
    for ln in open(PREDS):
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
            by_sess[r["session"]].append(r)
        except Exception:
            n_bad += 1
    print(f"preds sessions={len(by_sess)} bad_lines={n_bad} rollout_dirs={len(smap)}")

    fout = open(OUT_JSONL, "w")
    scenes_out = []
    for sess, rows in by_sess.items():
        info = smap.get(sess)
        if info is None:
            print(f"  WARN session {sess[:12]} has no rollout dir -> skip")
            continue
        scene8, scene_full = info["scene8"], info["scene_full"]
        cat = CATEGORY.get(scene8, "?")
        try:
            gt = load_gt(scene_full)
        except Exception as e:
            print(f"  WARN GT load failed {scene8}: {e}")
            gt = None
        # parquet outcomes
        series, scal = ({}, {})
        if info["parquet"]:
            try:
                series, scal = parquet_series(info["parquet"])
            except Exception as e:
                print(f"  WARN parquet failed {scene8}: {e}")
        outcome = score_rollout(scal) if scal else {}

        rows.sort(key=lambda r: r["t"])
        # dedupe identical timestamps (keep last)
        seen = {}
        for r in rows:
            seen[r["t"]] = r
        rows = [seen[k] for k in sorted(seen)]
        if len(rows) < 3 or gt is None:
            print(f"  skip {scene8}: steps={len(rows)} gt={'ok' if gt else 'none'}")
            continue
        ex_t = np.array([r["t"] for r in rows], dtype=float)
        ex_x = np.array([r["x"] for r in rows], dtype=float)
        ex_y = np.array([r["y"] for r in rows], dtype=float)
        ex_yaw = np.array([r["yaw"] for r in rows], dtype=float)
        ex_sp = np.array([r.get("speed", 0.0) for r in rows], dtype=float)
        t0 = ex_t[0]

        # plans in world frame per step: [n,4,2]
        plans_w = []
        for r in rows:
            x, y, yaw = r["x"], r["y"], r["yaw"]
            c, s = np.cos(yaw), np.sin(yaw)
            pw = []
            for (dx, dy) in r["pred_rig"]:
                pw.append([x + c * dx - s * dy, y + s * dx + c * dy])
            plans_w.append(pw)
        plans_w = np.array(plans_w)   # [n,4,2]

        def exec_at(T):
            return (float(np.interp(T, ex_t, ex_x)), float(np.interp(T, ex_t, ex_y)))

        def plan_eval(i, T):
            # plan i as a trajectory: anchor (ex_t[i], ego_i) + 4 waypoints at ex_t[i]+HOR
            tt = np.array([ex_t[i]] + [ex_t[i] + h * 1e6 for h in HOR])
            xx = np.array([ex_x[i]] + list(plans_w[i][:, 0]))
            yy = np.array([ex_y[i]] + list(plans_w[i][:, 1]))
            return float(np.interp(T, tt, xx)), float(np.interp(T, tt, yy))

        steps = []
        for i in range(len(rows)):
            x, y, yaw = ex_x[i], ex_y[i], ex_yaw[i]
            j, xte = nearest_gt(gt, x, y)                 # executed cross-track to corridor
            rec_fwd, rec_left = world_to_rig(gt["pos"][j, 0], gt["pos"][j, 1], x, y, yaw)
            # expert lookahead from the nearest corridor point (the recovery target), rig frame
            ref_look = []
            for h in HOR:
                gx, gy = gt_at_time(gt, gt["t"][j] + h * 1e6)
                ref_look.append([round(v, 3) for v in world_to_rig(gx, gy, x, y, yaw)])
            # plan waypoints' own cross-track to corridor (is the PLAN on-corridor?)
            plan_wp_xte = [float(nearest_gt(gt, plans_w[i][k, 0], plans_w[i][k, 1])[1])
                           for k in range(4)]
            # plan-vs-executed divergence at each horizon (where the ego actually went)
            ped = []
            for hi, h in enumerate(HOR):
                T = ex_t[i] + h * 1e6
                if T <= ex_t[-1]:
                    ex_p = exec_at(T)
                    ped.append(float(np.hypot(plans_w[i][hi, 0] - ex_p[0],
                                              plans_w[i][hi, 1] - ex_p[1])))
                else:
                    ped.append(None)
            # plan churn: this plan vs the next, matched on absolute future time
            churn = None
            if i + 1 < len(rows):
                cvals = []
                for h in HOR:
                    T = ex_t[i + 1] + h * 1e6
                    a = plan_eval(i, T)
                    b = plan_eval(i + 1, T)
                    cvals.append(float(np.hypot(a[0] - b[0], a[1] - b[1])))
                churn = float(np.mean(cvals))
            steps.append({
                "scene8": scene8, "category": cat, "step": i,
                "t_rel_s": round((ex_t[i] - t0) / 1e6, 3),
                "x": round(x, 3), "y": round(y, 3), "yaw": round(float(yaw), 4),
                "speed": round(float(ex_sp[i]), 3),
                "exec_xte_m": round(xte, 4),
                "recovery_rig_fwd_left": [round(rec_fwd, 3), round(rec_left, 3)],
                "ref_lookahead_rig": ref_look,
                "plan_rig": [[round(dx, 3), round(dy, 3)] for dx, dy in rows[i]["pred_rig"]],
                "plan_wp_xte_m": [round(v, 3) for v in plan_wp_xte],
                "plan_mean_xte_m": round(float(np.mean(plan_wp_xte)), 4),
                "plan_exec_dev_m": [None if v is None else round(v, 4) for v in ped],
                "plan_churn_m": None if churn is None else round(churn, 4),
            })
            fout.write(json.dumps(steps[-1]) + "\n")

        # ---- per-scene aggregates
        xtes = [s["exec_xte_m"] for s in steps]
        plan_xtes = [s["plan_mean_xte_m"] for s in steps]
        ped_short = [s["plan_exec_dev_m"][0] for s in steps]   # 0.5 s (tracking)
        ped_long = [s["plan_exec_dev_m"][3] for s in steps]    # 2.0 s (vision vs reality)
        churns = [s["plan_churn_m"] for s in steps]
        # departure onset: first step exec_xte crosses threshold
        onset = next((s for s in steps if s["exec_xte_m"] > XTE_ONSET_M), None)
        # temporal split (compounding on-policy degradation): first vs last third
        third = max(1, len(steps) // 3)
        early = lambda arr: mean(arr[:third])
        late = lambda arr: mean(arr[-third:])
        # executed offroad from the parquet's OWN series (AlpaSim metric, validated)
        off_frac, off_onset_frac = None, None
        if "offroad" in series:
            _t, _v = series["offroad"]
            off = (_v >= 0.5)
            off_frac = round(float(off.mean()), 3)
            idx = np.argmax(off) if off.any() else None
            off_onset_frac = round(float(idx) / max(1, len(off) - 1), 3) if idx is not None and off.any() else None
        sc = {
            "scene8": scene8, "scene_full": scene_full, "category": cat,
            "rollout_id": sess, "n_steps": len(steps),
            "outcome": outcome,
            "exec_xte_start_m": round(xtes[0], 4),          # frame-alignment sanity (should be small)
            "exec_xte_mean_m": round(mean(xtes), 4),
            "exec_xte_max_m": round(max(xtes), 4),
            "exec_xte_early_m": sround(early(xtes)),        # first third (compounding: early << late)
            "exec_xte_late_m": sround(late(xtes)),          # last third
            "plan_mean_xte_mean_m": round(mean(plan_xtes), 4),
            "plan_mean_xte_early_m": sround(early(plan_xtes)),
            "plan_mean_xte_late_m": sround(late(plan_xtes)),
            "exec_offroad_frac": off_frac,                  # AlpaSim's OWN per-step offroad (validated)
            "exec_offroad_onset_frac": off_onset_frac,      # rollout fraction at first offroad
            "plan_exec_dev_short_mean_m": None if mean(ped_short) is None else round(mean(ped_short), 4),
            "plan_exec_dev_long_mean_m": None if mean(ped_long) is None else round(mean(ped_long), 4),
            "plan_churn_mean_m": None if mean(churns) is None else round(mean(churns), 4),
            "departure_onset": None if onset is None else {
                "step": onset["step"], "t_rel_s": onset["t_rel_s"],
                "exec_xte_m": onset["exec_xte_m"],
                "plan_mean_xte_at_onset_m": onset["plan_mean_xte_m"]},
            # cross-validation vs AlpaSim's own metrics
            "xval_parquet_dist_to_gt": outcome.get("dist_to_gt_trajectory"),
            "xval_parquet_plan_deviation": outcome.get("plan_deviation"),
        }
        scenes_out.append(sc)
        print(f"  {scene8} {cat:12s} steps={len(steps):3d} pass={outcome.get('status','?'):4s} "
              f"exec_xte mean={sc['exec_xte_mean_m']:.2f} max={sc['exec_xte_max_m']:.2f} "
              f"plan_xte={sc['plan_mean_xte_mean_m']:.2f} churn={sc['plan_churn_mean_m']} "
              f"onset={'-' if onset is None else onset['t_rel_s']}")
    fout.close()

    # ---- category + overall aggregation
    def agg(rows):
        if not rows:
            return {}
        n = len(rows)
        return {
            "n_scenes": n,
            "n_steps": int(sum(r["n_steps"] for r in rows)),
            "pass_rate": round(sum(1 for r in rows if r["outcome"].get("status") == "pass") / n, 3),
            "at_fault_collision_rate": round(sum(r["outcome"].get("collision_at_fault", 0) for r in rows) / n, 3),
            "offroad_rate": round(sum(r["outcome"].get("offroad", 0) for r in rows) / n, 3),
            "exec_xte_mean_m": sround(mean([r["exec_xte_mean_m"] for r in rows])),
            "exec_xte_max_m": sround(mean([r["exec_xte_max_m"] for r in rows])),
            "exec_xte_early_m": sround(mean([r["exec_xte_early_m"] for r in rows])),
            "exec_xte_late_m": sround(mean([r["exec_xte_late_m"] for r in rows])),
            "exec_offroad_frac": sround(mean([r["exec_offroad_frac"] for r in rows]), 3),
            "plan_mean_xte_mean_m": sround(mean([r["plan_mean_xte_mean_m"] for r in rows])),
            "plan_mean_xte_early_m": sround(mean([r["plan_mean_xte_early_m"] for r in rows])),
            "plan_mean_xte_late_m": sround(mean([r["plan_mean_xte_late_m"] for r in rows])),
            "plan_exec_dev_short_mean_m": sround(mean([r["plan_exec_dev_short_mean_m"] for r in rows])),
            "plan_exec_dev_long_mean_m": sround(mean([r["plan_exec_dev_long_mean_m"] for r in rows])),
            "plan_churn_mean_m": sround(mean([r["plan_churn_mean_m"] for r in rows])),
            "n_with_departure_onset": int(sum(1 for r in rows if r["departure_onset"])),
            "median_onset_t_rel_s": (round(float(np.median(
                [r["departure_onset"]["t_rel_s"] for r in rows if r["departure_onset"]])), 3)
                if any(r["departure_onset"] for r in rows) else None),
        }

    cats = {}
    for c in ("intersection", "roundabout"):
        cats[c] = agg([r for r in scenes_out if r["category"] == c])
    summary = {
        "run": "Gate-1 on-policy rollout collection — REF-C-base closed-loop, junction scenes",
        "framing_MANDATORY": ("WITHIN-SIM RELATIVE, on NuRec reconstructions (~3.2x OOD, "
                              "RUN_RECIPE s13). Divergence PATTERN is the deliverable; absolute "
                              "metres are within-sim, not real-world."),
        "model": "REF-C base (/root/models/refc-base-30k/ckpt.pt, step 29999, 128 anchors)",
        "sceneset": SS, "resolution": "480x854", "control_hz": 5, "warmup_s": 3.0,
        "n_scenes": len(scenes_out),
        "signal_defs": {
            "exec_xte_m": "executed pose -> nearest GT corridor vertex (on-policy departure magnitude; ~= parquet dist_to_gt_trajectory)",
            "plan_mean_xte_m": "REF-C plan's 4 waypoints -> nearest GT corridor (is the PLAN on-corridor?)",
            "plan_exec_dev_short_m": "||plan@0.5s - executed@+0.5s|| (controller tracking)",
            "plan_exec_dev_long_m": "||plan@2.0s - executed@+2.0s|| (plan vision vs reality under replanning)",
            "plan_churn_m": "||plan_i - plan_{i+1}|| matched on absolute future time (replan instability)",
            "recovery_rig_fwd_left": "vector from ego to nearest GT corridor point, rig frame (recovery direction)",
            "ref_lookahead_rig": "GT expert path 0.5/1/1.5/2s ahead of the corridor projection, rig frame (recovery TARGET / DAgger label)",
        },
        "overall": agg(scenes_out),
        "per_category": cats,
        "per_scene": scenes_out,
    }
    json.dump(summary, open(OUT_SUM, "w"), indent=2)
    ov = summary["overall"]
    print("\n=== GATE-1 SUMMARY ===")
    print(f"scenes={ov.get('n_scenes')} steps={ov.get('n_steps')} pass_rate={ov.get('pass_rate')} "
          f"offroad_rate={ov.get('offroad_rate')} caf={ov.get('at_fault_collision_rate')}")
    print(f"exec_xte mean={ov.get('exec_xte_mean_m')} max={ov.get('exec_xte_max_m')} | "
          f"plan_xte mean={ov.get('plan_mean_xte_mean_m')}")
    print(f"plan_exec_dev short={ov.get('plan_exec_dev_short_mean_m')} long={ov.get('plan_exec_dev_long_mean_m')} | "
          f"churn={ov.get('plan_churn_mean_m')} | median_onset_s={ov.get('median_onset_t_rel_s')}")
    for c, d in cats.items():
        if d:
            print(f"  {c:12s} n={d['n_scenes']} pass={d['pass_rate']} off={d['offroad_rate']} "
                  f"exec_xte={d['exec_xte_mean_m']} plan_xte={d['plan_mean_xte_mean_m']} "
                  f"churn={d['plan_churn_mean_m']}")
    print("wrote", OUT_JSONL, OUT_SUM)


if __name__ == "__main__":
    main()
