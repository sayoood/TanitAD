#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score + PAIR closed-loop re-eval logdirs (AlpaSim's own offroad/collision
metrics from each rollout's metrics.parquet). Same score_rollout logic as
gate1_postprocess.py. Prints per-scene + aggregate per tag, and the PAIRED
per-scene offroad/pass delta between two tags.

Run: CUDA_VISIBLE_DEVICES="" .venv/bin/python gate1_score.py \
       base:/workspace/gate1_reeval_base ft:/workspace/gate1_reeval_ft
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
import polars as pl

CATEGORY = {
    "00169207": "intersection", "0810968d": "intersection", "41c06176": "intersection",
    "59cb0598": "intersection", "69efe005": "intersection", "780ece49": "intersection",
    "8b04d54e": "intersection",
    "3cc29c99": "roundabout", "471f2484": "roundabout", "6dcd2117": "roundabout",
    "bc843fa0": "roundabout", "fd3a49fa": "roundabout", "adb72a39": "roundabout",
    "c3d4065e": "roundabout", "d3267951": "roundabout",
}


def parquet_scalars(pq):
    df = pl.read_parquet(pq)
    if "valid" in df.columns:
        df = df.filter(pl.col("valid"))
    scal = {}
    for name in df["name"].unique().to_list():
        sub = df.filter(pl.col("name") == name).sort("timestamps_us")
        v = sub["values"].to_numpy().astype(float)
        if len(v) == 0:
            continue
        agg = sub["time_aggregation"][0]
        scal[name] = float({"max": v.max(), "min": v.min(), "mean": v.mean(),
                            "sum": v.sum(), "first": v[0]}.get(agg, v[-1]))
    return scal


def score_rollout(m):
    caf = 1.0 if (m.get("collision_front", 0) >= 0.5 or m.get("collision_lateral", 0) >= 0.5) else 0.0
    off = 1.0 if m.get("offroad", 0) >= 0.5 else 0.0
    passed = (caf == 0.0 and off == 0.0)
    prog = m.get("progress", m.get("progress_rel_to_total", 0.0))
    gtd = m.get("gt_dist_traveled_m", 999.0)
    pscore = 1.0 if gtd < 5.0 else min(max(min(prog, 1.0), 0.0) / 0.8, 1.0)
    return {"collision_at_fault": caf, "offroad": off,
            "status": "pass" if passed else "fail",
            "score": round(pscore if passed else 0.0, 4),
            "dist_to_gt": round(m.get("dist_to_gt_trajectory", 0.0), 3),
            "plan_dev": round(m.get("plan_deviation", 0.0), 3),
            "progress": round(prog, 3)}


def score_logdir(logdir):
    out = {}
    for d in sorted(glob.glob(f"{logdir}/rollouts/clipgt-*/*/")):
        scene8 = d.rstrip("/").split("/")[-2].replace("clipgt-", "")[:8]
        pq = os.path.join(d, "metrics.parquet")
        if not os.path.exists(pq):
            continue
        try:
            out[scene8] = score_rollout(parquet_scalars(pq))
        except Exception as e:
            out[scene8] = {"error": str(e)}
    return out


def agg(scored):
    rows = [v for v in scored.values() if "offroad" in v]
    n = len(rows)
    if n == 0:
        return {}
    return {"n": n,
            "offroad": sum(r["offroad"] for r in rows),
            "offroad_rate": round(sum(r["offroad"] for r in rows) / n, 3),
            "at_fault": sum(r["collision_at_fault"] for r in rows),
            "at_fault_rate": round(sum(r["collision_at_fault"] for r in rows) / n, 3),
            "pass": sum(1 for r in rows if r["status"] == "pass"),
            "pass_rate": round(sum(1 for r in rows if r["status"] == "pass") / n, 3),
            "mean_score": round(float(np.mean([r["score"] for r in rows])), 4),
            "mean_plan_dev": round(float(np.mean([r["plan_dev"] for r in rows])), 3),
            "mean_dist_to_gt": round(float(np.mean([r["dist_to_gt"] for r in rows])), 3)}


def cat_agg(scored):
    o = {}
    for c in ("intersection", "roundabout"):
        sub = {k: v for k, v in scored.items() if CATEGORY.get(k) == c}
        o[c] = agg(sub)
    return o


def main():
    tags = {}
    for arg in sys.argv[1:]:
        tag, logdir = arg.split(":", 1)
        tags[tag] = score_logdir(logdir)
    scenes = sorted(set().union(*[set(s) for s in tags.values()]))
    tlist = list(tags)

    print("=== PER-SCENE (offroad / at-fault-collision / pass) ===")
    hdr = "scene8    cat          " + "  ".join(f"{t:>22s}" for t in tlist)
    print(hdr)
    for s in scenes:
        cells = []
        for t in tlist:
            r = tags[t].get(s, {})
            if "offroad" in r:
                cells.append(f"off={int(r['offroad'])} caf={int(r['collision_at_fault'])} {r['status']:4s} pd={r['plan_dev']:.1f}")
            else:
                cells.append("--")
        print(f"{s}  {CATEGORY.get(s,'?'):12s} " + "  ".join(f"{c:>22s}" for c in cells))

    print("\n=== AGGREGATE ===")
    result = {"tags": {}, "paired": None}
    for t in tlist:
        a = agg(tags[t]); ca = cat_agg(tags[t])
        result["tags"][t] = {"overall": a, "per_category": ca}
        print(f"[{t}] overall: offroad {a.get('offroad')}/{a.get('n')} "
              f"({a.get('offroad_rate')})  at_fault {a.get('at_fault')}/{a.get('n')} "
              f"({a.get('at_fault_rate')})  pass {a.get('pass')}/{a.get('n')} "
              f"({a.get('pass_rate')})  mean_score {a.get('mean_score')}  "
              f"mean_plan_dev {a.get('mean_plan_dev')}")
        for c, d in ca.items():
            if d:
                print(f"     {c:12s} offroad {d['offroad']}/{d['n']} ({d['offroad_rate']}) "
                      f"at_fault {d['at_fault']}/{d['n']} pass {d['pass']}/{d['n']} "
                      f"plan_dev {d['mean_plan_dev']}")

    # paired delta between the first two tags
    if len(tlist) >= 2:
        a, b = tlist[0], tlist[1]
        common = [s for s in scenes if "offroad" in tags[a].get(s, {}) and "offroad" in tags[b].get(s, {})]
        d_off = sum(tags[b][s]["offroad"] - tags[a][s]["offroad"] for s in common)
        newly_on = [s for s in common if tags[b][s]["offroad"] > tags[a][s]["offroad"]]
        newly_off = [s for s in common if tags[b][s]["offroad"] < tags[a][s]["offroad"]]
        d_pass = sum((tags[b][s]["status"] == "pass") - (tags[a][s]["status"] == "pass") for s in common)
        result["paired"] = {
            "a": a, "b": b, "n_common": len(common),
            "offroad_a": sum(tags[a][s]["offroad"] for s in common),
            "offroad_b": sum(tags[b][s]["offroad"] for s in common),
            "offroad_delta_b_minus_a": d_off,
            "scenes_newly_offroad": newly_off_list(newly_on),
            "scenes_recovered_onroad": newly_off_list(newly_off),
            "pass_delta_b_minus_a": d_pass,
        }
        print(f"\n=== PAIRED ({b} - {a}) over {len(common)} scenes ===")
        print(f"  offroad: {int(result['paired']['offroad_a'])} -> {int(result['paired']['offroad_b'])} "
              f"(delta {int(d_off):+d})")
        print(f"  scenes newly OFFROAD (FT worse): {newly_on}")
        print(f"  scenes RECOVERED on-road (FT better): {newly_off}")
        print(f"  pass delta: {int(d_pass):+d}")
    json.dump(result, open("/workspace/gate1_reeval_scores.json", "w"), indent=2)
    print("\nwrote /workspace/gate1_reeval_scores.json")


def newly_off_list(xs):
    return list(xs)


if __name__ == "__main__":
    main()
