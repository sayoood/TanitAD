#!/usr/bin/env python3
"""Standalone per-CATEGORY paired aggregation from per-rollout metrics.parquet (recovers the
run whose runtime skipped results-summary.json because 1 scene failed a route sanity check).
Replicates AlpaSim scoring: caf = front|lateral collision (rear is not at-fault); pass = caf==0
& offroad==0; progress_score = min(clamp(progress,0,1)/0.8, 1) (=1 if gt_dist<5); score = pscore if pass.
Per scene picks the OLDEST rollout (the original clean run, not a partial re-sim).
Usage: scaled_aggregate2.py <flag_logdir> <refc_logdir> <labels_json> <out_json>
"""
import polars as pl, json, sys, os, glob, random
from math import comb

def rollout_scalars(pq):
    df = pl.read_parquet(pq)
    if "valid" in df.columns:
        df = df.filter(pl.col("valid"))
    out = {}
    for name in df["name"].unique().to_list():
        sub = df.filter(pl.col("name") == name).sort("timestamps_us")
        vals = sub["values"]
        if len(vals) == 0:
            continue
        agg = sub["time_aggregation"][0]
        try:
            if agg == "max": out[name] = float(vals.max())
            elif agg == "min": out[name] = float(vals.min())
            elif agg == "mean": out[name] = float(vals.mean())
            elif agg == "sum": out[name] = float(vals.sum())
            elif agg == "first": out[name] = float(vals[0])
            else: out[name] = float(vals[-1])   # last
        except Exception:
            pass
    return out

def score_rollout(m):
    caf = 1.0 if (m.get("collision_front", 0) >= 0.5 or m.get("collision_lateral", 0) >= 0.5) else 0.0
    off = 1.0 if m.get("offroad", 0) >= 0.5 else 0.0
    passed = (caf == 0.0 and off == 0.0)
    prog = m.get("progress", m.get("progress_rel_to_total", 0.0))
    gtd = m.get("gt_dist_traveled_m", 999.0)
    pscore = 1.0 if gtd < 5.0 else min(max(min(prog, 1.0), 0.0) / 0.8, 1.0)
    return {"collision_at_fault": caf, "collision_any": float(m.get("collision_any", 0.0) >= 0.5),
            "offroad": off, "status": "pass" if passed else "fail", "score": round(pscore if passed else 0.0, 4),
            "dist_to_gt_trajectory": round(m.get("dist_to_gt_trajectory", 0.0), 4),
            "progress": round(prog, 4), "plan_deviation": round(m.get("plan_deviation", 0.0), 4)}

def load(logdir):
    out = {}
    for scene in glob.glob(f"{logdir}/rollouts/clipgt-*"):
        clip8 = os.path.basename(scene).replace("clipgt-", "")[:8]
        rolls = glob.glob(f"{scene}/*/metrics.parquet")
        if not rolls:
            continue
        rolls.sort(key=os.path.getmtime)          # oldest = original clean run
        out[clip8] = {"clip8": clip8, "rollout_dir": os.path.dirname(rolls[0]).split("/")[-1],
                      **score_rollout(rollout_scalars(rolls[0]))}
    return out

def agg(rows):
    n = len(rows); f = lambda k: sum(r[k] for r in rows) / n if n else 0.0
    return {"n": n, "at_fault_collision_rate": round(f("collision_at_fault"), 4), "offroad_rate": round(f("offroad"), 4),
            "pass_rate": round(sum(1 for r in rows if r["status"] == "pass") / n, 4) if n else 0,
            "n_pass": sum(1 for r in rows if r["status"] == "pass"), "mean_score": round(f("score"), 4),
            "mean_dist_to_gt": round(f("dist_to_gt_trajectory"), 4), "mean_plan_deviation": round(f("plan_deviation"), 4)}

def boot(ds, B=20000, seed=0):
    if not ds: return [None, None]
    random.seed(seed); n = len(ds); m = sorted(sum(ds[random.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(m[int(0.025*B)], 4), round(m[int(0.975*B)], 4)]

def binom(k, n):
    if n == 0: return 1.0
    pmf = [comb(n, i) * 0.5**n for i in range(n+1)]
    return round(min(1.0, sum(p for p in pmf if p <= pmf[k] + 1e-12)), 4)

def paired(F, R, clips):
    ds = [F[c]["score"] - R[c]["score"] for c in clips]
    fb = sum(1 for d in ds if d > 1e-9); rb = sum(1 for d in ds if d < -1e-9)
    fp = sum(1 for c in clips if F[c]["status"]=="pass" and R[c]["status"]=="fail")
    rp = sum(1 for c in clips if R[c]["status"]=="pass" and F[c]["status"]=="fail")
    return {"n": len(clips), "mean_score_delta": round(sum(ds)/len(ds), 4) if ds else None, "score_delta_boot95": boot(ds),
            "score_sign_test": {"flag_better": fb, "refc_better": rb, "ties": len(ds)-fb-rb, "p": binom(min(fb, rb), fb+rb)},
            "pass_mcnemar": {"flag_pass_refc_fail": fp, "refc_pass_flag_fail": rp, "p": binom(min(fp, rp), fp+rp)}}

def main():
    fdir, rdir, labels_json, out_json = sys.argv[1:5]
    F, R = load(fdir), load(rdir)
    labels = {v["clip"][:8]: v["category"] for v in json.load(open(labels_json)).values()}
    clips = sorted(set(F) & set(R))
    cats = {}
    for cat in sorted(set(labels.values())):
        cc = [c for c in clips if labels.get(c) == cat]
        if cc:
            cats[cat] = {"clips": cc, "flagship_v1": agg([F[c] for c in cc]), "refc_base": agg([R[c] for c in cc]), "paired": paired(F, R, cc)}
    out = {"run": "scenario-stratified flagship-v1 vs REF-C-base — BALANCED scaled suite",
           "label": "on NuRec reconstructions (AlpaSim A40, 480x854, 1 rollout/scene)",
           "framing_MANDATORY": "WITHIN-SIM RELATIVE, ~3.2x-OOD (RUN_RECIPE s13); per-category RANKING trustworthy, absolute rates not real-world.",
           "scoring_note": "recovered from per-rollout metrics.parquet (runtime skipped results-summary.json: 1 scene 0580c069 failed a route sanity check). caf=front|lateral collision.",
           "n_scenes_total": len(clips), "n_dropped": "clip 0580c069 (traffic_light) failed route-sanity in both -> excluded",
           "overall": {"flagship_v1": agg([F[c] for c in clips]), "refc_base": agg([R[c] for c in clips]), "paired": paired(F, R, clips)},
           "per_category": cats,
           "per_scene": [{"clip8": c, "category": labels.get(c), "flag": F[c], "refc": R[c], "score_delta": round(F[c]["score"]-R[c]["score"], 4)} for c in clips]}
    json.dump(out, open(out_json, "w"), indent=2)
    print("=== SCALED SCENARIO SUITE (n=%d, recovered) ===" % len(clips))
    ov = out["overall"]
    print("OVERALL: flag pass %d/%d score %.3f | refc pass %d/%d score %.3f | dScore %.3f %s" % (
        ov["flagship_v1"]["n_pass"], ov["flagship_v1"]["n"], ov["flagship_v1"]["mean_score"],
        ov["refc_base"]["n_pass"], ov["refc_base"]["n"], ov["refc_base"]["mean_score"],
        ov["paired"]["mean_score_delta"], ov["paired"]["score_delta_boot95"]))
    for cat, d in cats.items():
        print("%-15s n=%2d | flag pass %d score %.3f off %.2f | refc pass %d score %.3f | dScore %+.3f" % (
            cat, d["flagship_v1"]["n"], d["flagship_v1"]["n_pass"], d["flagship_v1"]["mean_score"], d["flagship_v1"]["offroad_rate"],
            d["refc_base"]["n_pass"], d["refc_base"]["mean_score"], d["paired"]["mean_score_delta"]))
    print("wrote", out_json)

if __name__ == "__main__":
    main()
