#!/usr/bin/env python3
"""Per-CATEGORY paired flagship-v1 vs REF-C-base aggregation for the scaled scenario suite.
Usage: scaled_aggregate.py <flag_logdir> <refc_logdir> <labels_json> <out_json>
"""
import json, sys, random
from math import comb

def load(logdir):
    d = json.load(open(f"{logdir}/aggregate/results-summary.json"))
    out = {}
    for r in d["rollouts"]:
        m = r["metrics"]; clip = r.get("clipgt_id") or m.get("clipgt_id")
        c8 = clip.replace("clipgt-", "")[:8]
        out[c8] = {"clip": clip, "clip8": c8, "rollout_id": r.get("rollout_id") or m.get("rollout_id"),
                   "status": r["status"], "score": float(r["score"]),
                   "collision_at_fault": float(m["collision_at_fault"]), "offroad": float(m["offroad"]),
                   "dist_to_gt_trajectory": float(m["dist_to_gt_trajectory"]),
                   "progress": float(m.get("progress", 0.0)), "plan_deviation": float(m.get("plan_deviation", 0.0))}
    return out

def agg(rows):
    n = len(rows); f = lambda k: sum(r[k] for r in rows) / n if n else 0.0
    return {"n": n, "at_fault_collision_rate": round(f("collision_at_fault"), 4),
            "offroad_rate": round(f("offroad"), 4),
            "pass_rate": round(sum(1 for r in rows if r["status"] == "pass") / n, 4) if n else 0,
            "n_pass": sum(1 for r in rows if r["status"] == "pass"),
            "mean_score": round(f("score"), 4), "mean_dist_to_gt": round(f("dist_to_gt_trajectory"), 4),
            "mean_plan_deviation": round(f("plan_deviation"), 4)}

def boot_ci(deltas, B=20000, seed=0):
    if not deltas: return [None, None]
    random.seed(seed); n = len(deltas); means = []
    for _ in range(B):
        means.append(sum(deltas[random.randrange(n)] for _ in range(n)) / n)
    means.sort(); return [round(means[int(0.025*B)], 4), round(means[int(0.975*B)], 4)]

def binom_p(k, n):
    if n == 0: return 1.0
    pmf = [comb(n, i) * 0.5**n for i in range(n+1)]; obs = pmf[k]
    return round(min(1.0, sum(p for p in pmf if p <= obs + 1e-12)), 4)

def paired(F, R, clips):
    ds = [F[c]["score"] - R[c]["score"] for c in clips]
    fb = sum(1 for d in ds if d > 1e-9); rb = sum(1 for d in ds if d < -1e-9)
    fp_rf = sum(1 for c in clips if F[c]["status"]=="pass" and R[c]["status"]=="fail")
    rp_ff = sum(1 for c in clips if R[c]["status"]=="pass" and F[c]["status"]=="fail")
    return {"n": len(clips), "mean_score_delta": round(sum(ds)/len(ds), 4) if ds else None,
            "score_delta_boot95": boot_ci(ds),
            "score_sign_test": {"flag_better": fb, "refc_better": rb, "ties": len(ds)-fb-rb, "p": binom_p(min(fb, rb), fb+rb)},
            "pass_mcnemar": {"flag_pass_refc_fail": fp_rf, "refc_pass_flag_fail": rp_ff, "p": binom_p(min(fp_rf, rp_ff), fp_rf+rp_ff)}}

def main():
    flag_dir, refc_dir, labels_json, out_json = sys.argv[1:5]
    F = load(flag_dir); R = load(refc_dir)
    labels = {v["clip"][:8]: v["category"] for v in json.load(open(labels_json)).values()}
    clips = sorted(set(F) & set(R))
    cats = {}
    for cat in sorted(set(labels.values())):
        cc = [c for c in clips if labels.get(c) == cat]
        if not cc: continue
        cats[cat] = {"clips": cc, "flagship_v1": agg([F[c] for c in cc]),
                     "refc_base": agg([R[c] for c in cc]), "paired": paired(F, R, cc)}
    out = {"run": "scenario-stratified flagship-v1 vs REF-C-base — BALANCED scaled suite",
           "label": "on NuRec reconstructions (AlpaSim A40, 480x854, 1 rollout/scene)",
           "framing_MANDATORY": "WITHIN-SIM RELATIVE, ~3.2x-OOD (RUN_RECIPE s13); per-category RANKING trustworthy, absolute rates not real-world.",
           "n_scenes_total": len(clips),
           "overall": {"flagship_v1": agg([F[c] for c in clips]), "refc_base": agg([R[c] for c in clips]),
                       "paired": paired(F, R, clips)},
           "per_category": cats,
           "per_scene": [{"clip8": c, "category": labels.get(c), "flag": F[c], "refc": R[c],
                          "score_delta": round(F[c]["score"]-R[c]["score"], 4)} for c in clips]}
    json.dump(out, open(out_json, "w"), indent=2)
    print("=== SCALED SCENARIO SUITE (n=%d) ===" % len(clips))
    for cat, d in cats.items():
        print("%-15s n=%d | flag pass %d/%d score %.3f off %.2f | refc pass %d/%d score %.3f | dScore %.3f %s" % (
            cat, d["flagship_v1"]["n"], d["flagship_v1"]["n_pass"], d["flagship_v1"]["n"], d["flagship_v1"]["mean_score"],
            d["flagship_v1"]["offroad_rate"], d["refc_base"]["n_pass"], d["refc_base"]["n"], d["refc_base"]["mean_score"],
            d["paired"]["mean_score_delta"], d["paired"]["score_delta_boot95"]))
    print("wrote", out_json)

if __name__ == "__main__":
    main()
