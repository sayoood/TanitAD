#!/usr/bin/env python3
"""GATE-0 aggregation: REF-C-base WITH the free floor (floor-ON) vs WITHOUT (floor-OFF),
per category, paired. Reuses the VALIDATED scaled_aggregate2 scoring (caf=front|lateral,
pass=caf==0&offroad==0, progress score) for exact comparability with the baseline suite.
Pre-registered reading = OFF-ROAD departure, focus intersection + roundabout.
Usage: gate0_aggregate.py <on_logdir> <off_logdir> <labels_json> <out_json>
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
        if len(vals) == 0: continue
        agg = sub["time_aggregation"][0]
        try:
            out[name] = {"max": float(vals.max()), "min": float(vals.min()), "mean": float(vals.mean()),
                         "sum": float(vals.sum()), "first": float(vals[0])}.get(agg, float(vals[-1]))
        except Exception: pass
    return out

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
            "progress": round(prog, 4), "plan_deviation": round(m.get("plan_deviation", 0.0), 4)}

def load(logdir):
    out = {}
    for scene in glob.glob(f"{logdir}/rollouts/clipgt-*"):
        clip8 = os.path.basename(scene).replace("clipgt-", "")[:8]
        rolls = sorted(glob.glob(f"{scene}/*/metrics.parquet"), key=os.path.getmtime)
        if not rolls: continue
        out[clip8] = {"clip8": clip8, "rollout_dir": os.path.dirname(rolls[0]).split("/")[-1],
                      **score_rollout(rollout_scalars(rolls[0]))}
    return out

def agg(rows):
    n = len(rows); f = lambda k: sum(r[k] for r in rows) / n if n else 0.0
    return {"n": n, "at_fault_collision_rate": round(f("collision_at_fault"), 4),
            "offroad_rate": round(f("offroad"), 4),
            "pass_rate": round(sum(1 for r in rows if r["status"] == "pass") / n, 4) if n else 0,
            "n_pass": sum(1 for r in rows if r["status"] == "pass"), "mean_score": round(f("score"), 4),
            "mean_dist_to_gt": round(f("dist_to_gt_trajectory"), 4),
            "mean_plan_deviation": round(f("plan_deviation"), 4)}

def boot(ds, B=20000, seed=0):
    if not ds: return [None, None]
    random.seed(seed); n = len(ds)
    m = sorted(sum(ds[random.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(m[int(0.025 * B)], 4), round(m[int(0.975 * B)], 4)]

def binom(k, n):
    if n == 0: return 1.0
    pmf = [comb(n, i) * 0.5 ** n for i in range(n + 1)]
    return round(min(1.0, sum(p for p in pmf if p <= pmf[k] + 1e-12)), 4)

def paired(ON, OFF, clips):
    """ON minus OFF. score delta + OFFROAD delta (the pre-registered metric)."""
    ds = [ON[c]["score"] - OFF[c]["score"] for c in clips]
    do = [ON[c]["offroad"] - OFF[c]["offroad"] for c in clips]         # <0 = floor removed offroad
    onb = sum(1 for d in ds if d > 1e-9); offb = sum(1 for d in ds if d < -1e-9)
    # offroad flips
    fixed = sum(1 for c in clips if OFF[c]["offroad"] == 1 and ON[c]["offroad"] == 0)   # off->on-road
    broke = sum(1 for c in clips if OFF[c]["offroad"] == 0 and ON[c]["offroad"] == 1)   # on->off-road
    return {"n": len(clips),
            "mean_score_delta_ON_minus_OFF": round(sum(ds) / len(ds), 4) if ds else None,
            "score_delta_boot95": boot(ds),
            "mean_offroad_delta_ON_minus_OFF": round(sum(do) / len(do), 4) if do else None,
            "offroad_delta_boot95": boot(do),
            "offroad_fixed_off_to_onroad": fixed, "offroad_broke_onroad_to_off": broke,
            "offroad_mcnemar_p": binom(min(fixed, broke), fixed + broke),
            "score_sign_test": {"ON_better": onb, "OFF_better": offb, "ties": len(ds) - onb - offb,
                                "p": binom(min(onb, offb), onb + offb)}}

def main():
    ondir, offdir, labels_json, out_json = sys.argv[1:5]
    ON, OFF = load(ondir), load(offdir)
    labels = {v["clip"][:8]: v["category"] for v in json.load(open(labels_json)).values()}
    clips = sorted(set(ON) & set(OFF))
    cats = {}
    for cat in sorted(set(labels.get(c) for c in clips)):
        cc = [c for c in clips if labels.get(c) == cat]
        if cc:
            cats[cat] = {"clips": cc, "floor_on": agg([ON[c] for c in cc]),
                         "floor_off": agg([OFF[c] for c in cc]), "paired": paired(ON, OFF, cc)}
    junction = [c for c in clips if labels.get(c) in ("intersection", "roundabout")]
    out = {"run": "GATE-0 free inference-time floor: REF-C-base WITH floor (ON) vs WITHOUT (OFF)",
           "floor": "cost-guided selection over 128 denoised anchors (argmax conf - lam*offroad - mu*coll) + road-boundary safety clamp; lam=5.0 mu=0.0 clamp=0.75m",
           "label": "on NuRec reconstructions (AlpaSim A40, 480x854, 1 rollout/scene)",
           "framing_MANDATORY": "WITHIN-SIM RELATIVE, ~3.2x-OOD (RUN_RECIPE s13); per-category RANKING trustworthy, absolute rates not real-world. Off-road cost geometry validated (gate0_cost_validation.json).",
           "training": "ZERO (inference-time only)",
           "n_scenes_paired": len(clips),
           "overall": {"floor_on": agg([ON[c] for c in clips]), "floor_off": agg([OFF[c] for c in clips]),
                       "paired": paired(ON, OFF, clips)},
           "junction_intersection_plus_roundabout": {
               "clips": junction, "floor_on": agg([ON[c] for c in junction]),
               "floor_off": agg([OFF[c] for c in junction]), "paired": paired(ON, OFF, junction)},
           "per_category": cats,
           "per_scene": [{"clip8": c, "category": labels.get(c),
                          "off": OFF[c], "on": ON[c],
                          "offroad_delta": ON[c]["offroad"] - OFF[c]["offroad"],
                          "score_delta": round(ON[c]["score"] - OFF[c]["score"], 4)} for c in clips]}
    json.dump(out, open(out_json, "w"), indent=2)
    # console summary
    def line(tag, d):
        return ("%-16s n=%2d | OFF offroad %.2f pass %d/%d score %.3f caf %.2f | ON offroad %.2f pass %d/%d score %.3f caf %.2f | dOFFROAD %+.3f %s"
                % (tag, d["paired"]["n"], d["floor_off"]["offroad_rate"], d["floor_off"]["n_pass"], d["floor_off"]["n"],
                   d["floor_off"]["mean_score"], d["floor_off"]["at_fault_collision_rate"],
                   d["floor_on"]["offroad_rate"], d["floor_on"]["n_pass"], d["floor_on"]["n"],
                   d["floor_on"]["mean_score"], d["floor_on"]["at_fault_collision_rate"],
                   d["paired"]["mean_offroad_delta_ON_minus_OFF"], d["paired"]["offroad_delta_boot95"]))
    print("=== GATE-0 free-floor: floor-ON vs floor-OFF (n=%d paired) ===" % len(clips))
    print(line("OVERALL", out["overall"] | {"paired": out["overall"]["paired"]}))
    print(line("JUNCTION(int+rbt)", out["junction_intersection_plus_roundabout"]))
    for cat, d in cats.items():
        print(line(cat, d))
    ov = out["overall"]["paired"]; jn = out["junction_intersection_plus_roundabout"]
    print("\nOVERALL offroad delta (ON-OFF): %+.3f %s ; fixed off->on-road=%d broke=%d"
          % (ov["mean_offroad_delta_ON_minus_OFF"], ov["offroad_delta_boot95"],
             ov["offroad_fixed_off_to_onroad"], ov["offroad_broke_onroad_to_off"]))
    print("wrote", out_json)

if __name__ == "__main__":
    main()
