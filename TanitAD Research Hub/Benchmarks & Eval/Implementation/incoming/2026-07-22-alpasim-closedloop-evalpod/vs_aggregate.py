#!/usr/bin/env python3
"""Paired flagship-v1 vs REF-C-base closed-loop suite aggregation (on NuRec reconstructions).

Reads each model's AlpaSim runtime aggregate/results-summary.json, matches rollouts by
clipgt_id (paired within-sim), computes per-model aggregates + paired flagship-minus-REF-C
deltas with a paired scene-cluster bootstrap CI and an exact sign/McNemar test.

Usage: vs_aggregate.py <flag_logdir> <refc_logdir> <out_json> [canon_flag] [canon_refc]
"""
import json, sys, math, random

def load(logdir):
    d = json.load(open(f"{logdir}/aggregate/results-summary.json"))
    out = {}
    for r in d["rollouts"]:
        m = r["metrics"]
        clip = r.get("clipgt_id") or m.get("clipgt_id")
        # keep the LAST rollout per clip (a clean rebuilt logdir has exactly one)
        out[clip] = {
            "clip": clip,
            "rollout_id": r.get("rollout_id") or m.get("rollout_id"),
            "run_uuid": r.get("run_uuid") or m.get("run_uuid"),
            "status": r["status"],
            "score": float(r["score"]),
            "collision_at_fault": float(m["collision_at_fault"]),
            "collision_any": float(m.get("collision_any", 0.0)),
            "collision_front": float(m.get("collision_front", 0.0)),
            "offroad": float(m["offroad"]),
            "dist_to_gt_trajectory": float(m["dist_to_gt_trajectory"]),
            "progress": float(m.get("progress", 0.0)),
            "progress_rel": float(m.get("progress_rel", 0.0)),
            "plan_deviation": float(m.get("plan_deviation", 0.0)),
            "dist_traveled_m": float(m.get("dist_traveled_m", 0.0)),
            "duration_frac_20s": float(m.get("duration_frac_20s", 0.0)),
            "img_is_black": float(m.get("img_is_black", 0.0)),
        }
    return out

def agg(rows):
    n = len(rows)
    f = lambda k: sum(r[k] for r in rows) / n
    return {
        "n_scenes": n,
        "at_fault_collision_rate": f("collision_at_fault"),
        "collision_any_rate": f("collision_any"),
        "offroad_rate": f("offroad"),
        "pass_rate": sum(1 for r in rows if r["status"] == "pass") / n,
        "n_pass": sum(1 for r in rows if r["status"] == "pass"),
        "mean_score": f("score"),
        "mean_dist_to_gt": f("dist_to_gt_trajectory"),
        "mean_progress": f("progress"),
        "mean_progress_rel": f("progress_rel"),
        "mean_plan_deviation": f("plan_deviation"),
        "mean_dist_traveled_m": f("dist_traveled_m"),
        "img_is_black_rate": f("img_is_black"),
    }

def boot_ci(deltas, B=20000, seed=0):
    random.seed(seed); n = len(deltas)
    means = []
    for _ in range(B):
        s = sum(deltas[random.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * B)]; hi = means[int(0.975 * B)]
    return lo, hi

def binom_two_sided_p(k, n, p=0.5):
    # exact two-sided binomial p-value (sign/McNemar test on discordant pairs)
    if n == 0:
        return 1.0
    from math import comb
    pmf = [comb(n, i) * p**i * (1-p)**(n-i) for i in range(n+1)]
    obs = pmf[k]
    return min(1.0, sum(pi for pi in pmf if pi <= obs + 1e-12))

def main():
    flag_dir, refc_dir, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    canon_flag = sys.argv[4] if len(sys.argv) > 4 else ""
    canon_refc = sys.argv[5] if len(sys.argv) > 5 else ""
    res = sys.argv[6] if len(sys.argv) > 6 else "480x854"
    F = load(flag_dir); R = load(refc_dir)
    clips = sorted(set(F) & set(R))
    assert clips, "no shared clips"
    frows = [F[c] for c in clips]; rrows = [R[c] for c in clips]
    Fa = agg(frows); Ra = agg(rrows)

    # paired per-scene
    dscore = [F[c]["score"] - R[c]["score"] for c in clips]
    ddist  = [F[c]["dist_to_gt_trajectory"] - R[c]["dist_to_gt_trajectory"] for c in clips]
    score_lo, score_hi = boot_ci(dscore)
    dist_lo, dist_hi = boot_ci(ddist)

    # paired pass (McNemar): discordant pairs
    flag_pass_refc_fail = sum(1 for c in clips if F[c]["status"]=="pass" and R[c]["status"]=="fail")
    refc_pass_flag_fail = sum(1 for c in clips if R[c]["status"]=="pass" and F[c]["status"]=="fail")
    both_pass = sum(1 for c in clips if F[c]["status"]=="pass" and R[c]["status"]=="pass")
    both_fail = sum(1 for c in clips if F[c]["status"]=="fail" and R[c]["status"]=="fail")
    disc = flag_pass_refc_fail + refc_pass_flag_fail
    pass_p = binom_two_sided_p(min(flag_pass_refc_fail, refc_pass_flag_fail), disc)

    # paired at-fault collision: scenes where one collides at-fault and the other doesn't
    flag_col = lambda c: F[c]["collision_at_fault"] >= 0.5
    refc_col = lambda c: R[c]["collision_at_fault"] >= 0.5
    flag_col_refc_ok = sum(1 for c in clips if flag_col(c) and not refc_col(c))
    refc_col_flag_ok = sum(1 for c in clips if refc_col(c) and not flag_col(c))
    both_col = sum(1 for c in clips if flag_col(c) and refc_col(c))
    neither_col = sum(1 for c in clips if not flag_col(c) and not refc_col(c))
    coldisc = flag_col_refc_ok + refc_col_flag_ok
    col_p = binom_two_sided_p(min(flag_col_refc_ok, refc_col_flag_ok), coldisc)

    # sign test on score (scenes strictly better for flagship)
    flag_better = sum(1 for d in dscore if d > 1e-9)
    refc_better = sum(1 for d in dscore if d < -1e-9)
    ties = sum(1 for d in dscore if abs(d) <= 1e-9)
    sign_p = binom_two_sided_p(min(flag_better, refc_better), flag_better + refc_better)

    out = {
        "run": "flagship v1 (WM+tactical policy) vs REF-C base (open-loop anchored diffusion) — paired closed-loop suite",
        "label": "on NuRec reconstructions (AlpaSim NuRec, tanitad-eval A40)",
        "framing_MANDATORY": ("WITHIN-SIM RELATIVE comparison, NOT an absolute model measure. Both models are fed the "
            "SAME NuRec-reconstruction input, which is ~3.2x more OOD than REF-C's real-footage training "
            "(open-loop ADE 1.47 on these reconstructions vs 0.4728 on real PhysicalAI val, RUN_RECIPE s13). "
            "The paired design controls for scene + reconstruction fidelity, so the flagship-minus-REF-C "
            "delta isolates the PLANNER (WM+tactical-policy vs open-loop diffusion); it does NOT give either "
            "model's real-world closed-loop rate."),
        "n_scenes": len(clips),
        "one_rollout_per_scene": True,
        "resolution": res,
        "canon_flagship": canon_flag,
        "canon_refc": canon_refc,
        "models": {
            "flagship_v1": {"ckpt": "flagship-30k step 29999 (HF Sayood/tanitad-flagship-4b-speedjerk)", **Fa},
            "refc_base":   {"ckpt": "refc-base-30k step 29999 (104.2M, 128 anchors)", **Ra},
        },
        "paired_flagship_minus_refc": {
            "n_scenes": len(clips),
            "mean_score_delta": sum(dscore)/len(dscore),
            "mean_score_delta_boot95": [score_lo, score_hi],
            "mean_dist_to_gt_delta": sum(ddist)/len(ddist),
            "mean_dist_to_gt_delta_boot95": [dist_lo, dist_hi],
            "score_sign_test": {"flag_better": flag_better, "refc_better": refc_better,
                                "ties": ties, "two_sided_p": sign_p},
            "pass_mcnemar": {"flag_pass_refc_fail": flag_pass_refc_fail,
                             "refc_pass_flag_fail": refc_pass_flag_fail,
                             "both_pass": both_pass, "both_fail": both_fail, "two_sided_p": pass_p},
            "at_fault_collision_mcnemar": {"flag_collides_refc_ok": flag_col_refc_ok,
                             "refc_collides_flag_ok": refc_col_flag_ok,
                             "both_collide": both_col, "neither_collides": neither_col,
                             "two_sided_p": col_p},
        },
        "per_scene": [
            {"clip": c,
             "flag": {k: F[c][k] for k in ("rollout_id","run_uuid","status","score","collision_at_fault","offroad","dist_to_gt_trajectory")},
             "refc": {k: R[c][k] for k in ("rollout_id","run_uuid","status","score","collision_at_fault","offroad","dist_to_gt_trajectory")},
             "score_delta": F[c]["score"] - R[c]["score"]}
            for c in clips
        ],
    }
    json.dump(out, open(out_json, "w"), indent=2)

    # console summary
    print("=== PAIRED SUITE (n=%d, on NuRec reconstructions, %s) ===" % (len(clips), res))
    print("%-14s %8s %8s %8s %9s %9s" % ("metric","flagship","refc","delta","",""))
    print("at_fault_col   %8.3f %8.3f %8.3f" % (Fa["at_fault_collision_rate"], Ra["at_fault_collision_rate"], Fa["at_fault_collision_rate"]-Ra["at_fault_collision_rate"]))
    print("offroad        %8.3f %8.3f %8.3f" % (Fa["offroad_rate"], Ra["offroad_rate"], Fa["offroad_rate"]-Ra["offroad_rate"]))
    print("pass_rate      %8.3f %8.3f %8.3f  (%d vs %d of %d)" % (Fa["pass_rate"], Ra["pass_rate"], Fa["pass_rate"]-Ra["pass_rate"], Fa["n_pass"], Ra["n_pass"], len(clips)))
    print("mean_score     %8.3f %8.3f %8.3f  boot95 [%.3f, %.3f]" % (Fa["mean_score"], Ra["mean_score"], out["paired_flagship_minus_refc"]["mean_score_delta"], score_lo, score_hi))
    print("mean_dist_gt   %8.3f %8.3f %8.3f  boot95 [%.3f, %.3f]" % (Fa["mean_dist_to_gt"], Ra["mean_dist_to_gt"], out["paired_flagship_minus_refc"]["mean_dist_to_gt_delta"], dist_lo, dist_hi))
    print("PAIRED pass McNemar: flag>refc=%d, refc>flag=%d, both_pass=%d, both_fail=%d, p=%.3f" % (flag_pass_refc_fail, refc_pass_flag_fail, both_pass, both_fail, pass_p))
    print("PAIRED at-fault-collision: flag_col&refc_ok=%d, refc_col&flag_ok=%d, both=%d, neither=%d, p=%.3f" % (flag_col_refc_ok, refc_col_flag_ok, both_col, neither_col, col_p))
    print("score sign test: flag_better=%d refc_better=%d ties=%d p=%.3f" % (flag_better, refc_better, ties, sign_p))
    print("wrote", out_json)

if __name__ == "__main__":
    main()
