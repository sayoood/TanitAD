#!/usr/bin/env python3
"""Power and false-positive simulation of the Stage 0 decision rule on SYNTHETIC worlds that carry the REAL geometry (24 logs, 21 sessions, 1,705 tokens, 400 Step A tokens, D = 256).
Evidence class: ESTIMATED.  The label noise, session heterogeneity and the planted effect are synthetic: the absolute SEs (above all M3's regret) are a property of this generator, not of REFe.
What transfers is the SHAPE: which conjunct of the rule binds, and the ratio of the detectable gap to the achieved SE.  The real read prints its own achieved SE and MDE(80 %).

    CUDA_VISIBLE_DEVICES=-1 python power_sim.py --reps 24 --workers 12 --out ../raw/power_sim.json
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage0 as S0  # noqa: E402
import synth  # noqa: E402

KAPPAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)
WORLD = dict(D=256, frac=1.0, b_hi=5.0, sig_logit=2.0, het=0.7)


def one(args):
    kappa, rep = args
    seed = 1000 * rep + int(kappa * 100) + 7
    w = synth.make_world(seed=seed, kappa=kappa, **WORLD)
    d = S0.make_design(w)
    idx = S0.boot_indices(w.S)
    a0 = S0.run_arm(w, d, S0.R0, 0, idx)
    a1 = S0.run_arm(w, d, S0.R0, 1, idx)
    dec = S0.decide(a0["gaps"], a1["gaps"], a0["swap_null"], {})
    per = {}
    for m in ("M1", "M2", "M3"):
        g = a0["gaps"][m]
        pm = dec["per_measure"][m]
        per[m] = {"gap": g["point"], "se": g["se"], "lo_bonf": g["lo_bonf"], "hi_bonf": g["hi_bonf"], "nu": a0["swap_null"][f"nu_{m}"], "seed1": a1["gaps"][m]["point"],
                  "S": pm["separated_above_swap_null"], "M": pm["magnitude_ge_delta"], "R": pm["seed_replicate_agrees"], "L": pm["leave_one_session_out_stable"], "flag": pm["gap"]}
    return {"kappa": kappa, "rep": rep, "outcome": dec["outcome"], "sub": dec.get("sub"), "per": per, "seen_M1": a0["seen_minus_oof"]["M1"]["point"],
            "level_oof": a0["levels"]["OOF"], "level_in": a0["levels"]["IN"]}


def summarise(rows):
    out = {}
    for k in KAPPAS:
        rs = [r for r in rows if r["kappa"] == k]
        if not rs:
            continue
        e = {"n": len(rs), "P_GAP_any": float(np.mean([r["outcome"] == "GAP" for r in rs])), "P_NULL_powered": float(np.mean([r["outcome"] == "NULL" and r["sub"] == "POWERED" for r in rs]))}
        for m in ("M1", "M2", "M3"):
            g = np.array([r["per"][m]["gap"] for r in rs])
            e[m] = {"mean_gap": float(np.nanmean(g)), "sd_gap": float(np.nanstd(g, ddof=1)), "mean_se": float(np.nanmean([r["per"][m]["se"] for r in rs])),
                    "P_flag": float(np.mean([r["per"][m]["flag"] for r in rs])),
                    "P_statistical": float(np.mean([r["per"][m]["S"] and r["per"][m]["R"] and r["per"][m]["L"] for r in rs])),      # without the magnitude conjunct
                    "P_S": float(np.mean([r["per"][m]["S"] for r in rs])), "P_M": float(np.mean([r["per"][m]["M"] for r in rs])),
                    "P_R": float(np.mean([r["per"][m]["R"] for r in rs])), "P_L": float(np.mean([r["per"][m]["L"] for r in rs])),
                    "mean_nu": float(np.nanmean([r["per"][m]["nu"] for r in rs]))}
        out[str(k)] = e
    return out


def mde80(summary, m, key):
    """smallest MEAN realised gap at which P(key) >= 0.8, linear interpolation along the kappa grid; None if never reached"""
    pts = [(summary[str(k)][m]["mean_gap"], summary[str(k)][m][key]) for k in KAPPAS if str(k) in summary]
    for (g0, p0), (g1, p1) in zip(pts[:-1], pts[1:]):
        if p0 < 0.8 <= p1:
            return float(g0 + (0.8 - p0) / max(p1 - p0, 1e-12) * (g1 - g0))
    return float(pts[0][0]) if pts and pts[0][1] >= 0.8 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "raw", "power_sim.json"))
    a = ap.parse_args()
    jobs = [(k, r) for k in KAPPAS for r in range(a.reps)]
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(one, jobs, chunksize=1))
    summ = summarise(rows)
    res = {"what": "power / false-positive simulation of the Stage 0 rule; ESTIMATED (synthetic, real geometry)", "world": WORLD, "kappas": KAPPAS, "reps": a.reps,
           "constants_sha256": S0.constants_sha256(), "summary": summ, "mde80_statistical": {m: mde80(summ, m, "P_statistical") for m in ("M1", "M2", "M3")},
           "mde80_full_rule": {m: mde80(summ, m, "P_flag") for m in ("M1", "M2", "M3")}, "delta": S0.DELTA, "seconds": time.time() - t0, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(S0.jsonable(res), open(a.out, "w"), indent=1)
    print(json.dumps(S0.jsonable({k: res[k] for k in ("mde80_statistical", "mde80_full_rule", "seconds")}), indent=1))
    for k in KAPPAS:
        e = summ[str(k)]
        print(f"kappa {k:4}  P(GAP)={e['P_GAP_any']:.2f}  " + "  ".join(f"{m}: gap={e[m]['mean_gap']:+.4f} se={e[m]['mean_se']:.4f} P={e[m]['P_flag']:.2f}/{e[m]['P_statistical']:.2f}" for m in ("M1", "M2", "M3")))


if __name__ == "__main__":
    main()
