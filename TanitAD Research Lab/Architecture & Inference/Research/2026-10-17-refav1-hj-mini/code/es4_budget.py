"""PREREG_RA1_ES Addendum B (ES-4): A1 b0.08g50's own budget, selected on the HOLDOUT; eval-141 reported, NOT read as K.

usage: es4_budget.py <es dir holding st250/st500/st1000 verdicts + pw/> <out.json>
  st250 numbers come from es2_verdict.json (ES-2) + pw/A1_*.npz; st500 / st1000 from st<S>/es2_verdict.json + pw/st<S>_*.npz.
Paired episode-cluster bootstrap (B = 2000, seed 0) on the SAME 2,399 eval-141 windows (asserted)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..")), "taniteval"))

import numpy as np  # noqa: E402

from taniteval import ci  # noqa: E402

TIE_M = 0.01


def pair(a, b, eid):
    r = ci.paired_episode_cluster_bootstrap(np.asarray(a, np.float64), np.asarray(b, np.float64), eid, n_boot=2000, seed=0)
    return {k: r[k] for k in ("delta", "lo", "hi", "separated") if k in r}


def main(d, out):
    pw = os.path.join(d, "pw")
    paths = {250: {"v": os.path.join(d, "es2_verdict.json"), "pre": ""}}
    for S in (500, 1000):
        paths[S] = {"v": os.path.join(d, f"st{S}", "es2_verdict.json"), "pre": f"st{S}_"}
    res = dict(schema="ra1-es4-budget-v1", prereg="PREREG_RA1_ES.md Addendum B", cell="b0.08g50",
               estimator="paired episode-cluster bootstrap B=2000 seed 0 (EPISODE variance only)", budgets={})
    ref = None
    for S, p in paths.items():
        v = json.load(open(p["v"], encoding="utf-8"))
        hold = v["holdout"]["ade6_mean_by_run"]
        z = {r: np.load(os.path.join(pw, p["pre"] + r + ".npz"), allow_pickle=True)
             for r in ("A1_b0.08g50_s0", "A1_b0.08g50_s1", "A1x_b0.08g50_s0", "R0_s0", "R0_s1")}
        for r, x in z.items():
            if ref is None:
                ref = x
            assert np.array_equal(x["eid"], ref["eid"]) and np.array_equal(x["t"], ref["t"]), f"{S} {r}: eval windows differ"
        eid, p6 = ref["eid"], ref["p10_ade6"]
        a1 = [z[f"A1_b0.08g50_s{s}"]["ade6"] for s in (0, 1)]
        res["budgets"][str(S)] = dict(
            A1_holdout_ade6=[hold["A1:b0.08g50@0"], hold["A1:b0.08g50@1"]],
            A1_holdout_seed_mean=(hold["A1:b0.08g50@0"] + hold["A1:b0.08g50@1"]) / 2,
            A1x_holdout_ade6=hold.get("A1x:b0.08g50@0"),
            A1_eval_ade6=[float(x.mean()) for x in a1],
            A1_minus_P10_NOT_K=[pair(x, p6, eid) for x in a1],
            A1_minus_R0=[pair(a1[s], z[f"R0_s{s}"]["ade6"], eid) for s in (0, 1)],
            A1x_minus_A1_seed0=pair(z["A1x_b0.08g50_s0"]["ade6"], a1[0], eid),
            A1_seed_floor=abs(float(a1[0].mean()) - float(a1[1].mean())),
            gate_verdict=v["A1x_control"]["verdict"])
    hm = {int(S): b["A1_holdout_seed_mean"] for S, b in res["budgets"].items()}
    m = min(hm.values())
    res["S_star_A1"] = min(S for S, x in hm.items() if x <= m + TIE_M)
    res["holdout_seed_mean_by_budget"] = hm
    res["rule"] = f"lowest seed-mean A1 holdout ADE@6s; ties within {TIE_M} m -> the shorter budget"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for S, b in res["budgets"].items():
        k = b["A1_minus_P10_NOT_K"]
        g = b["A1x_minus_A1_seed0"]
        print(f"{S:>5} hold {b['A1_holdout_seed_mean']:.4f} | eval A1-P10 " + " / ".join("%+.4f [%+.4f, %+.4f]%s" % (x["delta"], x["lo"], x["hi"], " SEP" if x["separated"] else "") for x in k)
              + f" | A1x-A1 {g['delta']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]{' SEP' if g['separated'] else ''} | floor {b['A1_seed_floor']:.4f}")
    print("S*_A1 =", res["S_star_A1"], hm)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
