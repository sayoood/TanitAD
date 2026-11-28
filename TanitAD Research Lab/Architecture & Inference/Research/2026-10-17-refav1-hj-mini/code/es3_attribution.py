"""PREREG_RA1_ES Addendum A: the paired attribution deltas at the 250-step budget, from each run's eval/perwindow.npz.

usage: es3_attribution.py <dir with R0_s0.npz ... A1x_b0.08g50_s0.npz> <out.json>

Every delta is the paired episode-cluster bootstrap (B = 2000, seed 0) over the SAME 2,399 eval-141 windows (asserted), the estimator
`ra1_hjepa_eval.py verdict` uses. Readings are Addendum A's, in its order."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "taniteval"))

import numpy as np  # noqa: E402

from taniteval import ci  # noqa: E402

RUNS = ("R0_s0", "R0_s1", "A0_b0.08_s0", "A0_b0.08_s1", "A1_b0.08g50_s0", "A1_b0.08g50_s1", "A1x_b0.08g50_s0")


def pair(a, b, eid):
    r = ci.paired_episode_cluster_bootstrap(np.asarray(a, np.float64), np.asarray(b, np.float64), eid, n_boot=2000, seed=0)
    return {k: r[k] for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "estimator") if k in r}


def main(d, out):
    z = {r: np.load(os.path.join(d, r + ".npz"), allow_pickle=True) for r in RUNS}
    ref = z["R0_s0"]
    for r, x in z.items():
        assert np.array_equal(x["eid"], ref["eid"]) and np.array_equal(x["t"], ref["t"]), f"{r}: eval windows differ"
    eid, p6 = ref["eid"], ref["p10_ade6"]
    a = {r: z[r]["ade6"] for r in RUNS}
    res = dict(schema="ra1-es3-attribution-v1", prereg="PREREG_RA1_ES.md Addendum A", budget_steps=250, n_windows=int(len(eid)),
               estimator="paired episode-cluster bootstrap B=2000 seed 0 (EPISODE variance only)",
               ade6={r: float(v.mean()) for r, v in a.items()}, p10_ade6=float(p6.mean()), per_seed={})
    for s in (0, 1):
        res["per_seed"][f"seed{s}"] = dict(A0_minus_P10=pair(a[f"A0_b0.08_s{s}"], p6, eid), A1_minus_P10=pair(a[f"A1_b0.08g50_s{s}"], p6, eid),
                                           A1_minus_A0=pair(a[f"A1_b0.08g50_s{s}"], a[f"A0_b0.08_s{s}"], eid),
                                           A0_minus_R0=pair(a[f"A0_b0.08_s{s}"], a[f"R0_s{s}"], eid),
                                           A1_minus_R0=pair(a[f"A1_b0.08g50_s{s}"], a[f"R0_s{s}"], eid))
    res["A1x_minus_A0_seed0"] = pair(a["A1x_b0.08g50_s0"], a["A0_b0.08_s0"], eid)
    res["seed_floors"] = {arm: abs(float(a[f"{arm}_s0"].mean()) - float(a[f"{arm}_s1"].mean())) for arm in ("R0", "A0_b0.08", "A1_b0.08g50")}
    ps = res["per_seed"]
    sep_neg = lambda d: bool(d["separated"] and d["delta"] < 0)                   # noqa: E731
    r1 = all(sep_neg(ps[k]["A0_minus_P10"]) and not ps[k]["A1_minus_A0"]["separated"] for k in ps)
    r2 = all(not ps[k]["A0_minus_P10"]["separated"] and sep_neg(ps[k]["A1_minus_A0"]) for k in ps)
    if r1:
        res["reading"] = "READING 1: neither IDM nor temporal order is needed -- a WM loss acts as a regulariser (H-RA1-ES-2 supported)"
    elif r2:
        res["reading"] = ("READING 2: the IDM term carries the gain (A0 ~ P10, A1 < A0 separated at both seeds); A1x keeps it through its "
                          "gradient into the adapter. Next: an IDM-only arm (no prediction loss)")
    else:
        res["reading"] = "READING 3: UNRESOLVED -- both deltas and the seed floor are reported; next: a seed-2 replicate of the discordant arm"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for k, v in ps.items():
        print(k, {n: "%+.4f [%+.4f, %+.4f]%s" % (x["delta"], x["lo"], x["hi"], " SEP" if x["separated"] else "") for n, x in v.items()})
    x = res["A1x_minus_A0_seed0"]
    print("A1x - A0 s0: %+.4f [%+.4f, %+.4f]%s" % (x["delta"], x["lo"], x["hi"], " SEP" if x["separated"] else ""))
    print("floors", res["seed_floors"])
    print(res["reading"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
