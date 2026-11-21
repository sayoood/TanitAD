"""E-GEOM summariser - the 2x2 table and the PAIRED cross-arm deltas.

WHY A SEPARATE SCRIPT. The ladder's own `deltas_vs_p40` pairs arms INSIDE one
cache. The geometry arms live in DIFFERENT caches (different token grids,
different d_model), so that machinery cannot reach them - and differencing two
independent bootstrap CIs is NOT a paired test. This script loads the dumped
per-row predictions and runs `taniteval.ci.paired_episode_cluster_bootstrap`
through the ladder's own `paired_delta_r2c`, which is the estimator the
pre-registration commits to.

IT ASSERTS ROW IDENTITY RATHER THAN ASSUMING IT: the eval targets `yev` and the
episode ids `eev` must be element-wise equal across the two arms being paired,
or the comparison is refused. (They should be: the rows are copied from one
banked index and the eval subset is chosen by LABEL presence, not by features -
but "should be" is how unpaired comparisons get published.)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from encloc_ladder import paired_delta_r2c  # noqa: E402

ARMS = ["wide3f", "wide1f", "refa3f", "refa1f", "squash1f"]
TARGETS = ["ego_v0", "lead_gap", "lead_closing"]
# C104's published values, for the replication gate. Source:
# .../2026-08-18-pooling-ladder-ER10/raw/er10_dino.json  arms.p40.targets
C104 = {"ego_v0": 0.71733, "lead_gap": 0.44997, "lead_closing": 0.01713}
GATE_TOL = 0.002


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="the raw/ directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="wide3f")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    raw = Path(a.raw)

    res, preds, meta = {}, {}, {}
    for arm in ARMS:
        f = raw / ("encloc_%s.json" % arm)
        if not f.exists():
            print("[sum] MISSING %s - skipping arm" % f, flush=True)
            continue
        d = json.loads(f.read_text("utf-8"))
        pool = [k for k in d["arms"]][0]
        res[arm] = d["arms"][pool]["targets"]
        meta[arm] = {"pool_kernel": d["arms"][pool].get("pool_kernel"),
                     "n_units": d["arms"][pool].get("n_units"),
                     "n_raw_features": d["arms"][pool].get("n_raw_features"),
                     "token_grid": d.get("token_grid"),
                     "d_model": d.get("d_model"),
                     "label": d.get("arm_label"),
                     "intercept_col": d.get("ridge_intercept_col"),
                     "forbidden": d.get("forbidden")}
        p = raw / ("preds_%s.pkl" % arm)
        if p.exists():
            preds[arm] = pickle.loads(p.read_bytes())

    # ---- REPLICATION GATE -------------------------------------------------
    gate = {}
    if a.baseline in res:
        for t in TARGETS:
            got = res[a.baseline][t]["r2_ceiling_mean"]
            gate[t] = {"c104": C104[t], "ours": got,
                       "abs_diff": round(abs(got - C104[t]), 6),
                       "PASSES": bool(abs(got - C104[t]) <= GATE_TOL)}
    gate_ok = all(v["PASSES"] for v in gate.values()) if gate else False
    print("[sum] REPLICATION GATE vs C104: %s" % ("PASS" if gate_ok else "FAIL"))
    for t, v in gate.items():
        print("      %-13s C104 %.5f  ours %.5f  |d| %.6f  %s"
              % (t, v["c104"], v["ours"], v["abs_diff"],
                 "ok" if v["PASSES"] else "MISMATCH"))

    # ---- the 2x2 table ----------------------------------------------------
    table = {}
    for t in TARGETS:
        table[t] = {}
        for arm in ARMS:
            if arm not in res:
                continue
            n = res[arm][t]
            ps = n["per_seed"]
            pv0 = [ps[k].get("r2_ceiling_partial_v0") for k in sorted(ps)]
            pv0 = [x for x in pv0 if x is not None]
            table[t][arm] = {
                "r2_mean": n["r2_ceiling_mean"], "r2_sd": n["r2_ceiling_sd"],
                "r2_min": n["r2_ceiling_min"], "r2_max": n["r2_ceiling_max"],
                "r2_per_seed": [ps[k]["r2_ceiling"] for k in sorted(ps)],
                "partial_v0_mean": (round(float(np.mean(pv0)), 5) if pv0
                                    else None),
                "partial_v0_per_seed": pv0 or None,
                "K1_delta_mean": n["K1_delta_mean"],
                "K1_PASSES_seed0": ps[sorted(ps)[0]]["K1_PASSES"],
                "pred_sd_over_gt_sd": ps[sorted(ps)[0]]["pred_sd_over_gt_sd"],
                "n_eval": n["n_eval"], "n_eval_clusters": n["n_eval_clusters"],
                "retained_frac_of_baseline": None}
        base = table[t].get(a.baseline)
        if base:
            for arm, row in table[t].items():
                b = base["r2_mean"]
                row["retained_frac_of_baseline"] = (
                    round(row["r2_mean"] / b, 4) if abs(b) > 1e-9 else None)

    # ---- PAIRED cross-arm deltas -----------------------------------------
    deltas = {}
    if a.baseline in preds:
        pb = preds[a.baseline]
        for t in TARGETS:
            if t not in pb["scored"]:
                continue
            yb, eb, v0b = pb["scored"][t]
            deltas[t] = {}
            for arm in ARMS:
                if arm == a.baseline or arm not in preds:
                    continue
                pa = preds[arm]
                if t not in pa["scored"]:
                    continue
                ya, ea, v0a = pa["scored"][t]
                # REFUSE rather than silently mis-pair.
                if ya.shape != yb.shape or not np.allclose(ya, yb):
                    deltas[t][arm] = {"REFUSED": "eval targets differ between "
                                                 "%s and %s - not pairable"
                                                 % (arm, a.baseline)}
                    continue
                if list(map(str, ea)) != list(map(str, eb)):
                    deltas[t][arm] = {"REFUSED": "episode ids differ - not "
                                                 "pairable"}
                    continue
                per_seed, per_seed_pv0 = [], []
                armp, basep = pa["preds"][t], pb["preds"][t]
                ka = sorted(armp)[0]
                kb = sorted(basep)[0]
                seeds = sorted(set(armp[ka]) & set(basep[kb]))
                for s in seeds:
                    d1 = paired_delta_r2c(armp[ka][s], basep[kb][s], yb, eb,
                                          a.n_boot, seed=0)
                    per_seed.append(d1)
                    d2 = paired_delta_r2c(armp[ka][s], basep[kb][s], yb, eb,
                                          a.n_boot, seed=0, z=v0b)
                    per_seed_pv0.append(d2)
                deltas[t][arm] = {
                    "vs": a.baseline, "n_seeds": len(seeds),
                    "delta_r2_per_seed": [round(d["delta"], 5)
                                          for d in per_seed],
                    "delta_r2_ci_per_seed": [[d["lo"], d["hi"]] for d in per_seed],
                    "excludes_zero_all_seeds": all(d["separated"] for d in per_seed),
                    "delta_r2_partial_v0_per_seed": [round(d["delta"], 5)
                                                     for d in per_seed_pv0],
                    "delta_r2_partial_v0_ci_per_seed":
                        [[d["lo"], d["hi"]] for d in per_seed_pv0],
                    "excludes_zero_all_seeds_partial_v0": all(
                        d["separated"] for d in per_seed_pv0),
                    "estimator": "paired_episode_cluster_bootstrap "
                                 "(taniteval.ci, via paired_delta_r2c)"}

    out = {"_evidence_class": "MEASURED (ours; E-GEOM 2x2, frozen DINOv2, "
                              "geometry varied, same banked windows)",
           "eval_tier": "T0-DIAGNOSTIC (a linear readout, NEVER driving "
                        "performance)",
           "experiment": "E-GEOM (ENCODER_LOCALISATION.md §2)",
           "baseline": a.baseline,
           "replication_gate_vs_C104": {"tol": GATE_TOL, "PASSES": gate_ok,
                                        "per_target": gate},
           "estimator": "paired episode-cluster bootstrap over the eval "
                        "episodes; per-seed spread reported AS A SPREAD, "
                        "never as a confidence interval (C109)",
           "forbidden": "overlapping_holdout_se",
           "arm_meta": meta, "table": table, "paired_deltas": deltas}
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), "utf-8")
    print("[sum] wrote %s" % a.out)

    # ---- human-readable ---------------------------------------------------
    for t in TARGETS:
        print("\n== %s ==" % t)
        print("  %-8s %9s %9s %9s %8s %6s" %
              ("arm", "r2", "sd", "partialv0", "ret.frac", "K1"))
        for arm in ARMS:
            r = table.get(t, {}).get(arm)
            if not r:
                continue
            print("  %-8s %9.5f %9.5f %9s %8s %6s" %
                  (arm, r["r2_mean"], r["r2_sd"],
                   ("%.5f" % r["partial_v0_mean"]) if r["partial_v0_mean"]
                   is not None else "-",
                   ("%.3f" % r["retained_frac_of_baseline"])
                   if r["retained_frac_of_baseline"] is not None else "-",
                   "PASS" if r["K1_PASSES_seed0"] else "fail"))
        for arm, dd in deltas.get(t, {}).items():
            if "REFUSED" in dd:
                print("  delta %s: REFUSED (%s)" % (arm, dd["REFUSED"]))
            else:
                print("  delta %-8s vs %s: %s  excl0=%s | partial-v0 %s "
                      "excl0=%s"
                      % (arm, dd["vs"], dd["delta_r2_per_seed"],
                         dd["excludes_zero_all_seeds"],
                         dd["delta_r2_partial_v0_per_seed"],
                         dd["excludes_zero_all_seeds_partial_v0"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
