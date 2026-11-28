# price_maps_check.py : control m1 of card PREREG_V8F_PROOF_ADD47 (PRICE10K). Runs on the pod (numpy, json). T0.
# For each reader arm LR16_s0, LR16_s1: the knot positions of the maps at 0.35 and 0.50 equal those of the banked 0.20 map; at every knot the values are ordered
# 0.20 <= 0.35 <= 0.50 <= position; the coverage on the check rows rises with the level and is at most level + 0.05; the tool's unit check passes; fit and check
# clips are disjoint. The identity file gives g_used = g on a grid from 0 to 150 m.
# Usage: python price_maps_check.py --q20-dir <qmap dir of card ADD46> --sweep-dir <qmap_sweep> --identity <QMAP_IDENTITY.json> --out <PRICE_MAPS_CHECK.json>
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcal_map as QM  # noqa: E402

TOL = 1e-9


def one_arm(q20, q35, q50):
    xs = [np.asarray(d["xs"], np.float64) for d in (q20, q35, q50)]
    ys = [np.asarray(d["ys"], np.float64) for d in (q20, q35, q50)]
    cov = [float(d["coverage"]["share_below_Q"]) for d in (q20, q35, q50)]
    r = {"levels": [float(d["q_level"]) for d in (q20, q35, q50)], "coverage": cov, "n_knots": [int(len(x)) for x in xs],
         "knots_equal": bool(len(xs[0]) == len(xs[1]) == len(xs[2]) and np.max(np.abs(xs[0] - xs[1])) <= TOL and np.max(np.abs(xs[0] - xs[2])) <= TOL)}
    r["levels_as_declared"] = bool(r["levels"] == [20.0, 35.0, 50.0])
    r["values_ordered"] = bool(r["knots_equal"] and np.all(ys[0] <= ys[1] + TOL) and np.all(ys[1] <= ys[2] + TOL) and np.all(ys[2] <= xs[2] + TOL))
    r["coverage_rises"] = bool(cov[0] < cov[1] < cov[2])
    r["coverage_at_most_level_plus_005"] = bool(cov[1] <= 0.40 + TOL and cov[2] <= 0.55 + TOL)
    r["unit_checks"] = bool(all(d["unit_check"]["pass"] for d in (q20, q35, q50)))
    r["fit_check_disjoint"] = bool(all(d["fit_check_clips_disjoint"] for d in (q20, q35, q50)))
    r["same_rows_file"] = bool(q20["rows_sha256"] == q35["rows_sha256"] == q50["rows_sha256"])
    r["knot_values"] = {"q20": [round(float(v), 3) for v in ys[0]], "q35": [round(float(v), 3) for v in ys[1]], "q50": [round(float(v), 3) for v in ys[2]], "positions": [round(float(v), 3) for v in xs[0]]}
    r["PASS"] = bool(all(r[k] for k in ("levels_as_declared", "knots_equal", "values_ordered", "coverage_rises", "coverage_at_most_level_plus_005", "unit_checks", "fit_check_disjoint", "same_rows_file")))
    return r


def identity_ok(path):
    xs, ys = QM.load_map(path)
    grid = np.arange(0.0, 150.0001, 0.05)
    return bool(np.max(np.abs(QM.apply_map(grid, xs, ys) - grid)) <= TOL and np.isnan(QM.apply_map(np.array([np.nan]), xs, ys)[0]))


def selftest():
    def mk(level, ys, cov):
        return {"xs": [5.0, 10.0, 20.0], "ys": ys, "q_level": level, "coverage": {"share_below_Q": cov}, "unit_check": {"pass": True}, "fit_check_clips_disjoint": True, "rows_sha256": "a"}
    good = (mk(20.0, [3.0, 6.0, 12.0], 0.19), mk(35.0, [4.0, 7.5, 15.0], 0.33), mk(50.0, [4.5, 9.0, 18.0], 0.47))
    cases = [("good", good, True),
             ("values not ordered", (good[0], mk(35.0, [2.0, 7.5, 15.0], 0.33), good[2]), False),
             ("coverage does not rise", (good[0], mk(35.0, [4.0, 7.5, 15.0], 0.18), good[2]), False),
             ("coverage above level + 0.05", (good[0], good[1], mk(50.0, [4.5, 9.0, 18.0], 0.58)), False),
             ("value above its position", (good[0], good[1], mk(50.0, [4.5, 9.0, 21.0], 0.47)), False),
             ("wrong level", (good[0], mk(30.0, [4.0, 7.5, 15.0], 0.33), good[2]), False)]
    bad = 0
    for name, (a, b, c), want in cases:
        got = one_arm(a, b, c)["PASS"]
        bad += 0 if got is want else 1
        print("selftest %-30s want %-5s got %-5s" % (name, want, got))
    other = dict(good[1]); other["xs"] = [5.0, 11.0, 20.0]
    got = one_arm(good[0], other, good[2])["PASS"]
    bad += 0 if got is False else 1
    print("selftest %-30s want False got %-5s" % ("knot positions differ", got))
    print("ZZ SELFTEST %s" % ("PASS" if bad == 0 else "FAIL"))
    return 0 if bad == 0 else 1


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--q20-dir", required=True)
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--identity", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jl = lambda p: json.load(open(p, encoding="utf-8"))
    res = {"card": "PREREG_V8F_PROOF_ADD47", "control": "m1", "tier": "T0", "arms": {}}
    for k in (0, 1):
        res["arms"]["LR16_s%d" % k] = one_arm(jl(os.path.join(a.q20_dir, "QMAP_LR16_s%d.json" % k)), jl(os.path.join(a.sweep_dir, "QMAP_LR16_s%d_q35.json" % k)),
                                              jl(os.path.join(a.sweep_dir, "QMAP_LR16_s%d_q50.json" % k)))
    res["identity_file_is_identity"] = identity_ok(a.identity)
    res["PASS"] = bool(res["identity_file_is_identity"] and all(v["PASS"] for v in res["arms"].values()))
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("PRICE-MAPS-CHECK %s coverage %s identity %s" % ("PASS" if res["PASS"] else "FAIL", {k: v["coverage"] for k, v in res["arms"].items()}, res["identity_file_is_identity"]))
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
