#!/usr/bin/env python3
"""THE INFERENCE-SEED NOISE FLOOR for the p4 panel, from two banked arms that
differ in NOTHING but `--plan-seed`. iCEM is stochastic AT INFERENCE, so a
lever effect must be read against THIS, not against a paired CI alone
(CLAUDE.md / H-ESTIM-SEED-1: a separated CI from a one-seed arm is necessary,
not sufficient).

CONTROL that must read EXACTLY 0.0: the four non-planner floors (`ha`, `ha0`,
`ha0_ext`, `ol`) do not depend on the plan seed, so they must be bit-identical
across the pair. If they differ, the two arms are not on the same window grid
and nothing below is a seed measurement.  ASCII only.
"""
import json, sys

FAM = [("ADE_m", ("intervals", "metrics", "ade_dense_m", "mean")),
       ("FDE_m", ("intervals", "metrics", "fde_last_m", "mean")),
       ("LON speed_MAE", ("four_families", "longitudinal", "speed_mae_mps")),
       ("LON along_MAE", ("four_families", "longitudinal", "along_mae_m")),
       ("LON accel_MAE", ("four_families", "longitudinal", "accel_mae_mps2")),
       ("LAT heading_MAE", ("four_families", "lateral", "heading_mae_deg")),
       ("LAT yawrate_MAE", ("four_families", "lateral", "yaw_rate_mae_degps")),
       ("LAT curv_MAE", ("four_families", "lateral", "curvature_mae_1pm")),
       ("LAT cross_MAE", ("four_families", "lateral", "cross_mae_m")),
       ("TAC lat_acc", ("four_families", "tactical", "lateral_decision", "accuracy")),
       ("TAC lat_kappa", ("four_families", "tactical", "lateral_decision", "kappa")),
       ("TAC lon_acc", ("four_families", "tactical", "longitudinal_decision", "accuracy")),
       ("TAC m5_acc", ("four_families", "tactical", "maneuver_5way_collapsed", "accuracy")),
       ("TAC goalFDE_m", ("four_families", "tactical", "goal_setting", "goal_point_error_m")),
       ("TAC goalbear_deg", ("four_families", "tactical", "goal_setting", "goal_bearing_mae_deg"))]


def dig(d, ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return float("nan")
        d = d[k]
    return d


def main(pa, pb):
    A = json.load(open(pa, encoding="utf-8"))
    B = json.load(open(pb, encoding="utf-8"))
    print("A = %s  metric=%s W_KAPPA=%s seed?" % (A["arm"], A["cost"]["metric"],
                                                  A["cost"]["weights"]["W_KAPPA"]))
    print("B = %s  metric=%s W_KAPPA=%s" % (B["arm"], B["cost"]["metric"],
                                            B["cost"]["weights"]["W_KAPPA"]))
    print()
    print("CONTROL - the non-planner floors MUST be identical (same window grid):")
    ok = True
    for a in ("ha", "ha0", "ha0_ext", "ol"):
        va = dig(A["arms"][a], ("intervals", "metrics", "ade_dense_m", "mean"))
        vb = dig(B["arms"][a], ("intervals", "metrics", "ade_dense_m", "mean"))
        same = (va == vb)
        ok &= same
        print("   %-8s ADE %.4f vs %.4f  identical=%s" % (a, va, vb, same))
    print("   CONTROL %s" % ("PASSED" if ok else "FAILED - nothing below is a seed measurement"))
    print()
    print("SEED FLOOR on `cl` (the arm that MUST differ):")
    print("   %-18s %10s %10s %10s %9s" % ("metric", "seed A", "seed B", "abs diff", "rel %"))
    for name, ks in FAM:
        va, vb = dig(A["arms"]["cl"], ks), dig(B["arms"]["cl"], ks)
        try:
            d = abs(float(va) - float(vb))
            rel = 100.0 * d / max(abs(float(va)), 1e-12)
            print("   %-18s %10.5f %10.5f %10.5f %9.2f" % (name, va, vb, d, rel))
        except (TypeError, ValueError):
            print("   %-18s %10s %10s" % (name, va, vb))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
