#!/usr/bin/env python3
"""THE PER-METRIC INFERENCE-SEED FLOOR, EXTENDED to every statistic this package
quotes -- including the PER-CLASS tactical recalls and the friction-circle rate.

WHY THIS EXISTS (D-REFAV1-CG-SEEDFLOOR, and the withdrawal that sharpened it):
`seed_floor.py` printed 15 aggregate metrics. The package then quoted a PER-CLASS
recall against a PAIRED kappa's floor -- two different statistics under one loose
noun -- and had to withdraw the claim. The durable fix is that the floor table must
contain EVERY statistic that will be quoted, so no quote has to borrow a cousin's.

`kamm_over_rate` is NOT in the record; it is re-derived from the dumps with the same
`assert_feasible` the audit uses, so its floor lands in the SAME table as the rest.

CONTROL that must read exactly 0.0: the four non-planner arms (`ha`,`ha0`,`ha0_ext`,
`ol`) do not depend on the plan seed and must be bit-identical across the pair. If
they differ the two arms are not on one window grid and nothing below is a seed
measurement.  ASCII only.
"""
import glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, r"C:\Users\Admin\tanitad-wt\stack")
from tanitad.refs import feasible_decode as FD          # noqa: E402

DT, VMIN = 0.2, 2.0

FAM = [("ADE_m", ("intervals", "metrics", "ade_dense_m", "mean")),
       ("FDE_m", ("intervals", "metrics", "fde_last_m", "mean")),
       ("LON speed_MAE", ("four_families", "longitudinal", "speed_mae_mps")),
       ("LON speed_bias", ("four_families", "longitudinal", "speed_bias_mps")),
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

PERCLASS = [("lane_keep", "recall"), ("turn_left", "recall"), ("turn_right", "recall"),
            ("lane_keep", "precision"), ("turn_left", "precision"), ("turn_right", "precision")]


def dig(d, ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return float("nan")
        d = d[k]
    return d


def feas(dumpdir, key="cl", vmin=VMIN):
    out = []
    for f in sorted(glob.glob(dumpdir + "/ep*.npz")):
        z = np.load(f, allow_pickle=True)
        if key not in z.files:
            continue
        a = z[key]
        if vmin > 0.0 and "v0" in z.files:
            a = a[z["v0"] >= vmin]
        if len(a):
            out.append(a)
    if not out:
        return {}
    t = torch.from_numpy(np.concatenate(out, 0)).to(torch.float64)
    t = torch.cat([torch.zeros(t.shape[0], 1, 2, dtype=t.dtype), t], dim=1)
    r = FD.assert_feasible(t, dt=DT)
    r["_n"] = int(t.shape[0])
    return r


def row(name, va, vb):
    try:
        va, vb = float(va), float(vb)
        d = abs(va - vb)
        rel = 100.0 * d / max(abs(va), 1e-12)
        print("   %-22s %10.5f %10.5f %10.5f %9.2f" % (name, va, vb, d, rel))
    except (TypeError, ValueError):
        print("   %-22s %10s %10s" % (name, va, vb))


def main(pa, pb, da, db, label):
    A, B = (json.load(open(p, encoding="utf-8")) for p in (pa, pb))
    print("=" * 92)
    print("SEED-FLOOR PAIR: %s" % label)
    print("A = %s   W_KAPPA=%s  kamm_mu=%s  ladder=%s" % (
        A["arm"], A["cost"]["weights"]["W_KAPPA"],
        dig(A, ("refav1", "manifest", "plan_cfg", "kamm_mu")),
        dig(A, ("refav1", "manifest", "goal_rule", "seed_kappa_ladder"))))
    print("B = %s   W_KAPPA=%s  kamm_mu=%s  ladder=%s" % (
        B["arm"], B["cost"]["weights"]["W_KAPPA"],
        dig(B, ("refav1", "manifest", "plan_cfg", "kamm_mu")),
        dig(B, ("refav1", "manifest", "goal_rule", "seed_kappa_ladder"))))
    print("plan seed A=%s  B=%s" % (dig(A, ("refav1", "manifest", "plan_cfg", "seed")),
                                    dig(B, ("refav1", "manifest", "plan_cfg", "seed"))))
    print()
    print("CONTROL - the non-planner floors MUST be identical (same window grid):")
    ok = True
    for a in ("ha", "ha0", "ha0_ext", "ol"):
        va = dig(A["arms"][a], ("intervals", "metrics", "ade_dense_m", "mean"))
        vb = dig(B["arms"][a], ("intervals", "metrics", "ade_dense_m", "mean"))
        same = (va == vb)
        ok &= same
        print("   %-8s ADE %.4f vs %.4f  identical=%s" % (a, va, vb, same))
    print("   CONTROL %s" % ("PASSED" if ok else
                             "FAILED - nothing below is a seed measurement"))
    print()
    print("SEED FLOOR on `cl` (the arm that MUST differ):")
    print("   %-22s %10s %10s %10s %9s" % ("metric", "seed A", "seed B", "abs diff", "rel %"))
    for name, ks in FAM:
        row(name, dig(A["arms"]["cl"], ks), dig(B["arms"]["cl"], ks))
    for cls, stat in PERCLASS:
        ks = ("four_families", "tactical", "lateral_decision", "per_class", cls, stat)
        row("TACpc %s_%s" % (cls, stat[:3]), dig(A["arms"]["cl"], ks), dig(B["arms"]["cl"], ks))
    ra, rb = feas(da), feas(db)
    if ra and rb:
        print("   -- re-derived from the dumps (assert_feasible, v0 >= %.1f, nA=%d nB=%d) --"
              % (VMIN, ra["_n"], rb["_n"]))
        for k, nm in (("kamm_over_rate", "FEAS kamm_over"),
                      ("envelope_rate", "FEAS envelope"),
                      ("peak_g_max", "FEAS peak_g_max"),
                      ("max_abs_kappa", "FEAS max|kappa|"),
                      ("max_abs_accel", "FEAS max|a|")):
            row(nm, ra.get(k, float("nan")), rb.get(k, float("nan")))
        # the ground-truth control, in the same table
        ga, gb = feas(da, "g"), feas(db, "g")
        print("   CONTROL g (GT) kamm_over  A=%.4f  B=%.4f  (must both be 0.0000)"
              % (ga.get("kamm_over_rate", float("nan")),
                 gb.get("kamm_over_rate", float("nan"))))
    else:
        print("   FEAS rows ABSENT (a dump is missing) - not silently dropped")
    print()


if __name__ == "__main__":
    main(*sys.argv[1:6])
