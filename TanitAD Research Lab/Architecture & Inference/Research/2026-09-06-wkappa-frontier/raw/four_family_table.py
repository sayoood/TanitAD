#!/usr/bin/env python3
"""Four-family table for refav1_arm records. ADE alone is INCOMPLETE (binding).

Reads any number of rec_*.json and prints, per arm, the four families with their
episode-cluster-bootstrap intervals, plus the PAIRED (cl - floor) deltas the
record already carries. Every row is stamped with its tier and the arm's
(metric, W_JERK, W_KAPPA, W_VEND) -- an arm whose weights are not in its record
is not quotable.
ASCII only: the dev box is cp1252 and a non-ASCII print() is fatal.
"""
import json, sys, os

ARMS = ("cl", "ha", "ha0", "ha0_ext", "ol")   # default; overridden per record

def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d

def ci(m):
    if not isinstance(m, dict):
        return "     n/a"
    return "%.4f [%.4f, %.4f]" % (m.get("mean", float("nan")),
                                  m.get("lo", float("nan")),
                                  m.get("hi", float("nan")))

def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def head(rec, path):
    c = rec.get("cost", {})
    w = c.get("weights", {})
    print("=" * 100)
    print("ARM %-34s  n_windows=%s n_episodes=%s" %
          (rec.get("arm"), rec.get("n_windows"), rec.get("n_episodes")))
    print("  cost: metric=%-5s W_JERK=%-10.6g W_KAPPA=%-12.8g W_VEND=%-10.6g  source=%s"
          % (c.get("metric"), w.get("W_JERK", float('nan')),
             w.get("W_KAPPA", float('nan')), w.get("W_VEND", float('nan')),
             c.get("weights_source")))
    print("  record: %s" % path)

def famtable(rec):
    global ARMS
    ARMS = tuple(rec.get("arm_keys") or ARMS)
    print("-" * 100)
    print("  %-9s %-4s | %-24s | %-10s %-10s %-10s | %-9s" %
          ("arm", "tier", "ADE_m (episode-cluster CI)", "speed_MAE", "along_MAE",
           "tsa<1.0", "curv_MAE"))
    for a in ARMS:
        A = rec["arms"].get(a)
        if A is None:
            continue
        iv = g(A, "intervals", "metrics", default={})
        ff = A["four_families"]
        lon, lat = ff["longitudinal"], ff["lateral"]
        print("  %-9s %-4s | %-24s | %-10.4f %-10.4f %-10.4f | %-9.6f" % (
            a, A["tier"], ci(iv.get("ade_dense_m")),
            lon.get("speed_mae_mps", float("nan")),
            lon.get("along_mae_m", float("nan")),
            g(lon, "target_speed_acc", "within_1.0_mps", default=float("nan")),
            lat.get("curvature_mae_1pm", float("nan"))))
    print()
    print("  LONGITUDINAL (binding family 1)")
    print("    %-9s %-11s %-11s %-11s %-11s %-11s %-11s" %
          ("arm", "speed_MAE", "speed_bias", "speed_RMSE", "along_MAE",
           "accel_MAE", "tsa<0.5"))
    for a in ARMS:
        A = rec["arms"].get(a)
        if A is None: continue
        L = A["four_families"]["longitudinal"]
        print("    %-9s %-11.4f %-11.4f %-11.4f %-11.4f %-11.4f %-11.4f" % (
            a, L.get("speed_mae_mps", float('nan')), L.get("speed_bias_mps", float('nan')),
            L.get("speed_rmse_mps", float('nan')), L.get("along_mae_m", float('nan')),
            L.get("accel_mae_mps2", float('nan')),
            g(L, "target_speed_acc", "within_0.5_mps", default=float('nan'))))
    dk = g(rec, "arms", "cl", "four_families", "longitudinal", "distance_keeping", default={})
    print("    distance_keeping: %s  (n=%s)  reason=%s" % (
        dk.get("status"), dk.get("n"), str(dk.get("reason"))[:78]))
    print()
    print("  LATERAL (binding family 2)")
    print("    %-9s %-11s %-13s %-13s %-13s %-11s %-11s" %
          ("arm", "head_MAE_d", "yawrate_MAE", "curv_MAE_1pm", "curv_bias_1pm",
           "cross_MAE", "crossfin_MAE"))
    for a in ARMS:
        A = rec["arms"].get(a)
        if A is None: continue
        T = A["four_families"]["lateral"]
        print("    %-9s %-11.4f %-13.4f %-13.6f %-13.6f %-11.4f %-11.4f" % (
            a, T.get("heading_mae_deg", float('nan')), T.get("yaw_rate_mae_degps", float('nan')),
            T.get("curvature_mae_1pm", float('nan')), T.get("curvature_bias_1pm", float('nan')),
            T.get("cross_mae_m", float('nan')), T.get("cross_final_mae_m", float('nan'))))
    print()
    print("  TACTICAL (binding family 3) -- decision quality + goal setting")
    print("    %-9s %-8s %-8s %-8s %-8s %-8s %-10s %-10s" %
          ("arm", "lat_acc", "lat_kap", "lon_acc", "lon_kap", "m5_acc",
           "goalFDE_m", "goalbear_d"))
    for a in ARMS:
        A = rec["arms"].get(a)
        if A is None: continue
        T = A["four_families"]["tactical"]
        if T.get("status") != "OK":
            print("    %-9s %s" % (a, T.get("status"))); continue
        ld, lo_, m5 = T["lateral_decision"], T["longitudinal_decision"], T["maneuver_5way_collapsed"]
        gs = T["goal_setting"]
        print("    %-9s %-8.4f %-8.4f %-8.4f %-8.4f %-8.4f %-10.4f %-10.4f" % (
            a, ld.get("accuracy", float('nan')), ld.get("kappa", float('nan')),
            lo_.get("accuracy", float('nan')), lo_.get("kappa", float('nan')),
            m5.get("accuracy", float('nan')),
            gs.get("goal_point_error_m", float('nan')),
            gs.get("goal_bearing_mae_deg", float('nan'))))
    ld = g(rec, "arms", "cl", "four_families", "tactical", "lateral_decision", "per_class", default={})
    if ld:
        print("    cl lateral per-class: " + "  ".join(
            "%s n_true=%s recall=%s" % (k, v.get("n_true"), v.get("recall"))
            for k, v in ld.items()))
    print()
    print("  STRATEGIC (binding family 4)")
    S = g(rec, "arms", "cl", "four_families", "strategic", default={})
    print("    status=%s n=%s  reason=%s" % (S.get("status"), S.get("n"),
                                             str(S.get("reason"))[:80]))
    print()
    print("  PAIRED (episode-cluster bootstrap, the decision-grade estimator)")
    for k, blk in rec.get("paired_decision_grade", {}).items():
        for met, m in blk.items():
            if not isinstance(m, dict) or "delta" not in m:
                continue
            print("    %-26s %-16s delta=%+8.4f [%+8.4f, %+8.4f] separated=%s" % (
                blk.get("direction", k), met, m["delta"], m["lo"], m["hi"],
                m.get("separated")))

def main(paths):
    for p in paths:
        if not os.path.exists(p):
            print("MISSING %s" % p); continue
        rec = load(p)
        head(rec, p)
        famtable(rec)
        print()

if __name__ == "__main__":
    main(sys.argv[1:])
