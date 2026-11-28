"""Render the four metric families for every arm of a refcv3_arm.py record.

Usage: python ff_table_navpred.py <record.json> [<record.json> ...]
Emits an ASCII table per family; STRATEGIC is read from the record-level
`refcv3.strategic` block, because the per-arm four_families path reports it
UNAVAILABLE by design (landing read W-6).
"""
import json
import sys

ARMS = ["os", "os_navpred", "ha", "ha0_ext", "ha0", "os_navshuf", "os_navzero"]


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


for path in sys.argv[1:]:
    R = json.load(open(path, encoding="utf-8"))
    A = R.get("arms", {})
    print("=" * 110)
    print(path, " n_windows", R.get("n_windows"), " n_episodes", R.get("n_episodes"))
    print("=" * 110)

    print("\n--- LONGITUDINAL " + "-" * 80)
    print("%-12s %10s %10s %14s %12s %12s" % (
        "arm", "speedMAE", "bias", "tgtacc@0.5", "alongMAE", "accelMAE"))
    for a in ARMS:
        L = g(A, a, "four_families", "longitudinal")
        if not L:
            continue
        print("%-12s %10.4f %10.4f %14.4f %12.4f %12.4f" % (
            a, L.get("speed_mae_mps", float("nan")), L.get("speed_bias_mps", float("nan")),
            g(L, "target_speed_acc", "within_0.5_mps", default=float("nan")),
            L.get("along_mae_m", float("nan")), L.get("accel_mae_mps2", float("nan"))))

    print("\n--- LATERAL (read curvature against ha0, the straight-line floor) " + "-" * 30)
    print("%-12s %12s %14s %16s %12s" % (
        "arm", "headMAEdeg", "yawrateMAEdps", "CURVMAE_1pm", "crossMAE"))
    for a in ARMS:
        L = g(A, a, "four_families", "lateral")
        if not L:
            continue
        print("%-12s %12.4f %14.4f %16.6f %12.4f" % (
            a, L.get("heading_mae_deg", float("nan")), L.get("yaw_rate_mae_degps", float("nan")),
            L.get("curvature_mae_1pm", float("nan")), L.get("cross_mae_m", float("nan"))))

    print("\n--- TACTICAL " + "-" * 84)
    print("%-12s %8s %8s %8s %8s | %8s %8s %10s %10s" % (
        "arm", "LATacc", "LATk", "turnL", "turnR", "LONacc", "LONk", "brake_stop", "accel"))
    for a in ARMS:
        T = g(A, a, "four_families", "tactical")
        if not T:
            continue
        lat, lon = T.get("lateral_decision", {}), T.get("longitudinal_decision", {})
        print("%-12s %8.4f %8.4f %8.3f %8.3f | %8.4f %8.4f %10.3f %10.3f" % (
            a, lat.get("accuracy", float("nan")), lat.get("kappa", float("nan")),
            g(lat, "per_class", "turn_left", "recall", default=float("nan")),
            g(lat, "per_class", "turn_right", "recall", default=float("nan")),
            lon.get("accuracy", float("nan")), lon.get("kappa", float("nan")),
            g(lon, "per_class", "brake_stop", "recall", default=float("nan")),
            g(lon, "per_class", "accelerate", "recall", default=float("nan"))))
    for a in ARMS:
        T = g(A, a, "four_families", "tactical")
        if T and "goal" in json.dumps(T)[:0] + "":
            pass
    print("\n  goal/anchor selection (goal_point_error / goal_bearing):")
    for a in ARMS:
        T = g(A, a, "four_families", "tactical")
        if not T:
            continue
        gg = T.get("goal_setting") or T.get("goal") or {}
        if gg:
            print("   %-12s %s" % (a, json.dumps(
                {k: v for k, v in gg.items() if not k.startswith("_")})[:170]))

    print("\n--- STRATEGIC (record-level refcv3.strategic) " + "-" * 50)
    S = g(R, "refcv3", "strategic") or {}
    print("  n_windows %s  n_route_labeled %s  nav_valid_frac %s" % (
        S.get("n_windows"), S.get("n_route_labeled"), S.get("nav_valid_frac")))
    for cname, blk in (S.get("conditionings") or {}).items():
        if not isinstance(blk, dict) or blk.get("status") == "REFUSED":
            print("   %-14s REFUSED: %s" % (cname, str(blk.get("reason"))[:90]))
            continue
        print("   %-14s acc %s  kappa %s  echo %s (legacy raw %s)  maj %s" % (
            cname, blk.get("accuracy"), blk.get("kappa"), blk.get("nav_echo_index"),
            blk.get("nav_echo_index_RAW_INDEX_LEGACY"), blk.get("majority_class_rate")))
        pc = blk.get("per_class") or {}
        if pc:
            print("      per-class recall: " + "  ".join(
                "%s %.3f (n %s)" % (k, v.get("recall", float("nan")), v.get("n_true"))
                for k, v in pc.items()))
    for k in ("n_changed_subset", "changed_subset", "changed_exclusive_subset",
              "paired_true_minus_shuffled_accuracy"):
        if k in S:
            print("   %s: %s" % (k, json.dumps(S[k])[:400]))
    print()
