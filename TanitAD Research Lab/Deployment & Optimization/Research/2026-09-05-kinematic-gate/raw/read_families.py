#!/usr/bin/env python3
"""Render the FOUR FAMILIES from kingate_base.json -- per family, NEVER pooled.

ADE is printed BESIDE the families and never as "the result" (binding, PI 2026-08-02).
Every family carries its n and its estimator; a family that cannot be computed is printed
with its reason and its n, and `defect: true` is printed as a DEFECT, not as a refusal.
ASCII-only output.
"""
import json
import sys


def g(d, path, default=None):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def f(v, fmt="%.4f"):
    try:
        return fmt % float(v)
    except (TypeError, ValueError):
        return "     n/a"


def main():
    src = sys.argv[1]
    d = json.load(open(src, encoding="utf-8"))
    rules = ["model", "gate1", "gate2", "gate4", "kin_only", "oracle"]
    print("=" * 96)
    print("FOUR METRIC FAMILIES on the SELECTED path -- per family, NEVER pooled")
    print("tier=%s  device=%s  dt=%.2fs  estimator=paired episode-cluster bootstrap "
          "(cluster=clip)" % (d["tier"], d["device"], d["dt_s"]))
    print("=" * 96)
    for tag in ("A", "B"):
        dr = d["draws"][tag]
        fam = dr["families"]
        print("")
        print("---- DRAW %s : %d windows / %d episodes (episode-disjoint from the other) "
              "----" % (tag, dr["n_windows"], dr["n_episodes"]))

        print("")
        print("  [1] LONGITUDINAL  (target speed + distance keeping)")
        print("      %-10s %10s %10s %10s %10s %10s"
              % ("rule", "speed_mae", "speed_bias", "along_mae", "accel_mae", "n"))
        for r in rules:
            L = fam.get(r, {}).get("longitudinal", {})
            if L.get("unavailable"):
                print("      %-10s UNAVAILABLE defect=%s n=%s : %s"
                      % (r, L.get("defect"), L.get("n"), str(L.get("reason"))[:70]))
                continue
            print("      %-10s %10s %10s %10s %10s %10s"
                  % (r, f(L.get("speed_mae_mps")), f(L.get("speed_bias_mps")),
                     f(L.get("along_mae_m")), f(L.get("accel_mae_mps2")),
                     L.get("n_windows")))
        dk = g(fam.get("model", {}), "longitudinal.distance_keeping", {})
        print("      distance-keeping (headway/TTC to the lead): status=%s n=%s"
              % (dk.get("status"), dk.get("n", dk.get("n_windows"))))

        print("")
        print("  [2] LATERAL  (heading + CURVATURE + yaw-rate + cross-track)")
        print("      %-10s %10s %10s %10s %10s %10s"
              % ("rule", "cross_mae", "head_mae_d", "kappa_mae", "kappa_bias", "n_kappa"))
        for r in rules:
            L = fam.get(r, {}).get("lateral", {})
            if L.get("unavailable"):
                print("      %-10s UNAVAILABLE defect=%s n=%s : %s"
                      % (r, L.get("defect"), L.get("n"), str(L.get("reason"))[:70]))
                continue
            print("      %-10s %10s %10s %10s %10s %10s"
                  % (r, f(L.get("cross_mae_m")), f(L.get("heading_mae_deg")),
                     f(L.get("curvature_mae_1pm")), f(L.get("curvature_bias_1pm")),
                     L.get("n_steps_curvature")))

        print("")
        print("  [3] TACTICAL  (manoeuvre decision + goal setting)  factored, not 5-way only")
        print("      %-10s %8s %8s %8s %8s %10s %10s %8s"
              % ("rule", "lat_acc", "lat_kap", "lon_acc", "5way_acc", "goal_err_m",
                 "goal_brg_d", "n"))
        for r in rules:
            T = fam.get(r, {}).get("tactical", {})
            if T.get("unavailable"):
                print("      %-10s UNAVAILABLE defect=%s n=%s : %s"
                      % (r, T.get("defect"), T.get("n"), str(T.get("reason"))[:70]))
                continue
            print("      %-10s %8s %8s %8s %8s %10s %10s %8s"
                  % (r, f(g(T, "lateral_decision.accuracy")),
                     f(g(T, "lateral_decision.kappa")),
                     f(g(T, "longitudinal_decision.accuracy")),
                     f(g(T, "maneuver_5way_collapsed.accuracy")),
                     f(g(T, "goal_setting.goal_point_error_m")),
                     f(g(T, "goal_setting.goal_bearing_mae_deg")), T.get("n")))
        print("      anchor/goal SELECTION (this family's core row for a selection rule):")
        for r in rules:
            am = g(dr, "results.abs.%s.agrees_model.mean" % r)
            print("        %-10s agrees_model=%s   (fraction of windows where this rule "
                  "returns the model's own candidate)" % (r, f(am)))

        print("")
        print("  [4] STRATEGIC  (strategic decision + route/goal setting)")
        S = fam.get("model", {}).get("strategic", {})
        print("      status=%s  defect=%s  n=%s" % (S.get("status"), S.get("defect", False),
                                                     S.get("n")))
        print("      reason: %s" % str(S.get("reason", ""))[:300].replace("-", "-"))
        si = dr.get("strategic_identity", {})
        print("      IDENTITY for THIS lever: route head re-run by the gate = %s ; "
              "route_logits identical across rules = %s"
              % (si.get("route_head_rerun_by_the_gate"),
                 si.get("route_logits_bitwise_identical_across_rules")))
        print("      route argmax vs the nav_cmd it was FED = %s  <- an ECHO of an input, "
              "not strategic skill" % f(si.get("route_argmax_vs_nav_cmd_agreement")))

        print("")
        print("  [beside the families, never as 'the result'] ADE / FDE")
        print("      %-10s %10s %10s %10s" % ("rule", "ade_m(2s)", "fde_m", "d_ade"))
        for r in rules:
            A = g(dr, "results.abs.%s" % r, {})
            P = g(dr, "results.paired_vs_model.%s" % r, {})
            print("      %-10s %10s %10s %10s"
                  % (r, f(g(A, "ade_m.mean")), f(g(A, "fde_m.mean")),
                     f(g(P, "ade_m.delta"), "%+.4f")))

        print("")
        print("  paired per-window family deltas, gate2 - model (negative = gate better):")
        for nm, r in sorted(dr["family_paired"]["gate2"].items()):
            print("      %-24s %+10.5f  [%+.5f, %+.5f]  separated=%-5s n=%d/%dep"
                  % (nm, r["delta"], r["lo"], r["hi"], r["separated"],
                     r["n_windows"], r["n_episodes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
