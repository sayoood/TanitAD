"""Follow-ups to the K-yield curve.

(a) The inherited claim "stride cannot buy windows" -- tested, not repeated.
(b) The CI a K=185 closed-loop co-primary would carry at 600 clusters, projected
    from the gate's MEASURED 41-window interval. ESTIMATED, and labelled so.
(c) Where the yield curve crosses the ladder's n>=200 EPISODE-CLUSTER bar.
"""
import json
import math
import sys
from pathlib import Path

SC = Path(sys.argv[1])
W = 8
val = json.loads((SC / "verify_val600_pod2.json").read_text())
pub = json.loads((SC / "verify_evalpod40.json").read_text())
T600 = [e["T"] for e in val["episodes"]]
T40 = [e["T"] for e in pub["episodes"]]


def nwin(T, K, s):
    span = T - W - K
    return max(0, math.ceil(span / s)) if span > 0 else 0


def agg(Ts, K, s):
    per = [nwin(T, K, s) for T in Ts]
    return sum(per), sum(1 for x in per if x > 0)


# ---------------------------------------------------------------- (a) stride
stride_test = {}
for K in (120, 150, 185, 190):
    row = {}
    for s in (8, 4, 2, 1):
        w40, e40 = agg(T40, K, s)
        w600, e600 = agg(T600, K, s)
        row[f"stride_{s}"] = {"win_40ep": w40, "clusters_40ep": e40,
                              "win_600ep": w600, "clusters_600ep": e600}
    row["verdict"] = (
        f"stride 8->1 multiplies WINDOWS x"
        f"{row['stride_1']['win_600ep'] / max(1, row['stride_8']['win_600ep']):.1f} "
        f"on the 600 and changes CLUSTERS by "
        f"{row['stride_1']['clusters_600ep'] - row['stride_8']['clusters_600ep']:+d}.")
    stride_test[f"K={K}"] = row

# ------------------------------------------------------------------- (b) CI
# gate MEASURED: CDR@1.75 m, K=185, 41 windows / 40 episodes
GATE = {"point": 0.6388, "lo": 0.5565, "hi": 0.7128, "n_win": 41, "n_ep": 40}
hw40 = (GATE["hi"] - GATE["lo"]) / 2.0
w600_185, e600_185 = agg(T600, 185, 8)
scale = math.sqrt(GATE["n_ep"] / e600_185)
p = GATE["point"]
binom_se_40 = math.sqrt(p * (1 - p) / GATE["n_ep"])
binom_se_600 = math.sqrt(p * (1 - p) / e600_185)
ci_projection = {
    "MEASURED_at_40ep": {"point": p, "ci95": [GATE["lo"], GATE["hi"]],
                         "half_width": round(hw40, 4),
                         "n_windows": GATE["n_win"], "n_clusters": GATE["n_ep"],
                         "source": ("GATE_30K_RESULTS.md co-primary, "
                                    "episode_cluster_bootstrap B=2000")},
    "ESTIMATED_at_600ep": {
        "n_windows": w600_185, "n_clusters": e600_185,
        "sqrt_n_scale_factor": round(scale, 4),
        "projected_half_width": round(hw40 * scale, 4),
        "projected_ci95": [round(p - hw40 * scale, 4), round(p + hw40 * scale, 4)],
        "cross_check_binomial_half_width_40": round(1.96 * binom_se_40, 4),
        "cross_check_binomial_half_width_600": round(1.96 * binom_se_600, 4),
        "class": "ESTIMATED",
        "assumptions": ("1/sqrt(n_clusters) scaling at an unchanged departure "
                        "rate and ~1 window per cluster at K=185. It is a "
                        "projection, NOT a measurement -- the real interval "
                        "must come from running the block on the 600."),
    },
}

# -------------------------------------------------------- (c) the n>=200 bar
bar = []
for K in range(20, 200):
    w, e = agg(T600, K, 8)
    bar.append({"K": K, "clusters": e, "windows": w, "meets_200": e >= 200})
first_fail = next((r["K"] for r in bar if not r["meets_200"]), None)
max_K_all600 = min(T600) - W - 1
max_K_any600 = max(T600) - W - 1

out = {"a_stride": stride_test,
       "b_ci_projection_K185": ci_projection,
       "c_n200_bar": {
           "largest_K_with_at_least_200_clusters":
               max((r["K"] for r in bar if r["meets_200"]), default=None),
           "first_K_below_200_clusters": first_fail,
           "structural_K_ceiling_ANY_episode": max_K_any600,
           "structural_K_ceiling_EVERY_episode": max_K_all600,
           "clusters_at_K": {str(k): agg(T600, k, 8)[1]
                             for k in (20, 60, 90, 120, 150, 185, 190, 192, 194, 196)},
           "windows_at_K": {str(k): agg(T600, k, 8)[0]
                            for k in (20, 60, 90, 120, 150, 185, 190, 192, 194, 196)},
       }}
(SC / "horizon_analysis.json").write_text(json.dumps(out, indent=1))

print("(a) STRIDE -- does a smaller stride buy anything?")
for k, r in stride_test.items():
    print(f"  {k}: " + "  ".join(
        f"s={s}: {r['stride_'+str(s)]['win_600ep']}w/{r['stride_'+str(s)]['clusters_600ep']}c"
        for s in (8, 4, 2, 1)))
    print(f"        40 ep: " + "  ".join(
        f"s={s}: {r['stride_'+str(s)]['win_40ep']}w/{r['stride_'+str(s)]['clusters_40ep']}c"
        for s in (8, 4, 2, 1)))
print()
print("(b) K=185 CDR interval, MEASURED at 40 vs PROJECTED at 600")
m, e = ci_projection["MEASURED_at_40ep"], ci_projection["ESTIMATED_at_600ep"]
print(f"  MEASURED  40 clusters: {m['point']:.4f} [{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}]  "
      f"half-width {m['half_width']:.4f}")
print(f"  ESTIMATED {e['n_clusters']} clusters: {m['point']:.4f} "
      f"[{e['projected_ci95'][0]:.4f}, {e['projected_ci95'][1]:.4f}]  "
      f"half-width {e['projected_half_width']:.4f}  (x{e['sqrt_n_scale_factor']:.3f})")
print()
print("(c) the n>=200 EPISODE-CLUSTER bar on the 600")
c = out["c_n200_bar"]
print(f"  clusters by K: {c['clusters_at_K']}")
print(f"  largest K with >=200 clusters: {c['largest_K_with_at_least_200_clusters']} "
      f"({0.1*c['largest_K_with_at_least_200_clusters']:.1f} s)")
print(f"  structural ceiling: ANY episode K={c['structural_K_ceiling_ANY_episode']}, "
      f"EVERY episode K={c['structural_K_ceiling_EVERY_episode']}")
