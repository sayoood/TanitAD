"""K -> window/episode yield, computed on the REAL episode lengths of the 600.

The window rule is ONE line of taniteval/rollout.py:130

    starts = list(range(0, T - window - K, stride))          # window = 8

so an episode of length T yields ceil((T - W - K)/stride) windows at horizon K,
and ZERO once T - W - K <= 0.  ``corridor.horizon_ceiling`` states the same cap:
K_max(T) = T - W - 1.

The bootstrap's resampling unit is the EPISODE, so the number that decides power
is `episodes_with_at_least_one_window`, not the window count. Stride is reported
too, precisely because it is the thing that looks like it should help: at K=185
a smaller stride multiplies windows that sit 0.1 s apart inside the SAME episode
-- more rows, the same clusters, and the CI does not move.

Validated against two independently MEASURED published points before use:
  40 published episodes, W=8, stride=8, K=20  -> 881 windows   (MODEL_REGISTRY)
  40 published episodes, W=8, stride=8, K=185 -> 41 windows    (GATE_30K_RESULTS)
"""
import json
import math
import sys
from pathlib import Path

SC = Path(sys.argv[1])
W = 8
STRIDES = (8, 4, 2, 1)
KS = list(range(20, 191, 5))
KS_REPORT = [20, 40, 60, 80, 90, 100, 120, 140, 150, 160, 170, 185, 190]

val = json.loads((SC / "verify_val600_pod2.json").read_text())
pub = json.loads((SC / "verify_evalpod40.json").read_text())
T600 = [e["T"] for e in val["episodes"]]
T40 = [e["T"] for e in pub["episodes"]]
assert len(T600) == 600 and len(T40) == 40


def nwin(T, K, stride):
    span = T - W - K
    return max(0, math.ceil(span / stride)) if span > 0 else 0


def yield_at(Ts, K, stride=8):
    per = [nwin(T, K, stride) for T in Ts]
    return {"K": K, "horizon_s": round(K * 0.1, 2), "stride": stride,
            "n_windows": sum(per),
            "n_episodes_with_window": sum(1 for x in per if x > 0),
            "n_episodes_total": len(Ts),
            "windows_per_episode": round(sum(per) / max(1, len(Ts)), 3),
            "max_windows_in_an_episode": max(per)}


# ---- VALIDATION against the two published MEASURED points -------------------
v20 = yield_at(T40, 20, 8)
v185 = yield_at(T40, 185, 8)
validation = {
    "published_40ep_K20_stride8": {"computed": v20["n_windows"],
                                   "published": 881,
                                   "match": v20["n_windows"] == 881},
    "published_40ep_K185_stride8": {"computed": v185["n_windows"],
                                    "published": 41,
                                    "match": v185["n_windows"] == 41},
}
print("VALIDATION of the window rule against published MEASURED points")
for k, v in validation.items():
    print(f"  {k}: computed={v['computed']} published={v['published']} "
          f"{'MATCH' if v['match'] else 'MISMATCH'}")
print()

out = {
    "method": ("starts = range(0, T - W - K, stride) with W=8 "
               "(taniteval/rollout.py:130); K_max(T) = T - W - 1 "
               "(corridor.horizon_ceiling). Episode lengths T are MEASURED on "
               "the actual caches, not assumed."),
    "T_stats": {
        "val600": {"n": 600, "min": min(T600), "max": max(T600),
                   "mean": round(sum(T600) / 600, 2),
                   "n_below_198": sum(1 for t in T600 if t < 198)},
        "published40": {"n": 40, "min": min(T40), "max": max(T40),
                        "mean": round(sum(T40) / 40, 2)},
    },
    "validation": validation,
    "structural_ceiling_K": {
        "val600_max_K_any_episode": max(T600) - W - 1,
        "val600_max_K_ALL_episodes": min(T600) - W - 1,
        "published40_max_K_any_episode": max(T40) - W - 1,
        "published40_max_K_ALL_episodes": min(T40) - W - 1,
    },
    "curve_stride8_600": [yield_at(T600, K, 8) for K in KS],
    "curve_stride8_40": [yield_at(T40, K, 8) for K in KS],
    "stride_sensitivity_600": {
        str(s): [yield_at(T600, K, s) for K in KS_REPORT] for s in STRIDES},
}
(SC / "horizon_yield.json").write_text(json.dumps(out, indent=1))

print(f"T on the 600: min={min(T600)} max={max(T600)} "
      f"mean={out['T_stats']['val600']['mean']} "
      f"({out['T_stats']['val600']['n_below_198']} episodes shorter than the "
      f"published 40's minimum of {min(T40)})")
print(f"structural K ceiling: ANY episode {max(T600)-W-1} · "
      f"EVERY episode {min(T600)-W-1}")
print()
hdr = (f"{'K':>4} {'s':>6} | {'600: win':>9} {'eps':>5} {'w/ep':>6} | "
       f"{'40: win':>8} {'eps':>4} | 600 eps >= 200?")
print(hdr); print("-" * len(hdr))
for K in KS_REPORT:
    a = yield_at(T600, K, 8)
    b = yield_at(T40, K, 8)
    print(f"{K:>4} {K*0.1:>6.1f} | {a['n_windows']:>9} "
          f"{a['n_episodes_with_window']:>5} {a['windows_per_episode']:>6.2f} | "
          f"{b['n_windows']:>8} {b['n_episodes_with_window']:>4} | "
          f"{'YES' if a['n_episodes_with_window'] >= 200 else 'NO':>3}")
print()
print("STRIDE sensitivity on the 600 (windows / EPISODES-with-a-window):")
print(f"{'K':>4} " + " ".join(f"{'s='+str(s):>16}" for s in STRIDES))
for K in [20, 60, 120, 150, 185]:
    row = " ".join(
        f"{yield_at(T600,K,s)['n_windows']:>8}/"
        f"{yield_at(T600,K,s)['n_episodes_with_window']:<7}" for s in STRIDES)
    print(f"{K:>4} {row}")
