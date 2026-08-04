#!/usr/bin/env python3
"""Print the panel as the four families, per family, never pooled."""
import json
import sys

d = json.load(open(sys.argv[1]))
ARMS = ["refc-base-30k", "flagship-30k", "cv", "gt_oracle"]


def f(x, n=4):
    return "n/a" if x is None else (round(x, n) if isinstance(x, (int, float)) else x)


print(f"windows {d['n_windows']} eps {d['n_episodes']} states {d['window_states']}")
print(f"per-window/module max |diff| {d['_per_window_agreement_max']}")

print("\n=== LONGITUDINAL — target speed (dt=%.1f s, sparse view) ===" % d["_cadence"]["dt_s"])
print(f"{'arm':<16}{'speed_mae':>11}{'speed_bias':>12}{'along_mae':>11}{'along_fin_bias':>15}{'accel_mae':>11}{'ego_progress':>14}")
for a in ARMS:
    L = d["families"][a]["LONGITUDINAL"]
    print(f"{a:<16}{f(L['speed_mae_mps']):>11}{f(L['speed_bias_mps']):>12}{f(L['along_mae_m']):>11}"
          f"{f(L['along_final_bias_m']):>15}{f(L['accel_mae_mps2']):>11}{f(L['ego_progress'].get('progress')):>14}")

print("\n=== LONGITUDINAL — DISTANCE KEEPING (the family that was NOT COMPUTABLE) ===")
print(f"{'arm':<16}{'n':>5}{'/nwin':>7}{'headway_m':>11}{'time_gap_s':>12}{'n_tg':>6}{'min_ttc_s':>11}{'n_closing':>10}")
for a in ARMS:
    k = d["families"][a]["LONGITUDINAL"]["distance_keeping"]
    print(f"{a:<16}{k.get('n'):>5}{k.get('n_windows'):>7}{f(k.get('mean_headway_min_m')):>11}"
          f"{f(k.get('mean_time_gap_min_s')):>12}{k.get('n_time_gap'):>6}"
          f"{f(k.get('mean_min_ttc_s')):>11}{k.get('n_closing'):>10}")

print("\n=== LATERAL ===")
print(f"{'arm':<16}{'heading_deg':>12}{'yaw_rate_dps':>14}{'curv_1pm':>11}{'cross_mae_m':>13}{'cross_bias_m':>14}")
for a in ARMS:
    L = d["families"][a]["LATERAL"]
    print(f"{a:<16}{f(L.get('heading_mae_deg')):>12}{f(L.get('yaw_rate_mae_degps')):>14}"
          f"{f(L.get('curvature_mae_1pm'),6):>11}{f(L.get('cross_mae_m')):>13}{f(L.get('cross_bias_m')):>14}")

print("\n=== PAIRED DELTAS — distance keeping (paired episode-cluster bootstrap, B=2000) ===")
for key, blk in d["paired"].items():
    print(f"\n-- {key}")
    for m, v in blk["distance_keeping"].items():
        if v.get("status") != "OK":
            print(f"   {m:<18} {v.get('status')} — {v.get('reason')}")
            continue
        print(f"   {m:<18} delta {v['delta']:+.4f}  CI95 [{v['lo']:+.4f}, {v['hi']:+.4f}]  "
              f"sep={v['separated']}  n={v['n_used']} ({v['n_episodes']} eps)  "
              f"a_only={v['n_a_only']} b_only={v['n_b_only']}")

print("\n=== PAIRED DELTAS — the other family scalars ===")
for key, blk in d["paired"].items():
    print(f"\n-- {key}")
    for m, v in blk["families"].items():
        print(f"   {m:<22} delta {v['delta']:+.4f}  CI95 [{v['lo']:+.4f}, {v['hi']:+.4f}]  "
              f"sep={v['separated']}  n={v['n_used']}")

print("\n=== BY SPEED BAND — min-TTC / headway, per arm ===")
for a in ARMS:
    bs = d["by_speed"][a]
    print(f"\n-- {a}")
    for band, v in (bs.get("bands") or bs).items():
        if not isinstance(v, dict):
            continue
        print(f"   {band}: {json.dumps(v)[:260]}")

print("\n=== TACTICAL / STRATEGIC availability ===")
print(json.dumps(d["families_unavailable"], indent=1)[:2500])
