"""DDS v2 controls. A score that cannot fail is not a score."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dds2 import dds2  # noqa: E402

H = Path("C:/Users/Admin/tanitad-wt/_s2build/release")
st = pd.read_parquet(H / "b1_strata_cells.parquet")
sl = pd.read_parquet(H / "b1_stop_coverage_20s.parquet")[["clip", "has_stop_launch"]]
df = st.merge(sl.rename(columns={"clip": "clip_id"}), on="clip_id", how="inner")


def band(v):
    for lo, hi, lab in [(0, 2, "crawl"), (2, 8, "slow_urban"), (8, 14, "urban_arterial"),
                        (14, 20, "fast_arterial"), (20, 99, "highway")]:
        if lo <= v < hi:
            return lab
    return "highway"


df["speed_band"] = df.mean_ms.map(band)
print(f"clips with strata + stop-launch: {len(df)}")

# the DESIGN's grid: 3 road-class x 2 day/night x every country in the corpus
DESIGN_CELLS = [f"{r}|{d}|{c}" for r in ("urban", "intersection", "highway")
                for d in ("day", "night") for c in sorted(df.country.dropna().unique())]
P_DEFECT = float(df.speed_band.isin(["fast_arterial", "highway"]).mean())
P_SL = float(df.has_stop_launch.mean())
print(f"design grid: {len(DESIGN_CELLS)} cells | corpus defect mass {P_DEFECT*100:.1f}% "
      f"| corpus stop-launch {P_SL*100:.1f}%")

pool_slow = df[df.speed_band.isin(["crawl", "slow_urban"])]
pool_r0 = df[(df.mean_ms >= 2.0) & (df.mean_ms <= 14.0)]
pool_1c = df[df.country == df.country.value_counts().idxmax()]
pool_nostop = df[~df.has_stop_launch]
N = min(len(pool_slow), len(pool_r0), len(pool_1c), len(pool_nostop),
        int(df.country.value_counts().max()))
print(f"EQUAL-SIZE N for every arm = {N}\n")

arms = {"random": df.sample(N, random_state=0)}
# designed: balance speed bands AND require stop-launch presence proportionally
parts = []
for b, g in df.groupby("speed_band"):
    parts.append(g.sample(min(N // df.speed_band.nunique(), len(g)), random_state=1))
d1 = pd.concat(parts)
if len(d1) < N:
    d1 = pd.concat([d1, df.drop(d1.index).sample(N - len(d1), random_state=1)])
arms["designed_balanced"] = d1
arms["bad_all_slow"] = pool_slow.sample(N, random_state=0)
arms["bad_single_country"] = pool_1c.sample(N, random_state=0)
arms["bad_r0_gate"] = pool_r0.sample(N, random_state=0)
arms["bad_no_stop_launch"] = pool_nostop.sample(N, random_state=0)

print(f"{'arm':24s} {'n':>5s} {'DDS2':>7s} {'cover':>7s} {'divers':>7s} {'defect':>7s} {'stopL':>7s}")
res = {}
for k, a in arms.items():
    r = dds2(a, DESIGN_CELLS, P_DEFECT, P_SL)
    res[k] = r
    print(f"  {k:22s} {r['n']:5d} {r['dds2']:7.4f} {r['coverage_vs_design']:7.4f} "
          f"{r['cell_diversity']:7.4f} {r['defect_regime_mass']*100:6.1f}% "
          f"{r['stop_launch_share']*100:6.1f}%")
print("\nranking:", " > ".join(sorted(res, key=lambda k: -res[k]["dds2"])))

print("\n--- CONTROL ASSERTIONS ---")
ok = True
def chk(lab, c):
    global ok
    print(f"  [{'PASS' if c else 'FAIL'}] {lab}")
    ok = ok and c
chk("designed ABOVE random (not inert)", res["designed_balanced"]["dds2"] > res["random"]["dds2"])
chk("r0-gate BELOW random", res["bad_r0_gate"]["dds2"] < res["random"]["dds2"])
chk("all-slow BELOW random", res["bad_all_slow"]["dds2"] < res["random"]["dds2"])
chk("single-country BELOW random", res["bad_single_country"]["dds2"] < res["random"]["dds2"])
chk("⭐ no-stop-launch arm BELOW random (the NEW axis can fail a subset)",
    res["bad_no_stop_launch"]["dds2"] < res["random"]["dds2"])
chk("no-stop-launch arm has ZERO stop-launch share",
    res["bad_no_stop_launch"]["stop_launch_share"] == 0.0)
chk("r0-gate has ZERO defect mass (matches source gate)",
    res["bad_r0_gate"]["defect_regime_mass"] == 0.0)
chk("every arm SAME size", len({r["n"] for r in res.values()}) == 1)
print(f"\nCONTROLS {'PASS' if ok else 'FAIL'}")
json.dump(res, open(H / "dds2_controls.json", "w"), indent=1)
