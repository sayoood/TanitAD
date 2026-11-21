"""Run DDS against its controls. A score that cannot fail is not a score.

Controls, in increasing order of how much they would embarrass the score:
  1. RANDOM subset of the same size — the score must rank a designed subset ABOVE
     it, or the score is inert.
  2. DELIBERATELY BAD synthetic subsets (all-slow, single-country, one hour) —
     must rank LOW.
  3. ⭐ THE REAL KNOWN-BAD: the r0 DESIGNED corpus. `physicalai_r0.py:100` gates
     mean speed at 14 m/s, so it has 0.0 % fast-arterial and 0.0 % highway BY
     CONSTRUCTION. A score that ranks it well is refuted by source, not by
     opinion. This is a control with a KNOWN ANSWER, which beats a synthetic one.
  4. EQUAL SIZE throughout — a score that merely rewards more data has measured
     nothing, so every arm here is the same n.
"""
import gzip
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dds import dds, speed_band  # noqa: E402

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release/tanitad-v7-training-corpus")
META = Path("C:/Users/Admin/tanitad-data/physicalai/metadata")
RNG = np.random.default_rng(0)

# --- build the frame: parent catalogue + per-clip speed where we have it -----
dc = pd.read_parquet(META / "data_collection.parquet")
dc.index = dc.index.astype(str)
dc = dc.reset_index().rename(columns={"index": "clip_id"})

sp = pd.read_parquet(Path(__file__).with_name("b1_speed_profile.parquet"))
sp["speed_band"] = sp.mean_ms.map(speed_band)
sp = sp.rename(columns={"clip": "clip_id"})

b1 = pd.merge(sp[["clip_id", "mean_ms", "speed_band"]], dc, on="clip_id", how="inner")
print(f"B1 with speed + catalogue: {len(b1)}")

# ⚠️ The parent has no per-clip speed (egomotion is not downloaded corpus-wide),
# so the parent's speed distribution is IMPUTED from B1 — and B1 is itself a
# biased draw. That is stated, not hidden: it makes `defect_ratio_vs_parent` a
# comparison against OUR OWN best estimate of the parent, and the ordering of
# the arms below does not depend on it (they are compared to each other).
parent = b1.copy()

# ⛔ EQUAL SIZE IS THE WHOLE POINT — a score that merely rewards more data has
# measured nothing. N is therefore the LARGEST size every arm can actually
# reach, computed rather than assumed: the first run used N=800 and the
# single-country arm could only supply 302 (B1's biggest country), which failed
# the equal-size assertion. The assertion caught my arm construction, which is
# what it is for.
N = min(
    len(b1[b1.speed_band.isin(["crawl", "slow_urban"])]),
    int(b1.country.value_counts().max()),
    len(b1[(b1.mean_ms >= 2.0) & (b1.mean_ms <= 14.0)]),
)
print(f"equal-size N for every arm = {N}")


def sub(df, n=N):
    return df.sample(n=min(n, len(df)), random_state=0)


arms = {}
arms["random"] = sub(b1)
# a designed arm: force equal mass across speed bands, then spread countries
pieces = []
per = N // b1.speed_band.nunique()
for band, g in b1.groupby("speed_band"):
    pieces.append(g.sample(n=min(per, len(g)), random_state=1))
designed = pd.concat(pieces)
if len(designed) < N:
    designed = pd.concat([designed, b1.drop(designed.index).sample(
        n=N - len(designed), random_state=1)])
arms["designed_speed_balanced"] = designed
# deliberately bad
arms["bad_all_slow"] = sub(b1[b1.speed_band.isin(["crawl", "slow_urban"])])
top_c = b1.country.value_counts().idxmax()
arms["bad_single_country"] = sub(b1[b1.country == top_c])
# ⭐ the REAL known-bad: emulate the r0 gate exactly (2.0 <= mean_v <= 14.0)
arms["r0_gate_emulated"] = sub(b1[(b1.mean_ms >= 2.0) & (b1.mean_ms <= 14.0)])

print(f"\n{'arm':28s} {'n':>5s} {'DDS':>7s} {'cover':>7s} {'divers':>7s} {'defect%':>8s}")
res = {}
for name, a in arms.items():
    r = dds(a, parent)
    res[name] = r
    print(f"  {name:26s} {r['n']:5d} {r['dds']:7.4f} {r['coverage']:7.4f} "
          f"{r['effective_diversity']:7.4f} {r['defect_regime_mass']*100:7.1f}%")

order = sorted(res, key=lambda k: -res[k]["dds"])
print("\nranking (best first):", " > ".join(order))

print("\n--- CONTROL ASSERTIONS ---")
ok = True


def check(label, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    ok = ok and cond


check("designed ranks ABOVE random (score is not inert)",
      res["designed_speed_balanced"]["dds"] > res["random"]["dds"])
check("r0-gate-emulated ranks BELOW random (the real known-bad)",
      res["r0_gate_emulated"]["dds"] < res["random"]["dds"])
check("all-slow ranks BELOW random",
      res["bad_all_slow"]["dds"] < res["random"]["dds"])
check("single-country ranks BELOW random",
      res["bad_single_country"]["dds"] < res["random"]["dds"])
check("r0-gate-emulated has ZERO defect-regime mass (matches source)",
      res["r0_gate_emulated"]["defect_regime_mass"] == 0.0)
check("every arm is the SAME size (n rewards nothing)",
      len({r["n"] for r in res.values()}) == 1)
print(f"\nCONTROLS {'PASS' if ok else 'FAIL'}")
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_axis"}
           for k, v in res.items()},
          open(Path(__file__).with_name("dds_controls.json"), "w"), indent=1)
