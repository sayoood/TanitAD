"""Give the corpus back the ability to state its own composition.

PI-ordered design `2026-08-06-alpamayo-augmentation/DESIGN.md:42` specifies a
"selection manifest (clip_id + STRATA CELL)". The shipped manifest has only
`clip_id` + `t0_us`, so the corpus cannot say what it is made of and its
stratification is only INFERABLE. That gap is why a designed, PI-ordered
stratified selection was read — by me, in Part 1 — as "an availability
intersection with no selection".

⚠️ THE ERROR THAT MAKES THIS ARTIFACT WORTH BUILDING: I measured B1 == the
Alpamayo record set exactly (4,719/4,719 in, 0 out) and concluded "unselected".
**A set that equals its parent is not unselected — its PARENT was selected**, one
level up, when clips were chosen to send for labelling. The manifest is what
would have made that readable instead of inferable.

Derivations use the DESIGN'S OWN rules (`DESIGN.md:17`), not mine:
    highway       speed >= 20 m/s SUSTAINED
    intersection  a stop AND a heading change
    urban         low-speed varied (the remainder)
Labels may use ego (labels-may-use-ego rule); this is offline label derivation,
never an inference input.

Also emits the Codevilla INERTIA statistic in the same pass, since it needs the
same read: the STOPPED-FRAME FRACTION. Codevilla's failure mode was not "too
little high-speed data" — it was over-represented stopped frames creating a
spurious low-speed -> no-acceleration correlation. We have never measured it.
"""
import gzip
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release/tanitad-v7-training-corpus")
META = Path("C:/Users/Admin/tanitad-data/physicalai/metadata")
OUT = Path("C:/Users/Admin/tanitad-wt/_s2build/release")

V_STOP = 0.5          # m/s — a frame is "stopped" below this
V_HIGHWAY = 20.0      # DESIGN.md:17
SUSTAIN = 0.30        # "sustained" = at least this fraction of the clip
YAW_TURN_DEG = 25.0   # a heading change worth calling a junction

rows = [json.loads(x) for x in gzip.open(REL / "labels" / "s2_labels_v7.jsonl.gz",
                                         "rt", encoding="utf-8") if x.strip()]
t0_of = {r["clip_id"]: float(r.get("t0_s") or 0.0) for r in rows}
tf = tarfile.open(REL / "egomotion" / "egomotion_alpamayo.tar")

recs = []
for m in tf.getmembers():
    if not m.isfile():
        continue
    cid = m.name.split(".")[0]
    d = pd.read_parquet(io.BytesIO(tf.extractfile(m).read()),
                        columns=["vx", "vy", "qz", "qw", "timestamp"])
    v = np.hypot(d.vx.to_numpy(), d.vy.to_numpy())
    yaw = 2.0 * np.arctan2(d.qz.to_numpy(), d.qw.to_numpy())
    yaw_unwrapped = np.unwrap(yaw)
    stop_frac = float(np.mean(v < V_STOP))
    sustained_high = float(np.mean(v >= V_HIGHWAY))
    dyaw = float(np.degrees(np.abs(yaw_unwrapped[-1] - yaw_unwrapped[0])))
    yaw_span = float(np.degrees(np.ptp(yaw_unwrapped)))
    has_stop = stop_frac > 0.02
    if sustained_high >= SUSTAIN:
        road = "highway"
    elif has_stop and yaw_span >= YAW_TURN_DEG:
        road = "intersection"
    else:
        road = "urban"
    recs.append({"clip_id": cid, "mean_ms": float(np.nanmean(v)),
                 "p95_ms": float(np.nanpercentile(v, 95)),
                 "stopped_frame_frac": round(stop_frac, 4),
                 "sustained_high_frac": round(sustained_high, 4),
                 "yaw_span_deg": round(yaw_span, 1), "net_yaw_deg": round(dyaw, 1),
                 "road_class": road})
df = pd.DataFrame(recs)
print(f"clips: {len(df)}")

dc = pd.read_parquet(META / "data_collection.parquet")
dc.index = dc.index.astype(str)
dc = dc.reset_index().rename(columns={"index": "clip_id"})
df = df.merge(dc[["clip_id", "country", "hour_of_day", "month"]], on="clip_id", how="left")

# ⚠️ DAY/NIGHT IS AN APPROXIMATION AND IS DECLARED AS ONE. The design names
# day/night as a stratum but defines no rule, and we hold no lat/lon (egomotion
# is clip-local metres), so true solar elevation is NOT computable. This is a
# fixed local-clock rule; it will mislabel shoulder hours and high-latitude
# summers. The field is named `daynight_clock` so nobody reads it as measured.
df["daynight_clock"] = np.where((df.hour_of_day >= 7) & (df.hour_of_day < 19),
                                "day", "night")
df["strata_cell"] = (df.road_class + "|" + df.daynight_clock + "|"
                     + df.country.fillna("UNKNOWN"))

print("\n=== road_class (DESIGN.md:17 rules) ===")
print((df.road_class.value_counts(normalize=True) * 100).round(1).to_string())
print("\n=== day/night (clock approximation) ===")
print((df.daynight_clock.value_counts(normalize=True) * 100).round(1).to_string())
print(f"\ndistinct strata cells occupied: {df.strata_cell.nunique()}")
print(f"  equal-weight design over 3 road x 2 daynight x {df.country.nunique()} countries "
      f"= {3*2*df.country.nunique()} cells")
occ = df.strata_cell.value_counts()
print(f"  cells with >=5 clips: {int((occ>=5).sum())} | median clips/cell {occ.median():.0f} "
      f"| max {occ.max()}")

print("\n=== ⭐ CODEVILLA INERTIA STATISTIC — never measured before ===")
sf = df.stopped_frame_frac
print(f"  stopped-frame fraction: mean {sf.mean()*100:.1f}%  median {sf.median()*100:.1f}%  "
      f"p90 {sf.quantile(.9)*100:.1f}%")
print(f"  clips >50% stopped: {int((sf>0.5).sum())} = {(sf>0.5).mean()*100:.1f}%")
print(f"  clips >90% stopped: {int((sf>0.9).sum())} = {(sf>0.9).mean()*100:.1f}%")
print(f"  clips with NO stopped frames: {int((sf==0).sum())} = {(sf==0).mean()*100:.1f}%")

df.to_parquet(OUT / "b1_strata_cells.parquet", index=False)
man = [{"clip_id": r.clip_id, "t0_us": 5100000, "strata_cell": r.strata_cell,
        "road_class": r.road_class, "daynight_clock": r.daynight_clock,
        "country": r.country, "stopped_frame_frac": r.stopped_frame_frac}
       for r in df.itertuples()]
json.dump(man, open(OUT / "selection_manifest_with_strata.json", "w"), indent=0)
print(f"\nwrote selection_manifest_with_strata.json ({len(man)} rows) "
      f"and b1_strata_cells.parquet")
