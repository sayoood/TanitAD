"""Part 1 audit: what selected our 26 h, and how it sits against the parent.

Re-runnable. Every number in RESULT.md comes from here.

⛔ Reads the RELEASE blob (md5 pinned) — six copies of the labels exist across
three roots and they are not the same bytes.
"""
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release/tanitad-v7-training-corpus")
META = Path("C:/Users/Admin/tanitad-data/physicalai/metadata")
PINNED = "ee44875916ae7c0ac002c6716b9658ea"

blob = REL / "labels" / "s2_labels_v7.jsonl.gz"
md5 = hashlib.md5(blob.read_bytes()).hexdigest()
assert md5 == PINNED, f"not the release blob: {md5}"
b1 = {json.loads(x)["clip_id"] for x in gzip.open(blob, "rt", encoding="utf-8") if x.strip()}
print(f"B1 clips {len(b1)} (blob {md5[:12]})")

# --- 1. was B1 SELECTED, or is it an availability set? ----------------------
from tanitad.data import alpamayo_records as ARM        # noqa: E402
alp = set(ARM.available())
print(f"\nB1 within Alpamayo : {len(b1 & alp)} = {len(b1 & alp)/len(b1)*100:.2f}% of B1")
print(f"B1 outside Alpamayo: {len(b1 - alp)}   <- 0 means AVAILABILITY, not selection")
print(f"Alpamayo not in B1 : {len(alp - b1)}")

# --- 2. strata against the parent -------------------------------------------
dc = pd.read_parquet(META / "data_collection.parquet")
dc.index = dc.index.astype(str)
print(f"\nparent corpus {len(dc)} clips | catalogue columns: {list(dc.columns)}")
print("⚠️ no weather / road_type / traffic_density / surface column exists")
mine = dc[dc.index.isin(b1)]
for col in ("country", "hour_of_day", "platform_class"):
    p = dc[col].value_counts(normalize=True) * 100
    m = mine[col].value_counts(normalize=True) * 100
    j = pd.DataFrame({"parent %": p, "B1 %": m}).fillna(0.0)
    j["ratio"] = (j["B1 %"] / j["parent %"].replace(0, np.nan)).round(2)
    print(f"\n=== {col} ===")
    print(j.sort_values("parent %", ascending=False).head(10).round(2).to_string())

# --- 3. speed regime: the axis the metadata lacks ---------------------------
tf = tarfile.open(REL / "egomotion" / "egomotion_alpamayo.tar")
rows = []
for m in tf.getmembers():
    if not m.isfile():
        continue
    d = pd.read_parquet(io.BytesIO(tf.extractfile(m).read()), columns=["vx", "vy"])
    v = np.hypot(d.vx.to_numpy(), d.vy.to_numpy())
    rows.append((m.name.split(".")[0], float(np.nanmean(v)), float(np.nanpercentile(v, 95))))
sp = pd.DataFrame(rows, columns=["clip", "mean_ms", "p95_ms"])
BANDS = [(0, 2, "crawl <7 km/h"), (2, 8, "slow urban 7-29"), (8, 14, "urban/arterial 29-50"),
         (14, 20, "fast arterial 50-72"), (20, 99, "HIGHWAY >72 km/h")]
print(f"\n=== B1 speed regime (n={len(sp)}) ===")
for lo, hi, lab in BANDS:
    n = int(((sp.mean_ms >= lo) & (sp.mean_ms < hi)).sum())
    print(f"  {lab:24s} {n:5d}  {n/len(sp)*100:5.1f}%")

# --- 4. the DESIGNED corpus, for contrast ------------------------------------
r0p = Path("C:/Users/Admin/tanitad-data/physicalai/r0/r0_selection.parquet")
if r0p.exists():
    r0 = pd.read_parquet(r0p)
    print(f"\n=== r0 DESIGNED selection (local rows={len(r0)}) ===")
    print(f"  mean_speed median {r0.mean_speed.median():.2f} | max {r0.mean_speed.max():.2f} "
          f"| clips >14 m/s: {(r0.mean_speed > 14).sum()}")
    print("  ⛔ physicalai_r0.py:100 hard-gates mean_v <= 14.0 m/s (50.4 km/h),")
    print("     so highway scores 0 and is structurally unselectable.")
sp.to_parquet(Path(__file__).resolve().parent.parent / "raw" / "b1_speed_profile.parquet")
