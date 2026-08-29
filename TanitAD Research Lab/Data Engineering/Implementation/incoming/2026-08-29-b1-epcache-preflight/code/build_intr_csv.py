"""Build `physicalai_front_wide_intrinsics.csv` for the B1 4,719-clip corpus.

WHY THIS EXISTS (measured 2026-08-29): the epcache build for the NEW corpus
cannot resolve per-clip intrinsics -- `intrinsics_for_clip` returned the
corpus-median fallback for 200/200 sampled clips, and `cylindrical_rectify`
then correctly REFUSED (a single global cy is a rig-B value; using it on rig A
misplaces the horizon by ~215 px). So the B1 epcache was blocked, and the only
ways past the guard were a crash or `require_per_clip=False`, i.e. silently
building most of the corpus with the rig fix DISABLED.

The data was never missing -- 1,420 chunk parquets are already on disk. What was
missing is the clip->chunk mapping the resolver needs, which the B1 bundle
carries (`index/clip_to_chunk.parquet`). This writes resolution path (1), the
local CSV table, which takes precedence and carries per_clip=True.
"""
import pathlib

import pandas as pd

ROOT = pathlib.Path("C:/Users/Admin/tanitad-data/physicalai")
CAL = ROOT / "calibration" / "camera_intrinsics"
OUT = ROOT / "calibration" / "physicalai_front_wide_intrinsics.csv"
BUNDLE = pathlib.Path("C:/Users/Admin/tanitad-wt/_s2build/release/tanitad-v7-training-corpus")
CAM = "camera_front_wide_120fov"
COLS = ["cx", "cy", "width", "height"] + [f"fw_poly_{i}" for i in range(5)]

cix = pd.read_parquet(BUNDLE / "index" / "clip_to_chunk.parquet")
want = {str(c) for c in cix.index}
print(f"corpus clips: {len(want)} | chunk parquets on disk: "
      f"{len(list(CAL.glob('*.parquet')))}")

rows = []
for p in sorted(CAL.glob("*.parquet")):
    d = pd.read_parquet(p)
    if isinstance(d.index, pd.MultiIndex):
        d = d.reset_index()
    if "camera_name" in d.columns:
        d = d[d.camera_name == CAM]
    d["clip_id"] = d["clip_id"].astype(str)
    d = d[d.clip_id.isin(want)]
    if len(d):
        rows.append(d[["clip_id"] + COLS])

df = pd.concat(rows).drop_duplicates("clip_id")
print(f"resolved: {len(df)}/{len(want)}")
missing = want - set(df.clip_id)
assert not missing, f"{len(missing)} clips have no intrinsics, e.g. {list(missing)[:3]}"

# CONTENT assertions -- a table that exists is not a table that is right.
assert df[[f"fw_poly_{i}" for i in range(5)]].abs().sum(1).gt(0).all(), "zero poly rows"
assert df.width.eq(1920).all() and df.height.eq(1080).all(), "unexpected sensor size"
rigA = df[df.cy < 650]
rigB = df[df.cy >= 650]
print(f"rig A (cy<650): {len(rigA)} mean cy {rigA.cy.mean():.1f} | "
      f"rig B: {len(rigB)} mean cy {rigB.cy.mean():.1f}")
# cross-check against the INDEPENDENTLY built B1 cy table
cy = pd.read_parquet(BUNDLE / "index" / "front_wide_cy.parquet")
m = df.merge(cy[["clip_id", "cy"]], on="clip_id", suffixes=("", "_b1"))
d = (m.cy - m.cy_b1).abs()
print(f"cross-check vs B1 cy table: n={len(m)} max|dcy|={d.max():.6f} "
      f"-> {'MATCH' if d.max() < 1e-6 else 'MISMATCH'}")
assert d.max() < 1e-6

df.to_csv(OUT, index=False)
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB, {len(df)} rows)")
