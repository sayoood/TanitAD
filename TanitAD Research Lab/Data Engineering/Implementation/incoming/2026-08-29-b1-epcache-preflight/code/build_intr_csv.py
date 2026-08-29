"""Build `physicalai_front_wide_intrinsics.csv` for the B1 corpus. PORTABLE.

WHY THIS EXISTS (measured 2026-08-29): the epcache build could not resolve
per-clip intrinsics -- `intrinsics_for_clip` returned the corpus-median fallback
for 200/200 sampled clips, and `cylindrical_rectify` then correctly REFUSED (a
single global cy is a rig-B value; on rig A the horizon lands ~215 px wrong).
The only ways past that guard were a crash or `require_per_clip=False`, i.e.
silently building most of the corpus with the rig fix DISABLED.

The data was never missing -- the chunk parquets are on disk; what was missing is
the clip->chunk mapping, which the release bundle carries. This writes resolution
path (1), the local CSV table, which takes precedence and carries per_clip=True.

    python build_intr_csv.py --root <corpus root> --index <clip_to_chunk.parquet>
                             --clips <clip id source> [--cy-check <cy parquet>]
"""
import argparse
import json
import pathlib

import pandas as pd

CAM = "camera_front_wide_120fov"
COLS = ["cx", "cy", "width", "height"] + [f"fw_poly_{i}" for i in range(5)]

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True,
                help="corpus root; reads <root>/calibration/camera_intrinsics/, "
                     "writes <root>/calibration/physicalai_front_wide_intrinsics.csv")
ap.add_argument("--index", required=True, help="clip_to_chunk.parquet")
ap.add_argument("--clips", required=True,
                help="clip id source: .jsonl with clip_id, .parquet with a "
                     "clip_id column/index, or newline-separated .txt")
ap.add_argument("--cy-check", default="",
                help="optional front_wide_cy.parquet — cross-checks cy exactly")
a = ap.parse_args()

root = pathlib.Path(a.root)
cal = root / "calibration" / "camera_intrinsics"
out = root / "calibration" / "physicalai_front_wide_intrinsics.csv"
assert cal.is_dir(), f"missing {cal}"


def load_clip_ids(p: str) -> set:
    q = pathlib.Path(p)
    if q.suffix == ".jsonl":
        return {json.loads(x)["clip_id"] for x in
                open(q, encoding="utf-8") if x.strip()}
    if q.suffix == ".parquet":
        d = pd.read_parquet(q)
        if "clip_id" in d.columns:
            return set(d["clip_id"].astype(str))
        return set(d.index.astype(str))
    return {x.strip() for x in open(q, encoding="utf-8") if x.strip()}


want = load_clip_ids(a.clips)
chunks = sorted(cal.glob("*.parquet"))
print(f"corpus clips: {len(want)} | chunk parquets on disk: {len(chunks)}")

rows = []
for p in chunks:
    d = pd.read_parquet(p)
    if isinstance(d.index, pd.MultiIndex) or d.index.name:
        d = d.reset_index()
    if "camera_name" in d.columns:
        d = d[d.camera_name == CAM]
    d["clip_id"] = d["clip_id"].astype(str)
    d = d[d.clip_id.isin(want)]
    if len(d):
        rows.append(d[["clip_id"] + COLS])

assert rows, f"no rows matched under {cal}"
df = pd.concat(rows).drop_duplicates("clip_id")
print(f"resolved: {len(df)}/{len(want)}")
missing = want - set(df.clip_id)
assert not missing, f"{len(missing)} clips have no intrinsics, e.g. {sorted(missing)[:3]}"

# CONTENT assertions -- a table that exists is not a table that is right.
assert df[[f"fw_poly_{i}" for i in range(5)]].abs().sum(axis=1).gt(0).all(), "zero poly rows"
assert df.width.eq(1920).all() and df.height.eq(1080).all(), "unexpected sensor size"
rigA, rigB = df[df.cy < 650], df[df.cy >= 650]
print(f"rig A (cy<650): {len(rigA)} mean cy {rigA.cy.mean():.1f} | "
      f"rig B: {len(rigB)} mean cy {rigB.cy.mean():.1f}")
assert len(rigA) and len(rigB), "expected BOTH rigs; one is missing"

if a.cy_check:
    cy = pd.read_parquet(a.cy_check)
    m = df.merge(cy[["clip_id", "cy"]], on="clip_id", suffixes=("", "_ref"))
    dmax = (m.cy - m.cy_ref).abs().max()
    print(f"cross-check vs cy table: n={len(m)} max|dcy|={dmax:.6f} "
          f"-> {'MATCH' if dmax < 1e-6 else 'MISMATCH'}")
    assert dmax < 1e-6, "cy disagrees with the independently built table"

out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB, {len(df)} rows)")
