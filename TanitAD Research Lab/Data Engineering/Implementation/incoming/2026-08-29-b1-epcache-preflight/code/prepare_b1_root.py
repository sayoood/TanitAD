"""Build a corpus root the epcache builder can actually RESOLVE. PORTABLE.

⛔ THE LAYOUT IS NOT A PREFERENCE. `_physicalai_root_of` (physicalai.py:166)
recovers the corpus root by walking a clip path's parents for a directory named
**`r0`** and returns None otherwise. A camera bank at
`<root>/camera/camera_front_wide_120fov/` HAS NO `r0` ANCESTOR, so BOTH
`intrinsics_for_clip` and `extrinsics_for_clip` fall back SILENTLY -- the latter
returns None and callers then "treat the mount as level (optical axis ==
horizon)" (physicalai.py:450), with nothing raising. The mount pitch is what
locates the horizon, so that builds a horizon-wrong corpus with no error.

⇒ the camera bank MUST live at `<root>/r0/camera_front_wide_120fov/`.

And `_chunk_of_clip` reads `<root>/r0/r0_selection.parquet`; if it does not list
every corpus clip, extrinsics fall back for the ones it misses (MEASURED: a
500-row selection covered 38 of 4,719 -> 99.2 % silent fallback).

⛔ PARITY: this writes a NEW root and never touches a canonical corpus or its
r0_selection. The camera bank is ATTACHED, not copied (symlink on POSIX,
directory junction on Windows), so a 61.6 GB bank is not duplicated and a
running download is undisturbed.

    python prepare_b1_root.py --root <new root> --camera-src <dir of *.mp4>
        --index <clip_to_chunk.parquet> --clips <clip id source>
        --calib-src <dir holding camera_intrinsics/ + sensor_extrinsics/>
        [--intrinsics-csv <csv to copy in>]
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--camera-src", required=True)
ap.add_argument("--index", required=True)
ap.add_argument("--clips", required=True)
ap.add_argument("--calib-src", required=True)
ap.add_argument("--intrinsics-csv", default="")
ap.add_argument("--cam-name", default="camera_front_wide_120fov")
a = ap.parse_args()

root = pathlib.Path(a.root)
(root / "r0").mkdir(parents=True, exist_ok=True)
(root / "calibration").mkdir(parents=True, exist_ok=True)


def attach(link: pathlib.Path, target: pathlib.Path):
    """Symlink (POSIX) or directory junction (Windows). Never a copy."""
    if link.exists() or link.is_symlink():
        print(f"  attached already: {link.name}")
        return
    target = target.resolve()
    if os.name == "nt":
        # NOT text=True: mklink answers in the localised console codepage, which
        # blows up cp1252 decoding inside subprocess's reader thread.
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                           capture_output=True)
        if r.returncode or not link.exists():
            raise SystemExit(f"mklink failed for {link}: "
                             f"{(r.stdout + r.stderr).decode('utf-8', 'replace')}")
    else:
        link.symlink_to(target, target_is_directory=True)
    print(f"  attached {link.name} -> {target}")


def load_clip_ids(p: str) -> set:
    q = pathlib.Path(p)
    if q.suffix == ".jsonl":
        return {json.loads(x)["clip_id"] for x in open(q, encoding="utf-8") if x.strip()}
    if q.suffix == ".parquet":
        d = pd.read_parquet(q)
        return set(d["clip_id"].astype(str)) if "clip_id" in d.columns \
            else set(d.index.astype(str))
    return {x.strip() for x in open(q, encoding="utf-8") if x.strip()}


# --- 1. the r0 selection the chunk lookup needs ------------------------------
need = load_clip_ids(a.clips)
cix = pd.read_parquet(a.index)
if "clip_id" not in cix.columns:
    cix = cix.reset_index()
sel = cix[cix.clip_id.astype(str).isin(need)][["clip_id", "chunk"]].copy()
sel["clip_id"] = sel.clip_id.astype(str)
sel["chunk"] = sel.chunk.astype(int)
assert len(sel) == len(need), (
    f"r0_selection would cover {len(sel)} of {len(need)} clips — the missing ones "
    f"would fall back SILENTLY on extrinsics")
out = root / "r0" / "r0_selection.parquet"
sel.to_parquet(out, index=False)
print(f"wrote {out} — {len(sel)} clips, {sel.chunk.nunique()} chunks")

# --- 2. camera bank under an `r0` parent -------------------------------------
attach(root / "r0" / a.cam_name, pathlib.Path(a.camera_src))

# --- 3. calibration ----------------------------------------------------------
for kind in ("camera_intrinsics", "sensor_extrinsics"):
    src = pathlib.Path(a.calib_src) / kind
    if src.is_dir():
        attach(root / "calibration" / kind, src)
    else:
        print(f"  ⚠️ MISSING {src} — extrinsics/intrinsics will fall back SILENTLY")
if a.intrinsics_csv:
    csv = pathlib.Path(a.intrinsics_csv)
    dst = root / "calibration" / "physicalai_front_wide_intrinsics.csv"
    if csv.exists() and not dst.exists():
        dst.write_bytes(csv.read_bytes())
        print(f"  copied {csv.name} ({dst.stat().st_size/1e6:.2f} MB)")

# --- 4. prove it by RESOLVING, not by asserting the layout looks right -------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    from tanitad.data.physicalai import _physicalai_root_of
except ImportError:
    print("\n(tanitad not importable here — run preflight_epcache_build.py to verify)")
else:
    mp4 = next((root / "r0" / a.cam_name).glob("*.mp4"), None)
    rec = _physicalai_root_of(mp4) if mp4 else None
    ok = rec is not None and pathlib.Path(rec).resolve() == root.resolve()
    print(f"\nroot recovery from a clip path: {rec} -> {'OK' if ok else 'STILL BROKEN'}")
print(f"NEXT: python preflight_epcache_build.py --root {root} --sample 120")
