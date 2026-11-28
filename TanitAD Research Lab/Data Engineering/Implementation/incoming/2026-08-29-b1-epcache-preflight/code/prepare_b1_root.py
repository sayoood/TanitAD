"""Build a corpus root the epcache builder can actually RESOLVE. PORTABLE.

⛔ THE LAYOUT IS NOT A PREFERENCE. `_physicalai_root_of` (physicalai.py:166)
recovers the corpus root by walking a clip path's parents for a directory named
**`r0`** and returns None otherwise. A camera bank at
`<root>/camera/camera_front_wide_120fov/` HAS NO `r0` ANCESTOR, so BOTH
`intrinsics_for_clip` and `extrinsics_for_clip` fall back SILENTLY -- the latter
returns None and callers then "treat the mount as level (optical axis ==
horizon)" (physicalai.py:450), with nothing raising. The mount pitch is what
locates the horizon, so that builds a horizon-wrong corpus with no error.

⇒ the camera bank MUST live at `<root>/r0/camera_front_wide/` — the `r0`
ancestor for root recovery, and that EXACT directory name for discovery.

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
import gzip
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import zipfile

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--camera-src", required=True)
ap.add_argument("--index", required=True)
ap.add_argument("--clips", required=True)
ap.add_argument("--calib-src", required=True)
ap.add_argument("--intrinsics-csv", default="")
ap.add_argument("--cam-name", default="camera_front_wide",
                help="MUST be `camera_front_wide` — discovery globs that exact "
                     "directory name (physicalai.py:466); `_120fov` is the FEATURE "
                     "name and belongs in the FILE names only")
ap.add_argument("--timestamps-tar", default="",
                help="the bundle's timestamps.tar; extracted BESIDE the mp4s, "
                     "which is where discovery derives them by name")
ap.add_argument("--egomotion-tar", default="",
                help="the bundle's egomotion_alpamayo.tar; repackaged locally into "
                     "labels/egomotion/egomotion.chunk_NNNN.zip")
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
    """⚠️ .gz is not optional: the release bundle ships s2_labels_v7.jsonl.GZ, and
    the earlier version fell through to the plain-text branch and died on the
    gzip magic byte."""
    q = pathlib.Path(p)
    sfx = q.suffixes
    if ".jsonl" in sfx:
        op = (gzip.open(q, "rt", encoding="utf-8") if q.suffix == ".gz"
              else open(q, encoding="utf-8"))
        with op as fh:
            return {json.loads(x)["clip_id"] for x in fh if x.strip()}
    if q.suffix == ".parquet":
        d = pd.read_parquet(q)
        return set(d["clip_id"].astype(str)) if "clip_id" in d.columns \
            else set(d.index.astype(str))
    op = (gzip.open(q, "rt", encoding="utf-8") if q.suffix == ".gz"
          else open(q, encoding="utf-8"))
    with op as fh:
        return {x.strip() for x in fh if x.strip()}


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

# --- 2. camera bank at the name DISCOVERY expects ----------------------------
# ⛔ `discover_r0_clips` (physicalai.py:466) globs
# `root/r0/camera_front_wide/**/*.mp4` — the directory is literally
# `camera_front_wide`, NOT `camera_front_wide_120fov` (that string is the FEATURE
# name and appears in the FILE names, not the directory). MEASURED: with the
# _120fov directory name, discovery returned 0 clips SILENTLY and `measure`
# printed "0 clips" instead of naming a missing path.
attach(root / "r0" / a.cam_name, pathlib.Path(a.camera_src))

# --- 2a. timestamps BESIDE the mp4s -----------------------------------------
# `discover_r0_clips` derives the timestamps path from the mp4 path by
# `name.replace(".mp4", ".timestamps.parquet")` — so they must sit in the SAME
# directory, named to match the mp4s. Extracted from the bundle tar; no network.
if a.timestamps_tar:
    camdir = pathlib.Path(a.camera_src)
    with tarfile.open(a.timestamps_tar) as tf:
        n = 0
        for m in tf.getmembers():
            if not m.isfile():
                continue
            dst = camdir / pathlib.Path(m.name).name
            if dst.exists():
                continue
            data = tf.extractfile(m).read()
            assert data[:4] == b"PAR1" and data[-4:] == b"PAR1", f"{m.name}: not parquet"
            dst.write_bytes(data)
            n += 1
    tot = len(list(camdir.glob("*.timestamps.parquet")))
    print(f"  timestamps: {n} extracted, {tot} now beside the mp4s")

# --- 2b. egomotion: one .tar -> the per-chunk .zip layout discovery wants -----
# `discover_r0_clips` requires root/labels/egomotion/egomotion.chunk_NNNN.zip and
# `load_egomotion` reads the member ENDING IN `<clip>.egomotion.parquet`. Our
# bundle ships a single tar whose members are `<clip>.parquet` — so both the
# container and the member name have to change. Done LOCALLY: no network.
if a.egomotion_tar:
    egodir = root / "labels" / "egomotion"
    egodir.mkdir(parents=True, exist_ok=True)
    chunk_of = dict(zip(sel.clip_id, sel.chunk))
    with tarfile.open(a.egomotion_tar) as tf:
        members = {m.name.split(".")[0]: m for m in tf.getmembers() if m.isfile()}
        by_chunk: dict[int, list] = {}
        for cid, m in members.items():
            if cid in chunk_of:
                by_chunk.setdefault(int(chunk_of[cid]), []).append((cid, m))
        for ch, items in sorted(by_chunk.items()):
            zp = egodir / f"egomotion.chunk_{ch:04d}.zip"
            if zp.exists():
                continue
            with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
                for cid, m in items:
                    data = tf.extractfile(m).read()
                    assert data[:4] == b"PAR1", f"{cid}: not a parquet"
                    z.writestr(f"{cid}.egomotion.parquet", data)
    n_zip = len(list(egodir.glob("*.zip")))
    print(f"  egomotion: {len(members)} clips -> {n_zip} chunk zips in {egodir}")

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

# --- 4. prove it by RESOLVING and by DISCOVERING -----------------------------
# ⛔ TWO different questions, and instrumenting only the first is what let a
# 0-clip build reach Thor: root RECOVERY is about calibration, clip DISCOVERY is
# about layout. `discover_r0_clips` returns [] SILENTLY when the layout is wrong.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    from tanitad.data.physicalai import _physicalai_root_of, discover_r0_clips
except ImportError:
    print("(tanitad not importable here — run preflight_epcache_build.py to verify)")
else:
    mp4 = next((root / "r0" / a.cam_name).glob("*.mp4"), None)
    rec = _physicalai_root_of(mp4) if mp4 else None
    ok = rec is not None and pathlib.Path(rec).resolve() == root.resolve()
    print(f"root recovery: {rec} -> {'OK' if ok else 'STILL BROKEN'}")
    clips = discover_r0_clips(root)
    print(f"discover_r0_clips: {len(clips)}/{len(need)} clips")
    if len(clips) != len(need):
        mp4s = len(list((root / "r0" / a.cam_name).glob("*.mp4")))
        ts = len(list((root / "r0" / a.cam_name).glob("*.timestamps.parquet")))
        zips = len(list((root / "labels" / "egomotion").glob("*.zip")))
        raise SystemExit(
            f"⛔ DISCOVERY SHORT: {len(clips)} of {len(need)}. Discovery needs ALL "
            f"THREE per clip and reports none of them individually:"
            f"\n   mp4 under r0/{a.cam_name}/             {mp4s}"
            f"\n   <clip>.timestamps.parquet beside them  {ts}"
            f"\n   labels/egomotion/*.zip                 {zips}"
            f"\nWhichever is short is the cause; discovery itself returns [] silently.")
    print("✅ discovery OK — the build can start")
print(f"NEXT: python preflight_epcache_build.py --root {root} --sample 120")
