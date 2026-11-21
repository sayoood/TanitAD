#!/usr/bin/env python3
"""Extract ONLY the 40 val40 clips' obstacle.offline + egomotion parquets into one bundle.

Shipping the 37 whole chunk zips would be 3.1 GB / ~25 min at the measured 2.1 MB/s LAN rate to
Thor. The val40 episodes touch 40 clips out of the ~3,700 those chunks hold, so the per-clip
members are a tiny fraction. This builds `val40_lead_bundle.zip` with a flat layout:

    obstacle/<clip_id>.parquet
    egomotion/<clip_id>.parquet
    INDEX.json     ep_XXXXX.pt -> {clip_id, chunk, T}

PARITY: read-only over labels/*.zip and r0/phase0_selection.parquet. No clip is re-selected.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path("C:/Users/Admin/tanitad-data/physicalai")
SCR = Path(__file__).resolve().parent
CANDS = json.loads((SCR / "val40_clip_cands.json").read_text())
OUT = SCR / "val40_lead_bundle.zip"


def member(z: zipfile.ZipFile, clip: str, suffix: str) -> str | None:
    for n in z.namelist():
        if n.endswith(".parquet") and n.split("/")[-1].startswith(clip):
            return n
    return None


def main() -> None:
    index = {}
    by_chunk: dict[int, list[tuple[str, str]]] = {}
    for ep, v in CANDS.items():
        assert len(v["cands"]) == 1, f"{ep}: {len(v['cands'])} candidate clips"
        clip, chunk = v["cands"][0]
        index[ep] = {"clip_id": clip, "chunk": int(chunk), "T": int(v["T"]),
                     "prefix": v["prefix"]}
        by_chunk.setdefault(int(chunk), []).append((ep, clip))

    n_ob = n_eg = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for chunk in sorted(by_chunk):
            obz = ROOT / "labels" / "obstacle.offline" / f"obstacle.offline.chunk_{chunk:04d}.zip"
            egz = ROOT / "labels" / "egomotion" / f"egomotion.chunk_{chunk:04d}.zip"
            with zipfile.ZipFile(obz) as zo, zipfile.ZipFile(egz) as ze:
                for ep, clip in by_chunk[chunk]:
                    mo, me = member(zo, clip, "obstacle"), member(ze, clip, "egomotion")
                    if mo is None:
                        index[ep]["obstacle"] = "ABSENT_IN_CHUNK"
                        print(f"  {ep} {clip[:8]} chunk {chunk:04d}: NO obstacle member")
                    else:
                        b = zo.read(mo)
                        out.writestr(f"obstacle/{clip}.parquet", b)
                        index[ep]["obstacle"] = "present"
                        index[ep]["obstacle_sha256"] = hashlib.sha256(b).hexdigest()
                        index[ep]["obstacle_bytes"] = len(b)
                        n_ob += 1
                    if me is None:
                        index[ep]["egomotion"] = "ABSENT_IN_CHUNK"
                        print(f"  {ep} {clip[:8]} chunk {chunk:04d}: NO egomotion member")
                    else:
                        b = ze.read(me)
                        out.writestr(f"egomotion/{clip}.parquet", b)
                        index[ep]["egomotion"] = "present"
                        index[ep]["egomotion_sha256"] = hashlib.sha256(b).hexdigest()
                        index[ep]["egomotion_bytes"] = len(b)
                        n_eg += 1
            print(f"chunk {chunk:04d} done ({len(by_chunk[chunk])} clip(s))", flush=True)
        out.writestr("INDEX.json", json.dumps(index, indent=1))

    (SCR / "val40_lead_index.json").write_text(json.dumps(index, indent=1))
    size = OUT.stat().st_size
    print(f"\nWROTE {OUT}  {size/1e6:.2f} MB   obstacle={n_ob}/40  egomotion={n_eg}/40")
    print(f"sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    print(f"at 2.1 MB/s -> {size/1e6/2.1:.0f} s")


if __name__ == "__main__":
    main()
