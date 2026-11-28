#!/usr/bin/env python3
"""Byte-verify the C: runtime mirror's DATA against its D: sources (sha256, every file).
The devkit SOURCE copies are verified separately against git blobs (see RESULT.md §harness).
Usage: python verify_mirror.py <out.json> [<src_root> <dst_root> <relative dir or file>]...
Default pairs = the warmup set."""
import hashlib, json, os, sys, time
from pathlib import Path

SRC = Path("D:/Archive/devbox-C/navsim/data"); DST = Path("C:/Users/Admin/navsim-crun/data")
DEFAULT = ["maps/sg-one-north", "maps/us-ma-boston", "maps/us-pa-pittsburgh-hazelwood", "maps/nuplan-maps-v1.0.json",
           "openscene/warmup_two_stage/synthetic_scene_pickles"] + [
           f"openscene/navsim_logs/test/{n}.pkl" for n in (
               "2021.08.16.14.23.37_veh-45_00015_00132", "2021.08.30.14.54.34_veh-40_00439_00835",
               "2021.09.16.13.53.10_veh-42_00180_00342", "2021.09.16.15.47.30_veh-45_01199_01391",
               "2021.09.16.19.27.01_veh-45_01749_03230", "2021.09.29.19.02.14_veh-28_02911_03005",
               "2021.10.06.08.16.17_veh-52_01590_01725")]

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def files_under(rel: str):
    s = SRC / rel
    if s.is_file():
        return [rel]
    return sorted(str(Path(rel) / p.relative_to(s)).replace("\\", "/") for p in s.rglob("*") if p.is_file())

out = Path(sys.argv[1]); rels = sys.argv[2:] or DEFAULT
t0 = time.time(); rows = []; bad = []; n_bytes = 0
for rel in rels:
    for f in files_under(rel):
        a, b = SRC / f, DST / f
        ha = sha(a); hb = sha(b) if b.exists() else None
        rows.append({"file": f, "bytes": a.stat().st_size, "sha256": ha, "match": ha == hb})
        n_bytes += a.stat().st_size
        if ha != hb:
            bad.append(f)
res = {"src_root": str(SRC), "dst_root": str(DST), "n_files": len(rows), "n_bytes": n_bytes,
       "n_mismatch": len(bad), "mismatch": bad, "wall_s": round(time.time() - t0, 1), "files": rows}
out.write_text(json.dumps(res, indent=1))
print(f"verified {len(rows)} files, {n_bytes/2**20:.1f} MiB, mismatches={len(bad)} in {res['wall_s']} s")
sys.exit(1 if (bad or not rows) else 0)
