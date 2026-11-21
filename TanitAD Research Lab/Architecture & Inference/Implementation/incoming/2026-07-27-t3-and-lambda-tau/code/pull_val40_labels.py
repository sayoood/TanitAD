"""Pull ONLY the 40 eval-val clips' obstacle.offline + egomotion parquets.

Uses HTTP RANGE reads through HfFileSystem so a 64 MB chunk zip costs one
central-directory read plus the members we actually need (~0.6 MB each), not
4 GB of chunk downloads.

Token read IN PLACE from Keys.txt; never printed, never written to disk.
Output is raw gated content -> scratchpad only, NEVER the repo.
"""
import io
import json
import re
import time
import zipfile
from pathlib import Path

import truststore
truststore.inject_into_ssl()
from huggingface_hub import HfFileSystem, hf_hub_download

HERE = Path(__file__).resolve().parent
KEYS = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/Keys.txt")
TOK = re.search(r"hf_[A-Za-z0-9_]+", KEYS.read_text()).group(0)
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
FSREPO = f"datasets/{REPO}"
OUT = HERE / "pai_val40"
OUT.mkdir(exist_ok=True)

rows = json.load(open(HERE / "val40_map.json"))
by_chunk: dict[int, list[dict]] = {}
for r in rows:
    by_chunk.setdefault(int(r["chunk"]), []).append(r)

fs = HfFileSystem(token=TOK)
t0 = time.time()
got = {"obstacle.offline": 0, "egomotion": 0, "vehicle_dimensions": 0}
for ch in sorted(by_chunk):
    want = {r["clip_id"]: r for r in by_chunk[ch]}
    for kind in ("obstacle.offline", "egomotion"):
        dst_all = [OUT / f"{r['alias']}.{kind}.parquet" for r in by_chunk[ch]]
        if all(d.exists() for d in dst_all):
            got[kind] += len(dst_all)
            continue
        p = f"{FSREPO}/labels/{kind}/{kind}.chunk_{ch:04d}.zip"
        with fs.open(p, "rb") as f:
            z = zipfile.ZipFile(f)
            names = {n.split("/")[-1].split(".")[0]: n for n in z.namelist()
                     if n.endswith(".parquet")}
            for cid, r in want.items():
                dst = OUT / f"{r['alias']}.{kind}.parquet"
                if dst.exists():
                    got[kind] += 1
                    continue
                if cid not in names:
                    print(f"  MISS {kind} chunk {ch} {r['alias']}")
                    continue
                dst.write_bytes(z.read(names[cid]))
                got[kind] += 1
    # vehicle dimensions: a small per-chunk parquet, plain download
    vd = OUT / f"vehicle_dimensions.chunk_{ch:04d}.parquet"
    if not vd.exists():
        src = hf_hub_download(
            REPO, f"calibration/vehicle_dimensions/vehicle_dimensions.chunk_{ch:04d}.parquet",
            repo_type="dataset", token=TOK)
        vd.write_bytes(Path(src).read_bytes())
    got["vehicle_dimensions"] += 1
    print(f"[chunk {ch:04d}] {len(want)} clips  ({time.time()-t0:.0f}s)", flush=True)

print(json.dumps(got, indent=1))
tot = sum(f.stat().st_size for f in OUT.glob("*"))
print("total bytes", tot, "files", len(list(OUT.glob('*'))))
