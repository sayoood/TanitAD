"""fresh2_make_root.py -- PREREG_FRESH2 s1.2/s1.3: the FRESH2 pool (candidate ranks 421-1020) and its OWN fetched-root view.

The first pull's root F:/l1c/pai_root is shared with the fetch; l1c_build_inputs.step_prep REWRITES per-chunk egomotion zips, the intrinsics csv and
sel.parquet inside the root it is given.  So FRESH2 gets a NEW root F:/l1c/pai_root_fresh2 holding hard links (NTFS; copy fallback) to the fetched
FRESH2 files only -- the first pull's root and kits are never written.
  writes  F:/l1c/fresh2/fresh2_ids.json        [{clip_id, chunk}] in candidate (sha256 ascending) order      (raw ids; F: only)
          <pkg>/select/fresh2_pool_sha12.json  sha12 + rank + rig + chunk only                               (repo)
No clip id is printed.  NEW file (FRESH2 agent, 2026-10-27).
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

SEL = Path("F:/l1c/select")
SRC = Path("F:/l1c/pai_root")
DST = Path("F:/l1c/pai_root_fresh2")
OUT = Path("F:/l1c/fresh2")
PKG = Path(__file__).resolve().parent.parent
CAM = "camera_front_wide_120fov"
FIRST, COUNT = 420, 600


def link(src: Path, dst: Path) -> str:
    if dst.exists():
        if dst.stat().st_size != src.stat().st_size:
            raise SystemExit(f"[root] size mismatch on an existing file ({dst.name[-30:]})")
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copyfile(src, dst)
        return "copy"


def main():
    cand = [json.loads(l) for l in open(SEL / "candidates.jsonl", encoding="utf-8")]
    pool = cand[FIRST:FIRST + COUNT]
    assert len(pool) == COUNT and [int(r["rank"]) for r in pool][:1] and len({r["sha12"] for r in pool}) == COUNT
    for r in pool:
        assert hashlib.sha256(r["clip_id"].encode()).hexdigest()[:12] == r["sha12"]
    shas = [hashlib.sha256(r["clip_id"].encode()).hexdigest() for r in pool]
    assert shas == sorted(shas), "pool is not in sha256 order"
    man = {}
    for l in open(SEL / "fetch_manifest.jsonl", encoding="utf-8"):
        x = json.loads(l)
        if x.get("ok"):
            man[x["sha12"]] = x
    fetched = [r for r in pool if r["sha12"] in man]
    print(f"[root] pool {len(pool)}  fetch-ok {len(fetched)}  ranks {pool[0]['rank']}..{pool[-1]['rank']}", flush=True)
    how = {"link": 0, "copy": 0, "exists": 0}
    for r in fetched:
        c, ch = r["clip_id"], int(r["chunk"])
        for rel in (f"r0/camera_front_wide/{c}.{CAM}.mp4", f"r0/camera_front_wide/{c}.{CAM}.timestamps.parquet", f"labels/egomotion_flat/{c}.parquet",
                    f"calibration/camera_intrinsics/camera_intrinsics.chunk_{ch:04d}.parquet",
                    f"calibration/sensor_extrinsics/sensor_extrinsics.chunk_{ch:04d}.parquet"):
            s = SRC / rel
            if not s.exists() or s.stat().st_size == 0:
                raise SystemExit(f"[root] missing fetched member for {r['sha12']}: {rel.split('/')[0]}/{rel.split('/')[-1][-28:]}")
            how[link(s, DST / rel)] += 1
        mp4 = DST / f"r0/camera_front_wide/{c}.{CAM}.mp4"
        if hashlib.sha256(mp4.read_bytes()).hexdigest() != man[r["sha12"]]["mp4_sha256"]:
            raise SystemExit(f"[root] mp4 sha256 != fetch manifest for {r['sha12']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump([{"clip_id": r["clip_id"], "chunk": int(r["chunk"])} for r in fetched], open(OUT / "fresh2_ids.json", "w", encoding="utf-8"))
    pub = {"prereg": "PREREG_FRESH2.md s1 (sha256 956f87cd...)", "pool_ranks": [FIRST + 1, FIRST + COUNT], "n_pool": len(pool), "n_fetch_ok": len(fetched),
           "rows": [{"rank": int(r["rank"]), "sha12": r["sha12"], "rig": r["rig"], "chunk": int(r["chunk"]), "fetch_ok": r["sha12"] in man} for r in pool]}
    (PKG / "select").mkdir(exist_ok=True)
    (PKG / "select" / "fresh2_pool_sha12.json").write_text(json.dumps(pub, indent=1), encoding="utf-8")
    print(f"[root] {DST}: {how}  mp4 sha256 re-verified {len(fetched)}/{len(fetched)}  ZZF2ROOTZZ", flush=True)


if __name__ == "__main__":
    main()
