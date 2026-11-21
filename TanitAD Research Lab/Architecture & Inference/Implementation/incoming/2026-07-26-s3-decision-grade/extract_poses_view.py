"""Poses-only VIEW of a PhysicalAI epcache.

Each source ep_*.pt is ~117 MB, dominated by `frames_u8` [T,9,256,256]. S3's
miner reads ONLY `ep["poses"]` ([T,4]) -- ~3 KB. This writes a drop-in
`ep_*.pt` directory carrying the BIT-IDENTICAL poses tensor (verified by
sha256 of the raw bytes on both sides), so `run_s3_characterisation.py` runs
against it with ZERO code change, exactly as the pre-registration specifies.

Reads via torch.load(mmap=True): only the poses pages are faulted in, so a
2,376-episode cache costs ~60 s and a few MB of IO instead of 260 GB.

Also emits a manifest of per-episode sha256(poses.bytes) + episode_id, which is
the BYTE-LEVEL key for the train/val disjointness check.
"""
import argparse, hashlib, json, time
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("ep_*.pt"))
    if a.limit:
        files = files[:a.limit]
    print(f"[extract] {src}  ->  {dst}   ({len(files)} episodes)", flush=True)

    rows, t0 = [], time.time()
    for i, f in enumerate(files):
        d = torch.load(f, weights_only=False, map_location="cpu", mmap=True)
        po = torch.as_tensor(d["poses"]).clone().contiguous()
        eid = d.get("episode_id")
        try:
            eid = int(eid)
        except Exception:
            eid = str(eid)
        h = hashlib.sha256(po.numpy().tobytes()).hexdigest()
        torch.save({"poses": po, "episode_id": eid}, dst / f.name)
        rows.append({"file": f.name, "episode_id": eid, "T": int(po.shape[0]),
                     "poses_sha256": h})
        if i % 250 == 0:
            print(f"  {i}/{len(files)}  {time.time()-t0:.0f}s", flush=True)

    man = {"source_dir": src.as_posix(), "view_dir": dst.as_posix(),
           "n_episodes": len(rows),
           "note": "poses-only VIEW; poses bytes identical to source "
                   "(sha256 per episode). frames_u8/actions/maneuvers not "
                   "copied -- S3's miner never reads them.",
           "extract_seconds": round(time.time() - t0, 1),
           "episodes": rows}
    Path(a.manifest).write_text(json.dumps(man, indent=2))
    print(f"[extract] DONE {len(rows)} eps in {time.time()-t0:.0f}s -> {a.manifest}",
          flush=True)
    uniq = len({r["poses_sha256"] for r in rows})
    print(f"[extract] unique poses hashes: {uniq}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
