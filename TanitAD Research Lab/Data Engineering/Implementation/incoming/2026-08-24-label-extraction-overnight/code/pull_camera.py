"""Range-fetch camera video for a handful of clips, for the visual report.

The `camera_front_wide_120fov` chunks are **2.05 GB each**; a clip inside one is
~20 MB. Downloading a chunk to look at one clip is a 100x waste, so this reuses
the HTTP-range zip reader from `pull_egomotion_range.py`.

    python pull_camera.py <clip_id> [<clip_id> ...]

Writes `<clip>.mp4` into the camera store and skips what is already there.
"""
import truststore; truststore.inject_into_ssl()

import io
import os
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "pull_egomotion_range.py"), encoding="utf-8").read()
_g = {"__name__": "rangelib"}
exec(compile(_src.split("man = json.load")[0], "pull_egomotion_range.py", "exec"), _g)
HTTPRangeFile = _g["HTTPRangeFile"]

ROOT = "C:/Users/Admin/tanitad-data/physicalai"
OUT = f"{ROOT}/camera/camera_front_wide_120fov"
BASE = ("https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/"
        "resolve/main/camera/camera_front_wide_120fov/")
os.makedirs(OUT, exist_ok=True)


def pull(clip_ids: list[str]) -> dict[str, str]:
    ci = pd.read_parquet(f"{ROOT}/clip_index.parquet")
    by_chunk: dict[int, list[str]] = {}
    for cid in clip_ids:
        if os.path.exists(f"{OUT}/{cid}.mp4"):
            continue
        if cid not in ci.index:
            print(f"  {cid[:8]}: not in clip_index", flush=True)
            continue
        by_chunk.setdefault(int(ci.loc[cid, "chunk"]), []).append(cid)

    got: dict[str, str] = {}
    for chunk, ids in sorted(by_chunk.items()):
        url = f"{BASE}camera_front_wide_120fov.chunk_{chunk:04d}.zip"
        try:
            f = HTTPRangeFile(url)
            with zipfile.ZipFile(f) as z:
                names = z.namelist()
                for cid in ids:
                    member = next((m for m in names if m.split("/")[-1].startswith(cid)),
                                  None)
                    if not member:
                        print(f"  {cid[:8]}: not in chunk {chunk}", flush=True)
                        continue
                    data = z.read(member)
                    # Verify it is really a container before banking it: a short
                    # range read still "unzips". Checking the ftyp box is the
                    # content check, not the byte count.
                    if b"ftyp" not in data[:64]:
                        print(f"  {cid[:8]}: not an mp4 (no ftyp) — refused",
                              flush=True)
                        continue
                    p = f"{OUT}/{cid}.mp4"
                    with open(p + ".part", "wb") as fh:
                        fh.write(data)
                    os.replace(p + ".part", p)
                    got[cid] = p
                    print(f"  {cid[:8]}: {len(data)/1e6:.1f} MB  {member.split('/')[-1]}",
                          flush=True)
        except Exception as e:                            # noqa: BLE001
            print(f"  chunk {chunk} FAILED: {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
    return got


if __name__ == "__main__":
    ids = sys.argv[1:]
    if not ids:
        print(__doc__)
        raise SystemExit(2)
    out = pull(ids)
    print(f"pulled {len(out)}/{len(ids)}")
