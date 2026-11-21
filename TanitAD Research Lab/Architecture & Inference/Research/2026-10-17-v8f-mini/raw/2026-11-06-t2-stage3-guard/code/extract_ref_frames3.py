"""Stage 3, item 2 (AGENT K): `ref_frames.npz` for the 20 guard scenes, and the stage-1 bit-for-bit control re-run.

    python -I extract_ref_frames3.py [--check-only] [--workers 3]

Everything is `extract_ref_frames.py`'s (stage 2, AGENT J): frames 0, 60 and 120 of `camera_front_wide_120fov.mp4` -> `<scene folder>/ref_frames.npz` with `t2v8f_nocv2.extract_frames` (the function
stage 1 used), after the mp4's sha256 is checked against the listing's `lfs_sha256`. Those are the only mp4 reads of a stage-2/3 cell (`attach_actors_verified(frames=(0, 60, 120))`).
First the CONTROL: stage 1's 7c72937c npz must be reproduced array for array from its mp4 by this very method (`ERF.bitcheck`); if it is not, nothing is extracted.
Writes stage3/ref_frames/{BITCHECK_7c72937c.json, REF_FRAMES_SOURCES3.json, EXTRACT_RESULT3.json}.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import extract_ref_frames as ERF  # noqa: E402

T2 = ERF.T2
OUT = f"{T2}/stage3/ref_frames"
ERF.OUT = OUT                                                                    # `bitcheck()` writes BITCHECK_7c72937c.json into the module's OUT: stage 3's folder, stage 2's record stays


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    chk = ERF.bitcheck()
    print(json.dumps({"bitcheck_all_bit_equal": chk["all_bit_equal"], "n_arrays": chk["n_arrays"], "frames_0_60_120": chk["stage2_frames_0_60_120_bit_equal"]}), flush=True)
    assert chk["all_bit_equal"], "stage 1's frames are not reproduced bit for bit: STOP, the extraction method is not the one stage 1 used"
    if a.check_only:
        return 0
    dl = json.load(open(f"{OUT}/DOWNLOAD_RESULT3.json", encoding="utf-8"))
    assert dl["n_failed"] == 0 and dl["n_ok"] == 20, f"the download is not complete: {dl['n_ok']}/{dl['n_scenes']}"
    lst = json.load(open(f"{T2}/../t2_scenes/raw/RELEASE_LISTING.json", encoding="utf-8"))["scenes"]
    rows, t0 = {}, time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(ERF.one, sid, os.path.dirname(o["path"]), lst[sid][ERF.MP4]["lfs_sha256"]): sid for sid, o in dl["scenes"].items()}
        for f in cf.as_completed(futs):
            r = f.result()
            rows[r["scene"]] = r
            print(f"{r['scene'][:8]} {r['bytes']} B {r['shapes']['f0']} {r['seconds']} s", flush=True)
    assert len(rows) == 20
    src = {sid: {"path": r["path"], "bytes": r["bytes"], "sha256": r["sha256"]} for sid, r in rows.items()}
    with open(f"{OUT}/REF_FRAMES_SOURCES3.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(src, f, indent=1)
    with open(f"{OUT}/EXTRACT_RESULT3.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"n_scenes": len(rows), "frames": list(ERF.REF_IDX), "total_npz_bytes": sum(r["bytes"] for r in rows.values()), "seconds": round(time.time() - t0, 1), "scenes": rows}, f, indent=1)
    print(json.dumps({"extracted": len(rows), "total_npz_bytes": sum(r["bytes"] for r in rows.values())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
