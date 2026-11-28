"""Extract the reference frames of the 41 stage-2 scenes into `ref_frames.npz` (AGENT J) and check the method against stage 1.

    python -I extract_ref_frames.py [--check-only] [--workers 3]

For every scene of DOWNLOAD_RESULT.json (mp4 verified against `lfs_sha256`): frames 0, 60 and 120 of `camera_front_wide_120fov.mp4` -> `<scene folder>/ref_frames.npz`, with
`t2v8f_nocv2.extract_frames` (the function stage 1 used). Those are the only mp4 reads of a stage-2 cell: `actor_map.attach_actors_verified(frames=(0, 60, 120))`. The stage-2 driver has no
gap job (that was stage 1: `gap_ticks`), so no other frame is read; this is stated in the manifest (`frames`).

BIT-FOR-BIT CHECK against stage 1 (the method, the decoder and the mp4 are the same only if this passes): the kit scene 7c72937c's npz in the stage-1 tar was made from the mp4 by
`make_stage1.py` with frames {3 * gap_tick} | {0, 60, 120}. Here that mp4 is decoded again, the SAME frames are extracted into a scratch npz, and every array is compared with the stage-1 npz:
dtype, shape and bytes. (The .npz FILE differs in its zip timestamps, which is why the arrays are compared, and why MANIFEST2 also carries a content sha256 over the frame bytes.)
Writes: stage2/ref_frames/REF_FRAMES_SOURCES.json (scene -> npz path, for make_overlay2.py), EXTRACT_RESULT.json, BITCHECK_7c72937c.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import tarfile
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
T2 = os.path.dirname(HERE)
import numpy as np  # noqa: E402
import t2v8f_nocv2 as N  # noqa: E402

OUT = f"{T2}/stage2/ref_frames"
MP4 = "camera_front_wide_120fov.mp4"
KIT_SCENE = "7c72937c-c620-4776-9555-d57222c0081f"
KIT_MP4 = f"E:/thor_archive/2026-09-14/nurec_scenes/sample_set/26.04_release/{KIT_SCENE}/{MP4}"
STAGE1_TAR = f"{T2}/stage1/t2v8f_stage1.tar"
SCRATCH = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
REF_IDX = (0, 60, 120)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def one(sid, d, listing_sha):
    mp4, npz = f"{d}/{MP4}", f"{d}/ref_frames.npz"
    assert os.path.isfile(mp4), f"{mp4} is missing"
    got = sha256_file(mp4)
    assert got == listing_sha, f"{sid[:8]}: the mp4 sha256 {got} != the listing's lfs_sha256 {listing_sha}"
    assert not os.path.exists(npz), f"{npz} exists: move it aside first (nothing is overwritten)"
    t0 = time.time()
    shapes = N.extract_frames(mp4, REF_IDX, npz + ".tmp.npz")
    os.replace(npz + ".tmp.npz", npz)
    return {"scene": sid, "path": npz, "bytes": os.path.getsize(npz), "sha256": sha256_file(npz), "mp4_sha256": got, "shapes": {k: list(v) for k, v in shapes.items()},
            "seconds": round(time.time() - t0, 1)}


def bitcheck():
    """Re-extract stage 1's frames of the kit scene from its mp4 and compare every array with the npz stage 1 shipped."""
    t = tarfile.open(STAGE1_TAR)
    man = json.load(t.extractfile("pkg/MANIFEST.json"))
    want = next(e for e in man["files"] if e["path"] == f"refs/scenes/{KIT_SCENE}/ref_frames.npz")
    s1 = f"{SCRATCH}/s1x/pkg/refs/scenes/{KIT_SCENE}/ref_frames.npz"
    if not os.path.isfile(s1):
        t.extract(f"pkg/refs/scenes/{KIT_SCENE}/ref_frames.npz", f"{SCRATCH}/s1x", filter="data")
    assert sha256_file(s1) == want["sha256"], "the extracted stage-1 npz is not the one in stage 1's MANIFEST.json"
    gap = man["gap_ticks"][KIT_SCENE]
    idxs = sorted({man["stride"] * g for g in gap} | set(REF_IDX))
    pinned = json.load(open(f"{T2}/../t2_scenes/raw/RELEASE_LISTING.json", encoding="utf-8"))["scenes"][KIT_SCENE][MP4]["lfs_sha256"]
    mp4_sha = sha256_file(KIT_MP4)
    assert mp4_sha == pinned, f"the kit mp4 {mp4_sha} is not the pinned lfs_sha256 {pinned}"
    re_npz = f"{SCRATCH}/bitcheck_{KIT_SCENE[:8]}.npz"
    if os.path.exists(re_npz):
        os.replace(re_npz, re_npz + f".aside{int(time.time())}")
    N.extract_frames(KIT_MP4, idxs, re_npz)
    rows, ok = [], True
    with np.load(s1) as a, np.load(re_npz) as b:
        assert sorted(a.files) == sorted(b.files), f"array names differ: {sorted(a.files)} vs {sorted(b.files)}"
        for k in sorted(a.files, key=lambda x: int(x[1:])):
            x, y = a[k], b[k]
            same = bool(x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes())
            ok &= same
            rows.append({"array": k, "dtype": str(x.dtype), "shape": list(x.shape), "bit_equal": same, "sha256_stage1": hashlib.sha256(x.tobytes()).hexdigest()[:16], "sha256_reextracted": hashlib.sha256(y.tobytes()).hexdigest()[:16]})
    out = {"scene": KIT_SCENE, "mp4": KIT_MP4, "mp4_sha256": mp4_sha, "mp4_equals_pinned_lfs_sha256": True, "stage1_npz_sha256": want["sha256"], "frames_compared": idxs, "n_arrays": len(rows),
           "all_bit_equal": ok, "stage2_frames_0_60_120_bit_equal": all(r["bit_equal"] for r in rows if int(r["array"][1:]) in REF_IDX), "rows": rows,
           "note": "the arrays are compared, not the .npz files: a zip carries timestamps"}
    with open(f"{OUT}/BITCHECK_7c72937c.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    chk = bitcheck()
    print(json.dumps({"bitcheck_all_bit_equal": chk["all_bit_equal"], "n_arrays": chk["n_arrays"], "frames_0_60_120": chk["stage2_frames_0_60_120_bit_equal"]}), flush=True)
    assert chk["all_bit_equal"], "stage 1's frames are not reproduced bit for bit: STOP, the extraction method is not the one stage 1 used"
    if a.check_only:
        return 0
    dl = json.load(open(f"{OUT}/DOWNLOAD_RESULT.json", encoding="utf-8"))
    assert dl["n_failed"] == 0 and dl["n_ok"] == 41, f"the download is not complete: {dl['n_ok']}/{dl['n_scenes']}"
    lst = json.load(open(f"{T2}/../t2_scenes/raw/RELEASE_LISTING.json", encoding="utf-8"))["scenes"]
    rows, t0 = {}, time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, sid, os.path.dirname(o["path"]), lst[sid][MP4]["lfs_sha256"]): sid for sid, o in dl["scenes"].items()}
        for f in cf.as_completed(futs):
            r = f.result()
            rows[r["scene"]] = r
            print(f"{r['scene'][:8]} {r['bytes']} B {r['shapes']['f0']} {r['seconds']} s", flush=True)
    assert len(rows) == 41
    src = {sid: {"path": r["path"], "bytes": r["bytes"], "sha256": r["sha256"]} for sid, r in rows.items()}
    src[KIT_SCENE] = {"path": f"{SCRATCH}/s1x/pkg/refs/scenes/{KIT_SCENE}/ref_frames.npz", "bytes": os.path.getsize(f"{SCRATCH}/s1x/pkg/refs/scenes/{KIT_SCENE}/ref_frames.npz"),
                      "sha256": chk["stage1_npz_sha256"], "note": "stage 1's own npz (already in box 2's kit); hashed by the preflight, not shipped"}
    with open(f"{OUT}/REF_FRAMES_SOURCES.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(src, f, indent=1)
    with open(f"{OUT}/EXTRACT_RESULT.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"n_scenes": len(rows), "frames": list(REF_IDX), "total_npz_bytes": sum(r["bytes"] for r in rows.values()), "seconds": round(time.time() - t0, 1), "scenes": rows}, f, indent=1)
    print(json.dumps({"extracted": len(rows), "total_npz_bytes": sum(r["bytes"] for r in rows.values())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
