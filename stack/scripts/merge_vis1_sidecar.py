#!/usr/bin/env python3
"""merge_vis1_sidecar.py -- add NEW clips to an existing VIS-1 sidecar WITHOUT re-z-buffering the old ones.

refcv8 +203 train clips (2026-10-12): the old sidecar (``vis1_sidecar_refcv6b1_train4369_eval139.npz``, 4,508 clips,
859,497 frames) stays byte-for-byte what it was for every OLD clip; the 203 ADD clips' per-clip entries come from a
``vis1_subbuild_driver.py`` run of the UNMODIFIED ``precompute_vis1_sidecar.py --mode full``. This script only
RE-ASSEMBLES: per-clip entries -> the builder's own assembly block (copied verbatim below) -> ``vis1.write_sidecar``.

THE ASSEMBLY IS PROVEN BEFORE IT IS USED. Step 1 re-assembles the OLD sidecar's own per-clip entries with the OLD
meta and requires (i) every array equal (dtype and values), (ii) ``content_sha256_of_arrays`` equal to the old
meta's, and reports (iii) whether the written FILE is byte-identical (sha256). A merge whose no-op round trip does
not reproduce the old content hash is refused.

Clips are emitted in sha12 order (the builder's order), so the merged file has the layout a full rebuild would
produce; per-clip entries are copied, never recomputed.

sha12 only in every output; no raw clip id is written or printed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
STACK = HERE.parents[1]
if str(STACK) not in sys.path:
    sys.path.insert(0, str(STACK))

from tanitad.data import vis1 as V  # noqa: E402

ROW_KEYS = ("row", "track", "n_full", "n_img", "n_vis", "vis_rows", "cx", "cy")
ALL_KEYS = tuple(list(V._ROW_KEYS) + list(V._FRAME_KEYS) + list(V._CLIP_KEYS))


def log(msg: str) -> None:
    print(f"[vis1-merge {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5_file(p, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def sha12(s) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta_json"].tobytes()).decode("utf-8"))
        arrays = {k: z[k] for k in ALL_KEYS}
    return arrays, meta


def per_clip_entries(a: dict) -> dict:
    """sha12 -> the compact per-clip dict ``run_shard`` returns (the inverse of :func:`assemble`)."""
    out = {}
    cfp, fp = a["clip_frame_ptr"], a["frame_ptr"]
    for ci, s in enumerate(a["clip_sha12"].tolist()):
        f0, f1 = int(cfp[ci]), int(cfp[ci + 1])
        r0, r1 = int(fp[f0]), int(fp[f1])
        if f1 > f0 and not bool((a["frame_clip"][f0:f1] == ci).all()):
            raise SystemExit(f"[vis1-merge] clip {s}: frame_clip does not point back at the clip")
        d = {"sha12": s, "split": str(a["clip_split"][ci]), "mask_ftheta": bool(a["clip_mask_ftheta"][ci]),
             "frame_f": a["frame_f"][f0:f1], "frame_n_zh": a["frame_n_zh"][f0:f1],
             "frame_n_valid": a["frame_n_valid"][f0:f1],
             "frame_cnt": np.diff(fp[f0:f1 + 1]).astype(np.int64)}
        for k in ROW_KEYS:
            d[k] = a[k][r0:r1]
        out[s] = d
    return out


def assemble(per_clip: dict) -> dict:
    """The builder's own assembly block (precompute_vis1_sidecar.py, 'assemble the sidecar'), verbatim in logic."""
    order = sorted(per_clip)
    A = {k: [] for k in ROW_KEYS}
    fr = {k: [] for k in ("frame_clip", "frame_f", "frame_n_zh", "frame_n_valid")}
    frame_ptr = [0]
    clip_frame_ptr = [0]
    n_rows = 0
    for ci, s12 in enumerate(order):
        c = per_clip[s12]
        for k in A:
            A[k].append(c[k])
        nf = len(c["frame_f"])
        fr["frame_clip"].append(np.full(nf, ci, np.int32))
        fr["frame_f"].append(c["frame_f"])
        fr["frame_n_zh"].append(c["frame_n_zh"])
        fr["frame_n_valid"].append(c["frame_n_valid"])
        for cnt in c["frame_cnt"].tolist():
            n_rows += int(cnt)
            frame_ptr.append(n_rows)
        clip_frame_ptr.append(clip_frame_ptr[-1] + nf)
    arrays = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in A.items()}
    arrays["row"] = arrays["row"].astype(np.int16)
    arrays["track"] = arrays["track"].astype(np.int64)
    for k in ("n_full", "n_img", "n_vis", "vis_rows"):
        arrays[k] = arrays[k].astype(np.int32)
    for k in ("cx", "cy"):
        arrays[k] = arrays[k].astype(np.float32)
    for k, v in fr.items():
        arrays[k] = np.concatenate(v).astype(np.int32) if v else np.zeros(0, np.int32)
    arrays["frame_ptr"] = np.asarray(frame_ptr, np.int64)
    arrays["clip_frame_ptr"] = np.asarray(clip_frame_ptr, np.int64)
    arrays["clip_sha12"] = np.asarray(order, dtype="<U12")
    arrays["clip_mask_ftheta"] = np.asarray([bool(per_clip[s]["mask_ftheta"]) for s in order], np.bool_)
    arrays["clip_split"] = np.asarray([per_clip[s]["split"] for s in order], dtype="<U5")
    if int(arrays["frame_ptr"][-1]) != len(arrays["row"]):
        raise SystemExit("[vis1-merge] internal: frame pointers do not close over the rows")
    return arrays


def content_sha256(arrays: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    return h.hexdigest()


def arrays_equal(a: dict, b: dict) -> list:
    bad = []
    for k in ALL_KEYS:
        x, y = a[k], b[k]
        if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
            bad.append(k)
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", required=True, help="the existing sidecar .npz (its entries are copied unchanged)")
    ap.add_argument("--new", required=True, help="the vis1_subbuild_driver.py output holding the ADD clips")
    ap.add_argument("--add-list", required=True, help="JSON list of the sha12s to take from --new")
    ap.add_argument("--train-cache", required=True, help="the NEW train view dir (recorded; its _v2manifest.pt is read)")
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--extrinsics", required=True)
    ap.add_argument("--calib-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", required=True)
    a = ap.parse_args(argv)
    t_start = time.time()
    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    rec = {"tool": "merge_vis1_sidecar.py", "argv": sys.argv[1:], "tool_md5": md5_file(HERE),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "numpy": np.__version__}
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"[vis1-merge] refusing to overwrite {out}")
    old_sha = V.file_sha256(a.old)
    rec["old"] = {"path": a.old, "sha256": old_sha, "md5": md5_file(a.old)}
    old_a, old_meta = load_npz(a.old)
    new_a, new_meta = load_npz(a.new)
    rec["new_subbuild"] = {"path": a.new, "sha256": V.file_sha256(a.new), "md5": md5_file(a.new),
                           "n_clips": int(len(new_a["clip_sha12"]))}

    # ---- 1. the no-op round trip: is THIS assembly the one that wrote the old file? ---------------------- #
    old_pc = per_clip_entries(old_a)
    rt = assemble(old_pc)
    bad = arrays_equal(old_a, rt)
    rt_content = content_sha256(rt)
    rt_path = work / "roundtrip_old.npz"
    rt_meta = copy.deepcopy(old_meta)
    rt_dig = V.write_sidecar(rt_path, rt, rt_meta)
    rec["roundtrip"] = {"arrays_unequal": bad, "content_sha256": rt_content,
                        "old_meta_content_sha256": old_meta.get("content_sha256_of_arrays"),
                        "file_sha256": rt_dig["sha256"], "old_file_sha256": old_sha,
                        "FILE_BYTE_IDENTICAL": rt_dig["sha256"] == old_sha}
    rec["roundtrip"]["PASS"] = bool(not bad and rt_content == old_meta.get("content_sha256_of_arrays"))
    log(f"round trip of the OLD sidecar: arrays unequal {bad}, content {rt_content[:16]} vs "
        f"{str(old_meta.get('content_sha256_of_arrays'))[:16]}, file byte-identical "
        f"{rec['roundtrip']['FILE_BYTE_IDENTICAL']}")
    if not rec["roundtrip"]["PASS"]:
        (work / "merge_record.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        raise SystemExit("[vis1-merge] REFUSED: the no-op round trip does not reproduce the old content hash")
    rt_path.unlink()

    # ---- 2. the merge ---------------------------------------------------------------------------------- #
    add = [str(s) for s in json.load(open(a.add_list, encoding="utf-8"))]
    if len(set(add)) != len(add):
        raise SystemExit("[vis1-merge] --add-list has duplicates")
    new_pc = per_clip_entries(new_a)
    miss = [s for s in add if s not in new_pc]
    if miss:
        raise SystemExit(f"[vis1-merge] {len(miss)} ADD clips are not in the sub-build (e.g. {miss[:4]})")
    dup = [s for s in add if s in old_pc]
    if dup:
        raise SystemExit(f"[vis1-merge] {len(dup)} ADD clips are already in the old sidecar (e.g. {dup[:4]})")
    notrain = [s for s in add if new_pc[s]["split"] != "train"]
    if notrain:
        raise SystemExit(f"[vis1-merge] {len(notrain)} ADD clips are not split=train in the sub-build")
    nomask = [s for s in add if not new_pc[s]["mask_ftheta"]]
    if nomask:
        raise SystemExit(f"[vis1-merge] {len(nomask)} ADD clips were z-buffered WITHOUT the f-theta mask "
                         f"(e.g. {nomask[:4]}) -- the intrinsics are missing for their chunk")
    merged = dict(old_pc)
    for s in add:
        merged[s] = new_pc[s]
    arrays = assemble(merged)
    content = content_sha256(arrays)

    # ---- 3. meta ---------------------------------------------------------------------------------------- #
    import torch
    from tanitad.data import vis1_zbuffer as VZ
    def man_clips(cache):
        man = torch.load(Path(cache) / "_v2manifest.pt", map_location="cpu", weights_only=False)
        return [str(c) for c in man["clip_id"]], man
    tr_c, tr_man = man_clips(a.train_cache)
    ev_c, ev_man = man_clips(a.eval_cache)
    tr12, ev12 = sorted(sha12(c) for c in tr_c), sorted(sha12(c) for c in ev_c)
    in_train = sorted(s for s, p in zip(arrays["clip_sha12"].tolist(), arrays["clip_split"].tolist()) if p == "train")
    in_eval = sorted(s for s, p in zip(arrays["clip_sha12"].tolist(), arrays["clip_split"].tolist()) if p == "eval")
    if in_train != tr12 or in_eval != ev12:
        raise SystemExit(f"[vis1-merge] merged clip set != manifests (train {len(in_train)}/{len(tr12)}, "
                         f"eval {len(in_eval)}/{len(ev12)})")
    table = json.load(open(a.extrinsics, encoding="utf-8"))
    chunks = sorted({int(table[c].get("chunk", -1)) for c in tr_c + ev_c})
    calib = {}
    for ch in chunks:
        hit = None
        for d in a.calib_dir:
            for pat in VZ.INTRINSICS_GLOBS:
                q = Path(pat.format(calib_dir=d, ch=ch))
                if q.exists():
                    hit = q
                    break
            if hit:
                break
        calib[ch] = None if hit is None else md5_file(hit)
    n_missing = sum(1 for v in calib.values() if v is None)
    if n_missing:
        raise SystemExit(f"[vis1-merge] {n_missing} chunk(s) have no intrinsics file")
    meta = copy.deepcopy(old_meta)
    meta["splits"] = {"train": {"cache": a.train_cache, "n_clips": len(tr_c),
                                "clip_sha12_sha256": hashlib.sha256("\n".join(tr12).encode()).hexdigest()},
                      "eval": {"cache": a.eval_cache, "n_clips": len(ev_c),
                               "clip_sha12_sha256": hashlib.sha256("\n".join(ev12).encode()).hexdigest()}}
    if meta["splits"]["eval"]["clip_sha12_sha256"] != old_meta["splits"]["eval"]["clip_sha12_sha256"]:
        raise SystemExit("[vis1-merge] the eval clip set changed vs the old sidecar")
    for k in ("agent_join", "join3d"):
        if new_meta["sources"][k]["md5"] != old_meta["sources"][k]["md5"]:
            raise SystemExit(f"[vis1-merge] {k} md5 differs between the old build and the sub-build "
                             f"({old_meta['sources'][k]['md5']} vs {new_meta['sources'][k]['md5']}) -- the old rows "
                             f"were keyed to a different join")
    meta["sources"] = copy.deepcopy(old_meta["sources"])
    meta["sources"]["extrinsics"] = {"path": a.extrinsics, "md5": md5_file(a.extrinsics)}
    meta["sources"]["intrinsics"] = {
        "calib_dirs": list(a.calib_dir), "n_chunks": len(chunks), "n_missing": 0,
        "md5_of_sorted_chunk_md5s": hashlib.md5(json.dumps(sorted(calib.items()), default=str).encode()).hexdigest()}
    meta["zbuffer"] = copy.deepcopy(new_meta["zbuffer"])
    if meta["zbuffer"]["function_sha256"] != old_meta["zbuffer"]["function_sha256"]:
        raise SystemExit("[vis1-merge] the z-buffer function digests differ between the old build and the sub-build")
    meta["n_clips"] = int(len(arrays["clip_sha12"]))
    meta["n_frames"] = int(len(arrays["frame_f"]))
    meta["n_rows"] = int(len(arrays["row"]))
    meta["n_clips_bounds_only"] = int((~arrays["clip_mask_ftheta"]).sum())
    meta["max_clips_smoke"] = 0
    meta["content_sha256_of_arrays"] = content
    n_zero = int((np.diff(arrays["clip_frame_ptr"]) == 0).sum())
    meta["merge"] = {
        "what": "old sidecar entries copied unchanged + ADD clips from an unmodified-builder sub-build; "
                "zero-frame clips = clips with no agent-join record (agent_label False -> _vis1_item never looks up)",
        "old_sidecar_sha256": old_sha, "old_sidecar_content_sha256_of_arrays": old_meta["content_sha256_of_arrays"],
        "old_zbuffer_module_md5": old_meta["zbuffer"]["module_md5"],
        "old_intrinsics_md5_of_sorted_chunk_md5s": old_meta["sources"]["intrinsics"]["md5_of_sorted_chunk_md5s"],
        "subbuild_sha256": rec["new_subbuild"]["sha256"], "n_add": len(add),
        "n_add_with_frames": int(sum(1 for s in add if len(new_pc[s]["frame_f"]) > 0)),
        "n_zero_frame_clips": n_zero,
        "merge_tool_md5": rec["tool_md5"], "roundtrip_old_content_ok": True,
        "roundtrip_old_file_byte_identical": rec["roundtrip"]["FILE_BYTE_IDENTICAL"]}
    dig = V.write_sidecar(out, arrays, meta)
    sc = V.VIS1Sidecar(out)                                   # the reader the trainer uses
    rec["merged"] = {"path": str(out), **dig, "n_clips": sc.n_clips, "n_frames": sc.n_frames, "n_rows": sc.n_rows,
                     "n_zero_frame_clips": n_zero, "content_sha256_of_arrays": content, "readback": sc.stamp()}
    rec["wall_s"] = round(time.time() - t_start, 1)
    (work / "merge_record.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    log(f"MERGED {out.name}: {sc.n_clips} clips ({n_zero} zero-frame), {sc.n_frames} frames, {sc.n_rows} rows, "
        f"sha256 {dig['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
