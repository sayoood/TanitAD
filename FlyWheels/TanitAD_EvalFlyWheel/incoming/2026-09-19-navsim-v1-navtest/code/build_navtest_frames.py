#!/usr/bin/env python3
"""navtest frame bank, per VERIFIED camera shard (TANITAD VENV, CPU). CAM_F0/L0/R0 only.

⛔ THE STITCH IS NOT RE-DERIVED. Every pixel comes from E2's navhard-ready builder,
``…/2026-09-19-navsim-refcv4b-bridge/code/build_frames.py`` — IMPORTED by path, never copied —
whose geometry is itself proven bit-exact against the DataFlyWheel bank (E2 control KB). Each
frame is stitched by calling E2's ``build_scene`` on that ONE frame's camera dict: its per-frame
work (``rig_key`` → ``build_map`` → ``sample``) is independent across frames, so a scene's array is
the stack of its frames' arrays — and KB1 (``--verify-e2``) proves it by running E2's builder
UNMODIFIED, as its own process, on the same tokens and requiring the per-scene
``sha256(arr.tobytes())[:16]`` to be bit-identical.

WHY UNIQUE FRAMES. navtest windows are 4 consecutive 2 Hz frames and tokens are dense inside a
log, so one frame serves up to 4 tokens. D: is exFAT with 1 MiB clusters, so the bank is ONE
``.npy`` per shard (``frames_sNN.npy`` u8 [n_frames, 256, 640, 3]) plus ONE index — never a file
per scene. The observed-pixel map depends only on the rig and is stored once per rig.

Per shard (only if its receipt row reads ``COMPLETE_VERIFIED``):
  1. stream the ``.tgz`` once; extract ONLY the F0/L0/R0 jpgs some navtest token needs, into a
     D: scratch dir (the archive is kept);
  2. build every token whose 4 × 3 jpgs are all present (KB3: otherwise it is carried to the next
     shard, and refused at the end if never completed);
  3. stitch each needed frame once, assert ``mean_px ≥ 1.0`` (KB2), write the shard bank;
  4. delete the extracted jpgs (unless ``--keep-scratch``); write ``shard_NN.DONE.json``.

    python build_navtest_frames.py --inputs <export.json.gz> --bank D:/…/frame_bank --shards 0
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tarfile
import time

import numpy as np

E2_BUILDER = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
              "2026-09-19-navsim-refcv4b-bridge/code/build_frames.py")
SHARD_DIR = "D:/Archive/devbox-C/navsim/data/openscene-v1.1/openscene_sensor_test_camera"
RECEIPT = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
           "2026-09-19-navhard-download/raw/receipt_navtest_camera.json")
LOG_DIR = "D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
PREFIX = "openscene-v1.1/sensor_blobs/test/"
CAMS = ("cam_l0", "cam_f0", "cam_r0")
H, W = 256, 640


def load_e2():
    spec = importlib.util.spec_from_file_location("e2_build_frames", E2_BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def token_paths(rec: dict) -> list:
    """The 12 jpg paths (4 frames x 3 cameras) one navtest token needs, forward-slashed."""
    return [p.replace("\\", "/") for c in CAMS for p in rec["cams"][c]]


def complete_partial(toks: dict, need: dict, extracted: set, skip=()) -> tuple:
    """(complete, partial) tokens given the jpgs on hand (KB3).

    COMPLETE = all 12 of its jpgs are present; PARTIAL = some but not all — that happens when a
    log SPANS two shards, and those jpgs are carried over rather than deleted. ``skip`` is the
    set of tokens ALREADY in the bank: frames are shared between neighbouring tokens, so a
    carried jpg would otherwise resurrect a built token as "partial" and carry its jpgs forever.
    Pinned by tests/test_frame_bank_selection.py, including the cross-shard case."""
    skip = set(skip)
    cand = {t for p in extracted for t in need.get(p, ()) if t not in skip}
    complete, partial = [], []
    for t in sorted(cand):
        ps = token_paths(toks[t])
        (complete if all(p in extracted for p in ps) else partial).append(t)
    return complete, partial


def shard_name(i: int) -> str:
    return f"openscene_sensor_test_camera_{i}.tgz"


def verified_shards() -> dict:
    """{index: receipt row} for rows reading COMPLETE_VERIFIED (sha256 == HF ETag)."""
    rec = json.load(open(RECEIPT, encoding="utf-8"))
    out = {}
    for name, row in rec.get("files", {}).items():
        if row.get("status") in ("COMPLETE_VERIFIED", "ALREADY_COMPLETE_VERIFIED") \
                and row.get("sha256") == row.get("x_linked_etag"):
            out[int(name.rsplit("_", 1)[1].split(".")[0])] = row
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="export_navtest_inputs.py output (.json.gz)")
    ap.add_argument("--bank", required=True, help="bank root on D:")
    ap.add_argument("--shards", default="all", help="comma list or 'all' (verified ones only)")
    ap.add_argument("--tokens", default="", help="optional JSON list/doc restricting the tokens")
    ap.add_argument("--max-logs", type=int, default=0, help="smoke: stop after N navtest logs")
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:                                                 # noqa: BLE001
        pass
    e2 = load_e2()
    bank = pathlib.Path(a.bank)
    (bank / "rigs").mkdir(parents=True, exist_ok=True)
    doc = json.load(gzip.open(a.inputs, "rt", encoding="utf-8"))
    toks = doc["tokens"]
    if a.tokens:
        want = json.load(open(a.tokens, encoding="utf-8"))
        want = set(want["tokens"] if isinstance(want, dict) else want)
        toks = {t: r for t, r in toks.items() if t in want}
    # needed jpg (path relative to sensor_blobs/test) -> set of tokens
    need: dict = {}
    for t, r in toks.items():
        for p in token_paths(r):
            need.setdefault(p, set()).add(t)
    ver = verified_shards()
    shards = sorted(ver) if a.shards == "all" else [int(x) for x in a.shards.split(",")]
    unverified = [s for s in shards if s not in ver]
    if unverified:
        print(f"⛔ shards {unverified} are not COMPLETE_VERIFIED in the receipt — refused", flush=True)
        shards = [s for s in shards if s in ver]
    index_path = bank / "index.json"
    index = json.load(open(index_path, encoding="utf-8")) if index_path.exists() else {
        "frame": None, "tokens": {}, "shards": {}, "rigs": {}}
    rig_cache: dict = {}
    log_cache: dict = {}

    def log_frames(ln):
        if ln not in log_cache:
            log_cache.clear()                                   # one log pickle in RAM at a time
            fr = e2.load(os.path.join(LOG_DIR, f"{ln}.pkl"))
            log_cache[ln] = ({f["token"]: i for i, f in enumerate(fr)}, fr)
        return log_cache[ln]

    rc = 0
    for si in shards:
        done_p = bank / f"shard_{si:02d}.DONE.json"
        if done_p.exists():
            print(f"[bank] shard {si} already DONE — skipped", flush=True)
            continue
        t0 = time.time()
        scratch = pathlib.Path(f"{bank}/_scratch_s{si:02d}")
        scratch.mkdir(parents=True, exist_ok=True)
        tgz = os.path.join(SHARD_DIR, shard_name(si))
        logs_seen, n_members, n_extracted, extracted = [], 0, 0, set()
        cur_log, n_navtest_logs = None, 0
        with tarfile.open(tgz, "r|gz") as tf:
            for m in tf:
                n_members += 1
                if not m.name.startswith(PREFIX):
                    continue
                rel = m.name[len(PREFIX):]
                ln = rel.split("/", 1)[0]
                if ln != cur_log:
                    if a.max_logs and cur_log is not None and any(
                            p.startswith(cur_log + "/") for p in extracted):
                        n_navtest_logs += 1
                        if n_navtest_logs >= a.max_logs:
                            break
                    cur_log = ln
                    logs_seen.append(ln)
                if not m.isfile() or rel not in need:
                    continue
                f = tf.extractfile(m)
                dst = scratch / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(dst, "wb") as fo:
                    shutil.copyfileobj(f, fo, 1 << 20)
                extracted.add(rel)
                n_extracted += 1
        t_extract = time.time() - t0
        # CARRY-OVER (a log may span two shards): jpgs of tokens left incomplete by an earlier
        # shard live in <bank>/_carry; move them into this shard's scratch (same volume: rename).
        carry = bank / "_carry"
        n_carried_in = 0
        if carry.exists():
            for fp in list(carry.rglob("*.jpg")):
                rel = fp.relative_to(carry).as_posix()
                dst = scratch / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    os.replace(fp, dst)
                    extracted.add(rel)
                    n_carried_in += 1
        # which tokens are complete NOW (KB3)
        complete, partial = complete_partial(toks, need, extracted, skip=index["tokens"])
        # unique frames of complete tokens
        frame_list, frame_row = [], {}
        for t in complete:
            for ft in toks[t]["frame_tokens"]:
                if ft not in frame_row:
                    frame_row[ft] = len(frame_list)
                    frame_list.append((toks[t]["log_name"], ft))
        out_npy = bank / f"frames_s{si:02d}.npy"
        arr = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.uint8,
                                        shape=(max(len(frame_list), 1), H, W, 3))
        frame_meta, fails = {}, []
        t1 = time.time()
        for k, (ln, ft) in enumerate(frame_list):
            try:
                idx, fr = log_frames(ln)
                cd = fr[idx[ft]]["cams"]
                one, src, rk, obs = e2.build_scene([cd], scratch, rig_cache)
                if one.shape != (1, H, W, 3) or one.dtype != np.uint8:
                    raise ValueError(f"stitch returned {one.dtype}{one.shape}")
                mpx = float(one.mean())
                if mpx < 1.0:
                    raise ValueError(f"mean_px {mpx:.3f} < 1.0 (an all-black frame)")
                arr[k] = one[0]
                frame_meta[ft] = {"row": k, "rig_key": rk, "observed_frac": round(obs, 6),
                                  "mean_px": round(mpx, 2)}
                if rk not in index["rigs"]:
                    np.save(bank / "rigs" / f"{rk}.src.npy", src)
                    index["rigs"][rk] = {"observed_frac": round(float((src >= 0).mean()), 6)}
            except Exception as e:                                    # noqa: BLE001
                fails.append({"frame_token": ft, "log": ln, "reason": f"{type(e).__name__}: {str(e)[:160]}"})
            if (k + 1) % 500 == 0:
                print(f"[bank] shard {si}: {k + 1}/{len(frame_list)} frames, "
                      f"{(time.time() - t1) / (k + 1):.3f} s/frame", flush=True)
        arr.flush()
        del arr
        built, refused = 0, []
        for t in complete:
            fts = toks[t]["frame_tokens"]
            if not all(ft in frame_meta for ft in fts):
                refused.append({"token": t, "reason": "a frame failed to stitch"})
                continue
            rows = [frame_meta[ft]["row"] for ft in fts]
            mm = np.load(out_npy, mmap_mode="r")
            stack = np.ascontiguousarray(mm[rows])
            index["tokens"][t] = {
                "shard": si, "file": out_npy.name, "rows": rows,
                "log_name": toks[t]["log_name"], "rig_key": frame_meta[fts[-1]]["rig_key"],
                "observed_frac": min(frame_meta[ft]["observed_frac"] for ft in fts),
                "mean_px": round(float(stack.mean()), 2),
                "sha256": hashlib.sha256(stack.tobytes()).hexdigest()[:16]}
            built += 1
        index["frame"] = {"h": e2.FRAME.height, "w": e2.FRAME.width, "f_ref": e2.FRAME.f_ref,
                          "projection": e2.FRAME.projection}
        rep = {"shard": si, "tgz": tgz, "receipt_sha256": ver[si]["sha256"], "n_members": n_members,
               "logs_in_shard": logs_seen, "n_jpgs_extracted": n_extracted,
               # ⚠️ `cand` was a local of the inline completeness loop that moved into
               # complete_partial(); referencing it here survived the refactor and killed the
               # shard AFTER its extraction and stitching (MEASURED 2026-09-20, shard 0, rc 1).
               "n_tokens_touched": len(complete) + len(partial),
               "n_tokens_already_in_bank": len(index["tokens"]),
               "n_tokens_complete": len(complete),
               "n_tokens_partial": len(partial), "partial_tokens": partial[:50],
               "n_frames": len(frame_list), "n_frame_failures": len(fails), "frame_failures": fails[:40],
               "n_tokens_built": built, "n_tokens_refused": len(refused), "refused": refused[:40],
               "bank_file": str(out_npy), "bank_bytes": os.path.getsize(out_npy),
               "extract_s": round(t_extract, 1), "stitch_s": round(time.time() - t1, 1),
               "wall_s": round(time.time() - t0, 1), "max_logs": a.max_logs}
        index["shards"][str(si)] = {k: rep[k] for k in ("n_tokens_built", "n_frames", "bank_file",
                                                        "logs_in_shard", "n_tokens_partial")}
        json.dump(index, open(index_path, "w", encoding="utf-8"))
        # jpgs of still-PARTIAL tokens go to the carry dir; everything else is deleted
        keep = {p for t in partial for p in token_paths(toks[t])}
        n_carried_out = 0
        for rel in sorted(keep & extracted):
            src_p = scratch / rel
            if src_p.exists():
                dst = carry / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(src_p, dst)
                n_carried_out += 1
        rep["n_jpgs_carried_in"] = n_carried_in
        rep["n_jpgs_carried_out"] = n_carried_out
        if not a.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
            rep["scratch_deleted"] = not scratch.exists()
        json.dump(rep, open(done_p if not a.max_logs else bank / f"shard_{si:02d}.SMOKE.json",
                            "w", encoding="utf-8"), indent=1)
        print(json.dumps({k: rep[k] for k in ("shard", "n_jpgs_extracted", "n_tokens_complete",
                                              "n_tokens_partial", "n_frames", "n_tokens_built",
                                              "n_frame_failures", "extract_s", "stitch_s")}), flush=True)
        if fails or refused:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
