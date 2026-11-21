#!/usr/bin/env python3
"""P6b - LEVER L4 (data): LiDAR BEV GT + locally rebuilt trunk frames for extra TRAINING clips.

Diagnosis that motivates it (MEASURED in this package): the frozen-trunk head OVERFITS its 82
training clips -- val AP peaks at step 1,000 (0.4548) and falls to ~0.40 by step 3,500 while
train loss keeps falling. More clips is the lever aimed at that cause.

⛔ WHAT THIS DOES NOT TOUCH: the 139-clip B1 EVAL corpus, its split, its test clips, or any
file the pre-registered panel reads. Extra clips can only ever be TRAINING data; the test set
stays the 34 held-out eval clips.

Selection -- deterministic and content-blind, fixed before any extra clip was fetched:
  * universe = the v7.2 TRAIN label release (4,572 clips; md5 0ff902130ce76886b8a925eceed9e3a5,
    the file refcv5-v2 trained on -- so these images are IN-SAMPLE for the TRUNK, never for
    the BEV target, which the trunk never saw);
  * available on the dev box: mp4 + timestamps, egomotion, LiDAR + camera extrinsics, a
    per-clip intrinsics row, a clip_index chunk;
  * not an eval-join clip;
  * sorted by sha256(clip_id)[:12]; the first N.

Frames: rebuilt from mp4 exactly as `v2_compressed._resampled` (P6a measured the rebuild at
<= 3 levels max / 0.75 mean abs from the pod-built PNGs; NOT bit-identical -- the token-level
effect is measured before any head trains on it). Saved as FRAMES-ONLY payloads with NO
poses/actions keys, so a planner trainer pointed at them fails loudly instead of training.

Usage:  python p6b_extra_corpus.py --n 200 --fetch-conc 4 --build-workers 4
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

WORK = Path(r"C:\Users\Admin\tanitad-caches\bevhead-20260913")
FRAMES_DIR = Path(os.environ.get("BEVHEAD_FRAMES_DIR", str(WORK / "bevframes_extra")))
GT_EXTRA = WORK / "bev_gt_extra"
TRAIN_LABELS_GLOB = str(HERE.parents[3] / "**" / "s2_labels_v7.2_train.jsonl.gz")


def select_clips(n: int) -> tuple[list[str], dict]:
    import pyarrow.parquet as pq
    import p2_build_corpus as B
    from lidar_fetch import CLIP_INDEX, is_protected, sha12
    files = glob.glob(TRAIN_LABELS_GLOB, recursive=True)
    if len(files) != 1:
        raise SystemExit(f"expected exactly one v7.2 train label file, found {len(files)}")
    import hashlib
    md5 = hashlib.md5(open(files[0], "rb").read()).hexdigest()
    if md5 != "0ff902130ce76886b8a925eceed9e3a5":
        raise SystemExit(f"train label md5 {md5} is not the file refcv5-v2 trained on")
    train = []
    with gzip.open(files[0], "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            train.append(r.get("clip_id"))
    eval_join = set(B.join_clips())
    chunk_of = dict(zip(*pq.read_table(CLIP_INDEX, columns=["clip_id", "chunk"]).to_pydict().values()))
    csv = Path(B.CAM_DIR).parents[1] / "calibration" / "physicalai_front_wide_intrinsics.csv"
    intr_ids = set()
    import csv as _csv
    with open(csv, newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            intr_ids.add(row["clip_id"])
    stats = {"train_universe": len(train), "reject": {}}
    ok = []

    def rej(k):
        stats["reject"][k] = stats["reject"].get(k, 0) + 1

    for c in train:
        if c in eval_join:
            rej("in_eval_join"); continue
        if is_protected(c):
            rej("protected_qwen_clip"); continue
        if not (Path(B.CAM_DIR) / f"{c}.mp4").exists() or not (Path(B.CAM_DIR) / f"{c}.timestamps.parquet").exists():
            rej("no_mp4_or_timestamps"); continue
        if not (Path(B.EGO_DIR) / f"{c}.parquet").exists():
            rej("no_egomotion"); continue
        if c not in intr_ids:
            rej("no_per_clip_intrinsics"); continue
        if c not in chunk_of:
            rej("no_clip_index_chunk"); continue
        try:
            B.extrinsics(c, "lidar_top_360fov")
            B.extrinsics(c, "camera_front_wide_120fov")
        except KeyError:
            rej("no_extrinsics"); continue
        ok.append(c)
    ok.sort(key=sha12)
    stats["available"] = len(ok)
    stats["selected"] = min(n, len(ok))
    stats["train_label_md5"] = md5
    return ok[:n], stats


def frames_payload(clip_id: str) -> dict:
    """Rebuild the trunk's frames from mp4 and write a FRAMES-ONLY payload. Worker-safe."""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import torch
    import torchvision.io as tvio
    torch.set_num_threads(2)
    import p2_build_corpus as B
    from lidar_fetch import redact_text, sha12
    from p6a_frames_equivalence import rebuild_frames
    out = FRAMES_DIR / f"{clip_id}.v2ep.pt"      # local cache only; never staged
    rec = {"clip_sha12": sha12(clip_id)}
    t0 = time.time()
    try:
        if not out.exists():
            g = B.episode_grid(clip_id)
            vid = rebuild_frames(clip_id, g["cam_frame_idx"])           # [T,3,256,640] u8
            if vid.shape[1:] != (3, 256, 640) or int(vid.amax()) == 0:
                raise RuntimeError(f"rebuilt frames degenerate {tuple(vid.shape)}")
            pngs = [tvio.encode_png(vid[i].contiguous()) for i in range(vid.shape[0])]
            FRAMES_DIR.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".tmp")
            torch.save({"jpeg_buf": torch.cat(pngs),
                        "jpeg_len": torch.tensor([int(p.numel()) for p in pngs], dtype=torch.int64),
                        "n_stack": 3, "image_h": 256, "image_w": 640, "codec": "png",
                        "frame": {"height": 256, "width": 640, "f_ref": 305.5774907364391,
                                  "projection": "cylindrical"},
                        "schema": "tanitad.bevframes/1",
                        "WARNING": "FRAMES ONLY, rebuilt locally from mp4 (P6a). NO poses/actions: "
                                   "not a v2ep training payload."}, tmp)
            os.replace(tmp, out)
        rec["frames_s"] = round(time.time() - t0, 1)
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = redact_text(f"{type(e).__name__}: {e}")[:400]
    return rec


def build_extra(clip_id: str) -> dict:
    """Frames payload, then the P2 builder with its frame source pointed at the payload dir."""
    fr = frames_payload(clip_id)
    if not fr.get("ok"):
        return {**fr, "stage": "frames"}
    import p2_build_corpus as B
    B.V2EP_DIR = str(FRAMES_DIR)                 # this worker process only
    r = B.build_clip(clip_id, out_dir=str(GT_EXTRA))
    r["frames_s"] = fr.get("frames_s")
    r["frames_source"] = "local mp4 rebuild (P6a), not the pod-built v2ep PNGs"
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--fetch-conc", type=int, default=4)
    ap.add_argument("--build-workers", type=int, default=4)
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--rebuild-eval", action="store_true",
                    help="AMENDMENT A1: frames-only payloads for ALL 139 eval-join clips "
                         "(run with BEVHEAD_FRAMES_DIR pointing at a separate dir)")
    args = ap.parse_args()
    if args.rebuild_eval:
        import p2_build_corpus as B
        from lidar_fetch import sha12
        if "BEVHEAD_FRAMES_DIR" not in os.environ:
            raise SystemExit("refusing: --rebuild-eval must write to an explicit BEVHEAD_FRAMES_DIR")
        clips = B.join_clips()
        t0 = time.time()
        recs = []
        with ProcessPoolExecutor(max_workers=args.build_workers) as ex:
            for r in ex.map(frames_payload, clips):
                recs.append(r)
        n_ok = sum(bool(r.get("ok")) for r in recs)
        (HERE.parent / "raw" / "p6b_rebuild_eval_frames.json").write_text(json.dumps(
            {"n": len(recs), "ok": n_ok, "wall_s": round(time.time() - t0, 1),
             "failed": [r for r in recs if not r.get("ok")]}, indent=1), encoding="utf-8")
        print(f"[p6b] rebuilt eval frames ok={n_ok}/{len(recs)} wall={time.time() - t0:.0f} s", flush=True)
        return 0 if n_ok == len(recs) else 2
    from lidar_fetch import cached_path, chunk_map, fetch_clip, is_protected, redact_text, sha12

    clips, stats = select_clips(args.n)
    GT_EXTRA.mkdir(parents=True, exist_ok=True)
    (HERE.parent / "raw" / "p6b_extra_selection.json").write_text(json.dumps(
        {**stats, "selected_sha12": [sha12(c) for c in clips],
         "rule": "v7.2 TRAIN universe, available locally, not eval-join, not protected, sorted by "
                 "sha12, first N"}, indent=1), encoding="utf-8")
    print(f"[p6b] {json.dumps(stats)}", flush=True)
    if args.select_only:
        return 0
    man_path = GT_EXTRA / "manifest.jsonl"
    done = set()
    if man_path.exists():
        for line in man_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("ok"):
                done.add(r["clip_sha12"])
    todo = [c for c in clips if sha12(c) not in done]
    chunks = chunk_map()
    t0 = time.time()
    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=args.fetch_conc) as fx, \
            ProcessPoolExecutor(max_workers=args.build_workers) as bx:
        fetch_f = {fx.submit(fetch_clip, c, chunks[c]): c for c in todo if not cached_path(c).exists()}
        build_f = {bx.submit(build_extra, c): c for c in todo if cached_path(c).exists()}
        while fetch_f or build_f:
            dn, _ = wait(list(fetch_f) + list(build_f), return_when=FIRST_COMPLETED)
            for f in dn:
                if f in fetch_f:
                    c = fetch_f.pop(f)
                    r = f.result()
                    if r["ok"]:
                        build_f[bx.submit(build_extra, c)] = c
                    else:
                        n_fail += 1
                        with open(man_path, "a", encoding="utf-8") as mf:
                            mf.write(json.dumps({"clip_sha12": sha12(c), "ok": False, "stage": "fetch",
                                                 "error": r.get("error")}) + "\n")
                else:
                    c = build_f.pop(f)
                    r = f.result()
                    deleted = False
                    if r.get("ok") and not is_protected(c):
                        try:
                            cached_path(c).unlink()
                            deleted = True
                        except OSError as e:
                            r["delete_error"] = redact_text(str(e))
                    r["parquet_deleted"] = deleted
                    with open(man_path, "a", encoding="utf-8") as mf:
                        mf.write(json.dumps(r) + "\n")
                    n_ok += int(bool(r.get("ok")))
                    n_fail += int(not r.get("ok"))
                    print(f"[p6b] {r.get('clip_sha12')} ok={r.get('ok')} marg48={r.get('polar48_marginal_observed_infield')} "
                          f"fails={r.get('failures')} err={r.get('error', '')} [{n_ok} ok / {n_fail} fail, "
                          f"{time.time() - t0:.0f} s]", flush=True)
    print(f"[p6b] DONE ok={n_ok} fail={n_fail} wall={time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
