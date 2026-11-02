#!/usr/bin/env python3
"""stage_crops.py -- dev box, CPU only.  python -I stage_crops.py frames | align | shards [--workers 3]
frames : the unique (clip, t) frame list (24,000 train windows + 608 eval-141 windows) -> raw/frames.npz
align  : the 32-frame known-answer pack (cache-geometry full frames as lossless PNG + the same frames' crops as PNG and JPEG) -> SCR/align_pack.zip
shards : decode + rectify at native scale + crop + JPEG q95 for every frame, ~100 clips per shard, ZIP_STORED -> SCR/shards/shard_NNNN.zip (+ .idx.json); resumable (a shard is done iff its .done marker exists)."""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW = f"{HERE}/raw"
SCR = "F:/TanitAD-scratch/trigger_probe5_scratch"
sys.path.insert(0, HERE)
import crops_lib as C  # noqa: E402

P1 = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe/raw"
WIN_SHA = "a9a57b32ffac558929b8f57e49a979826b88d17be17e5e9d7f9347e5875783ec"
CLIPS_PER_SHARD = 100
TRAIN_CACHE = "F:/TanitAD-scratch/speedread/cache/train/clips"
EVAL_CACHE = "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/clips"
MANIFEST_TRAIN = "C:/Users/Admin/box2_ship/logs/v8f_dscale/manifest_train.jsonl"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build_frames():
    assert sha256_file(f"{P1}/windows_train.npz") == WIN_SHA, "windows_train.npz changed"
    A = np.load(f"{P1}/windows_train.npz", allow_pickle=False)
    clips = A["clips"]
    keys = {}
    for pop in ("pstop", "pmove"):
        for w in np.load(f"{P1}/sample_{pop}.npz")["idx"]:
            keys.setdefault((str(clips[int(A["clip_i"][w])]), int(A["t"][w])), set()).add("train")
    n_train = len(keys)
    E = np.load(f"{P1}/eval141_features.npz", allow_pickle=True)
    for c, t in zip(E["clip"], E["t0"]):
        keys.setdefault((str(c), int(t)), set()).add("eval")
    ks = sorted(keys)
    clip = np.array([k[0] for k in ks])
    t = np.array([k[1] for k in ks], np.int32)
    in_tr = np.array(["train" in keys[k] for k in ks])
    in_ev = np.array(["eval" in keys[k] for k in ks])
    np.savez(f"{RAW}/frames.npz", clip=clip, t=t, in_train=in_tr, in_eval=in_ev)
    meta = {"n_frames": len(ks), "n_train_pairs": n_train, "n_clips": int(len(set(clip.tolist()))), "n_eval_pairs": int(in_ev.sum()), "n_both": int((in_tr & in_ev).sum()), "t_min": int(t.min()), "t_max": int(t.max())}
    json.dump(meta, open(f"{RAW}/frames_meta.json", "w"), indent=1)
    print(meta)


def load_frames():
    F = np.load(f"{RAW}/frames.npz")
    return F["clip"], F["t"], F["in_train"], F["in_eval"]


def clip_frames(clip, ts):
    """-> (raw idx per t, vid uint8 [n,3,1080,1920]) for the v2ep frames 2t."""
    fi = C.frame_idx_for(f"{C.MP4_DIR}/{clip}.timestamps.parquet")
    assert all(2 * int(t) < len(fi) for t in ts), (clip, ts, len(fi))
    raw = np.array([fi[2 * int(t)] for t in ts])
    return raw, C.raw_frames(f"{C.MP4_DIR}/{clip}.mp4", raw, threads=1)


def png_bytes(chw):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(chw.permute(1, 2, 0).numpy())).save(b, "PNG")
    return b.getvalue()


def do_shard(args):
    si, clist = args
    import torch
    torch.set_num_threads(1)
    zp, done = f"{SCR}/shards/shard_{si:04d}.zip", f"{SCR}/shards/shard_{si:04d}.done"
    if os.path.exists(done):
        return si, "skip"
    F = np.load(f"{RAW}/frames.npz")
    clip, t = F["clip"], F["t"]
    fr = C.frame_full()
    members, t0 = [], time.time()
    with zipfile.ZipFile(zp + ".tmp", "w", zipfile.ZIP_STORED) as z:
        for c in clist:
            ts = t[clip == c]
            raw, vid = clip_frames(c, ts)
            intr = C.intr_for(c)
            outs = {k: C.rectify(vid, intr, fr, C.BOXES[k]) for k in C.CROPS}
            for j, tt in enumerate(ts):
                for k in C.CROPS:
                    z.writestr(f"{c}__{int(tt)}__{k}.jpg", C.jpeg_bytes(outs[k][j]))
                members.append([c, int(tt), int(raw[j])])
    os.replace(zp + ".tmp", zp)
    json.dump({"shard": si, "clips": clist, "frames": members, "zip_bytes": os.path.getsize(zp), "seconds": time.time() - t0}, open(zp.replace(".zip", ".idx.json"), "w"))
    open(done, "w").write("ok")
    return si, f"{len(members)} frames {time.time() - t0:.0f}s"


def stage_shards(workers):
    import multiprocessing as mp
    os.makedirs(f"{SCR}/shards", exist_ok=True)
    clip, t, _, _ = load_frames()
    uc = sorted(set(clip.tolist()))
    jobs = [(i // CLIPS_PER_SHARD, uc[i:i + CLIPS_PER_SHARD]) for i in range(0, len(uc), CLIPS_PER_SHARD)]
    print(len(uc), "clips", len(jobs), "shards", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        for si, msg in pool.imap_unordered(do_shard, jobs):
            print("shard", si, msg, time.strftime("%H:%M:%S"), flush=True)
    print("SHARDS-DONE", flush=True)


def stage_align():
    import torch
    torch.set_num_threads(2)
    clip, t, in_tr, in_ev = load_frames()
    rng = np.random.default_rng(11)
    tr = rng.choice(np.flatnonzero(in_tr), 24, replace=False)
    ev = rng.choice(np.flatnonzero(in_ev), 8, replace=False)
    sel = [(int(i), "train") for i in tr] + [(int(i), "eval") for i in ev]
    os.makedirs(SCR, exist_ok=True)
    meta = []
    with zipfile.ZipFile(f"{SCR}/align_pack.zip.tmp", "w", zipfile.ZIP_STORED) as z:
        for n, (i, src) in enumerate(sel):
            c, tt = str(clip[i]), int(t[i])
            raw, vid = clip_frames(c, [tt])
            intr = C.intr_for(c)
            full = C.rectify(vid, intr, C.frame_cache())[0]
            z.writestr(f"a{n:02d}_full.png", png_bytes(full))
            nat = C.rectify(vid, intr, C.frame_full())
            for k in C.CROPS:
                r0, r1, c0, c1 = C.BOXES[k]
                cr = nat[0, :, r0:r1, c0:c1]
                z.writestr(f"a{n:02d}_{k}.png", png_bytes(cr))
                z.writestr(f"a{n:02d}_{k}.jpg", C.jpeg_bytes(cr))
            meta.append({"n": n, "clip": c, "t": tt, "src": src, "raw_idx": int(raw[0])})
        z.writestr("meta.json", json.dumps(meta))
    os.replace(f"{SCR}/align_pack.zip.tmp", f"{SCR}/align_pack.zip")
    json.dump(meta, open(f"{RAW}/align_meta.json", "w"), indent=1)
    print("align pack", os.path.getsize(f"{SCR}/align_pack.zip"), "bytes", sha256_file(f"{SCR}/align_pack.zip"))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "frames":
        build_frames()
    elif cmd == "align":
        stage_align()
    elif cmd == "shards":
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3
        stage_shards(w)
