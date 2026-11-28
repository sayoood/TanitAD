#!/usr/bin/env python3
"""encode_crops.py -- BOX 2 (RTX 4060), frozen DINOv3-L/16, bf16.  PREREG_FROZEN.md sections 2-3.
  --mode align : 32 full cache-geometry frames (256x640 PNG) -> pool4x10 (fp32 4x4 average, the cache's own recipe) + the same frames' crops from PNG and from JPEG -> out/align_*.npy
  --mode crops : every shard zip in --shards -> out/feat_<shard>.npy  fp16 [n_frames, 2 (TL, LEAD), 10 (CLS, mean patch, 2x4 grid x8), 1024] in the frame order of the shard's .idx.json
Hard-fails without CUDA; resumable per shard (atomic .tmp -> os.replace); pauses (never kills) while the GPU is >= 78 C."""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

MID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
W_SHA = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
CROP_GRID = {"TL": (21, 72), "LEAD": (21, 60)}
CROPS = ("TL", "LEAD")
BATCH = 2


def gpu_temp():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def cool():
    t = gpu_temp()
    if t >= 78:
        print(f"COOL-PAUSE gpu {t} C", flush=True)
        while gpu_temp() >= 72:
            time.sleep(20)


def build_model():
    from transformers import AutoImageProcessor, AutoModel
    assert torch.cuda.is_available(), "ZZNO-CUDAZZ"
    proc = AutoImageProcessor.from_pretrained(MID, local_files_only=True)
    model = AutoModel.from_pretrained(MID, local_files_only=True, dtype=torch.bfloat16).eval().to("cuda")
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).to("cuda", torch.bfloat16)
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1).to("cuda", torch.bfloat16)
    n_special = 1 + getattr(model.config, "num_register_tokens", 0)
    assert n_special == 5 and tuple(proc.image_mean) == (0.485, 0.456, 0.406), "encoder geometry changed"
    return model, mean, std


def weights_sha():
    p = glob.glob("D:/tanitad/hf/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/*/model.safetensors")
    h = hashlib.sha256()
    with open(p[0], "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest(), p[0]


@torch.inference_mode()
def tokens(model, mean, std, x_u8):
    x = x_u8.to("cuda").to(torch.bfloat16) / 255.0                  # the cache's own ops (dinov3_fp8_encode_ship.encode_episode)
    return model(pixel_values=(x - mean) / std).last_hidden_state


def crop_feats(h, gh, gw):
    B = h.shape[0]
    assert h.shape[1] == 5 + gh * gw, h.shape
    cls = h[:, 0].float()
    p = h[:, 5:].float()
    mp = p.mean(1)
    g = F.adaptive_avg_pool2d(p.reshape(B, gh, gw, -1).permute(0, 3, 1, 2), (2, 4)).flatten(2).permute(0, 2, 1)      # [B, 8, 1024] row-major
    return torch.cat([cls[:, None], mp[:, None], g], 1)


def pool4x10(h):
    f = h[:, 5:].float()
    T = f.shape[0]
    assert f.shape[1] == 640
    return f.reshape(T, 4, 4, 10, 4, -1).mean(dim=(2, 4)).reshape(T, 40, -1)       # == pull_encode_pooled.pool_field


def img(b):
    from PIL import Image
    return torch.from_numpy(np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))).permute(2, 0, 1).contiguous()


def atomic_npy(path, arr):
    np.save(path + ".tmp.npy", arr)
    os.replace(path + ".tmp.npy", path)


def run_align(a, model, mean, std):
    z = zipfile.ZipFile(a.pack)
    meta = json.loads(z.read("meta.json"))
    n = len(meta)
    t0 = time.time()
    full = torch.stack([img(z.read(f"a{i:02d}_full.png")) for i in range(n)])
    pools = []
    for j in range(0, n, 8):
        pools.append(pool4x10(tokens(model, mean, std, full[j:j + 8])).cpu())
    atomic_npy(f"{a.out}/align_full_pool4x10.npy", torch.cat(pools).numpy().astype(np.float32))
    for ext in ("png", "jpg"):
        res = []
        for k in CROPS:
            gh, gw = CROP_GRID[k]
            fs = []
            for j in range(0, n, BATCH):
                x = torch.stack([img(z.read(f"a{i:02d}_{k}.{ext}")) for i in range(j, min(n, j + BATCH))])
                fs.append(crop_feats(tokens(model, mean, std, x), gh, gw).cpu())
            res.append(torch.cat(fs))
        atomic_npy(f"{a.out}/align_crops_{ext}.npy", torch.stack(res, 1).numpy().astype(np.float32))
    sha, p = weights_sha()
    info = {"weights_sha256": sha, "weights_ok": sha == W_SHA, "weights_path": p, "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "n": n, "seconds": time.time() - t0, "gpu_temp_end": gpu_temp()}
    json.dump(info, open(f"{a.out}/align_info.json", "w"), indent=1)
    print("ALIGN-DONE", json.dumps(info), flush=True)


def run_shard(a, model, mean, std, zp):
    name = os.path.basename(zp)[:-4]
    out = f"{a.out}/feat_{name}.npy"
    if os.path.exists(out) and os.path.exists(out.replace(".npy", ".json")):
        return name, "skip"
    idx = json.load(open(zp.replace(".zip", ".idx.json")))
    fr = idx["frames"]
    z = zipfile.ZipFile(zp)
    n = len(fr)
    res = np.zeros((n, 2, 10, 1024), np.float16)
    t0 = time.time()
    ex = ThreadPoolExecutor(3)

    def load(k, lo, hi):
        return torch.stack(list(ex.map(lambda r: img(z.read(f"{fr[r][0]}__{fr[r][1]}__{k}.jpg")), range(lo, hi))))

    G = 16                                                           # decode-ahead group (frames)
    for ci, k in enumerate(CROPS):
        gh, gw = CROP_GRID[k]
        fut = ex.submit(load, k, 0, min(n, G))
        for lo in range(0, n, G):
            x = fut.result()
            if lo + G < n:
                fut = ex.submit(load, k, lo + G, min(n, lo + 2 * G))
            for j in range(0, x.shape[0], BATCH):
                f = crop_feats(tokens(model, mean, std, x[j:j + BATCH]), gh, gw)
                res[lo + j:lo + j + f.shape[0], ci] = f.cpu().numpy().astype(np.float16)
        cool()
    ex.shutdown()
    assert np.isfinite(res).all() and float(np.abs(res).mean()) > 1e-3
    atomic_npy(out, res)
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    json.dump({"shard": name, "n": n, "sha256": sha, "seconds": time.time() - t0, "frames": fr, "gpu_temp": gpu_temp()}, open(out.replace(".npy", ".json"), "w"))
    return name, f"{n} frames {time.time() - t0:.0f}s gpu {gpu_temp()} C"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("align", "crops"))
    ap.add_argument("--pack", default="D:/tanitad/jobs/probe5/align_pack.zip")
    ap.add_argument("--shards", default="D:/tanitad/jobs/probe5/shards")
    ap.add_argument("--out", default="D:/tanitad/jobs/probe5/out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(2)
    model, mean, std = build_model()
    print("MODEL-OK", torch.__version__, torch.cuda.get_device_name(0), "temp", gpu_temp(), flush=True)
    if a.mode == "align":
        run_align(a, model, mean, std)
    else:
        for zp in sorted(glob.glob(f"{a.shards}/shard_*.zip")):
            if not os.path.exists(zp.replace(".zip", ".idx.json")):
                continue
            cool()
            nm, msg = run_shard(a, model, mean, std, zp)
            print("shard", nm, msg, time.strftime("%H:%M:%S"), flush=True)
        print("ENCODE-DONE", flush=True)
