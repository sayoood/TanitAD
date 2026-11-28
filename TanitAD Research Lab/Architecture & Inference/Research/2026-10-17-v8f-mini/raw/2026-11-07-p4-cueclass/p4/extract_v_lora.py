#!/usr/bin/env python3
"""extract_v_lora.py -- the probe windows' features from L's encoder, for PREREG_P4_CUECLASS.md section 4.  RUNS ON THE POD (the v2ep frames live only there).

    python -I extract_v_lora.py --stack /workspace/TanitAD/stack --ckpt <L>/ckpt_last.pt --frames-root <dir of <clip>.v2ep.pt> --probe-raw <dir holding windows_train.npz + sample_*.npz> --out <dir> [--device cuda]
    python -I extract_v_lora.py --selftest-zero-b --stack <stack>          # CPU: B = 0 LoRA == the frozen encoder, bit for bit, on a synthetic frame

What it writes: <out>/V_L_{pstop,pmove}.npy, fp16 [12000, 40960]; row r = window sample_<pop>.npz["idx"][r] = (clip, cache step t) of windows_train.npz; the value is
the CACHE'S tensor for that frame from the MERGED L encoder: W + (alpha/r) B A folded into the q and v slices of every block (v8f_lora_encoder.DinoV3LoRAEncoder.merged_encoder),
builder-shaped batches of 8, fp8 round trip, frame = raw frame 2t of <clip>.v2ep.pt (the same frame the banked V used: pool4x10[t]).  Also <out>/V_L_meta.json (checkpoint sha256, step,
LoRA meta, merge audit) and <out>/V_L_progress.json (resumable: finished clips are skipped).
The LoRA tensors are read from the checkpoint's `lora` key (= enc.lora_state_dict(), saved by stack/scripts/v8f_mini_train.py).
STATUS: written and covered by a CPU zero-B test on a synthetic frame; NOT run on a real v2ep frame (none on the dev box), so the tie to the banked pool4x10 of a real frame is UNVERIFIED."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

D_V = 40 * 1024


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def build_merged(stack, ckpt_path, device):
    """-> (merged twin encoder, meta).  The twin is a plain backbone: no adapters, same preprocessing and pooling."""
    import torch
    sys.path.insert(0, stack)
    from tanitad.models import v8f_lora_encoder as LE
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["lora"]
    mt = sd["meta"]
    assert (mt["r"], float(mt["alpha"]), list(mt["targets"])) == (32, 32.0, ["q", "v"]), f"the card pins r 32 / alpha 32 / q,v; the checkpoint says {mt}"
    enc = LE.DinoV3LoRAEncoder.from_pretrained(r=mt["r"], alpha=mt["alpha"], targets=tuple(mt["targets"]), dropout=0.0, seed=mt["seed"], device="cpu", dtype=torch.bfloat16)
    enc.load_lora_state_dict(sd)
    nz = sum(float(t.float().abs().sum()) > 0 for n, t in sd["tensors"].items() if "lora_B" in n)
    merged = enc.merged_encoder().to(device).eval()
    meta = {"ckpt": os.path.abspath(ckpt_path), "ckpt_sha256": sha256_file(ckpt_path), "step": ck.get("step"), "lora_meta": mt, "n_lora_B_nonzero": nz,
            "n_lora_tensors": len(sd["tensors"]), "builder_encode_episode_sha256": LE.assert_builder_unchanged()}
    return LE, merged, meta


def extract(stack, ckpt, frames_root, probe_raw, out, device, pops):
    import torch
    torch.set_num_threads(4)
    os.makedirs(out, exist_ok=True)
    LE, merged, meta = build_merged(stack, ckpt, device)
    A = np.load(f"{probe_raw}/windows_train.npz", allow_pickle=False)
    clips = [str(c) for c in A["clips"]]
    prog_p = f"{out}/V_L_progress.json"
    prog = json.load(open(prog_p)) if os.path.exists(prog_p) else {}
    by = {}
    mm = {}
    for pop in pops:
        idx = np.load(f"{probe_raw}/sample_{pop}.npz")["idx"]
        path = f"{out}/V_L_{pop}.npy"
        mm[pop] = np.lib.format.open_memmap(path, mode="r+" if os.path.exists(path) else "w+", dtype=np.float16, shape=(len(idx), D_V))
        for row, w in enumerate(idx):
            by.setdefault(int(A["clip_i"][w]), []).append((pop, row, int(A["t"][w])))
    t0 = time.time()
    for j, (ci, items) in enumerate(sorted(by.items())):
        c = clips[ci]
        if prog.get(c):
            continue
        o = torch.load(f"{frames_root}/{c}.v2ep.pt", map_location="cpu", weights_only=False)
        ts = sorted({t for _, _, t in items})
        assert max(ts) < LE.n_cache_frames(o), (c, max(ts), LE.n_cache_frames(o))
        frames = LE.decode_cache_frames(o, ts).to(device)
        with torch.no_grad():
            pooled = LE.pooled_builder_shaped(merged, frames).float().cpu().reshape(len(ts), -1)
        pos = {t: i for i, t in enumerate(ts)}
        for pop, row, t in items:
            mm[pop][row] = pooled[pos[t]].to(torch.float16).numpy()
        prog[c] = True
        if j % 100 == 0:
            for m in mm.values():
                m.flush()
            json.dump(prog, open(prog_p, "w"))
            print(j, len(by), f"{time.time() - t0:.0f}s", flush=True)
    for m in mm.values():
        m.flush()
    json.dump(prog, open(prog_p, "w"))
    meta.update({"rows": {p: int(m.shape[0]) for p, m in mm.items()}, "clips": len(by), "seconds": time.time() - t0})
    json.dump(meta, open(f"{out}/V_L_meta.json", "w"), indent=1)
    return meta


def selftest_zero_b(stack):
    """CPU: a B = 0 LoRA, merged, must reproduce the frozen cache-builder encoder BIT FOR BIT on a frame (synthetic uint8 256x640: no real v2ep frame exists on the dev box)."""
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, stack)
    from tanitad.models import v8f_lora_encoder as LE
    enc = LE.DinoV3LoRAEncoder.from_pretrained(r=32, alpha=32.0, targets=("q", "v"), device="cpu", dtype=torch.bfloat16, verify_sha=False)
    assert enc.lora_state_dict()["tensors"], "no LoRA tensors"
    assert all(float(t.float().abs().sum()) == 0 for n, t in enc.lora_state_dict()["tensors"].items() if "lora_B" in n), "B is not zero at build"
    merged = enc.merged_encoder().eval()
    ref = LE.frozen_plain_encoder(device="cpu", dtype=torch.bfloat16).eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 256, (8, 3, 256, 640), generator=g, dtype=torch.uint8)
    with torch.no_grad():
        a, b = LE.pooled_builder_shaped(merged, x), LE.pooled_builder_shaped(ref, x)
    return {"bit_for_bit": bool(torch.equal(a, b)), "max_abs_diff": float((a - b).abs().max()), "shape": list(a.shape)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--selftest-zero-b", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--frames-root")
    ap.add_argument("--probe-raw")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pops", default="pstop,pmove")
    a = ap.parse_args()
    if a.selftest_zero_b:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print(json.dumps(selftest_zero_b(a.stack)))
    else:
        print(json.dumps(extract(a.stack, a.ckpt, a.frames_root, a.probe_raw, a.out, a.device, a.pops.split(",")), indent=1))
