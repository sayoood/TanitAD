"""G2 -- encoder fidelity: the in-loop encoder on the LOGGED frame of a cached eval-141 window reproduces the banked pooled tokens.
Thresholds fixed before the first measurement: rel-L2 <= 0.05 and cosine >= 0.995 (CPU bf16 vs the GPU-bf16 cache; the sizing's residue is 2.5-2.8 %,
ROUTE.md's device floor 2.9-3.5 %), and the rectified pixels within 3 levels of the cache's own PNG. One frame (ViT-L on CPU)."""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch

import t2a_v8f_encoder as E
import t2a_v8f_policy as P

TREE = P.bootstrap()
ART = "E:/Projects/TanitAD-artifacts"
CACHE = ART + "/sfresh_inputs/cache/eval/clips"
V2EP = "C:/Users/Admin/tanitad-data/refav1-eval141-256rb"
PAI = "C:/Users/Admin/tanitad-data/physicalai"
CLIP = "01be5919-6e61-45ba-a193-5f447efb4b51"
STEP = 40                                    # cache step j; raw frame 2j
MUTATIONS = ("wrong_norm", "n_special_1", "neighbour_frame", "transposed_grid")
THR = {"rel_l2": 0.05, "cosine": 0.995, "pix_max": 3}
_ENC = {}


def encoder(dtype="bf16"):
    if dtype not in _ENC:
        _ENC[dtype] = E.InLoopEncoder(dtype=dtype)
    return _ENC[dtype]


def cyl_frame(raw_index: int) -> torch.Tensor:
    import torchvision
    o = torch.load(f"{V2EP}/{CLIP}.v2ep.pt", map_location="cpu", weights_only=True)
    lens = o["jpeg_len"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    return torchvision.io.decode_png(o["jpeg_buf"][offs[raw_index]:offs[raw_index + 1]].clone())      # uint8 [3,256,640]


def mp4_frame_cyl(raw_index: int, enc):
    """Stage A: the clip's mp4 frame at the builder's frame choice -> in-loop rectify with the clip's per-clip intrinsics. None if the mp4 is absent."""
    import av
    import pandas as pd
    mp4 = f"{PAI}/camera/camera_front_wide_120fov/{CLIP}.mp4"
    tsf = f"{PAI}/camera/camera_front_wide_120fov/{CLIP}.timestamps.parquet"
    if not (os.path.isfile(mp4) and os.path.isfile(tsf)):
        return None, "mp4/timestamps absent"
    from tanitad.data.physicalai import intrinsics_for_clip
    ts = pd.read_parquet(tsf)
    tf = ts[next(c for c in ts.columns if "time" in c.lower())].to_numpy(np.float64)
    span, unit = tf[-1] - tf[0], 1.0
    for cand in (1e9, 1e6, 1e3):
        if span / cand > 1.0:
            unit = cand
            break
    n_target = max(int(span / unit * 10), 4)
    fidx = int(np.searchsorted(tf, np.linspace(tf[0], tf[-1], n_target)).clip(0, len(tf) - 1)[raw_index])
    with av.open(mp4) as c:
        for i, vf in enumerate(c.decode(c.streams.video[0])):
            if i == fidx:
                img = vf.to_ndarray(format="rgb24")
                break
    intr = intrinsics_for_clip(CLIP, PAI)
    if not intr.per_clip:
        return None, "no per-clip intrinsics"
    return enc.rectify(img, intr), f"mp4 frame {fidx} -> raw {raw_index}"


def run_g2(mut=None, assert_pass=False, with_fp32=False):
    enc = encoder("bf16")
    banked = torch.load(f"{CACHE}/{CLIP}.pt", map_location="cpu", weights_only=True)["pool4x10"][STEP].float()
    j = STEP + (1 if mut == "neighbour_frame" else 0)
    cyl = cyl_frame(2 * j)
    saved = (enc.mean.clone(), enc.std.clone(), enc.n_special)
    try:
        if mut == "wrong_norm":
            enc.mean, enc.std = torch.zeros_like(enc.mean), torch.ones_like(enc.std)
        if mut == "n_special_1":
            enc.n_special = 1
        if mut == "transposed_grid":
            f = enc.encode_cyl(cyl[None])
            tok = E.pool_field(f.reshape(1, E.GH, E.GW, -1).transpose(1, 2).reshape(1, E.GH * E.GW, -1)).to(torch.float8_e4m3fn)[0]
        else:
            tok = enc.tokens(cyl, already_cyl=True)
    finally:
        enc.mean, enc.std, enc.n_special = saved
    r = {"clip": CLIP, "step": STEP, "mutation": mut, "dtype": "bf16 CPU", "rel_l2": E.rel_l2(tok.float(), banked), "cosine": E.cosine(tok.float(), banked)}
    if mut is None:
        cyl_a, note = mp4_frame_cyl(2 * STEP, enc)
        r["stageA"] = {"note": note}
        if cyl_a is not None:
            d = (cyl_a.int() - cyl.int()).abs()
            r["stageA"].update(max_abs_level=int(d.max()), frac_equal=float((d == 0).float().mean()))
            tokA = enc.tokens(cyl_a, already_cyl=True)
            r["stageA"].update(tokens_from_mp4_rel_l2=E.rel_l2(tokA.float(), banked), tokens_from_mp4_cosine=E.cosine(tokA.float(), banked))
        if with_fp32:
            t32 = encoder("fp32").tokens(cyl, already_cyl=True)
            r["fp32_cpu"] = {"rel_l2": E.rel_l2(t32.float(), banked), "cosine": E.cosine(t32.float(), banked), "bf16_vs_fp32_rel_l2": E.rel_l2(tok.float(), t32.float())}
        r["snapshot"] = enc.snapshot
    pix_ok = ("stageA" not in r) or ("max_abs_level" not in r["stageA"]) or r["stageA"]["max_abs_level"] <= THR["pix_max"]
    r["pass"] = bool(r["rel_l2"] <= THR["rel_l2"] and r["cosine"] >= THR["cosine"] and pix_ok)
    if assert_pass:
        assert r["pass"], f"G2 RED: rel-L2 {r['rel_l2']:.4f}, cosine {r['cosine']:.4f}"
    return r


if __name__ == "__main__":
    import json
    out = run_g2(with_fp32=True)
    print(json.dumps(out, indent=1))
    os.makedirs(os.path.join(HERE, "..", "raw"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "..", "raw", "G2_result.json"), "w"), indent=1)
