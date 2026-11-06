#!/usr/bin/env python3
"""Same-device sensitivity references for identity_pod.py (clip 8f387977, banked frames a, MODULE path, pod GPU).
What size of token change does (i) the AR-2 pixel-scale mutant and (ii) a different encoder batch shape (16 instead of 8) cause on THIS device?
Reads the numbers beside the 2e-2 device effect, so the 'mutant caught' control is not read against a floor it cannot beat."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/root/identity_pod")
import identity_pod as I
import torch

I.D = I.load_driver(); I.E = I.D.E
D, E = I.D, I.E
enc = D.build_encoder("cuda")
pool_field = E.builder_pool_field()
clip = "8f387977-4086-466d-bba4-b04f8cdc0743"
o = torch.load(I.FRAMES_A / "train" / f"{clip}.v2ep.pt", map_location="cpu", weights_only=False)
n = E.n_cache_frames(o)
_, base = I.encode_module(enc, o, n, pool_field)
res = {"clip": clip, "n_frames": n}
restore = D.apply_mutant(enc, D.DEFAULT_MUTANT, torch.device("cuda"))
try:
    _, fm = I.encode_module(enc, o, n, pool_field)
finally:
    restore()
res["mutant_PIXEL_SCALE_255_to_256_vs_unmutated_same_device"] = I.compare(fm, base)
feats = []
with torch.inference_mode():
    for j in range(0, n, 16):
        feats.append(enc.field(E.decode_cache_frames(o, list(range(j, min(j + 16, n))))).cpu())
f16 = pool_field(torch.cat(feats), 4, 10).to(torch.float8_e4m3fn).contiguous()
res["batch16_vs_batch8_same_device"] = I.compare(f16, base)
json.dump(res, open("/root/identity_pod/out/ident_extra.json", "w"), indent=1)
print(json.dumps({k: (v["rel_l2"], v["frac_bitwise_equal"]) for k, v in res.items() if isinstance(v, dict)}))
