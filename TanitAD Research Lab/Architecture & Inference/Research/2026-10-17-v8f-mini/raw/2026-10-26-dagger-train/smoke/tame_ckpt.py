"""Smoke helper: take a checkpoint the trainer wrote and make the closed loop STABLE enough for the bank generator (a random synthetic prior is an unstable loop and every own state is cut for heading).

The same taming the DATA tests apply (``tiny_model``: prior x 0.02, a small normal residual on head_o). The architecture and the config are untouched, so the result is still a valid ``--init-ckpt`` of the same arm.
usage: tame_ckpt.py <in ckpt_last.pt> <out ckpt>
"""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
blob = torch.load(src, map_location="cpu", weights_only=False)
sd = blob["model"]
for k in list(sd):
    if k.startswith("prior.") and sd[k].is_floating_point():
        sd[k] = sd[k] * 0.02
    if k.startswith("head_o.2.weight"):
        sd[k] = torch.randn(sd[k].shape, generator=torch.Generator().manual_seed(0)) * 0.05
torch.save(blob, dst)
print("tamed", src, "->", dst)
