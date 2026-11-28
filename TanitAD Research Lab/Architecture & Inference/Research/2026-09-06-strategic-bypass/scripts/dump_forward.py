# -*- coding: utf-8 -*-
"""Dump a deterministic RefCV3 forward, for the OFF-is-byte-identical proof.

Run in BOTH trees (pre-patch baseline and post-patch) with the SAME argument.
It must never mention `no_strategic`: the baseline tree does not have the
field, and a script that branches on it would not be the same script.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "stack"))

from tanitad.refs import refc_v3 as v3   # noqa: E402

out_path = sys.argv[1]

torch.manual_seed(0)
cfg = v3.refc_v3_smoke_config(True)       # hier arm, tiny -- the default build
model = v3.RefCV3Model(cfg)
model.eval()

h, w = cfg.core.encoder.image_hw()
g = torch.Generator().manual_seed(1234)
b = 3
frames = torch.rand(b, cfg.core.window, cfg.core.encoder.in_channels, h, w,
                    generator=g)

dump = {}
for nav_idx in (0, 1, 2, 3):
    kw = dict(frames=frames,
              nav_cmd=torch.full((b,), nav_idx, dtype=torch.long),
              v0=torch.tensor([3.0, 7.0, 11.0]))
    with torch.no_grad():
        o = model(**kw)
    for k, v in o.items():
        if torch.is_tensor(v):
            dump[f"nav{nav_idx}/{k}"] = v.detach().clone()

# the parameters too: same seed must give the same init, or the RNG stream
# moved and "byte-identical" would be measuring the wrong thing
for k, p in model.state_dict().items():
    dump[f"sd/{k}"] = p.detach().clone()

torch.save(dump, out_path)
print(f"wrote {out_path}: {len(dump)} tensors "
      f"({sum(1 for k in dump if k.startswith('sd/'))} state_dict entries)")
