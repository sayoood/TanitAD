"""Helper of test_v8f_optb3.py (C10): print one JSON digest of a built fixed_k WM (state dict, every loss term, the outputs, the gradients, the RNG state) for the tree whose root is argv[1].
mode ``absent`` never names ``wm_rsp_target`` (so it also runs on the 86da79cd tree, which has no such field); ``explicit`` names it as 1.0.
usage: python optb3_digest.py <tree root> <absent|explicit> <work dir>"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

root, mode, work = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3])
for p in (root / "stack", root / "taniteval", root / "stack" / "scripts"):
    sys.path.insert(0, str(p))
from tanitad.models import v8f_mini as M  # noqa: E402

assert Path(M.__file__).resolve().is_relative_to(root), M.__file__
torch.set_num_threads(1)
dim = M.V8FMiniConfig.__dataclass_fields__["wm_dim"].default
g = np.random.RandomState(0)
f = work / "readout.pt"
torch.save({"tag": "T", "K": 10, "lambda": 1.0, "RA": {"mu": g.randn(dim), "sd": np.abs(g.randn(dim)) + 0.5, "W": g.randn(dim, 1) * 0.1, "ym": np.zeros(1)}, "s_cal_swap_heldout": [0.2, 0.04]}, f)
sha = hashlib.sha256(f.read_bytes()).hexdigest()
extra = {"wm_rsp_target": 1.0} if mode == "explicit" else {}
cfg = M.V8FMiniConfig(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, wm="b", hmax=28, kin_mode="yaw", wm_rsp_w=1.0, wm_rsp_readout="fixed_k",
                      wm_rsp_probe=str(f), wm_rsp_probe_sha256=sha, wm_freeze_proj="on", **extra)
torch.manual_seed(0)
wm = M.WMAux(cfg)
gen = torch.Generator().manual_seed(3)
am = torch.randn(2, cfg.hmax, cfg.d, generator=gen, requires_grad=True)
act = torch.randn(2, cfg.hmax, 2, generator=gen)
out = wm.forward_mean(am, act)
(out["rsp/L1"] * 1.0).backward()
h = hashlib.sha256()
for k, v in wm.state_dict().items():
    h.update(k.encode())
    h.update(v.detach().cpu().contiguous().numpy().tobytes())
hg = hashlib.sha256()
for n, p in wm.named_parameters():
    if p.grad is not None:
        hg.update(n.encode())
        hg.update(p.grad.detach().cpu().contiguous().numpy().tobytes())
ho = hashlib.sha256()
for k in sorted(out):
    ho.update(k.encode())
    ho.update(out[k].detach().cpu().contiguous().numpy().tobytes())
print(json.dumps({"state": h.hexdigest(), "grads": hg.hexdigest(), "outputs": ho.hexdigest(), "terms": sorted(out), "rng": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                  "rsp_L1": repr(float(out["rsp/L1"].detach()))}))
