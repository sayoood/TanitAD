"""c2 worker: load a checkpoint into the model of ONE tree and dump the outputs of forward (p_gt 0) and forward_infer on the fixed tiny batch.
Usage: lc_dump.py <tree: base|patch> <ckpt> <out.pt> <lever: off|on> [perturb]"""
import sys

import torch

tree, ckpt, out, lever = sys.argv[1:5]
import lc_common as C

C.use_tree(tree)
from tanitad.models import v8f_mini as M

ck = torch.load(ckpt, map_location="cpu", weights_only=False)
cfg_d = dict(ck["cfg"])
if lever == "on":
    cfg_d["lead_cond"] = "on"
cfg = M.V8FMiniConfig(**cfg_d)
batch, J = C.world_batch("c2", hmax=cfg.hmax)
model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
res = model.load_state_dict(ck["model"], strict=(lever == "off"))
if lever == "on":
    assert sorted(res.missing_keys) == ["lead_cond.bias", "lead_cond.weight"] and not res.unexpected_keys, res
model.eval()
with torch.no_grad():
    o1 = model.forward(batch, p_gt=0.0)
    o2 = model.forward_infer({k: v for k, v in batch.items() if k in M.INFERENCE_INPUT_KEYS})
keep = lambda o: {k: v.detach().clone() for k, v in o.items() if torch.is_tensor(v)}
torch.save({"forward": keep(o1), "infer": keep(o2), "n_keys_state": len(model.state_dict()), "missing": list(res.missing_keys)}, out)
print("dumped", tree, lever, len(keep(o1)), len(keep(o2)))
