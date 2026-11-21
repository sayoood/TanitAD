"""make_synth_ckpt.py TREE SRC_CKPT OUT_CKPT [--route-emb-scale S] -- TEST FIXTURE ONLY (never a result): the N1DH checkpoint rebuilt as an aux_route=cond model whose route_head is seeded-random and route_emb small seeded-random,
so the Z-POST / K-LIM machinery sees a model that emits route_logits.  Deleted after the tests (DELETIONS.log)."""
import sys
import os
tree, src, dst = sys.argv[1:4]
scale = float(sys.argv[sys.argv.index("--route-emb-scale") + 1]) if "--route-emb-scale" in sys.argv else 0.02
sys.dont_write_bytecode = True
sys.path[:0] = [tree + "/stack", tree + "/taniteval"]
import torch
from tanitad.models import v8f_mini as M
ck = torch.load(src, map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"])
cfg["aux_route"], cfg["w_aux_route"] = "cond", 0.5
c = M.V8FMiniConfig(**cfg)
m = M.V8FMini(c, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
missing, unexpected = m.load_state_dict(ck["model"], strict=False)
print("missing", missing, "unexpected", unexpected)
g = torch.Generator().manual_seed(7)
with torch.no_grad():
    m.route_head.weight.copy_(torch.randn(m.route_head.weight.shape, generator=g) * 0.05)
    m.route_head.bias.copy_(torch.randn(m.route_head.bias.shape, generator=g) * 0.5)
    m.route_emb.weight.copy_(torch.randn(m.route_emb.weight.shape, generator=g) * scale)
out = dict(ck)
out["cfg"] = cfg
out["model"] = m.state_dict()
torch.save(out, dst)
print("saved", dst, os.path.getsize(dst))
