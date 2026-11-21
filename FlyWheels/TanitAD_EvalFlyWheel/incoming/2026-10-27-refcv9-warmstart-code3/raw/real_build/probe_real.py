"""Real refcv8-r101-s0 build (CPU, refcv7_loader.build_model, strict load of ckpt_ema_48852.pt): the declared
grad-unreachable subtrees, the frozen-at-build parameters, and per-module tensor/value counts.
usage: python probe_real.py <tree> <config.json> <ckpt> <out.json>"""
import json, os, sys, time, collections
tree, cfgp, ckpt, outp = sys.argv[1:5]
os.environ["REFCV7_REPO"] = tree
sys.path.insert(0, os.path.join(tree, "stack"))
from tanitad.eval import refcv7_loader as L
L.bootstrap()
import tanitad, torch
from tanitad.models import _gradreach as GR
torch.set_num_threads(4)
t0 = time.time()
model, cfg, args, rec = L.build_model(json.load(open(cfgp, encoding="utf-8")), ckpt, device="cpu")
decl = GR.grad_unreachable_prefixes(model)
frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
mods = collections.OrderedDict()
for n, p in model.named_parameters():
    parts = n.split(".")
    t = ".".join(parts[:2]) if parts[0] in ("core", "_perception", "_map_hires") else parts[0]
    e = mods.setdefault(t, [0, 0, 0, 0]); e[0] += 1; e[1] += int(p.numel())
    if not p.requires_grad:
        e[2] += 1; e[3] += int(p.numel())
json.dump({"tanitad_file": tanitad.__file__, "build_s": round(time.time() - t0, 1),
           "argv_flags": {k: getattr(args, k, None) for k in ("no_strategic", "graft_tac8_prior", "r8_no_rc", "w_tac_v6",
                                                               "r8_tf_start", "r8_tf_end", "r8_alloc_emit_start", "steps")},
           "declared": decl, "frozen_at_build": frozen,
           "modules": mods, "dvb": rec.get("declared_vs_built"),
           "params": [(n, list(p.shape), int(p.numel()), bool(p.requires_grad)) for n, p in model.named_parameters()]},
          open(outp, "w", encoding="utf-8"), indent=0)
print("tanitad", tanitad.__file__, "declared", len(decl), "frozen", len(frozen), "build_s", round(time.time() - t0, 1))
