"""The FULL gradient-reach census at v7-tiny's EXACT config -- not just norms.

The 1-D `.weight` audit could only ever see LayerNorms. This runs the same
measurement over EVERY parameter, so a dead Linear inside a live group is
visible too.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "stack"))
sys.path.insert(0, str(HERE / "stack" / "scripts"))

from tanitad.models.v6 import V6Stack, apply_stage_freeze
from train_v6_staged import (V6LossWeights, grad_reach_census,
                             synthetic_train_batch, v6_loss_step)
from p1_audit2 import build_cfg, build_weights

RUN = Path("C:/Users/Admin/tanitad-caches/mm-e19-assets-20260901/"
           "v7tiny_postrain30k")
CFG = json.loads((RUN / "config.json").read_text())
print("run=%s stage=%s horizons=%s o5_k=%s"
      % (RUN.name, CFG["stage"], CFG["v6_config"]["predictor"]["horizons"],
         CFG["args"].get("o5_k")))

torch.manual_seed(0)
stack = V6Stack(build_cfg(CFG))
assert sum(p.numel() for p in stack.parameters()) == CFG["param_report"]["total"]
apply_stage_freeze(stack, CFG["stage"])
trainable = [p for p in stack.parameters() if p.requires_grad]
n_train_numel = sum(p.numel() for p in trainable)
assert n_train_numel == CFG["freeze"]["n_trainable"]
print("[ctrl] geometry + freeze BOTH bit-identical to the run's config.json")

stack.zero_grad(set_to_none=True)
b = synthetic_train_batch(stack, batch=2, k=10, seed=0)
b["gt_wp"] = torch.randn(2, 10, 2, generator=torch.Generator().manual_seed(0))
out = v6_loss_step(stack, b, stage=CFG["stage"], weights=build_weights(CFG),
                   o1_k=10, o5_k=int(CFG["args"].get("o5_k", 1)),
                   o5_form=CFG["args"].get("o5_form", "l1"),
                   generator=torch.Generator().manual_seed(0),
                   sigreg_generator=torch.Generator().manual_seed(0))
out["loss"].backward()
c = grad_reach_census(stack, trainable)

print("")
print("OPTIMIZER tensors           : %d" % c["n_in_optimizer"])
print("  reached by a gradient     : %d" % c["n_reached"])
print("  UNREACHED (grad is None)  : %d" % c["n_unreached"])
print("trainable params (numel)    : %d (%.2f M)"
      % (n_train_numel, n_train_numel / 1e6))
print("UNREACHED params (numel)    : %d (%.2f M) = %.1f %% of the declared "
      "trainable budget" % (c["unreached_numel"], c["unreached_numel"] / 1e6,
                            100.0 * c["unreached_numel"] / n_train_numel))
print("")
roll: dict[str, int] = {}
for m, e in c["unreached_modules"].items():
    roll[m.split(".")[0]] = roll.get(m.split(".")[0], 0) + e["numel"]
print("UNREACHED, rolled up to the top-level module:")
for m, v in sorted(roll.items(), key=lambda kv: -kv[1]):
    g = stack.group_of([n for n in c["unreached_tensors"]
                        if n.startswith(m + ".")][0])
    print("  %-22s %10d params (%.2f M)  group=%s" % (m, v, v / 1e6, g))

# CONTROL: the same census with the objectives ON must shrink the unreached set
import dataclasses
w_on = dataclasses.replace(build_weights(CFG), o1_ctrl=1.0, o1_fact=1.0,
                           o1_scene=1.0, o3_masked=1.0)
torch.manual_seed(0)
s2 = V6Stack(build_cfg(CFG))
apply_stage_freeze(s2, CFG["stage"])
tr2 = [p for p in s2.parameters() if p.requires_grad]
s2.zero_grad(set_to_none=True)
b2 = synthetic_train_batch(s2, batch=2, k=10, seed=0)
b2["gt_wp"] = torch.randn(2, 10, 2, generator=torch.Generator().manual_seed(0))
o2 = v6_loss_step(s2, b2, stage=CFG["stage"], weights=w_on, o1_k=10,
                  o5_k=int(CFG["args"].get("o5_k", 1)),
                  o5_form=CFG["args"].get("o5_form", "l1"),
                  generator=torch.Generator().manual_seed(0),
                  sigreg_generator=torch.Generator().manual_seed(0))
o2["loss"].backward()
c2 = grad_reach_census(s2, tr2)
print("")
print("MUTATION CONTROL (O1/O3 switched ON, same geometry, same stage):")
print("  UNREACHED tensors %d -> %d   params %.2f M -> %.2f M"
      % (c["n_unreached"], c2["n_unreached"], c["unreached_numel"] / 1e6,
         c2["unreached_numel"] / 1e6))
print("  still unreached: %s" % sorted(c2["unreached_modules"]))

(HERE / "p1_fullcensus.json").write_text(json.dumps(
    {"run": RUN.name, "stage": CFG["stage"],
     "trainable_numel": n_train_numel, "off": c, "on_mutation": c2}, indent=1))
print("")
print("[bank] wrote p1_fullcensus.json")
