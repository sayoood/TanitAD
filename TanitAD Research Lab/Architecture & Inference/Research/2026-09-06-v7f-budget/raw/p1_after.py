"""AFTER: the v7-tiny census with the grad-unreachable declaration in force.

CONTROLS, all three of which must hold or the change is not what it claims:
  1. TOTAL parameter count is UNCHANGED  -> nothing was deleted.
  2. `n_trainable_by_group_map_alone` == the run's OWN recorded
     `freeze.n_trainable` -> the new number is a WITHHOLDING, not a different
     model; the old number is exactly reproduced by the old rule.
  3. the checkpoint still loads STRICTLY -> `requires_grad` is not serialised.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "stack"))
sys.path.insert(0, str(HERE / "stack" / "scripts"))

from tanitad.models.v6 import V6Stack, apply_stage_freeze     # noqa: E402
from train_v6_staged import (grad_reach_census,               # noqa: E402
                             synthetic_train_batch, v6_loss_step)
from p1_audit2 import build_cfg, build_weights                # noqa: E402

RUN = Path("C:/Users/Admin/tanitad-caches/mm-e19-assets-20260901/"
           "v7tiny_postrain30k")
CFG = json.loads((RUN / "config.json").read_text())
OLD_TRAINABLE = int(CFG["freeze"]["n_trainable"])
OLD_TOTAL = int(CFG["param_report"]["total"])

torch.manual_seed(0)
stack = V6Stack(build_cfg(CFG))
n_tot = sum(p.numel() for p in stack.parameters())
print("[ctrl 1] TOTAL params %d vs run's config.json %d -> %s"
      % (n_tot, OLD_TOTAL, "UNCHANGED" if n_tot == OLD_TOTAL else "CHANGED"))
assert n_tot == OLD_TOTAL, "a parameter was added or removed -- not the deal"

rep = apply_stage_freeze(stack, CFG["stage"])
print("[ctrl 2] group-map-alone trainable %d vs run's recorded %d -> %s"
      % (rep["n_trainable_by_group_map_alone"], OLD_TRAINABLE,
         "IDENTICAL" if rep["n_trainable_by_group_map_alone"] == OLD_TRAINABLE
         else "MISMATCH"))
assert rep["n_trainable_by_group_map_alone"] == OLD_TRAINABLE

sd = torch.load(RUN / "ckpt.pt", map_location="cpu", weights_only=False)
sd = sd.get("model", sd.get("stack", sd))
missing, unexpected = stack.load_state_dict(sd, strict=True), None
print("[ctrl 3] STRICT load of the banked 30k checkpoint -> OK "
      "(missing=%s unexpected=%s)" % (list(missing.missing_keys),
                                      list(missing.unexpected_keys)))

trainable = [p for p in stack.parameters() if p.requires_grad]
n_train = sum(int(p.numel()) for p in trainable)

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
print("=" * 74)
print("v7-tiny (v7tiny_postrain30k) -- BEFORE vs AFTER")
print("=" * 74)
print("  declared trainable   %10d  ->  %10d   (-%d)"
      % (OLD_TRAINABLE, n_train, OLD_TRAINABLE - n_train))
print("  optimizer tensors           138  ->  %10d" % c["n_in_optimizer"])
print("  UNREACHED tensors            42  ->  %10d" % c["n_unreached"])
print("  UNREACHED params        5305667  ->  %10d" % c["unreached_numel"])
print("  overstatement            52.2 %%  ->  %8.1f %%"
      % (100.0 * c["unreached_numel"] / max(n_train, 1)))
print("  withheld by declaration : %d params in %d subtrees"
      % (rep["n_grad_unreachable"], len(rep["grad_unreachable"])))
for k, v in rep["grad_unreachable"].items():
    print("      %-26s %s" % (k, v[:74]))
print("")
print("  STILL unreached (the honest residue -- starved objectives, not dead):")
for m, e in c["unreached_modules"].items():
    print("      %-42s %8d  group=%s" % (m, e["numel"], e["group"]))

(HERE / "p1_after.json").write_text(json.dumps(
    {"run": RUN.name, "old_trainable": OLD_TRAINABLE, "new_trainable": n_train,
     "total_unchanged": n_tot, "freeze": rep, "census": c}, indent=1))
print("")
print("[bank] wrote p1_after.json")
