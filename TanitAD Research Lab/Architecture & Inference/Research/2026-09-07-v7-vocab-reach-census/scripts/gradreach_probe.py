#!/usr/bin/env python3
"""Which parameter tensors actually RECEIVE GRADIENT under the live arm's flags.

⛔ This is the discriminator the tac_goal_head docstring itself names:
``p.grad is None`` means the head is NOT WIRED, and it is distinguishable from
"switched off" only because that module's loss is unguarded. Run the REAL
trainer's config pin, the REAL model, the REAL loss, one backward.
"""
from __future__ import annotations
import json, sys, os
sys.stdout.reconfigure(encoding="utf-8")

import torch

sys.path.insert(0, os.environ["STACK"])
sys.path.insert(0, os.path.join(os.environ["STACK"], "scripts"))
import refc_v3_train as T  # noqa: E402

LIVE = os.environ["LIVE_ARGV"].split("\x1f")

parser = T.build_parser()
args = parser.parse_args(LIVE)
print("[probe] parsed args:",
      {k: v for k, v in vars(args).items()
       if k in ("arm", "v7_labels", "tac_goal_tok_head", "nav_from_v7",
                "goal_str", "max_speed_input", "ego_state_inject",
                "sel_refined", "agents", "sampler", "no_strategic")})

cfg = T.v3.refc_v3_smoke_config(True)
cfg = T._pin_trainer_cfg(cfg, args)
print("[probe] cfg.tac_vocab_version =", getattr(cfg, "tac_vocab_version", None))
print("[probe] cfg.tac_goal_tok_head =", getattr(cfg, "tac_goal_tok_head", None))
print("[probe] cfg.max_speed_input   =", getattr(cfg, "max_speed_input", None))

torch.manual_seed(0)
model = T.v3.RefCV3Model(cfg)
model._w_goal_point = float(getattr(args, "goal_point_w", 0.0) or 0.0)
model.train()
print("[probe] model.tac_goal_tok_head =",
      type(getattr(model, "tac_goal_tok_head", None)).__name__)
print("[probe] model.max_speed_cond    =",
      type(getattr(model, "max_speed_cond", None)).__name__)

eps = T._synth_episodes(2, cfg.core, seed=0)
ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                 channels=cfg.core.encoder.in_channels)
batch = torch.utils.data.default_collate([ds[0], ds[1]])

v7l = T.v7l
n_lat, n_lon = len(v7l.HEADS["tac_lat"]), len(v7l.HEADS["tac_lon"])
batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
batch["lon_v7"] = torch.tensor([n_lon - 1, v7l.IGNORE_ID], dtype=torch.long)
if getattr(args, "nav_from_v7", False):
    pass
    batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
    batch["nav_valid"] = torch.tensor([True, True])

losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
print("[probe] loss keys:", sorted(losses))
# ⛔ THE TRAINER'S OWN TOTAL, not a sum of everything that happens to carry
# grad -- summing diagnostics would ADD gradient paths the run does not have
# and turn an unwired head into a reached one.
total = losses["loss"]
assert total.requires_grad, "the trainer's total loss does not require grad"

# ⭐ MUTATION CONTROL: with MUTATE=1 attach a loss to the head's own logits and
# prove the NOT-WIRED verdict is REACHABLE in the other direction. A detector
# that cannot go GREEN on a wired head certifies nothing when it says NOT WIRED.
if os.environ.get("MUTATE") == "1":
    cache = model(batch["frames"].float(), v0=batch.get("v0"),
                  ego_state=None) if False else None
    tg = getattr(model, "tac_goal_tok_head", None)
    assert tg is not None
    z = torch.ones(2, tg.net.in_features if hasattr(tg.net, "in_features")
                   else tg.net[0].in_features)
    total = total + tg(z).sum()
    print("[probe] MUTATION ACTIVE: a loss now touches tac_goal_tok_head")
total.backward()

# ---- per-module gradient census ---------------------------------------- #
mods = {}
for name, mod in model.named_children():
    ps = list(mod.parameters())
    if not ps:
        continue
    n_par = sum(p.numel() for p in ps)
    n_none = sum(1 for p in ps if p.grad is None)
    gnorm = sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None)
    mods[name] = {"n_tensors": len(ps), "n_params": n_par,
                  "n_grad_none": n_none, "grad_abs_sum": gnorm,
                  "verdict": ("NOT WIRED (all grads None)" if n_none == len(ps)
                              else "ZERO GRAD" if gnorm == 0.0
                              else "GRADIENT REACHES")}
for k in sorted(mods):
    m = mods[k]
    print(f"  {k:28s} params={m['n_params']:>10,d} "
          f"grad_none={m['n_grad_none']}/{m['n_tensors']} "
          f"|g|={m['grad_abs_sum']:.6g}  {m['verdict']}")

out = {"argv": LIVE, "modules": mods,
       "losses": {k: (float(v) if torch.is_tensor(v) and v.numel() == 1
                      else str(type(v).__name__))
                  for k, v in losses.items()},
       "tac_goal_tok_head_built":
           getattr(model, "tac_goal_tok_head", None) is not None,
       "max_speed_cond_built":
           getattr(model, "max_speed_cond", None) is not None}
json.dump(out, open(os.environ["OUT"], "w", encoding="utf-8"), indent=1)
print("[probe] wrote", os.environ["OUT"])
