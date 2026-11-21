"""BITWISE PARITY: is a `--agents off` / `--agents head` arm unchanged by the
agent_gt patch?  Dumps (params, losses) for one arm from one tree.

Same seed, same config, same batch, same RNG sequence.  The two trees are
separate processes, so each dumps a .pt and a comparator does torch.equal.
"""
from __future__ import annotations
import importlib.util, os, sys

import torch

STACK = os.environ["TANIT_STACK"]
ARM = os.environ["TANIT_ARM"]          # off | head | oracle
OUT = os.environ["TANIT_OUT"]
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(STACK, "scripts"))
_spec = importlib.util.spec_from_file_location(
    "refc_v3_train_parity", os.path.join(STACK, "scripts", "refc_v3_train.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

ARGV = ["--arm", "hier", "--out", "/tmp/parity", "--seed", "0",
        "--agents", ARM, "--agent-queries", "8"]
if ARM == "head":
    ARGV += ["--w-agent", "1.0", "--agent-join", "j.xz"]

args = T.build_parser().parse_args(ARGV)
cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)

torch.manual_seed(1234)
model = T.v3.RefCV3Model(cfg)
model._w_goal_point = 0.0
model._w_tac_goal = 0.0
model._tac_goal_pos_weight = None
model._tac_goal_class_mask = None
model.train()

eps = T._synth_episodes(2, cfg.core, seed=0)
ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                 channels=cfg.core.encoder.in_channels)
batch = torch.utils.data.default_collate([ds[0], ds[1]])
batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
batch["nav_valid"] = torch.tensor([True, True])

# ⛔ The GT boxes are injected on EVERY arm, including `off`/`head`.  Injecting
# them only on the oracle arm would confound "the branch did not fire" with
# "the batch was different", and the OFF comparison would pass for the wrong
# reason.  With the gate correct, `off`/`head` must ignore them entirely.
B, N = 2, 8
g = torch.Generator().manual_seed(7)
batch["agent_box"] = torch.rand(B, N, 4, generator=g) * 20.0 + 1.0
batch["agent_yaw"] = torch.rand(B, N, generator=g) * 2.0 - 1.0
batch["agent_cls"] = torch.randint(0, 3, (B, N), generator=g)
batch["agent_valid"] = torch.ones(B, N, dtype=torch.bool)
batch["agent_valid"][1, N // 2:] = False
batch["agent_occ"] = torch.zeros(B, N)
batch["agent_rates"] = torch.zeros(B, N, 3)
batch["agent_rates_mask"] = torch.zeros(B, N, dtype=torch.bool)
batch["agent_label"] = torch.ones(B, dtype=torch.bool)

params = {k: v.detach().clone() for k, v in model.state_dict().items()}

torch.manual_seed(99)                      # identical RNG state at the call
losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
loss_vals = {k: (v.detach().clone() if torch.is_tensor(v)
                 else torch.tensor(float(v)))
             for k, v in losses.items()}
losses["loss"].backward()
grads = {k: (p.grad.detach().clone() if p.grad is not None else None)
         for k, p in model.named_parameters()}

torch.save({"params": params, "losses": loss_vals, "grads": grads,
            "arm": ARM, "stack": STACK}, OUT)
print(f"OK arm={ARM} n_params={len(params)} n_losses={len(loss_vals)} "
      f"loss={float(loss_vals['loss']):.10f} -> {OUT}")
