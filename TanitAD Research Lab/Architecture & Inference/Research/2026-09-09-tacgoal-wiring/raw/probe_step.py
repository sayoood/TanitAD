"""⭐ THE PARAMETER COMPARISON, MADE DISCRIMINATING.

MEASURED first, and reported: comparing parameters straight after construction
reads **0 differing for BOTH arms**, because no optimizer step has run -- the
values are seed-determined and the loss cannot have moved them yet. That
comparison proves the model CONSTRUCTION is identical (no RNG drift, no shape
change) and is **structurally blind** to whether the loss changed. Presenting it
as proof of the ON case would be the "check that shares the defect it checks
for" family.

So this probe takes ONE optimizer step and compares AFTER it. Now:
  * OFF vs HEAD  must be exactly equal over all 167 shared tensors;
  * ON  vs HEAD  must DIFFER -- and if it does not, the comparison is blind and
    the OFF result proves nothing.

Run:  python probe_step.py <tree>/stack <label>
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

STACK = os.path.abspath(sys.argv[1])
LABEL = sys.argv[2]
SCRIPTS = os.path.join(STACK, "scripts")
for p in (STACK, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_gradient import LIVE_ARGV  # noqa: E402


def trainer():
    spec = importlib.util.spec_from_file_location(
        "refc_v3_train_step", os.path.join(SCRIPTS, "refc_v3_train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(T, extra_argv):
    args = T.build_parser().parse_args(LIVE_ARGV + list(extra_argv))
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    model = T.v3.RefCV3Model(cfg)
    model._w_goal_point = float(getattr(args, "goal_point_w", 0.0) or 0.0)
    model._w_tac_goal = float(getattr(args, "w_tac_goal", 0.0) or 0.0)
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None
    model.train()

    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    v7l = T.v7l
    n_lon = len(v7l.HEADS["tac_lon"])
    batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
    batch["lon_v7"] = torch.tensor([n_lon - 1, v7l.IGNORE_ID], dtype=torch.long)
    batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
    batch["nav_valid"] = torch.tensor([True, True])
    K = len(v7l.TAC_GOAL_TOKENS)
    y = torch.zeros(2, K)
    w = torch.zeros(2, K)
    y[0, 0] = 1.0
    w[0, 0] = 1.0
    w[0, 1] = 1.0
    w[0, 2] = 1.0
    batch["tac_goal_y"] = y
    batch["tac_goal_w"] = w

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt.zero_grad(set_to_none=True)
    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    losses["loss"].backward()
    opt.step()                       # <-- the step the naive probe was missing
    return {
        "loss_total": float(losses["loss"].detach()),
        "post_step": {n: float(p.detach().double().abs().sum())
                      for n, p in sorted(model.named_parameters())},
    }


def main():
    T = trainer()
    out = {"label": LABEL, "arms": {}}
    for arm, extra in (("OFF", []), ("ON", ["--w-tac-goal", "1.0"])):
        try:
            out["arms"][arm] = run(T, extra)
        except SystemExit as e:
            out["arms"][arm] = {"REFUSED": str(e)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
