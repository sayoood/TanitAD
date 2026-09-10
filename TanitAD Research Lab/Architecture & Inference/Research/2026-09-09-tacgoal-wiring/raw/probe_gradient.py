"""MEASURE: does a gradient reach `tac_goal_tok_head`, and is OFF bit-identical?

Run as:  python probe_gradient.py <tree>/stack <label>

Prints one JSON blob. ⛔ Every expectation lives in the CALLER (the RESULT
tables and the pytest guard), never here -- this file only MEASURES.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

# ⚠️ Import-safe: this module is also imported by `probe_corrupt.py` for its
# LIVE_ARGV and census, and a module-level `sys.argv[1]` would crash there.
STACK = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else ""
LABEL = sys.argv[2] if len(sys.argv) > 2 else ""
SCRIPTS = os.path.join(STACK, "scripts") if STACK else ""
for p in (STACK, SCRIPTS):
    if p and p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

TRAINER_PY = os.path.join(SCRIPTS, "refc_v3_train.py")


def trainer():
    spec = importlib.util.spec_from_file_location("refc_v3_train_probe",
                                                  TRAINER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: ⛔ THE RECORDED argv OF THE LIVE refcv5-v2 ARM -- copied verbatim from
#: stack/tests/test_built_heads_receive_gradient.py::LIVE_ARGV, which took it
#: from the arm's own config.json. Data paths are pod paths and never opened.
LIVE_ARGV = [
    "--arm", "hier", "--size", "base",
    "--v2-cache", "/root/data/train",
    "--v7-labels", "/workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz",
    "--image-hw", "256", "640", "--steps", "40284", "--batch", "20",
    "--lr", "1e-4", "--warmup", "2000", "--seed", "0", "--u8-batches",
    "--out", "/workspace/experiments/refcv5-v2-noagents-b1-v72-40k",
    "--nav-from-v7", "--ego-state-inject", "--ego-dropout", "0.5",
    "--n-anchors", "117", "--anchor-v0-conditioned",
    "--anchor-control-units", "alat", "--sel-accel-max", "2.0",
    "--sampler", "ddim", "--w-u0", "0.5", "--sel-refined",
    "--sel-score-emitted", "--goal-str", "--tac-goal-tok-head",
    "--agents", "off",
]


def module_grad_census(model):
    out = {}
    for name, mod in model.named_children():
        ps = [p for p in mod.parameters() if p.requires_grad]
        if not ps:
            continue
        n_none = sum(1 for p in ps if p.grad is None)
        gsum = sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None)
        out[name] = {
            "n_tensors": len(ps),
            "n_params": sum(p.numel() for p in ps),
            "n_grad_none": n_none,
            "grad_abs_sum": gsum,
            "verdict": ("NOT_WIRED" if n_none == len(ps)
                        else "ZERO_GRAD" if gsum == 0.0
                        else "GRADIENT_REACHES"),
        }
    return out


def build_and_backward(T, extra_argv, *, corrupt=None):
    """Build the arm, run the TRAINER'S OWN loss, one backward.

    ``corrupt`` is the DELIBERATE-REGRESSION hook: a callable that mutates the
    model AFTER construction and BEFORE the forward. A comparison that cannot
    be made to fail is not evidence of identity.
    """
    args = T.build_parser().parse_args(LIVE_ARGV + list(extra_argv))
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    model = T.v3.RefCV3Model(cfg)
    model._w_goal_point = float(getattr(args, "goal_point_w", 0.0) or 0.0)
    model._w_tac_goal = float(getattr(args, "w_tac_goal", 0.0) or 0.0)
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None
    if corrupt is not None:
        corrupt(model)
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
    # The goal-set target, injected the same way lat_v7/lon_v7 are: the smoke
    # dataset has no v7 join. Row 0 carries evidence on the first 3 tokens,
    # row 1 is entirely ignored -- so `n_supervised` has a LITERAL expectation.
    K = len(v7l.TAC_GOAL_TOKENS)
    y = torch.zeros(2, K)
    w = torch.zeros(2, K)
    y[0, 0] = 1.0
    w[0, 0] = 1.0
    w[0, 1] = 1.0
    w[0, 2] = 1.0
    batch["tac_goal_y"] = y
    batch["tac_goal_w"] = w

    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    total = losses["loss"]
    assert total.requires_grad, "UNPOWERED: total loss does not require grad"
    total.backward()
    return model, module_grad_census(model), losses


def param_fingerprint(model):
    """A seed-stable fingerprint of every SHARED parameter, by name."""
    return {n: float(p.detach().double().abs().sum())
            for n, p in sorted(model.named_parameters())}


def run():
    T = trainer()
    res = {"label": LABEL, "stack": STACK, "arms": {}}

    for arm, extra in (("OFF", []), ("ON", ["--w-tac-goal", "1.0"])):
        try:
            model, census, losses = build_and_backward(T, extra)
        except SystemExit as e:
            res["arms"][arm] = {"REFUSED": str(e)}
            continue
        except Exception as e:                       # noqa: BLE001
            res["arms"][arm] = {"ERROR": f"{type(e).__name__}: {e}"}
            continue
        tg = census.get("tac_goal_tok_head")
        res["arms"][arm] = {
            "tac_goal_tok_head": tg,
            "loss_total": float(losses["loss"].detach()),
            "tac_goal_term": (float(losses["tac_goal"].detach())
                              if "tac_goal" in losses else None),
            "tac_goal_n_supervised": (float(losses["tac_goal_n_supervised"])
                                      if "tac_goal_n_supervised" in losses
                                      else None),
            "fingerprint": param_fingerprint(model),
            "n_children_wired": sum(
                1 for v in census.values()
                if v["verdict"] == "GRADIENT_REACHES"),
        }
    print(json.dumps(res))


if __name__ == "__main__":
    run()
