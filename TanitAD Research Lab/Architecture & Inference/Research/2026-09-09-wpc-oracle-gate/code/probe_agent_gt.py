"""MEASUREMENT: does `agent_gt` reach RefCV3Model.forward under --agents oracle?

Positive assertion on a REAL forward, three ways:
  (1) a spy on RefCV3Model.forward records the agent_gt kwarg it received;
  (2) the ORACLE seam's emitted slot boxes are compared BITWISE to the
      batch's `agent_box` (degrade_boxes at sigma=0/miss=0 returns the input
      UNCHANGED, so equality proves the tensor arrived intact);
  (3) the trainer's own total loss is finite and backwards.
Run with PYTHONPATH=<tree>/stack; the tree decides base vs patched.
"""
from __future__ import annotations
import importlib.util, json, os, sys, traceback

import torch

STACK = os.environ["TANIT_STACK"]
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(STACK, "scripts"))

_spec = importlib.util.spec_from_file_location(
    "refc_v3_train_probe", os.path.join(STACK, "scripts", "refc_v3_train.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

ARGV = ["--arm", "hier", "--out", "/tmp/probe_oracle",
        "--agents", "oracle", "--agent-queries", "8", "--seed", "0"]

res = {"stack": STACK,
       "agent_gt_token_count": open(
           os.path.join(STACK, "scripts", "refc_v3_train.py"),
           encoding="utf-8").read().count("agent_gt"),
       "agent_box_token_count_CONTROL": open(
           os.path.join(STACK, "scripts", "refc_v3_train.py"),
           encoding="utf-8").read().count("agent_box")}

try:
    args = T.build_parser().parse_args(ARGV)
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    res["cfg_agents_enable"] = bool(cfg.core.agents.enable)
    res["cfg_agents_oracle"] = bool(cfg.core.agents.oracle)
    res["cfg_oracle_sigma_range_m"] = float(cfg.core.agents.oracle_sigma_range_m)
    res["cfg_oracle_miss_rate"] = float(cfg.core.agents.oracle_miss_rate)

    torch.manual_seed(0)
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

    # --- the ORACLE's tokens: ground-truth boxes, injected into the batch ---
    B, N = 2, 8
    g = torch.Generator().manual_seed(7)
    box = torch.rand(B, N, 4, generator=g) * 20.0 + 1.0
    yaw = torch.rand(B, N, generator=g) * 2.0 - 1.0
    cls = torch.randint(0, 3, (B, N), generator=g)
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[1, N // 2:] = False
    batch["agent_box"] = box
    batch["agent_yaw"] = yaw
    batch["agent_cls"] = cls
    batch["agent_valid"] = valid

    seen = {}
    real_fwd = T.v3.RefCV3Model.forward

    def spy(self, *a, **kw):
        seen["called"] = seen.get("called", 0) + 1
        seen["agent_gt_in_kwargs"] = "agent_gt" in kw
        ag = kw.get("agent_gt")
        seen["agent_gt_is_None"] = ag is None
        if ag is not None:
            seen["agent_gt_keys"] = sorted(ag.keys())
            seen["agent_gt_box_shape"] = list(ag["box"].shape)
            seen["agent_gt_box_equals_batch"] = bool(
                torch.equal(ag["box"].cpu(), box))
            seen["agent_gt_yaw_equals_batch"] = bool(
                torch.equal(ag["yaw"].cpu(), yaw))
            seen["agent_gt_cls_equals_batch"] = bool(
                torch.equal(ag["cls"].cpu(), cls))
            seen["agent_gt_valid_equals_batch"] = bool(
                torch.equal(ag["valid"].cpu(), valid))
        out = real_fwd(self, *a, **kw)
        if "agent_slots" in out:
            seen["out_has_agent_slots"] = True
            seen["slot_box_equals_batch_BITWISE"] = bool(
                torch.equal(out["agent_slots"]["box"].detach().cpu(), box))
            seen["slot_box_shape"] = list(out["agent_slots"]["box"].shape)
        return out

    T.v3.RefCV3Model.forward = spy
    try:
        losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    finally:
        T.v3.RefCV3Model.forward = real_fwd
    res["forward_spy"] = seen
    res["loss"] = float(losses["loss"])
    res["loss_finite"] = bool(torch.isfinite(losses["loss"]))
    losses["loss"].backward()
    res["backward_ok"] = True
    res["VERDICT"] = ("GREEN: agent_gt reached the forward"
                      if seen.get("agent_gt_is_None") is False
                      and seen.get("slot_box_equals_batch_BITWISE")
                      else "RED: agent_gt did NOT reach the forward")
except BaseException as e:                                  # noqa: BLE001
    res["VERDICT"] = "RED"
    res["exc_type"] = type(e).__name__
    res["exc"] = str(e)[:600]
    res["tb_tail"] = traceback.format_exc()[-900:]

print(json.dumps(res, indent=2))
