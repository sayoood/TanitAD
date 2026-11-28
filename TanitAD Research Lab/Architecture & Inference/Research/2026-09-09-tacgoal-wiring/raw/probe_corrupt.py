"""⭐ THE DISCRIMINATING CONTROL. A comparison that cannot be made to FAIL is
not evidence of identity.

WP-B's removability proof was GREEN FOR A REASON UNRELATED TO WP-B: a zero-init
gate multiplied its whole branch away, so the proof passed *with the head
deliberately corrupted*. This file corrupts THIS term, deliberately, two
independent ways, and reports what each one moves:

  C1  the HEAD'S PARAMETERS are perturbed.
      OFF total must NOT move  (the head is genuinely outside the OFF graph)
      ON  total MUST move      (the head is genuinely inside the ON graph)

  C2  `tac_goal_loss` itself is replaced by a scaled version.
      OFF total must NOT move  (the TERM is genuinely absent at w = 0)
      ON  total MUST move      (the comparison is sensitive to the term)

⛔ If C1/C2 leave the ON total unmoved, the ON measurement is uninformative and
the OFF identity proves nothing -- exactly the WP-B failure.

Run:  python probe_corrupt.py <tree>/stack
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

STACK = os.path.abspath(sys.argv[1])
SCRIPTS = os.path.join(STACK, "scripts")
for p in (STACK, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_gradient import LIVE_ARGV, module_grad_census  # noqa: E402


def trainer():
    spec = importlib.util.spec_from_file_location(
        "refc_v3_train_corrupt", os.path.join(SCRIPTS, "refc_v3_train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_arm(T, extra_argv, *, corrupt_params=False, scale_loss=None):
    args = T.build_parser().parse_args(LIVE_ARGV + list(extra_argv))
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    model = T.v3.RefCV3Model(cfg)
    model._w_goal_point = float(getattr(args, "goal_point_w", 0.0) or 0.0)
    model._w_tac_goal = float(getattr(args, "w_tac_goal", 0.0) or 0.0)
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None

    # ---- C1: perturb the head's parameters ------------------------------
    if corrupt_params:
        with torch.no_grad():
            for p in model.tac_goal_tok_head.parameters():
                p.add_(7.5)          # large, deterministic, unmistakable

    # ---- C2: scale the loss function itself ------------------------------
    restore = None
    if scale_loss is not None:
        tgh = T._tac_goal_head
        orig = tgh.tac_goal_loss

        def scaled(*a, **k):
            loss, n = orig(*a, **k)
            return loss * scale_loss, n

        tgh.tac_goal_loss = scaled
        restore = (tgh, orig)

    try:
        model.train()
        eps = T._synth_episodes(2, cfg.core, seed=0)
        ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                         channels=cfg.core.encoder.in_channels)
        batch = torch.utils.data.default_collate([ds[0], ds[1]])
        v7l = T.v7l
        n_lon = len(v7l.HEADS["tac_lon"])
        batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
        batch["lon_v7"] = torch.tensor([n_lon - 1, v7l.IGNORE_ID],
                                       dtype=torch.long)
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
        losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
        losses["loss"].backward()
        cen = module_grad_census(model)["tac_goal_tok_head"]
        return {"loss_total": float(losses["loss"].detach()),
                "tac_goal_term": (float(losses["tac_goal"].detach())
                                  if "tac_goal" in losses else None),
                "grad_abs_sum": cen["grad_abs_sum"],
                "verdict": cen["verdict"]}
    finally:
        if restore is not None:
            restore[0].tac_goal_loss = restore[1]


def main():
    T = trainer()
    out = {}
    for name, kw in (
            ("clean",           {}),
            ("C1_params",       {"corrupt_params": True}),
            ("C2_loss_x1000",   {"scale_loss": 1000.0}),
    ):
        for arm, extra in (("OFF", []), ("ON", ["--w-tac-goal", "1.0"])):
            out[f"{name}/{arm}"] = run_arm(T, extra, **kw)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
