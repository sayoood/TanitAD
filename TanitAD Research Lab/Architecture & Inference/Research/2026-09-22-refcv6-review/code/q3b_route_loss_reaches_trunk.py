"""Q3b — under the strategic BYPASS, does the ROUTE head still push gradient
into the SHARED TRUNK?

SOURCE (positive assertion, with its control):
  * `refc_v3_train.py:3633` — `... + ROUTE_WEIGHT * loss_route + ...` with NO
    `no_strategic` guard;   `ROUTE_WEIGHT = 0.1` (`refc_train.py:79`).
  * CONTROL, same breath, ten lines below: `refc_v3_train.py:3674` — the
    STRATEGIC GOAL loss IS guarded:
    `if lan is not None and not bool(getattr(core, "no_strategic", False))`.
    So a search that found "no guard" for the route term is not a search that
    cannot find guards.

The trainer's own comment at `:3664-3667` states the reason the guard exists:
*"Supervising it anyway would push gradient through `str_goal_head` ->
`StrategicCtx` -> the SHARED ENCODER, i.e. the strategic layer would still
shape the trunk that produces the plan. That is a SECOND variable inside a
one-variable arm."* This probe asks whether that sentence is true of
`route_head` too, and answers it by MEASUREMENT rather than by reading.

⛔ MEASURED, NOT INFERRED: backprop a route cross-entropy through a model built
with `no_strategic=True` and count trunk parameters with non-zero gradient.
⭐ CONTROLS:
  (1) a NO-LOSS control — the same model, zero backward — must read 0 tensors
      with gradient, or "non-zero grad" means nothing;
  (2) a REFERENCE control — the trajectory loss — must reach the trunk, so a
      zero route reading would be about the route path, not about the trunk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_refcv6_probe_model as B            # noqa: E402

OUT = HERE.parent / "raw"
OUT.mkdir(parents=True, exist_ok=True)

TRUNK_PREFIX = "core.encoder."


def _grad_census(model, prefix: str) -> dict:
    n_t = n_nz = 0
    tot = 0.0
    for name, p in model.named_parameters():
        if not name.startswith(prefix):
            continue
        n_t += 1
        if p.grad is not None:
            g = float(p.grad.abs().sum())
            tot += g
            if g > 0:
                n_nz += 1
    return {"n_tensors": n_t, "n_with_nonzero_grad": n_nz,
            "sum_abs_grad": round(tot, 8)}


def _zero(model):
    for p in model.parameters():
        p.grad = None


def main():
    res = {}
    model, cfg = B.build(no_strategic=True)
    model.train()
    f, kw = B.batch(model, cfg, nav=1)

    # (1) NO-LOSS control
    _zero(model)
    res["control_no_backward"] = _grad_census(model, TRUNK_PREFIX)

    # (2) the ROUTE loss, exactly the trainer's term
    _zero(model)
    out = model(f, **kw)
    n_route = int(out["route_logits"].shape[-1])
    tgt = torch.zeros(out["route_logits"].shape[0], dtype=torch.long)
    loss_route = F.cross_entropy(out["route_logits"], tgt)
    (0.1 * loss_route).backward()                 # ROUTE_WEIGHT = 0.1
    res["route_loss_grad_into_trunk"] = _grad_census(model, TRUNK_PREFIX)
    res["route_loss_grad_into_route_head"] = _grad_census(model, "core.route_head.")
    res["route_loss_grad_into_strategic_gru"] = _grad_census(model, "core.strategic.")
    res["route_loss_grad_into_tac_decoder"] = _grad_census(model, "tac_decoder_v6.")

    # (3) REFERENCE control — the trajectory loss must also reach the trunk
    _zero(model)
    out2 = model(f, **kw)
    loss_traj = out2["traj"].pow(2).mean()
    loss_traj.backward()
    res["reference_control_traj_loss_grad_into_trunk"] = _grad_census(
        model, TRUNK_PREFIX)

    res["cfg_no_strategic"] = True
    res["ROUTE_WEIGHT"] = 0.1
    res["source"] = {
        "ungated_route_term": "stack/scripts/refc_v3_train.py:3633",
        "gated_gstr_term_CONTROL": "stack/scripts/refc_v3_train.py:3674",
        "route_weight": "stack/scripts/refc_train.py:79 (ROUTE_WEIGHT = 0.1)",
        "route_prior_IS_gated": "stack/tanitad/refs/refc.py:4139-4140",
    }
    p = OUT / "q3b_route_grad.json"
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"\n[banked] {p}")


if __name__ == "__main__":
    main()
