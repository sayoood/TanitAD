"""Which EXACT-ZERO gradients are GATED (recoverable) and which are DEAD?

The 139-clip forward reads ``grad_abs_sum`` EXACTLY 0.0 on three modules with
every parameter holding a real (all-zero) gradient tensor. That is the
``tac_goal_tok_head`` signature -- 11,286 params at exactly 0 for 40,284 steps.
A zero is NOT automatically a defect: a zero-init GATE reads zero at step 0 and
then opens. So the zeros must be SEPARATED, and only a mutation can do it:

  GATED  — one side of the product is zero, the other is not, so the zero side
           receives ``dL/dgate = other_side != 0`` and can leave zero.
  DEAD   — BOTH sides of a product are exactly zero, so each one's gradient is
           proportional to the other and NEITHER can ever move. Self-sustaining.

⛔ PROVEN BY MUTATION, not by reading the source. Each arm breaks exactly ONE
zero and the OTHER side's gradient is read back. A control arm keeps both zero
and must read exactly 0.0, or the probe cannot see what it is cited for.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e2e_1024 import (build_args, build_dataset, build_model, cache_dir,  # noqa: E402
                      load_trainer, one_window_per_episode, trainer_argv)

import torch  # noqa: E402

#: ⛔ PARAMETER granularity, not module granularity, and that is load-bearing:
#: `ego_to_cond` is `Linear(32, d)` whose BIAS receives a real gradient (it is
#: an unconditional offset on the condition) while its WEIGHT receives exactly
#: zero. Reading the MODULE would sum the two, report non-zero, and hide the
#: dead half -- the arithmetic equivalent of pooling a per-class number.
PAIRS = {
    # name: (side A, side B) -- the two factors of the suspected product
    "ego_history": ("core.ego_hist.out.weight",
                    "core.decoder.ego_to_cond.weight"),
    "target_latent": ("tac_latent_proj.weight",
                      "core.decoder.tgt_film.to_scale_shift.weight"),
    "goal_scorer": ("scorer", "goal_gate"),
}
WATCH = ["core.ego_hist", "core.ego_hist.out.weight", "core.ego_hist.out.bias",
         "core.decoder.ego_to_cond", "core.decoder.ego_to_cond.weight",
         "core.decoder.ego_to_cond.bias",
         "tac_latent_proj", "tac_latent_proj.weight",
         "core.decoder.tgt_proj.weight",
         "core.decoder.tgt_film.to_scale_shift.weight",
         "core.decoder.tgt_film.to_scale_shift.bias",
         "scorer", "goal_gate", "core.route_head", "core.strategic",
         "core.encoder", "core.decoder"]


def grads(model, names):
    out = {}
    for nm in names:
        try:
            mod = model.get_submodule(nm)
            ps = list(mod.parameters(recurse=True))
        except (AttributeError, TypeError):
            p = dict(model.named_parameters()).get(nm)
            ps = [p] if p is not None else []
        if not ps:
            out[nm] = None
            continue
        out[nm] = {
            "n_params": int(sum(p.numel() for p in ps)),
            "grad_abs_sum": float(sum(float(p.grad.detach().abs().sum())
                                      for p in ps if p.grad is not None)),
            "n_grad_none": int(sum(1 for p in ps if p.grad is None)),
            "weight_abs_sum": float(sum(float(p.detach().abs().sum())
                                        for p in ps)),
        }
    return out


def one_step(trainer, model, batch):
    model.zero_grad(set_to_none=True)
    losses = trainer.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    key = "loss" if "loss" in losses else "total"
    losses[key].backward()
    return float(losses[key].detach()), grads(model, WATCH)


def break_zero(model, name, scale=0.05):
    """Give ONE side of a product real weights. Returns a restore closure."""
    try:
        mod = model.get_submodule(name)
        ps = list(mod.parameters(recurse=True))
    except (AttributeError, TypeError):
        ps = [dict(model.named_parameters())[name]]
    saved = [p.detach().clone() for p in ps]
    with torch.no_grad():
        for p in ps:
            p.copy_(torch.randn_like(p) * scale)

    def restore():
        with torch.no_grad():
            for p, s in zip(ps, saved):
                p.copy_(s)
    return restore


def main():
    pkg = HERE.parent
    raw = pkg / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    trainer = load_trainer()
    argv = trainer_argv(os.environ.get(
        "E2E_CACHE",
        "D:/Projects/TanitAD-artifacts/v2ep-eval139-256x1024cyl"),
        trunk_name="resnet101.a1_in1k", out=str(pkg / "_runout"),
        v7_labels=os.environ.get("E2E_V7"),
        anchors=os.environ.get("E2E_ANCHORS"))
    _ap, args = build_args(trainer, argv)
    cfg, model, _art, _ = build_model(trainer, args)
    eps, ds, _par, join = build_dataset(trainer, cfg, args)
    e_i, d_i = one_window_per_episode(ds, 1)[0]
    batch = torch.utils.data.default_collate([ds[d_i]])

    res = {"evidence_class": "MEASURED", "v7_join": join,
           "image_hw": list(cfg.core.encoder.image_hw()), "arms": {}}

    t0 = time.time()
    loss0, g0 = one_step(trainer, model, batch)
    res["arms"]["CONTROL_as_shipped"] = {"loss": loss0, "grads": g0}
    print("[forensics] control loss=%.4f (%.0fs)" % (loss0, time.time() - t0),
          flush=True)

    for pair, (a, b) in PAIRS.items():
        for side in (a, b):
            try:
                restore = break_zero(model, side)
            except Exception as exc:
                res["arms"]["MUT_%s" % side] = {"error": str(exc)}
                continue
            t = time.time()
            loss, g = one_step(trainer, model, batch)
            restore()
            res["arms"]["MUT_%s" % side] = {
                "pair": pair, "broken": side, "loss": loss, "grads": g}
            print("[forensics] MUT %-42s loss=%.4f (%.0fs)"
                  % (side, loss, time.time() - t), flush=True)

    # ---- the verdict, computed from the arms, not asserted -----------------
    verdicts = {}
    for pair, (a, b) in PAIRS.items():
        def gsum(arm, nm):
            d = res["arms"].get(arm, {}).get("grads", {}).get(nm)
            return None if d is None else d["grad_abs_sum"]
        base_a, base_b = gsum("CONTROL_as_shipped", a), gsum("CONTROL_as_shipped", b)
        a_when_b_broken = gsum("MUT_%s" % b, a)
        b_when_a_broken = gsum("MUT_%s" % a, b)
        both_zero_at_init = (base_a == 0.0 and base_b == 0.0)
        verdicts[pair] = {
            "sides": [a, b],
            "grad_a_as_shipped": base_a, "grad_b_as_shipped": base_b,
            "grad_a_when_b_broken": a_when_b_broken,
            "grad_b_when_a_broken": b_when_a_broken,
            "both_zero_as_shipped": both_zero_at_init,
            # DEAD == both sides read exactly 0 as shipped AND each recovers
            # only when the OTHER is broken: a mutual-zero product.
            # DEAD  == both factors read exactly 0 as shipped AND each one
            #          recovers ONLY when the other is broken: a mutual zero,
            #          self-sustaining, no gradient can ever start it.
            # GATED == exactly one factor is zero; the OTHER one is the gate
            #          and it has a gradient, so it can open and the zero side
            #          then learns. A zero at step 0 here is NOT a defect.
            "verdict": ("DEAD_ZERO_PRODUCT" if (
                both_zero_at_init
                and (a_when_b_broken or 0) > 0 and (b_when_a_broken or 0) > 0)
                else "DEAD_BUT_MUTATION_DID_NOT_RECOVER" if both_zero_at_init
                else "GATED_RECOVERABLE" if (base_a == 0.0 or base_b == 0.0)
                else "LIVE"),
            "gate_side": (b if base_a == 0.0 and (base_b or 0) > 0 else
                          a if base_b == 0.0 and (base_a or 0) > 0 else None),
        }
    res["verdicts"] = verdicts
    (raw / "zero_grad_forensics.json").write_text(json.dumps(res, indent=1),
                                                  encoding="utf-8")
    print(json.dumps(verdicts, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
