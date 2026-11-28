"""F3 / F4 re-measured after removing a ZERO-INIT bottleneck.

⚠️ WHY THIS FILE EXISTS. `diag_f1_f9_liveness.py`'s first F3/F4 arms perturbed
`cascade.control_heads[i]` / `adaln[i]` and read a trajectory delta of EXACTLY
0.0 — which reads like a dead wire and is not one. `control_head` and every
`CascadeHeads.control_head` are **zero-init** (`refc.py:1705-1706`,
`refcv6_diffusion.py:304-305`), so at construction ``du = 0`` and
``u0_hat = x_n``: the emitted trajectory is the anchored Gaussian and is
INDEPENDENT OF EVERY DECODER WEIGHT. Any liveness probe that mutates an
upstream module and watches `traj` therefore reads 0.0 for a perfectly wired
model. ⭐ This is the advisory's class-F shape inverted — an assertion that
*cannot come out the other way* is not a control.

The fix is to give the emitting head a non-zero weight first (the state any
trained model is in after step 1) and re-run the SAME mutation. Both arms are
reported so the artifact is visible rather than hidden.

Also measured here: F3's DETACH (DD `transfuser_model_v2.py:379`) — with the
heads un-zeroed, a backward from the LAST stage must put gradient on the last
layer's attention and NONE on the first layer's.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from diag_f1_f9_liveness import build, fwd, inputs, tele
from tanitad.models import refcv6_diffusion as rv6


def unzero_(dec, scale=0.05, seed=5):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if dec.control_head is not None:
            dec.control_head.weight.copy_(
                torch.randn(dec.control_head.weight.shape, generator=g) * scale)
        if dec.cascade is not None:
            for h in dec.cascade.control_heads:
                h.weight.copy_(torch.randn(h.weight.shape, generator=g) * scale)


def main():
    out = {}

    # ------------------------------------------------------------------ F3
    f3 = {}
    dec = build(rv6.DiffusionFlags(f3_per_layer=True))
    f3["ZEROINIT_du_is_zero"] = float(
        fwd(dec, seed=3)["u0_hat"].abs().sum()
        - fwd(dec, seed=3)["anchor_bank"].abs().sum()) == 0.0
    unzero_(dec)
    base = fwd(dec, seed=3)["traj"].clone()
    base_layers = [t.clone() for t in fwd(dec, seed=3)["layer_u0_hat"]]
    for i in range(len(dec.cascade.control_heads)):
        with torch.no_grad():
            dec.cascade.control_heads[i].bias.add_(1.0)
        o = fwd(dec, seed=3)
        f3[f"mutate_stage{i}__traj_delta"] = float((o["traj"] - base).abs().max())
        f3[f"mutate_stage{i}__layer_u0_deltas"] = [
            round(float((a - b).abs().max()), 6)
            for a, b in zip(o["layer_u0_hat"], base_layers)]
        with torch.no_grad():
            dec.cascade.control_heads[i].bias.sub_(1.0)

    # DETACH, with non-zero heads so gradient can actually flow
    d2 = build(rv6.DiffusionFlags(f3_per_layer=True))
    unzero_(d2)
    fmap, m, v = inputs()
    d2.train(False)
    o = d2(fmap, m, steps=2, v_ms=v)
    o["layer_u0_hat"][-1].sum().backward()
    g = {}
    for i, ly in enumerate(d2.layers):
        w = ly.cross.out_proj.weight.grad
        g[f"layer{i}.cross.out_proj"] = None if w is None else float(w.abs().sum())
    for i, h in enumerate(d2.cascade.control_heads):
        w = h.weight.grad
        g[f"cascade.control_heads[{i}]"] = (None if w is None
                                            else float(w.abs().sum()))
    f3["grad_from_LAST_stage"] = g
    f3["detach_holds"] = bool(
        (g.get("layer0.cross.out_proj") in (None, 0.0))
        and (g.get(f"layer{len(d2.layers)-1}.cross.out_proj") or 0.0) > 0.0)

    # CONTROL that must read a KNOWN value: with the cascade OFF the same
    # backward must reach layer0 (no detach exists to cut it).
    d3 = build(rv6.DiffusionFlags(f2_dd_step=True))
    unzero_(d3)
    d3.train(False)
    o3 = d3(fmap, m, steps=2, v_ms=v)
    o3["u0_hat"].sum().backward()
    f3["control_no_cascade_layer0_grad"] = (
        None if d3.layers[0].cross.out_proj.weight.grad is None
        else float(d3.layers[0].cross.out_proj.weight.grad.abs().sum()))
    out["F3"] = f3

    # ------------------------------------------------------------------ F4
    f4 = {}
    dec4 = build(rv6.DiffusionFlags(f4_adaln=True))
    unzero_(dec4)
    base4 = fwd(dec4, seed=3)["traj"].clone()
    for i in range(len(dec4.adaln)):
        with torch.no_grad():
            dec4.adaln[i].scale_shift_mlp[-1].bias.add_(0.5)
        f4[f"mutate_adaln{i}__traj_delta"] = float(
            (fwd(dec4, seed=3)["traj"] - base4).abs().max())
        with torch.no_grad():
            dec4.adaln[i].scale_shift_mlp[-1].bias.sub_(0.5)
    # zero-init variant: identity at construction (DD's own value is False)
    z = build(rv6.DiffusionFlags(f4_adaln=True, f4_zero_init=True))
    unzero_(z)
    n = build(None)
    unzero_(n)
    f4["zeroinit_traj_equals_no_f4"] = float(
        (fwd(z, seed=3)["traj"] - fwd(n, seed=3)["traj"]).abs().max())
    dflt = build(rv6.DiffusionFlags(f4_adaln=True))
    unzero_(dflt)
    f4["default_init_traj_differs_from_no_f4"] = float(
        (fwd(dflt, seed=3)["traj"] - fwd(n, seed=3)["traj"]).abs().max())
    f4["DD_zero_init_default"] = bool(rv6.DiffusionFlags().f4_zero_init)
    out["F4"] = f4

    # ------------------------------------------------------- F8 clamp probe
    f8 = {}
    on = build(rv6.DiffusionFlags(f8_flat_waypoint_noise=True), space="metre")
    unzero_(on, scale=0.5)
    # instrument the schedule step to see whether x_n leaves [-1, 1]
    seen = {"max_abs_after_step": 0.0, "n_steps": 0}
    orig = on.sched.step

    def patched(x0_hat, x_t, t, t_prev):
        r = orig(x0_hat, x_t, t, t_prev)
        seen["max_abs_after_step"] = max(seen["max_abs_after_step"],
                                         float(r.abs().max()))
        seen["n_steps"] += 1
        return r
    on.sched.step = patched
    fwd(on, seed=4)
    on.sched.step = orig
    f8["max_abs_x_n_after_sched_step"] = round(seen["max_abs_after_step"], 5)
    f8["n_sched_steps"] = seen["n_steps"]
    f8["exceeds_DD_clamp_box"] = bool(seen["max_abs_after_step"] > 1.0)
    out["F8_clamp"] = f8

    txt = json.dumps(out, indent=2, default=str)
    print(txt)
    (Path(__file__).resolve().parents[1] / "raw"
     / "f3_f4_zeroinit.json").write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
