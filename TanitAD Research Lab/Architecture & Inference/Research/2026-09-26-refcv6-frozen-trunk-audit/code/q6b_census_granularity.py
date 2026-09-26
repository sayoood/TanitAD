"""Q6b -- would the programme's OWN "every built head receives gradient" guard have caught the F3
defect?  `stack/tests/test_built_heads_receive_gradient.py::_module_grad_census` classifies
`model.named_children()` -- the TOP-LEVEL children -- as NOT_WIRED only when EVERY parameter tensor
of the child has `grad is None` (its own docstring: "a module with SOME None gradients is partially
masked, which is ordinary"). The F3 per-stage heads live INSIDE `core`, which is fed by everything.

Constructed here, on the refcv6 diffusion flags (F1-F6) and the run's own 117-anchor vocabulary,
smoke width, synthetic episodes, the trainer's own `compute_losses_v3` + one backward:
  census at the guard's granularity (the test's function, imported)  -> predicted: `core` reads
      GRADIENT_REACHES, i.e. the guard stays GREEN over the defect;
  census at LEAF granularity                                        -> predicted: the stage-0..2
      heads and AdaLN 0..2 read NOT_WIRED (grad None);
  the same leaf census with the 2-line pass-through                 -> they read GRADIENT_REACHES.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.bootstrap()
import torch  # noqa: E402

ANCH = str(C.KIT / "data/anchors/refc_anchors_6s_v0cond_alat_117.pt")
ARGV = ["--arm", "hier", "--smoke", "--device", "cpu", "--batch", "2",
        "--sampler", "ddim", "--anchors", ANCH, "--anchor-v0-conditioned", "--n-anchors", "117",
        "--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln", "--f5-focal",
        "--f5-emitting-conf", "--f6-w-u0-zero", "--out", "unused"]


def load_census_fn():
    p = C.STACK / "tests/test_built_heads_receive_gradient.py"
    spec = importlib.util.spec_from_file_location("tbhrg_for_audit", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._module_grad_census


def leaf_census(model, prefixes) -> dict:
    out = {}
    for name, mod in model.named_modules():
        if not any(name.startswith(pf) for pf in prefixes):
            continue
        ps = [p for p in mod.parameters(recurse=False) if p.requires_grad]
        if not ps:
            continue
        n_none = sum(1 for p in ps if p.grad is None)
        g = sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None)
        out[name] = ("NOT_WIRED" if n_none == len(ps) else "ZERO_GRAD" if g == 0.0
                     else "GRADIENT_REACHES")
    return out


def build_and_backward(passthrough: bool):
    T = C.trainer_module()
    from tanitad.refs import refc
    args = T.build_parser().parse_args(ARGV)
    art = T._read_anchor_artifact(args)
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    model = T.v3.RefCV3Model(cfg)
    model.core.decoder.load_anchors(art.anchors, art.controls)
    model._w_goal_point = 0.0
    model._w_tac_goal = 0.0
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None
    model.train()
    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    orig = refc.RefCModel.forward

    def fwd(self, *a, **k):
        o = orig(self, *a, **k)
        dec = self.decoder
        if getattr(dec, "cascade", None) is not None and getattr(dec, "_rv6_layer_u0", None):
            o["layer_u0_hat"] = dec._rv6_layer_u0
            o["layer_logits"] = dec._rv6_layer_conf
        return o
    if passthrough:
        refc.RefCModel.forward = fwd
    try:
        losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
        losses["loss"].backward()
    finally:
        refc.RefCModel.forward = orig
    return model, losses


def main():
    C.ram_guard("q6b_census_granularity (light job; the brief's 8 GB floor applies to every job)")
    census = load_census_fn()
    res = {"what": "the gradient-reach guard's granularity vs the F3 defect",
           "evidence_class": "MEASURED (ours, CPU, smoke width, the run's 117-anchor file)"}
    for arm, pt in (("as_shipped", False), ("passthrough_fix", True)):
        model, losses = build_and_backward(pt)
        top = census(model)
        res[arm] = {"cascade_in_losses": "cascade" in losses,
                    "guard_granularity_named_children": {k: v["verdict"] for k, v in top.items()},
                    "leaf": leaf_census(model, ("core.decoder.cascade", "core.decoder.adaln",
                                                "core.decoder.control_head",
                                                "core.decoder.offset_head"))}
        del model
    a, b = res["as_shipped"], res["passthrough_fix"]
    res["verdict"] = {
        "guard_green_over_defect": a["guard_granularity_named_children"].get("core") == "GRADIENT_REACHES",
        "leaf_stage0_not_wired_as_shipped": a["leaf"].get("core.decoder.cascade.control_heads.0") == "NOT_WIRED",
        "leaf_stage0_reached_with_fix": b["leaf"].get("core.decoder.cascade.control_heads.0") == "GRADIENT_REACHES"}
    print(json.dumps(res, indent=1))
    C.write_json("q6b_census_granularity.json", res)


if __name__ == "__main__":
    main()
