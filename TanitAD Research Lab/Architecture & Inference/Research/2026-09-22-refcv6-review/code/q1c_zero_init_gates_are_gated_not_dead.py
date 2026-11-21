"""Q1c — the two SELECTION gates are ZERO-INIT. Gated, or DEAD?

SPEC §5 puts nav into selection *"via a parameter-free nav-compliance term
behind a zero-init gate"*, and §4 puts the valid-behaviour set into selection
through another zero-init graft. MEASURED in Q1: at step 0 both contribute
EXACTLY 0.0 to the score. That is by design — but only if the gate can LEAVE
zero, and a zero-init parameter that receives no gradient never does.

⛔ The programme has measured this exact family: *"42 of 138 tensors with a
declared budget and no gradient"* (2026-09-06). `refc.py:1804`'s own comment
asserts *"the gradient (compliance * dL/dscore) is non-zero — gated, not
dead"*. This probe tests that assertion instead of quoting it.

⭐ CONTROLS: (a) a no-backward census must read 0, or "non-zero" means nothing;
(b) a selection parameter known to be live must read non-zero in the same
backward; (c) the nav-compliance gate's gradient must VANISH when nav is not
supplied (`nav_cmd=None` -> `nav_cmd_sel=None`), which is the discriminating
half — a gradient that is non-zero either way is not evidence the nav term is
what feeds it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_refcv6_probe_model as B            # noqa: E402

OUT = HERE.parent / "raw"
OUT.mkdir(parents=True, exist_ok=True)


def _g(model, name):
    for n, p in model.named_parameters():
        if n == name:
            return (None if p.grad is None
                    else round(float(p.grad.abs().sum()), 12))
    return "NO SUCH PARAMETER"


def main():
    res = {}
    # ⛔ BOTH §4 selection seams must be BUILT or this probe measures a dead
    # graft that was never wired — `graft_behaviour_sel` defaults to False and
    # the instrument's first run read 'no gradient' for exactly that reason.
    model, cfg = B.build(tac8_prior=True, behaviour_sel=True)
    model.train()
    names = ["core.decoder.navc_gate", "tac_behaviour_gate_v6.proj.weight",
             "core.decoder.tac8_lat_to_anchor.weight",
             "core.decoder.tac8_lon_to_anchor.weight"]
    live_control = "core.decoder.conf_head.weight"
    res["parameters_found"] = {n: _g(model, n) for n in names}

    def _run(nav, tag):
        for p in model.parameters():
            p.grad = None
        res[f"{tag}_control_no_backward"] = {n: _g(model, n) for n in names}
        f, kw = B.batch(model, cfg, nav=nav)
        out = model(f, **kw)
        # a selection-shaped loss: push the ranked score around
        loss = out["sel_score_v3"].pow(2).mean()
        loss.backward()
        res[tag] = {n: _g(model, n) for n in names}
        res[tag]["CONTROL_live_selection_param"] = _g(model, live_control)
        res[tag]["navc_gate_value"] = round(
            float(model.core.decoder.navc_gate.detach()), 8)

    _run(1, "nav_SUPPLIED")
    _run(None, "nav_NONE_discriminating_control")

    res["VERDICT"] = {
        "navc_gate_is_gated_not_dead": (
            isinstance(res["nav_SUPPLIED"].get("core.decoder.navc_gate"), float)
            and res["nav_SUPPLIED"]["core.decoder.navc_gate"] > 0),
        "navc_gate_gradient_vanishes_without_nav": (
            res["nav_NONE_discriminating_control"].get("core.decoder.navc_gate")
            in (0.0, None)),
        "behaviour_gate_is_gated_not_dead": (
            isinstance(res["nav_SUPPLIED"].get("tac_behaviour_gate_v6.proj.weight"),
                       float)
            and res["nav_SUPPLIED"]["tac_behaviour_gate_v6.proj.weight"] > 0),
        "tac8_grafts_are_gated_not_dead": all(
            isinstance(res["nav_SUPPLIED"].get(k), float)
            and res["nav_SUPPLIED"][k] > 0
            for k in ("core.decoder.tac8_lat_to_anchor.weight",
                      "core.decoder.tac8_lon_to_anchor.weight")),
    }
    p = OUT / "q1c_zero_init_gates.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    print(f"\n[banked] {p}")


if __name__ == "__main__":
    main()
