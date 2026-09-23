"""Q3 — with the strategic layer "deactivated", are the heads actually NOT BUILT?

SPEC_REFCV6_V2 §1, verbatim: *"Deactivated for this experiment: the whole
strategic layer — route head, `g_str`, strategic GRU. No head estimates the
route (PI). The flags remain but default OFF and **the heads are not built**."*

⛔ THE QUESTION IS ABOUT PARAMETERS, NOT ABOUT A CONFIG FLAG. A head that
EXISTS but is unsupervised is a different failure from one that does not exist:
it takes checkpoint space, it appears in `model.parameters()` and therefore in
the optimiser, it can still receive gradient from any consumer that was not
found, and — the case this programme has measured before — it can be reported
as "the strategic layer is off" while its output is still computed and cached.

So this probe reads `state_dict()` keys and `named_parameters()`, never the
config, and it runs BOTH arms (`--no-strategic` off and on) so the answer is a
DIFFERENCE, not an assertion. It also runs a forward in each arm and asks
whether `g_str` / `route_logits` still come out.

⭐ CONTROL IN THE SAME BREATH: a key set that reads empty because the probe's
prefix was wrong is indistinguishable from a head that was not built. Every
"absent" row below is paired with the TOTAL key count and with at least one
prefix that must read non-zero.
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

#: The three things SPEC §1 names, and the parameter prefixes that would hold
#: them. Derived by reading the constructors, then CHECKED against a live
#: state_dict (a prefix that matches nothing in BOTH arms is reported as
#: UNRESOLVED, never as "not built").
TARGETS = {
    "route_head (refc.py:3575)": ["core.route_head."],
    "strategic GRU (refc.py:3419 StrategicCtx)": ["core.strategic."],
    "g_str head (refc_v3.py:943 str_goal_head)": ["str_goal_head."],
    "g_str -> tactical FiLM (refc_v3.py:946-950)": ["gstr_embed.", "gstr_film."],
    "ctx -> operative cond (refc.py:1737 ctx_to_cond)": [
        "core.decoder.ctx_to_cond."],
}
#: ⭐ the same-breath control: these MUST be non-empty in both arms, or the
#: probe is reading the wrong object and every absence above is meaningless.
CONTROLS = ["core.decoder.layers.", "tac_decoder_v6."]


def _scan(model) -> dict:
    sd = model.state_dict()
    keys = list(sd)
    def hit(prefixes):
        ks = [k for k in keys if any(k.startswith(p) for p in prefixes)]
        return {"n_keys": len(ks),
                "n_params": int(sum(sd[k].numel() for k in ks)),
                "keys": ks[:8]}
    return {
        "total_state_dict_keys": len(keys),
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "targets": {name: hit(pfx) for name, pfx in TARGETS.items()},
        "controls": {c: hit([c]) for c in CONTROLS},
    }


def main():
    res = {}
    for arm, no_strat in (("strategic_flag_DEFAULT", False),
                          ("no_strategic_TRUE", True)):
        model, cfg = B.build(no_strategic=no_strat)
        scan = _scan(model)
        f, kw = B.batch(model, cfg, nav=1)
        with torch.no_grad():
            out = model(f, **kw)
        scan["forward_emits"] = {
            k: (list(out[k].shape) if torch.is_tensor(out.get(k)) else None)
            for k in ("g_str", "g_str_raw", "route_logits", "g_tac", "z_tac")}
        # ⭐ does the route head's output still MOVE with the input? A head
        # that is built AND live is a third state, distinct from built-and-dead.
        f2, kw2 = B.batch(model, cfg, nav=1, seed=7)
        with torch.no_grad():
            out2 = model(f2, **kw2)
        scan["forward_moves_with_input"] = {
            k: (round(float((out[k] - out2[k]).abs().max()), 8)
                if torch.is_tensor(out.get(k)) and torch.is_tensor(out2.get(k))
                else None)
            for k in ("g_str", "route_logits", "z_tac")}
        scan["cfg_no_strategic"] = bool(getattr(cfg.core, "no_strategic", False))
        res[arm] = scan

    a, b = res["strategic_flag_DEFAULT"], res["no_strategic_TRUE"]
    res["DELTA_params_default_minus_nostrategic"] = (
        a["total_params"] - b["total_params"])
    res["VERDICT"] = {
        name: {
            "built_default": a["targets"][name]["n_params"] > 0,
            "built_no_strategic": b["targets"][name]["n_params"] > 0,
            "params_default": a["targets"][name]["n_params"],
            "params_no_strategic": b["targets"][name]["n_params"],
        } for name in TARGETS
    }
    p = OUT / "q3_strategic_heads.json"
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["VERDICT"], indent=1))
    print("\ntotal params default / --no-strategic :",
          a["total_params"], "/", b["total_params"],
          "delta", res["DELTA_params_default_minus_nostrategic"])
    print("controls (must be non-zero in both):",
          {c: (a["controls"][c]["n_keys"], b["controls"][c]["n_keys"])
           for c in CONTROLS})
    print("forward emits (default):", a["forward_emits"])
    print("forward emits (--no-strategic):", b["forward_emits"])
    print("moves with input (default):", a["forward_moves_with_input"])
    print("moves with input (--no-strategic):", b["forward_moves_with_input"])
    print(f"\n[banked] {p}")


if __name__ == "__main__":
    main()
