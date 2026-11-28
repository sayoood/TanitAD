"""WP-B end-to-end on the REAL refcv5 config -- not a fake layer stub.

⛔ THE STUB TRAP: `check_trainer.py` exercises `assert_seams_are_built` against
hand-built fake layers, and a fake that cannot express the real fields is
doubly inert (the `CLAUDE.md` conditioning-test failure: a flat two-attribute
stub checking a map whose real fields are nested). This script builds the ACTUAL
`RefCV3Model` from the ACTUAL parsed args and asserts on the built modules.
"""
import json
import sys

sys.path.insert(0, r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
sys.path.insert(0, r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack/scripts")

import torch                                                       # noqa: E402
import refc_v3_train as T                                          # noqa: E402
from tanitad.refs import refc_v3 as v3                             # noqa: E402
from tanitad.refs.refc import param_breakdown                      # noqa: E402

BASE = ["--arm", "hier", "--out", "/tmp/wpb", "--steps", "40284",
        "--agents", "oracle", "--agent-join", "joins/train2400_agents.jsonl.xz",
        "--image-hw", "256", "640"]
out = {}


def build(extra):
    args = T.build_parser().parse_args(BASE + extra)
    cfg = v3.RefCV3Config()
    T._pin_refcv5_seams(cfg, args)
    torch.manual_seed(1234)
    model = v3.RefCV3Model(cfg)
    stamp = T._seam_stamp(cfg, args)
    return args, cfg, model, stamp


for name, extra in (("index_off", []), ("index_on", ["--wp-index", "on"])):
    args, cfg, model, stamp = build(extra)
    core = getattr(model, "core", model)
    dec = core.decoder
    layers = list(dec.layers)
    rec = {
        "n_decoder_layers": len(layers),
        "n_heads": cfg.core.decoder.n_heads,
        "layers_with_cross_agent": sum(1 for ly in layers
                                       if ly.cross_agent is not None),
        "layers_with_wp_index": sum(1 for ly in layers
                                    if ly.wp_index is not None),
        "wp_index_cfg_mode": getattr(dec.wp_index_cfg, "mode", None),
        "wp_index_params_method": dec.wp_index_params(),
        "param_breakdown_wp_index": param_breakdown(core)["wp_index"],
        "param_breakdown_decoder": param_breakdown(core)["decoder"],
        "param_breakdown_total": param_breakdown(core)["total"],
        "state_dict_wp_index_keys": sum(1 for k in model.state_dict()
                                        if "wp_index" in k),
        "stamp_wp_index_enable": (stamp["wp_index"] or {}).get("enable"),
    }
    try:
        T.assert_seams_are_built(model, stamp)
        rec["assert_seams_are_built"] = "PASSED"
    except SystemExit as exc:
        rec["assert_seams_are_built"] = " ".join(str(exc).split())[:200]
    out[name] = rec

# the deltas a reader needs
off, on = out["index_off"], out["index_on"]
out["deltas"] = {
    "wp_index_params": on["param_breakdown_wp_index"]
    - off["param_breakdown_wp_index"],
    "decoder_line_unchanged": on["param_breakdown_decoder"]
    == off["param_breakdown_decoder"],
    "total_delta": on["param_breakdown_total"] - off["param_breakdown_total"],
    "analytic_expected": (8 * 32 + 32 + 32 * on["n_heads"] + on["n_heads"])
    * on["n_decoder_layers"],
}
print(json.dumps(out, indent=2, default=str))
