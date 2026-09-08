"""WP-B removability measurement -> raw/removability.json.

⛔⛔ THE REQUIRED PROOF is "flag OFF == a build that never had the seam", and it is
BITWISE. Everything beyond that is a bonus and is reported with its honest number.

⚠️ THE TRAP THIS SCRIPT EXISTS TO MAKE VISIBLE: with `agent_gate` at its zero
init the whole agent branch is multiplied away, so the on/off comparison passes
EVEN WITH A CORRUPTED BIAS HEAD. That row is measured here explicitly, beside
the live-gate row, so nobody can read the first as evidence for the second.
"""
import argparse
import json
import os

import torch

from tanitad.refs import refc_agents as ra
from tanitad.refs import refc_wp_index as wi
from tanitad.refs.refc import RefCModel, param_breakdown, refc_smoke_config

SEED = 1234


def build(index, gate=None, corrupt=False, seed=SEED):
    cfg = refc_smoke_config()
    cfg.agents = ra.AgentSeamConfig(enable=True, queries=6, d_model=32,
                                    depth=1, n_heads=2, enforce_band=False)
    cfg.decoder.cross_agent = True
    if index is not None:
        cfg.decoder.wp_index = index
    torch.manual_seed(seed)
    m = RefCModel(cfg)
    with torch.no_grad():
        if gate is not None:
            for ly in m.decoder.layers:
                ly.agent_gate.fill_(float(gate))
        if corrupt:
            for ly in m.decoder.layers:
                torch.nn.init.normal_(ly.wp_index.mlp[-1].weight, std=10.0)
                torch.nn.init.normal_(ly.wp_index.mlp[-1].bias, std=10.0)
    return cfg, m


def compare(off, on, cfg):
    off.eval(), on.eval()
    x = torch.randn(2, cfg.window, cfg.encoder.in_channels, 64, 64)
    torch.manual_seed(7)
    a = off(x)
    torch.manual_seed(7)
    b = on(x)
    keys = [k for k in a if torch.is_tensor(a[k])]
    bit = all(torch.equal(a[k], b[k]) for k in keys)
    md = max(float((a[k] - b[k]).abs().max()) for k in keys
             if a[k].is_floating_point())
    return {"keys_equal": sorted(set(a)) == sorted(set(b)),
            "n_tensor_keys": len(keys),
            "bitwise_equal": bool(bit),
            "max_abs_diff_over_all_tensors": md,
            "max_abs_diff_traj": float((a["traj"] - b["traj"]).abs().max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rec = {"torch": torch.__version__, "seed": SEED,
           "rig": "refc_smoke_config + AgentSeamConfig(queries=6, d_model=32)",
           "device": "cpu", "dtype": "float32"}

    # --- 1. the REQUIRED proof: agents on, index OFF vs never-had-the-field --
    cfgA, mA = build(None)
    _, mB = build(wi.WaypointIndexConfig(enable=False))
    sa, sb = mA.state_dict(), mB.state_dict()
    rec["required_flag_off_vs_no_field"] = {
        "state_dict_keys_identical": list(sa) == list(sb),
        "n_params_compared": len(sa),
        "all_params_bitwise_equal": all(torch.equal(sa[k], sb[k]) for k in sa),
        "any_wp_index_key_in_state_dict": any("wp_index" in k for k in sa),
        **compare(mA, mB, cfgA)}

    # --- 2. shared parameters between index-off and index-on ----------------
    _, off = build(None)
    _, on = build(wi.WaypointIndexConfig(enable=True))
    so, sn = off.state_dict(), on.state_dict()
    shared = [k for k in so if k in sn]
    rec["shared_params_off_vs_on"] = {
        "n_shared": len(shared), "n_off": len(so), "n_on": len(sn),
        "n_extra_keys_on": len(sn) - len(so),
        "shared_all_bitwise_equal": all(torch.equal(so[k], sn[k])
                                        for k in shared),
        "differing": [k for k in shared if not torch.equal(so[k], sn[k])]}

    # --- 3. the confound, MEASURED, and the live-gate number ---------------
    rows = {}
    for name, gate, corrupt in (("gate_0_zero_init_bias", 0.0, False),
                                ("gate_0_CORRUPTED_bias", 0.0, True),
                                ("gate_1_zero_init_bias", 1.0, False),
                                ("gate_1_CORRUPTED_bias", 1.0, True)):
        cfg0, o = build(None, gate=gate)
        _, n = build(wi.WaypointIndexConfig(enable=True), gate=gate,
                     corrupt=corrupt)
        rows[name] = compare(o, n, cfg0)
    rec["agent_gate_sweep"] = rows
    rec["reading"] = (
        "gate_0_CORRUPTED_bias being bitwise-equal is the CONFOUND: with "
        "agent_gate at its zero init the whole agent branch is multiplied "
        "away, so the shipped-init removability proof passes even on a "
        "corrupted head. The load-bearing number is gate_1_zero_init_bias, "
        "which is NOT bitwise equal - it is float32 kernel rounding from "
        "supplying an attn_mask at all - and gate_1_CORRUPTED_bias is what a "
        "real lever looks like on the same rig.")

    # --- 4. parameter cost --------------------------------------------------
    bo, bn = param_breakdown(off), param_breakdown(on)
    rec["param_cost"] = {
        "wp_index_off": bo["wp_index"], "wp_index_on": bn["wp_index"],
        "decoder_line_unchanged": bo["decoder"] == bn["decoder"],
        "total_delta": bn["total"] - bo["total"],
        "analytic_per_layer_8h_h_hH_H": (8 * 32 + 32 + 32 * 4 + 4),
        "layers": len(off.decoder.layers),
        "n_heads": off.cfg.decoder.n_heads,
        "note": ("smoke config: hidden 32, n_heads 4, layers 2 -> 420*2 = 840. "
                 "At refcv5's n_heads 8 / layers 4 the same formula gives "
                 "552*4 = 2208.")}

    p = os.path.join(args.out, "removability.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
    print(json.dumps(rec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
