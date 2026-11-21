#!/usr/bin/env python3
"""G0 -- the offline re-pick must reproduce REFe's banked `pick` BIT-FOR-BIT on every token, before anything else.

Arms:
  GREEN  torch-f32 navsim_v1_lane on table.logits      == table.pick           (1058/1058 required)
  GREEN  same on the ego probe's logits_all_A0         == probe pick[:, A0]    (1058/1058 required)
  GREEN  numpy-f32 / numpy-f64 variants                (reported; must also be 1058/1058 to call the rule implementation-robust)
  RED-1  rule navsim_v1 (lane factor dropped)          must NOT reproduce -> the gate must FAIL on it
  RED-2  the old `logitsum` rule (planner.py:581-582)  must NOT reproduce -> the gate must FAIL on it
  RED-3  argmin instead of argmax (mutation)           must NOT reproduce
Exit 0 only if every GREEN passes AND every RED arm fails the gate.
"""
import json
import sys

import numpy as np

import sel_common as sc


def gate(pred, ref):
    n_eq = int((np.asarray(pred) == np.asarray(ref)).sum())
    return {"n_equal": n_eq, "n": int(len(ref)), "PASS": n_eq == len(ref)}


def main():
    D = sc.load()
    out = {}
    out["green_torch_table"] = gate(sc.pick_from_logits(D["logits"], "torch"), D["pick"])
    out["green_torch_probeA0"] = gate(sc.pick_from_logits(D["logits_A0"], "torch"), D["pick_A0_probe"])
    out["logits_table_vs_probe_bit_identical"] = bool((D["logits"] == D["logits_A0"]).all())
    out["green_np32_table"] = gate(sc.pick_from_logits(D["logits"], "np32"), D["pick"])
    out["green_np64_table"] = gate(sc.pick_from_logits(D["logits"], "np64"), D["pick"])
    out["red1_no_lane"] = gate(sc.pick_from_logits(D["logits"], "torch", lane=False), D["pick"])
    out["red2_logitsum"] = gate(np.argmax(D["logits"][..., :6].astype(np.float32).sum(-1), -1), D["pick"])
    out["red3_argmin"] = gate(np.argmin(sc.agg_torch(D["logits"]), -1), D["pick"])
    # near-tie census: how close is the runner-up to the max (a fragile pick would flip under a tiny re-implementation change)
    a = sc.agg_np(D["logits"], np.float64)
    s = np.sort(a, -1)
    out["min_top2_gap_f64"] = float((s[:, -1] - s[:, -2]).min())
    out["n_top2_gap_lt_1e-6"] = int(((s[:, -1] - s[:, -2]) < 1e-6).sum())
    greens = [k for k in out if k.startswith("green_torch")]
    reds = [k for k in out if k.startswith("red")]
    out["G0_PASS"] = bool(all(out[k]["PASS"] for k in greens) and out["logits_table_vs_probe_bit_identical"])
    out["RED_ARMS_FAIL_AS_REQUIRED"] = bool(all(not out[k]["PASS"] for k in reds))
    out["EXIT_OK"] = out["G0_PASS"] and out["RED_ARMS_FAIL_AS_REQUIRED"]
    json.dump(out, open(sc.OUT + "/g0_gate.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["EXIT_OK"] else 1)


if __name__ == "__main__":
    main()
