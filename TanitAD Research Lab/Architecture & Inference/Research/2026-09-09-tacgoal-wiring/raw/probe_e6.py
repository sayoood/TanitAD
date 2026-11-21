"""Exercise the CONTRACT that `train()`'s split-fitting block (E6) depends on.

⛔ THE GAP THIS CLOSES IN MY OWN WORK. Both gradient probes set
`model._tac_goal_pos_weight = None` by hand, and the dataset tests set
`ds.tac_goal_targets` by hand — so NOTHING executes the block in `train()` that
fits `pos_weight` and `class_mask` on the loaded split. A wrong key there
(`mask_report(...)["mask"]` vs `["masks"]`) would surface only at a real launch,
after the corpus mounts and a GPU is already paid for.

This drives the REAL functions with synthetic labels and asserts, as literals,
every key and shape E6 reads.

Run:  python probe_e6.py <tree>/stack
"""
from __future__ import annotations

import json
import os
import sys

STACK = os.path.abspath(sys.argv[1])
if STACK not in sys.path:
    sys.path.insert(0, STACK)

import torch                                                     # noqa: E402
from tanitad.data import v7_labels as v7l                        # noqa: E402
from tanitad.refs import tac_goal_head as tgh                    # noqa: E402


def _label(clip, goals, t0=8.0):
    return v7l.V7Label(
        clip_id=clip, tac_lat="LANE_KEEP", tac_lon="CRUISE",
        str_action="HOLD_MAIN_ROAD", str_goal="HOLD_MAIN_ROAD",
        tac_anchor=None, bands={"tactical_s": [0.0, 60.0]}, t0_s=t0,
        horizon={}, tac_goals=frozenset(goals),
        tac_goal_meta={k: dict(v) for k, v in goals.items()})


def main():
    v7l._MEASURED_GEOMETRY_TOKENS = frozenset(
        {"FOLLOW_LANE", "SPEED_BAND", "STOP_POINT", "TURN_L", "TURN_R",
         "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R"})
    labels = [
        _label("c1", {"FOLLOW_LANE": {"provenance": "geometry"},
                      "SPEED_BAND": {"provenance": "geometry"}}),
        _label("c2", {"TURN_L": {"provenance": "geometry"},
                      "SPEED_BAND": {"provenance": "geometry"}}),
        _label("c3", {"FOLLOW_LANE": {"provenance": "geometry"}}),
    ]

    # --- exactly the four calls E6 makes, in order -------------------------
    census = v7l.goal_supervision_census(labels)
    mask = tgh.mask_report(census)
    pw = v7l.goal_pos_weight(labels)
    pw_t = torch.tensor(pw, dtype=torch.float32)
    mask_t = torch.tensor(mask["mask"], dtype=torch.float32)

    out = {
        "n_tokens": len(v7l.TAC_GOAL_TOKENS),
        "census_len": len(census),
        "mask_keys": sorted(mask),
        "n_trainable": int(mask["n_trainable"]),
        "n_total": int(mask["n_total"]),
        "trainable_is_list": isinstance(mask["trainable"], list),
        "masked_why_is_dict": isinstance(mask["masked_why"], dict),
        "pos_weight_len": len(pw),
        "pw_tensor_shape": list(pw_t.shape),
        "mask_tensor_shape": list(mask_t.shape),
        "pw_dtype": str(pw_t.dtype),
        "mask_dtype": str(mask_t.dtype),
        # the loss must accept them in exactly this form
        "loss_accepts": None,
    }
    K = len(v7l.TAC_GOAL_TOKENS)
    logits = torch.zeros(2, K, requires_grad=True)
    y = torch.zeros(2, K)
    w = torch.ones(2, K)
    loss, n = tgh.tac_goal_loss(logits, y, w, pos_weight=pw_t,
                                class_mask=mask_t)
    loss.backward()
    out["loss_accepts"] = {
        "loss": float(loss.detach()),
        "n_supervised": int(n),
        "grad_is_not_none": logits.grad is not None,
    }
    # and the JSON-serialisability of the config.json stamp E6 writes
    stats = {
        "negatives": "measured",
        "n_trainable": int(mask["n_trainable"]),
        "n_total": int(mask["n_total"]),
        "trainable": list(mask["trainable"]),
        "masked_why": mask["masked_why"],
        "pos_weight": [float(x) for x in pw],
        "census": census,
    }
    out["stamp_json_bytes"] = len(json.dumps(stats))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
