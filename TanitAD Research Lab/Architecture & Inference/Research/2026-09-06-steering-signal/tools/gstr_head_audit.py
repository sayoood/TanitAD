"""D-GSTR-1 P1c -- is `str_goal_head` TRAINED, and where does the ~30 deg
clockwise offset live: in the BIAS, or in the weight?

Reads the banked checkpoint ONLY (CPU, no forward pass). Three questions:
  Q1  does the checkpoint carry optimizer state for `str_goal_head`, and is its
      Adam `step` non-zero?  (a head excluded from the optimizer is FROZEN)
  Q2  what is the BIAS direction atan2(b1, b0) in degrees, and how big is the
      bias next to the weight rows?  (a bias-dominated head is a CONSTANT
      bearing with a small vision-driven wobble -- exactly the observed shape)
  Q3  a same-breath CONTROL: the same statistics for heads we KNOW are trained
      (`route_head`, `lat_head_tac`) and for a FRESH init of the same shape.
      Without the control, "the numbers look small" is not evidence.

ASCII-only.
"""
import json
import math
import os
import sys

import torch

CKPT = os.environ.get("GSTR_CKPT", "/home/nvidia/refcv4b/ckpt_40284_FINAL.pt")
OUT = os.environ.get("GSTR_OUT3", "/home/nvidia/navroute/GSTR_HEAD_AUDIT.json")


def stat(t):
    t = t.detach().float()
    return {"shape": list(t.shape),
            "absmean": round(float(t.abs().mean()), 6),
            "std": round(float(t.std()), 6),
            "max_abs": round(float(t.abs().max()), 6),
            "l2": round(float(t.norm()), 6)}


def main():
    d = torch.load(CKPT, map_location="cpu", weights_only=False)
    res = {"tool": "D-GSTR-1 P1c str_goal_head audit", "ckpt": CKPT,
           "evidence_class": "MEASURED (ours)",
           "ckpt_top_keys": sorted([k for k in d]) if isinstance(d, dict) else
                            str(type(d))}
    sd = None
    for key in ("model", "state_dict", "model_state", "ema"):
        if isinstance(d, dict) and key in d and isinstance(d[key], dict):
            sd = d[key]
            res["state_dict_key"] = key
            break
    if sd is None and isinstance(d, dict):
        sd = {k: v for k, v in d.items() if torch.is_tensor(v)}
        res["state_dict_key"] = "<flat>"
    names = [k for k in sd if "str_goal_head" in k]
    res["str_goal_head_params"] = names
    if not names:
        res["ERROR"] = "no str_goal_head in the checkpoint"
        print(json.dumps(res, indent=1))
        return

    W = sd[[n for n in names if n.endswith("weight")][0]].float()
    b = sd[[n for n in names if n.endswith("bias")][0]].float()
    d_ctx = W.shape[1]
    res["Q2_bias_and_weight"] = {
        "d_ctx": int(d_ctx),
        "bias": [round(float(x), 6) for x in b.tolist()],
        "bias_angle_deg_atan2_b1_b0": round(
            math.degrees(math.atan2(float(b[1]), float(b[0]))), 4),
        "bias_xy_l2": round(float(b[:2].norm()), 6),
        "weight_row0_l2": round(float(W[0].norm()), 6),
        "weight_row1_l2": round(float(W[1].norm()), 6),
        "weight_row2_l2": round(float(W[2].norm()), 6),
        "_reads": ("if the bearing offset lives in the BIAS, "
                   "bias_angle_deg is near the measured -29.0 deg AND "
                   "bias_xy_l2 is large next to the weight rows"),
    }
    # ---- Q3 CONTROL: heads known to be trained, and a fresh init ----
    ctrl = {}
    for pat in ("route_head", "lat_head_tac", "lon_head_tac", "gp_head",
                "goal_head", "tac_goal_head"):
        ns = [k for k in sd if pat in k and k.endswith("weight")]
        if ns:
            ctrl[pat] = stat(sd[ns[0]])
            bn = ns[0][:-6] + "bias"
            if bn in sd:
                ctrl[pat + ".bias"] = stat(sd[bn])
    torch.manual_seed(0)
    fresh = torch.nn.Linear(int(d_ctx), 3)
    ctrl["FRESH_INIT_Linear(d_ctx,3).weight"] = stat(fresh.weight)
    ctrl["FRESH_INIT_Linear(d_ctx,3).bias"] = stat(fresh.bias)
    ctrl["_init_bound_1_over_sqrt_d"] = round(1.0 / math.sqrt(d_ctx), 6)
    ctrl["str_goal_head.weight"] = stat(W)
    ctrl["str_goal_head.bias"] = stat(b)
    res["Q3_controls"] = ctrl

    # ---- Q1 optimizer state ----
    q1 = {"optimizer_state_present": False}
    for key in ("opt", "optim", "optimizer", "optimizer_state_dict"):
        if isinstance(d, dict) and key in d:
            q1["optimizer_state_present"] = True
            q1["key"] = key
            o = d[key]
            try:
                st = o.get("state", {})
                q1["n_params_with_state"] = len(st)
                steps = [float(v["step"]) for v in st.values()
                         if isinstance(v, dict) and "step" in v]
                if steps:
                    q1["adam_step_min"] = min(steps)
                    q1["adam_step_max"] = max(steps)
                q1["_note"] = ("Adam state is keyed by param INDEX, not name; "
                               "a param with no entry was never stepped")
            except Exception as e:                       # noqa: BLE001
                q1["ERROR"] = repr(e)
            break
    res["Q1_optimizer"] = q1
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
