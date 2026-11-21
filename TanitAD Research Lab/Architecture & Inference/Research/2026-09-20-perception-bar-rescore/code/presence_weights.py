"""Has the PRESENCE PROJECTION collapsed? A weight inspection — no forward pass at all.

`a15de2d` narrowed it: the decoder emits 100 distinct boxes while presence is near-constant across
slots AND near-invariant across windows. So it is not a trunk or capacity failure; the remaining
explanation is local to the presence term or its PROJECTION.

⭐ THE PROJECTION HALF IS DECIDABLE FROM THE CHECKPOINT ALONE. `agent_slots.py` emits every slot
field from ONE linear head sliced by `SLOT_SLICES` (`:368`), and the presence bias is initialised
to logit(0.05) at `:314`. If the presence ROWS of that weight matrix have collapsed toward zero,
the logit is bias-dominated and therefore CONSTANT by construction — which would explain the
across-slot constancy and the across-window invariance in one stroke.

⛔ THE CONTROL IS THE OTHER ROWS OF THE SAME MATRIX. `box`, `cls`, `size` come from the identical
layer and the same optimiser. If presence's row norm is comparable to theirs, the projection is
healthy and the cause lies in the loss, not the weights. Comparing presence against ITSELF at init
would measure nothing.
⛔ No model forward, no GPU, read-only.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack"))
from tanitad.models.agent_slots import SLOT_SLICES        # noqa: E402  canonical layout

CKPT = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run/ckpt_5000.pt")


def main() -> int:
    d = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = d.get("model") or d.get("state_dict") or d
    if not isinstance(sd, dict):
        print("⛔ unexpected checkpoint layout:", type(sd))
        return 3

    total = sum(int(SLOT_SLICES[k].stop - SLOT_SLICES[k].start) for k in SLOT_SLICES)
    print(f"SLOT_SLICES fields: {len(SLOT_SLICES)}, total width {total}")

    # find the slot head: a weight whose OUT dim equals the slot width
    cands = [(k, tuple(v.shape)) for k, v in sd.items()
             if hasattr(v, "shape") and v.dim() == 2 and int(v.shape[0]) == total]
    if not cands:
        print("⛔ no 2-D weight with out-dim", total, "— candidate keys containing 'head':")
        for k, v in sd.items():
            if "head" in k.lower() and hasattr(v, "shape"):
                print("   ", k, tuple(v.shape))
        return 4
    for k, s in cands:
        print(f"  slot-head candidate: {k} {s}")
    key = [k for k, _ in cands if "slot" in k.lower() or "head" in k.lower()] or [cands[0][0]]
    W = sd[key[0]].float()
    print(f"\nusing {key[0]}  shape {tuple(W.shape)}")

    rows = {}
    for name, sl in SLOT_SLICES.items():
        block = W[sl]
        n = int(block.shape[0])
        rows[name] = {"rows": n,
                      "mean_row_L2": round(float(block.norm(dim=1).mean()), 6),
                      "max_abs": round(float(block.abs().max()), 6)}

    pres = rows.get("presence")
    others = {k: v for k, v in rows.items() if k != "presence"}
    med_other = sorted(v["mean_row_L2"] for v in others.values())[len(others) // 2]
    ratio = pres["mean_row_L2"] / med_other if med_other else float("nan")

    bkey = [k for k in sd if k.startswith(key[0].rsplit(".", 1)[0]) and k.endswith("bias")]
    bias_pres = None
    if bkey:
        b = sd[bkey[0]].float()[SLOT_SLICES["presence"]]
        bias_pres = round(float(b.mean()), 6)

    res = {"_what": "has the presence PROJECTION collapsed? weight inspection, no forward pass",
           "_evidence_class": "MEASURED (ours), read-only, no GPU",
           "_control": "the OTHER slot fields' rows in the SAME linear layer",
           "head_key": key[0], "per_field": rows,
           "presence_mean_row_L2": pres["mean_row_L2"],
           "median_other_field_row_L2": round(med_other, 6),
           "ratio_presence_to_median_other": round(ratio, 4),
           "presence_bias_mean": bias_pres,
           "init_bias_logit_0p05": round(math.log(0.05 / 0.95), 6)}
    res["_VERDICT"] = (
        "⛔ PROJECTION COLLAPSED — presence rows are near-zero beside their siblings; the logit is "
        "bias-dominated and constant BY CONSTRUCTION" if ratio < 0.15 else
        "the presence projection is COMPARABLE to its siblings ⇒ the weights are NOT the cause and "
        "the explanation lies in the loss/optimisation, not the layer" if ratio > 0.5 else
        "⚠️ INTERMEDIATE — presence rows are smaller than their siblings but not degenerate")
    print(json.dumps(res, indent=1))
    print("\n" + res["_VERDICT"])
    pathlib.Path("presence_weights.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
