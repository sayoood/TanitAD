"""P1d - ARE THE THREE LANE-CHANGE LOGITS DEAD IN THE TRAINED HEAD?

`p.grad is None` is the right discriminator for "the loss never weighted it",
but the trainer is not running and the A40 is off limits, so the reachable
equivalent on a FINISHED run is the trained parameter itself: a softmax class
with zero positive examples anywhere in the corpus is only ever pushed DOWN by
the cross-entropy denominator, so its bias goes strongly negative relative to
the classes that do occur.

CONTROL: the classes that DO occur (LANE_KEEP / NUDGE_* / TURN_*) must show a
clearly different bias regime, or this probe is reading noise.
"""
import json
import sys

import numpy as np
import torch

import _env  # noqa: F401

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CKPT = r"C:\Users\Admin\refcv4b_final\ckpt_40284_FINAL.pt"


def main():
    from tanitad.models.vocab_v7 import (TACTICAL_LAT_ACTIONS_V7,
                                         TACTICAL_LON_ACTIONS_V7)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    d = sd.get("model", sd)
    heads = [k for k in d if "tac" in k.lower() and
             ("head" in k.lower() or "lat" in k.lower() or "lon" in k.lower())]
    print("tactical-head tensors in the checkpoint:")
    for k in sorted(heads):
        print(f"  {k:<60} {tuple(d[k].shape)}")
    n_lat, n_lon = len(TACTICAL_LAT_ACTIONS_V7), len(TACTICAL_LON_ACTIONS_V7)
    out = {}
    for axis, names in (("lat", TACTICAL_LAT_ACTIONS_V7),
                        ("lon", TACTICAL_LON_ACTIONS_V7)):
        n = len(names)
        wk = [k for k in d if k.endswith("weight") and d[k].ndim == 2
              and d[k].shape[0] == n and f".{axis}" in k.lower()]
        bk = [k for k in d if k.endswith("bias") and d[k].ndim == 1
              and d[k].shape[0] == n and f".{axis}" in k.lower()]
        if not wk:
            wk = [k for k in d if k.endswith("weight") and d[k].ndim == 2
                  and d[k].shape[0] == n and "tac" in k.lower()]
            bk = [k for k in d if k.endswith("bias") and d[k].ndim == 1
                  and d[k].shape[0] == n and "tac" in k.lower()]
        print(f"\n=== {axis.upper()} head ({n} classes) ===")
        print(f"  weight tensors: {wk}")
        print(f"  bias   tensors: {bk}")
        if not wk:
            print("  INCONCLUSIVE: no tensor of this output width found")
            continue
        W = d[wk[0]].float()
        B = d[bk[0]].float() if bk else torch.zeros(n)
        wn = W.norm(dim=1).numpy()
        bb = B.numpy()
        print(f"  {'class':<24}{'bias':>10}{'||w_row||':>12}"
              f"{'bias rank':>11}")
        order = np.argsort(-bb)
        rank = {int(i): int(r) + 1 for r, i in enumerate(order)}
        for i, nm in enumerate(names):
            mark = "   <== ZERO POSITIVES IN THE CORPUS" if nm in (
                "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC",
                "YIELD_MERGE") else ""
            print(f"  {nm:<24}{bb[i]:>10.4f}{wn[i]:>12.4f}"
                  f"{rank[i]:>11}{mark}")
        out[axis] = {"weight_key": wk[0], "bias_key": bk[0] if bk else None,
                     "bias": {nm: float(bb[i]) for i, nm in enumerate(names)},
                     "w_row_norm": {nm: float(wn[i])
                                    for i, nm in enumerate(names)}}
    with open("out_p1d_head_dead_classes.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote out_p1d_head_dead_classes.json")


if __name__ == "__main__":
    main()
