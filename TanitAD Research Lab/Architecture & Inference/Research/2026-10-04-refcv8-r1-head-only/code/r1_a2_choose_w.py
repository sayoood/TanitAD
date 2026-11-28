"""SPEC_R1_A2-2: choose the guidance scale w on the TRAIN-FIT windows (never on eval).
Rule (registered): the smallest w in {1, 3, 10} whose pooled forced-direction controllability (TURN_L -> pick dir +1,
TURN_R -> pick dir -1) on the classified train-fit windows is >= 0.95; if none, w = 10. Reads
arms/H5g_w<w>/trainfit_s0.pt; writes arms/a2_choice.json and prints the chosen w."""
import json
import sys
from pathlib import Path

import numpy as np

A = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/nvidia/refcv8_r1/arms")
CACHE = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/nvidia/refcv8_r1/cache")


def main():
    import torch
    sel = json.loads((CACHE / "train" / "selection.json").read_text())
    cls_of = {w["i"]: w["cls"] for w in sel["windows"]}
    out = {"rule": "smallest w in {1,3,10} with pooled TURN_L/TURN_R controllability >= 0.95 on classified "
                   "train-fit windows, else 10 (SPEC_R1_A2-2)", "per_w": {}}
    chosen = None
    for w in (1, 3, 10):
        r = torch.load(A / f"H5g_w{w}" / "trainfit_s0.pt", map_location="cpu", weights_only=False)
        cs = r["ctrl_sel_dir"].numpy()
        cls = np.array([cls_of[int(i)] for i in r["b_r1_i"]])
        m = np.isin(cls, ["turnL", "turnR", "straight", "gentle"]) & (cs[:, 0] != -9)
        lk = m & np.isin(cls, ["turnL", "turnR", "straight"])
        tl = float((cs[m, 0] == 1).mean())
        trr = float((cs[m, 1] == -1).mean())
        pooled = float(np.concatenate([cs[m, 0] == 1, cs[m, 1] == -1]).mean())
        out["per_w"][str(w)] = {"n": int(m.sum()), "TURN_L": tl, "TURN_R": trr,
                                "LANE_KEEP": float((cs[lk, 2] == 0).mean()), "pooled": pooled}
        if chosen is None and pooled >= 0.95:
            chosen = w
    out["chosen_w"] = chosen if chosen is not None else 10
    out["any_w_reached_0.95"] = chosen is not None
    (A / "a2_choice.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print(f"CHOSEN_W={out['chosen_w']}")


if __name__ == "__main__":
    main()
