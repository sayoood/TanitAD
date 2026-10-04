"""WP-D -- did the box3d head's "learned" reference points (`--slot-query-select learned_ref`) move during refcv7?

Reads the run's milestone checkpoints (model-only, md5 in D:/refcv7_eval_kit/ckpt/MD5SUMS) on CPU and compares
`_perception.box_refpts.xy` [300, 2] (metres, planner grid 60 m x +-16 m) with its initialisation, which is
REPRODUCED here from the module's own rule (slot_query_select.LearnedRefPoints: a seed-0 torch.Generator, x ~ U(0, 60),
y ~ U(-16, 16)) -- the init is never read from a file, so a match is evidence, not an identity.
Also records the presence-logit biases of both slot heads (init logit(0.01) = -4.595) and the 10 cm map's class
biases. Output: raw/refpts_by_step.json. Evidence: MEASURED (checkpoints of refcv7-r101-s0).
Control: the step-5000 anchors must be within 1 m of the reproduced init for every query (if the init rule were
mis-reproduced, displacements would be ~uniform-random, i.e. tens of metres).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

CK = Path(sys.argv[1] if len(sys.argv) > 1 else "D:/refcv7_eval_kit/ckpt")
OUT = Path(__file__).resolve().parents[1] / "raw" / "refpts_by_step.json"
STEPS = [5000, 15000, 20000, 30000, 50400]


def main():
    g = torch.Generator().manual_seed(0)
    k = 300
    init = torch.stack([torch.rand(k, generator=g) * 60.0, (torch.rand(k, generator=g) * 2 - 1) * 16.0], -1).numpy()
    rec = {"init_rule": "slot_query_select.LearnedRefPoints(seed=0): x~U(0,60), y~U(-16,16)",
           "evidence": "MEASURED (refcv7-r101-s0 milestone checkpoints, CPU)", "by_step": {}}
    xs = [-1e9, 0, 10, 20, 30, 40, 50, 60, 1e9]
    ys = [0, 4, 8, 12, 16, 1e9]
    for s in STEPS:
        sd = torch.load(CK / f"ckpt_{s}.pt", map_location="cpu", weights_only=False)["model"]
        xy = sd["_perception.box_refpts.xy"].float().numpy()
        d = np.linalg.norm(xy - init, axis=1)
        if s == STEPS[0]:
            assert d.max() < 1.0, f"control FAIL: step {s} anchors are {d.max():.2f} m from the reproduced init"
        rec["by_step"][str(s)] = {
            "displacement_m": {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
                               "max": float(d.max())},
            "x_hist_edges_m": xs[1:-1], "x_hist": np.histogram(xy[:, 0], bins=xs)[0].tolist(),
            "abs_y_hist_edges_m": ys[:-1], "abs_y_hist": np.histogram(np.abs(xy[:, 1]), bins=ys)[0].tolist(),
            "box3d_presence_bias": float(sd["_perception.box_dec.head.bias"][0]),
            "agent_presence_bias": float(sd["core.agent_head.head.bias"][0]),
            "map_cls_bias": [round(float(v), 4) for v in sd["_map_hires.refine.cls.bias"]]}
        print(s, rec["by_step"][str(s)]["displacement_m"])
        del sd
    rec["control"] = "step-5000 anchors within 1 m of the reproduced init: PASS"
    OUT.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
