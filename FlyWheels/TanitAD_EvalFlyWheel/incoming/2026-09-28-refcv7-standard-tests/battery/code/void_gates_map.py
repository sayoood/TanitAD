"""SPEC §3.6 VOID gates 5 and 6 (model-free / artifact-only; CPU).

    python void_gates_map.py --g0 <g0.json> --out <json>

Gate 5 (the A7 anchored-coordinate control): for every eval clip that has BOTH the `/2` GT
(`sam3_gt_eval_thor137`, 600 x 320) and the `/3` GT (`sam3_gt_v3_eval`, 1000 x 600), the `/3`
`fine_codes` inside the old 60 m x +-16 m window -- rows 0..599, cols `anchor_offset` (140) .. +320 --
must equal `/2` byte for byte on every frame. Also a literal check of the refcv6 upsampling hook:
`coarse_to_fine_codes` of a one-hot 0.5 m map places cell (i, j) on fine rows 5i..5i+4 and cols
140+5j..140+5j+4 and NO_PREDICTION everywhere outside.
Gate 6 (perception is seed-invariant): in the G0 artifact every MAP10 / DETECTION / MATCHED term
has rel spread exactly 0 over the 8 inference seeds (the perception heads run before the DDIM
planner). If not, the perception pass must be rolled at both seeds.
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", default=None)
    ap.add_argument("--kit", default="D:/refcv6_eval_kit/data")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from taniteval import map_hires_metrics as MH
    from tanitad.data.semantic_map_gt_fine import EXTENT_REFCV7, FINE_CELL_M
    out = {"tool": "void_gates_map.py"}
    d2 = os.path.join(a.kit, "sam3_gt_eval_thor137")
    d3 = os.path.join(a.kit, "sam3_gt_v3_eval")
    n2 = {f for f in os.listdir(d2) if f.endswith(".npz")}
    n3 = {f for f in os.listdir(d3) if f.endswith(".npz")}
    both = sorted(n2 & n3)
    c0 = int(EXTENT_REFCV7.anchor_offset(FINE_CELL_M))
    bad, frames = [], 0
    for f in both:
        with np.load(os.path.join(d2, f), allow_pickle=False) as z2, \
                np.load(os.path.join(d3, f), allow_pickle=False) as z3:
            a2, a3 = z2["fine_codes"], z3["fine_codes"]
        if a2.shape[0] != a3.shape[0]:
            bad.append({"clip": f.split(".")[0], "why": f"frame count {a2.shape[0]} vs {a3.shape[0]}"})
            continue
        win = a3[:, 0:a2.shape[1], c0:c0 + a2.shape[2]]
        frames += int(a2.shape[0])
        if not np.array_equal(win, a2):
            bad.append({"clip": f.split(".")[0], "n_frames_differ":
                        int((win != a2).reshape(a2.shape[0], -1).any(1).sum())})
    out["gate5_gt_window_identity"] = {"n_clips_both": len(both), "n_only_v2": len(n2 - n3),
                                       "n_only_v3": len(n3 - n2), "n_frames": frames,
                                       "anchor_col_offset": c0, "failures": bad[:20],
                                       "pass": bool(both) and not bad}
    # the refcv6 hook, a literal
    oh = np.zeros((8, 120, 64), np.float32)
    oh[0] = 1.0
    oh[:, 3, 7] = 0.0
    oh[5, 3, 7] = 1.0                                    # cell (3, 7) is class 5, the rest class 0
    fine = MH.coarse_to_fine_codes(oh, extent=EXTENT_REFCV7)
    blk = fine[15:20, c0 + 35:c0 + 40]
    outside = np.concatenate([fine[600:].ravel(), fine[:600, :c0].ravel(), fine[:600, c0 + 320:].ravel()])
    lit = {"block_is_class5": bool((blk == 5).all()), "n_class5_cells": int((fine == 5).sum()),
           "outside_all_no_prediction": bool((outside == MH.NO_PREDICTION).all()),
           "shape": list(fine.shape)}
    lit["pass"] = bool(lit["block_is_class5"] and lit["n_class5_cells"] == 25
                       and lit["outside_all_no_prediction"] and lit["shape"] == [1000, 600])
    out["gate5_refcv6_hook_literal"] = lit
    if a.g0:
        g = json.load(open(a.g0, encoding="utf-8"))
        terms = (g.get("verdict") or {}).get("terms") or {}
        nz = {k: v.get("rel_spread") for k, v in terms.items()
              if str(v.get("class")) in ("MAP10_COUNTS", "MAP10_IOU", "DETECTION", "MATCHED")
              and v.get("rel_spread") not in (None, 0, 0.0)}
        n_all = sum(1 for v in terms.values()
                    if str(v.get("class")) in ("MAP10_COUNTS", "MAP10_IOU", "DETECTION", "MATCHED")
                    and v.get("rel_spread") is not None)
        out["gate6_perception_seed_invariant"] = {"n_terms": n_all, "n_nonzero_spread": len(nz),
                                                  "nonzero_first10": dict(list(nz.items())[:10]),
                                                  "pass": n_all > 0 and not nz}
    out["pass"] = all(v.get("pass") for k, v in out.items() if isinstance(v, dict))
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in out.items()}))


if __name__ == "__main__":
    main()
