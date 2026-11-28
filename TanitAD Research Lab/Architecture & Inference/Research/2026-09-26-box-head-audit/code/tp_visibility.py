#!/usr/bin/env python3
"""tp_visibility.py -- box-head audit: how much of refcv6's detection picture is GT visibility?

Joins, per window, the refcv6 head's greedy 2 m matches (bha_analyze.Scorer.window: the trainer's target set, the
programme's AP matcher) with each target's camera visibility (vis_zbuf rows, same window, same row order), then:
  * the visibility of the TARGETS the head HITS and of those it MISSES, at gate 0.5;
  * the head re-scored under VIS-1 (POSITIVE = target, IGNORE = removed from the PR count, DROPPED = not a target):
    precision / recall / AP at gate 0.5 and at the P = R gate, beside the filter-only scoring.
The re-score re-runs the greedy matcher on the VIS-1 target set (a detection on a DROPPED row becomes a FP unless it
is within 2 m of an IGNORE row; a detection matched to an IGNORE row leaves the count).
sha12 only.
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bha_analyze import Scorer, at_gate, pr_equal_gate  # noqa: E402
from vis_rules import label  # noqa: E402


def main():
    tree, dump_dir, sset, vis_boxes, out = sys.argv[1:6]
    import glob
    import torch
    rows = []
    for f in sorted(glob.glob(f"{dump_dir}/{sset}_chunk*.pt")):
        rows += torch.load(f, weights_only=False)
    rec = json.load(open(f"{dump_dir}/{sset}_record.json"))
    sc = Scorer(tree, (rec.get("model") or {}).get("cls_class_weight"))
    vb = json.load(open(vis_boxes))
    byw = {}
    for r in vb:
        byw.setdefault((r["sha12"], int(r["w"])), []).append(r)
    res = {"set": sset}
    for head in ("box3d", "agent"):
        hit_vis, miss_vis = [], []
        rows_f, rows_v = [], []
        n_gt_f = n_gt_v = 0
        n_skip = 0
        for r in rows:
            if not r["agent_label"]:
                continue
            vr = byw.get((r["sha12"], int(r["t"])), [])
            A = int(r["agent_box"].shape[0])
            if len(vr) != A:
                n_skip += 1
                continue
            w = sc.window(r, head)
            conf = w["p"] >= 0.5
            hitc = {int(j) for i, j in enumerate(w["hcol"]) if j >= 0 and conf[i]}
            for j in np.nonzero(w["vis"])[0]:
                (hit_vis if int(j) in hitc else miss_vis).append(vr[j]["vis_frac"])
            rows_f += w["conf_rows"]
            n_gt_f += w["n_vis"]
            # VIS-1 re-score: greedy on POSITIVES only; IGNORE rows absorb detections within 2 m
            lab = [label(x) if x["target"] else "none" for x in vr]
            pos = [j for j in range(A) if lab[j] == "positive"]
            ign = [j for j in range(A) if lab[j] == "ignore"]
            n_gt_v += len(pos)
            box = w["box"]
            gb = w["gbox"]
            order = np.argsort(-w["p"], kind="stable")
            taken = set()
            for i in order:
                d_pos = [(math.hypot(*(box[i, :2] - gb[j, :2])), j) for j in pos if j not in taken]
                best = min(d_pos) if d_pos else (math.inf, -1)
                if best[0] <= 2.0:
                    taken.add(best[1])
                    rows_v.append((float(w["p"][i]), 1))
                    continue
                if any(math.hypot(*(box[i, :2] - gb[j, :2])) <= 2.0 for j in ign):
                    continue                                   # IGNORE: out of the count
                rows_v.append((float(w["p"][i]), 0))
        hv, mv = np.asarray(hit_vis, float), np.asarray(miss_vis, float)
        pe_f, pe_v = pr_equal_gate(rows_f, n_gt_f), pr_equal_gate(rows_v, n_gt_v)
        res[head] = {
            "n_windows_skipped_row_mismatch": n_skip,
            "hits_at_0.5": {"n": int(hv.size), "frac_vis_lt_0.10": float((hv < 0.1).mean()) if hv.size else None,
                            "frac_vis_lt_0.30": float((hv < 0.3).mean()) if hv.size else None},
            "misses_at_0.5": {"n": int(mv.size), "frac_vis_lt_0.10": float((mv < 0.1).mean()) if mv.size else None,
                              "frac_vis_lt_0.30": float((mv < 0.3).mean()) if mv.size else None},
            "filter_only": {"n_gt": n_gt_f, "at_0.5": at_gate(rows_f, n_gt_f, 0.5), "pr": pe_f},
            "vis1": {"n_gt": n_gt_v, "at_0.5": at_gate(rows_v, n_gt_v, 0.5), "pr": pe_v},
        }
        print(head, json.dumps(res[head], default=float)[:1500], flush=True)
    json.dump(res, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
