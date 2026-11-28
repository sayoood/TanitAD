#!/usr/bin/env python3
"""glive_arms.py -- the refcv6 38k READINGS of the two proposed G-LIVE checks, per set, so the red arms are MEASURED:

  G-LIVE-GATE   conf_ratio = sum(n slots >= gate) / sum(targets)   band [0.5, 2.0]
  G-LIVE-COUNT  sum(p_hat) / sum(targets)                          band [0.7, 1.4]

Arms read on the SAME windows:
  * refcv6 rule: gate 0.5 on sigmoid(logit)                          (the run as configured -> must FAIL GATE)
  * TRAIN-derived gate: the P = R gate measured on the TRAIN set, applied unchanged to eval
  * p_hat = sigmoid(logit - ln 10) (the tilt-corrected, calibrated probability)  (COUNT, correct arm)
  * p_hat = sigmoid(logit) (the tilt IGNORED)                                    (COUNT, red arm -> must FAIL)
Targets = the trainer's filter-only targets AND, separately, VIS-1 POSITIVES (vis_zbuf rows).
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bha_analyze import Scorer, at_gate  # noqa: E402
from vis_rules import label  # noqa: E402


def main():
    tree, dump_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    train_gate = json.load(open(f"{dump_dir}/analyze_train.json"))["sets"]["train"]
    import glob
    import torch
    res = {}
    for sset, vis_boxes in (("clipgrid", "/home/nvidia/bha_2325/ov/out/vis_clipgrid_boxes.json"),
                            ("inrun", "/home/nvidia/bha_2325/ov/out/vis_inrun_boxes.json"),
                            ("train", "/home/nvidia/bha_2325/ov/out/vis_train_boxes.json")):
        rows = []
        for f in sorted(glob.glob(f"{dump_dir}/{sset}_chunk*.pt")):
            rows += torch.load(f, weights_only=False)
        rec = json.load(open(f"{dump_dir}/{sset}_record.json"))
        sc = Scorer(tree, (rec.get("model") or {}).get("cls_class_weight"))
        vb = {}
        for r in json.load(open(vis_boxes)):
            vb.setdefault((r["sha12"], int(r["w"])), []).append(r)
        res[sset] = {}
        for head in ("box3d", "agent"):
            g_tr = float(train_gate[head]["pr"]["pr_equal"]["gate"])
            n_t = n_pos = 0
            s_raw = s_tilt = 0.0
            n05 = ntr = 0
            conf_rows = []
            for r in rows:
                if not r["agent_label"]:
                    continue
                w = sc.window(r, head)
                n_t += w["n_vis"]
                vr = vb.get((r["sha12"], int(r["t"])), [])
                n_pos += sum(1 for x in vr if x["target"] and label(x) == "positive")
                s_raw += float(w["p"].sum())
                s_tilt += float((1 / (1 + np.exp(-(w["logit"] - math.log(10))))).sum())
                n05 += int((w["p"] >= 0.5).sum())
                ntr += int((w["p"] >= g_tr).sum())
                conf_rows += w["conf_rows"]
            a_tr = at_gate(conf_rows, n_t, g_tr)
            res[sset][head] = {
                "n_targets_filter_only": n_t, "n_vis1_positives": n_pos, "train_derived_gate": g_tr,
                "GATE_conf_ratio@0.5_vs_targets": n05 / max(n_t, 1),
                "GATE_conf_ratio@train_gate_vs_targets": ntr / max(n_t, 1),
                "GATE_conf_ratio@train_gate_vs_vis1": ntr / max(n_pos, 1),
                "P_R@train_gate": [a_tr["precision"], a_tr["recall"]],
                "COUNT_sum_p_tilt_vs_targets": s_tilt / max(n_t, 1),
                "COUNT_sum_p_tilt_vs_vis1": s_tilt / max(n_pos, 1),
                "COUNT_sum_p_raw_vs_targets (red arm)": s_raw / max(n_t, 1)}
            print(sset, head, json.dumps(res[sset][head]), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
