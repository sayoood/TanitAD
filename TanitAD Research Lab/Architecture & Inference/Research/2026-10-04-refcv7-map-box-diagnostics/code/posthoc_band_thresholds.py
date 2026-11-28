"""POST-HOC (not in SPEC.md; logged as such in RESULT.md): per-class x PER-BAND one-vs-rest thresholds on phat,
fitted on the TRAIN-DIAG histograms, scored EXACTLY (bin edges) on the EVAL-DIAG histograms. Zero GPU.
Compared with the pre-registered single tau per class (thr_phat). Output: posthoc_band_thresholds.json."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_metrics as dm  # noqa: E402


def main(train_acc, eval_acc, out):
    tr = torch.load(train_acc, map_location="cpu", weights_only=False)
    ev = torch.load(eval_acc, map_location="cpu", weights_only=False)
    bk = list(ev["band_keys"])
    edges = dm.hist_edges()
    res = {"definition": "tau_{c,b} = argmax TRAIN IoU of {logit(phat_c) >= tau} within band b; EVAL IoU at "
                         "tau_{c,b} (exact at bin edges); 'pooled' sums inter/pred/gt over bands",
           "classes": {}}
    for score in ("phat",):
        ht, he = tr["hist"][score].numpy(), ev["hist"][score].numpy()
        for c, name in enumerate(dm.CLASS_KEYS):
            row, I, P, G = {}, 0.0, 0.0, 0.0
            I1, P1, G1 = 0.0, 0.0, 0.0
            i_glob, _ = dm.best_threshold(ht[c].sum(0)[0], ht[c].sum(0)[1])
            for b, key in enumerate(bk):
                i_b, tr_iou = dm.best_threshold(ht[c, b, 0], ht[c, b, 1])
                iou_b, pred, inter, g = dm.iou_at_index(he[c, b, 0], he[c, b, 1], i_b)
                iou_g, pred1, inter1, _ = dm.iou_at_index(he[c, b, 0], he[c, b, 1], i_glob)
                row[key] = {"tau_prob": float(1 / (1 + np.exp(-edges[i_b]))), "train_iou": tr_iou,
                            "eval_iou_band_tau": iou_b, "eval_iou_global_tau": iou_g, "n_gt_eval": g}
                I, P, G = I + inter, P + pred, G + g
                I1, P1, G1 = I1 + inter1, P1 + pred1, G1 + g
            row["pooled"] = {"eval_iou_band_tau": dm.iou(I, P, G), "eval_iou_global_tau": dm.iou(I1, P1, G1)}
            res["classes"][name] = row
    Path(out).write_text(json.dumps(res, indent=1))
    for name, row in res["classes"].items():
        print(f"{name}: pooled band-tau {row['pooled']['eval_iou_band_tau']:.4f} vs global-tau "
              f"{row['pooled']['eval_iou_global_tau']:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
