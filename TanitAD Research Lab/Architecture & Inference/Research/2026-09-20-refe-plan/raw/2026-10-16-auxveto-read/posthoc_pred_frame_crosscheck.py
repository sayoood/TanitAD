#!/usr/bin/env python3
"""posthoc_pred_frame_crosscheck.py -- POST-HOC PIPELINE CROSS-CHECK, NOT IN THE PREREG, DECIDES NOTHING.

The IoU identity (§9.2) proves the predicted MASK sits in the GT frame.  This asks the same of the whole PREDICTED-GRID path (2 m cells, `PRED_GEOM`, EDT at 2 m, bilinear sampling):
feed it the GT mask DOWN-SAMPLED to the head's 2 m grid (a 'perfect head on this grid') and compare its proposal-level veto with NAVSIM's DAC = 0, next to the 0.5 m GT raster.
An x/y transposition or a half-cell shift in the predicted-grid path would collapse its agreement.  Rasters the head never predicted; nothing here touches AUXVETO_READ.json.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import auxveto_read as av  # noqa: E402
import geodac as G  # noqa: E402

av.assert_prereg()
w, _ = av.load_world()
gt = av.load_gt_masks(w)
dac0 = w.sub[:, :, 1] == 0
out = {"note": "POST-HOC; decides nothing"}

def conf(veto, label):
    tp, fp, fn, tn = int((veto & label).sum()), int((veto & ~label).sum()), int((~veto & label).sum()), int((~veto & ~label).sum())
    n = tp + fp + fn + tn
    po, pe = (tp + tn) / n, ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return {"agreement": po, "kappa": (po - pe) / (1 - pe), "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "share_vetoed": (tp + fp) / n}

mg_gt = av.margins_from_masks(w, gt, 0.5, G.GT_GEOM)
mg_ds = av.margins_from_masks(w, {t: av.gt_mask_to_2m(gt[t]) for t in w.token}, 2.0, G.PRED_GEOM)
mg_ds_flip = av.margins_from_masks(w, {t: av.gt_mask_to_2m(gt[t])[:, ::-1] for t in w.token}, 2.0, G.PRED_GEOM)
mg_ds_tr = av.margins_from_masks(w, {t: np.roll(av.gt_mask_to_2m(gt[t]), 1, axis=0) for t in w.token}, 2.0, G.PRED_GEOM)
for name, mg in (("GT raster 0.5 m", mg_gt), ("GT down-sampled to 2 m (pred-grid path)", mg_ds), ("same, y-flipped (control)", mg_ds_flip), ("same, shifted one 2 m cell in x (control)", mg_ds_tr)):
    out[name] = conf(av.vetoed(mg, av.MARGIN), dac0)
json.dump(out, open(HERE / "POSTHOC_PRED_FRAME_CROSSCHECK.json", "w"), indent=1)
for k, v in out.items():
    if isinstance(v, dict):
        print(f"{k:45s} agreement {v['agreement']:.4f} kappa {v['kappa']:.4f} sens {v['sensitivity']:.4f} spec {v['specificity']:.4f} vetoed {v['share_vetoed']:.4f}")
