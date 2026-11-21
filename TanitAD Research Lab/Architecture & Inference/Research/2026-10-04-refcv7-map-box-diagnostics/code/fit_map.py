"""Fit the map decision parameters on TRAIN-DIAG ONLY (SPEC.md sec. 2, M-c iii / iv) -> fit JSON for the EVAL pass.

* tau_phat_logit[c], tau_q_logit[c]: the lower bin edge (logit space) that maximises pooled TRAIN IoU of the
  one-vs-rest mask {logit(s_c) >= tau} over all bands, from the exact TRAIN histograms.
* delta[c] (delta_0 = 0): coordinate ascent (grid -3..+6 step 0.25, 3 sweeps, classes 1..7 in order) of the mean
  IoU over classes 1..7, all bands pooled, on the uniform 1 % TRAIN cell subsample, for the partition rule
  argmax_c (z_c - ln w_c + delta_c). Ties keep the earlier (smaller |delta|-first is NOT assumed: the grid is
  scanned in ascending order and only a STRICT improvement moves delta).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_metrics as dm  # noqa: E402


def confusion_iou(z, y, offs, classes=range(8)):
    pred = (z + offs).argmax(1)
    cm = torch.bincount(y * 8 + pred, minlength=64).view(8, 8).double()
    tp = cm.diag()
    union = cm.sum(1) + cm.sum(0) - tp
    iou = torch.where(union > 0, tp / union.clamp_min(1), torch.full_like(tp, float("nan")))
    return iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-acc", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = torch.load(a.train_acc, map_location="cpu", weights_only=False)
    edges = dm.hist_edges()
    fit = {"source": str(a.train_acc), "split": "TRAIN-DIAG", "classes": list(dm.CLASS_KEYS)}
    for score in ("phat", "q"):
        h = d["hist"][score].sum(dim=1).numpy()          # [8, 2, NB] over bands
        taus, ious = [], []
        for c in range(8):
            i, best = dm.best_threshold(h[c, 0], h[c, 1])
            taus.append(float(edges[i]))
            ious.append(best)
        fit[f"tau_{score}_logit"] = taus
        fit[f"tau_{score}_prob"] = [float(1 / (1 + np.exp(-t))) for t in taus]
        fit[f"train_iou_at_tau_{score}"] = ious
    sub = d["sub"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = sub["z"].to(dev).float()
    y = sub["y"].to(dev).long()
    lw = torch.log(d["class_weight"].float()).to(dev)
    grid = np.arange(-3.0, 6.0 + 1e-9, 0.25)
    delta = np.zeros(8)

    def obj(dl):
        iou = confusion_iou(z, y, (-lw + torch.tensor(dl, dtype=torch.float32, device=dev)).view(1, -1))
        return float(torch.nanmean(iou[1:])), iou
    base, base_iou = obj(delta)
    hist = [{"sweep": 0, "objective": base}]
    for sweep in range(3):
        for c in range(1, 8):
            best_v, best_o = delta[c], obj(delta)[0]
            for v in grid:
                trial = delta.copy()
                trial[c] = v
                o = obj(trial)[0]
                if o > best_o + 1e-12:
                    best_v, best_o = v, o
            delta[c] = best_v
        hist.append({"sweep": sweep + 1, "objective": obj(delta)[0], "delta": delta.tolist()})
    final, final_iou = obj(delta)
    raw_o, raw_iou = (lambda r: (float(torch.nanmean(r[1:])), r))(confusion_iou(z, y, torch.zeros(1, 8, device=dev)))
    fit.update({"delta": delta.tolist(), "delta_grid": [float(grid[0]), float(grid[-1]), 0.25],
                "objective": "mean IoU over classes 1..7, all bands pooled, 1 % TRAIN cell subsample",
                "n_subsample_cells": int(y.numel()),
                "subsample_iou": {"pc": base_iou.tolist(), "raw": raw_iou.tolist(), "off": final_iou.tolist()},
                "subsample_mean_iou_1to7": {"pc": base, "raw": raw_o, "off": final},
                "ascent": hist})
    Path(a.out).write_text(json.dumps(fit, indent=1))
    print(f"[fit] delta {np.round(delta, 2).tolist()} objective pc {base:.4f} raw {raw_o:.4f} off {final:.4f}",
          flush=True)
    print(f"[fit] tau_phat_prob {np.round(fit['tau_phat_prob'], 4).tolist()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
