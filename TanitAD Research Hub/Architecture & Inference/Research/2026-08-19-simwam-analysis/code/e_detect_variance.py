"""Is a sub-prior arm DEGENERATE, or is it overfitting?

⛔ WHY THIS MATTERS. Four grid arms scored BELOW the closed-form `prior`, which
has two completely different explanations:

  (a) the head COLLAPSED to a near-constant map that is simply a worse constant
      than the prior — a training failure;
  (b) the head learned the prior's shape correctly and then added frame-specific
      variation that does not generalise — overfitting.

They call for opposite fixes, and AP cannot tell them apart. This decomposes the
out-of-fold prediction map into ACROSS-CELL variance (the prior's shape, shared
by every frame) and ACROSS-FRAME variance (the only component that can encode
what is in THIS frame), and correlates each arm's mean map with the true prior.

MEASURED 2026-08-21: every arm has frame-frac 0.69-0.83 with mean-map
correlation +0.91..+0.98 against the prior. So (b): the shape IS learned, and
the frame-specific part is net-harmful. It happens for RAW PIXELS too, which is
why no arm here can yet be read as a statement about a representation.
"""
from __future__ import annotations

import glob
import json

import numpy as np

import e_detect_prep as P
import e_trunk2_probe as T


def decompose(p: np.ndarray, occ: np.ndarray) -> dict:
    per_cell = p.mean(0)
    across_cell = float(per_cell.var())
    across_frame = float((p - per_cell[None, :]).var())
    return {
        "across_cell_var": round(across_cell, 4),
        "across_frame_var": round(across_frame, 4),
        "frame_fraction": round(across_frame / (across_cell + across_frame), 4),
        "corr_meanmap_vs_prior": round(
            float(np.corrcoef(per_cell, occ.mean(0))[0, 1]), 4),
    }


def main() -> None:
    keys = [tuple(k) for k in
            json.loads((P.FEAT / "keys.json").read_text(encoding="utf-8"))]
    ep = [k[0] for k in keys]
    occ = np.load(P.OUT / "occ.npy")
    folds = T.episode_folds(ep)
    pr = np.zeros_like(occ, np.float32)
    for k, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        pr[te] = occ[tr].mean(0)[None, :]

    out = {"_evidence_class": "MEASURED (ours)",
           "reading": "frame_fraction ~0 => near-CONSTANT map (collapse); "
                      "high frame_fraction with high corr_meanmap_vs_prior => "
                      "the prior shape was learned and the frame-specific part "
                      "is what fails to generalise (overfitting)",
           "arms": {"prior": decompose(pr, occ)}}
    for f in sorted(glob.glob("e_detect_pred_*.npy")):
        n = f.replace("e_detect_pred_", "").replace(".npy", "")
        out["arms"][n] = decompose(np.load(f), occ)
    for n, v in out["arms"].items():
        print(f"  {n:<16} across-cell {v['across_cell_var']:8.4f}  "
              f"across-FRAME {v['across_frame_var']:8.4f}  "
              f"frame-frac {v['frame_fraction']:6.3f}   "
              f"corr(mean-map, prior) {v['corr_meanmap_vs_prior']:+.3f}")
    with open("e_detect_variance.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\n-> e_detect_variance.json")


if __name__ == "__main__":
    main()
