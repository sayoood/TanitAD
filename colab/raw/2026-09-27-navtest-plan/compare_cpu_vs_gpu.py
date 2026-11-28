"""Cross-BACKEND numerical deviation of REFe's forward, MEASURED: the same snapshot (016), the same 4 tokens, the same
unchanged seam script -- once on the dev box's RTX 4060 (the landed E-6 dump) and once on the dev box's CPU
(CUDA_VISIBLE_DEVICES=-1, `--device cpu`, run for this plan; outputs in the session scratchpad).

The GPU path uses cuDNN TF32 for the patch-embedding conv (torch default `cudnn.allow_tf32=True` on Ampere+) and
cuBLAS fp32 GEMMs + SDPA; the CPU path is plain fp32. A Colab GPU (L4 = Ada like the 4060; G4 = Blackwell) differs
from the 4060 only in kernels and reduction order, NOT in the TF32 rounding, so this CPU-vs-GPU gap is a
conservative (upper-side) scale for what a GPU-to-GPU move can do. Writes cpu_vs_gpu_deviation.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GPU = "D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep016/proposals.npz"
SEAM_GPU = "D:/Projects/TanitAD/data/refe_navtest/seams/refe_sub200_ep016.npz"
S = ("C:/Users/Admin/AppData/Local/Temp/claude/D--Projects-TanitAD/91effc67-8c1e-4b66-9a63-341a3109dfc1/"
     "scratchpad/cpu_vs_gpu")


def agg(lg):
    s = 1.0 / (1.0 + np.exp(-lg.astype(np.float32)))
    return s[..., 0] * s[..., 1] * (5 * s[..., 2] + 5 * s[..., 3] + 2 * s[..., 4]) / np.float32(12.0)


def main() -> int:
    g = np.load(GPU)
    c = np.load(f"{S}/props_cpu_ep016.npz")
    sg = np.load(SEAM_GPU)
    sc = np.load(f"{S}/seam_cpu_ep016.npz")
    gi = {t: i for i, t in enumerate(g["token"])}
    sgi = {t: i for i, t in enumerate(sg["token"])}
    rows = []
    for j, t in enumerate(c["token"]):
        i = gi[t]
        dl = np.abs(c["logits"][j].astype(np.float64) - g["logits"][i].astype(np.float64))
        dp = np.abs(c["proposals"][j].astype(np.float64) - g["proposals"][i].astype(np.float64))
        ag, ac = agg(g["logits"][i]), agg(c["logits"][j])
        srt = np.sort(ag)
        seam_d = float(np.abs(sc["poses"][j].astype(np.float64) - sg["poses"][sgi[t]].astype(np.float64)).max())
        rows.append({"token": str(t), "pick_gpu": int(g["pick"][i]), "pick_cpu": int(c["pick"][j]),
                     "pick_agrees": bool(g["pick"][i] == c["pick"][j]),
                     "gpu_margin_top1_top2": float(srt[-1] - srt[-2]),
                     "max_abs_dlogit": float(dl.max()), "median_abs_dlogit": float(np.median(dl)),
                     "max_abs_daggregate": float(np.abs(ac.astype(np.float64) - ag.astype(np.float64)).max()),
                     "max_abs_dpose_all_64_proposals": float(dp.max()),
                     "max_abs_dpose_executed_seam_row": seam_d})
    rep = {"what": "dev-box CPU fp32 forward vs the landed dev-box RTX 4060 dump, snapshot 016, same tokens",
           "tokens": rows,
           "max_abs_dlogit": max(r["max_abs_dlogit"] for r in rows),
           "max_abs_daggregate": max(r["max_abs_daggregate"] for r in rows),
           "max_abs_dpose_all_proposals_m": max(r["max_abs_dpose_all_64_proposals"] for r in rows),
           "picks_agree": f"{sum(r['pick_agrees'] for r in rows)}/{len(rows)}"}
    json.dump(rep, open(os.path.join(HERE, "cpu_vs_gpu_deviation.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
