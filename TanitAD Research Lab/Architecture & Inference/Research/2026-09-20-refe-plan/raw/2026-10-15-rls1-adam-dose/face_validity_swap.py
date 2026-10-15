#!/usr/bin/env python3
"""Negative control for face_validity.py (post-hoc, deciding nothing): same V statistic, but each scene is scored against ANOTHER scene's cached visual_ctx (cyclic shift by 1).
If V stays high the statistic is insensitive to sctx and 'ADMISSIBLE' would be weak evidence; the absolute logit difference is the sharper read."""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import adam_common as AC
SF = AC.load_trainer(); a, _ = AC.r0_args(SF); holder, cfg = AC.build_scorer(SF, a)
pv = AC.Pool(AC.prepare_pool(SF, a, cfg, "val")).load_visual_all(); tr = AC.Trainer(SF, a, holder, pv)
z = np.load("C:/Users/Admin/rls1b_dl/r0/final/scores_step0_base.npz", allow_pickle=False)
idx = {(str(z["log"][i]), str(z["token"][i]), int(z["step"][i]), int(z["rank"][i])): i for i in range(len(z["token"]))}
banked = np.stack([z["scores"][idx[k]] for k in pv.keys]).astype(np.float64)
n = len(pv.keys); out = {}
for tag, shift in (("true_ctx", 0), ("swapped_ctx_shift1", 1)):
    mine = []
    for i in range(n):
        cxy = torch.from_numpy(pv.cxy[[i]]); sctx = AC.bf16bits_to_f32(pv.vis[(i + shift) % n])[None]
        with torch.no_grad(): mine.append(tr.base(cxy, sctx).float().numpy()[0])
    mine = np.stack(mine).astype(np.float64)
    r = [np.mean([np.corrcoef(mine[k, :, j], banked[k, :, j])[0, 1] for k in range(n) if mine[k, :, j].std() > 0]) for j in range(7)]
    out[tag] = {"V_median_over_heads": float(np.median(r)), "mean_abs_logit_diff_per_head": [float(np.abs(mine[..., j] - banked[..., j]).mean()) for j in range(7)],
                "mean_abs_logit_diff_all": float(np.abs(mine - banked).mean())}
    print(tag, {k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in out[tag].items()})
json.dump(out, open(os.path.join(HERE, "face_validity_swap.json"), "w"), indent=1)
