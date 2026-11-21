#!/usr/bin/env python3
"""H-d (path/precision): can the numeric path difference between the two reads explain any of the gap?  CPU, read-only.

The held-out dump is produced INSIDE the trainer: trunk under bf16 autocast, so the head read bf16 ctx (and was trained on bf16 ctx).  The navtest read applies the head to the PLANNER path's fp32 ctx
(eval/aux_navtest_forward.py `--head-ctx planner`, the default; the S1 read used it).  MEASURED upstream on 32 tokens (raw/2026-10-13-aux-navtest-forward/RESULT.md section 4): ctx rel L2 between the two paths
1.05 / 1.27 / 1.46 % (min / mean / max); the effect on a TRAINED head was marked UNVERIFIED.  Here: apply the TRAINED H1-s0 head to the banked fp32 navtest ctx under perturbations at or above that size and read the
change in P1 [0, 40) on a fixed random subset of tokens.  If a 3 % relative-L2 perturbation moves P1 by far less than the 0.135 gap, the precision path cannot be the gap.
Variants (same subset, same labels): A control fp32 | B ctx rounded to bf16, head fp32 | C ctx bf16 + head under bf16 autocast (the trainer's head regime) | D..F ctx + Gaussian noise at 1.5 / 3 / 6 % rel L2 (3 draws each).
NOT done: the trunk itself is not re-run (needs the GPU and the u400 checkpoint); the noise is isotropic, the true bf16-trunk deviation is structured.  So this BOUNDS sensitivity; it does not replace
`aux_navtest_forward.py --head-ctx trainer`, the one-command GPU experiment that does."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shf_common as C  # noqa: E402
import shf_arrays as SA  # noqa: E402
import aux_navtest_forward as FW  # noqa: E402
import torch  # noqa: E402

ARM = "H1-s0"
N_SUB = int(os.environ.get("SHF_NSUB", "200"))
Dn = SA.domain("navtest", (ARM,))
rng = np.random.default_rng(20261016)
sub = np.sort(rng.choice(Dn["n"], N_SUB, replace=False))
head, info = FW.load_aux_head(f"{C.S1CACHE}/arms/{ARM}/aux_drv.pt")
b = SA.band()[None]
variants = ["A_fp32", "B_ctx_bf16", "C_ctx_bf16_head_autocast", "D_noise_1.5pct", "E_noise_3pct", "F_noise_6pct"]
logits = {v: [] for v in variants}
nz = {"D_noise_1.5pct": 0.015, "E_noise_3pct": 0.03, "F_noise_6pct": 0.06}
gen = torch.Generator().manual_seed(7)
t0 = time.time()
rel = []
for n, j in enumerate(sub):
    ctx = torch.from_numpy(np.load(FW.cache_path(C.CTX_DIR, Dn["tokens"][j])))[None]
    logits["A_fp32"].append(FW.apply_head(head, ctx)[0][0])
    cb = ctx.to(torch.bfloat16).float()
    rel.append(float((cb - ctx).norm() / ctx.norm()))
    logits["B_ctx_bf16"].append(FW.apply_head(head, cb)[0][0])
    logits["C_ctx_bf16_head_autocast"].append(FW.apply_head(head, ctx.to(torch.bfloat16), autocast=True)[0][0])
    for v, s in nz.items():
        e = torch.randn(ctx.shape, generator=gen)
        e = e * (s * ctx.norm() / e.norm())
        logits[v].append(FW.apply_head(head, ctx + e)[0][0])
    if (n + 1) % 50 == 0:
        print(f"{n + 1}/{len(sub)} {time.time() - t0:.0f}s", flush=True)
tm = Dn["TM"][sub] & b


def iou(L):
    pm = (np.stack(L) > 0)
    return float((tm & pm & b).sum() / ((tm | pm) & b).sum()), pm


res = {"n_subset": int(N_SUB), "bf16_rounding_rel_l2_mean": float(np.mean(rel)), "subset_seed": 20261016}
base, pm0 = iou(logits["A_fp32"])
res["A_fp32"] = base
# identity check: the subset IoU of the control equals the IoU computed from the regenerated pred file on the same rows (the file was written by the same head and ctx)
pm_file = Dn["PM"][ARM][sub]
res["control_equals_regenerated_file_mask_cells_differing"] = int((pm0 != pm_file).sum())
for v in variants[1:]:
    val, pmv = iou(logits[v])
    res[v] = dict(iou=val, delta_vs_fp32=val - base, cells_flipped_share=float((pmv != pm0).mean()))
    print(v, res[v], flush=True)
res["verdict_scale"] = "the gap to explain is 0.135; the largest perturbation here is 6 % rel L2, 4x the measured 1.46 % upstream maximum"
C.jdump(res, os.path.join(C.HERE, "e1_precision_sensitivity.json"))
print("ZZE1_DONE")
