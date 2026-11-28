"""SH5 C-REACH DIAGNOSIS driver, grid 5 (Research Lab, 2026-10-20).  `diag_pre3.main` (UNCHANGED, sha256-pinned by PREREG_GRID3) with two additions, both applied by swapping module
globals of diag_pre3 in THIS process (no pinned file is edited):
  * `DA` -> `diag_attn5`: adds the lever option "fp32": true (attention in fp32, autocast off inside SDPA); without it `apply_lever` delegates to `diag_attn.apply_lever` unchanged;
  * `make_learners` -> a wrapper that, for a learner entry with "adamw_wd": w, REPLACES the sealed Adam(lr, weight_decay=L.WD = 1e-4, L2-coupled) by AdamW(lr, betas, eps of the sealed
    recipe) with DECOUPLED weight decay w on the weight MATRICES only (ndim >= 2, excluding `t_emb`); biases, LayerNorms, `t_emb` and the lever's temperature parameter `s` get 0.
    Without "adamw_wd" the learner keeps the sealed optimizer exactly as diag_pre3 builds it.
PASS rule, ablations, checkpoints, attention diagnostic, the GPU lock (owner `sh5_diag`): all diag_pre3's, unchanged.
usage: PY diag_pre5.py --grid grid5.json --out OUTDIR [--max-wait S]"""
from __future__ import annotations

import sys

sys.path.insert(0, "C:/Users/Admin/par18/sh5diag/code")
import diag_pre3 as D3                # noqa: E402
import diag_attn5 as DA5              # noqa: E402

_make_learners3 = D3.make_learners


def wd_groups(model, wd: float) -> list:
    dec, nodec = [], []
    for n, p in model.named_parameters():
        (dec if (p.ndim >= 2 and not n.endswith("t_emb")) else nodec).append(p)
    return [{"params": dec, "weight_decay": float(wd)}, {"params": nodec, "weight_decay": 0.0}]


def make_learners5(D, run, provider, device):
    import torch
    specs, lns = _make_learners3(D, run, provider, device)
    for ln, lc in zip(lns, run["learners"]):
        if lc.get("adamw_wd") is not None:
            ln.opt = torch.optim.AdamW(wd_groups(ln.model, lc["adamw_wd"]), lr=lc["lr"], betas=D3.L.BETAS, eps=D3.L.EPS)
        ln.diag_opt = type(ln.opt).__name__
    return specs, lns


def install():
    D3.DA = DA5
    D3.make_learners = make_learners5


if __name__ == "__main__":
    install()
    sys.exit(D3.main())
