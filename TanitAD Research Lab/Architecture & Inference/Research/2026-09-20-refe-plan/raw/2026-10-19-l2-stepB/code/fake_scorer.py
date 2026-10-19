#!/usr/bin/env python3
"""A TINY stand-in for `real_scorer` (same two entry points: `load_scorer`, `make_score_fn`) so the Step B plumbing, the head installer and the WHOLE of g_reach_stepb.py can be tested in seconds without the
1.3 GB checkpoint.  It is set-conditional like the real scorer (each proposal's token sees the mean token of the set) and ends in the same `score_head = nn.Linear(256, 6)` with EP at row 2.
Used by test_stepB.py ONLY; it carries no claim about P2."""
from __future__ import annotations

import types

import numpy as np


class FakeScorerModule:
    def __init__(self, seed: int = 1234):
        self.seed = seed

    def load_scorer(self, ckpt, refe_dir=None, device="cpu"):
        import torch
        import torch.nn as nn
        g = torch.Generator().manual_seed(self.seed)
        ns = types.SimpleNamespace()
        ns.enc = nn.Sequential(nn.Linear(60, 256), nn.Tanh())
        ns.mix = nn.Linear(256, 256)
        ns.score_head = nn.Linear(256, 6)
        with torch.no_grad():
            for m in (ns.enc[0], ns.mix, ns.score_head):
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.25)
                m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)
        for m in (ns.enc, ns.mix, ns.score_head):
            m.eval()

        def score_trajectories(trajs, sctx):
            tok = ns.enc(trajs.reshape(*trajs.shape[:2], -1))                       # [B,K,256]
            tok = tok + ns.mix(tok.mean(dim=1, keepdim=True))                         # set-conditional
            return ns.score_head(tok)

        ns.score_trajectories = score_trajectories
        return ns, {"ckpt": ckpt, "depth": 0, "n_scorer_tensors": 6, "meta_step": 0}

    def make_score_fn(self, ckpt, ctx_path, tokens, device="cpu", refe_dir=None, autocast="none", ctx_shape=None):
        import torch
        ns, info = self.load_scorer(ckpt, refe_dir, device)                           # attribute lookup: a wrapper installed on this instance is honoured, as with the real module
        st = {"bound": None}

        def bind(n):
            st["bound"] = int(n)

        def fn(S):
            if st["bound"] is None:
                raise RuntimeError("score_fn called before bind(n)")
            x = torch.from_numpy(np.ascontiguousarray(S, dtype=np.float32))[None]
            with torch.no_grad():
                out = ns.score_trajectories(x, None)
            return out[0].float().cpu().numpy()

        fn.bind, fn.info = bind, info
        return fn
