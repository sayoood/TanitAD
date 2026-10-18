#!/usr/bin/env python3
"""The REAL scorer behind `stepA.py`'s `score_fn`: P2-s0's three scorer modules loaded from `model_final.pt` and driven by the LANDED `REFe.score_trajectories` (refe/model.py:699).

Only `score_q_mlp`, `score_dec`, `score_head` are built (78 tensors; the 322 M-parameter trunk is never instantiated), so this runs on CPU in a few hundred MB.  The method called is the
landed one, BOUND to those modules (`types.MethodType`), not a re-implementation: the scoring path is the model's own.

`visual_ctx` is read from `ctx.npy` ([N, P, D] uint16 = raw bf16 bits, memory-mapped; `make_stepA_bundle.py`'s output) or a headerless `.u16`, one token at a time (`bind(n)`).  Numerics: `autocast='none'` is fp32 (exact on CPU);
`'bf16'` mimics the trainer's autocast.  A CPU result is NOT bit-comparable with logits the GPU produced: gate G2b reports the max |difference| to the banked logits and the runner records the device.

Needs torch; imported lazily so `stepA.py` stays numpy-only.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np

DEFAULT_REFE_DIR = os.environ.get("L1_REFE_DIR") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "2026-09-20-refe-plan", "refe"))


def _model_module(refe_dir):
    p = os.path.join(refe_dir, "model.py")
    if not os.path.exists(p):
        raise FileNotFoundError(f"refe/model.py not found under {refe_dir!r}: pass --refe-dir or set L1_REFE_DIR (no silent fallback)")
    spec = importlib.util.spec_from_file_location("refe_model_l1", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m                       # a @dataclass with `from __future__ import annotations` resolves its module through sys.modules
    spec.loader.exec_module(m)
    return m


def load_scorer(ckpt: str, refe_dir: str | None = None, device: str = "cpu"):
    """-> (ns, info): ns.score_trajectories(trajs [B,K,T,3], sctx [B,P,D]) -> [B,K,6], the landed method bound to the checkpoint's own modules"""
    import torch
    import torch.nn as nn
    M = _model_module(refe_dir or DEFAULT_REFE_DIR)
    ck = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["model"]
    depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("score_dec."))
    cfg = M.REFeConfig()
    ns = types.SimpleNamespace()
    ns.score_q_mlp = nn.Sequential(nn.Linear(cfg.horizon_steps * cfg.traj_dim, cfg.dec_width), nn.GELU(), nn.Linear(cfg.dec_width, cfg.dec_width))
    ns.score_dec = nn.ModuleList([M.CrossBlock(cfg.dec_width, cfg.dec_heads, cfg.dec_width) for _ in range(depth)])
    ns.score_head = nn.Linear(cfg.dec_width, cfg.n_score_components)
    for name in ("score_q_mlp", "score_dec", "score_head"):
        mod = getattr(ns, name)
        mod.load_state_dict({k[len(name) + 1:]: v.clone() for k, v in sd.items() if k.startswith(name + ".")}, strict=True)     # strict: a missing or extra key is an error
        mod.eval().to(device)
    ns.score_trajectories = types.MethodType(M.REFe.score_trajectories, ns)
    info = {"ckpt": ckpt, "depth": depth, "dec_heads": cfg.dec_heads, "dec_width": cfg.dec_width, "n_scorer_tensors": sum(1 for k in sd if k.split(".")[0] in ("score_q_mlp", "score_dec", "score_head")),
            "meta_step": ck.get("meta", {}).get("step")}
    return ns, info


def make_score_fn(ckpt: str, ctx_path: str, tokens, device: str = "cpu", refe_dir: str | None = None, autocast: str = "none", ctx_shape=None):
    """score_fn(set [K,20,3]) -> logits [K,6] float32 for the token last passed to `.bind(n)`"""
    import torch
    ns, info = load_scorer(ckpt, refe_dir, device)
    N = len(tokens)
    if str(ctx_path).endswith(".npy"):
        mm = np.load(ctx_path, mmap_mode="r")
        if mm.dtype != np.uint16 or mm.shape[0] != N:
            raise ValueError(f"ctx file {ctx_path}: dtype {mm.dtype} shape {mm.shape} does not match {N} tokens of uint16 bf16 bits")
    else:
        mm = np.memmap(ctx_path, dtype=np.uint16, mode="r", shape=tuple(ctx_shape) if ctx_shape else (N, 7680, 256))
    st = {"ctx": None}

    def bind(n):
        raw = np.array(mm[n], dtype=np.uint16)           # a writable copy (a memmap row is read-only)
        t = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)[None].to(device)
        st["ctx"] = t if autocast == "bf16" else t.float()

    def fn(S):
        if st["ctx"] is None:
            raise RuntimeError("score_fn called before bind(n): no visual_ctx loaded")
        x = torch.from_numpy(np.ascontiguousarray(S, dtype=np.float32))[None].to(device)
        with torch.no_grad():
            if autocast == "bf16":
                with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
                    out = ns.score_trajectories(x, st["ctx"])
            else:
                out = ns.score_trajectories(x, st["ctx"])
        return out[0].float().cpu().numpy()

    fn.bind, fn.info = bind, info
    return fn
