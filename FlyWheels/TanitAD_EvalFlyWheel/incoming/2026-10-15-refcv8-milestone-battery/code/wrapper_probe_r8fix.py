"""wrapper_probe_r8fix.py -- the PROPOSED fix for the G0 wrapper clause on refcv8 (EvalFlyWheel, 2026-10-16).

DEFECT (instrument, not model): `wrapper_probe_r7.probe` neutralises the DDIM noise by patching ONLY
`torch.randn_like -> zeros` (wrapper_probe_r7.py:140-141). That covered refcv7, whose sampler drew
`eps = torch.randn_like(x0_n)` (refc.py:3030). refcv8 (cand_src set) draws its noise at three OTHER sites:
  * base eps   `torch.randn((b, _nb) + ...)`                        refc.py:3036  (global stream, generator-less)
  * extras eps `self.r8_gen.randn_eval(..., EVAL_SITE_EXTRA_EPS)`    refc.py:3042  (fresh generator seeded off the
                                                                     global RNG state, refcv8_conditioning.py:186-208)
  * source-anchor choice `gen.rand_eval(..., EVAL_SITE_SELECT_SOURCE)` refcv8_conditioning.py:445-446
None is neutralised, so every arm of the probe (plain / permuted / micro13 / micro31) samples DIFFERENT noise per row,
the `floor` and `wrapper` statistics are both noise-redraw deviations, and `wrapper <= 3*floor` becomes a coin toss.

FIX: neutralise all four sites for the duration of the probe (generator-carrying `torch.randn` calls -- fixed seeded
projections such as refc_v3.py:3122 -- are left untouched). Drop-in: call `probe(...)` from here instead of
`wrapper_probe_r7.probe(...)` (same signature) -- e.g. in g0_refcv8.py before `G.main()`:
`import wrapper_probe_r7, wrapper_probe_r8fix; wrapper_probe_r7.probe = wrapper_probe_r8fix.probe` -- or wrap the
existing call in `with neutralise_eval_draws(): ...`.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

import wrapper_probe_r7 as W

#: the UNPATCHED probe, captured at import so `W.probe = probe` (a monkey-patch from g0_refcv8.py) cannot recurse
_ORIG_PROBE = W.probe


@contextmanager
def neutralise_eval_draws():
    from tanitad.refs import refcv8_conditioning as r8c
    o_rl, o_rn = torch.randn_like, torch.randn
    o_re, o_ue = r8c.R8Generator.randn_eval, r8c.R8Generator.rand_eval

    def randn0(*size, **kk):
        if kk.get("generator") is not None:
            return o_rn(*size, **kk)
        shp = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else size
        return torch.zeros(tuple(shp), device=kk.get("device"), dtype=kk.get("dtype") or torch.float32)

    torch.randn_like = lambda x, *aa, **kk: torch.zeros_like(x)
    torch.randn = randn0
    r8c.R8Generator.randn_eval = lambda self, shape, device, site, dtype=torch.float32: torch.zeros(
        tuple(shape), device=device, dtype=dtype)
    r8c.R8Generator.rand_eval = lambda self, shape, device, site: torch.full(tuple(shape), 0.5, device=device)
    try:
        yield
    finally:
        torch.randn_like, torch.randn = o_rl, o_rn
        r8c.R8Generator.randn_eval, r8c.R8Generator.rand_eval = o_re, o_ue


def probe(*a, **kw):
    with neutralise_eval_draws():
        rec = _ORIG_PROBE(*a, **kw)
    rec["eval_draws_neutralised"] = ["torch.randn_like", "torch.randn (generator-less)", "R8Generator.randn_eval",
                                     "R8Generator.rand_eval"]
    return rec
