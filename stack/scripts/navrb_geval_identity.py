#!/usr/bin/env python3
"""navrb_geval_identity.py -- PREREG_V8F_NAVRB2 G-EVAL identity at the evaluator's OWN batch size (the WH K-ID lesson: batch 64 vs 256 moved fp32 round-off past 1e-5, so the identity is read where the evaluator runs).

  pair    CKPT OUTDIR   writes ``zon.pt`` (the N1R / S1R checkpoint with ``e_nav_rb.weight`` set to exactly 0.0, cfg nav_rb on) and ``zoff.pt`` (the same weights with the ``e_nav_rb.*`` keys dropped and cfg nav_rb off).
                        The two differ in nothing the model computes: a zero channel weight adds exactly 0.0 to the nav token.
  compare A.npz B.npz   reads two evaluator dumps (``eval_<tag>.npz`` of the T0 tool, or a T1 tick dump) and requires every float / int array bit-equal (names the first difference).

Pod use (done by the chain's post step, at the tool's default batch: T0 --batch 64, T1 --batch 32; never a different one):
  python v8f_mini_eval_rb.py run --ckpt zon.pt  --nav-rb on  --nav-rb-npz T --nav-rb-sha256 S ... --tag zon  --out-dir D
  python v8f_mini_eval_rb.py run --ckpt zoff.pt ...                                               --tag zoff --out-dir D
  python navrb_geval_identity.py compare D/eval_zon.npz D/eval_zoff.npz
The T1 pair is the same with ``v8f_t1_eval_rb.py`` / ``v8f_t1_eval.py`` and ``--dump-ticks``.
"""
from __future__ import annotations

import os
import sys

import numpy as np


def pair(ckpt, outdir):
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if ck["cfg"].get("nav_rb") != "on" or "e_nav_rb.weight" not in ck["model"]:
        raise SystemExit("REFUSED: the checkpoint has no nav_rb channel (cfg nav_rb != on or no e_nav_rb.weight)")
    os.makedirs(outdir, exist_ok=True)
    on = {**ck, "cfg": dict(ck["cfg"]), "model": dict(ck["model"])}
    on["model"]["e_nav_rb.weight"] = torch.zeros_like(ck["model"]["e_nav_rb.weight"])
    off = {**ck, "cfg": {**ck["cfg"], "nav_rb": "off"}, "model": {k: v for k, v in on["model"].items() if not k.startswith("e_nav_rb.")}}
    torch.save(on, os.path.join(outdir, "zon.pt"))
    torch.save(off, os.path.join(outdir, "zoff.pt"))
    return {"zon": os.path.join(outdir, "zon.pt"), "zoff": os.path.join(outdir, "zoff.pt"), "dropped_keys": sorted(k for k in on["model"] if k.startswith("e_nav_rb."))}


def compare(pa, pb):
    """[] iff every array of the two dumps is bit-equal (same names, shapes, dtypes); else the differences named."""
    a, b = np.load(pa, allow_pickle=False), np.load(pb, allow_pickle=False)
    bad = []
    if sorted(a.files) != sorted(b.files):
        bad.append(f"array names differ: only A {sorted(set(a.files) - set(b.files))}, only B {sorted(set(b.files) - set(a.files))}")
    for k in sorted(set(a.files) & set(b.files)):
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            bad.append(f"{k}: shape/dtype {x.shape}/{x.dtype} vs {y.shape}/{y.dtype}")
        elif x.dtype.kind in "fiub" and not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            bad.append(f"{k}: not bit-equal, max abs diff {np.nanmax(d):.3e}")
    return bad


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if len(a) == 3 and a[0] == "pair":
        print(pair(a[1], a[2]))
        return 0
    if len(a) == 3 and a[0] == "compare":
        bad = compare(a[1], a[2])
        print("PASS" if not bad else "REFUSED: " + "; ".join(bad))
        return 0 if not bad else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
