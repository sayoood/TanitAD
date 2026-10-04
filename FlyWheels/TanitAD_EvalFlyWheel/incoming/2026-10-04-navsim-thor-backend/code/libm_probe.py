#!/usr/bin/env python3
"""Cross-platform float64 transcendental probe (any numpy python). Run on BOTH machines, diff the output.

Evaluates np.exp / np.cos / np.sin / np.arctan2 and math.exp / math.cos on a FIXED deterministic grid
(PCG64 seed 0) over the ranges the NavSim scorer feeds them, and writes the SHA-256 of each result's
bytes plus the first 2,000 values as hex, so the two machines can be compared value by value. Also
records numpy's runtime CPU-dispatch features (which SIMD kernels numpy may pick).

    python libm_probe.py --n 1000000 --out probe_<host>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--keep", type=int, default=20000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.Generator(np.random.PCG64(0))
    x_exp = -rng.random(a.n) * 60.0                     # Gaussian-kernel exponents are <= 0
    x_ang = (rng.random(a.n) * 2.0 - 1.0) * np.pi
    # uniform only: standard_normal's ziggurat itself calls exp/log in its tail, i.e. is platform-dependent
    y = (rng.random(a.n) * 2.0 - 1.0) * 3.0
    xx = (rng.random(a.n) * 2.0 - 1.0) * 3.0
    res = {"np.exp": np.exp(x_exp), "np.cos": np.cos(x_ang), "np.sin": np.sin(x_ang),
           "np.arctan2": np.arctan2(y, xx),
           "math.exp": np.array([math.exp(v) for v in x_exp[:a.keep]]),
           "math.cos": np.array([math.cos(v) for v in x_ang[:a.keep]])}
    try:
        from numpy.core._multiarray_umath import __cpu_features__ as feats
        feats = {k: bool(v) for k, v in feats.items() if v}
    except Exception:                                                     # noqa: BLE001
        feats = None
    out = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
           "numpy": np.__version__, "cpu_features_numpy_sees": feats, "n": a.n,
           "inputs_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in (("x_exp", x_exp), ("x_ang", x_ang), ("y", y), ("x", xx))},
           "sha256": {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in res.items()},
           "head_hex": {k: [float(t).hex() for t in v[:a.keep]] for k, v in res.items()}}
    json.dump(out, open(a.out, "w", encoding="utf-8"))
    print(json.dumps({"inputs": {k: v[:12] for k, v in out["inputs_sha256"].items()}, **{k: v[:16] for k, v in out["sha256"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
