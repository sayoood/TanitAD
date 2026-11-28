#!/usr/bin/env python3
"""Bit-for-bit digests of the UNAMENDED Stage 0 pipeline on small synthetic worlds (amendment A1, identity at S = 1).

`digests(mod)` runs the arm, the exclusion-deleted arm and the shuffled-label training control of ANY stage0 module through the integer-seed API that both the sealed code and the amended
code expose, and returns sha256 digests of (a) every JSON-able statistic and (b) the raw bytes of every cached composite score.  The GOLDEN values pinned in `test_stage0.py` were produced by
running this file against the SEALED `spess0/code/stage0.py` (sha256 71a9ccf5...), before any line of the amendment existed:

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 python golden_digests.py C:/Users/Admin/par18/spess0/code/stage0.py

The amended module at S = 1 must return the same digests, which is what "S = 1 reproduces the unamended pipeline bit for bit" means.  Reads no real token.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402


def load_module_from(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _h(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def _cache_hash(cache) -> str:
    h = hashlib.sha256()
    for key in sorted(cache.agg):
        h.update(repr(key).encode())
        h.update(np.ascontiguousarray(cache.agg[key]).tobytes())
        h.update(np.ascontiguousarray(cache.agg_a[key]).tobytes())
        h.update(np.ascontiguousarray(cache.agg_p[key]).tobytes())
    return h.hexdigest()


def _arm_digest(S0, a: dict) -> dict:
    audit = [{k: v for k, v in row.items() if k != "seed"} for row in a["fit"]["audit"]]      # the amendment ANNOTATES each audit row with its training seed; nothing else in a row changed
    stats = {"gaps": a["gaps"], "swap_null": a["swap_null"], "seen_minus_oof": a["seen_minus_oof"], "levels": a["levels"], "fit_audit": audit}
    return {"stats": _h(S0.jsonable(stats)), "scores": _cache_hash(a["cache"])}


def digests(S0, synth) -> dict:
    """S0 / synth: the stage0 module under test and the synth module bound to it.  Integer seeds only (the API both versions share)."""
    out = {}
    rec = S0.Recipe(steps=40)
    for tag, (seed_w, kappa) in {"null": (4, 0.0), "planted": (3, 2.0)}.items():
        w = synth.make_world(seed=seed_w, kappa=kappa, D=32, frac=0.3)
        d = S0.make_design(w)
        idx = S0.boot_indices(w.S)
        for s in (0, 1):
            out[f"{tag}_seed{s}"] = _arm_digest(S0, S0.run_arm(w, d, rec, s, idx))
        if tag == "planted":
            m = S0.run_arm(w, d, rec, 0, idx, fit=S0.fit_all_unexcluded(w, d, rec, 0))
            out["planted_exclusion_deleted_seed0"] = _arm_digest(S0, m)
            sh = S0.control_shuffle_world(w, d, S0.Recipe(steps=20), idx, seeds=(9101, 9102))
            out["planted_shuffle_trained_2worlds"] = _h(S0.jsonable(sh))
    return out


if __name__ == "__main__":
    path = sys.argv[1]
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    S0 = load_module_from(path, "stage0")                       # synth.py does `import stage0 as S0`, so register under that name
    import synth  # noqa: E402
    print(json.dumps({"stage0_py": os.path.abspath(path), "stage0_sha256_lf": S0.sha256_file_lf(path), "digests": digests(S0, synth)}, indent=1))
