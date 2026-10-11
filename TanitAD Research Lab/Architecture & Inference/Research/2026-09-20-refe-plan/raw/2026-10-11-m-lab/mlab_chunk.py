#!/usr/bin/env python3
"""M-LAB supplementary probe: the cost of SPLITTING a 64-proposal set into chunks (SPEC_P6 section 3.1: "a set may be split into
proposal chunks across workers for latency"). score_plans always simulates PDM-Closed (index 0) plus the M proposals, so a chunk
of M proposals costs  a + b*M  with a = the fixed part (the PDM-Closed simulation, scorer setup). Measures CPU per call for
M in (64, 32, 16, 8, 4, 1) on parts already in RAM, fits a + b*M, and checks chunk-invariance of the labels (each proposal is
normalised against PDM-Closed alone, so a chunk must reproduce the same pdms as the full set).

Run AFTER the main chain (it would perturb the N=1 primaries): python mlab_chunk.py  (navsim venv, same env as the driver).
"""
from __future__ import annotations

import json
import lzma
import os
import pickle
import sys
import time

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PKG, "refe"))
_P = psutil.Process()


def cpu():
    c = _P.cpu_times()
    return c.user + c.system


def main() -> int:
    import navsim_pdm_targets as PT
    PT._setup()
    plan = json.load(open(os.path.join(HERE, "plan", "cached_r3.json"), encoding="utf-8"))
    npz = np.load(os.path.join(HERE, "plan", "cached_r3.npz"))
    toks = [str(t) for t in npz["token"]][:40]
    props = npz["proposals"][:40].astype(np.float64)
    Ms = (64, 32, 16, 8, 4, 1)
    tot = {m: 0.0 for m in Ms}
    calls = {m: 0 for m in Ms}
    inv_max = 0.0
    for k, t in enumerate(toks):
        mc = pickle.loads(lzma.decompress(open(plan["paths"][t], "rb").read()))
        parts = (mc.trajectory, mc.ego_state, mc.observation, mc.centerline, list(mc.route_lane_ids), mc.drivable_area_map)
        if k == 0:
            PT.score_plans(parts, props[k])            # untimed warm-up on the first token
        full = PT.score_plans(parts, props[k])["pdms"]
        for m in Ms:
            nch = 64 // m
            c0 = cpu()
            for j in range(nch):
                out = PT.score_plans(parts, props[k][j * m:(j + 1) * m])
                if m == 16:
                    inv_max = max(inv_max, float(np.abs(np.asarray(out["pdms"]) - np.asarray(full)[j * m:(j + 1) * m]).max()))
            tot[m] += cpu() - c0                       # CPU to label the WHOLE 64-set as 64/m chunks
            calls[m] += nch
    n = len(toks)
    per_set = {m: tot[m] / n for m in Ms}                          # CPU-s to label one 64-set as 64/m chunks
    per_call = {m: tot[m] / calls[m] for m in Ms}
    x = np.array(Ms, float)
    y = np.array([per_call[m] for m in Ms])
    b, a = np.polyfit(x, y, 1)
    res = {"n_sets": n, "cpu_s_per_64set_as_chunks": per_set, "cpu_s_per_call": per_call,
           "fit_per_call_a_fixed_s": float(a), "fit_per_call_b_per_proposal_s": float(b),
           "chunk16_pdms_vs_full_max_abs_diff": inv_max,
           "note": "per_set[m] = CPU to label one 64-proposal set as 64/m chunks of m (parts already in RAM)"}
    json.dump(res, open(os.path.join(HERE, "chunk_probe.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
