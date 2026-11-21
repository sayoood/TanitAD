#!/usr/bin/env python3
"""bank_rollouts.py -- bank the SMOOTH-T1 rollout outputs into the package as FEW large files (the 141 per-episode dump files would cost 1 MiB each on exFAT): per run one concatenated dump npz
(same keys as the per-episode files, concatenated in file order), the ticks npz (main runs only), the result JSON and the done marker.  Main runs = the card's five; replicate / diagnostic runs: dump + JSON + done only."""
import glob
import json
import os
import shutil
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

S = "F:/TanitAD-scratch/smooth_t1"
DST = "E:/Projects/TanitAD-artifacts/dagland/smooth_t1/raw/rollouts"
JOBS = [("out", t, True) for t in ("K1OFF_DAGOACC_s0", "SMOOTH_DAGOACC_s0", "SMOOTH_DAGOACC_s1", "SMOOTH_SOUP_s0", "SMOOTH_SOUP_s1")] + \
       [("out_t2", t, False) for t in ("SMOOTH_DAGOACC_s0", "SMOOTH_DAGOACC_s1", "SMOOTH_SOUP_s0", "SMOOTH_SOUP_s1")] + [("out_diag", t, False) for t in ("K1U_DAGOACC_s0", "K1P3_DAGOACC_s0")]
for sub, tag, ticks in JOBS:
    d = f"{DST}/{sub}/{tag}"
    os.makedirs(d, exist_ok=True)
    fs = sorted(glob.glob(f"{S}/{sub}/{tag}/dump_{tag}/ep*.npz"))
    assert len(fs) == 141, (sub, tag, len(fs))
    zs = [np.load(f) for f in fs]
    np.savez_compressed(f"{d}/dump_{tag}_concat.npz", **{k: np.concatenate([z[k] for z in zs]) for k in zs[0].files})
    for f in glob.glob(f"{S}/{sub}/{tag}/t1v8f_*.json") + [f"{S}/{sub}/done_{tag}.txt"]:
        shutil.copy2(f, d)
    if ticks:
        shutil.copy2(f"{S}/{sub}/{tag}/ticks/ticks_{tag}.npz", d)
    print("banked", sub, tag)
