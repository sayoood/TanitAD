"""bank_rollouts_a1.py -- bank the nine 16-thread rollouts of AMENDMENT_1 (X: four sg72; Y': four off; Y'': one off) into raw_a1/rollouts/out_m16 as FEW large files: per run one concatenated dump npz
(same keys as the per-episode files, concatenated in file order), the ticks npz, the result JSON and the done marker (derived from bank_rollouts.py)."""
import glob
import json
import os
import shutil
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

S = "F:/TanitAD-scratch/smooth_t1"
DST = "E:/Projects/TanitAD-artifacts/dagland/smooth_t1/raw_a1/rollouts"
JOBS = [("out_m16", t, True) for t in ("K1OFF_DAGOACC_s0", "OFFREP_DAGOACC_s0", "OFF_DAGOACC_s1", "OFF_SOUP_s0", "OFF_SOUP_s1", "SMOOTH_DAGOACC_s0", "SMOOTH_DAGOACC_s1", "SMOOTH_SOUP_s0", "SMOOTH_SOUP_s1")]
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
