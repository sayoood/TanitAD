"""analyze.py -- TREEDRIFT: pairwise comparison of the per-window plans / RNG states / ADE from F:/treedrift/out/*.  CPU only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("F:/treedrift/out")


def load(name):
    p = OUT / name
    if not (p / "META.json").is_file():
        return None
    z = np.load(p / "records.npz", allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["ade"] = np.load(p / "ade_per_window.npy")
    d["meta"] = json.loads((p / "META.json").read_text(encoding="utf-8"))
    return d


def key(d):
    return {(int(a), int(b)): i for i, (a, b) in enumerate(zip(d["e_i"], d["t"]))}


def pair(a, b, label=""):
    A, B = load(a), load(b)
    if A is None or B is None:
        return {"pair": f"{a} vs {b}", "missing": [n for n, x in ((a, A), (b, B)) if x is None]}
    ka, kb = key(A), key(B)
    common = sorted(set(ka) & set(kb))
    ia = np.array([ka[c] for c in common])
    ib = np.array([kb[c] for c in common])
    dt = np.abs(A["traj"][ia].astype(np.float64) - B["traj"][ib].astype(np.float64)).max(axis=(1, 2))
    dz = np.abs(A["z"][ia].astype(np.float64) - B["z"][ib].astype(np.float64)).max()
    cu_eq = np.array([np.array_equal(A["cuda_state"][i], B["cuda_state"][j]) for i, j in zip(ia, ib)])
    cp_eq = np.array([np.array_equal(A["cpu_state"][i], B["cpu_state"][j]) for i, j in zip(ia, ib)])
    return {"pair": f"{a} vs {b}", "n": len(common), "max_abs_traj_m": float(dt.max()), "median_abs_traj_m": float(np.median(dt)),
            "n_windows_traj_bytes_differ": int((dt > 0).sum()), "n_windows_gt_1e-4": int((dt > 1e-4).sum()), "n_windows_gt_1e-5": int((dt > 1e-5).sum()),
            "z_tac_max_abs": float(dz), "n_windows_cuda_state_equal_before_fwd": int(cu_eq.sum()), "n_windows_cpu_state_equal_before_fwd": int(cp_eq.sum()),
            "mean_ade_2s": [A["meta"]["mean_ade_2s_os"], B["meta"]["mean_ade_2s_os"]],
            "mean_paired_dADE": float((A["ade"][ia] - B["ade"][ib]).mean()), "sd_paired_dADE": float((A["ade"][ia] - B["ade"][ib]).std(ddof=1))}


if __name__ == "__main__":
    pairs = [a.split(",") for a in sys.argv[1:]]
    for a, b in pairs:
        print(json.dumps(pair(a, b)))
