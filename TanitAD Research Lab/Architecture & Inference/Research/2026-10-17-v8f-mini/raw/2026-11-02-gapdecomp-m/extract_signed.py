#!/usr/bin/env python3
"""extract_signed.py -- per-window SIGNED plan-vs-GT behaviour of one arm and of P10 on the landed windows (zero training, CPU only; Beside block of PREREG_FROZEN.md).

    python extract_signed.py --arm CTRL --seed 0 --out F:/TanitAD-scratch/gapdecomp_m/signed/signed_CTRL_s0.npz

Reads the arm's dump (the `cl` track of the arm, the `fl` track of P10, the GT track `g`; dump_dir from the landed array's meta.json), checks it is window-aligned with the landed
arrays, and writes per-window arrays computed with the instrument's own geometry (four_families._seq_geometry, dt 0.2): mean speed over the 30 steps; mean |kappa| and the mean
signed kappa error over the steps where the plan and the GT are both pair-valid (the mask of comps()).  It computes no subclass number and no share.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

ART = "E:/Projects/TanitAD-artifacts"
LAND = f"{ART}/dagland/gapdecomp/raw/arrays"
TREE = f"{ART}/dagt1_tree_b43c7a9a"
DT = 0.2


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_dir(d):
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    assert len(fs) == 141, (d, len(fs))
    zs = [np.load(f) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "fl", "v0", "eid", "clip_index", "ws")}


def curv_stats(torch, P, Q):
    """-> (mean |kappa| of the plan, mean |kappa| of the GT, mean signed (plan - GT)) per window over the steps where both are pair-valid; NaN where none."""
    bp = P["pair_valid"] & Q["pair_valid"]
    c = bp.sum(1)
    def mm(x):
        s = torch.where(bp, x, torch.zeros_like(x)).sum(1)
        return torch.where(c > 0, s / c.clamp_min(1), torch.full_like(s, float("nan"))).numpy().astype(np.float64)
    kp, kq = P["curvature"], Q["curvature"]
    return mm(kp.abs()), mm(kq.abs()), mm(kp - kq)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("CTRL", "SOUP"))
    ap.add_argument("--seed", required=True, type=int, choices=(0, 1))
    ap.add_argument("--out", required=True)
    a_ = ap.parse_args(argv)
    import torch
    os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", os.path.join(TREE, "stack").replace("/", os.sep))
    sys.path.insert(0, os.path.join(TREE, "taniteval"))
    from taniteval import four_families as FF
    meta = json.load(open(f"{LAND}/{a_.arm}_s{a_.seed}.npz.meta.json", encoding="utf-8"))
    d = load_dir(meta["dump_dir"])
    z = np.load(f"{LAND}/{a_.arm}_s{a_.seed}.npz")
    aligned = all(np.array_equal(d[k], z[k]) for k in ("g", "v0", "eid", "clip_index", "ws"))
    if not aligned or len(d["eid"]) != 2399:
        print("REFUSED: the dump is not window-aligned with the landed arrays", file=sys.stderr)
        return 2
    geo = lambda X: FF._seq_geometry(torch.as_tensor(np.asarray(X, np.float64)), DT)   # noqa: E731
    P, F, Q = geo(d["cl"]), geo(d["fl"]), geo(d["g"])
    out = {"eid": d["eid"], "v0": d["v0"], "spd_x": P["speed"].mean(1).numpy(), "spd_p10": F["speed"].mean(1).numpy(), "spd_gt": Q["speed"].mean(1).numpy()}
    out["kabs_x"], out["kabs_gt_x"], out["kerr_x"] = curv_stats(torch, P, Q)
    out["kabs_p10"], out["kabs_gt_p10"], out["kerr_p10"] = curv_stats(torch, F, Q)
    os.makedirs(os.path.dirname(os.path.abspath(a_.out)), exist_ok=True)
    np.savez_compressed(a_.out, **out)
    m = {"arm": a_.arm, "seed": a_.seed, "dump_dir": meta["dump_dir"], "landed_array_sha256": sha256_file(f"{LAND}/{a_.arm}_s{a_.seed}.npz"), "out_sha256": sha256_file(a_.out), "aligned": aligned}
    with open(a_.out + ".meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(m, f, indent=1, sort_keys=True)
    print("EXTRACT_SIGNED OK", json.dumps(m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
