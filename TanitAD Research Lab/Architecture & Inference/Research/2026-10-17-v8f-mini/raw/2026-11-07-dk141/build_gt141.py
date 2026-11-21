#!/usr/bin/env python3
"""build_gt141.py -- ONE-TIME build of the eval-141 ground truth used by dk141.py: gt [2399,30,2], eid, clip, frame, v0, dt.

Rebuilt from the tip's own code (`tanitad.data.v8f_mini_data.eval_context(kin_mode="yaw")`, the call `v8f_dk_read.ground_truth` makes) on the pinned
inputs under ref/.  Writes gt141.npz.  The proof that it is the right grid is NOT here: dk141.py REFUSES any arm whose `||pred - gt||` differs from the
arm's own saved ade6 (max dev <= 1e-5 m), so a wrong gt cannot score.
usage: CUDA_VISIBLE_DEVICES=-1 python -I build_gt141.py
"""
import hashlib
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
for p in (f"{HERE}/tip_export/stack", f"{HERE}/tip_export/taniteval"):
    sys.path.insert(0, p)
import numpy as np  # noqa: E402

from tanitad.data import v8f_mini_data as D  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


grid = json.load(open(f"{HERE}/ref/eval_grid_windows.json", encoding="utf-8"))
X = D.eval_context(grid, f"{HERE}/ref/eval141_kin.npz", D.load_join(f"{HERE}/ref/join_eval141.npz"), kin_mode="yaw")
order = list(grid["order"])
eid = np.asarray(X["eid"])
out = dict(gt=np.asarray(X["gt"], np.float64), eid=eid, clip=np.array([order[e] for e in eid]), frame=2 * np.asarray(X["t"], np.int64),
           v0=np.asarray(X["kin"][:, 0], np.float64), dt=np.float64(D.DT))
np.savez_compressed(f"{HERE}/gt141.npz", **out)
print("gt141.npz", {k: (v.shape, str(v.dtype)) for k, v in out.items()}, "sha256", sha(f"{HERE}/gt141.npz"))
print("tanitad from", os.path.abspath(sys.modules["tanitad"].__file__))
