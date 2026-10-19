"""PRIVGEOM-SAFE, POST-HOC DESCRIPTIVE (no authority over any branch; not pinned): how much of the NC label does the `nc_min_dist` scalar reproduce, and what is the rest?

Reads the join arrays and the gate's `labels_sub.npy` (the labels were opened by the gate).  Writes run/descr/nc_identity.json.
  * of the NC = 0 proposals, the share with nc_min_dist == 0 (contact) / in (0, 0.5] / in (0.5, 2] / > 2 m;
  * of the NC = 1 proposals, the same split;  NC = 0.5 reported apart.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                       # noqa: E402

import pg_run as R                                                                       # noqa: E402


def main() -> int:
    A = R.load_join(labels=True)
    d = A["g"][..., 11].astype(np.float64)
    nc = A["lab"][..., 0]
    bins = [("== 0 (contact)", lambda x: x == 0.0), ("(0, 0.5] m", lambda x: (x > 0) & (x <= 0.5)), ("(0.5, 2] m", lambda x: (x > 0.5) & (x <= 2.0)), ("(2, 20) m", lambda x: (x > 2.0) & (x < 20.0)), (">= 20 m (cap)", lambda x: x >= 20.0)]
    out = {"n_proposals": int(nc.size)}
    for name, m in (("NC=0", nc == 0.0), ("NC=0.5", nc == 0.5), ("NC=1", nc == 1.0)):
        x = d[m]
        out[name] = {"n": int(m.sum()), **{b: float(f(x).mean()) if x.size else None for b, f in bins}}
    q = d[(nc == 0.0) & (d > 0)]
    out["NC=0 with nc_min_dist>0: quantiles of nc_min_dist (m) 10/25/50/75/90"] = [float(v) for v in np.quantile(q, [0.1, 0.25, 0.5, 0.75, 0.9])] if q.size else None
    out["n_sets_with_any_NC0"] = int(((nc == 0.0).any(axis=1)).sum())
    out["n_sets_with_any_NC0_and_all_those_dist_gt0"] = int(((nc == 0.0).any(axis=1) & ~(((nc == 0.0) & (d == 0.0)).any(axis=1))).sum())
    os.makedirs(f"{R.ROOT}/run/descr", exist_ok=True)
    R.jdump(out, f"{R.ROOT}/run/descr/nc_identity.json")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
