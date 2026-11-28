"""PRIVGEOM-SAFE, POST-HOC DESCRIPTIVE (no authority over any branch; not pinned): does the DAC-nosign companion still let a LINEAR read recover the sign of the clearance?

The companion keeps |dac_clear_min| and the nine |dac_clear_tXX|.  Logic (not data): c_min is the minimum signed clearance over ALL steps, so c_t >= c_min at every key step.
If c_min >= 0 (a DAC-safe proposal), then |c_t| >= |c_min| at every key step.  Hence a key step with |c_t| < |c_min| PROVES c_min < 0, i.e. DAC = 0.
A column pair (|c_min|, |c_t|) therefore carries the label's sign through their CONTRAST, and a linear fit that weights them with opposite signs can exploit it.
This script measures, on the real arrays: precision of that rule (must be 1.0 by the logic), its recall, and the raw within-set F-AUC of the contrast x = min_k |c_tk| - |c_min| (no fit).
Writes run/descr/dac_signleak.json.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                       # noqa: E402

import pg_lib as P                                                                       # noqa: E402
import pg_run as R                                                                       # noqa: E402


def main() -> int:
    A = R.load_join(labels=True)
    an = P.Analyzer(A["lab"], A["code"], A["log_idx"], A["n_logs"])
    g = A["g"].astype(np.float64)
    cmin, ct = g[..., 0], g[..., 2:11]
    unsafe = A["lab"][..., 1] == 0.0
    safe = A["lab"][..., 1] == 1.0
    F = A["code"] == 4
    cond = (np.abs(ct) + 1e-5 < np.abs(cmin)[..., None]).any(axis=-1)
    x = (np.abs(ct).min(axis=-1) - np.abs(cmin))
    out = {"n_proposals": int(cmin.size), "n_unsafe": int(unsafe.sum()), "n_unsafe_F": int((unsafe & F).sum()),
           "precision_P(DAC=0 | some key step has |c_t| < |c_min|)": float(unsafe[cond].mean()), "n_cond": int(cond.sum()),
           "n_cond_and_safe (must be 0 by the logic)": int((cond & safe).sum()),
           "recall_all": float(cond[unsafe].mean()), "recall_F": float(cond[unsafe & F].mean()),
           "min_k|c_t| - |c_min| quantiles unsafe (10/50/90)": [float(v) for v in np.quantile(x[unsafe], [0.1, 0.5, 0.9])],
           "min_k|c_t| - |c_min| quantiles safe (10/50/90)": [float(v) for v in np.quantile(x[safe], [0.1, 0.5, 0.9])]}
    for nm, z in (("contrast x = min_k|c_t| - |c_min| (larger = safer)", x), ("indicator 1[no key step with |c_t| < |c_min|]", (~cond).astype(np.float64))):
        out["raw_F_auc " + nm] = an.auc(an.per_log_stats(z.astype(np.float32), "dac"), "Q4")
    os.makedirs(f"{R.ROOT}/run/descr", exist_ok=True)
    R.jdump(out, f"{R.ROOT}/run/descr/dac_signleak.json")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
