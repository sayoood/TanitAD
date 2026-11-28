"""Runs the LEADSUP label rule on the real probe join of this box (C:/Users/Admin/lorapod/ship2/join_train.npz: arrays clip, off, n, lead_valid, lead_gap, as read by spatial/s0e) and writes the counts and the
sha256 of the three label arrays: a real-data check that the vectorised rule runs on the real join and what share of rows carries each label. Reads the join only (34.7 MB). T0 bookkeeping, no model.
Usage: real_join_label_stats.py <tree root>  (PYTHONPATH = the tree's stack)"""
import hashlib
import json
import os
import sys

import numpy as np

tree = sys.argv[1]
import tanitad  # noqa: E402

assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(tree).lower()), tanitad.__file__
from tanitad.data import v8f_mini_data as D  # noqa: E402

CK = os.path.dirname(os.path.abspath(__file__))
P = "C:/Users/Admin/lorapod/ship2/join_train.npz"
z = np.load(P)
J = {"off": z["off"].astype(np.int64), "n": z["n"].astype(np.int64), "lead_valid": z["lead_valid"], "lead_gap": z["lead_gap"]}
L = D.lead_sup_label_arrays(J)
N = L["n_rows"]
fin_gap_raw = np.isfinite(J["lead_gap"].astype(np.float64))
res = {"join": P, "join_sha256": hashlib.sha256(open(P, "rb").read()).hexdigest(), "n_clips": int(len(J["n"])), "n_rows": N, "n_pres_lead_valid_1": L["n_pres"], "share_pres": L["n_pres"] / N,
       "n_finite_raw_gap": int(fin_gap_raw.sum()), "n_gap_label_0_60": L["n_gap"], "share_gap": L["n_gap"] / N, "n_clos_label": L["n_clos"], "share_clos": L["n_clos"] / N,
       "label_sha256": L["sha256"],
       "clos_quantiles_ms": {str(q): float(np.nanquantile(L["clos"], q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)},
       "gap_quantiles_m": {str(q): float(np.nanquantile(L["gap"], q)) for q in (0.05, 0.5, 0.95)},
       "rows_with_presence_but_no_gap_label": int(((L["pres"] == 1) & ~np.isfinite(L["gap"])).sum()),
       "rows_with_gap_label_but_no_presence": int(((L["pres"] == 0) & np.isfinite(L["gap"])).sum()), "tanitad_file": tanitad.__file__}
json.dump(res, open(os.path.join(CK, "real_join_label_stats.json"), "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps(res, indent=1))
