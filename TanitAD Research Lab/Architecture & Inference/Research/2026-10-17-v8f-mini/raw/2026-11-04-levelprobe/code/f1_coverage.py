#!/usr/bin/env python3
"""f1_coverage.py -- pre-freeze fact F1 of PREREG_LEVELPROBE.md: which of the T1 rig's 2,399 eval windows does probe 3's G8 grid cover (v0 < 6 only), and where does the M1 mass sit?
Reads only banked artifacts (read-only).  python -I code/f1_coverage.py  ->  raw/F1_coverage.json
"""
import json
import os
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

ART = "E:/Projects/TanitAD-artifacts"
PK = f"{ART}/dagland/levelprobe"
z = np.load(f"{ART}/dagland/kappa_t1/raw/arrays/SREP_DAGOACC_s0.npz", allow_pickle=True)
labels = [str(q) for q in z["labels"]]
i = labels.index("M1 crawl share")
v0, ws, eid = z["v0"], z["ws"], z["eid"]
a = z[f"a_{i}"]
m1 = np.isfinite(a) & np.isfinite(z[f"b_{i}"])
f = np.load(f"{ART}/dagland/trigger_probe/raw/eval141_features.npz", allow_pickle=False)
sel = f["sel"]
cov = np.zeros(len(v0), bool)
cov[sel] = True
chk = {"sel_v0_equals_rig_v0": bool(np.array_equal(f["v0"], v0[sel])), "sel_t0_equals_rig_ws": bool(np.array_equal(f["t0"], ws[sel])), "sel_eid_equals_rig_eid": bool(np.array_equal(f["eid"], eid[sel])),
       "G8_eval141_covers_exactly_the_v0_lt_6_windows": bool(np.array_equal(cov, v0 < 6.0))}
out = {"n_windows": int(len(v0)), "n_episodes": int(len(np.unique(eid))), "n_covered_v0_lt_6": int(cov.sum()), "n_uncovered_v0_ge_6": int((~cov).sum()), "frac_uncovered": float((~cov).mean()),
       "n_m1_windows": int(m1.sum()), "n_m1_windows_covered": int((m1 & cov).sum()), "n_m1_windows_uncovered": int((m1 & ~cov).sum()), "frac_m1_windows_uncovered": float((m1 & ~cov).sum() / m1.sum()),
       "n_y_m1_nonzero": int((a > 0).sum()), "n_y_m1_nonzero_covered": int(((a > 0) & cov).sum()), "n_y_m1_nonzero_uncovered": int(((a > 0) & ~cov).sum()),
       "train_windows_pstop_pmove": 24000, "train_label_range": "v0 in [0, 6) by construction of P-STOP (v0<0.5) and P-MOVE (0.5<=v0<6)", "checks": chk,
       "source": "kappa_t1/raw/arrays/SREP_DAGOACC_s0.npz (v0, ws, eid, M1 rows), trigger_probe/raw/eval141_features.npz (sel)", "evidence_class": "MEASURED (ours)"}
os.makedirs(f"{PK}/raw", exist_ok=True)
json.dump(out, open(f"{PK}/raw/F1_coverage.json", "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
sys.exit(0 if all(chk.values()) else 3)
