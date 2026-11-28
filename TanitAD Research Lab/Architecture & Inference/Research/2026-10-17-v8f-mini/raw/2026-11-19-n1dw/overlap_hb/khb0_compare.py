"""khb0_compare.py BANKED_DUMP_DIR NEW_DUMP_DIR OUT_JSON -- control K-HB0 of PREREG_N1DW_HB_OVERLAP.md.
PASS iff the dump made through v8f_t1_eval_hb.py --kappa-max none holds the same ep*.npz files as the banked dump, with the same keys,
and every array bit-equal (np.array_equal, dtype and shape equal). Exit 0 on PASS, 3 on FAIL."""
import glob
import json
import os
import sys

import numpy as np

a, b, out = sys.argv[1:4]
fa = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a, "ep*.npz")))
fb = sorted(os.path.basename(p) for p in glob.glob(os.path.join(b, "ep*.npz")))
rep = {"control": "K-HB0", "banked": a, "new": b, "n_files_banked": len(fa), "n_files_new": len(fb), "mismatches": []}
if fa != fb or not fa:
    rep["mismatches"].append("file sets differ or empty")
else:
    for f in fa:
        za, zb = np.load(os.path.join(a, f)), np.load(os.path.join(b, f))
        if sorted(za.files) != sorted(zb.files):
            rep["mismatches"].append(f"{f}: keys differ")
            continue
        for k in za.files:
            x, y = za[k], zb[k]
            if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y, equal_nan=(x.dtype.kind == "f")):
                rep["mismatches"].append(f"{f}:{k}")
rep["n_mismatches"] = len(rep["mismatches"])
rep["mismatches"] = rep["mismatches"][:50]
rep["pass"] = rep["n_mismatches"] == 0
json.dump(rep, open(out, "w"), indent=1)
print("K-HB0", "PASS" if rep["pass"] else "FAIL", rep["n_files_banked"], rep["n_files_new"], rep["n_mismatches"])
sys.exit(0 if rep["pass"] else 3)
