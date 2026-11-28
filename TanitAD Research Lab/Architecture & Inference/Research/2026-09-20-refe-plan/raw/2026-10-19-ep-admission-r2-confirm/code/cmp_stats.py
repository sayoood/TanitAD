"""Bit-for-bit comparison of a fresh stats.npz with the banked EP-ADMISSION-R1 stats.npz.  python -I cmp_stats.py <fresh> <banked> <out.json>"""
import json
import sys

import numpy as np

a, b, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
ka, kb = sorted(a.files), sorted(b.files)
ok = ka == kb and all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and np.array_equal(a[k], b[k]) for k in ka)
res = {"fresh": sys.argv[1], "banked": sys.argv[2], "n_keys_fresh": len(ka), "n_keys_banked": len(kb), "same_keys": ka == kb, "all_arrays_bit_identical": bool(ok)}
json.dump(res, open(out, "w"), indent=1)
print("STATS_IDENTICAL", res)
sys.exit(0 if ok else 3)
