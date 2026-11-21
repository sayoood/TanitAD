"""Exact comparison of two npz files (every array: shape, dtype, values). Prints ZZGTEQ-<same keys>-<n arrays>-<n differing>ZZ."""
import sys
import numpy as np
a, b = np.load(sys.argv[1], allow_pickle=True), np.load(sys.argv[2], allow_pickle=True)
ka, kb = sorted(a.files), sorted(b.files)
diff = [k for k in ka if k in kb and not (a[k].shape == b[k].shape and a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]))]
print("keys", ka, "| differing", diff)
print(f"ZZGTEQ-{int(ka == kb)}-{len(ka)}-{len(diff)}ZZ")
