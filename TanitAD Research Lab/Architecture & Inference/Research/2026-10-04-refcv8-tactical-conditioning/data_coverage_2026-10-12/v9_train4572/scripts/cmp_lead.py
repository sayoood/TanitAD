"""Compare two lead npz files record-for-record keyed by (sha12, k): bitwise on every value array."""
import sys, json
import numpy as np

A, B_ = sys.argv[1], sys.argv[2]
za, zb = np.load(A, allow_pickle=True), np.load(B_, allow_pickle=True)
keys = ["sha12", "k", "has", "lead", "gap", "tg", "s", "v"]


def table(z):
    s = z["sha12"].astype(str)
    k = z["k"].astype(np.int64)
    # sort by (sha12, k) for a canonical order
    order = np.lexsort((k, s))
    return {kk: z[kk][order] for kk in keys}


ta, tb = table(za), table(zb)
print("records a/b:", len(ta["k"]), len(tb["k"]))
same_n = len(ta["k"]) == len(tb["k"])
bad = {}
if same_n:
    for kk in keys:
        a, b = ta[kk], tb[kk]
        if a.dtype.kind == "U":
            bad[kk] = int((a != b).sum())
        else:
            bad[kk] = int((np.ascontiguousarray(a).view(np.uint8).reshape(len(a), -1) != np.ascontiguousarray(b).view(np.uint8).reshape(len(b), -1)).any(axis=1).sum())
print("records differing per field (bitwise):", bad)
print("clips a/b:", len(set(za["sha12"].astype(str))), len(set(zb["sha12"].astype(str))))
print("LEAD_CMP", "IDENTICAL" if same_n and not any(bad.values()) else "DIFFERENT")
