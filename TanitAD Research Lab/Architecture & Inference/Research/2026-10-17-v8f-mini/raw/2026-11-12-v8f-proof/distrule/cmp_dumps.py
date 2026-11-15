#!/usr/bin/env python
"""cmp_dumps.py A_DIR B_DIR : compare two T1-kin dump folders array for array (controls r1 and r2 of card PREREG_V8F_PROOF_ADD26 with note N1).
The episode files are matched by sorted position after the arm tag is stripped from the name; every numeric array must be equal (NaN equals NaN).
Arrays of strings or objects (tags, paths) are listed and not compared. Prints one line `CMP files=<n> differing=<k> max_abs=<x>` and exits 0 only when
n is 141 and k is 0. A self-check runs first: a copy of the first file with one value moved by 1e-6 must be reported as differing."""
import os
import sys

import numpy as np


def load(p):
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def diff(a, b):
    """(number of differing numeric arrays, max abs difference, list of keys present in one only)."""
    only = sorted(set(a) ^ set(b))
    nd, mx = 0, 0.0
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if x.dtype.kind in "OUS" or y.dtype.kind in "OUS":
            continue
        if x.shape != y.shape:
            nd += 1
            mx = float("inf")
            continue
        if x.dtype.kind == "f" or y.dtype.kind == "f":
            same = np.array_equal(x, y, equal_nan=True)
        else:
            same = np.array_equal(x, y)
        if not same:
            nd += 1
            with np.errstate(invalid="ignore"):
                d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            d = d[np.isfinite(d)]
            mx = max(mx, float(d.max()) if d.size else float("inf"))
    return nd, mx, only


def main(argv):
    A, B = argv[1], argv[2]
    fa = sorted(f for f in os.listdir(A) if f.endswith(".npz"))
    fb = sorted(f for f in os.listdir(B) if f.endswith(".npz"))
    if len(fa) != len(fb) or not fa:
        print("CMP files=%d/%d differing=-1 max_abs=nan (file counts differ)" % (len(fa), len(fb)))
        return 2
    # self-check: the comparison must see a planted change
    a0 = load(os.path.join(A, fa[0]))
    num = [k for k in sorted(a0) if a0[k].dtype.kind == "f" and a0[k].size and np.isfinite(a0[k]).any()]
    if not num:
        print("CMP self-check impossible: no finite float array in " + fa[0])
        return 2
    b0 = {k: v.copy() for k, v in a0.items()}
    flat = b0[num[0]].reshape(-1)
    j = int(np.flatnonzero(np.isfinite(flat))[0])
    flat[j] = flat[j] + (1e-3 if flat.dtype == np.float16 else 1e-6) * max(1.0, abs(float(flat[j]))) * (4096 if flat.dtype == np.float32 else 1)
    if diff(a0, b0)[0] != 1:
        print("CMP self-check FAILED: a planted change was not seen")
        return 2
    k, mx, strange = 0, 0.0, set()
    for x, y in zip(fa, fb):
        nd, m, only = diff(load(os.path.join(A, x)), load(os.path.join(B, y)))
        strange.update(only)
        if nd or only:
            k += 1
            mx = max(mx, m)
    print("CMP files=%d differing=%d max_abs=%g keys_in_one_only=%s self_check=ok" % (len(fa), k, mx, sorted(strange)))
    return 0 if (k == 0 and len(fa) == 141) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
