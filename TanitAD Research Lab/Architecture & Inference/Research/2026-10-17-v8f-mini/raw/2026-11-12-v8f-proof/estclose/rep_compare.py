#!/usr/bin/env python
"""rep_compare.py -- compares every npz array of two run folders of the same run name (the queue run and its replicate): per file the count of arrays, the count of arrays that are
bit-equal and the largest absolute difference of the numeric arrays. Prints one summary line per group (dump files, tick files) and a last line ZZ REP-COMPARE.
Usage: rep_compare.py RUN_DIR_A RUN_DIR_B"""
import glob
import os
import sys

import numpy as np


def cmp_npz(pa, pb):
    a, b = np.load(pa, allow_pickle=False), np.load(pb, allow_pickle=False)
    ka, kb = sorted(a.files), sorted(b.files)
    if ka != kb:
        return {"keys_equal": False, "arrays": len(ka), "bit_equal": 0, "max_abs": float("inf"), "worst": "keys"}
    n_eq, mx, worst = 0, 0.0, ""
    for k in ka:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape or x.dtype != y.dtype:
            return {"keys_equal": True, "arrays": len(ka), "bit_equal": n_eq, "max_abs": float("inf"), "worst": k + " shape/dtype"}
        if x.tobytes() == y.tobytes():
            n_eq += 1
            continue
        if x.dtype.kind in "fiub":
            xf, yf = x.astype(np.float64), y.astype(np.float64)
            both_nan = np.isnan(xf) & np.isnan(yf)
            d = np.where(both_nan, 0.0, np.abs(xf - yf))
            m = float(np.nanmax(d)) if d.size else 0.0
            if np.isnan(d).any():
                m = float("inf")
        else:
            m = float("inf")
        if m > mx:
            mx, worst = m, k
    return {"keys_equal": True, "arrays": len(ka), "bit_equal": n_eq, "max_abs": mx, "worst": worst}


def main(A, B):
    groups = {"dump": "dump_*/ep*.npz", "ticks": "ticks/*.npz"}
    ok_all = True
    for g, pat in groups.items():
        fa = sorted(glob.glob(os.path.join(A, pat)))
        rel = [os.path.relpath(p, A) for p in fa]
        fb_missing = [r for r in rel if not os.path.isfile(os.path.join(B, r))]
        nb = len(glob.glob(os.path.join(B, pat)))
        tot = eq = 0
        mx, worst = 0.0, ""
        files_bit_equal = 0
        for r in rel:
            if r in fb_missing:
                continue
            c = cmp_npz(os.path.join(A, r), os.path.join(B, r))
            tot += c["arrays"]
            eq += c["bit_equal"]
            files_bit_equal += int(c["bit_equal"] == c["arrays"] and c["keys_equal"])
            if c["max_abs"] > mx:
                mx, worst = c["max_abs"], r + ":" + c["worst"]
        same = bool(fa) and not fb_missing and nb == len(fa) and tot == eq
        ok_all = ok_all and same
        print("%s files A %d B %d missing in B %d; files bit-equal %d; arrays %d bit-equal %d; largest abs difference %.3e %s" % (g, len(fa), nb, len(fb_missing), files_bit_equal, tot, eq, mx, worst))
    print("ZZ REP-COMPARE %s" % ("BIT-EQUAL" if ok_all else "DIFFERENT"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
