#!/usr/bin/env python
"""qhead_types_n2.py -- prints the arm types that are read under note PREREG_V8F_PROOF_ADD42_N2 section 2 (both checkpoints of the type inside the h5 band on the
check clips), space separated, from the fit records in --fit-dir. The band and the type table come from qhead_names_n2.py. Prints nothing if no type is read."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qhead_names_n2 as N2  # noqa: E402


def read_types(fit_dir):
    out = []
    for ty, m in N2.TYPES.items():
        ok = True
        for s in N2.SEEDS:
            f = N2.jl(os.path.join(fit_dir, "%s_s%d.qhead.json" % (m["base"], s)))
            c = ((f or {}).get("coverage_check") or {}).get("share_q_above_label")
            ok = ok and bool(c is not None and N2.H5_LO <= c <= N2.H5_HI)
        if ok:
            out.append(ty)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--fit-dir", required=True)
    a = ap.parse_args()
    print(" ".join(read_types(a.fit_dir)))
