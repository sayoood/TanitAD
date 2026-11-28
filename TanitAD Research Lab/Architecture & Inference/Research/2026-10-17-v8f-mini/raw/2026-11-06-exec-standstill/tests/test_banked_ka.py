"""Known answer on the BANKED arrays: the reader machinery used by exec_read.py reproduces RESULT_RI_C0.md's logged-frame M1 excess of C0 against P10
(s0 +0.0532, s1 +0.0594) from the banked per-window arrays, and the identical-input dE is exactly 0.
Run:  python -I tests/test_banked_ka.py
"""
import importlib.util
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
CODE = os.path.join(os.path.dirname(HERE), "code").replace("\\", "/")
spec = importlib.util.spec_from_file_location("exec_read", f"{CODE}/exec_read.py")
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)
REF = "E:/Projects/TanitAD-artifacts/dagland/recover_next/read/raw/arrays"
WANT = {0: 0.0532, 1: 0.0594}


def main():
    rr = X.load_rr()
    rr.heavy_init()
    GD = rr.GD
    i1 = GD.LABELS.index(X.M1)
    for s in (0, 1):
        z = rr.read_arrays(f"{REF}/C0_s{s}.npz")
        cls, _ = GD.classify(z["v0"], z["g"])
        r = GD.decompose(z[f"a_{i1}"], z[f"b_{i1}"], z["eid"], cls, GD.POOL[X.M1], 2000)
        assert abs(r["delta"] - WANT[s]) < 6e-5, (s, r["delta"], WANT[s])
        d = z[f"a_{i1}"] - z[f"b_{i1}"]
        r0 = X.delta_e(GD, d, d, z["eid"], cls, 2000)
        assert r0["delta"] == 0.0 and r0["lo"] == 0.0 and r0["hi"] == 0.0
        print(f"PASS s{s}: banked C0 M1 excess {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] (RESULT_RI_C0 logged {WANT[s]:+.4f}); identical-input dE = 0 with CI [0, 0]; n={r['n_windows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
