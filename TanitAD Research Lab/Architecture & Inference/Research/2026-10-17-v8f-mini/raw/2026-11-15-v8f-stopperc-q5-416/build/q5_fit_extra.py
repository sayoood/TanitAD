"""q5_fit_extra.py -- the one beside arm s0p_fit.py does not have: V40@416+(E-limit) (AMENDMENT 7 section 2.2, Q7).  s0p_fit.py stays UNEDITED: its ARMS / GRID_OF / build_features are
extended in this process only, and `fit_one` (CV, refit, score eval once, same grids) is called as-is.

  python q5_fit_extra.py --train <dir> --eval <npz|dir> --work <work dir with proj_*.npy from `s0p_fit.py prep`> --model linear|mlp
-> <work>/fit_V40+E-limit_<model>.{json,npy}
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q5_controls as C  # noqa: E402

ARM = "V40+E-limit"


def extend(F):
    """Register the arm in the s0p_fit module object (this process only)."""
    if ARM in F.GRID_OF:
        return F
    F.GRID_OF[ARM] = "v40"
    orig = F.build_features

    def build_features(arm, P, E, dim):
        if arm == ARM:
            return np.concatenate([P[:, :, :dim].reshape(len(P), -1), np.asarray(E)[:, :F.L.E_NO_LIMIT]], 1).astype(np.float32)
        return orig(arm, P, E, dim)

    F.build_features = build_features
    return F


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", choices=("linear", "mlp"), required=True)
    a = ap.parse_args(argv)
    F, _ = C.s0p_modules()
    extend(F)
    tr, ev = F.load_any(a.train), F.load_any(a.eval)
    F.fit_one(ARM, a.model, a.work, tr, ev)


if __name__ == "__main__":
    main()
