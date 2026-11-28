"""PREREG_V8F_DECINFO amendment 1 (Master Mind, 2026-11-09; the card's Log states why): the four-frame grid rungs R4 and R6, ridge reader only, on the top 8 PCA channels per frame (1,280 features, the count of the one-frame rungs). Control 2 and its bars are unchanged. Conv rows are untouched. TUNE windows only; CPU. Tier T0, a readout diagnostic.
Writes amend1/amend1.json and prints one line per row."""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import decinfo_lib as DL
import run_decinfo as RD

K_RIDGE = 8
M = 4
OUT = os.path.join(HERE, "amend1")


def read(D, rung, tag, grid):
    X8 = grid[:, :, :K_RIDGE, :]
    assert X8.shape[2:] == (K_RIDGE, 40)
    Xs = DL.stack_ch(X8, M)
    Xf = Xs[:D.nf].reshape(D.nf * Xs.shape[1], -1)
    Xh = Xs[D.nf:].reshape((D.n - D.nf) * Xs.shape[1], -1)
    assert Xf.shape[1] == K_RIDGE * 40 * M == 1280
    a, z = DL.planted_check(Xf, Xh)
    res = dict(rung=rung, tag=tag, m=M, k_ridge=K_RIDGE, n_features=int(Xf.shape[1]), planted=a, zeroed=z, control2_pass=bool(a >= 0.80 and z <= 0.02), rows=[])
    RD.log(rung, tag, f"planted {a:.3f} zeroed {z:.3f}")
    ys_all = DL.targets(D)
    for ti, tgt in enumerate(RD.TARGETS):
        y = ys_all[tgt]
        ysh = RD.shuffled(D, y, seed=100 + ti)
        yS, ySh = y[:, M - 1:], ysh[:, M - 1:]
        r, nf_, nh_ = DL.ridge_read(D, Xs, yS, yS)
        rs, _, _ = DL.ridge_read(D, Xs, yS, ySh)
        res["rows"].append(dict(target=tgt, reader="ridge", r2=r, r2_shuffled=rs, n_fit=nf_, n_held=nh_))
        RD.log(rung, tag, "ridge", tgt, f"{r:+.3f} shuf {rs:+.3f}")
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    D, _ = DL.load_tune()
    assert D.n == 982 and D.nf == 744
    out = {"tier": "T0", "amendment": 1, "units": []}
    for tag in ("M2b_s0", "M2b_s1"):
        DL.wait_ram()
        model, _ = DL.load_any(DL.M2B[tag])
        grid, _ = DL.adapter_grid(model, D)
        out["units"].append(read(D, "R4", tag, grid))
        del grid
    DL.wait_ram()
    grid, _ = DL.encoder_grid(D)
    out["units"].append(read(D, "R6", "encoder", grid))
    json.dump(out, open(os.path.join(OUT, "amend1.json"), "w", encoding="utf-8"), indent=1)
    RD.log("amend1 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
