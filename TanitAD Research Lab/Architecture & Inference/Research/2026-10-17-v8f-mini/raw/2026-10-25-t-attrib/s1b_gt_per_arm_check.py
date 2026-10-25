"""s1b_gt_per_arm_check.py -- does every arm's own GT-oracle block equal the GT I score against (same T tick, same windows)? MEASURED check, eval-141."""
import numpy as np
from tcommon import *
L = "C:/Users/Admin/box2_ship/logs/"
T = load(); G = np.where(np.isfinite(T["gt_std"]), T["gt_std"], 0.0); has = T["has"]
for arm, (d, tag) in {"grid8x20_s0": ("v8f_grid", "M2b-8x20_s0"), "hist10_s0": ("v8f_hist", "M2b-h10_s0"), "pgt_s0": ("v8f_pgt", "M2b-pgt_s0"),
                      "wc_s0": ("v8f_wc", "M2b-wc_s0"), "df50_s0": ("v8f_dscale", "M2b-df50_s0"), "M1_s0": ("v8f_tacgrp2", "M1_s0")}.items():
    z = np.load(L + f"{d}/eval/eval_{tag}-tacorc.npz")
    if "tac_code" not in z.files:
        print(f"{arm:12s} tacorc has no tac_code"); continue
    g = z["tac_code"][:, 34:60].astype(float)
    bad = np.abs(g - G)[has].max(1) > 1e-4
    print(f"{arm:12s} windows whose arm-own GT block != my GT block: {int(bad.sum())} of {int(has.sum())}"
          + (f"; their t values {sorted(set(T['tt'][has][bad].tolist()))[:6]}" if bad.any() else ""))
