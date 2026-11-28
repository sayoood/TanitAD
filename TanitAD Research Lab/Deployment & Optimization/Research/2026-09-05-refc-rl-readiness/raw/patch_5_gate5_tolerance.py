"""Patch 5: gate 5 (step-0 trust-region divergence) passes on an ABSOLUTE tolerance
of 1e-8 m^2 (~0.1 mm displacement) instead of exact zero. MEASURED 2026-09-05 on the
4060: live-vs-frozen-deepcopy 9.06e-11 m^2 with live-vs-live 0 — float-rounding
scale, ten orders of magnitude under the 2.774 m^2 mode mismatch the gate exists to
catch (TRAIN-C5). The exact-zero flag is still recorded."""
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
D = os.path.join(ROOT, "stack/scripts/rl_refcv3_min.py")
s = open(D, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, f"expected {count}, found {n}: {old[:70]!r}"
    s = s.replace(old, new)


rep("EXPECT_BASE_STEP = 40284\n",
    "EXPECT_BASE_STEP = 40284\n"
    "#: gate 5 tolerance. A MODE MISMATCH reads m^2-scale (2.774 m^2 on the v2.1 pilot,\n"
    "#: TRAIN-C5); a frozen deepcopy on CUDA reads 1e-11 m^2 (MEASURED 2026-09-05, refcv3\n"
    "#: @ 40,284, live-vs-live exactly 0) — kernel-selection rounding, not policy drift.\n"
    "ANCHOR_STEP0_TOL_M2 = 1e-8\n")
rep('''                           "wis": [int(w) for w in b["wis"]], "n_frozen": n_frozen,
                           "PASS": float(div.abs().max()) == 0.0}
''', '''                           "wis": [int(w) for w in b["wis"]], "n_frozen": n_frozen,
                           "exact_zero": float(div.abs().max()) == 0.0,
                           "tolerance_m2": ANCHOR_STEP0_TOL_M2,
                           "PASS": float(div.abs().max()) <= ANCHOR_STEP0_TOL_M2}
''')
rep("    # 5. step-0 trust-region divergence must read EXACTLY 0 ----------------------\n",
    "    # 5. step-0 trust-region divergence must read ~0 (<= ANCHOR_STEP0_TOL_M2) -------\n")
open(D, "w", encoding="utf-8", newline="\n").write(s)
print("patch 5 applied")
