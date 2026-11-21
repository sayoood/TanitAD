"""Synthetic checks for s0c_fusion.py (no fit-pack data read). python test_s0c.py -> 'ZZS0C_TEST_PASS <n>'"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0c_fusion as S  # noqa: E402

n = 0
rng = np.random.default_rng(0)
rule = rng.random((50, 64))
# 1. p_ok == 1 -> the fused pick IS the canonical pick (C-REG identity), first index on ties
assert (S.fused_pick(rule, np.ones_like(rule)) == rule.argmax(1)).all(); n += 1
tie = np.zeros((1, 64)); tie[0, [3, 9]] = 1.0
assert S.fused_pick(tie, np.ones_like(tie))[0] == 3; n += 1
# 2. p_ok = 0 on the canonical pick moves the pick
p = np.ones_like(rule); p[np.arange(50), rule.argmax(1)] = 0.0
assert (S.fused_pick(rule, p) != rule.argmax(1)).all(); n += 1
# 3. features: F0 has 6 columns, F-X adds exactly the two unpacked flag columns in proposal order
lg = rng.normal(size=(2, 64, 6)).astype(np.float32)
fl = np.zeros((2, 64), bool); fl[0, 5] = True; fl[1, 63] = True
F = {"R-t0__nc": np.packbits(fl, axis=1), "R-t0__ttc": np.packbits(~fl, axis=1)}
X0 = S.features(lg, F, "F0"); X1 = S.features(lg, F, "F-R-t0")
assert X0.shape == (128, 6) and X1.shape == (128, 8); n += 1
assert X1[5, 6] == 1.0 and X1[64 + 63, 6] == 1.0 and X1[:, 6].sum() == 2 and (X1[:, 7] == 1 - X1[:, 6]).all(); n += 1
# 4. outcome rows, mechanically
def rep(c0, ct0, csh, g0=0.1, gt0=0.2):
    return {"F0": {"capture": c0, "gain": g0}, "F-R-t0": {"capture": ct0, "gain": gt0}, "F-C-SHUF": {"capture": csh}}
assert S.outcome(rep(0.1, 0.3, 0.1), 0.0, 10)[0] == "S0C-FUSE-GO"; n += 1
assert S.outcome(rep(0.3, 0.3, 0.3, g0=0.5, gt0=0.4), 0.0, 10)[0] == "S0C-RERANK"; n += 1
assert S.outcome(rep(0.1, 0.1, 0.1), 0.0, 10)[0] == "S0C-NULL"; n += 1
assert S.outcome(rep(0.1, 0.3, 0.1), 1e-9, 10)[0] == "INVALID"; n += 1          # C-REG nonzero
assert S.outcome(rep(0.1, 0.3, 0.25), 0.0, 10)[0] == "INVALID"; n += 1          # shuffled flags add >= 0.10
assert S.outcome(rep(0.1, 0.3, 0.1, g0=0.3, gt0=0.2), 0.0, 10)[0] == "S0C-NULL"; n += 1  # F-t0 not above F0 and F0 < 0.25
print("ZZS0C_TEST_PASS", n)
