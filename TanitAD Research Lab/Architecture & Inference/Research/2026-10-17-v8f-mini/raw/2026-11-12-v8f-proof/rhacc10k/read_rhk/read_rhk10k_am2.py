"""read_rhk10k_am2.py -- the reader of card ADD140 under its amendment 2: read_rhk10k.py with one change, made by asserted source replacement in memory.

Amendment 2: control n1 of read_longroll10k_n4 (no-preroll runs start at the origin) is read on card ADD139's runs H_RLcA_P0_s0..s2 (same 1,271-window grid, no
pre-roll), loaded beside the two R3 arms and registered as card ADD139's reader registered them; they enter no pair and no bar. Nothing else changes; read_rhk10k.py
is imported unedited. Same command line as read_rhk10k.py.
"""
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_rhk10k as M  # noqa: E402

N1_ARMS = ("H_RLcA_P0",)
N1_SPEC = ("RLc", "A", 0, "deadreckon", "30", "plan", 5, 0.2)   # as read_rhacc10k.py line 450 registers H_RLcA_P0

src_reg = inspect.getsource(M.register_arms)
old_reg = "            R.ALL_RUNS[key_of(arm, m)] = R3_SPEC\n"
assert src_reg.count(old_reg) == 1
src_reg = src_reg + "    for arm in N1_ARMS:\n        for m in SEEDS:\n            R.ALL_RUNS[key_of(arm, m)] = N1_SPEC\n"

src_r3 = inspect.getsource(M.r3_engine)
old_r3 = 'out = R.read_grid(ns, E, "30", keys, list(P_ARMS), SPREAD_ARM_R3, {"R3": R3_PAIR}, {}, log)'
assert src_r3.count(old_r3) == 1
src_r3 = src_r3.replace(old_r3, 'out = R.read_grid(ns, E, "30", keys, list(P_ARMS) + list(N1_ARMS), SPREAD_ARM_R3, {"R3": R3_PAIR}, {}, log)')
src_r3 = src_r3.replace('"preroll_steps": {k: 30 for k in P_ARMS}}', '"preroll_steps": {k: 30 for k in P_ARMS}, "n1_arms_amendment2": list(N1_ARMS)}')
assert '"n1_arms_amendment2"' in src_r3

M.N1_ARMS, M.N1_SPEC = N1_ARMS, N1_SPEC
exec(compile(src_reg, "<register_arms, amendment 2>", "exec"), M.__dict__)
exec(compile(src_r3, "<r3_engine, amendment 2>", "exec"), M.__dict__)

if __name__ == "__main__":
    sys.exit(M.main())
