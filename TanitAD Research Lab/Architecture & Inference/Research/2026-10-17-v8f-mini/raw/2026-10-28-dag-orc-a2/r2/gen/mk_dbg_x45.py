#!/usr/bin/env python3
"""Debug aid: assemble dbg_x45.py = the self-test head + the C2 definitions + load_mod + put + sections X4 and X5 only.  python mk_dbg_x45.py && python dbg_x45.py"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
t = (HERE / "selftest_chain.py").read_text(encoding="utf-8")


def cut(a, b):
    i = t.index(a)
    return t[i:t.index(b, i)]


head = t[:t.index("# ====================================================================== A. the gate asserter")]
c2 = cut("# ====================================================================== C2.", "qf = bp(QUEUE)")
lm = cut("def load_mod(", "def cmut(")
pt = cut("def put(p, data):", "def r1_state(")
pre = ('\nCSRC = (R2SHIP / "v8f_dagoaccorc_r2_chain_checks.py").read_text(encoding="utf-8")\nLIBSRC = (R2SHIP / "oaf_dagoaccorc_r2_lib.sh").read_text(encoding="utf-8")\n'
       'RUNS_CMD = r"D:\\tanitad\\runs\\v8f_mini_full"\nARMS2 = ("dagoaccorc", "dagoaccorcrep")\n')
x45 = cut("# ====================================================================== X4.", "# ====================================================================== summary")
post = ('\nnpass, nfail = sum(1 for _, ok in RESULTS if ok), sum(1 for _, ok in RESULTS if not ok)\nnkill, nsurv = sum(1 for _, ok in MUTANTS if ok), sum(1 for _, ok in MUTANTS if not ok)\n'
        'print("DEBUG-X45:", npass, "tests pass,", nfail, "fail;", nkill, "mutants killed,", nsurv, "survivors")\n')
(HERE / "dbg_x45.py").write_text(head + c2 + pre + lm + pt + x45 + post, encoding="utf-8", newline="\n")
print("dbg_x45.py written")
