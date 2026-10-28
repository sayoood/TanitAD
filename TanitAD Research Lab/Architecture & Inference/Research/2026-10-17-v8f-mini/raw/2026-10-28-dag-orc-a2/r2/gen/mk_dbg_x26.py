#!/usr/bin/env python3
"""Debug aid: assemble dbg_x26.py = the self-test head + r1's sc_set / sc_del helpers + the C2 definitions + load_mod / cmut + sections X2, X4 and X6 only (the stream-record clause, the r2 invariants, the r2 asserter + the real fixed-trainer records).
python mk_dbg_x26.py && python dbg_x26.py"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
t = (HERE / "selftest_chain.py").read_text(encoding="utf-8")


def cut(a, b):
    i = t.index(a)
    return t[i:t.index(b, i)]


head = t[:t.index("# ====================================================================== A. the gate asserter")]
helpers = cut("def sc_set(", "def argv_edit(")
c2 = cut("# ====================================================================== C2.", "qf = bp(QUEUE)")
pre = ('\nqf = bp(QUEUE)\nCSRC = (R2SHIP / "v8f_dagoaccorc_r2_chain_checks.py").read_text(encoding="utf-8")\nLIBSRC = (R2SHIP / "oaf_dagoaccorc_r2_lib.sh").read_text(encoding="utf-8")\n'
       'RUNS_CMD = r"D:\\tanitad\\runs\\v8f_mini_full"\nARMS2 = ("dagoaccorc", "dagoaccorcrep")\nother = SCR / "other_stop.out"\n')
lm = cut("def load_mod(", "PRED_MUTS = [")
x2 = cut("# ====================================================================== X2.", "# ====================================================================== X3.")
x46 = cut("# ====================================================================== X4.", "# ====================================================================== X5.")
post = ('\nnpass, nfail = sum(1 for _, ok in RESULTS if ok), sum(1 for _, ok in RESULTS if not ok)\nnkill, nsurv = sum(1 for _, ok in MUTANTS if ok), sum(1 for _, ok in MUTANTS if not ok)\n'
        'print("DEBUG-X26:", npass, "tests pass,", nfail, "fail;", nkill, "mutants killed,", nsurv, "survivors")\n')
(HERE / "dbg_x26.py").write_text(head + helpers + c2 + pre + lm + x2 + x46 + post, encoding="utf-8", newline="\n")
print("dbg_x26.py written")
