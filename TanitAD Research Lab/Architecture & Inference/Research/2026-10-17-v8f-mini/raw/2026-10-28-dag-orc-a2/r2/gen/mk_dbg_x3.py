#!/usr/bin/env python3
"""Debug aid: assemble dbg_x3.py = the self-test head + the shell helpers + section X3 only (the aside on a scratch tree), to iterate on X3 without the 7 minutes of sections before it.  python mk_dbg_x3.py && python dbg_x3.py"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
t = (HERE / "selftest_chain.py").read_text(encoding="utf-8")
a = t.index("# ====================================================================== A. the gate asserter")
head = t[:a]
h1, h2 = t.index("def sh(cmdline"), t.index("def ship2(")
x3a, x3b = t.index("# ====================================================================== X3."), t.index("# the chain-level lib pin catches a changed lib")
pre = ('\nFIX = R2 / "fixtures"\nimport inspect\nLIBSRC = (R2SHIP / "oaf_dagoaccorc_r2_lib.sh").read_text(encoding="utf-8")\n'
       'CSRC = (R2SHIP / "v8f_dagoaccorc_r2_chain_checks.py").read_text(encoding="utf-8")\n')
post = ('\nnpass, nfail = sum(1 for _, ok in RESULTS if ok), sum(1 for _, ok in RESULTS if not ok)\nnkill, nsurv = sum(1 for _, ok in MUTANTS if ok), sum(1 for _, ok in MUTANTS if not ok)\n'
        'print("DEBUG-X3:", npass, "tests pass,", nfail, "fail;", nkill, "mutants killed,", nsurv, "survivors")\n')
(HERE / "dbg_x3.py").write_text(head + pre + t[h1:h2] + t[x3a:x3b] + post, encoding="utf-8", newline="\n")
print("dbg_x3.py written")
