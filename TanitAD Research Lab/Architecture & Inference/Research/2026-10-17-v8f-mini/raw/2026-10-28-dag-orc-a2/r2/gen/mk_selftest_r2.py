#!/usr/bin/env python3
"""Assemble gen/selftest_chain.py = r2 head + r1's selftest_chain.py sections A / A-mut / K VERBATIM (lines from '# ==== A.' up to '# ==== Q.') + r2 tail.  python mk_selftest_r2.py"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
R1 = Path("E:/Projects/TanitAD-artifacts/dagorc_rebase/chain/selftest_chain.py").read_text(encoding="utf-8").split("\n")
a = next(i for i, x in enumerate(R1) if x.startswith("# ====") and " A. the gate asserter" in x)
q = next(i for i, x in enumerate(R1) if x.startswith("# ====") and " Q. the queue" in x)
mid = "\n".join(R1[a:q]) + "\n"
out = (HERE / "selftest_r2_head.py").read_text(encoding="utf-8") + mid + "\n" + (HERE / "selftest_r2_tail.py").read_text(encoding="utf-8")
(HERE / "selftest_chain.py").write_bytes(out.encode("utf-8"))
print("wrote selftest_chain.py", len(out), "bytes; r1 sections copied: lines", a + 1, "to", q)
