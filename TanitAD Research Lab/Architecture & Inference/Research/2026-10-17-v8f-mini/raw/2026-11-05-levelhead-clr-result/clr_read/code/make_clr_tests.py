#!/usr/bin/env python3
"""make_clr_tests.py -- builds tests/test_clr_read.py and tests/test_clr_chain_gates.py from the CL package's tests (sha256 pinned): role rename CL -> CLR, plain replacements, and exact-once edits / spans (an edit that does
not match exactly once aborts).  The CLR-specific tests (attribution, outcome mapping, A2 card, RAPR-P01 base, recovery marker, readout clone, generators) live in the edits below.   usage: python -I make_clr_tests.py [--check]"""
import hashlib
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
CLT = "E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_read/tests"
import importlib.util
_s = importlib.util.spec_from_file_location("clr_test_edits", f"{HERE}/clr_test_edits.py")
TE = importlib.util.module_from_spec(_s)
_s.loader.exec_module(TE)


def rename(t):
    for a, b in (("RICLC", "RICLRC"), ("RICL", "RICLR"), ("FZCL", "FZCLR"), ("CLC", "CLRC"), ("CL", "CLR")):
        t = re.sub(r"(?<![A-Za-z0-9])" + a + r"(?![A-Za-z0-9])", b, t)
    return t.replace("PREREG_LEVELHEAD_CLR", "PREREG_LEVELHEAD_CL")


def build(src_name, spec):
    raw = open(f"{CLT}/{src_name}", "rb").read()
    assert hashlib.sha256(raw).hexdigest() == spec["sha"], f"{src_name} is not the pinned file"
    t = rename(raw.decode("utf-8"))
    for e in spec["edits"]:
        kind = e[0]
        if kind == "all":
            if t.count(e[2]) < 1:
                raise SystemExit(f"all {e[1]!r}: 0 matches")
            t = t.replace(e[2], e[3])
        elif kind == "edit":
            if t.count(e[2]) != 1:
                raise SystemExit(f"edit {e[1]!r}: {t.count(e[2])} matches (must be 1)")
            t = t.replace(e[2], e[3])
        else:
            _, name, start, end, new = e
            if t.count(start) != 1 or t.count(end) != 1:
                raise SystemExit(f"span {name!r}: start/end match {t.count(start)}/{t.count(end)} times (must be 1/1)")
            i = t.index(start)
            t = t[:i] + new + t[t.index(end, i):]
    return t


if __name__ == "__main__":
    ok = True
    for src, dst, spec in (("test_cl_read.py", "test_clr_read.py", TE.READ), ("test_cl_chain_gates.py", "test_clr_chain_gates.py", TE.GATES)):
        out = build(src, spec)
        p = f"{PK}/tests/{dst}"
        if "--check" in sys.argv:
            same = os.path.isfile(p) and open(p, encoding="utf-8", newline="").read() == out
            print(dst, "== generated" if same else "DIFFERS from generated")
            ok &= same
        else:
            open(p, "w", encoding="utf-8", newline="\n").write(out)
            print("wrote", p, len(out))
    sys.exit(0 if ok else 1)
