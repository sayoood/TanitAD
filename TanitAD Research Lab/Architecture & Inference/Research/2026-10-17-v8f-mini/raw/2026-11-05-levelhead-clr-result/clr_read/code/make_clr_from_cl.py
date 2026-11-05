#!/usr/bin/env python3
"""make_clr_from_cl.py -- builds code/clr_read.py from the CL reader (cl_read/code/cl_read.py, sha256 pinned below): the role names are renamed (RICL -> RICLR, RICLC -> RICLRC, FZCL -> FZCLR, CLC -> CLRC, CL -> CLR)
and every other change is one EXACT-ONCE replacement listed in EDITS (a replacement that does not match exactly once aborts).  The generated file is then the reader; this script is the provenance of its diff against CL's.
usage: python -I make_clr_from_cl.py [--check]     (--check: regenerate in memory and compare with code/clr_read.py; exit 1 on a difference)"""
import hashlib
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
SRC = "E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_read/code/cl_read.py"
SRC_SHA = "483dd200881d4a8ee8a719505bb6cf252ebe74e233552d81611223b29f31028c"
DST = f"{PK}/code/clr_read.py"

RENAMES = [("RICLC", "RICLRC"), ("RICL", "RICLR"), ("FZCL", "FZCLR"), ("CLC", "CLRC"), ("CL", "CLR")]
import importlib.util
_s = importlib.util.spec_from_file_location("clr_edits", f"{HERE}/clr_edits.py")
CE = importlib.util.module_from_spec(_s)
_s.loader.exec_module(CE)
DOC_NEW, EDITS = CE.DOC_NEW, CE.EDITS


def build():
    raw = open(SRC, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == SRC_SHA, "cl_read.py is not the pinned file"
    t = raw.decode("utf-8")
    i = t.index('"""', t.index('"""') + 3) + 3                      # end of the module docstring
    body = t[i:]
    for a, b in RENAMES:        # the role names, also when preceded by '_' (rollout_RICL_s0, split_RICL_minus_RIC0); the docstring is inserted AFTER the renaming
        body = re.sub(r"(?<![A-Za-z0-9])" + a + r"(?![A-Za-z0-9])", b, body)
    body = body.replace("PREREG_LEVELHEAD_CLR", "PREREG_LEVELHEAD_CL")      # the CL card keeps its name
    t = '#!/usr/bin/env python3\n"""' + DOC_NEW.strip("\n") + '\n"""' + body
    for e in EDITS:
        if e[0] == "edit":
            _, name, old, new = e
            n = t.count(old)
            if n != 1:
                raise SystemExit(f"edit {name!r}: old text matches {n} times (must be 1)")
            t = t.replace(old, new)
        else:
            _, name, start, end, new = e
            if t.count(start) != 1:
                raise SystemExit(f"span {name!r}: start matches {t.count(start)} times (must be 1)")
            i = t.index(start)
            if t.count(end, i) != 1:
                raise SystemExit(f"span {name!r}: end matches {t.count(end, i)} times after start (must be 1)")
            t = t[:i] + new + t[t.index(end, i):]
    return t


if __name__ == "__main__":
    out = build()
    if "--check" in sys.argv:
        cur = open(DST, encoding="utf-8", newline="").read()
        print("clr_read.py == generated" if cur == out else "clr_read.py DIFFERS from generated")
        sys.exit(0 if cur == out else 1)
    open(DST, "w", encoding="utf-8", newline="\n").write(out)
    print("wrote", DST, len(out))
