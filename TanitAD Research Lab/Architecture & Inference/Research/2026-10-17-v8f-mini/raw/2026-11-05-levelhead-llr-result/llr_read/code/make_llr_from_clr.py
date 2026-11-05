#!/usr/bin/env python3
"""make_llr_from_clr.py -- builds code/llr_read.py from the CLR reader (clr_read/code/clr_read.py, sha256 pinned below).  Role names are renamed (RICLRC -> RILLRC, RICLR -> RILLR, FZCLR -> FZLLR, CLRC -> LLRC,
CLR -> LLR) AFTER the attribution base is protected (CLR's old base RIRAPRP01 / RAPRP01 become the placeholders for the roles RICLR / CLR, which are now the BANKED attribution base); every other change is one
EXACT-ONCE replacement listed in llr_edits.py (a replacement that does not match exactly once aborts).  This script is the provenance of the diff against CLR's reader.
usage: python -I make_llr_from_clr.py [--check]     (--check: regenerate in memory and compare with code/llr_read.py; exit 1 on a difference)"""
import hashlib
import importlib.util
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
SRC = "E:/Projects/TanitAD-artifacts/dagland/motion_c/clr_read/code/clr_read.py"
SRC_SHA = "ba12db24038c31d8daf34c25a81b1c3f35e16a3c96d792c31e4e0be7e584bc43"
DST = f"{PK}/code/llr_read.py"
RENAMES = [("RICLRC", "RILLRC"), ("RICLR", "RILLR"), ("FZCLR", "FZLLR"), ("CLRC", "LLRC"), ("CLR", "LLR")]
PROTECT = [("RIRAPRP01", "@@BASERI@@"), ("RAPRP01", "@@BASELG@@")]
_s = importlib.util.spec_from_file_location("llr_edits", f"{HERE}/llr_edits.py")
LE = importlib.util.module_from_spec(_s)
_s.loader.exec_module(LE)


def sub_tokens(body, pairs):
    for a, b in pairs:
        body = re.sub(r"(?<![A-Za-z0-9])" + re.escape(a) + r"(?![A-Za-z0-9])", b, body)
    return body


def apply(t, edits):
    for e in edits:
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


def build():
    raw = open(SRC, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == SRC_SHA, "clr_read.py is not the pinned file"
    t = raw.decode("utf-8")
    i = t.index('"""', t.index('"""') + 3) + 3
    body = sub_tokens(t[i:], PROTECT)
    body = sub_tokens(body, RENAMES)
    body = body.replace("@@BASERI@@", "RICLR").replace("@@BASELG@@", "CLR").replace("clr_", "llr_")
    t = '#!/usr/bin/env python3\n"""' + LE.DOC_NEW.strip("\n") + '\n"""' + body
    return apply(t, LE.EDITS)


if __name__ == "__main__":
    out = build()
    if "--check" in sys.argv:
        cur = open(DST, encoding="utf-8", newline="").read()
        print("llr_read.py == generated" if cur == out else "llr_read.py DIFFERS from generated")
        sys.exit(0 if cur == out else 1)
    open(DST, "w", encoding="utf-8", newline="\n").write(out)
    print("wrote", DST, len(out))
