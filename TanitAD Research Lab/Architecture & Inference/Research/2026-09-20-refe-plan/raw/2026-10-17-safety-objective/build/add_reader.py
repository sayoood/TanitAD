#!/usr/bin/env python3
"""add_reader.py <PK dir> : add the two NEW files of the S-reader (bar S of PREREG_SAFEOBJ.md) to eval/:  eval/safeobj_read.py and eval/test_safeobj_read.py  (sources: build/safeobj_read.py,
build/test_safeobj_read.py).  Neither exists at the tip; an existing file is a hard error (the tree is not pristine).  Touches no existing file."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patchlib as PL  # noqa: E402

FILES = [("safeobj_read.py", "eval/safeobj_read.py"), ("test_safeobj_read.py", "eval/test_safeobj_read.py")]


def main():
    pk = PL.pk_arg()
    here = os.path.dirname(os.path.abspath(__file__))
    for src, rel in FILES:
        dst = os.path.join(pk, rel)
        if os.path.exists(dst):
            raise SystemExit(f"{rel} already exists: refusing to overwrite (is the tree pristine?)")
        text = PL.read(os.path.join(here, src))
        PL.write(dst, text)
        print({"wrote": rel, "blob": PL.git_blob(text), "bytes": len(text.encode("utf-8"))})


if __name__ == "__main__":
    main()
