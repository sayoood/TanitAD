#!/usr/bin/env python3
"""make_pins.py -- writes PINS.txt (sha256 of every file the LF read chain must find unchanged) and MANIFEST.txt (every file of the package).  lf_read_chain.sh refuses to start (exit 2) when a pinned code / test / script file
differs, when the tree's tool differs from lf_build/PINS.txt.  Run LAST, after the tests, the mutation proof and the known answers.   usage: python -I code/make_pins.py"""
import hashlib
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
CODE = ["lf_read.py", "lf_precheck.py", "run_mutations_lf.py", "make_pins.py"]
TESTS = ["test_lf_read.py", "test_lf_chain_gates.py"]
OTHER = ["lf_read_chain.sh", "lf_read_chain.cmd"]
SHARED = [("llr_read_py", f"{ART}/dagland/motion_c/llr_read/code/llr_read.py"), ("llr_read_json", f"{ART}/dagland/motion_c/llr_read/raw/LLR_READ.json"), ("a3_card", f"{ART}/dagland/motion_c/a3_prereg/PREREG_LEVELFLOOR_A3.md")]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    L = [f"# LF read pins (written {time.strftime('%Y-%m-%dT%H:%M:%S%z')}); lf_read_chain.sh refuses to start when a pinned file differs."]
    for k, p in SHARED:
        L.append(f"{k} sha256 {sha(p)}")
    for f in [f"code/{c}" for c in CODE] + [f"tests/{t}" for t in TESTS] + OTHER:
        L.append(f"{f} sha256 {sha(f'{PK}/{f}')}")
    open(f"{PK}/PINS.txt", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    M = ["# MANIFEST: package E:/Projects/TanitAD-artifacts/dagland/motion_c/lf_read (staged, NOT in git; the Master Mind lands).  scratch: F:/TanitAD-scratch/motion_c/lf_read/ (tree/, out/, smoke/, work/, pkgtest/)", "# path\tbytes\tsha256"]
    for root, dirs, files in os.walk(PK):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".pytest_cache"))
        for f in sorted(files):
            p = os.path.join(root, f).replace("\\", "/")
            rel = p[len(PK) + 1:]
            if rel != "MANIFEST.txt":
                M.append(f"{rel}\t{os.path.getsize(p)}\t{sha(p)}")
    open(f"{PK}/MANIFEST.txt", "w", encoding="utf-8", newline="\n").write("\n".join(M) + "\n")
    print("wrote PINS.txt (%d lines) and MANIFEST.txt (%d files)" % (len(L) - 1, len(M) - 2))


if __name__ == "__main__":
    main()
