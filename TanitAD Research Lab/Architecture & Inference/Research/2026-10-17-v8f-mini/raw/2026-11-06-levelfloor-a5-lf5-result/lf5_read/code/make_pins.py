#!/usr/bin/env python3
"""make_pins.py -- writes PINS.txt (sha256 of every file the LF5 chain must find unchanged) and MANIFEST.txt. lf5_chain.sh refuses to start (exit 2) when a pinned file differs. Run LAST, after the tests, the mutation proof and the known answers.
usage: python -I code/make_pins.py"""
import hashlib
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
MC = "E:/Projects/TanitAD-artifacts/dagland/motion_c"
CODE = ["lf5_read.py", "lf5_precheck.py", "run_mutations_lf5_read.py", "make_pins.py", "sanitize_objseed.py"]
TESTS = ["test_lf5_read.py"]
OTHER = ["lf5_chain.sh", "lf5_chain.cmd"]
SHARED = [("a5_card", f"{MC}/a5_prereg/PREREG_LEVELFLOOR_A5.md"), ("lfo_read_py", f"{MC}/lfo_read/code/lfo_read.py"), ("lfo_precheck_py", f"{MC}/lfo_read/code/lfo_precheck.py"), ("lf_read_py", f"{MC}/lf_read/code/lf_read.py"),
          ("lf_precheck_py", f"{MC}/lf_read/code/lf_precheck.py"), ("llr_read_py", f"{MC}/llr_read/code/llr_read.py"), ("lfo_read_json", f"{MC}/lfo_read/raw/LFO_READ.json")]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    L = [f"# LF5 read pins (written {time.strftime('%Y-%m-%dT%H:%M:%S%z')}); lf5_chain.sh refuses to start when a pinned file differs."]
    for k, p in SHARED:
        L.append(f"{k} sha256 {sha(p)}")
    for f in [f"code/{c}" for c in CODE] + [f"tests/{t}" for t in TESTS] + OTHER:
        L.append(f"{f} sha256 {sha(f'{PK}/{f}')}")
    open(f"{PK}/PINS.txt", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    M = ["# MANIFEST: package E:/Projects/TanitAD-artifacts/dagland/motion_c/lf5_read (staged, NOT in git; the Master Mind lands). scratch: F:/TanitAD-scratch/motion_c/lf5_read/ (tree/, out/, smoke/, work/, pt/)", "# path  bytes  sha256"]
    for root, dirs, files in os.walk(PK):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".pytest_cache") and not d.startswith("_old"))
        for f in sorted(files):
            p = os.path.join(root, f).replace("\\", "/")
            if f in ("MANIFEST.txt",) or f.endswith(".npz"):
                continue
            M.append(f"{os.path.relpath(p, PK).replace(chr(92), '/')}  {os.path.getsize(p)}  {sha(p)}")
    open(f"{PK}/MANIFEST.txt", "w", encoding="utf-8", newline="\n").write("\n".join(M) + "\n")
    print("PINS.txt and MANIFEST.txt written")


if __name__ == "__main__":
    main()
