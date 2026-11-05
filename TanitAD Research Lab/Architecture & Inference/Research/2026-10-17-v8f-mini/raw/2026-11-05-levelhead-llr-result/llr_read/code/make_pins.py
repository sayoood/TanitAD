#!/usr/bin/env python3
"""make_pins.py -- writes PINS.txt (sha256 of every file the LLR read chain must find unchanged) and MANIFEST.txt (every file of the package).  llr_read_chain.sh refuses to start (exit 2) when a pinned code / test / script file
differs, when the LLR tree's tool / model / trainer / data differ, or when the CLR package's evidence differs.  Run LAST, after the tests, the mutation proof and the known answers.   usage: python -I code/make_pins.py"""
import hashlib
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
TREE = "F:/TanitAD-scratch/motion_c/llr_read/tree"
CODE = ["llr_read.py", "llr_precheck.py", "llr_readout.py", "llr_edits.py", "make_llr_from_clr.py", "make_llr_chain.py", "chain_src.sha", "ka_new2.py", "build_llr_read_tree.py", "make_synthetic_llr.py", "run_mutations_llr.py", "make_pins.py"]
TESTS = ["test_llr_read.py", "test_llr_chain_gates.py", "test_llr_readout.py"]
OTHER = ["llr_read_chain.sh", "llr_read_chain.cmd"]
SHARED = [("tool", f"{TREE}/stack/scripts/v8f_t1_eval_ri.py"), ("model", f"{TREE}/stack/tanitad/models/v8f_mini.py"), ("trainer", f"{TREE}/stack/scripts/v8f_mini_train.py"), ("data", f"{TREE}/stack/tanitad/data/v8f_mini_data.py"),
          ("clr_read_json", f"{ART}/dagland/motion_c/clr_read/raw/CLR_READ.json"), ("clr_precheck_json", f"{ART}/dagland/motion_c/clr_read/raw/CLR_PRECHECK.json")]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    L = [f"# LLR read pins (written {time.strftime('%Y-%m-%dT%H:%M:%S%z')}); llr_read_chain.sh refuses to start when a pinned file differs."]
    for k, p in SHARED:
        L.append(f"{k} sha256 {sha(p)}")
    for f in [f"code/{c}" for c in CODE] + [f"tests/{t}" for t in TESTS] + OTHER:
        L.append(f"{f} sha256 {sha(f'{PK}/{f}')}")
    open(f"{PK}/PINS.txt", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    M = ["# MANIFEST: package E:/Projects/TanitAD-artifacts/dagland/motion_c/llr_read (staged, NOT in git; the Master Mind lands).  scratch: F:/TanitAD-scratch/motion_c/llr_read/ (tree/, out/, smoke/, mut/, pt/, work/, readout/)", "# path\tbytes\tsha256"]
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
