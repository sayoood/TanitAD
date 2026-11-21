#!/usr/bin/env python3
"""make_pins.py -- writes PINS.txt (sha256 of every file the CLR read chain must find unchanged) and MANIFEST.txt.  clr_read_chain.sh refuses to start (exit 2) when a pinned code / test / script file differs.
Run LAST, after the tests, the mutation proof and the known answers.   usage: python -I code/make_pins.py"""
import hashlib
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
TREE = "F:/TanitAD-scratch/motion_c/cl_read/tree"
CODE = ["clr_read.py", "clr_precheck.py", "clr_readout.py", "clr_merge_readout.py", "clr_wiring.py", "clr_edits.py", "clr_test_edits.py", "make_clr_from_cl.py", "make_clr_precheck.py", "make_clr_readout.py",
        "make_clr_tests.py", "run_mutations_clr.py", "make_pins.py"]
TESTS = ["test_clr_read.py", "test_clr_chain_gates.py", "test_clr_wiring.py"]
OTHER = ["clr_read_chain.sh", "clr_read_chain.cmd"]
LOGS = ["logs/pytest_clr_read.log", "logs/pytest_clr_chain_gates.log", "logs/pytest_clr_wiring.log", "logs/mutations_clr.log"]
RAW = ["raw/KA_READER_ka-null.json", "raw/KA_READER_ka-ri.json", "raw/KA_READER_ka-cl.json"]
SHARED = [("tool", f"{TREE}/stack/scripts/v8f_t1_eval_ri.py"), ("model", f"{TREE}/stack/tanitad/models/v8f_mini.py"), ("trainer", f"{TREE}/stack/scripts/v8f_mini_train.py"),
          ("cl_ka_new", f"{ART}/dagland/motion_c/cl_read/raw/KA_NEW.json"), ("cl_smoke_level_on", f"{ART}/dagland/motion_c/cl_read/raw/SMOKE_LEVEL_ON.json"), ("cl_read_json", f"{ART}/dagland/motion_c/cl_read/raw/CL_READ.json")]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    L = [f"# CLR read pins (written {time.strftime('%Y-%m-%dT%H:%M:%S%z')}); clr_read_chain.sh refuses to start when a pinned file differs."]
    for k, p in SHARED:
        L.append(f"{k} sha256 {sha(p)}")
    for rel in [f"code/{c}" for c in CODE] + [f"tests/{t}" for t in TESTS] + OTHER + LOGS + RAW:
        L.append(f"{rel} sha256 {sha(f'{PK}/{rel}')}")
    open(f"{PK}/PINS.txt", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    M = [f"# MANIFEST: package {PK} (staged, NOT in git; the Master Mind lands).  scratch: F:/TanitAD-scratch/motion_c/clr_read/ (out/, smoke/, mut/, pt/, work/, readout/); the shared read tree {TREE} is read only.", "# path\tbytes\tsha256"]
    for root, dirs, files in os.walk(PK):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            p = os.path.join(root, f).replace("\\", "/")
            rel = p[len(PK) + 1:]
            if rel == "MANIFEST.txt":
                continue
            M.append(f"{rel}\t{os.path.getsize(p)}\t{sha(p)}")
    open(f"{PK}/MANIFEST.txt", "w", encoding="utf-8", newline="\n").write("\n".join(M) + "\n")
    print("wrote PINS.txt and MANIFEST.txt")


if __name__ == "__main__":
    main()
