#!/usr/bin/env python3
"""make_pins.py -- writes PINS.txt (the sha256 lines cl_read_chain.sh gate 0 checks) and MANIFEST.txt (every deliverable of this package with its size, sha256 and where it lives).
Run LAST, after the unit-test logs, the mutation log and the known answers exist.   usage: python -I code/make_pins.py"""
import hashlib
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
TREE = "F:/TanitAD-scratch/motion_c/cl_read/tree"
PINNED_FILES = ["code/cl_read.py", "code/cl_precheck.py", "code/ka_new.py", "code/ka_new_summary.py", "code/run_tool_subset.py", "code/smoke_level_on.py", "code/build_read_tree.py", "code/make_pins.py",
                "code/run_mutations_cl.py", "tests/test_cl_read.py", "tests/test_cl_chain_gates.py", "cl_read_chain.sh", "cl_read_chain.cmd"]
PINNED_LOGS = ["logs/pytest_cl_read.log", "logs/pytest_cl_chain_gates.log", "logs/mutations_cl.log", "logs/ka_new_summary.out", "logs/smoke_level_on.out"]
PINNED_RAW = ["raw/KA_NEW.json", "raw/SMOKE_LEVEL_ON.json", "raw/KA_READER_ka-null.json", "raw/KA_READER_ka-ri.json", "raw/REHEARSAL.json"]
TREE_FILES = {"tool": "stack/scripts/v8f_t1_eval_ri.py", "model": "stack/tanitad/models/v8f_mini.py", "trainer": "stack/scripts/v8f_mini_train.py"}
EXPECT = {"tool": "0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7", "model": "a363427677c881cb7c82cf20fcfad317e864a9cde58cf58306e9fed24ef3658e",
          "trainer": "9ffdf1b4788358d3eb2698f11ca4008bffb042c5ee3887995bac4f01adb390ae"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import json
    rj = json.load(open(f"{PK}/raw/REHEARSAL.json", encoding="utf-8"))
    assert rj["PASS"] is True and rj.get("reader_sha256") == sha(f"{PK}/code/cl_read.py"), "raw/REHEARSAL.json is not a PASS of THIS reader (re-run code/rehearsal_production.py)"
    for k in ("ka-null", "ka-ri"):
        kj = json.load(open(f"{PK}/raw/KA_READER_{k}.json", encoding="utf-8"))
        assert kj["KA_PASS"] is True and kj["reader_sha256"] == sha(f"{PK}/code/cl_read.py"), f"raw/KA_READER_{k}.json is not a PASS of THIS reader (re-run code/cl_read.py --mode {k})"
    lines = [f"# CL read pins (written {time.strftime('%Y-%m-%dT%H:%M:%S%z')}); cl_read_chain.sh refuses to start when a pinned file differs."]
    for k, rel in TREE_FILES.items():
        got = sha(f"{TREE}/{rel}")
        assert got == EXPECT[k], f"{k}: the read tree's {rel} is {got}, not {EXPECT[k]}"
        lines.append(f"{k} sha256 {got}")
    for rel in PINNED_FILES + PINNED_LOGS + PINNED_RAW:
        lines.append(f"{rel} sha256 {sha(f'{PK}/{rel}')}")
    open(f"{PK}/PINS.txt", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    skip_dirs = {"__pycache__", "arrays"}
    m = [f"# MANIFEST: package {PK} (staged, NOT in git; the Master Mind lands).  scratch: F:/TanitAD-scratch/motion_c/cl_read/ (tree/, smoke/, mut/, pt/, rehearsal/, work/); nothing else was written.",
         "# path\tbytes\tsha256"]
    for root, dirs, files in os.walk(PK):
        dirs[:] = sorted(d for d in dirs if d not in skip_dirs)
        for f in sorted(files):
            p = os.path.join(root, f).replace("\\", "/")
            rel = p[len(PK) + 1:]
            if rel in ("MANIFEST.txt",) or f.endswith(".pyc"):
                continue
            m.append(f"{rel}\t{os.path.getsize(p)}\t{sha(p)}")
    open(f"{PK}/MANIFEST.txt", "w", encoding="utf-8", newline="\n").write("\n".join(m) + "\n")
    print(f"PINS.txt {len(lines) - 1} lines; MANIFEST.txt {len(m) - 2} files")


if __name__ == "__main__":
    main()
