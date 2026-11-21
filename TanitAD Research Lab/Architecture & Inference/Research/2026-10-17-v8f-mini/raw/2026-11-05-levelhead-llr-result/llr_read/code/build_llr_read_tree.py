#!/usr/bin/env python3
"""build_llr_read_tree.py -- the LLR read tree on F:  the shared CL read tree (level-head stack) with THREE files replaced by the code commit 83c9ee0 blobs (the lead head, the build the LLR
checkpoints were trained with); the RI tool is kept.

  SRC   F:/TanitAD-scratch/motion_c/cl_read/tree/                 (READ ONLY: model a3634276, data 209da7a6, trainer 9ffdf1b4, RI tool 0417c838)
  BLOBS `git -C C:/Users/Admin/tanitad-push show 83c9ee0:<path>`   the full sha256 pins are read from llr_launch/PINS.json ("stack"), whose prefixes this script asserts: v8f_mini.py 5fd4baea,
        v8f_mini_data.py 22672d87, v8f_mini_train.py e808cf1a (and v8f_recover.py, which the tree already carries, must equal its pin)
  DST   F:/TanitAD-scratch/motion_c/llr_read/tree/                = a copy of SRC with the three files replaced; stack/scripts/v8f_t1_eval_ri.py (0417c838) kept.

The blob bytes are read as BYTES from git (no autocrlf) and sha256-checked against the pin BEFORE they are written and again after; a destination that exists is MOVED to _old_<time>, never deleted.
usage: python -I build_llr_read_tree.py"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SRC = "F:/TanitAD-scratch/motion_c/cl_read/tree"
DST = "F:/TanitAD-scratch/motion_c/llr_read/tree"
GIT = "C:/Users/Admin/tanitad-push"
COMMIT = "83c9ee0dc4e7d73370c8506e671d8c3e22db2edc"
LAUNCH_PINS = "E:/Projects/TanitAD-artifacts/dagland/motion_c/llr_launch/PINS.json"
REPLACE = {"stack/tanitad/models/v8f_mini.py": "5fd4baea", "stack/tanitad/data/v8f_mini_data.py": "22672d87", "stack/scripts/v8f_mini_train.py": "e808cf1a"}
KEEP_RECOVER = ("stack/tanitad/data/v8f_recover.py", "0257ccb9")
TOOL_REL, TOOL_SHA = "stack/scripts/v8f_t1_eval_ri.py", "0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7"


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pins():
    d = json.load(open(LAUNCH_PINS, encoding="utf-8"))
    return {e["rel"]: e["sha256"] for e in d["stack"]}


def blob(rel):
    r = subprocess.run(["git", "-C", GIT, "show", f"{COMMIT}:{rel}"], capture_output=True)
    if r.returncode:
        raise SystemExit(f"REFUSED: git show {COMMIT[:7]}:{rel} failed: {r.stderr.decode(errors='replace')[:300]}")
    return r.stdout


def main():
    pn = pins()
    for rel, pre in REPLACE.items():
        assert pn[rel].startswith(pre), f"{rel}: llr_launch/PINS.json carries {pn[rel][:8]}, expected {pre}"
    assert pn[KEEP_RECOVER[0]].startswith(KEEP_RECOVER[1])
    assert sha(f"{SRC}/{TOOL_REL}") == TOOL_SHA, "the shared tree's RI tool is not 0417c838"
    blobs = {}
    for rel in list(REPLACE) + [KEEP_RECOVER[0]]:
        b = blob(rel)
        assert sha_bytes(b) == pn[rel], f"REFUSED: the 83c9ee0 blob of {rel} is {sha_bytes(b)}, not the pin {pn[rel]}"
        blobs[rel] = b
    if os.path.exists(DST):
        old = f"{DST}_old_{time.strftime('%H%M%S')}"
        os.rename(DST, old)
        print(f"BUILDTREE moved the existing tree to {old}")
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(SRC, DST, ignore=ign)
    for rel in list(REPLACE) + [KEEP_RECOVER[0]]:
        with open(f"{DST}/{rel}", "wb") as f:
            f.write(blobs[rel])
        assert sha(f"{DST}/{rel}") == pn[rel], f"{rel}: the copy differs from the pin"
    assert sha(f"{DST}/{TOOL_REL}") == TOOL_SHA
    n = sum(len(fs) for _, _, fs in os.walk(DST))
    print(f"BUILDTREE OK {DST}: {n} files; model {pn['stack/tanitad/models/v8f_mini.py'][:8]} data {pn['stack/tanitad/data/v8f_mini_data.py'][:8]} trainer {pn['stack/scripts/v8f_mini_train.py'][:8]} "
          f"recover {KEEP_RECOVER[1]} tool {TOOL_SHA[:8]} verified (blobs of {COMMIT[:7]})")


if __name__ == "__main__":
    main()
