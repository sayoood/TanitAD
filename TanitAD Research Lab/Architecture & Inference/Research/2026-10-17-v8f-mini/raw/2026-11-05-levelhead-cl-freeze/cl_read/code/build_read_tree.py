#!/usr/bin/env python3
"""build_read_tree.py -- the CL read tree: the NEW stack (level-head lever, motion_c/build) + the RI tool, on F:.

  SRC  F:/TanitAD-scratch/motion_c/build/new/{stack,taniteval}   (v8f_mini.py a3634276, v8f_mini_train.py 9ffdf1b4; motion_c/build/SHA_CHANGES.txt)
  TOOL E:/Projects/TanitAD-artifacts/dagland/motion_c/ri/code/v8f_t1_eval_ri.py   (sha256 0417c838...)
  DST  F:/TanitAD-scratch/motion_c/cl_read/tree/   = stack/{tanitad,scripts,pyproject.toml} + taniteval/ (without results/, tests/ and __pycache__) + the tool at stack/scripts/

Own copy, never a link into build/new (the build agent's suite still runs there).  Source shas are checked FIRST (SHA_CHANGES.txt 'new' lines for the model and the trainer); the tool sha is checked before and after
the copy; a destination that exists is MOVED to _old_<time> (never deleted).  usage: python -I build_read_tree.py"""
import hashlib
import os
import re
import shutil
import sys
import time

sys.dont_write_bytecode = True
SRC = "F:/TanitAD-scratch/motion_c/build/new"
DST = "F:/TanitAD-scratch/motion_c/cl_read/tree"
TOOL = "E:/Projects/TanitAD-artifacts/dagland/motion_c/ri/code/v8f_t1_eval_ri.py"
TOOL_SHA = "0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7"
SHA_CHANGES = "E:/Projects/TanitAD-artifacts/dagland/motion_c/build/SHA_CHANGES.txt"
PINNED = {"stack/tanitad/models/v8f_mini.py": "a363427677c881cb7c82cf20fcfad317e864a9cde58cf58306e9fed24ef3658e",
          "stack/scripts/v8f_mini_train.py": "9ffdf1b4788358d3eb2698f11ca4008bffb042c5ee3887995bac4f01adb390ae"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    txt = open(SHA_CHANGES, encoding="utf-8").read()
    for rel, want in PINNED.items():
        assert re.search(rf"{re.escape(rel)}.*?new ({want})", txt, re.S), f"{rel}: pin {want[:8]} is not the 'new' line of SHA_CHANGES.txt"
        got = sha(f"{SRC}/{rel}")
        assert got == want, f"REFUSED: {rel} in the build tree is {got}, not {want}"
    assert sha(TOOL) == TOOL_SHA, f"REFUSED: the RI tool is {sha(TOOL)}, not {TOOL_SHA}"
    if os.path.exists(DST):
        old = f"{DST}_old_{time.strftime('%H%M%S')}"
        os.rename(DST, old)
        print(f"BUILDTREE moved the existing tree to {old}")
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    os.makedirs(f"{DST}/stack", exist_ok=True)
    for d in ("tanitad", "scripts"):
        shutil.copytree(f"{SRC}/stack/{d}", f"{DST}/stack/{d}", ignore=ign)
    shutil.copy2(f"{SRC}/stack/pyproject.toml", f"{DST}/stack/pyproject.toml")
    shutil.copytree(f"{SRC}/taniteval", f"{DST}/taniteval", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "results", "tests"))
    shutil.copy2(TOOL, f"{DST}/stack/scripts/v8f_t1_eval_ri.py")
    for rel, want in PINNED.items():
        assert sha(f"{DST}/{rel}") == want, f"{rel}: the copy differs from the pin"
    assert sha(f"{DST}/stack/scripts/v8f_t1_eval_ri.py") == TOOL_SHA
    n = sum(len(fs) for _, _, fs in os.walk(DST))
    print(f"BUILDTREE OK {DST}: {n} files; model a3634276 trainer 9ffdf1b4 tool 0417c838 verified; results/ and tests/ of taniteval left out")


if __name__ == "__main__":
    main()
