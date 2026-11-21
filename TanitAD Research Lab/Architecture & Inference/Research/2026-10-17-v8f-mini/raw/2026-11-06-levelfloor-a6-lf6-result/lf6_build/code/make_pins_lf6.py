#!/usr/bin/env python3
"""make_pins_lf6.py -- writes PINS.txt of the LF6 build (keys read by lf6_chain.sh / lf6_read.py: patched_tool, level_floor, model, data, trainer). Run LAST, after the tests, the mutation proof and the known answers."""
import hashlib
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
MC = "E:/Projects/TanitAD-artifacts/dagland/motion_c"
TREE = "F:/TanitAD-scratch/motion_c/lf6_read/tree"
LF5_TOOL_SHA = "88b275a1ca67894be01907f72b71302b72f6e0b4fa0af067f4e2c3d5286aba77"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    tool, tree_tool = sha(f"{PK}/tool/v8f_t1_eval_ri_lf6.py"), sha(f"{TREE}/stack/scripts/v8f_t1_eval_ri.py")
    assert tool == tree_tool, "the tree's tool differs from lf6_build/tool"
    out = f"{PK}/_regen.py"
    subprocess.run([sys.executable, "-I", f"{PK}/code/apply_lf6_patch.py", f"{MC}/lf5_build/tool/v8f_t1_eval_ri_lf5.py", out], check=True, capture_output=True)
    regen = sha(out)
    os.remove(out)
    assert sha(f"{MC}/lf5_build/tool/v8f_t1_eval_ri_lf5.py") == LF5_TOOL_SHA
    L = ["# PINS of the LF6 build (sha256, full)", "", "# keys (key=value; the chain and the reader read patched_tool and level_floor)", f"patched_tool={tool}", f"patched_tool_path={TREE}/stack/scripts/v8f_t1_eval_ri.py",
         f"lf5_tool={LF5_TOOL_SHA}", f"level_floor={sha(f'{TREE}/stack/scripts/level_floor.py')}", f"model={sha(f'{TREE}/stack/tanitad/models/v8f_mini.py')}", f"data={sha(f'{TREE}/stack/tanitad/data/v8f_mini_data.py')}",
         f"trainer={sha(f'{TREE}/stack/scripts/v8f_mini_train.py')}", f"a6_card={sha(f'{MC}/a6_prereg/PREREG_LEVELFLOOR_A6.md')}", f"regeneration_check={'YES' if regen == tool else 'NO'}", "",
         f"written {time.strftime('%Y-%m-%dT%H:%M:%S%z')}"]
    for f in ("code/apply_lf6_patch.py", "code/ka_lf6.py", "code/run_mutations_lf6.py", "code/make_pins_lf6.py", "tests/test_lf6.py", "raw/KA_LF6.json", "logs/MUTATIONS_LF6.txt", "logs/pytest_lf6.log"):
        L.append(f"{sha(f'{PK}/{f}')}  {f}")
    open(f"{PK}/PINS.txt", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("regeneration check:", regen == tool, "patched_tool", tool[:16])
    return 0 if regen == tool else 1


if __name__ == "__main__":
    sys.exit(main())
