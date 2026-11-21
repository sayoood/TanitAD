#!/usr/bin/env python3
"""make_pins_lf5.py -- writes PINS.txt of the LF5 build (keys read by lf5_chain.sh / lf5_read.py: patched_tool, level_floor, model, data, trainer). Run LAST, after the tests, the mutation proof and the known answers."""
import hashlib
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
MC = "E:/Projects/TanitAD-artifacts/dagland/motion_c"
TREE = "F:/TanitAD-scratch/motion_c/lf5_read/tree"
LFO_TOOL_SHA = "6744d3e4686e344fb72e02ad8f659c779eb73261de13c4a3844e15243667dcc6"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    tool, tree_tool = sha(f"{PK}/tool/v8f_t1_eval_ri_lf5.py"), sha(f"{TREE}/stack/scripts/v8f_t1_eval_ri.py")
    assert tool == tree_tool, "the tree's tool differs from lf5_build/tool"
    out = f"{PK}/_regen.py"
    subprocess.run([sys.executable, "-I", f"{PK}/code/apply_lf5_patch.py", f"{MC}/lfo_build/tool/v8f_t1_eval_ri_lfo.py", out], check=True, capture_output=True)
    regen = sha(out)
    os.remove(out)
    assert sha(f"{MC}/lfo_build/tool/v8f_t1_eval_ri_lfo.py") == LFO_TOOL_SHA
    L = ["# PINS of the LF5 build (sha256, full)", "", "# keys (key=value; the chain and the reader read patched_tool and level_floor)", f"patched_tool={tool}", f"patched_tool_path={TREE}/stack/scripts/v8f_t1_eval_ri.py",
         f"lfo_tool={LFO_TOOL_SHA}", f"level_floor={sha(f'{TREE}/stack/scripts/level_floor.py')}", f"model={sha(f'{TREE}/stack/tanitad/models/v8f_mini.py')}", f"data={sha(f'{TREE}/stack/tanitad/data/v8f_mini_data.py')}",
         f"trainer={sha(f'{TREE}/stack/scripts/v8f_mini_train.py')}", f"a5_card={sha(f'{MC}/a5_prereg/PREREG_LEVELFLOOR_A5.md')}", f"regeneration_check={'YES' if regen == tool else 'NO'}", "",
         f"written {time.strftime('%Y-%m-%dT%H:%M:%S%z')}"]
    for f in ("code/apply_lf5_patch.py", "code/ka_lf5.py", "code/run_mutations_lf5.py", "code/make_pins_lf5.py", "tests/test_lf5.py", "raw/KA_LF5.json", "logs/MUTATIONS_LF5.txt", "logs/pytest_lf5.log"):
        L.append(f"{sha(f'{PK}/{f}')}  {f}")
    open(f"{PK}/PINS.txt", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("regeneration check:", regen == tool, "patched_tool", tool[:16])
    return 0 if regen == tool else 1


if __name__ == "__main__":
    sys.exit(main())
