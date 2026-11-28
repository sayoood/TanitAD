"""One stage-4 rollout cell (AGENT N): the stage-2 cell entry with the composite arms C0C_s0 / C0C_s1 registered first.

    python -I t2v8f_stage4_cell.py [--executor base|hold] [--hold-eps 0.1] <the t2v8f_run.py rollout arguments>

`t2v8f_c0c.install(R)` extends `t2v8f_run`'s ARMS / CKPT_FILE and wraps `R.build_arm` BEFORE `R.setup()` registers the arms (the stage-2 cell calls `R.setup()` inside `apply_executor`), then the
stage-2 cell runs unchanged. Every other arm is built exactly as before; the stage-2 and stage-3 drivers spawn `t2v8f_stage2_cell.py`, which never imports this file.
The RI tool is read from `T2V8F_RI_TOOL` (the stage-4 driver sets it to the pinned byte copy in the package; `t2v8f_c0c.load_ri_tool` refuses any other sha256).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def run_cell(argv):
    import t2v8f_c0c
    import t2v8f_run as R
    import t2v8f_stage2_cell as C
    t2v8f_c0c.install(R)
    return C.run_cell(argv)


if __name__ == "__main__":
    run_cell(sys.argv[1:])
