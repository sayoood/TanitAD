"""gpu_watch_r8: our refcv8 roll is OURS, a real foreign job stays FOREIGN, and the upstream allow-list alone is the defect.

The command lines are LITERALS copied from the 2026-10-18 raw-30k watch log and the dev box process table, not
expressions over the code under test.
"""
from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_watch_r8  # noqa: E402

OUR_ROLL = ("C:\\Users\\Admin\\venvs\\tanitad\\Scripts\\python.exe C:\\Users\\Admin\\qland\\hold_r8batt\\stage\\FlyWheels\\"
            "TanitAD_EvalFlyWheel\\incoming\\2026-10-15-refcv8-milestone-battery\\code\\roll_seed_r8.py --ckpt E:/refcv8")
FOREIGN = ("\"C:\\Users\\Admin\\AppData\\Roaming\\uv\\python\\cpython-3.11-windows-x86_64-none\\python.exe\"  "
           "C:\\Users\\Admin\\srttc_s0\\code\\srttc_s0_dump.py --ckpt D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt")

needs_upstream = pytest.mark.skipif(not os.path.isfile(os.path.join(gpu_watch_r8.BAT, "gpu_watch.py")),
                                    reason="upstream gpu_watch.py not present at R8_BAT")


def _fake_q(cmds):
    def q(args, timeout=60):
        if args[0] == "nvidia-smi":
            return "".join(f"{pid}, python.exe\n" for pid in cmds)
        return cmds[int(re.search(r"ProcessId=(\d+)", args[-1]).group(1))]
    return q


@needs_upstream
def test_our_roll_is_ours_and_foreign_stays_foreign():
    gw = gpu_watch_r8.load()
    gw._q = _fake_q({1648: OUR_ROLL, 7852: FOREIGN})
    snap = {x["pid"]: x["ours"] for x in gw.snapshot()}
    assert snap == {1648: True, 7852: False}


@needs_upstream
def test_mutation_upstream_allow_list_alone_reproduces_the_false_collision():
    """Deliberate regression: without the refcv8 entries our own roll reads FOREIGN (the 173/175 defect)."""
    gw = gpu_watch_r8.load()
    gw.OURS = tuple(o for o in gw.OURS if "refcv8" not in o)
    gw._q = _fake_q({1648: OUR_ROLL})
    assert [x["ours"] for x in gw.snapshot()] == [False]
