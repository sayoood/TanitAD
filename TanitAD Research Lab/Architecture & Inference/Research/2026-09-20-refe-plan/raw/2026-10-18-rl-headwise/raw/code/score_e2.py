"""RL-1 V1 relaunch wrapper (Master Mind, 2026-10-18). Runs the SAME score_e.py (W3 harness, unchanged) with ONE
process-local fix: os.path.relpath falls back to the absolute path when the two paths are on different drives.

Why: all three first attempts crashed at run_v1.py:459 `os.path.relpath(dst, PKG)` (dst on C:, PKG on E:) AFTER the
RAM guard had aborted the devkit, so the row-count checks C2/C3 never ran and a 4,433-row partial reconstruction from
attempt 1 sat in the run dir. The relpath value is a report label only; no score, check or path that is read depends on it.
    python score_e2.py <score_e.py args>
"""
import os
import runpy
import sys

SP = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/a085b71c-4fb3-409b-ae6b-65f605ed575e/scratchpad"
_relpath = os.path.relpath


def _relpath_xdrive(path, start=os.curdir):
    try:
        return _relpath(path, start)
    except ValueError:  # "path is on mount 'C:', start on mount 'E:'"
        return os.path.abspath(path).replace(os.sep, "/")


os.path.relpath = _relpath_xdrive
sys.argv = [f"{SP}/score_e.py"] + sys.argv[1:]
runpy.run_path(f"{SP}/score_e.py", run_name="__main__")
