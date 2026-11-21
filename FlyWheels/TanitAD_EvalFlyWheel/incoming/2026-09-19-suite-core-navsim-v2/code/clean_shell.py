#!/usr/bin/env python3
"""Run a command in a CLEAN, EXPLICIT environment (the W1 acceptance's "clean shell").

⚠️ Why not ``env -i`` in MSYS bash: MSYS hands the child a POSIX-style ``PATH``
(``/c/Windows/System32``), which Windows' ``CreateProcess`` does not understand — MEASURED
2026-09-20: the suite's ``gpu-gap`` probe then could not find ``nvidia-smi`` and reported the
memory as UNKNOWN (the conservative direction — no gap — but for the wrong reason). This launcher
builds the environment as a dict, so what the child gets is exactly what is printed.

    python clean_shell.py -- <program> [args...]

Kept variables: SystemRoot, PATH (venv Scripts + System32 + Wbem), TEMP, TMP, USERPROFILE,
PYTHONPATH, PYTHONIOENCODING. ⛔ NOTHING NavSim-related (no NUPLAN_*, NAVSIM_*, OPENSCENE_*, no
PYTHONHASHSEED): the suite must set every one of them itself, or the acceptance is not clean.
"""
from __future__ import annotations

import json
import subprocess
import sys

ENV = {
    "SystemRoot": r"C:\Windows",
    "PATH": r"C:\Users\Admin\venvs\tanitad\Scripts;C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem",
    "TEMP": r"C:\Users\Admin\AppData\Local\Temp",
    "TMP": r"C:\Users\Admin\AppData\Local\Temp",
    "USERPROFILE": r"C:\Users\Admin",
    "PYTHONPATH": "D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval",
    "PYTHONIOENCODING": "utf-8",
}
FORBIDDEN_PREFIXES = ("NUPLAN", "NAVSIM", "OPENSCENE", "PYTHONHASHSEED", "OMP_", "MKL_", "CUDA")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    if not argv:
        print(__doc__)
        return 2
    bad = [k for k in ENV if k.upper().startswith(FORBIDDEN_PREFIXES)]
    assert not bad, f"the clean environment must not preset {bad}"
    print("CLEAN_ENV=" + json.dumps(ENV, indent=1), flush=True)
    print("CMD=" + " ".join(argv), flush=True)
    return subprocess.run(argv, env=ENV, cwd="D:/Projects/TanitAD").returncode


if __name__ == "__main__":
    sys.exit(main())
