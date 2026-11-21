#!/usr/bin/env python3
"""patch_pyguard.py -- Master Mind 2026-10-21 22:50 Berlin.  Thor's GPU-roll guards counted EVERY python process as "busy", so the
CPU-only, nice-19 NT-DECIDE data pipeline (r8ns/nt/: nt_e1_run.py, nt_joins.py, ...) kept the GPU idle for hours.  This rewrites the
one-line python count in the named scripts so python processes whose args match the NT CPU pipeline are NOT counted.  The
`nvidia-smi --query-compute-apps` check (GPU exclusivity) is untouched.  Writes <file>.bak_pyguard first; refuses unless the old
line occurs exactly once; asserts the new line is present afterwards."""
import shutil
import sys

OLD = "pys=$(ps -u nvidia -o pid=,comm= | awk '$2 ~ /^python/' | wc -l)"
NEW = ("pys=$(ps -u nvidia -o pid=,args= | awk '$2 ~ /python/' | grep -vE 'r8ns/nt/|code/nt_[a-z0-9_]+[.]py' | wc -l)"
       "  # NT-DECIDE CPU pipeline excluded (MM 2026-10-21)")
for f in sys.argv[1:]:
    t = open(f, encoding="utf-8").read()
    n = t.count(OLD)
    if n != 1:
        print(f"REFUSE {f}: old line count {n}")
        continue
    shutil.copyfile(f, f + ".bak_pyguard")
    open(f, "w", encoding="utf-8", newline="\n").write(t.replace(OLD, NEW))
    assert open(f, encoding="utf-8").read().count(NEW) == 1
    print(f"PATCHED {f}")
