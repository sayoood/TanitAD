"""Run each test file on the FINAL tree (the export) in its own pytest process; print one summary line per file."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.argv[2]
FILES = sys.argv[3:]
env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.pathsep.join([str(ROOT / "stack"), str(ROOT / "taniteval")])}
tot = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
for f in FILES:
    t0 = time.time()
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"stack/tests/{f}"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    lines = [x for x in r.stdout.strip().splitlines() if x.strip()]
    last = lines[-1] if lines else "<no output>"
    fails = [x for x in lines if x.startswith(("FAILED", "ERROR"))]
    print(f"{f}: rc={r.returncode} | {last} | {time.time() - t0:.0f}s", flush=True)
    for x in fails[:8]:
        print("    ", x[:200], flush=True)
    for k in tot:
        import re
        m = re.search(rf"(\d+) {k}", last)
        if m:
            tot[k] += int(m.group(1))
print("TOTAL", tot, flush=True)
