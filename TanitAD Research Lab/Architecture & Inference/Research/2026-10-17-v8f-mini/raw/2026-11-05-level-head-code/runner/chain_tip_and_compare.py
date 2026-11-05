"""Detached driver: wait for run_files.py new to write ALL-DONE, then run the unmodified tip on the files that failed on NEW, then compare (logs/COMPARE.txt). Polling is a sleep in a script, not a model turn."""
import subprocess
import time
from pathlib import Path

ART = Path("E:/Projects/TanitAD-artifacts/dagland/motion_c/build")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
newlog = ART / "logs" / "files_NEW.log"
while not (newlog.exists() and "ALL-DONE" in newlog.read_text(encoding="utf-8", errors="replace")):
    time.sleep(30)
subprocess.run([PY, str(ART / "runner" / "run_files.py"), "tip", "--only-failing-of", str(newlog)], stdin=subprocess.DEVNULL)
subprocess.run([PY, str(ART / "runner" / "compare_suites.py")], stdin=subprocess.DEVNULL)
