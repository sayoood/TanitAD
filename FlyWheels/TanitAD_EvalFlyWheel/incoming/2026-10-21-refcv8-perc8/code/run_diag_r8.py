"""run_diag_r8.py -- the refcv7 map+box DIAGNOSTICS harness (run_diag.py, unchanged) re-pointed at refcv8 on Thor: r8_repoint.apply()
FIRST (remap file from env R8_REMAP_FILE, pose-sync policy from env R8_NO_POSE_SYNC), then run_diag.main().  Same argv as run_diag.py."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_repoint  # noqa: E402

r8_repoint.apply()
import run_diag  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_diag.main())
