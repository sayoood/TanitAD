"""Run eval/aux_navtest_forward.py with the nuPlan environment the seam / c3_run use (eval_checkpoint.env_driverl()), from the driverl-eval venv.
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe launch.py <out_dir> [aux_navtest_forward.py args ...]
The script takes the shared dev-box GPU lock itself (wait_idle.py + O_EXCL) and releases it on exit; exit codes are the script's (0 ok/partial, 2 identity FAIL, 4 refused, 5 GPU busy)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(os.path.dirname(os.path.dirname(HERE)), "eval")
sys.path.insert(0, EVAL)
import eval_checkpoint as EC  # noqa: E402

if len(sys.argv) < 2:
    sys.exit("usage: launch.py <out_dir> [args ...]")
cmd = [EC.DRIVERL_PY, os.path.join(EVAL, "aux_navtest_forward.py"), "--out", sys.argv[1]] + sys.argv[2:]
print("CMD", cmd, flush=True)
sys.exit(subprocess.call(cmd, cwd=EVAL, env=EC.env_driverl()))
