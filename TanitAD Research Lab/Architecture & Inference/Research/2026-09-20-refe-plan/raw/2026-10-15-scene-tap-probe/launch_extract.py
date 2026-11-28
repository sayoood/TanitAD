"""Run extract_scene_ctx.py with the nuPlan environment the seam / c3_run use (eval_checkpoint.env_driverl()), from the driverl-eval venv.
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe launch_extract.py [extract_scene_ctx.py args ...]
Exit codes are the script's (0 ok | 4 refused | 5 GPU busy | 6 not the same forward | 7 identity failure)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PKG, "eval"))
import eval_checkpoint as EC  # noqa: E402

cmd = [EC.DRIVERL_PY, os.path.join(HERE, "extract_scene_ctx.py")] + sys.argv[1:]
print("CMD", cmd, flush=True)
sys.exit(subprocess.call(cmd, cwd=os.path.join(PKG, "eval"), env=EC.env_driverl()))
