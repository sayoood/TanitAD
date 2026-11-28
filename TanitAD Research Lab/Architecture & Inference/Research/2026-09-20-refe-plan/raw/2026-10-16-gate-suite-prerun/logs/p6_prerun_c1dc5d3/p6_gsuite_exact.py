"""P6 G-SUITE pre-run through the GATE'S OWN CODE (Master Mind 2026-10-16): gate_checks.check_suite(PROFILES["refe_p6"]) in-process,
from P6's shipped tree (/workspace/refe-plan-p6 @ c1dc5d3), whose suite has NEVER run on the pod (last pod suite run 2026-10-14 was an older ship).
Environment = what p6_pod.sh gives launch_gate.py at stage 11:
  - p6_start.env (set -a) + teacher_env.sh (the wrapper sources both);
  - PKG, REFE_BACKBONE_ROOT, PYTHONUNBUFFERED, OMP_NUM_THREADS=4, cwd $PKG/refe;
  - HF_TOKEN and DRIVERL_EVAL_ROUTE_LANE_RANK absent (p6_pod.sh:49, 192-200, 304, 397-498).
Differences, both deliberate:
  - the GPU is hidden (P2-s1 trains on it);
  - TANITAD_GATE_KEY_FILE is a THROWAWAY key outside the tree, so P6's real key is not created early.
Writes /workspace/ops/p6_gsuite_exact.json + ZZPRERUN_DONE."""
import json
import os
import sys
import time

WORK, DATA = "/workspace", "/workspace/data"
PKG = f"{WORK}/refe-plan-p6"
HERE = f"{PKG}/refe"
KEY = "/workspace/ops/p6_prerun_key/gate.key"
os.environ.update({
    "PKG": PKG, "TANITAD_GATE_KEY_FILE": KEY, "REFE_BACKBONE_ROOT": f"{DATA}/backbones", "PYTHONUNBUFFERED": "1",
    "NUPLAN_DATA_ROOT": DATA, "REFE_NUPLAN_DB_ROOT": f"{DATA}/navtrain_dbs", "REFE_NAVTRAIN_YAML": f"{PKG}/splits/navtrain.yaml",
    "OMP_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": ""})
for k in ("HF_TOKEN", "DRIVERL_EVAL_ROUTE_LANE_RANK"):
    os.environ.pop(k, None)
sys.path.insert(0, HERE)
os.chdir(HERE)
import gate_checks as G  # noqa: E402
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402

GT.load_key(KEY, create=True, forbid_under=[PKG])
t0 = time.time()
v = G.check_suite(L.PROFILES["refe_p6"])
out = {"seconds": round(time.time() - t0, 1), "ship": "c1dc5d3", "verdict": v}
open("/workspace/ops/p6_gsuite_exact.json", "w").write(json.dumps(out, indent=1, default=str))
print(json.dumps({"seconds": out["seconds"], "status": v.get("status"), "reasons": v.get("reasons"),
                  "scripts": (v.get("details") or {}).get("scripts")}, default=str))
print("ZZPRERUN_DONE")
