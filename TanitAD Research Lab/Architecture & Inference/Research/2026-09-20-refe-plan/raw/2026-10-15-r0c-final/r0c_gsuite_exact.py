"""R0c G-SUITE pre-run through the GATE'S OWN CODE (Master Mind 2026-10-15, PI "run R0c"): gate_checks.check_suite(PROFILES["refe_rls1c"]) in-process, from the
shipped tree, with the environment rls1_pod.sh gives launch_gate.py on the pod: TANITAD_GATE_KEY_FILE=$R/launch_gate.key, PKG, REFE_BACKBONE_ROOT, the nuPlan roots,
OMP_NUM_THREADS=4, HF_TOKEN absent.  The GPU is hidden (P2-s1 trains on it) and the caller runs this at nice 19.  Writes the verdict JSON + ZZPRERUN_DONE."""
import json
import os
import sys
import time

PKG = "/workspace/refe-plan-rls1"
HERE = PKG + "/refe"
DATA = "/workspace/data"
os.environ.update({"TANITAD_GATE_KEY_FILE": DATA + "/rls1/launch_gate.key", "OMP_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": "", "PKG": PKG,
                   "REFE_BACKBONE_ROOT": DATA + "/backbones", "NUPLAN_DATA_ROOT": DATA, "REFE_NUPLAN_DB_ROOT": DATA + "/navtrain_dbs",
                   "REFE_NAVTRAIN_YAML": PKG + "/splits/navtrain.yaml", "PYTHONUNBUFFERED": "1"})
os.environ.pop("HF_TOKEN", None)
sys.path.insert(0, HERE)
os.chdir(HERE)
import gate_checks as G  # noqa: E402
import launch_gate as L  # noqa: E402

t0 = time.time()
v = G.check_suite(L.PROFILES["refe_rls1c"])
out = {"seconds": round(time.time() - t0, 1), "verdict": v}
open("/workspace/ops/r0c_gsuite_exact.json", "w").write(json.dumps(out, indent=1, default=str))
print(json.dumps({"seconds": out["seconds"], "status": v.get("status"), "reasons": v.get("reasons"),
                  "scripts": (v.get("details") or {}).get("scripts")}, default=str))
print("ZZPRERUN_DONE")
