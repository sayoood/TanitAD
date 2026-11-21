"""S1c G-SUITE pre-run through the GATE'S OWN CODE (Master Mind 2026-10-16): gate_checks.check_suite(PROFILES["refe_aux_s1c"]) in-process,
from the shipped tree (/workspace/refe-plan-s1aux @ 10d4fae), with the environment s1aux_pod.sh gives launch_gate.py on the pod:
S1AUX_ARMS="H1u10k-s0", TANITAD_GATE_KEY_FILE=$R/launch_gate.key, REFE_BACKBONE_ROOT, AUX_LABEL_STORE, AUX_NAVTEST, NUPLAN_* / REFE_NUPLAN_*,
OMP_NUM_THREADS=4, HF_TOKEN absent (s1aux_pod.sh:25-54, 204-220, 291, 396).  The GPU is hidden (P2-s1 trains on it).
The caller sources /workspace/teacher_env.sh first (s1aux_pod.sh:204).  Writes /workspace/ops/s1c_gsuite_exact.json + ZZPRERUN_DONE."""
import json
import os
import sys
import time

WORK, DATA = "/workspace", "/workspace/data"
PKG = f"{WORK}/refe-plan-s1aux"
HERE = f"{PKG}/refe"
R = f"{DATA}/s1aux"
os.environ.update({
    "S1AUX_ARMS": "H1u10k-s0", "PKG": PKG, "TANITAD_GATE_KEY_FILE": f"{R}/launch_gate.key", "REFE_BACKBONE_ROOT": f"{DATA}/backbones",
    "PYTHONUNBUFFERED": "1", "AUX_LABEL_STORE": f"{DATA}/aux_drivable/label_store", "AUX_NAVTEST": f"{DATA}/aux_drivable/navtest.json",
    "NUPLAN_DATA_ROOT": DATA, "REFE_NUPLAN_DB_ROOT": f"{DATA}/navtrain_dbs", "REFE_NAVTRAIN_YAML": f"{PKG}/splits/navtrain.yaml",
    "OMP_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": ""})
os.environ.pop("HF_TOKEN", None)
sys.path.insert(0, HERE)
os.chdir(HERE)
import gate_checks as G  # noqa: E402
import launch_gate as L  # noqa: E402

t0 = time.time()
v = G.check_suite(L.PROFILES["refe_aux_s1c"])
out = {"seconds": round(time.time() - t0, 1), "verdict": v}
open("/workspace/ops/s1c_gsuite_exact.json", "w").write(json.dumps(out, indent=1, default=str))
print(json.dumps({"seconds": out["seconds"], "status": v.get("status"), "reasons": v.get("reasons"),
                  "scripts": (v.get("details") or {}).get("scripts")}, default=str))
print("ZZPRERUN_DONE")
