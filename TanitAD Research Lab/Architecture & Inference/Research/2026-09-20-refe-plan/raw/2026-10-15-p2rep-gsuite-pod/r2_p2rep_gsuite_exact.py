"""P2-s1 G-SUITE pre-run through the GATE'S OWN CODE (Master Mind 2026-10-15): gate_checks.check_suite(PROFILES["refe_p2rep"]) in-process, from the shipped
tree, with the environment p2rep_pod.sh gives launch_gate.py on the pod: P2REP_S1_ABANDONED=1 (the launch override that broke the 05:04Z run),
TANITAD_GATE_KEY_FILE, OMP_NUM_THREADS=4, HF_TOKEN absent.  The GPU is hidden (S1 trains on it).  Writes the verdict JSON + ZZPRERUN_DONE."""
import json
import os
import sys
import time

HERE = "/workspace/refe-plan-p2rep/refe"
os.environ.update({"P2REP_S1_ABANDONED": "1", "TANITAD_GATE_KEY_FILE": "/workspace/data/refe_p2rep/launch_gate.key", "OMP_NUM_THREADS": "4",
                   "CUDA_VISIBLE_DEVICES": "", "PKG": "/workspace/refe-plan-p2rep"})
os.environ.pop("HF_TOKEN", None)
sys.path.insert(0, HERE)
os.chdir(HERE)
import gate_checks as G  # noqa: E402
import launch_gate as L  # noqa: E402

t0 = time.time()
v = G.check_suite(L.PROFILES["refe_p2rep"])
out = {"seconds": round(time.time() - t0, 1), "verdict": v}
open("/workspace/ops/p2rep_gsuite_exact.json", "w").write(json.dumps(out, indent=1, default=str))
print(json.dumps({"seconds": out["seconds"], "status": v.get("status"), "reasons": v.get("reasons"),
                  "scripts": (v.get("details") or {}).get("scripts")}, default=str))
print("ZZPRERUN_DONE")
