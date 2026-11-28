"""Adoption run: the unchanged seam with NO --goal-fix flag (so the planner's DEFAULT path, GOAL_FIX = "pdm_route", is what
runs) on the pdm_route-changed tokens outside Amendment 9's confirmation set (316 selection + 25 frame-control) plus 8 of
the registered controls. Together with Amendment 9's 3,045 confirmation rows and the OFF rows of the unchanged tokens it
completes the full 12,146-token navtest under the adopted fix."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_a9_arms as R  # noqa: E402
import run_a9_chunked as C  # noqa: E402

R.FLAGS["adopted"] = []                          # no flag: the default goal path
ts = json.load(open(f"{R.OUT}/tokens_adopted_rest.json", encoding="utf-8"))
me = json.dumps({"job": "refe-adopted-rest", "pid": os.getpid(), "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
R.take_lock(me)
print(f"lock taken {time.strftime('%F %T')}", flush=True)
try:
    env = {k: R.fix(v) for k, v in R.EC.env_driverl().items()}
    ok = C.run_chunk("adopted", "rest", ts["tokens"], ts["token_log"], env)
finally:
    R.release(me)
print("ZZADOPTED_REST", "OK" if ok else "FAILED", flush=True)
