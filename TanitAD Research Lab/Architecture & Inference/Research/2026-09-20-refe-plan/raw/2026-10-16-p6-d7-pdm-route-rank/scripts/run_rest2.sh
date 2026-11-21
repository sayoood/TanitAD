#!/bin/bash
# D7 verification, part 2: rank-3 twins (the bank carries 3,390 of them; the MM's disk estimate listed only ranks 1+2), the `obs` canon diagnosis, the final-tree unit tests.
# Sequential, <= 2 CPU processes at any time, scratch under /workspace/ops/d7fix only.
set -u
source /workspace/teacher_env.sh
D=/workspace/ops/d7fix
PKG=$D/pkg
export PYTHONPATH=$PKG/code:${PYTHONPATH:-}
cd $PKG/refe
L=$D/logs; R=$D/results
PY=$DRIVERL_EVAL_PYTHON
DBS=/workspace/data/navtrain_dbs
run() { env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK -u REFE_D7_MUTATE -u REFE_PAUSE_FILE OMP_NUM_THREADS=1 nice -n 15 "$@"; }
stamp() { echo "$(date -u +%FT%TZ) $*" >> $L/rest2_progress.log; }

# ---------- rank 3: 9 twins from 9 logs (seeded), through the REAL cmd_build + the verifier
$PY - <<'PYEOF'
import json, os, random
D = "/workspace/ops/d7fix"
rng = random.Random(0)
by = {}
for ln in open("/workspace/data/refe_navtrain/train_grow_cmd/targets_rank1.jsonl"):
    r = json.loads(ln)
    a = r.get("aug")
    if isinstance(a, dict) and a.get("kind") == "lane_rank" and int(a["value"]) == 3:
        by.setdefault(r["log_name"], []).append(r)
logs = sorted(by)
rng.shuffle(logs)
rows = [sorted(by[l], key=lambda r: r["token"])[0] for l in logs[:9]]
os.makedirs(D + "/bank_k3", exist_ok=True)
with open(D + "/bank_k3/targets_rank1.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
# the logged parts of those tokens (read-only source) into the scratch cache so the verifier finds both
import shutil
for r in rows:
    s = f"/workspace/data/refe_navtrain/metric_cache/{r['log_name']}/{r['token']}.pkl.xz"
    d = f"{D}/mc_test/{r['log_name']}/{r['token']}.pkl.xz"
    if os.path.isfile(s) and not os.path.isfile(d):
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(s, d)
print("bank_k3", len(rows), "div>=0.5:", sum(r["aug"]["divergence_m"] >= 0.5 for r in rows))
PYEOF
stamp "k3 build start"
run $PY build_metric_cache.py build --targets $D/bank_k3 --out $D/mc_test --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers 2 --lane-rank 3 > $L/build_k3.log 2>&1
stamp "k3 build rc=$?"
run $PY diag_d7_twin_route.py verify --bank $D/bank_k3 --k 3 --twin-cache $D/mc_test --logged-cache $D/mc_test --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --out $R/verify_k3.json > $L/verify_k3.log 2>&1
stamp "k3 verify rc=$?"

# ---------- content-level comparisons (file BYTES are not deterministic across two builds of the same token -- MEASURED -- so the proof is content: canon of the parts + D9's score_plans equality)
REAL=/workspace/data/refe_navtrain/metric_cache
# (1) control: rank-0 rebuilt with the fixed code vs the existing cache entry
for t in "2021.06.08.12.54.54_veh-26_02323_02479 31e76d971d415db9" "2021.07.16.16.27.22_veh-26_02282_03814 13459d66a08c558d"; do
  set -- $t
  run $PY $D/obs_diag.py $D/mc_rank0/$1/$2.pkl.xz $REAL/$1/$2.pkl.xz > $L/obs_diag_control_$2.log 2>&1
done
# (3) chain-style vs standalone twin (two builds of the same token, same code)
f=$(find $D/mc_chainstyle -name "9e642a0d*__lane_rank_1.pkl.xz" | head -1)
g=$D/mc_test/$(basename $(dirname $f))/$(basename $f)
run $PY $D/obs_diag.py $f $g > $L/obs_diag_chain_9e642a0d.log 2>&1
stamp "content diag done"
stamp "ALL DONE 2"
