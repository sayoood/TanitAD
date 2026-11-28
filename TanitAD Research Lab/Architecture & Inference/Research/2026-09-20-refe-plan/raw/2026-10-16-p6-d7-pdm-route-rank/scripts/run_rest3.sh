#!/bin/bash
# D7 verification, part 3: content-level control + proof, the production-regime cost point, the roadblock-disagreement probe, the final-tree test suites.
# Sequential, <= 2 CPU processes at any time (cost run: 1 worker), scratch under /workspace/ops/d7fix only.
set -u
source /workspace/teacher_env.sh
D=/workspace/ops/d7fix
PKG=$D/pkg
export PYTHONPATH=$PKG/code:${PYTHONPATH:-}
cd $PKG/refe
L=$D/logs; R=$D/results
PY=$DRIVERL_EVAL_PYTHON
DBS=/workspace/data/navtrain_dbs
REAL=/workspace/data/refe_navtrain/metric_cache
run() { env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK -u REFE_D7_MUTATE -u REFE_PAUSE_FILE OMP_NUM_THREADS=1 nice -n 15 "$@"; }
stamp() { echo "$(date -u +%FT%TZ) $*" >> $L/rest3_progress.log; }

cp -n $R/control_rank0.json $R/control_rank0_run1_raw_canon.json
stamp "compare rank0 start"
run $PY diag_d7_twin_route.py compare --bank $D/bank_test --a $D/mc_rank0 --b $REAL --out $R/control_rank0.json --n-sets 4 --min-keys 10 > $L/control_rank0_v2.log 2>&1
stamp "compare rank0 rc=$?"
run $PY diag_d7_twin_route.py compare --bank $D/bank_chain --k 1 --a $D/mc_chainstyle --b $D/mc_test --out $R/proof_chainstyle_vs_standalone.json --n-sets 4 --min-keys 6 > $L/proof_chainstyle.log 2>&1
stamp "compare chainstyle rc=$?"

# ---------- cost, production regime: ONE log with many rank-1 twins, 1 worker
$PY - <<'PYEOF'
import json, os, collections
D = "/workspace/ops/d7fix"
by = collections.defaultdict(list)
for ln in open("/workspace/data/refe_navtrain/train_grow_cmd/targets_rank1.jsonl"):
    r = json.loads(ln)
    a = r.get("aug")
    if isinstance(a, dict) and a.get("kind") == "lane_rank" and int(a["value"]) == 1:
        by[r["log_name"]].append(r)
big = sorted(by, key=lambda l: (-len(by[l]), l))
print("largest rank-1 logs:", [(l[:24], len(by[l])) for l in big[:3]], "median twins/log:", sorted(len(v) for v in by.values())[len(by) // 2])
lg = big[0]
os.makedirs(D + "/bank_cost2", exist_ok=True)
rows = by[lg][:40]
with open(D + "/bank_cost2/targets_rank1.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print("bank_cost2", len(rows), "twins in", lg)
PYEOF
stamp "cost2 start"
rm -rf $D/mc_cost2
S=$(date +%s.%N)
run $PY build_metric_cache.py build --targets $D/bank_cost2 --out $D/mc_cost2 --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers 1 --lane-rank 1 > $L/cost2_w1.log 2>&1
E=$(date +%s.%N)
echo "wall_s=$(python3 -c "print(round($E - $S, 1))") loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" >> $L/cost2_w1.log
stamp "cost2 done"

# ---------- where is DriveRL's start roadblock relative to PDM's when they disagree
run $PY $D/probe_blocks.py $R/verify_k1.json $R/verify_k2.json $R/verify_k3.json > $L/probe_blocks.log 2>&1
stamp "probe_blocks done"

# ---------- the final-tree suites (no GPU, no pod resources beyond 1 process each)
run $PY test_pdm_route_rank.py > $L/test_pdm_route_rank.log 2>&1;  echo "rc=$?" >> $L/test_pdm_route_rank.log;  stamp "test_pdm_route_rank rc=$(tail -1 $L/test_pdm_route_rank.log)"
run $PY test_d9_accept.py > $L/test_d9_accept.log 2>&1;            echo "rc=$?" >> $L/test_d9_accept.log;            stamp "test_d9_accept rc=$(tail -1 $L/test_d9_accept.log)"
run timeout 1500 $PY test_p6_chain.py > $L/test_p6_chain.log 2>&1;       echo "rc=$?" >> $L/test_p6_chain.log;       stamp "test_p6_chain rc=$(tail -1 $L/test_p6_chain.log)"
run timeout 900 $PY test_p6_cpu_probe.py > $L/test_p6_cpu_probe.log 2>&1; echo "rc=$?" >> $L/test_p6_cpu_probe.log; stamp "test_p6_cpu_probe rc=$(tail -1 $L/test_p6_cpu_probe.log)"
stamp "ALL DONE 3"
