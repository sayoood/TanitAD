#!/bin/bash
# D7 verification, remaining pod jobs -- strictly sequential, <= 2 CPU processes at any time (2 pool workers; the parent only waits), scratch under /workspace/ops/d7fix only.
set -u
source /workspace/teacher_env.sh
D=/workspace/ops/d7fix
PKG=$D/pkg
export PYTHONPATH=$PKG/code:${PYTHONPATH:-}
cd $PKG/refe
L=$D/logs; R=$D/results; mkdir -p $L $R
PY=$DRIVERL_EVAL_PYTHON
REAL=/workspace/data/refe_navtrain/metric_cache
DBS=/workspace/data/navtrain_dbs
run() { env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK -u REFE_D7_MUTATE -u REFE_PAUSE_FILE OMP_NUM_THREADS=1 nice -n 15 "$@"; }
stamp() { echo "$(date -u +%FT%TZ) $*" >> $L/rest_progress.log; }

# ---------- mini banks
$PY - <<'PYEOF'
import json, os
D = "/workspace/ops/d7fix"
rows = [json.loads(l) for l in open(D + "/bank_test/targets_rank1.jsonl")]
r1 = [r for r in rows if r["aug"]["value"] == 1]
ver = json.load(open(D + "/results/verify_k1.json"))["twins"]
distinct = {t["token"] for t in ver if "error" not in t and not t["expected_equals_pdm_lane0"]}
def put(name, rs):
    os.makedirs(f"{D}/{name}", exist_ok=True)
    with open(f"{D}/{name}/targets_rank1.jsonl", "w") as fh:
        for r in rs:
            fh.write(json.dumps(r) + "\n")
    print(name, len(rs))
# cost: 4 logs x 3 twins
logs = sorted({r["log_name"] for r in r1})[:4]
put("bank_cost", [r for r in r1 if r["log_name"] in logs][:12])
# mutation + chain-style: 6 twins whose expected lane differs from PDM's own (so a defective build reads 0 % distinct)
six = [r for r in r1 if r["token"] in distinct][:6]
put("bank_mut", six)
put("bank_chain", six)
PYEOF

# ---------- (d) CONTROL: rank-0 parts rebuilt with the fixed code, through the REAL cmd_build, 2 workers
stamp "control build start"
rm -rf $D/mc_rank0
run $PY build_metric_cache.py build --targets $D/bank_test --out $D/mc_rank0 --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers 2 > $L/build_rank0.log 2>&1
stamp "control build rc=$?"
run $PY diag_d7_twin_route.py control --bank $D/bank_test --new $D/mc_rank0 --ref $REAL --out $R/control_rank0.json > $L/control_rank0.log 2>&1
stamp "control compare rc=$?"

# ---------- COST: single worker, 12 twins in 4 logs
stamp "cost start"
rm -rf $D/mc_cost
S=$(date +%s.%N)
run $PY build_metric_cache.py build --targets $D/bank_cost --out $D/mc_cost --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers 1 --lane-rank 1 > $L/cost_w1.log 2>&1
E=$(date +%s.%N)
echo "wall_s=$(python3 -c "print(round($E - $S, 1))") loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" >> $L/cost_w1.log
stamp "cost done"

# ---------- (e) MUTATION 1: initializer removed, guard live -> the build must FAIL loud and write nothing
stamp "mutation1 start"
rm -rf $D/mc_mut1
REFE_D7_MUTATE=no_initializer env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK OMP_NUM_THREADS=1 nice -n 15 timeout 600 $PY build_metric_cache.py build --targets $D/bank_mut --out $D/mc_mut1 --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers 2 --lane-rank 1 > $L/mut1.log 2>&1
echo "rc=$?" >> $L/mut1.log
echo "files_written=$(find $D/mc_mut1 -name '*.pkl.xz' 2>/dev/null | wc -l)" >> $L/mut1.log
stamp "mutation1 done"

# ---------- (e) MUTATION 2: initializer AND guard removed (= P6's defect) -> the build 'succeeds'; (a) and (b) must FAIL
stamp "mutation2 start"
rm -rf $D/mc_mut2
REFE_D7_MUTATE=no_initializer_no_guard env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK OMP_NUM_THREADS=1 nice -n 15 timeout 600 $PY build_metric_cache.py build --targets $D/bank_mut --out $D/mc_mut2 --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers 2 --lane-rank 1 > $L/mut2.log 2>&1
echo "rc=$?" >> $L/mut2.log
run $PY diag_d7_twin_route.py verify --bank $D/bank_mut --k 1 --twin-cache $D/mc_mut2 --logged-cache $D/mc_test --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --out $R/verify_mut2.json > $L/verify_mut2.log 2>&1
echo "rc=$?" >> $L/verify_mut2.log
stamp "mutation2 done"

# ---------- PROOF the chain's later skip is safe: (1) the chain-style invocation (env var + pause-file + other nice/workers) writes BYTE-IDENTICAL files; (2) re-running it over the standalone output skips every file
stamp "chainstyle start"
rm -rf $D/mc_chainstyle
env -u HF_TOKEN DRIVERL_EVAL_ROUTE_LANE_RANK=1 OMP_NUM_THREADS=1 REFE_PAUSE_FILE=$D/no_such_pause_file nice -n 10 $PY build_metric_cache.py build --targets $D/bank_chain --out $D/mc_chainstyle --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers 2 --lane-rank 1 > $L/chainstyle.log 2>&1
$PY - <<'PYEOF' >> $L/chainstyle.log 2>&1
import hashlib, json, os, glob
D = "/workspace/ops/d7fix"
rows = [json.loads(l) for l in open(D + "/bank_chain/targets_rank1.jsonl")]
eq = 0; n = 0
for r in rows:
    a = f"{D}/mc_chainstyle/{r['log_name']}/{r['token']}__lane_rank_1.pkl.xz"
    b = f"{D}/mc_test/{r['log_name']}/{r['token']}__lane_rank_1.pkl.xz"
    ha = hashlib.sha256(open(a, "rb").read()).hexdigest() if os.path.isfile(a) else None
    hb = hashlib.sha256(open(b, "rb").read()).hexdigest() if os.path.isfile(b) else None
    n += 1; eq += int(ha is not None and ha == hb)
    print("ZZPROOF_FILE", r["token"][:8], (ha or "MISSING")[:12], (hb or "MISSING")[:12], ha == hb)
print(f"ZZPROOF_BYTES_IDENTICAL {eq}/{n}")
PYEOF
# (2) skip: chain-style over the standalone output (mc_test already holds these twins): mtimes must not move
find $D/mc_test -name '*__lane_rank_1.pkl.xz' -printf '%p %T@\n' | sort > $D/mtimes_before.txt
env -u HF_TOKEN DRIVERL_EVAL_ROUTE_LANE_RANK=1 OMP_NUM_THREADS=1 REFE_PAUSE_FILE=$D/no_such_pause_file nice -n 10 $PY build_metric_cache.py build --targets $D/bank_chain --out $D/mc_test --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers 2 --lane-rank 1 > $L/chainstyle_skip.log 2>&1
find $D/mc_test -name '*__lane_rank_1.pkl.xz' -printf '%p %T@\n' | sort > $D/mtimes_after.txt
if cmp -s $D/mtimes_before.txt $D/mtimes_after.txt; then echo "ZZPROOF_SKIP_UNCHANGED $(wc -l < $D/mtimes_after.txt) files" >> $L/chainstyle_skip.log; else echo "ZZPROOF_SKIP_CHANGED" >> $L/chainstyle_skip.log; fi
stamp "chainstyle done"

# ---------- p6_prep.lane-distinct on the test cache, as the chain would read it (0.9 and 0.8)
for k in 1 2; do
  for mf in 0.9 0.8; do
    $PY p6_prep.py lane-distinct --bank $D/bank_test --cache $D/mc_test --k $k --n 40 --min-frac $mf > $L/lane_distinct_k${k}_$mf.log 2>&1
    echo "rc=$?" >> $L/lane_distinct_k${k}_$mf.log
  done
done
stamp "ALL DONE"
