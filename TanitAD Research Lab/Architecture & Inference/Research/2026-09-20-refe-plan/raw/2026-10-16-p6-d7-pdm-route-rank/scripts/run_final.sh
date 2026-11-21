#!/bin/bash
# D7 final-tree verification: the FROZEN tree ($PKG) rebuilds the mini-banks from scratch through the real cmd_build and passes (a)(b) again; the skip is cheap; suites re-run.
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
stamp() { echo "$(date -u +%FT%TZ) $*" >> $L/final_progress.log; }
rm -f $L/final_progress.log

# ---- the frozen tree's hashes
( cd $PKG && sha256sum refe/build_metric_cache.py refe/pdm_route_rank.py refe/p6_pod.sh refe/test_p6_chain.py refe/test_pdm_route_rank.py refe/diag_d7_twin_route.py > FIXED_FILES.sha256 )
stamp "hashes written"

# ---- (1) suites that import the changed modules
run $PY test_pdm_route_rank.py > $L/final_test_pdm_route_rank.log 2>&1; stamp "test_pdm_route_rank rc=$? $(tail -2 $L/final_test_pdm_route_rank.log | tr '\n' ' ')"
run $PY test_d9_accept.py > $L/final_test_d9_accept.log 2>&1;           stamp "test_d9_accept rc=$? $(tail -1 $L/final_test_d9_accept.log)"

# ---- (2) skip is cheap and exact: a chain-style run over a complete cache builds nothing and moves no mtime
find $D/mc_test -name '*__lane_rank_1.pkl.xz' -printf '%p %T@\n' | sort > $D/mtimes_before2.txt
S=$(date +%s.%N)
env -u HF_TOKEN DRIVERL_EVAL_ROUTE_LANE_RANK=1 OMP_NUM_THREADS=1 REFE_PAUSE_FILE=$D/no_such_pause_file nice -n 10 $PY build_metric_cache.py build --targets $D/bank_test --out $D/mc_test --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers 2 --lane-rank 1 > $L/final_skip_all.log 2>&1
E=$(date +%s.%N)
find $D/mc_test -name '*__lane_rank_1.pkl.xz' -printf '%p %T@\n' | sort > $D/mtimes_after2.txt
if cmp -s $D/mtimes_before2.txt $D/mtimes_after2.txt; then echo "ZZFINAL_SKIP_UNCHANGED $(wc -l < $D/mtimes_after2.txt) files wall_s=$(python3 -c "print(round($E - $S, 1))")" >> $L/final_skip_all.log; else echo "ZZFINAL_SKIP_CHANGED" >> $L/final_skip_all.log; fi
stamp "skip-all: $(grep -E 'ZZMETRIC|ZZFINAL' $L/final_skip_all.log | tr '\n' ' ')"

# ---- (3) partial: 3 of the 6 chain tokens deleted -> only those are rebuilt, and the rebuilt twins equal the independently built ones (mc_chainstyle)
$PY - <<'PYEOF'
import json, os
D = "/workspace/ops/d7fix"
rows = [json.loads(l) for l in open(D + "/bank_chain/targets_rank1.jsonl")]
for r in rows[:3]:
    p = f"{D}/mc_test/{r['log_name']}/{r['token']}__lane_rank_1.pkl.xz"
    os.remove(p)
print("removed", [r["token"][:8] for r in rows[:3]])
PYEOF
S=$(date +%s.%N)
env -u HF_TOKEN DRIVERL_EVAL_ROUTE_LANE_RANK=1 OMP_NUM_THREADS=1 REFE_PAUSE_FILE=$D/no_such_pause_file nice -n 10 $PY build_metric_cache.py build --targets $D/bank_chain --out $D/mc_test --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers 2 --lane-rank 1 > $L/final_skip_partial.log 2>&1
E=$(date +%s.%N)
echo "wall_s=$(python3 -c "print(round($E - $S, 1))")" >> $L/final_skip_partial.log
stamp "partial rebuild: $(grep -E 'ZZMETRIC|wall_s' $L/final_skip_partial.log | tr '\n' ' ')"
run $PY diag_d7_twin_route.py compare --bank $D/bank_chain --k 1 --a $D/mc_test --b $D/mc_chainstyle --out $R/final_partial_vs_chainstyle.json --n-sets 4 --min-keys 6 > $L/final_partial_compare.log 2>&1
stamp "partial compare rc=$? $(grep ZZD7C_ $L/final_partial_compare.log | tail -1)"

# ---- (4) the frozen tree rebuilds the three mini-banks FROM SCRATCH (real cmd_build, 2 workers) and passes (a)(b)
rm -rf $D/mc_final && mkdir -p $D/mc_final
$PY - <<'PYEOF'
import json, os, shutil
D = "/workspace/ops/d7fix"
for bank in ("bank_test", "bank_k3"):
    for ln in open(f"{D}/{bank}/targets_rank1.jsonl"):
        r = json.loads(ln)
        s = f"{D}/mc_test/{r['log_name']}/{r['token']}.pkl.xz"
        d = f"{D}/mc_final/{r['log_name']}/{r['token']}.pkl.xz"
        if os.path.isfile(s) and not os.path.isfile(d):
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copyfile(s, d)
print("logged parts copied:", sum(len(f) for _, _, f in os.walk(D + "/mc_final")))
PYEOF
for spec in "1 bank_test" "2 bank_test" "3 bank_k3"; do
  set -- $spec
  K=$1; B=$2
  env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK -u REFE_D7_MUTATE -u REFE_PAUSE_FILE OMP_NUM_THREADS=1 nice -n 15 $PY build_metric_cache.py build --targets $D/$B --out $D/mc_final --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers 2 --lane-rank $K > $L/final_build_k$K.log 2>&1
  stamp "final build k=$K rc=$? $(grep ZZMETRIC $L/final_build_k$K.log)"
  run $PY diag_d7_twin_route.py verify --bank $D/$B --k $K --twin-cache $D/mc_final --logged-cache $D/mc_final --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --out $R/verify_final_k$K.json > $L/final_verify_k$K.log 2>&1
  stamp "final verify k=$K rc=$? $(grep -o 'ZZD7V_[A-Z]* a=[A-Za-z]* b=[A-Za-z]* c=[A-Za-z]*' $L/final_verify_k$K.log)"
done
# the final-tree content must equal the pre-freeze build's (same code path => same parts)
for spec in "1 bank_test" "2 bank_test" "3 bank_k3"; do
  set -- $spec
  run $PY diag_d7_twin_route.py compare --bank $D/$2 --k $1 --a $D/mc_final --b $D/mc_test --out $R/final_vs_prefreeze_k$1.json --n-sets 2 --min-keys 5 > $L/final_vs_prefreeze_k$1.log 2>&1
  stamp "final-vs-prefreeze k=$1 $(grep ZZD7C_SUMMARY $L/final_vs_prefreeze_k$1.log | cut -c1-260)"
done
run timeout 900 $PY test_p6_cpu_probe.py > $L/final_test_p6_cpu_probe.log 2>&1; stamp "test_p6_cpu_probe rc=$? $(tail -1 $L/final_test_p6_cpu_probe.log)"
stamp "FINAL DONE"
