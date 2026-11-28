#!/bin/bash
# 74-twin cross-pod reproducibility control (token-free). Re-runs the banked 2026-10-17 control on THIS pod and compares bit for bit.
set -u
W=/workspace; C=$W/ops/bank3c/controls74; DBS=$W/data/navtrain_dbs
R="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
T="$R/raw/2026-10-17-twin-bank-3c"; TIP=f2f604f5d8067352287e113d90bc925f76c9641e
mkdir -p $C/banked $C/bank_in $C/bank_logged $C/rank0_ref $C/mc $C/logs $DBS
log() { echo "$(date -u +%FT%TZ) $*" >> $C/ctl.log; }
fail() { log "ZZCTL_FAIL $*"; echo "$*" > $C/FAILED; exit 1; }
until grep -q "^ZZOK teacher env ready" $W/data/refe_p6/teacher_env.log 2>/dev/null; do
  grep -q "^ZZFAIL" $W/data/refe_p6/teacher_env.log 2>/dev/null && fail "teacher env build failed"; sleep 30; done
log "teacher env ready"
source $W/ops/bank3c/code/env_3c.sh
export RANK0=$C/rank0_ref/targets_rank0.jsonl
G="git -C $W/repo"
$G archive $TIP:"$T/results/controls/pod" bank_i3 merged held | tar -x -C $C/banked || fail archive
cp $C/banked/bank_i3/targets_rank1.jsonl $C/bank_in/targets_rank1.jsonl
[ "$(sha256sum $C/bank_in/targets_rank1.jsonl | cut -c1-16)" = 94e0280d74ba40c4 ] || fail bank_sha
$PY - <<'PYEOF'
import json
C = '/workspace/ops/bank3c/controls74'
rows = [json.loads(l) for l in open(C + '/bank_in/targets_rank1.jsonl', encoding='utf-8') if l.strip()]
lane = [r for r in rows if isinstance(r.get('aug'), dict) and r['aug'].get('kind') == 'lane_rank']
toks = sorted({(r['log_name'], r['token']) for r in lane})
open(C + '/bank_logged/targets_rank0.jsonl', 'w').write(''.join(json.dumps({'log_name': a, 'token': b, 'rank': 0}) + '\n' for a, b in toks))
kept = [json.loads(l) for l in open(C + '/banked/merged/targets_rank1.jsonl', encoding='utf-8') if l.strip()]
kept = [r for r in kept if isinstance(r.get('aug'), dict) and r['aug'].get('reroll') == '3c']
open(C + '/rank0_ref/targets_rank0.jsonl', 'w').write(''.join(json.dumps({'log_name': r['log_name'], 'token': r['token'], 'rank': 0, 'traj': r['traj']}) + '\n' for r in kept))
open(C + '/mini_logs.txt', 'w').write('\n'.join(sorted({r['log_name'] for r in lane})) + '\n')
print('lane', len(lane), 'tokens', len(toks), 'kept_ref', len(kept), 'logs', len({r['log_name'] for r in lane}))
PYEOF
log "inputs built: $(wc -l < $C/mini_logs.txt) logs"
cd $PKG/code && env -u HF_TOKEN $PY fetch_navtrain_dbs.py --out $DBS --held-root /nonexistent --logs-file $C/mini_logs.txt --jobs 8 > $C/logs/dbs.log 2>&1
have=0; for l in $(cat $C/mini_logs.txt); do [ -f $DBS/$l.db ] && have=$((have+1)); done
[ "$have" = "$(wc -l < $C/mini_logs.txt)" ] || fail "dbs: have $have of $(wc -l < $C/mini_logs.txt) (logs/dbs.log)"
log "dbs ok $have"
cd $PKG/refe; export PYTHONPATH=$PKG/code:${PYTHONPATH:-}
env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK OMP_NUM_THREADS=1 REFE_PAUSE_FILE=/nonexistent nice -n 10 $PY build_metric_cache.py build --targets $C/bank_logged --out $C/mc --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers 6 > $C/logs/parts_logged.out 2>&1
grep -q "^ZZMETRIC_CACHE_DONE .* failed=0 " $C/logs/parts_logged.out || fail "logged parts (logs/parts_logged.out)"
for K in 1 2 3; do
  env -u HF_TOKEN DRIVERL_EVAL_ROUTE_LANE_RANK=$K OMP_NUM_THREADS=1 REFE_PAUSE_FILE=/nonexistent nice -n 10 $PY build_metric_cache.py build --targets $C/bank_in --out $C/mc --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers 6 --lane-rank $K > $C/logs/parts_k$K.out 2>&1
  grep -q "^ZZMETRIC_CACHE_DONE .* failed=0 " $C/logs/parts_k$K.out || fail "twin parts K=$K (logs/parts_k$K.out)"
done
log "parts ok: $(ls $C/mc | wc -l) top entries"
export C PKG PY CODE3C
seq 0 15 | xargs -P 6 -I{} bash -c 'i={}; s=$(printf "s%04d" $i); source /workspace/ops/bank3c/code/env_3c.sh; export RANK0=$C/rank0_ref/targets_rank0.jsonl; cd $PKG/refe; run3c $PY $CODE3C/reroll_bank_3c.py run --bank $C/bank_in --rank0 $RANK0 --out $C/new/shards/$s --mc $C/mc --log-shard $i/16 --code-dir $CODE3C --refe-dir $PKG/refe --pkg-code-dir $PKG/code > $C/logs/held_$s.log 2>&1'
nd=$(ls $C/new/shards/*/DONE 2>/dev/null | wc -l); log "reroll shards done $nd of 16"
python3 $W/ops/bank3c/ctl74_compare.py $C/new $C/banked $C/banked/merged/targets_rank1.jsonl --json $C/ctl74_result.json > $C/ctl74_verdict.txt 2>&1
log "verdict: $(cat $C/ctl74_verdict.txt | cut -c1-300)"
echo done > $C/RUN_DONE
