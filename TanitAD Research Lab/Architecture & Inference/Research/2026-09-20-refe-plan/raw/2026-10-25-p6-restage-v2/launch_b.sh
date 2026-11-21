#!/bin/bash
# P6 option B relaunch on the migrated pod (MM, 2026-10-25; PREREG_P6 Amendment (c)).  Run DETACHED, after the volume migration and the DB fetch.
# verify migrated inputs -> move the discarded run aside (kept) -> place the step-2,603 rollback run -> link the kept 3-C twins (POST_MERGE (d))
# -> the chain's OWN D7 check on all ranks -> P6_D7_FALLBACK=0 -> invalidate lanes (the chain re-runs D7 + lab/mem/egr/launch_gate: a NEW token)
# -> relaunch the rebuild driver (all its steps .done -> straight to the final chain) -> coverage guard on the first labeller events.
# Any check failing STOPS before the chain starts (nothing on the volume is deleted; the discarded run stays as *.deviant_4332_discarded).
set -u
W=/workspace; O=/root/tad/launch_b; NB=$W/data/refe_navtrain; MC=$NB/metric_cache; SRC=$NB/metric_cache_twins_3c; CB=$NB/train_grow_cmd_3c
P6=$W/data/refe_p6; RUNS=$W/data/refe_runs; RUN=$RUNS/p6_paper_faithful; PKG=$W/refe-plan-p6; PY=$W/venv-teacher/bin/python; ENVF=$W/ops/p6_start.env
RB=/root/tad/rb2603/p6_paper_faithful; CKSHA=eda651447f31846a
mkdir -p $O
log() { echo "$(date -u +%FT%TZ) $*" >> $O/launch_b.log; }
fail() { log "ZZLB_FAIL $*"; echo "$*" > $O/FAILED; exit 1; }
rm -f $O/FAILED $O/DONE
log "start pid $$"

# 0. preconditions: no chain / trainer / GPU process, DBs complete, migration complete
ps -eo args= | grep -qE "[p]6_pod.sh|[t]rain.py|[r]ebuild/driver.sh|[f]etch_navtrain_dbs" && fail "a chain, trainer, driver or DB fetch is running"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || fail "GPU busy (speed experiments must be stopped first)"
[ "$(ls /root/tad/navtrain_dbs/*.db 2> /dev/null | wc -l)" -ge 1192 ] || fail "DBs: $(ls /root/tad/navtrain_dbs/*.db | wc -l)/1192"
[ "$(readlink $W/data/navtrain_dbs)" = /root/tad/navtrain_dbs ] || fail "navtrain_dbs symlink"
[ -z "$(find $W/ops $W/refe-plan-p6 $W/venv-teacher $NB $P6 $RUNS -maxdepth 3 -user nobody 2> /dev/null | head -1)" ] || fail "nobody-owned placeholders remain"
$PY -c "import torch, numpy, yaml, omegaconf" 2> $O/import.err || fail "venv-teacher import: $(tail -n 1 $O/import.err)"
[ "$(sha256sum < $CB/targets_rank0.jsonl | cut -c1-16)" = daa23b962e951367 ] || fail "3-C rank0 sha"
[ "$(sha256sum < $CB/targets_rank1.jsonl | cut -c1-16)" = 2a63acc2f6340481 ] || fail "3-C rank1 sha"
for k in 1 2 3; do echo "k$k $(find $SRC -name "*__lane_rank_$k.pkl.xz" | wc -l)"; done > $O/twins_src.txt
[ "$(tr '\n' ' ' < $O/twins_src.txt)" = "k1 32929 k2 8090 k3 3390 " ] || fail "twins_3c counts: $(tr '\n' ' ' < $O/twins_src.txt)"
[ "$(find $MC -name '*__lane_rank_*.pkl.xz' | wc -l)" = 0 ] || fail "metric_cache already holds lane files"
log "ZZLB_INPUTS_OK dbs 1192, 3-C shas, twins_3c 32929/8090/3390, metric_cache plain parts $(find $MC -name '*.pkl.xz' | wc -l)"

# 1. the discarded run aside (a rename on one volume; nothing deleted), the rollback run in place
[ -f $RUN/ckpt_last.pt ] || fail "no migrated run dir at $RUN"
[ -e $RUN.deviant_4332_discarded ] && fail "$RUN.deviant_4332_discarded already exists"
mv $RUN $RUN.deviant_4332_discarded || fail "mv run aside"
mkdir -p $RUN && cp -p $RB/* $RUN/ || fail "copy rollback files"
[ "$(sha256sum < $RUN/ckpt_last.pt | cut -c1-16)" = $CKSHA ] || fail "placed ckpt sha"
python3 -I /root/tad/restage_helpers.py verify --run $RUN --expect-step 2603 > $O/verify.out 2>&1 || fail "verify: $(tail -n 1 $O/verify.out)"
log "ZZLB_RUN_PLACED step 2603 ckpt $CKSHA; discarded run kept at $RUN.deviant_4332_discarded"

# 2. POST_MERGE (d): link the kept twins, plan then apply
cd /root/tad
$PY -I pod_helpers.py parts --bank $CB --src $SRC --dst $MC --p6refe $PKG/refe --mode plan > $O/parts_plan.out 2>&1
grep -q "state=plan_ok" $O/parts_plan.out || fail "parts plan: $(grep -a ZZV $O/parts_plan.out | cut -c1-300)"
$PY -I pod_helpers.py parts --bank $CB --src $SRC --dst $MC --p6refe $PKG/refe --mode apply --how link --report $O/parts_apply.json > $O/parts_apply.out 2>&1
L=$(grep -a "^ZZV parts" $O/parts_apply.out)
for kv in state=ok linked=21821 errors=0 dst_missing_after=0; do echo " $L " | grep -q " $kv " || fail "parts apply ($kv): $(echo $L | cut -c1-300)"; done
log "ZZLB_LINKED $(echo $L | cut -c1-260)"

# 3. the chain's OWN D7 command (deterministic: first 40 twins in bank order), every rank OK
cd $PKG/refe
for k in 1 2 3; do
  env -u HF_TOKEN $PY p6_prep.py lane-distinct --bank $CB --cache $MC --k $k --n 40 --min-frac 0.9 > $O/distinct_$k.out 2>&1
  grep -q "^ZZD7_DISTINCT_OK " $O/distinct_$k.out || fail "D7 rank $k: $(tail -n 1 $O/distinct_$k.out | cut -c1-200)"
  log "ZZLB_D7 $(tail -n 1 $O/distinct_$k.out | cut -c1-160)"
done

# 4. fail-closed from now on; invalidate lanes (the chain's own semantics: it re-runs D7 and voids lab/mem/egr/launch_gate)
cp -p $ENVF $ENVF.pre_launch_b
sed -i 's/^P6_D7_FALLBACK=1$/P6_D7_FALLBACK=0/' $ENVF
grep -q "^P6_D7_FALLBACK=0$" $ENVF || fail "could not set P6_D7_FALLBACK=0"
rm -f $P6/stage/lanes.ok
log "ZZLB_INVALIDATED lanes; P6_D7_FALLBACK=0 (backup $ENVF.pre_launch_b)"
echo "$(date -u +%FT%TZ) MM launch_b: PREREG_P6 Amendment (c) -- rollback to step 2,603 (HF fc2adb00b59e), 21,821 kept 3-C twins linked, D7 OK, lanes invalidated -> D7 + lab/mem/egr/launch_gate re-run, then resume" >> $P6/p6_stdout.log

# 5. relaunch, then the coverage guard (never kills training)
cd / && setsid -f nohup bash $W/ops/rebuild/driver.sh > $O/driver.out 2>&1 < /dev/null
log "ZZLB_RELAUNCHED driver"
OFF=$(stat -c %s $RUN/metrics.jsonl)
python3 -I /root/tad/restage_helpers.py guard --metrics $RUN/metrics.jsonl --offset $OFF --wait-s 14400 --lines 5 --alert $O/ALERT_COVERAGE.txt > $O/guard.out 2>&1
log "ZZLB_GUARD rc $? $(tail -n 1 $O/guard.out | cut -c1-200)"
date -u +%FT%T > $O/DONE
