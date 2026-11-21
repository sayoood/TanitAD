#!/bin/bash
# r9_recqueue24.sh <L SHA40> <K2DC02s1> -- MM 4a3d05, 2026-11-20 (Thor kit; derived from r9_recqueue23.sh md5 a29b59a6). K2DC02s1's launch-gate RECORDS: the section 4.4 REPLICATE of K2DC02 (the same flags, --seed 1)
# (PREREG_K2DC02_2026-11-18.md section 4). RECORDS ARE GATE PREP: the K2DC02s1 TRAINING still needs the PI's go. G-MAP-OVERFIT / G-BOX-OVERFIT bind the WHOLE launch-argv sha256,
# so the arm binds its own map-banded + box records, then K0's explore-gate records at the SAME commit via kit/r9_recqueue19.sh (SKIPPED when queue 19 already ended DONE at
# this commit: the arm's sibling queue made them, and q19 refuses a second run there).
# WAITS for the K2DC02-TAIL read: before its first GPU step the queue holds until stageB/ROLL_STATUS.txt carries AWAIT-S6C AND no roll / trainer / gate process AND no compute app
# (two polls 60 s apart), then the usual thor_gpu.lock flock. A stopped / aborted read ends the queue ZZRECQ24-FAIL-READSTOP.
# Events: $R1/logs/recq24_<c12>.status (+ recq19_<c12>.status). Ends ZZRECQ24-DONEZZ or ZZRECQ24-FAIL-<step>ZZ. R9Q24_DRY=1: all dry-runs, no GPU, no wait.
X=${1:?usage: r9_recqueue24.sh <L SHA40> <K2DC02s1>}; K2=${2:?K2 arm}; C12=${X:0:12}
case "$K2" in K2DC02s1) ;; *) echo "K2 arm must be K2DC02s1 (this kit binds only that arm)"; exit 2;; esac
R1=/home/nvidia/r8ns/r9e1; LT=$R1/rectree_$C12/launch_tree; BIND=$LT/stack/ops/r9e1_bind.sh
LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock; ST=$R1/logs/recq24_$C12.status; L=$R1/logs
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $ST; }
fail() { ev "$1"; echo "ZZRECQ24-FAIL-$2ZZ" >> $ST; exit "${3:-4}"; }
cd $R1 || exit 2
[ "$(cat $R1/rectree_$C12/launch_tree.commit 2>/dev/null)" = "$X" ] || fail "REFUSED $LT is not at $X" TREE 2
[ -f $LT/stack/ops/runs.d/r9e2_$K2.argv.json ] || fail "REFUSED no r9e2_$K2.argv.json in $LT" NOK2 2
grep -q '"25.0"' $LT/stack/ops/runs.d/r9e2_$K2.argv.json || fail "REFUSED r9e2_$K2.argv.json carries no --w-map-hires 25.0 (the initial weight)" NOREDOSE 2
grep -q '"--map-hires-share-target"' $LT/stack/ops/runs.d/r9e2_$K2.argv.json || fail "REFUSED r9e2_$K2.argv.json carries no --map-hires-share-target (not a controlled arm)" NODC 2
grep -q '"0.02"' $LT/stack/ops/runs.d/r9e2_$K2.argv.json && grep -q '"0.1,25"' $LT/stack/ops/runs.d/r9e2_$K2.argv.json || fail "REFUSED r9e2_$K2.argv.json lacks target 0.02 or bounds 0.1,25 (not a K2DC02 descendant)" NOK2DC02 2
/usr/bin/python3 -c "import json,sys; a=json.load(open(sys.argv[1]))['argv']; sys.exit(0 if a.count('--seed') == 1 and a[a.index('--seed') + 1] == '1' and a.count('--r9-hist-supervise-geom') == 1 else 1)" $LT/stack/ops/runs.d/r9e2_$K2.argv.json 2>/dev/null || fail "REFUSED r9e2_$K2.argv.json is not the K2DC02s1 argv (--seed 1 exactly once and --r9-hist-supervise-geom exactly once)" NOARM 2
[ -s $R1/kit/r9_recqueue19.sh ] || fail "REFUSED no kit/r9_recqueue19.sh (K0's explore-gate records)" NOQ19 2
ev "QUEUE24-START $C12 code $LT k2 $K2 "
dry() {   # $1 tag, rest = bind args; any ZZR9E1BIND-DRY*OKZZ marker passes
  local tag=$1; shift
  bash $BIND --dry-run "$@" > $L/recq24_${C12}_${tag}.dry.out 2>&1 < /dev/null
  grep -qE 'ZZR9E1BIND-DRY[A-Za-z-]*OKZZ' $L/recq24_${C12}_${tag}.dry.out && { ev "$tag dry-run OK"; return 0; }
  fail "$tag dry-run FAILED: $(tail -2 $L/recq24_${C12}_${tag}.dry.out | tr '\n' ' ' | cut -c1-220)" "${tag}-DRY" 3
}
gpu() {   # $1 tag, rest = bind args; the step's own ZZR9E1BIND-<kind>-<arm>-OKZZ marker passes
  local tag=$1 i w M; shift
  for i in 1 2 3 4 5 6; do
    for w in $(seq 1 1440); do flock -n $LOCKF true && break; sleep 60; done
    ev "$tag try $i: lock free, calling bind"
    bash $BIND "$@" > $L/recq24_${C12}_${tag}.$i.out 2>&1 < /dev/null
    M=$(grep -oE 'ZZR9E1BIND-[A-Za-z0-9-]*ZZ' $L/recq24_${C12}_${tag}.$i.out | tail -1)
    case "$M" in *-OKZZ) ev "$tag OK ($M)"; return 0;; esac
    grep -q "stayed held" $L/recq24_${C12}_${tag}.$i.out && { ev "$tag try $i lost the lock race; retry"; continue; }
    fail "$tag FAILED (${M:-no marker}): $(tail -3 $L/recq24_${C12}_${tag}.$i.out | tr '\n' ' ' | cut -c1-240)" "$tag"
  done
  fail "$tag: 6 lost races" "${tag}-RACES" 5
}
TAIL=/home/nvidia/r8ns/r9e2/runs/E2TAIL_K2DC02
wait_read() {   # MM 2026-11-20: the K2DC02-TAIL read owns the Thor GPU (it takes thor_gpu.lock per stage). Records start only after it RELEASED it.
  local i RS=$TAIL/stageB/ROLL_STATUS.txt ok=0 NB NG
  ev "WAIT-READ: waiting for AWAIT-S6C in $RS, then no roll / trainer / gate process and no compute app on two polls 60 s apart (cap 24 h)"
  for i in $(seq 1 1440); do
    # MM 11-20: only lines AFTER the last 'ALLSTEP RESUME' count -- a STOP that a later RESUME answered is history, not a stopped read
    { awk '/ALLSTEP RESUME/{b=""} {b=b $0 "\n"} END{printf "%s", b}' $TAIL/CHAIN_ALL_STATUS.txt 2>/dev/null | grep -q 'ALLSTEP STOP' || [ -e $TAIL/stageB/ROLL_ABORTED ]; } && fail "WAIT-READ: the K2DC02-TAIL read stopped or aborted ($TAIL): records wait for an MM decision" READSTOP 6
    if grep -q 'AWAIT-S6C' $RS 2>/dev/null; then
      NB=$(ps -eo args | grep -c -E '[r]9e2_roll[.]sh|[r]efc_v3_train[.]py|[r]9gate3_chain[.]sh|[r]9gate3_thor_gate[.]sh')
      NG=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
      if [ "$NB" = 0 ] && [ "$NG" = 0 ]; then ok=$((ok + 1)); [ $ok -ge 2 ] && { ev "WAIT-READ: the read released the GPU (AWAIT-S6C, no process, no compute app)"; return 0; }; else ok=0; fi
    fi
    sleep 60
  done
  fail "WAIT-READ: 24 h without AWAIT-S6C and an idle GPU" READWAIT 6
}
RE2="--argv-prefix r9e2"
dry mapb_$K2 map-banded $K2 $RE2 --commit $X --code $LT
dry box_$K2 box $K2 $RE2 --commit $X --code $LT
ev "dry-runs OK"
if [ "${R9Q24_DRY:-}" = 1 ]; then
  if [ "$(tail -n 1 $R1/logs/recq19_$C12.status 2>/dev/null)" = ZZRECQ19-DONEZZ ]; then ev "K0 records (queue 19) already DONE at $C12: dry-run skipped"; else
  R9Q19_DRY=1 bash $R1/kit/r9_recqueue19.sh $X > $L/recq24_${C12}_q19dry.out 2>&1 < /dev/null
  grep -q ZZRECQ19-DRYONLYZZ $R1/logs/recq19_$C12.status || fail "K0 (queue 19) dry-run FAILED: $(tail -2 $R1/logs/recq19_$C12.status | tr '\n' ' ' | cut -c1-200)" Q19-DRY 3
  fi
  ev "DRY ONLY (R9Q24_DRY=1): K2DC02s1 + K0 dry-runs OK, no GPU step"; echo "ZZRECQ24-DRYONLYZZ" >> $ST; exit 0
fi
wait_read
gpu mapb_$K2 map-banded $K2 $RE2 --commit $X --code $LT
gpu box_$K2 box $K2 $RE2 --commit $X --code $LT
ev "$K2-RECORDS-READY at $C12 (map-banded + box own, K2DC02s1 argv)"
if [ "$(tail -n 1 $R1/logs/recq19_$C12.status 2>/dev/null)" = ZZRECQ19-DONEZZ ]; then ev "K0 records (queue 19) already DONE at $C12: not re-run"; else
bash $R1/kit/r9_recqueue19.sh $X > $L/recq24_${C12}_q19.out 2>&1 < /dev/null
grep -q ZZRECQ19-DONEZZ $R1/logs/recq19_$C12.status || fail "K0 records (queue 19) FAILED: $(tail -2 $R1/logs/recq19_$C12.status | tr '\n' ' ' | cut -c1-200)" Q19 4
fi
ev "K0-RECORDS-READY at $C12 (queue 19)"
ev "QUEUE24-DONE"; echo "ZZRECQ24-DONEZZ" >> $ST
