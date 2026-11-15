#!/usr/bin/env bash
# deploy_k.sh <NAME> <QUEUE> <BASEARM> <TREE> "<RUNS>" [SFX] [REF_SFX] : dev-box side, script polling. Card PREREG_V8F_PROOF_ADD40, note N4.
# Pushes the tool of note N4 (source cand-imagkin, second diagnostic) into its OWN pod folder /workspace/v8proof/imagplan_k (sha256-checked; a file that is already there with
# the same sha256 is not written again, a file with another sha256 stops the script: no run of that folder ever sees a changed file), starts pod_imagplan_seq4.sh for one
# checkpoint, waits for its end, pulls the logs of the runs it made (imag log, p4/p5 log, run text). Deletes nothing. Last line: ZZ DEPLOY-K <NAME> DONE or STOP.
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
H=/f/TanitAD-scratch/v8proof/imagplan; W=/workspace/v8proof; D=$W/imagplan_k; R=$W/runs/imagplan
NAME=${1:?name}; Q=${2:?queue}; BASE=${3:?base}; TREE=${4:?tree}; RUNS=${5:?runs}; SFX=${6:-x}; RSFX=${7-}
ts() { date -u +%H:%M:%SZ; }
cd "$H" || exit 1
timeout 60 bash "$P" "mkdir -p $D" 2>/dev/null
for f in imag_plan.py v8f_t1_eval_imagplan.py pod_imagplan_run.sh pod_imagplan_seq4.sh; do
  b=$(sha256sum "$H/$f" | cut -d' ' -f1)
  a=$(timeout 60 bash "$P" "[ -f $D/$f ] && sha256sum $D/$f | cut -d' ' -f1" 2>/dev/null | tr -cd 'a-f0-9')
  if [ -z "$a" ]; then
    timeout 180 bash "$P" "cat > $D/$f" < "$H/$f" 2>/dev/null
    a=$(timeout 60 bash "$P" "sha256sum $D/$f | cut -d' ' -f1" 2>/dev/null | tr -cd 'a-f0-9')
    [ "$a" = "$b" ] || { echo "ZZ DEPLOY-K $NAME STOP push mismatch $f"; exit 3; }
    echo "[$(ts)] pushed $f sha256 ${a:0:16}"
  elif [ "$a" = "$b" ]; then echo "[$(ts)] on the pod already: $f sha256 ${a:0:16}"
  else echo "ZZ DEPLOY-K $NAME STOP $f on the pod has another sha256 (${a:0:16} against ${b:0:16}): the folder is not changed"; exit 3; fi
done
timeout 60 bash "$P" "cd $D && bash -n pod_imagplan_seq4.sh && bash -n pod_imagplan_run.sh && [ ! -e $R/SEQ_${NAME}_DONE.txt ] && (IP_SFX='$SFX' IP_REF_SFX='$RSFX' nohup setsid bash pod_imagplan_seq4.sh $NAME $Q $BASE $TREE '$RUNS' > $R/seq_$NAME.nohup 2>&1 < /dev/null &); sleep 6; tail -n 2 $R/seq_$NAME.log | cut -c1-260" 2>/dev/null | sed '/hf_/d'
k=0
while :; do
  s=$(timeout 60 bash "$P" "ls $R/SEQ_${NAME}_DONE.txt 2>/dev/null | wc -l; grep -c 'ZZ SEQ STOP' $R/seq_$NAME.log 2>/dev/null" 2>/dev/null | tr -d '\r' | tr '\n' ' ')
  set -- $s
  [ "${1:-0}" = 1 ] && break
  [ "${2:-0}" -ge 1 ] 2>/dev/null && { timeout 60 bash "$P" "cut -c1-260 $R/seq_$NAME.log" 2>/dev/null | sed '/hf_/d'; echo "ZZ DEPLOY-K $NAME STOP"; exit 2; }
  k=$((k+1)); [ "$k" -gt 480 ] && { echo "ZZ DEPLOY-K $NAME STOP not ended after 8 h"; exit 4; }
  sleep 60
done
timeout 60 bash "$P" "cut -c1-230 $R/seq_$NAME.log" 2>/dev/null | sed '/hf_/d'
for t in $RUNS; do
  [ "$t" = IPo ] && new=IPo${RSFX}_$BASE || new=${t}${SFX}_$BASE
  mkdir -p "$H/pull/$new"
  for f in imag_log_$new.npz p45_$new.npz imagplan_run_$new.txt; do
    a=$(timeout 60 bash "$P" "[ -f $R/t1/$new/ticks/$f ] && sha256sum $R/t1/$new/ticks/$f | cut -d' ' -f1" 2>/dev/null | tr -cd 'a-f0-9')
    [ -n "$a" ] || continue
    [ -f "$H/pull/$new/$f" ] && [ "$(sha256sum "$H/pull/$new/$f" | cut -d' ' -f1)" = "$a" ] && continue
    bash "$C" "$R/t1/$new/ticks/$f" "$H/pull/$new/$f" < /dev/null > /dev/null 2>&1
    b=$(sha256sum "$H/pull/$new/$f" 2>/dev/null | cut -d' ' -f1)
    if [ "$a" = "$b" ]; then echo "[$(ts)] pulled $new/$f sha256 ${a:0:16}"; else echo "[$(ts)] PULL-MISMATCH $new/$f"; fi
  done
done
echo "ZZ DEPLOY-K $NAME DONE $(ts)"
