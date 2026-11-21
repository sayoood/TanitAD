#!/usr/bin/env bash
# deploy_q1.sh (note PREREG_V8F_PROOF_ADD40_N7, form of deploy_k6.sh) <NAME> <QUEUE> <BASEARM> <TREE> "<RUNS>" [SFX] [REF_SFX] : dev-box side, script polling. Card PREREG_V8F_PROOF_ADD40, note N4.
# Sequence with the memory guard (pod_imagplan_seq_q.sh): pushes the tool files into the pod folder /workspace/v8proof/imagplan_k (sha256-checked;
# a file that is there with the same sha256 is not written again, one with another sha256 stops the script), starts the sequence for one checkpoint, waits for its end
# (done or stopped), and pulls the logs of every run of the list that has its T1_DONE marker (also after a stop). Deletes nothing.
# Last line: ZZ DEPLOY-Q <NAME> DONE or ZZ DEPLOY-Q <NAME> STOP.
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
H=/f/TanitAD-scratch/v8proof/imagplan_q; W=/workspace/v8proof; D=$W/imagplan_q; R=$W/runs/imagplan
NAME=${1:?name}; Q=${2:?queue}; BASE=${3:?base}; TREE=${4:?tree}; RUNS=${5:?runs}; SFX=${6:-x}; RSFX=${7-}
ts() { date -u +%H:%M:%SZ; }
cd "$H" || exit 1
timeout 60 bash "$P" "mkdir -p $D" 2>/dev/null
for f in imag_plan.py v8f_t1_eval_imagplan_q.py qhead_model.py pod_imagplan_run_q.sh pod_imagplan_seq_q.sh; do
  b=$(sha256sum "$H/$f" | cut -d' ' -f1)
  a=$(timeout 60 bash "$P" "[ -f $D/$f ] && sha256sum $D/$f | cut -d' ' -f1" 2>/dev/null | tr -cd 'a-f0-9')
  if [ -z "$a" ]; then
    timeout 180 bash "$P" "cat > $D/$f" < "$H/$f" 2>/dev/null
    a=$(timeout 60 bash "$P" "sha256sum $D/$f | cut -d' ' -f1" 2>/dev/null | tr -cd 'a-f0-9')
    [ "$a" = "$b" ] || { echo "ZZ DEPLOY-Q $NAME STOP push mismatch $f"; exit 3; }
    echo "[$(ts)] pushed $f sha256 ${a:0:16}"
  elif [ "$a" = "$b" ]; then echo "[$(ts)] on the pod already: $f sha256 ${a:0:16}"
  else echo "ZZ DEPLOY-Q $NAME STOP $f on the pod has another sha256 (${a:0:16} against ${b:0:16}): the folder is not changed"; exit 3; fi
done
timeout 60 bash "$P" "cd $D && bash -n pod_imagplan_seq_q.sh && bash -n pod_imagplan_run_q.sh && [ ! -e $R/SEQ_${NAME}_DONE.txt ] && [ ! -e $R/SEQ_${NAME}_STOPPED.txt ] && (IP_SFX='$SFX' IP_REF_SFX='$RSFX' IP_NEED_GB='${IP_NEED_GB:-24}' IP_NEED_NOISE_GB='${IP_NEED_NOISE_GB:-40}' IP_MAX_WINDOWS='${IP_MAX_WINDOWS:-}' nohup setsid bash pod_imagplan_seq_q.sh $NAME $Q $BASE $TREE '$RUNS' > $R/seq_$NAME.nohup 2>&1 < /dev/null &); sleep 6; tail -n 2 $R/seq_$NAME.log | cut -c1-300" 2>/dev/null | sed '/hf_/d'
k=0; st=DONE
while :; do
  s=$(timeout 60 bash "$P" "ls $R/SEQ_${NAME}_DONE.txt $R/SEQ_${NAME}_STOPPED.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '" 2>/dev/null | tr -d '\r')
  case "$s" in *STOPPED*) st=STOP; break;; *DONE*) break;; esac
  k=$((k+1)); [ "$k" -gt 600 ] && { st=STOP; echo "[$(ts)] not ended after 10 h"; break; }
  sleep 60
done
timeout 60 bash "$P" "cut -c1-260 $R/seq_$NAME.log" 2>/dev/null | sed '/hf_/d'
for t in $RUNS; do
  [ "$t" = IPo ] && new=IPo${RSFX}_$BASE || new=${t}${SFX}_$BASE
  d=$(timeout 60 bash "$P" "[ -f $R/t1/$new/T1_DONE ] && echo yes" 2>/dev/null | tr -cd 'a-z')
  [ "$d" = yes ] || { echo "[$(ts)] $new has no T1_DONE: not pulled"; continue; }
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
echo "ZZ DEPLOY-Q $NAME $st $(ts)"
