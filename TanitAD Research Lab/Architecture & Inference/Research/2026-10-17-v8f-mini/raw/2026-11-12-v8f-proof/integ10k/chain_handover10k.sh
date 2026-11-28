#!/usr/bin/env bash
# chain_i10k.sh -- DEV-BOX chain of card PREREG_V8F_PROOF_ADD46 (INTEG10K-PRE). Started by the lead; nothing here starts by itself.
#   1 push the code files into the pod folder $W/build (sha256 compare; a file that is there with another sha256 stops the chain; the pod folder is created, nothing else on the pod is touched)
#   2 start the pod queue (nohup setsid; refuses if a START marker exists)
#   3 wait by a polling loop that EXITS on HANDOVER10K_DONE, HANDOVER10K_STOP or after 12 h (a poll is a script step, not a model turn)
#   4 pull: per rollout folder and for the small files, one tar on the pod, copied with pcat.sh, sha256 compared, extracted into its own folder under $PULL, the local tar then cleared (path logged)
# Last line: "ZZ CHAIN-HANDOVER10K DONE" or "ZZ CHAIN-HANDOVER10K STOP <reason>".   Usage: bash chain_i10k.sh [--no-start]  (--no-start only pushes and pulls what exists)
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
H=/f/TanitAD-scratch/v8proof; I=$H/integ10k
W=/workspace/v8proof/integ10k; B=$W/build
PULL=$I/pull; TARS=$I/work/tars; LOGF=$I/chain_handover10k.log
NOSTART=0; [ "${1:-}" = "--no-start" ] && NOSTART=1
ts() { date -u +%H:%M:%SZ; }
say() { echo "[$(ts)] $*" | tee -a "$LOGF"; }
fin() { echo "ZZ CHAIN-HANDOVER10K STOP $*" | tee -a "$LOGF"; exit 3; }
pod() { timeout "${PODT:-60}" bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
mkdir -p "$PULL" "$TARS"

FILES=("$I/v8f_t1_eval_reader_rule.py" "$I/tok_reader.py" "$I/tok_probe_lr.py" "$I/qmap_fit.py" "$I/qcal_map.py" "$I/reader_controls.py" "$I/pod_handover10k_queue2.sh" "$I/v8f_t1_eval_reader_rule_ro.py" "$H/distrule/cmp_dumps.py"
       "$I/taniteval_cdbff971.tar" "$H/leversheet/LS_REF40_WINDOWS.json" "$H/distrule/dist_rule.py" "$H/distrule/v8f_t1_eval_rule.py" "$H/distrule/v8f_t1_eval_rule_mix.py")
pod "mkdir -p $B" > /dev/null
for f in "${FILES[@]}"; do
  n=$(basename "$f")
  [ -f "$f" ] || fin "missing local file $n"
  b=$(sha256sum "$f" | cut -d' ' -f1)
  a=$(pod "[ -f $B/$n ] && sha256sum $B/$n | cut -d' ' -f1" | tr -cd 'a-f0-9')
  if [ -z "$a" ]; then
    PODT=180 pod "cat > $B/$n" < "$f" > /dev/null
    a=$(pod "sha256sum $B/$n | cut -d' ' -f1" | tr -cd 'a-f0-9')
    [ "$a" = "$b" ] || fin "push mismatch $n"
    say "pushed $n sha256 ${a:0:16}"
  elif [ "$a" = "$b" ]; then say "on the pod already: $n sha256 ${a:0:16}"
  else fin "$n on the pod has another sha256 (${a:0:16} against ${b:0:16}); the folder is not changed"; fi
done
[ "$(pod "bash -n $B/pod_handover10k_queue2.sh && echo ok" | tr -cd 'a-z')" = ok ] || fin "bash -n of the queue failed on the pod"
# overlay tree for the pinned tool: stack = link to the staged stack, taniteval from the tar (extracted once)
if [ "$(pod "[ -d $B/tree_ov/taniteval/taniteval ] && [ -e $B/tree_ov/stack/scripts/v8f_t1_eval.py ] && echo yes" | tr -cd 'a-z')" != yes ]; then
  pod "mkdir -p $B/tree_ov && cd $B/tree_ov && tar xf $B/taniteval_cdbff971.tar && { [ -e $B/tree_ov/stack ] || ln -s $W/tree/stack $B/tree_ov/stack; }" > /dev/null
  [ "$(pod "[ -d $B/tree_ov/taniteval/taniteval ] && [ -e $B/tree_ov/stack/scripts/v8f_t1_eval.py ] && echo yes" | tr -cd 'a-z')" = yes ] || fin "overlay tree not built"
  say "overlay tree built"
fi

if [ $NOSTART = 0 ]; then
  s=$(pod "ls $W/START_HANDOVER10K.txt 2>/dev/null | wc -l" | tr -cd '0-9')
  [ "$s" = 0 ] || fin "START_HANDOVER10K.txt exists on the pod: a queue was started"
  pod "cd $W && nohup setsid bash $B/pod_handover10k_queue2.sh > $W/queue_handover_stdout.txt 2>&1 < /dev/null & sleep 2; ls $W/START_HANDOVER10K.txt 2>&1 | tail -n 1" > /dev/null
  say "queue started"
fi

k=0; st=DONE
while :; do
  s=$(pod "ls $W/HANDOVER10K_DONE.txt $W/HANDOVER10K_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '")
  case "$s" in *STOP*) st=STOP; break;; *DONE*) break;; esac
  [ $NOSTART = 1 ] && break
  k=$((k+1)); [ "$k" -gt 720 ] && { st=STOP; say "not ended after 12 h"; break; }
  sleep 60
done
[ "$st" = STOP ] && pod "cat $W/HANDOVER10K_STOP.txt | cut -c1-300" | tee -a "$LOGF"

pull_dir() { # name remote-folder local-folder
  local name=$1 rdir=$2 ldir=$3 a b
  [ -f "$ldir/PULLED.txt" ] && return 0
  pod "tar cf $W/pull_$name.tar -C $rdir . && sha256sum $W/pull_$name.tar | cut -d' ' -f1" > "$TARS/$name.sha" 2>/dev/null
  a=$(tr -cd 'a-f0-9' < "$TARS/$name.sha")
  [ ${#a} = 64 ] || { say "no tar of $name"; return 1; }
  bash "$C" "$W/pull_$name.tar" "$TARS/$name.tar" < /dev/null > /dev/null 2>&1
  b=$(sha256sum "$TARS/$name.tar" 2>/dev/null | cut -d' ' -f1)
  [ "$a" = "$b" ] || { say "PULL-MISMATCH $name"; return 1; }
  mkdir -p "$ldir" && tar xf "$TARS/$name.tar" -C "$ldir" && echo "$a" > "$ldir/PULLED.txt"
  say "pulled $name sha256 ${a:0:16}"
  local T="$TARS/$name.tar" S="$TARS/$name.sha"
  sz=$(stat -c %s "${T:?}" 2>/dev/null); rm -f "${T:?}" "${S:?}"; say "cleared $T (${sz:-?} bytes) and $S"
}
for m in 0 1; do
  for nm in I10R0HID I10R0C5 I10R0B5 I10R0C10; do
    d=$(pod "[ -f $W/t1/${nm}_s$m/T1_DONE ] && echo yes" | tr -cd 'a-z'); [ "$d" = yes ] && pull_dir "${nm}_s$m" "$W/t1/${nm}_s$m" "$PULL/${nm}_s$m"
  done
  d=$(pod "[ -f $W/t1/I10RRx_s$m/T1_DONE ] && echo yes" | tr -cd 'a-z'); [ "$d" = yes ] && pull_dir "I10RRx_s$m" "$W/t1/I10RRx_s$m" "$PULL/I10RRx_s$m"
done
# small files: reader logs, controls, maps, rows records, queue log (the rows npz stay on the pod)
pod "mkdir -p $W/small_handover && cp -f $W/qmap_sweep/* $W/qmap_sweep.log $W/rows/*.json $W/lr16_checklog.txt $W/lr16/TRAIN_DONE.json $W/lr16/train_meta.json $W/queue_i10k.log $W/k8_n2.log $W/diag/out/DIAG_TAIL.json $W/diag/out/DIAG_TAIL2.json $W/diag/out/DIAG_TAIL3.json $W/small_handover/ 2>/dev/null; for k in 0 1; do cp -f $W/lr16/TK16_s\$k/train_log.jsonl $W/small_handover/train_log_s\$k.jsonl 2>/dev/null; sha256sum $W/lr16/TK16_s\$k/ckpt_last.pt | cut -c1-64 > $W/small_handover/ckpt_sha_s\$k.txt; done; ls $W/small_handover | wc -l" > /dev/null
pull_dir small_handover "$W/small_handover" "$PULL/small_handover"
if [ "$st" = DONE ]; then echo "ZZ CHAIN-HANDOVER10K DONE $(ts)" | tee -a "$LOGF"; else echo "ZZ CHAIN-HANDOVER10K STOP queue-stop-or-timeout $(ts)" | tee -a "$LOGF"; fi
