#!/usr/bin/env bash
# replay_n4b.sh : dev-box side, script polling. Card PREREG_V8F_PROOF_ADD40, note N3 section 3: where does the tool's imagined response differ from the reader's?
# The wrapper of this folder now also replays the reader's own stimulus through the tool's state path (the logged actions with a constant acceleration offset of
# -2, -1, +1 m/s2 on all ten steps, key imag_gap_logged_off of the p4/p5 log). It is pushed into its OWN pod folder (/workspace/v8proof/imagplan_n4b, with copies
# of the unchanged files), so that no run of the folder imagplan sees a changed file. Two runs: IPh_LFT_s0 and IPh_LA_s0 (cand-imag, family decel, p4/p5 log).
# Each waits out a refusal (3 GPU jobs or under 20 GB) up to 60 min. Then the logs are pulled. Deletes nothing. Last line: ZZ REPLAY-N4B DONE or STOP.
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
H=/f/TanitAD-scratch/v8proof/imagplan; W=/workspace/v8proof; D=$W/imagplan_n4b; R=$W/runs/imagplan
ts() { date -u +%H:%M:%SZ; }
cd "$H" || exit 1
timeout 60 bash "$P" "mkdir -p $D" 2>/dev/null
for f in imag_plan.py v8f_t1_eval_imagplan.py pod_imagplan_run.sh; do
  timeout 180 bash "$P" "cat > $D/$f" < "$H/$f" 2>/dev/null
  a=$(timeout 60 bash "$P" "sha256sum $D/$f | cut -d' ' -f1" 2>/dev/null | tr -cd 'a-f0-9'); b=$(sha256sum "$H/$f" | cut -d' ' -f1)
  [ -n "$a" ] && [ "$a" = "$b" ] || { echo "ZZ REPLAY-N4B STOP push mismatch $f"; exit 3; }
  echo "[$(ts)] pushed $f sha256 ${a:0:16}"
done
timeout 60 bash "$P" "cat > $D/run_replay_b.sh" <<'EOS' 2>/dev/null
#!/usr/bin/env bash
# run_replay_b.sh : ON THE POD. Two replay runs, serial; a refusal for slot or memory is waited out (60 x 60 s).
W=/workspace/v8proof; D=$W/imagplan_n4b; R=$W/runs/imagplan; LOG=$R/seq_replay_n4b.log
export IP_FAMILY=decel IP_P45=1 IP_LIBDIRS=$W/distrule:$W/probe
one() { local new=$1 q=$2 base=$3 tree=$4 n=0 rc
  while :; do
    bash $D/pod_imagplan_run.sh $new $q $base cand-imag $tree > $R/run_$new.out 2>&1; rc=$?
    [ $rc = 0 ] && { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $new done" >> $LOG; return 0; }
    if [ $rc = 2 ] && [ ! -e $R/t1/$new ] && grep -q "free container memory\|GPU jobs already run" $R/run_$new.out; then n=$((n+1)); [ $n -gt 60 ] && break; sleep 60; continue; fi
    break
  done
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ZZ SEQ STOP $new rc=$rc: $(tail -n 1 $R/run_$new.out | cut -c1-200)" >> $LOG; return 1; }
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] replay start wrapper $(sha256sum $D/v8f_t1_eval_imagplan.py | cut -c1-16)" >> $LOG
one IPh_LFT_s0 leadcond_a LFT_s0 $W/tree_leadcond && one IPh_LA_s0 actcons_a LA_s0 $W/tree_actcons && { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ZZ SEQ replay_n4b DONE" >> $LOG; touch $R/SEQ_replay_n4b_DONE.txt; }
EOS
timeout 60 bash "$P" "cd $D && bash -n run_replay_b.sh && (nohup setsid bash run_replay_b.sh > $R/seq_replay_n4b.nohup 2>&1 < /dev/null &); sleep 5; cat $R/seq_replay_n4b.log | cut -c1-160" 2>/dev/null | sed '/hf_/d'
k=0
while :; do
  s=$(timeout 60 bash "$P" "ls $R/SEQ_replay_n4b_DONE.txt 2>/dev/null | wc -l; grep -c 'ZZ SEQ STOP' $R/seq_replay_n4b.log 2>/dev/null" 2>/dev/null | tr -d '\r' | tr '\n' ' ')
  set -- $s
  [ "${1:-0}" = 1 ] && break
  [ "${2:-0}" -ge 1 ] 2>/dev/null && { timeout 60 bash "$P" "cut -c1-260 $R/seq_replay_n4b.log" 2>/dev/null | sed '/hf_/d'; echo "ZZ REPLAY-N4B STOP"; exit 2; }
  k=$((k+1)); [ "$k" -gt 100 ] && { echo "ZZ REPLAY-N4B STOP not ended after 100 min"; exit 4; }
  sleep 60
done
timeout 60 bash "$P" "cut -c1-200 $R/seq_replay_n4b.log" 2>/dev/null | sed '/hf_/d'
for new in IPh_LFT_s0 IPh_LA_s0; do
  mkdir -p "$H/pull/$new"
  for f in imag_log_$new.npz p45_$new.npz imagplan_run_$new.txt; do
    bash "$C" "$R/t1/$new/ticks/$f" "$H/pull/$new/$f" < /dev/null > /dev/null 2>&1
    a=$(timeout 60 bash "$P" "sha256sum $R/t1/$new/ticks/$f | cut -d' ' -f1" 2>/dev/null | tr -cd 'a-f0-9'); b=$(sha256sum "$H/pull/$new/$f" 2>/dev/null | cut -d' ' -f1)
    if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "[$(ts)] pulled $new/$f sha256 ${a:0:16}"; else echo "[$(ts)] PULL-MISMATCH $new/$f"; fi
  done
done
echo "ZZ REPLAY-N4B DONE $(ts)"
