#!/usr/bin/env bash
# wait_lr_smoke_b.sh -- pushes the files of the longroll smoke (second build) to the pod build folder, starts pod_lr_smoke_b.sh, waits for its marker (script polling, no model turn)
# and pulls its stdout. Known values and shapes only; no outcome number. Last line: ZZ WAIT-LRSMOKEB DONE | STOP <why> | TIMEOUT
set -u
I=/f/TanitAD-scratch/v8proof/integ10k; P=/c/Users/Admin/v8pod/pssh.sh
W=/workspace/v8proof/integ10k; B=$W/build; S=$W/lr_smoke_b
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "${PODT:-60}" bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
for f in v8f_t1_eval_reader_rule_rzgfhq.py check_preroll_b.py make_longroll_grid.py pod_lr_smoke_b.sh; do
  b=$(sha256sum "$I/$f" | cut -c1-64)
  a=$(pod "[ -f $B/$f ] && sha256sum $B/$f | cut -c1-64" | tr -cd 'a-f0-9')
  if [ -z "$a" ]; then
    bash "$P" "cat > $B/$f" < "$I/$f" 2>/dev/null
    a=$(pod "sha256sum $B/$f | cut -c1-64" | tr -cd 'a-f0-9')
  fi
  [ "$a" = "$b" ] || { echo "ZZ WAIT-LRSMOKEB STOP push mismatch $f"; exit 3; }
  echo "[$(ts)] on the pod: $f ${b:0:16}"
done
for f in v8f_t1_eval_reader_rule_rzgfh.py cmp_dumps.py; do
  b=$(sha256sum "$I/$f" 2>/dev/null | cut -c1-64); a=$(pod "sha256sum $B/$f | cut -c1-64" | tr -cd 'a-f0-9')
  [ -n "$a" ] || { echo "ZZ WAIT-LRSMOKEB STOP missing on the pod: $f"; exit 3; }
  [ -z "$b" ] || [ "$a" = "$b" ] || { echo "ZZ WAIT-LRSMOKEB STOP pod copy differs: $f"; exit 3; }
  echo "[$(ts)] present on the pod: $f ${a:0:16}"
done
[ "$(pod "ls $B/dist_rule.py $W/qmap_sweep/QMAP_LR16_s0_q50.json 2>/dev/null | wc -l" | tr -cd '0-9')" = 2 ] || { echo "ZZ WAIT-LRSMOKEB STOP dist_rule.py or the map file is missing"; exit 3; }
[ "$(pod "bash -n $B/pod_lr_smoke_b.sh && echo ok" | tr -cd 'a-z')" = ok ] || { echo "ZZ WAIT-LRSMOKEB STOP bash -n failed"; exit 3; }
fr=$(pod 'mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk "/^anon /{print \$2}" /sys/fs/cgroup/memory.stat); echo $(( (mm - an) / 1073741824 ))' | tr -cd '0-9')
echo "[$(ts)] pod memory free ${fr} GB"; [ "${fr:-0}" -ge 20 ] || { echo "ZZ WAIT-LRSMOKEB STOP memory ${fr} GB"; exit 3; }
s=$(pod "ls -d $S 2>/dev/null | wc -l" | tr -cd '0-9')
if [ "$s" = 0 ]; then
  pod "cd $B && QMAP_FILE=$W/qmap_sweep/QMAP_LR16_s0_q50.json nohup setsid bash $B/pod_lr_smoke_b.sh > $W/lr_smoke_b_stdout.txt 2>&1 < /dev/null & sleep 2; echo started" > /dev/null
  echo "[$(ts)] started on the pod"
else
  echo "[$(ts)] folder exists: waiting only"
fi
k=0; st=TIMEOUT
while [ "$k" -le 150 ]; do
  s=$(pod "ls $S/LRSMOKEB_DONE.txt $S/LRSMOKEB_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '")
  case "$s" in *STOP*) st=STOP; break;; *DONE*) st=DONE; break;; esac
  if [ "$k" -gt 2 ] && [ "$(pod "pgrep -f 'pod_lr_smoke_[b].sh' | wc -l" | tr -cd '0-9')" = 0 ]; then
    s=$(pod "ls $S/LRSMOKEB_DONE.txt $S/LRSMOKEB_STOP.txt 2>/dev/null | wc -l" | tr -cd '0-9'); [ "$s" = 0 ] && { st="STOP script ended without a marker"; break; }
  fi
  k=$((k+1)); sleep 30
done
pod "cat $W/lr_smoke_b_stdout.txt" | cut -c1-400 > "$I/lr_smoke_b_stdout.txt"
echo "[$(ts)] pulled stdout: $(wc -l < "$I/lr_smoke_b_stdout.txt") lines"
tail -n 3 "$I/lr_smoke_b_stdout.txt" | cut -c1-300
echo "ZZ WAIT-LRSMOKEB $st $(ts)"
