#!/usr/bin/env bash
# wait_patchcheck_hb.sh -- pushes pod_hbrecord_check.sh, starts it on the pod, waits for its marker (the script polls, not a model turn) and pulls the patch and the two compare files.
# last line: ZZ WAIT-HBREC DONE | STOP <why>
set -u
I=/f/TanitAD-scratch/v8proof/integ10k; P=/c/Users/Admin/v8pod/pssh.sh
W=/workspace/v8proof/integ10k; H=$W/hbrec
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "${PODT:-60}" bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
a=$(pod "[ -f $W/build/pod_hbrecord_check.sh ] && sha256sum $W/build/pod_hbrecord_check.sh | cut -c1-64" | tr -cd 'a-f0-9'); b=$(sha256sum "$I/pod_hbrecord_check.sh" | cut -c1-64)
if [ -z "$a" ]; then
  bash "$P" "cat > $W/build/pod_hbrecord_check.sh" < "$I/pod_hbrecord_check.sh" 2>/dev/null
  a=$(pod "sha256sum $W/build/pod_hbrecord_check.sh | cut -c1-64" | tr -cd 'a-f0-9')
fi
[ "$a" = "$b" ] || { echo "ZZ WAIT-HBREC STOP push mismatch"; exit 3; }
[ "$(pod "bash -n $W/build/pod_hbrecord_check.sh && echo ok" | tr -cd 'a-z')" = ok ] || { echo "ZZ WAIT-HBREC STOP bash -n failed"; exit 3; }
s=$(pod "ls $H/HBREC_START.txt 2>/dev/null | wc -l" | tr -cd '0-9')
if [ "$s" = 0 ]; then
  pod "mkdir -p $H && cd $H && nohup setsid bash $W/build/pod_hbrecord_check.sh > $H/hbrec_stdout.txt 2>&1 < /dev/null & sleep 2; echo started" > /dev/null
  echo "[$(ts)] started on the pod"
else
  echo "[$(ts)] started before: waiting only"
fi
k=0; st=""
while :; do
  s=$(pod "ls $H/HBREC_DONE.txt $H/HBREC_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '")
  case "$s" in *STOP*) st=STOP; break;; *DONE*) st=DONE; break;; esac
  k=$((k+1)); [ "$k" -gt 120 ] && { st=TIMEOUT; break; }
  sleep 30
done
pod "cat $H/hbrec_stdout.txt | cut -c1-240 | tail -n 14"
mkdir -p "$I/hbrec"
for f in CMP_RecOld.txt CMP_RecHb.txt hbrec_stdout.txt; do
  pod "cat $H/$f" > "$I/hbrec/$f"
  x=$(pod "sha256sum $H/$f | cut -c1-64"); y=$(sha256sum "$I/hbrec/$f" | cut -c1-64)
  [ "$x" = "$y" ] && echo "[$(ts)] pulled $f sha256 ${x:0:16}" || echo "[$(ts)] PULL-MISMATCH $f"
done
echo "ZZ WAIT-HBREC $st $(ts)"
