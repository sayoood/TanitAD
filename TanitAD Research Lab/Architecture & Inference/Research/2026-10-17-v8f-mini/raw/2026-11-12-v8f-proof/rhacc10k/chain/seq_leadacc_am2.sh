#!/usr/bin/env bash
# seq_leadacc_am2.sh -- card ADD153 amendment 2 on box 2, one job at a time, AFTER card ADD154's job (box-2 order agreed with the coordinator).
# 1. Waits until box 2 was seen busy after this script started and then stayed free (no BOX2_BUSY.txt, no tanitad python) for 15 consecutive minutes;
#    if box 2 is never seen busy within 90 minutes, it starts once box 2 is free. 2. The two training jobs. 3. mk_jobs_am2.py read writes the ensemble job
#    from the pulled ACC.json files; it is shipped and run. Markers: ZZSEQ <job> <end line>, ZZSEQ-END <state>. Script polling, not model turns.
set -u
C=F:/TanitAD-scratch/v8proof/rhacc10k/chain
LA=F:/TanitAD-scratch/v8proof/rulefc/leadacc
B=/c/Users/Admin/box2_ship
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
npy() { $SSH "powershell -NoProfile -Command \"@(Get-CimInstance Win32_Process -Filter \\\"Name = 'python.exe'\\\" | Where-Object { \$_.CommandLine -match 'tanitad' }).Count\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$'; }
free_now() { [ ! -f $B/BOX2_BUSY.txt ] && [ "$(npy)" = "0" ]; }
wait_after_other() {
  local fm=0 i=0 seen=0
  while :; do
    if free_now; then fm=$((fm + 1)); else fm=0; seen=1; fi
    if [ $seen -eq 1 ] && [ $fm -ge 15 ]; then echo "$(ts) box 2 seen busy, then free for 15 min"; return 0; fi
    if [ $seen -eq 0 ] && [ $i -ge 90 ] && [ $fm -ge 1 ]; then echo "$(ts) box 2 not seen busy in 90 min; free now"; return 0; fi
    i=$((i + 1)); [ $((i % 30)) -eq 0 ] && echo "$(ts) waiting (free minutes $fm, seen busy $seen)"
    [ $i -ge 2880 ] && { echo "ZZSEQ-END waited-48h $(ts)"; exit 0; }
    sleep 60
  done
}
pull() {
  mkdir -p $LA/box2_$1
  printf -- '-get /D:/tanitad/logs/%s.log %s/box2_%s/%s.log\n-get /%s/ACC.json %s/box2_%s/ACC.json\n-get /%s/ENSN.json %s/box2_%s/ENSN.json\n' \
    "$1" "$LA" "$1" "$1" "$2" "$LA" "$1" "$2" "$LA" "$1" > $LA/box2_$1/pull.sftp
  sftp -o BatchMode=yes -b $LA/box2_$1/pull.sftp tanitad-box2 > /dev/null 2>&1
}
run() {
  local e n=0
  while :; do
    bash $C/chain_rhacc.sh $1 leadacc > $C/chain_$1.out 2>&1
    e=$(grep "ZZRHACC-END" $C/chain_$1.out | tail -1)
    case "$e" in *"busy-held"*) n=$((n + 1)); [ $n -ge 240 ] && break; sleep 60; until free_now; do sleep 60; done ;; *) break ;; esac
  done
  echo "ZZSEQ $1 $e"
  pull $1 $2
  case "$e" in *"queue-end"*) ;; *) echo "ZZSEQ-END stopped-after-$1 $(ts)"; exit 0 ;; esac
}
echo "$(ts) seq_leadacc_am2 start"
wait_after_other
run acc_a2_K25D_s150 D:/tanitad/leadacc/K25D_s150
run acc_a2_K25DS_s149 D:/tanitad/leadacc/K25DS_s149
$PY -I $LA/mk_jobs_am2.py read > $LA/mk_jobs_am2_read.out 2>&1 || { echo "ZZSEQ-END read-job-refused $(ts)"; exit 0; }
printf 'put %s/jobs/acc_a2_ens4.cmd /D:/tanitad/jobs/acc_a2_ens4.cmd\n' "$LA" > $LA/ship_a2.sftp
sftp -o BatchMode=yes -b $LA/ship_a2.sftp tanitad-box2 > /dev/null 2>&1 || { echo "ZZSEQ-END ship-failed $(ts)"; exit 0; }
run acc_a2_ens4 D:/tanitad/leadacc/ens4
echo "ZZSEQ-END all-done $(ts)"
