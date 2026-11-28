#!/usr/bin/env bash
# seq_leadacc.sh -- card ADD153 (ACCHIST10K) on box 2, one job at a time (PI 2026-11-28: "Card, build and run it").
# 1. Waits until box 2 is free (no BOX2_BUSY.txt, no python.exe with a D:\tanitad command line) and card ADD152 is through (LOOP_STATE says hold 155 landed,
#    or box 2 was seen busy after this script started and then stayed free for 45 consecutive minutes; a box never seen busy does not count). 2. Smoke job; stops if its projection of the card exceeds 11 h (the PI is asked).
# 3. The four stage-1 arms. 4. select_stage2.py picks the arm with the highest held-out A; its two stage-2 jobs are shipped and run.
# Each job: chain_rhacc.sh (BOX2_BUSY, telemetry, temperature rule matching "leadacc"); the next starts only after queue-end; results pulled after each job.
# Markers: ZZSEQ <job> <end line>, ZZSEQ-END <state>. Script polling, not model turns.
set -u
C=F:/TanitAD-scratch/v8proof/rhacc10k/chain
LA=F:/TanitAD-scratch/v8proof/rulefc/leadacc
B=/c/Users/Admin/box2_ship
LS=/e/Projects/TanitAD-artifacts/dagland/LOOP_STATE_V8PROOF_2026-11-12.md
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
npy() { $SSH "powershell -NoProfile -Command \"@(Get-CimInstance Win32_Process -Filter \\\"Name = 'python.exe'\\\" | Where-Object { \$_.CommandLine -match 'tanitad' }).Count\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$'; }
free_now() { [ ! -f $B/BOX2_BUSY.txt ] && [ "$(npy)" = "0" ]; }
wait_first() {
  local fm=0 i=0 seen=0
  while :; do
    if free_now; then fm=$((fm + 1)); else fm=0; seen=1; fi
    if [ $fm -ge 1 ] && grep -qE "hold 155 .*landed|hold 155 landed" $LS; then echo "$(ts) box 2 free, hold 155 landed"; return 0; fi
    if [ $seen -eq 1 ] && [ $fm -ge 45 ]; then echo "$(ts) box 2 seen busy, then free for 45 min"; return 0; fi
    i=$((i + 1)); [ $((i % 30)) -eq 0 ] && echo "$(ts) waiting (free minutes $fm, seen busy $seen)"
    [ $i -ge 2880 ] && { echo "ZZSEQ-END waited-48h $(ts)"; exit 0; }
    sleep 60
  done
}
pull() {   # pull <job> <remote out dir>: ACC.json / SMOKE.json and the job log into rulefc/leadacc/box2_<job>/
  mkdir -p $LA/box2_$1
  printf -- '-get /D:/tanitad/logs/%s.log %s/box2_%s/%s.log\n-get /%s/ACC.json %s/box2_%s/ACC.json\n-get /%s/SMOKE.json %s/box2_%s/SMOKE.json\n' \
    "$1" "$LA" "$1" "$1" "$2" "$LA" "$1" "$2" "$LA" "$1" > $LA/box2_$1/pull.sftp
  sftp -o BatchMode=yes -b $LA/box2_$1/pull.sftp tanitad-box2 > /dev/null 2>&1
}
run() {    # run <job> <remote out dir>
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
echo "$(ts) seq_leadacc start"
wait_first
run acc_smoke D:/tanitad/leadacc/smoke
st=$($PY -I -c "import json,sys; r=json.load(open(sys.argv[1])); p=r.get('projection',{}); print('STOP' if p.get('stop', True) else 'GO', p.get('total_hours'))" $LA/box2_acc_smoke/SMOKE.json 2>&1)
echo "$(ts) smoke projection: $st"
case "$st" in GO*) ;; *) echo "ZZSEQ-END stop-projection $(ts)"; exit 0 ;; esac
for a in K25D K25DM K10D K40D; do run acc_$a D:/tanitad/leadacc/$a; done
$PY -I $LA/select_stage2.py > $LA/SELECT.out 2>&1; rc=$?
best=$($PY -I -c "import json,sys; print(json.load(open(sys.argv[1]))['selected'])" $LA/SELECT.json)
echo "$(ts) stage 2 selected: $best (rc $rc)"
[ $rc -eq 0 ] || { echo "ZZSEQ-END no-stage1-arm-passed $(ts)"; exit 0; }
printf 'put %s/jobs/acc_s2rep.cmd /D:/tanitad/jobs/acc_s2rep.cmd\nput %s/jobs/acc_s2smooth.cmd /D:/tanitad/jobs/acc_s2smooth.cmd\n' "$LA" "$LA" > $LA/ship_s2.sftp
sftp -o BatchMode=yes -b $LA/ship_s2.sftp tanitad-box2 > /dev/null 2>&1 || { echo "ZZSEQ-END ship-s2-failed $(ts)"; exit 0; }
run acc_s2rep D:/tanitad/leadacc/${best}_s149
run acc_s2smooth D:/tanitad/leadacc/${best}S
echo "ZZSEQ-END all-done $(ts)"
