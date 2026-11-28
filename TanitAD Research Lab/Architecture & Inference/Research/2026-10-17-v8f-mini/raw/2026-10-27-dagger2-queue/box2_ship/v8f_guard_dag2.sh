#!/usr/bin/env bash
# COPY of v8f_guard_dag.sh (itself a copy of v8f_guard.sh) for PREREG_V8F_DAGGER2's chain, byte-identical in its logic -- the end test is ANCHORED (^ZZV8FMINI|ZZBETA|ZZV8FTAC -DONE|-STOP) and reads the chain log named by V8F_CHAIN_OUT: the ONLY change is that kill_ours also stops the bank generators (v8f_dagger_bank / v8f_dagger_queue_tool), which read cache_full on the USB HDD.
# Box 2 guard for the v8F-mini chain (PREREG_V8F_MINI Amendment 4, operational rules, fixed before any run):
#   TEMPERATURE: the bus-powered USB HDD (33k power-on hours) holds cache_full; temp_guard.sh's rule, unchanged:
#                STOP if >= 4 of the last 20 telemetry samples (15 s apart) read >= 61 C, or any one reads >= 63 C.
#   TIME:        once the running trainer has logged >= 1,000 steps, its rate over the last 20 log records (1,000 steps) x the run's
#                step total must stay <= 10 h. Above that the input pipeline, not the model, is the experiment: STOP and report.
# Stop = Stop-Process on OUR job's python processes only (v8f_mini_train / v8f_mini_eval / v8f_lora_smoke / v8f_lora_identity_dev / v8f_kscore) and their workers. A stopped run resumes
# from ckpt_last.pt on the same argv. Exits when the chain ends. Script polling, not model turns.
# 2026-10-20: the end test is ANCHORED (^MARKER) and knows PREREG_V8F_TAC's chain (ZZV8FTAC-DONE / -STOP). Unanchored, the same
# test in ra1runs_guard.sh matched a chain's own "waiting for <marker>" line and switched itself off (2026-10-20 22:43).
B=/c/Users/Admin/box2_ship; L=$B/logs
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
MAXH=${V8F_MAX_RUN_H:-10}
CHAIN_OUT=${V8F_CHAIN_OUT:-$L/chain_v8f_mini.out}       # chain_beta.sh points this at its own log; the default is the v8F-mini chain's
kill_ours() {
  $SSH "powershell -NoProfile -Command \"\$p = @(Get-CimInstance Win32_Process -Filter \\\"Name = 'python.exe'\\\"); \$ours = @(\$p | Where-Object { \$_.CommandLine -match 'v8f_mini_train|v8f_mini_eval|v8f_dagger_bank|v8f_dagger_queue_tool|v8f_lora_smoke|v8f_lora_identity_dev|v8f_kscore' }); \$ids = @(\$ours | ForEach-Object { \$_.ProcessId }); \$kids = @(\$p | Where-Object { \$ids -contains \$_.ParentProcessId }); (\$ours + \$kids) | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue; 'stopped ' + \$_.ProcessId }\"" 2>/dev/null | tr -d '\r'
}
echo "$(date '+%m-%d %H:%M') v8F guard armed (USB >= 4/20 samples >= 61 C or one >= 63 C; projected run > ${MAXH} h after 1,000 steps)"
hi=0
while :; do
  grep -q -E '^(ZZV8FMINI|ZZBETA|ZZV8FTAC)-(DONE|STOP)' "$CHAIN_OUT" 2>/dev/null && { echo "$(date '+%m-%d %H:%M') chain ended; guard off (max USB ${hi} C)"; exit 0; }
  t=$($SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 20\"" 2>/dev/null | tr -d '\r' | grep -E '^20' | cut -d, -f11 | grep -E '^[0-9]+$')
  if [ -n "$t" ]; then
    n61=$(echo "$t" | awk '$1 >= 61' | wc -l); m=$(echo "$t" | sort -n | tail -1)
    [ "$m" -gt "$hi" ] && { hi=$m; echo "$(date '+%m-%d %H:%M') new max USB temp $m C"; }
    if [ "$n61" -ge 4 ] || [ "$m" -ge 63 ]; then
      echo "ZZGUARD-STOP TEMP $(date '+%m-%d %H:%M') USB HDD: $n61 of the last 20 samples >= 61 C, max $m C"; kill_ours; exit 0
    fi
  fi
  run=$(cat ${V8F_CURRENT:-$L/v8f_current.txt} 2>/dev/null)
  if [ -n "$run" ]; then
    $SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/runs/v8f_mini_full/$run/train_log.jsonl) { Get-Content D:/tanitad/runs/v8f_mini_full/$run/train_log.jsonl -Tail 21 }\"" 2>/dev/null | tr -d '\r' > $L/v8f_guard_tail.jsonl
    $SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/runs/v8f_mini_full/$run/selfcheck.json) { Get-Content D:/tanitad/runs/v8f_mini_full/$run/selfcheck.json -Raw }\"" 2>/dev/null | tr -d '\r' > $L/v8f_guard_sc.json
    p=$($PY - "$(cygpath -m $L/v8f_guard_tail.jsonl)" "$(cygpath -m $L/v8f_guard_sc.json)" <<'EOF'
import json, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip().startswith("{")]
    total = json.load(open(sys.argv[2], encoding="utf-8"))["steps_total"]
except Exception:
    sys.exit(0)
rows = [r for r in rows if "step" in r and "sec" in r]
if len(rows) < 21 or rows[-1]["step"] < 1000:
    sys.exit(0)
a, b = rows[0], rows[-1]
if b["sec"] <= a["sec"] or b["step"] <= a["step"]:          # a resume restarted the clock inside the window: wait for 20 clean records
    sys.exit(0)
rate = (b["sec"] - a["sec"]) / (b["step"] - a["step"])
print(f"{rate * total / 3600:.2f} {rate:.3f} {b['step']} {total}")
EOF
)
    if [ -n "$p" ]; then
      set -- $p
      echo "$(date '+%m-%d %H:%M') $run: step $3/$4, $2 s/step, projected $1 h" >> $L/v8f_guard_rate.log
      if awk -v h="$1" -v m="$MAXH" 'BEGIN{exit !(h > m)}'; then
        echo "ZZGUARD-STOP TIME $(date '+%m-%d %H:%M') $run projects $1 h (> ${MAXH} h) at $2 s/step"; kill_ours; exit 0
      fi
    fi
  fi
  sleep 60
done
