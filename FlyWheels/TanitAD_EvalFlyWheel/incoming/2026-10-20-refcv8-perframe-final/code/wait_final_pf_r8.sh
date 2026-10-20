#!/bin/sh
# wait_final_pf_r8.sh -- GUARDIAN of the per-frame final read.   usage: nohup sh wait_final_pf_r8.sh < /dev/null > <out> 2>&1 &
#   NEW file (EvalFlyWheel 2026-10-20, hold_r8pf).  Owns three duties, nothing else:
#   1. runs chain_final_pf_r8.sh in the FOREGROUND (argv token `as=chain_final_r8`, so yield48852_r8.sh-aware chains yield to it); relaunches it ONCE if it ends without a terminal marker
#      (ZZPFENDZZ / ZZPFABORTZZ after its last ZZPFSTARTZZ) -- the chain is resumable from its artifacts;
#   2. GUARANTEES the per-clip chain_final_r8.sh comes back on every path: if the chain did not restart it (state file cfn_restarted absent), the guardian restarts it -- but only once
#      the pf GPU job no longer holds the lock and no run_battery_r8_pf process is alive (cap 6 h; a foreign lock is never touched);
#   3. BESIDE refresh: waits (cap 60 h) for the per-clip verdict SHIP_R8_step48852ema_unc.json and re-runs ship_r8_verdict_pf.py once so the per-clip verdict is printed beside.
#   It refuses to start while chain_final_r8.sh is alive: the operator stops that chain first, by explicit PID (the pf-first guarantee is that it is NOT running before stage P).
#   Markers ZZPFG...ZZ -> E:/refcv8_eval_kit/chain/chain_final_pf.log (the chain's own log).
HOLDW="${PF_HOLDW:-C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final}"
HOLDL="${PF_HOLDL:-/c/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final}"
PKGL="${PF_R8PKGL:-/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
KIT="${PF_KIT:-/e/refcv8_eval_kit}"
PY="${PF_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
CHAIN="${PF_CHAIN:-$HOLDL/code/chain_final_pf_r8.sh}"
STATE="${PF_STATE:-/c/Users/Admin/refcv8_pf/state}"
LOG="${PF_LOG:-$KIT/chain/chain_final_pf.log}"
LOCKF="${PF_LOCKF:-/c/Users/Admin/qland/work/refcv7/devbox_gpu.lock}"
JOB="refcv8-battery-step48852ema_pf"
CFN_PIDF="${PF_CFN_PIDF:-$KIT/chain/chain_final.pid}"
CFN_RESTART="${PF_CFN_RESTART:-cd $PKGL && nohup sh code/chain_final_r8.sh < /dev/null >> $KIT/chain/chain_final.out 2>&1 &}"
PERCLIP_V="${PF_PERCLIP_V:-$PKGL/raw/ship_r8/SHIP_R8_step48852ema_unc.json}"
POLL="${PF_POLL_S:-60}"; RESTART_CAP="${PF_RESTART_CAP_S:-21600}"; BESIDE_CAP="${PF_BESIDE_CAP_S:-216000}"
mkdir -p "$STATE" "$(dirname "$LOG")"
mark() { echo "ZZPFG$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
cfn_alive() { [ -s "$CFN_PIDF" ] && kill -0 "$(cat "$CFN_PIDF" 2>/dev/null)" 2>/dev/null; }
GPIDF="$STATE/wait_final_pf.pid"
if [ -s "$GPIDF" ] && kill -0 "$(cat "$GPIDF")" 2>/dev/null; then mark DUP "another guardian (pid $(cat "$GPIDF")) is alive"; exit 5; fi
echo $$ > "$GPIDF"
trap '[ "$(cat "$GPIDF" 2>/dev/null)" = "$$" ] && rm -f "$GPIDF"' EXIT
trap 'exit 143' TERM INT HUP
if cfn_alive; then mark REFUSE "chain_final_r8 (pid $(cat "$CFN_PIDF")) is alive: stop it by explicit PID first (pf-first guarantee)"; exit 30; fi
mark START "pid $$ chain $CHAIN"
terminal() { awk '/ZZPFSTARTZZ/{e=0} /ZZPF(END|ABORT)ZZ/{e=1} END{exit e?0:1}' "$LOG" 2>/dev/null; }
n=0
while :; do
  sh "$CHAIN" as=chain_final_r8 >> "$STATE/chain_final_pf.out" 2>&1
  if terminal; then mark CHAINTERMINAL "$(grep -aoE 'ZZPF(END|ABORT)ZZ.*' "$LOG" | tail -n 1 | cut -c1-200)"; break; fi
  n=$((n+1))
  if [ "$n" -ge 2 ]; then mark CHAINGIVEUP "the chain ended twice without a terminal marker"; break; fi
  mark CHAINRELAUNCH "no terminal marker after its last START: relaunching once (resumable)"
  sleep "$POLL"
done
# ---- duty 2: the per-clip chain comes back ------------------------------------------------------------------- #
lock_ours() { grep -q "\"job\": *\"$JOB\"" "$LOCKF" 2>/dev/null; }
# a PYTHON process (column 6 = the executable) running run_battery_r8_pf.py for THIS tag.  A plain `grep -c` also counted shell wrappers whose command
# line merely QUOTES the name (MEASURED 2026-10-20: 2 decoys, 0 real) -- the self-match trap -- and would have blocked the per-clip restart for 6 h.
pf_py_alive() { ps -ef 2>/dev/null | awk -v t="--tag step48852ema_pf" '$6 ~ /python(\.exe)?$/ && index($0, "run_battery_r8_pf.py") && index($0, t) {n++} END {print n+0}'; }
if [ -e "$STATE/cfn_restarted" ]; then mark CFNOK "the chain restarted chain_final_r8 itself"
else
  _t0=$(date +%s)
  while lock_ours || [ "$(pf_py_alive)" != "0" ]; do
    [ $(( $(date +%s) - _t0 )) -gt "$RESTART_CAP" ] && { mark CFNSTUCK "the pf job still holds the lock / a pf battery process is alive after ${RESTART_CAP} s: NOT restarting chain_final_r8 (a human must look)"; exit 31; }
    sleep "$POLL"
  done
  if cfn_alive; then mark CFNALIVE "chain_final_r8 already running"
  else sh -c "$CFN_RESTART"; : > "$STATE/cfn_restarted"; mark CFNRESTART "guardian restarted chain_final_r8.sh (per-clip reads, never binding)"; fi
fi
sleep 15
if cfn_alive; then mark CFNVERIFIED "chain_final_r8 pid $(cat "$CFN_PIDF") alive"; else mark CFNNOTALIVE "chain_final_r8 is NOT alive 15 s after the restart (its own log: $KIT/chain/chain_final.log; ZZCFNDUPZZ / ZZCFNABORTZZ explain it)"; fi
# ---- duty 3: print the per-clip verdict beside --------------------------------------------------------------- #
[ -s "$HOLDL/raw/ship_r8/SHIP_R8_step48852ema_pf.json" ] || { mark NOVERDICT "no pf verdict to refresh"; exit 0; }
_t0=$(date +%s)
until [ -s "$PERCLIP_V" ]; do
  [ $(( $(date +%s) - _t0 )) -gt "$BESIDE_CAP" ] && { mark BESIDETIMEOUT "no per-clip verdict after ${BESIDE_CAP} s: the pf verdict says NOT YET COMPUTED beside"; exit 0; }
  sleep "$POLL"
done
"$PY" -X utf8 "$HOLDW/code/ship_r8_verdict_pf.py" --tree step48852ema_pf > "$STATE/logs/verdict_refresh.log" 2>&1
OV=$("$PY" -I -c "import json,sys;j=json.load(open(sys.argv[1],encoding='utf-8'));print(j['overall'], (j['beside']['per_clip_verdict_deliberate_regression_arm'] or {}).get('overall'))" \
     "$HOLDW/raw/ship_r8/SHIP_R8_step48852ema_pf.json" 2>/dev/null)
mark BESIDEREFRESH "pf / per-clip-beside: ${OV:-UNREADABLE}"
mark END ""
