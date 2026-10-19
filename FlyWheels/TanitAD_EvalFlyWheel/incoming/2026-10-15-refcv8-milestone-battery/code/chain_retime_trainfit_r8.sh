#!/bin/sh
# chain_retime_trainfit_r8.sh -- the TRAIN-clip rolls for the deployable RETIME test (PREREG_TRAINFIT.md), queued BEHIND the dev-box GPU queue.  NEW file.
#   start (detached, ONCE):  nohup sh <PKG>/code/chain_retime_trainfit_r8.sh < /dev/null > /e/refcv8_eval_kit/chain/trainfit_chain.out 2>&1 &
#   alive?     ps -ef | grep -c '[c]hain_retime_trainfit_r8'          progress:  grep ZZTF /e/refcv8_eval_kit/chain/trainfit_chain.log | tail
#   resumable: re-run the same command; a seed whose roll json already says ok is skipped.
# ORDER (the PI/Master Mind queue as of 2026-10-19 03:05; this chain is LAST, and the step-40k battery outranks it):
#   K-draw queue (holds the GPU lock) -> it writes /c/lgs/RELEASE_R7 -> refcv7 chains + K2/K4 -> (fp64 probe, CPU, last, 16 GB gate).
#   G1  wait for /c/lgs/RELEASE_R7 (cap 24 h)            G2  none of chain_r7_matched_phase / chain_r7on_50400 / queue_k2k4_r8 / kdraw_queue_r8 alive
#   G3  step-40k yield (AMENDMENT 1 A1.5, REAL markers of snap40k_r8.sh in $KIT/chain/snap40k.log): (a) snap40k_r8 alive and no terminal marker after the last ZZS40STARTZZ -> yield;
#       (b) DONE/ALREADY (ckpt_full_40000.pt banked) -> yield until step40000ema_unc has a battery_summary.json; any other terminal marker -> no 40k battery can follow, no yield.
#       Safety cap 48 h from the chain's G3 start. NO 40k battery chain exists on the dev box yet (not invented here); (b) waits until one is wired.
#   M   FreeVirtualMemory >= 10 GB on TWO checks 60 s apart (the dev box COMMIT limit binds)        L  the shared O_EXCL GPU lock (a foreign holder is NEVER broken)
# One lock hold per inference seed (~45 min each, os-only); G2/G3/M are re-evaluated before each seed.  Assert on ARTIFACTS, never an exit code, never through a pipe.
# After both rolls: runs code/chain_retime_trainfit_post_r8.sh (CPU: panels, freeze, forward, gap, report) if that file exists.  Nothing is written to the repo or to G:.
PKGW="C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery"
PKGL="/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery"
BAT="C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code"
KIT="/e/refcv8_eval_kit"; KITW="E:/refcv8_eval_kit"
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
TREE="step30000ema_train139"
TFW="$KITW/trainfit/$TREE"; TF="$KIT/trainfit/$TREE"
LOG="$KIT/chain/trainfit_chain.log"
JOB="refcv8-trainfit"
NEED="${TF_NEED_GB:-10}"
RELEASE_FILE="${TF_RELEASE_FILE:-/c/lgs/RELEASE_R7}"
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6
mkdir -p "$KIT/chain" "$TF" "$PKGL/raw/retime_trainfit/logs"
mark() { echo "ZZTF$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
PIDF="$KIT/chain/trainfit_chain.pid"
if [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then mark DUP "another trainfit chain (pid $(cat "$PIDF")) is alive; exiting"; exit 5; fi
echo $$ > "$PIDF"
HAVE_LOCK=0
gpu_release() {
  [ "$HAVE_LOCK" = 1 ] || return 0
  "$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$KIT/chain/trainfit_lockwait.log" 2>&1
  "$PY" "$BAT/chain_marker.py" remove --chain "$JOB" --pid $$ >> "$KIT/chain/trainfit_lockwait.log" 2>&1
  HAVE_LOCK=0; mark RELEASED "$JOB"
}
finish() { gpu_release; [ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] && rm -f "$PIDF"; }
trap finish EXIT
trap 'exit 143' TERM INT HUP
mark QSTART "pid $$ tree $TREE need ${NEED} GB"
jq_ok() {   # $1 = file, $2 = python boolean expression over `j`
  _f=$(cygpath -m "$1" 2>/dev/null || echo "$1")
  "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)
except Exception: sys.exit(1)" "$_f"
}
free_gb() { v=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)" 2>/dev/null | tr -d '\r ' | tr ',' '.'); case "$v" in ''|*[!0-9.]*) echo 0;; *) echo "$v";; esac; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit (a+0>=b+0)?0:1}'; }
roll_ok() { jq_ok "$TF/roll_s$1.json" "j.get('smoke') is False and j.get('n_windows',0)>=4000 and j.get('n_episodes')==139 and j.get('device')=='cuda'"; }
others_alive() { c=$(ps -ef 2>/dev/null | grep -cE '[c]hain_r7_matched_phase|[c]hain_r7on_50400|[q]ueue_k2k4_r8|[k]draw_queue_r8|[f]orce_s2_launcher'); [ "${c:-0}" -ge 1 ] 2>/dev/null; }
SNAPLOG="${TF_SNAPLOG:-$KIT/chain/snap40k.log}"
# 2026-10-19 03:45 (orchestrator): SSHFAIL removed from the FAILED set -- snap40k_r8.sh logs it per failed poll and RETRIES; only SSHDEAD is terminal. As a FAILED member, one transient wifi miss would end the 40k yield for good.
snap_state() {   # RUNNING | DONE | FAILED | ABSENT, from the markers AFTER the last ZZS40STARTZZ (opaque tokens, searched with awk, never grep -c on a result code)
  alive=$(ps -ef 2>/dev/null | grep -c '[s]nap40k_r8')
  t=$(awk '/ZZS40STARTZZ/{s=NR} END{print s+0}' "$SNAPLOG" 2>/dev/null)
  v=$(awk -v s="${t:-0}" 'NR>s && /ZZS40(DONE|ALREADY)ZZ/{d=1} NR>s && /ZZS40(TIMEOUT|TOOLATE|MD5MISMATCH|SIZEMISMATCH|NODISK|ABORTLATER|SSHDEAD|INCONCLUSIVE|BADSIZE|CHANGED)ZZ/{f=1} END{if(d)print "DONE"; else if(f)print "FAILED"; else print "NONE"}' "$SNAPLOG" 2>/dev/null)
  case "$v" in DONE) echo DONE;; FAILED) echo FAILED;; *) if [ "${alive:-0}" -ge 1 ] 2>/dev/null; then echo RUNNING; else echo ABSENT; fi;; esac
}
b40_reason() {   # prints the reason to yield, or nothing
  case "$(snap_state)" in
    RUNNING) echo "G3a snap40k chain alive, no terminal marker yet" ;;
    DONE) if [ -s "/c/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc/battery_summary.json" ] || [ -s "$KIT/battery/step40000ema_unc/battery_summary.json" ]; then :; else echo "G3b ckpt_full_40000 banked, the 40k battery has not run"; fi ;;
    *) : ;;
  esac
}
# ---- G1 --------------------------------------------------------------------------------------------------------------- #
n=0; until [ -f "$RELEASE_FILE" ]; do n=$((n+1)); [ $((n % 30)) -eq 1 ] && mark G1WAIT "no $RELEASE_FILE yet (K-draw queue still ahead)"; [ $n -gt 720 ] && { mark G1TIMEOUT "24 h"; break; }; sleep 120; done
mark G1OK "$(head -1 "$RELEASE_FILE" 2>/dev/null)"
sleep 900; mark G1GRACE "900 s so that the refcv7 chains (launched at once) and K2/K4 (launched at +480 s) by r7_after_kdraw.sh are alive before G2 looks"
T40=$(date +%s)
NFAIL=0
for s in 0 1; do
  if roll_ok "$s"; then mark ROLLHAVE "seed $s"; continue; fi
  t0=$(date +%s); last=""
  while :; do   # ---- G2 / G3 / M, all must hold on one pass
    if others_alive; then r="G2 refcv7 chains / K2K4 / kdraw alive"
    elif [ -n "$(b40_reason)" ] && [ $(( $(date +%s) - T40 )) -lt 172800 ]; then r="$(b40_reason)"
    else
      h1=$(free_gb)
      if ge "$h1" "$NEED"; then sleep 60; h2=$(free_gb); if ge "$h2" "$NEED"; then mark MEMOK "free commit ${h1}/${h2} GB"; break; fi; r="M second check ${h2} GB < ${NEED}"; else r="M free commit ${h1} GB < ${NEED}"; fi
    fi
    [ "$r" != "$last" ] && { mark WAIT "seed $s: $r"; last="$r"; }
    sleep 120
  done
  rm -f "$KIT/trainfit/lock_s$s.json"
  "$PY" "$BAT/gpu_lock.py" acquire --job "$JOB" --pid $$ --max-wait-s 28800 --out "$KITW/trainfit/lock_s$s.json" >> "$KIT/chain/trainfit_lockwait.log" 2>&1
  jq_ok "$KIT/trainfit/lock_s$s.json" "j.get('ok') is True" || { mark NOLOCK "seed $s: not acquired within 8 h (a foreign holder is never broken)"; NFAIL=$((NFAIL+1)); break; }
  HAVE_LOCK=1
  "$PY" "$BAT/chain_marker.py" create --chain "$JOB" --pid $$ >> "$KIT/chain/trainfit_lockwait.log" 2>&1
  mark LOCKED "seed $s"
  rm -rf "$TF/dump_s$s"; rm -f "$TF/roll_s$s.json"
  "$PY" -X utf8 "$PKGW/code/roll_seed_r8_train.py" --ckpt "$KITW/ckpt/ckpt_ema_30000.pt" --config "$KITW/ckpt/config.json" --seed "$s" \
      --dump-dir "$TFW/dump_s$s" --out-json "$TFW/roll_s$s.json" --device cuda >> "$PKGL/raw/retime_trainfit/logs/roll_s$s.log" 2>&1
  gpu_release
  if roll_ok "$s"; then mark ROLL_s${s}_OK "$(grep -o '"wall_s": [0-9.]*' "$TF/roll_s$s.json" | head -1)"; else mark ROLLFAIL "seed $s: artifact not ok (see $PKGL/raw/retime_trainfit/logs/roll_s$s.log)"; NFAIL=$((NFAIL+1)); fi
done
if [ "$NFAIL" -eq 0 ]; then
  mark ROLLSDONE "both seeds ok"
  if [ -f "$PKGL/code/chain_retime_trainfit_post_r8.sh" ]; then sh "$PKGL/code/chain_retime_trainfit_post_r8.sh" >> "$KIT/chain/trainfit_post.out" 2>&1; mark POSTRAN "see $KIT/chain/trainfit_post.out"; else mark NOPOST "post script absent"; fi
fi
mark END "rc=$NFAIL"
