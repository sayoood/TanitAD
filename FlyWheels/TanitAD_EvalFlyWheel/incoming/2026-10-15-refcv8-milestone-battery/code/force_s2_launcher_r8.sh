#!/bin/sh
# force_s2_launcher_r8.sh -- GATED launcher of the S2 force re-roll (force_s2_reroll_r8.py).  CPU ONLY.  Start it ONCE, detached:
#     nohup sh <PKG>/code/force_s2_launcher_r8.sh >> /e/refcv8_eval_kit/chain/force_s2.out 2>&1 &
# and verify it is alive:  ps -ef | grep -c '[f]orce_s2'
#
# It WAITS for ALL of (in this order):
#   G1  the 30k queue has ended:                    the end marker in E:/refcv8_eval_kit/chain/queue_30000_r8.log
#   G2  the float64 probe has ended:                a TERMINAL marker in <PKG>/raw/step30000/wrapper_fp64/raw/run.log that appears AFTER the LAST wait marker
#                                                   of that file.  (The file ALREADY holds a stale combine-fail line from the aborted 19:56 attempt -- a plain
#                                                   grep for the terminal markers would open this gate at once.  Only lines after the last wait line count.)
#                                                   If the probe's script is dead and left no terminal marker, the gate opens with a loud F64DEAD marker.
#   G3  the earlier force_lon roll has finished     (its end marker in E:/refcv8_eval_kit/force_lon_scratch/roll_full.log) or its log has gone stale (> 15 min
#                                                   without a write = not alive).  ADDED BY THE IMPLEMENTER beyond the brief's list: two 8-thread CPU rolls
#                                                   would squeeze each other and the 7.6 GB-per-child commit; the gate costs nothing when it is already done.
#   G4  free commit >= 12 GB on TWO consecutive checks 3 min apart (any miss restarts the pair).
# then runs  select -> roll --driver 4 -> analyze  and writes ZZFS2...ZZ markers to E:/refcv8_eval_kit/chain/force_s2.log.  Each stage is judged by its
# ARTIFACT (never by an exit code, never through a pipe): select by plan_raw.json + s2_ref.npz + its own end marker in this run's log, the roll by the
# `status` stage reading roll_full.json against the plan, analyze by FORCE_S2.json + its own end marker.
#
# TEST HOOKS (all optional; defaults are the real thing): FS2_PKG FS2_QUEUE_LOG FS2_F64_LOG FS2_FL_LOG FS2_LOG FS2_SCR FS2_POLL_S FS2_GAP_S FS2_NEED_GB
#   FS2_MAX_WAIT_S FS2_FREE_CMD (prints free GB) FS2_F64_ALIVE_CMD (prints a count) FS2_DRY=1 (stop after the gate opens; no stage runs) FS2_PY.
PKG=${FS2_PKG:-C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}
Q=${FS2_QUEUE_LOG:-/e/refcv8_eval_kit/chain/queue_30000_r8.log}
F=${FS2_F64_LOG:-$PKG/raw/step30000/wrapper_fp64/raw/run.log}
FLL=${FS2_FL_LOG:-/e/refcv8_eval_kit/force_lon_scratch/roll_full.log}
LOG=${FS2_LOG:-/e/refcv8_eval_kit/chain/force_s2.log}
SCRD=${FS2_SCR:-/e/refcv8_eval_kit/force_s2_scratch}
POLL=${FS2_POLL_S:-60}
GAP=${FS2_GAP_S:-180}
NEED=${FS2_NEED_GB:-12}
MAXW=${FS2_MAX_WAIT_S:-172800}
PY=${FS2_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}
OUTD=$PKG/raw/force_s2

mk() { echo "ZZFS2$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }

# ---- single-launcher guard (a mkdir lock holding our pid; a dead holder is reclaimed)
mkdir -p "$SCRD" 2>/dev/null
LK=$SCRD/launcher.lock
if ! mkdir "$LK" 2>/dev/null; then
  op=$(cat "$LK/pid" 2>/dev/null)
  if [ -n "$op" ] && ps -p "$op" >/dev/null 2>&1; then mk ALREADY "another launcher holds $LK (pid $op)"; exit 6; fi
  rm -rf "$LK"; mkdir "$LK" 2>/dev/null || { mk ALREADY "cannot take $LK"; exit 6; }
fi
echo $$ > "$LK/pid"
trap 'rm -rf "$LK"' EXIT

hr() {      # free commit in whole GB; anything that is not a plain integer counts as 0 (an unreadable probe never counts as "enough")
  if [ -n "$FS2_FREE_CMD" ]; then v=$(sh -c "$FS2_FREE_CMD" 2>/dev/null | tr -d '\r '); else
    v=$(powershell -NoProfile -Command '$o=Get-CimInstance Win32_OperatingSystem; [math]::Floor($o.FreeVirtualMemory/1MB)' 2>/dev/null | tr -d '\r '); fi
  case "$v" in ''|*[!0-9]*) v=0;; esac
  echo "$v"
}
q_done() { grep -q ZZQQUEUEENDZZ "$Q" 2>/dev/null; }
f64_term() {  # terminal markers counted ONLY after the last wait marker
  [ -f "$F" ] || return 1
  c=$(awk '/ZZF64WAITZZ/{c=0; next} /ZZF64(DONE|ARMFAIL|COMBINEFAIL|TIMEOUT)ZZ/{c++} END{print c+0}' "$F" 2>/dev/null)
  [ "${c:-0}" -ge 1 ]
}
f64_alive() {
  if [ -n "$FS2_F64_ALIVE_CMD" ]; then c=$(sh -c "$FS2_F64_ALIVE_CMD" 2>/dev/null | tr -d '\r '); else c=$(ps -ef 2>/dev/null | grep -c '[r]un_fp64_30k'); fi
  [ "${c:-0}" -ge 1 ] 2>/dev/null
}
fl_busy() {   # the force_lon roll is alive: no end marker yet AND its log was written within the last 15 minutes
  grep -q ZZFLDRIVERENDZZ "$FLL" 2>/dev/null && return 1
  [ -f "$FLL" ] || return 1
  [ -n "$(find "$FLL" -mmin -15 2>/dev/null)" ]
}

t0=$(date +%s); last_hb=0
mk ARMED "gate: queue end + float64 probe end + force_lon roll done + free commit >= ${NEED} GB twice ${GAP}s apart; poll ${POLL}s"
while :; do
  now=$(date +%s)
  if [ $((now - t0)) -gt "$MAXW" ]; then mk TIMEOUT "gate not open after ${MAXW}s"; exit 4; fi
  g1=0; g2=0; g3=0
  q_done && g1=1
  if [ $g1 -eq 1 ]; then
    if f64_term; then g2=1; elif ! f64_alive; then g2=2; fi
  fi
  if [ $g1 -eq 1 ] && [ $g2 -ge 1 ]; then fl_busy || g3=1; fi
  if [ $g1 -eq 1 ] && [ $g2 -ge 1 ] && [ $g3 -eq 1 ]; then
    h1=$(hr)
    if [ "$h1" -ge "$NEED" ]; then
      sleep "$GAP"
      h2=$(hr)
      if [ "$h2" -ge "$NEED" ]; then break; fi
      mk FREE "second check ${h2} GB < ${NEED}: the pair restarts"
      continue
    fi
  fi
  if [ $((now - last_hb)) -ge 1800 ]; then
    last_hb=$now; mk WAIT "queue_end=$g1 f64=$g2 (1=terminal 2=probe dead) force_lon_done=$g3 free=$(hr) GB"
  fi
  sleep "$POLL"
done
[ "$g2" -eq 2 ] && mk F64DEAD "the float64 probe script is gone and left no terminal marker: proceeding"
mk GATEOPEN "free commit ${h1}/${h2} GB"
[ -n "$FS2_DRY" ] && { mk DRYEND "FS2_DRY set: no stage was run"; exit 0; }

export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 R8_NO_POSE_SYNC=1 PYTHONIOENCODING=utf-8
export PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKG/code"
cd "$PKG/code" || { mk FAIL "cd $PKG/code"; exit 2; }

mk SELECTSTART "e1_decomp loader + dv2 sets (CPU)"
"$PY" -X utf8 -B force_s2_reroll_r8.py select > "$SCRD/select.log" 2>&1
if [ -s "$SCRD/plan_raw.json" ] && [ -s "$SCRD/s2_ref.npz" ] && grep -q ZZFS2SELECTOKZZ "$SCRD/select.log"; then
  mk SELECTOK "plan_raw.json + s2_ref.npz written; this run's log carries its end marker"
else
  mk SELECTFAIL "see $SCRD/select.log (a set-size control refusal reads: regenerated sets != 434 / 424)"; exit 3
fi

pass=0
while [ $pass -lt 2 ]; do
  pass=$((pass + 1))
  mk ROLLSTART "driver 4, pass $pass (resumable)"
  "$PY" -X utf8 -B -u force_s2_reroll_r8.py roll --tag full --driver 4 --save-every 2 >> "$SCRD/roll_full.log" 2>&1
  st=$("$PY" -X utf8 -B force_s2_reroll_r8.py status --tag full 2>/dev/null | grep -o 'rolled=[0-9]*/[0-9]*' | head -1)
  a=${st#rolled=}; b=${a#*/}; a=${a%/*}
  case "$a$b" in ''|*[!0-9]*) a=0; b=0;; esac
  if [ "$b" -gt 0 ] && [ "$a" -eq "$b" ]; then mk ROLLDONE "rolled=$a/$b"; break; fi
  mk ROLLINCOMPLETE "rolled=$a/$b after pass $pass"
done
if [ "$b" -le 0 ] || [ "$a" -ne "$b" ]; then mk ROLLFAIL "rolled=$a/$b: analyze NOT run (re-run the roll command from HANDOFF.md; it is resumable)"; exit 5; fi

mk ANALYZESTART "numpy on the banked arrays + the roll artifact"
"$PY" -X utf8 -B force_s2_reroll_r8.py analyze --tag full > "$SCRD/analyze_full.log" 2>&1
if [ -s "$OUTD/FORCE_S2.json" ] && grep -q ZZFS2ANALYZEDONEZZ "$SCRD/analyze_full.log"; then
  mk ANALYZEDONE "$(grep -m1 '"verdict"' "$SCRD/analyze_full.log" | tr -d '\r')"
else
  mk ANALYZEFAIL "see $SCRD/analyze_full.log"; exit 7
fi
exit 0
