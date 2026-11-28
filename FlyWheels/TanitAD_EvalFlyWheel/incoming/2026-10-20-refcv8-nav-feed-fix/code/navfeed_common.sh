#!/bin/sh
# navfeed_common.sh -- SOURCED by run_navfeed.sh (the GPU wrapper) and wait_navfeed.sh (the waiter): ONE definition of paths, env and the start conditions.
#   (i) the GPU lock is free (taken through the SAME gpu_lock.py the battery chains use; a foreign lock is never broken)
#   (ii) >= $MIN_FREE_MB MB free on the OUTPUT volume (Get-PSDrive + a real 16 MB write test; never `df` alone)
#   (iii) the final (step 48,852) chain is NOT active: `bfinal_pending` below is the yield48852_r8.sh rule verbatim (weights banked AND chain alive AND no
#         END/ABORT AND the EMA battery / K-draw still open), PLUS no PYTHON process whose command line names the 48,852 tag (the Master Mind's wording),
#         PLUS no chain-active marker.  Before the weights are banked the chain is idle in its pull wait (the trainer may be a day away) and is NOT "running".
HOLDW="${NF_HOLDW:-C:/Users/Admin/qland/hold_r8navfix/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-nav-feed-fix}"
HOLDL="${NF_HOLDL:-/c/Users/Admin/qland/hold_r8navfix/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-nav-feed-fix}"
OUTW="${NF_OUTW:-C:/Users/Admin/refcv8_navfeed_out}"; OUTL="${NF_OUTL:-/c/Users/Admin/refcv8_navfeed_out}"
BAT="${NF_BAT:-C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code}"
CKPT="${NF_CKPT:-E:/refcv8_eval_kit/ckpt/ckpt_ema_40000.pt}"; CKPT_L="${NF_CKPT_L:-/e/refcv8_eval_kit/ckpt/ckpt_ema_40000.pt}"
CKPT_MD5="${NF_CKPT_MD5:-57807456bd7169800be159ff95871d22}"          # ckpt_40000.md5 sidecar line for ckpt_ema_40000.pt (E:/refcv8_eval_kit/ckpt/ckpt_40000.md5)
CFG="${NF_CFG:-E:/refcv8_eval_kit/ckpt/config.json}"
ARCH_DUMP="${NF_ARCH_DUMP:-C:/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc/dump_s0}"
PY="${NF_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
JOB="refcv8-navfeed"
MIN_FREE_MB="${NF_MIN_FREE_MB:-3072}"
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 R8_NO_POSE_SYNC=1
# R8_NO_POSE_SYNC=1 = the STANDALONE pose clock the step40000ema_unc tree was rolled on (archived roll log line 1: "pose_sync policy: OFF (R8_NO_POSE_SYNC)")
mkdir -p "$OUTL/state" "$OUTL/logs" "$OUTL/smoke" "$OUTL/seam_tele"
export R8_SEAM_TELE_DIR="$OUTW/seam_tele"        # r8_repoint.patch_seam_guard writes its saturation telemetry here (a count; no behaviour change)
LOG="$OUTL/navfeed.log"
mark() { echo "ZZNF$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
jq_ok() {   # $1 = file, $2 = python boolean expression over `j`; exit 0 iff the file parses and the expression is true
  _f=$(cygpath -m "$1" 2>/dev/null || echo "$1")
  "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)
except Exception: sys.exit(1)" "$_f"
}
free_mb() {   # integer MB free on the OUTPUT volume (Get-PSDrive, not df)
  _d=$(echo "$OUTW" | cut -c1)
  _v=$(powershell -NoProfile -Command "[math]::Floor((Get-PSDrive $_d).Free/1MB)" 2>/dev/null | tr -d '\r ')
  case "$_v" in ''|*[!0-9]*) echo 0 ;; *) echo "$_v" ;; esac
}
write_test() {   # a REAL write test on the output volume (16 MB), then remove it
  dd if=/dev/zero of="$OUTL/.nf_writetest" bs=1048576 count=16 2>/dev/null
  _sz=$(wc -c < "$OUTL/.nf_writetest" 2>/dev/null | tr -d ' ')
  rm -f "$OUTL/.nf_writetest"
  [ "$_sz" = "16777216" ]
}
# ---- the final-chain yield rule: COPIED from yield48852_r8.sh `bfinal_pending` (battery package, 2026-10-19), unchanged in meaning -------------
Y48_BANK="${Y48_BANK:-/e/refcv8_eval_kit/ckpt/ckpt_full_48852.pt}"
Y48_SUM="${Y48_SUM:-/c/Users/Admin/refcv8_eval_archive/battery/step48852ema_unc/battery_summary.json}"
Y48_LOG="${Y48_LOG:-/e/refcv8_eval_kit/chain/chain_final.log}"
bfinal_pending() {
  [ -f "$Y48_BANK" ] || return 1
  _y48c=$(ps -ef 2>/dev/null | grep -c '[c]hain_final_r8'); [ "${_y48c:-0}" -ge 1 ] 2>/dev/null || return 1
  awk '/ZZCFNSTARTZZ/{e=0} /ZZCFN(ABORT|END)ZZ/{e=1} END{exit e}' "$Y48_LOG" 2>/dev/null || return 1
  if [ ! -s "$Y48_SUM" ]; then
    awk '/ZZCFNSTARTZZ/{e=0} /ZZCFNEMAFAILZZ/{e=1} END{exit e}' "$Y48_LOG" 2>/dev/null && return 0
  fi
  awk '/ZZCFNSTARTZZ/{k=0} /ZZCFNKD(DONE|HAVE|SKIP|FAIL)ZZ/{k=1} END{exit k}' "$Y48_LOG" 2>/dev/null && return 0
  return 1
}
final_py_children() {   # count of PYTHON processes whose command line names the 48,852 tag.  The tag is assembled here so THIS command line never contains it.
  powershell -NoProfile -Command "\$t='488'+'52'; \$n=@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.ProcessId -ne \$PID -and \$_.CommandLine -and \$_.CommandLine.Contains(\$t) -and \$_.CommandLine -notmatch 'navfeed' }).Count; Write-Output ('ZZPYC'+\$n+'ZZ')" 2>/dev/null | tr -d '\r' | sed -n 's/.*ZZPYC\([0-9]*\)ZZ.*/\1/p' | head -n 1
}
chain_marker_present() { [ -e /c/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active ]; }
conditions_ok() {   # echoes nothing; returns 0 iff (ii) disk and (iii) the final chain is not active
  _fm=$(free_mb)
  [ "$_fm" -ge "$MIN_FREE_MB" ] 2>/dev/null || { mark NODISK "free ${_fm} MB < $MIN_FREE_MB MB on the output volume"; return 1; }
  write_test || { mark NOWRITE "the 16 MB write test failed on the output volume"; return 1; }
  if bfinal_pending; then mark CHAINPENDING "the final (48,852) chain is pending/active (yield rule)"; return 1; fi
  _pc=$(final_py_children); [ "${_pc:-x}" = "0" ] || { mark CHAINPY "python processes of the final chain: '${_pc}'"; return 1; }
  if chain_marker_present; then mark CHAINMARKER "a chain-active marker exists"; return 1; fi
  return 0
}
