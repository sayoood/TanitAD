#!/usr/bin/env bash
# RA-1a -- the exact commands for the pooled frozen-DINOv3 cache on the dev-box RTX 4060. Git Bash:
#     bash run_encode.sh [A|B|C|D|all]          (default: all)  -- each stage is RESUMABLE; rerun after any interruption
#
#   A  600 TRAIN clips   grids 4x10 + 8x20     the first arm set of PREREG_HJ_STAGE0 H-HJ-SENS-1 .......... ~47 min   12.5 GB
#   B  141 EVAL clips    grids 4x10 + 8x20     local native 256x640 v2ep (NO Thor read) ................... ~11 min    2.9 GB
#   C  all 4,572 TRAIN   grids 4x10 only       the 600 of A are skipped (their grids are a superset) ...... ~5.2 h    17.3 GB
#   D  (optional, MATCHED_EVAL=1) 141 EVAL via the SAME 416->256 resample route as TRAIN, grids 4x10 ...... ~11 min    0.6 GB
#
# Time = clips x 4.72 s, the banked rate on this 4060 (trunk-probe raw/train/pull_encode.log: 600 clips in 2,834 s, serial
# scp + encode + 66 MB save). This pipeline prefetches and saves 4-21 MB, so it should be no slower; NOT re-measured on GPU.
# Disk: 4x10-only 4.36 MB/clip, 4x10+8x20 20.9 MB/clip (MEASURED on a real output file). A+B+C = 32.8 GB on C:.
# 8x20 for all 4,572 would be 95.6 GB and does NOT fit on C: (63 GB free 2026-10-17) -- hence 8x20 on A and B only.
#
# GPU etiquette (memory devbox-drive-letter-and-detached-jobs): the 4060 is shared via
# C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock. This script WAITS for it (gpu_lock.py: stale locks are renamed aside,
# never deleted; a live holder is waited on) and then also for nvidia-smi memory.used <= GPU_MEM_MAX_MIB for 2 polls.
# Launch DETACHED (a Claude session's children die with it) -- see ../HANDOFF.md for the Invoke-CimMethod line.
set -u
STAGES="${1:-all}"
PY="${PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
CODE="$(cd "$(dirname "$0")" && pwd -W)"
OUT="${OUT:-C:/Users/Admin/refav1_hj_cache}"
GPU_MEM_MAX_MIB="${GPU_MEM_MAX_MIB:-2500}"
LOCK_MAX_WAIT_S="${LOCK_MAX_WAIT_S:-28800}"
export PYTHONIOENCODING=utf-8 PYTHONPATH="E:/Projects/TanitAD/stack;E:/Projects/TanitAD/taniteval" OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
LOGS="$OUT/logs"; mkdir -p "$LOGS"
stamp() { date '+%F %T'; }
say() { echo "[run_encode $(stamp)] $*"; }

if [ "${DRY_RUN:-0}" = 1 ]; then      # preflight: clip-set assertions + parity gate + SOURCE resolution (1 tiny ssh) + resume plan. No GPU, no lock.
  say "DRY_RUN: preflight only (no lock, no model, nothing encoded)"
  "$PY" "$CODE/gpu_lock.py" status
  "$PY" -u "$CODE/pull_encode_pooled.py" --dry-run --split train --n 600 --grids 4x10,8x20 --out "$OUT" 2>&1 | grep -v "Loading weights" | tail -8
  "$PY" -u "$CODE/pull_encode_pooled.py" --dry-run --split eval --grids 4x10,8x20 --out "$OUT" 2>&1 | grep -v "Loading weights" | tail -5
  "$PY" -u "$CODE/pull_encode_pooled.py" --dry-run --split train --n 0 --grids 4x10 --out "$OUT" 2>&1 | grep -v "Loading weights" | tail -3
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>&1 | head -2
  df -h "$OUT" 2>/dev/null | tail -1
  say "ZZDRY-RUN-DONEZZ"
  exit 0
fi

WINPID="$(cat /proc/$$/winpid 2>/dev/null || echo $$)"
LOCKARG=(); [ -n "${GPU_LOCK:-}" ] && LOCKARG=(--lock "$GPU_LOCK")          # tests only; default = the real shared lock
release() { "$PY" "$CODE/gpu_lock.py" release --pid "$WINPID" "${LOCKARG[@]}" >/dev/null 2>&1; }
trap release EXIT

say "stages=$STAGES out=$OUT winpid=$WINPID"
"$PY" "$CODE/gpu_lock.py" status "${LOCKARG[@]}"
"$PY" "$CODE/gpu_lock.py" acquire --pid "$WINPID" --owner ra1a --job refav1-hj-pooled-encode --max-wait-s "$LOCK_MAX_WAIT_S" "${LOCKARG[@]}" || { say "ZZLOCK-TIMEOUTZZ"; exit 5; }
idle=0
while [ "$idle" -lt 2 ]; do
  g="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r')"
  case "$g" in ''|*[!0-9]*) g=99999 ;; esac
  if [ "$g" -le "$GPU_MEM_MAX_MIB" ]; then idle=$((idle + 1)); else idle=0; say "GPU memory.used=${g} MiB > ${GPU_MEM_MAX_MIB}: another CUDA job is on the card"; fi
  [ "$idle" -lt 2 ] && sleep 60
done
say "GPU idle (memory.used <= ${GPU_MEM_MAX_MIB} MiB twice) and the lock is ours"

# encode <log> <args...>: rc 3 = some clips skipped (torn transfer) -> rerun, only the missing ones are redone. rc 4/7/other = stop.
encode() {
  local log="$1"; shift
  local try rc=1
  for try in 1 2 3; do
    "$PY" -u "$CODE/pull_encode_pooled.py" --device cuda --dtype bf16 "$@" >> "$log" 2>&1; rc=$?
    [ "$rc" -eq 0 ] && return 0
    [ "$rc" -ne 3 ] && break
    say "rc=3 (skipped clips) -> rerun $try"
  done
  say "ENCODE FAILED rc=$rc (see $log)"; return "$rc"
}
verify() { local log="$1"; shift; "$PY" -u "$CODE/verify_cache.py" "$@" >> "$log" 2>&1; local rc=$?; tail -1 "$log"; return "$rc"; }
want() { [ "$STAGES" = all ] || [[ "$STAGES" == *"$1"* ]]; }

if want A; then
  say "STAGE A: 600 train, 4x10+8x20"
  encode "$LOGS/A_train600.log" --split train --n 600 --grids 4x10,8x20 --out "$OUT" || exit 10
  verify "$LOGS/A_verify.log" --split train --n 600 --grids 4x10,8x20 --sample 100 --redo 5 --device cuda --dtype bf16 --out "$OUT" || { say "ZZVERIFY-A-FAILZZ"; exit 11; }
fi
if want B; then
  say "STAGE B: 141 eval, 4x10+8x20 (local native v2ep)"
  encode "$LOGS/B_eval141.log" --split eval --grids 4x10,8x20 --out "$OUT" || exit 20
  verify "$LOGS/B_verify.log" --split eval --grids 4x10,8x20 --sample 0 --redo 5 --device cuda --dtype bf16 --out "$OUT" || { say "ZZVERIFY-B-FAILZZ"; exit 21; }
fi
if want C; then
  say "STAGE C: all 4,572 train, 4x10 (the 600 of A are skipped)"
  encode "$LOGS/C_train4572.log" --split train --n 0 --grids 4x10 --out "$OUT" || exit 30
  verify "$LOGS/C_verify.log" --split train --n 0 --grids 4x10 --sample 200 --out "$OUT" || { say "ZZVERIFY-C-FAILZZ"; exit 31; }
fi
if [ "${MATCHED_EVAL:-0}" = 1 ] && want D; then
  say "STAGE D: matched-preprocessing eval (416->256 resample route), 4x10"
  encode "$LOGS/D_eval141_m416.log" --split eval --tag _m416 --source thor416 --grids 4x10 --out "$OUT" || exit 40
  verify "$LOGS/D_verify.log" --split eval --tag _m416 --grids 4x10 --sample 0 --out "$OUT" || true
fi
say "ZZRUN-ENCODE-DONEZZ"
