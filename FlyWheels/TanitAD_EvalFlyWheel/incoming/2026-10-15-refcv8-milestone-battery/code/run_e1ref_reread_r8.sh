#!/bin/bash
# run_e1ref_reread_r8.sh -- D-R8-E1-REF: re-read banked refcv8 trees under the ha0_ext_pose reference (CPU only, memory-gated, ONE tree at a time).
#   usage: sh run_e1ref_reread_r8.sh <tree> [<tree> ...]            e.g. step30000ema_unc step30000_unc ...
#   NEW file (2026-10-19).  Does not touch chain_40k_r8.sh, which calls the same `compliant_reread_r8.py read --tree T --force` CLI.
#   Markers (opaque, disjoint from anything this script greps for):  ZZE1REFSTARTZZ  ZZE1REFTREE<ok|bad> <tree>ZZ  ZZE1REFENDZZ  in raw/compliant_reread/logs/e1ref_series.log
#   The ARTIFACT is the evidence: a tree counts as done only if its JSON says controls.ALL_PASS true AND carries the pose-scored BAR-R8-E1 reference field.
PKG=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
LOGD=$PKG/raw/compliant_reread/logs
SER=$LOGD/e1ref_series.log
NEED_GB=${E1REF_NEED_GB:-5}
mkdir -p "$LOGD"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8
export PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKG/code"
free_gb() { v=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)" 2>/dev/null | tr -d '\r ' | tr ',' '.'); case "$v" in ''|*[!0-9.]*) echo 0;; *) echo "$v";; esac; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit (a+0>=b+0)?0:1}'; }
gate() {   # two checks 20 s apart (the dev box COMMIT limit binds, not RAM)
  while :; do
    h1=$(free_gb)
    if ge "$h1" "$NEED_GB"; then sleep 20; h2=$(free_gb); ge "$h2" "$NEED_GB" && return 0; fi
    echo "waiting: free commit ${h1} GB < ${NEED_GB} ($(date +%T))" >> "$1"
    sleep 30
  done
}
echo "ZZE1REFSTARTZZ $(date +%FT%T) trees: $*" >> "$SER"
for T in "$@"; do
  L=$LOGD/e1ref_$T.log
  gate "$L"
  echo "start $(date +%T) free commit $(free_gb) GB" >> "$L"
  ( cd "$PKG/code" && "$PY" -X utf8 compliant_reread_r8.py read --tree "$T" --force ) >> "$L" 2>&1
  ok=$("$PY" -I -c "
import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8'))
    print('yes' if j['controls']['ALL_PASS'] is True and j['bars']['compliant']['capped']['BAR-R8-E1'].get('reference','').startswith('ha0_ext_pose') else 'no')
except Exception:
    print('no')" "$PKG/raw/compliant_reread/trees/$T.json")
  if [ "$ok" = "yes" ]; then echo "ZZE1REFTREEok $T $(date +%T)ZZ" >> "$SER"; else echo "ZZE1REFTREEbad $T $(date +%T)ZZ (see $L)" >> "$SER"; fi
done
echo "ZZE1REFENDZZ $(date +%FT%T)" >> "$SER"
