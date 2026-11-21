#!/bin/bash
# run_e1ref_all_r8.sh -- D-R8-E1-REF: the WHOLE re-read / mutation / strict / assemble / series / pytest sequence, ONE heavy job at a time, memory-gated, CPU only.
#   usage: sh run_e1ref_all_r8.sh            (detached: nohup sh code/run_e1ref_all_r8.sh > raw/compliant_reread/logs/e1ref_all.out 2>&1 &)
#   NEW file (2026-10-19).  Does not touch chain_40k_r8.sh (pid 1152752), which calls the same `compliant_reread_r8.py read --tree T --force` CLI at ~22:30.
#   Opaque markers in raw/compliant_reread/logs/e1ref_series.log:  ZZE1REFTREEok|bad <tree>ZZ  ZZE1REFSTAGE <name> <ok|bad>ZZ  ZZE1REFALLDONEZZ
#   The ARTIFACT is the evidence (never an exit code): each stage's marker is decided from the file it should have written.
PKG=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
LOGD=$PKG/raw/compliant_reread/logs
SER=$LOGD/e1ref_series.log
OUT=$PKG/raw/compliant_reread
NEED_GB=${E1REF_NEED_GB:-5}
mkdir -p "$LOGD"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8
export PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKG/code"
free_gb() { v=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)" 2>/dev/null | tr -d '\r ' | tr ',' '.'); case "$v" in ''|*[!0-9.]*) echo 0;; *) echo "$v";; esac; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit (a+0>=b+0)?0:1}'; }
gate() {   # $1 = logfile ; two checks 20 s apart (the dev box COMMIT limit binds, not RAM)
  while :; do
    h1=$(free_gb)
    if ge "$h1" "$NEED_GB"; then sleep 20; h2=$(free_gb); ge "$h2" "$NEED_GB" && return 0; fi
    echo "waiting: free commit ${h1} GB < ${NEED_GB} ($(date +%T))" >> "$1"
    sleep 30
  done
}
stage_mark() { echo "ZZE1REFSTAGE $1 $2 $(date +%T)ZZ" >> "$SER"; }
cd "$PKG/code" || exit 9
echo "ZZE1REFALLSTARTZZ $(date +%FT%T)" >> "$SER"

# ---- 1. re-read every banked tree under the pose reference (EMA first; the primary tree first of all)
bash run_e1ref_reread_r8.sh step30000ema_unc step30000_unc step20000ema_unc step20000_unc step15000ema_unc step15000_unc step10500ema_unc step10500_unc step5000ema_unc step5000_unc

# ---- 2. the DELIBERATE-REGRESSION arm (control c): pose prior swapped back to ha0_ext; writes mutation/step30000ema_unc.json, never trees/
L=$LOGD/e1ref_mutation.log; rm -f "$OUT/mutation/step30000ema_unc.json"
gate "$L"; "$PY" -X utf8 compliant_reread_r8.py read --tree step30000ema_unc --mutate-pose-to-ha0-ext --force >> "$L" 2>&1
if [ -s "$OUT/mutation/step30000ema_unc.json" ]; then stage_mark mutation ok; else stage_mark mutation bad; fi

# ---- 3. strict-cap sensitivity for the primary tree, now with the pose reference beside the echo
L=$LOGD/e1ref_strict.log
gate "$L"; "$PY" -X utf8 compliant_reread_r8.py strict --tree step30000ema_unc >> "$L" 2>&1
if [ -s "$OUT/STRICT_CAP_step30000ema_unc.json" ]; then stage_mark strict ok; else stage_mark strict bad; fi

# ---- 4. assemble + series (reads the per-tree JSONs only)
"$PY" -X utf8 compliant_reread_r8.py assemble >> "$LOGD/e1ref_assemble.log" 2>&1
"$PY" -X utf8 compliant_reread_r8.py series >> "$LOGD/e1ref_assemble.log" 2>&1
if [ -s "$OUT/E1REF_SERIES.md" ] && [ -s "$OUT/COMPLIANT_REREAD.md" ]; then stage_mark assemble ok; else stage_mark assemble bad; fi

# ---- 5. pytest of every file touched (+ the chain test the brief names); logs hold the counts
for T in test_compliant_reread_e1ref_r8 test_chain_40k_r8 test_kdraw_e1_r8 test_a0k0_headroom_r8; do
  L=$LOGD/pytest_$T.log
  gate "$L"
  timeout 1500 "$PY" -X utf8 -m pytest -q -p no:cacheprovider "$T.py" > "$L" 2>&1
  last=$(tail -n 1 "$L" | tr -d '\r')
  echo "ZZE1REFPYTEST $T :: $last ZZ" >> "$SER"
done
echo "ZZE1REFALLDONEZZ $(date +%FT%T)" >> "$SER"
