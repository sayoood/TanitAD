#!/bin/bash
# P6 layout fix acceptance runner (RUNBOOK_P6_READ step 0.2-0.3 on today's layout). NO env overrides for the archive / artifacts / commands / P2 table:
# the root resolvers must find them themselves. `python -X utf8 -I` (isolated: no PYTHON* env, no cwd on sys.path). CPU only.
ART=E:/Projects/TanitAD-artifacts/p6read_layout_fix
PKG="E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
unset MSYS_NO_PATHCONV P6_COMMANDS_JSON REFE_P2_TABLE REFE_NAVTEST_EXPORT TANITAD_NAVSIM_ROOT TANITAD_ARTIFACTS_ROOT
tag=${1:-after}
cd "$PKG/eval"
run() { n="$1"; shift; ( $PY -X utf8 -I "$@" ) > "$ART/logs/${tag}_${n}.log" 2>&1 < /dev/null; echo "$(date -u +%T) $n rc=$? $(tail -n 1 "$ART/logs/${tag}_${n}.log" | cut -c1-200)" >> "$ART/logs/${tag}_status.txt"; }
: > "$ART/logs/${tag}_status.txt"
$PY -X utf8 -I harness_pins.py --verify > "$ART/logs/${tag}_harness_pins_verify.json" 2>&1; echo "$(date -u +%T) harness_pins rc=$?" >> "$ART/logs/${tag}_status.txt"
run test_p6_read_gaps test_p6_read_gaps.py &
run test_command_adherence test_command_adherence.py &
( run selftest_stage3_full selftest_stage3_full.py; run stage3_selfcheck stage3_full.py --selfcheck ) &
wait
echo "$(date -u +%T) ALL DONE" >> "$ART/logs/${tag}_status.txt"
