#!/bin/bash
# MM's independent acceptance re-run of the P6-read layout fix: E: tree == the staged bytes; NO export / root override (the resolver itself is exercised)
S=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/c216c849-5fe7-4d43-a3be-32429241f811/scratchpad/p6read_pre
PKG="E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset REFE_NAVTEST_EXPORT TANITAD_NAVSIM_ROOT P6_COMMANDS_JSON REFE_P2_TABLE
i=0
for t in "test_p6_read_gaps.py" "selftest_stage3_full.py" "test_command_adherence.py" "stage3_full.py --selfcheck"; do
  i=$((i + 1)); echo "== $(date -u +%T) $t"
  ( cd "$PKG/eval" && env -u MSYS_NO_PATHCONV $PY $t ) > "$S/v2_$i.log" 2>&1 < /dev/null
  echo "rc=$? $(tail -n 2 "$S/v2_$i.log" | tr '\n' ' ' | cut -c1-250)"
done; echo "== $(date -u +%T) done"
