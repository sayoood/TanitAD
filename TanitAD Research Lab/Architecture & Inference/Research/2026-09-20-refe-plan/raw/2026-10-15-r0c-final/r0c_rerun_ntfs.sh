#!/bin/bash
# Master Mind 2026-10-15: re-run the five suites that failed with TMP on exFAT E: (git dubious-ownership, no exec bits, no symlinks), now with TMP on NTFS C:.
set -u
NSP=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/c216c849-5fe7-4d43-a3be-32429241f811/scratchpad
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
V=C:/Users/Admin/AppData/Local/Temp/r0f
P11=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
OUT=$NSP/r0cf; T=C:/Users/Admin/AppData/Local/Temp/t0c; rm -rf "$T"; mkdir -p "$T"
cd "$V/$PK/refe" || exit 1
run() { local py=$1 tag=$2; shift 2; local T0=$(date +%s); local nm=$(echo "$* $tag ntfs" | tr ' /.+' '____');
  PYTHONIOENCODING=utf-8 TMP=$T TEMP=$T "$py" -u "$@" > "$OUT/$nm.log" 2>&1; local rc=$?
  echo "$* [$tag, TMP=C:] rc $rc $(( $(date +%s) - T0 ))s $(grep -o -E 'ZZ[A-Z0-9_]*(PASS|SKIP)[A-Z_]* ?[0-9]*' "$OUT/$nm.log" | tail -1)"; }
run $P11 t2.11 test_launch_gate.py --static
run $P11 t2.11 test_p2rep_chain.py
run $P11 t2.11 test_s1aux_chain.py
run $P11 t2.11 test_p6_chain.py
run $P11 t2.11 test_rls1_queue.py
rm -rf "$T"
echo ZZR0C_NTFS_DONE
