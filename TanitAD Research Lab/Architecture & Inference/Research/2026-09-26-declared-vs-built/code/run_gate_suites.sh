#!/usr/bin/env bash
# Batch-1 test evidence on trees built by `git archive 2c510fb` (the current tip), memory-bounded:
# ONE pytest process per file, junit per file, the Master Mind's RAM floor (run_suite_bounded.py).
#  0. TAU    : the nav-compliance tau on the FULL train split (PI ruling E1: all three ON)
#  1. FIXNEW : the 6 new test files on tip + batch 1 (an early failure surfaces first)
#  2. TIPRED : the same files on tip + ONLY the two new modules and the tests (no fixes)
#  3. FIX    : the related suites on tip + batch 1
#  4. TIP    : the same suites on the tip
PK="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-declared-vs-built"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
R="$PK/code/run_suite_bounded.py"
O="$PK/raw/gate"
FIX=C:/Users/Admin/dvb_fix_0926g
TIP=C:/Users/Admin/dvb_tip_0926d
RED=C:/Users/Admin/dvb_tipred_0926c
cd "$PK/code"
echo "[gate] $(date '+%F %T') trees FIX=$FIX TIP=$TIP TIPRED=$RED base=$(cat "$PK/raw/gate_tree_tip_commit.txt")"
echo "[gate] $("$PY" -c 'import run_suite_bounded as r; print(r.FLOOR_NOTE)')"
OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" "$PY" "$PK/code/gated_cmd.py" "$O/nav_compliance_tau_train.log" -- \
  "$PY" "$PK/code/derive_navc_tau.py" $FIX \
  "C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/7c618562-39eb-4a61-af81-755048086a13/scratchpad/thor_pull/refcv6-b1-416x1024-train___v2manifest.pt" \
  "D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz" "$PK/raw/nav_compliance_tau_train.json"
echo "[gate] $(date '+%F %T') TAU step done"
"$PY" "$R" $FIX "$PK/raw/new_stack_files.txt" "$O/FIXNEW_stack.jsonl" --rundir stack
"$PY" "$R" $FIX "$PK/raw/tipred_taniteval_files.txt" "$O/FIXNEW_taniteval.jsonl" --rundir taniteval
"$PY" "$R" $RED "$PK/raw/new_stack_files.txt" "$O/TIPRED_stack.jsonl" --rundir stack
"$PY" "$R" $RED "$PK/raw/tipred_taniteval_files.txt" "$O/TIPRED_taniteval.jsonl" --rundir taniteval
"$PY" "$R" $FIX "$PK/raw/related_stack_files_FIX.txt" "$O/FIX_stack.jsonl" --rundir stack
"$PY" "$R" $FIX "$PK/raw/related_taniteval_files_FIX.txt" "$O/FIX_taniteval.jsonl" --rundir taniteval
"$PY" "$R" $TIP "$PK/raw/related_stack_files_TIP.txt" "$O/TIP_stack.jsonl" --rundir stack
"$PY" "$R" $TIP "$PK/raw/related_taniteval_files_TIP.txt" "$O/TIP_taniteval.jsonl" --rundir taniteval
echo "ZZGATESUITESDONEZZ $(date '+%F %T')"
