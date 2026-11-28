#!/bin/bash
# pull6.sh -- AMEND_HWMPLAN_6 step 1: the streamed pull + level-1 latents of the 859 new TRAIN clips. Dev-box CPU; box 2 is READ through sftp (no process there).
# Launched detached by launch6.ps1 -Job pull6.cmd.  Last line of amend6/pull.log:  ZZ PULL6 DONE ZZ | ZZ PULL6 FAIL ZZ <why>
# Resumable: rerun the same launcher after any stop; finished chunks (sha256 in lat6/CHUNKS.json) are kept.
export PATH="/usr/bin:/bin:$PATH"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_tree
export PYTHONPATH="F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval;E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\wmact;F:\TanitAD-scratch\hwm_plan"
ROOT=F:/TanitAD-scratch/actrsp/lonplan/amend6
B=$ROOT/build
export TMP=$ROOT/work/tmp TEMP=$ROOT/work/tmp
mkdir -p "$TMP" "$ROOT/lat6"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$ROOT/pull.log
fail() { echo "ZZ PULL6 FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
want=$(cut -c1-64 "$ROOT/AMEND_HWMPLAN_6.sha256")
got=$(sha256sum "$ROOT/AMEND_HWMPLAN_6.md" | cut -c1-64)
[ "$got" = "$want" ] || fail "AMEND_HWMPLAN_6.md sha256 $got is not the frozen $want"
$PY -u F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
cd "$B" || fail "no build folder"
echo "$(date -u +%FT%TZ) pull start" >> "$LOG"
# up to 5 passes: sftp / ssh drops are survived by the resume (a pass that ends rc 0 with all clips banked is the last)
for i in 1 2 3 4 5; do
  $PY -u pull_latents6.py run >> "$LOG" 2>&1 && break
  echo "$(date -u +%T) pass $i ended rc $?; resuming" >> "$LOG"
  sleep 60
done
n=$($PY -c "import json;print(sum(len(c['clips']) for c in json.load(open('$ROOT/lat6/CHUNKS.json'))['chunks']))" 2>> "$LOG")
[ "$n" = "859" ] || fail "only $n of 859 new clips banked (resume with the same launcher; checkpoints 141 / 423 are usable by fit_ra6.py)"
echo "ZZ PULL6 DONE ZZ" | tee -a "$LOG"
