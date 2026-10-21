#!/bin/sh
# pull_and_analyze_pk.sh -- PREREG R8-PICKER (sha256 e6ec30e7...): AFTER the Thor waiter wrote ZZPKWDONEZZ.
#  1. pull R0 / R1 (fan_extras.npz, refcv6_extras.npz, fan_roll_s*.json, roll_s*.json, MD5_R*.txt, pkroll.log) and md5-verify every file against Thor's list (K11);
#  2. build the three picker trees (build_picker_trees.py build) -> BUILD_PICKER_TREES.json + archive/PICKS_s*.npz;
#  3. the SHIP-R8 lane on each tree (pk_ship_lane.sh, sequential, CPU);
#  4. analyze_picker_r8.py full -> PICKER_R8.json + RESULT.md (mechanical, from the prereg).
# Idempotent: a step whose artifact exists is skipped (delete it to redo).  Markers ZZPKP<NAME>ZZ in logs/pull_analyze.log.
set -u
P=C:/Users/Admin/refcv8_pf/picker; T=/home/nvidia/r8ns/picker/out
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$P/logs/pull_analyze.log; mkdir -p $P/logs
mark() { echo "ZZPKP$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
mark START "pid $$"
ssh -n -o ConnectTimeout=20 tanitad-thor "grep -o 'ZZPKWDONEZZ.*' $T/waiter.log" > $P/logs/thor_done.txt 2>/dev/null
grep -q 'R0=0 R1=0' $P/logs/thor_done.txt || { mark NOTDONE "$(cat $P/logs/thor_done.txt)"; exit 2; }
for s in 0 1; do
  D=$P/runs/R$s; mkdir -p $D/dump
  if [ ! -s $D/MD5_R$s.txt ]; then
    scp -q tanitad-thor:$T/R$s/dump/fan_extras.npz tanitad-thor:$T/R$s/dump/refcv6_extras.npz $D/dump/ && \
    scp -q tanitad-thor:$T/R$s/fan_roll_s$s.json tanitad-thor:$T/R$s/roll_s$s.json tanitad-thor:$T/R$s/pkroll.log tanitad-thor:$T/R$s/MD5_R$s.txt $D/ \
      || { mark PULLFAIL "R$s"; exit 3; }
  fi
  ( cd $D && md5sum -c MD5_R$s.txt > md5_check.txt 2>&1 ) && mark "MD5OK_R$s" "$(wc -l < $D/MD5_R$s.txt) files" || { mark MD5BAD "R$s $(cat $D/md5_check.txt | tr '\n' ' ')"; exit 4; }
done
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
if [ ! -s $P/BUILD_PICKER_TREES.json ]; then
  "$PY" -X utf8 $P/code/build_picker_trees.py build --fan s0=$P/runs/R0/dump/fan_extras.npz --fan s1=$P/runs/R1/dump/fan_extras.npz \
      --out-json $P/BUILD_PICKER_TREES.json > $P/logs/build.log 2>&1
  [ -s $P/BUILD_PICKER_TREES.json ] && mark BUILT "" || { mark BUILDFAIL "see logs/build.log"; exit 5; }
fi
# the three lanes run SEQUENTIALLY: Master Mind constraint 2026-10-21 21:45 (PI decision) -- while the C2 run is live
# (C:/Users/Admin/par18/c2run/logs/c2_supervise.log has no ZZSUP_DONE) every dev-box process stays < 4 GB and C2's GPU stages need 8 GB host RAM free;
# one lane at a time keeps this pipeline's footprint to a single <=~2.5 GB process.  No dev-box GPU anywhere (CUDA_VISIBLE_DEVICES=-1).
for tree in pkship_pf pkbase_pf pkmbr32_pf; do
  if [ ! -s $P/lane/$tree/ship_r8/SHIP_R8_$tree.json ]; then
    sh $P/code/pk_ship_lane.sh $tree > $P/logs/lane_$tree.out 2>&1
  fi
  mark "LANE_$tree" "$(grep -oE 'ZZPKL(VERDICT_[A-Z]+|FAIL_[a-z_0-9]+|NOTREE|PAIRFAIL|INSTALLBAD|VERDICTMISSING)ZZ' $P/lane/$tree/lane.log | tail -n 1)"
done
"$PY" -X utf8 $P/code/analyze_picker_r8.py full > $P/logs/analyze.log 2>&1
[ -s $P/PICKER_R8.json ] && [ -s $P/RESULT.md ] && mark ANALYZED "$(grep -o 'ZZPKANALYSISDONEZZ.*' $P/logs/analyze.log | cut -c1-300)" || { mark ANALYZEFAIL "see logs/analyze.log"; exit 6; }
mark END ""
