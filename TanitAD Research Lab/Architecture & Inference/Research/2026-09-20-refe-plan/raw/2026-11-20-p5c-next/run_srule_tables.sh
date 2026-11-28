#!/bin/bash
# MM 2026-11-20: S-RULE (PREREG_P5C_NEXT.md sha256 ac0e03f9..., frozen 06:26Z before any full table) - three per-proposal tables,
# one arm at a time, 2 shards each (<= 2 scoring processes), then each arm's table read and the S-RULE read.
# The scorer gates on commit free >= 5.5 GB itself (start + every 200 tokens). Resumable. Prints ZZSRULE_* lines.
cd /e/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next || exit 1
NAV="F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
REFE="F:/TanitAD_p5cread_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"
PYN=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
PYT=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
SEAMS=E:/Projects/TanitAD/data/refe_navtest/seams
declare -A NPZ=( [C1]=$SEAMS/stage2_p5c_c1.proposals.npz [C0]=$SEAMS/stage2_p5c_c0.proposals.npz [base]=$SEAMS/stage2_p7_full.proposals.npz )
for arm in C1 C0 base; do
  t0=$(date +%s)
  pids=()
  for s in 0 1; do
    PYTHONPATH="$NAV;$REFE" TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 "$PYN" p5c_prop_score.py --arm $arm --proposals "${NPZ[$arm]}" --out propscore/full_${arm}_s${s}of2.jsonl \
      --shard $s/2 > propscore/full_${arm}_s${s}of2.log 2>&1 < /dev/null &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=$?; done
  n=$(cat propscore/full_${arm}_s0of2.jsonl propscore/full_${arm}_s1of2.jsonl 2>/dev/null | wc -l)
  echo "ZZSRULE_TABLE $arm rc=$rc rows=$n $(( ($(date +%s) - t0) / 60 )) min $(date -u +%FT%H:%MZ)"
  tail -n 1 propscore/full_${arm}_s0of2.log | cut -c1-200; tail -n 1 propscore/full_${arm}_s1of2.log | cut -c1-200
  [ "$rc" = 0 ] || { echo "ZZSRULE_FAIL table $arm"; exit 0; }
done
echo "ZZSRULE_TABLES_DONE $(date -u +%FT%H:%MZ)"
