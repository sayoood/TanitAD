#!/bin/bash
# chain7: the bank-position injection (CUDA state = (seed 0, offset 4 x bank row)) on fb47a58 and tip, plus a +1-position control.  Waits for chain6, then takes the gpu lock.
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TMP=F:/c_overflow_2026-10-26/tmp TEMP=F:/c_overflow_2026-10-26/tmp
BAT=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code
LOCKPY=F:/treedrift/fb47a58/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
D=F:/treedrift; CH=$D/logs/chain7.log; JOB=treedrift-bankpos
mk() { echo "$(date +%T) $*" >> $CH; }
roll() {
  local name=$1 tree=$2 seed=$3; shift 3
  [ -s $D/out/$name/META.json ] && { mk "SKIP $name"; return 0; }
  while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ' | tr -d '\r'); [ -n "$u" ] && [ "$u" -lt 4300 ] 2>/dev/null && break; mk "WAIT gpu used=${u}MiB"; sleep 20; done
  mk "START $name tree=$tree seed=$seed $*"; rm -rf $D/out/$name
  R8_LAUNCH_TREE=$D/$tree PYTHONPATH="$D/$tree/stack;$D/$tree/taniteval;$BAT" $PY -X utf8 $D/roll_tree.py --tree $D/$tree --seed $seed --out $D/out/$name --tag $name "$@" > $D/logs/$name.log 2>&1
  if [ -s $D/out/$name/META.json ]; then mk "DONE $name: $(grep -o 'mean_ade_2s=[0-9.]*' $D/logs/$name.log | tail -1)"; else mk "FAIL $name (no META.json)"; fi
}
mk "CHAIN7 START (waiting for CHAIN6 END)"
until grep -q "CHAIN6 END" $D/logs/chain6.log 2>/dev/null; do [ "$(date +%H%M)" -ge 1710 ] && { mk "GAVEUP"; exit 1; }; sleep 15; done
rm -f $D/logs/lock7.json
$PY $LOCKPY acquire --job $JOB --pid $$ --max-wait-s 3600 --out $D/logs/lock7.json >> $D/logs/lock7.log 2>&1
$PY -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" $D/logs/lock7.json || { mk "NOLOCK"; exit 1; }
trap '$PY $LOCKPY release --job $JOB --pid $$ >> $D/logs/lock7.log 2>&1; mk "LOCK RELEASED"' EXIT
mk "LOCKED"
roll bankpos_fb  fb47a58 0 --rng-inject $D/out/rng_bankpos.npz --inject-cuda-only
roll bankpos_tip 4b5f7fb 0 --rng-inject $D/out/rng_bankpos.npz --inject-cuda-only
roll bankpos1_fb fb47a58 0 --rng-inject $D/out/rng_bankpos_plus1.npz --inject-cuda-only
roll bankpos_c3766 3766dcd 0 --rng-inject $D/out/rng_bankpos.npz --inject-cuda-only
mk "CHAIN7 END"
