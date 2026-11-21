#!/bin/bash
# TREEDRIFT chain v3: takes the dev-box GPU lock (gpu_lock.py, the binding protocol) ONCE for the whole ~25 min batch, then rolls everything.
# Success is judged on META.json, never on an exit code.
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONIOENCODING=utf-8 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TMP=F:/c_overflow_2026-10-26/tmp TEMP=F:/c_overflow_2026-10-26/tmp
BAT=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code
LOCKPY=F:/treedrift/fb47a58/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
D=F:/treedrift; CH=$D/logs/chain6.log; JOB=treedrift-bisect
mk() { echo "$(date +%T) $*" >> $CH; }
roll() {  # name tree seed [extra args...]
  local name=$1 tree=$2 seed=$3; shift 3
  [ -s $D/out/$name/META.json ] && { mk "SKIP $name (META exists)"; return 0; }
  while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' 
'); [ -n "$u" ] && [ "$u" -lt 4300 ] 2>/dev/null && break; mk "WAIT gpu used=${u}MiB (another python process holds the card)"; sleep 30; done
  mk "START $name tree=$tree seed=$seed $*"
  rm -rf $D/out/$name
  R8_LAUNCH_TREE=$D/$tree PYTHONPATH="$D/$tree/stack;$D/$tree/taniteval;$BAT" \
    $PY -X utf8 $D/roll_tree.py --tree $D/$tree --seed $seed --out $D/out/$name --tag $name "$@" > $D/logs/$name.log 2>&1
  if [ -s $D/out/$name/META.json ]; then mk "DONE $name: $(grep -o 'mean_ade_2s=[0-9.]*' $D/logs/$name.log | tail -1)"; else mk "FAIL $name (no META.json)"; fi
}
mk "CHAIN6 START: MM order = LONHEAD -> R9-PF K-OFF/K-EQ -> treedrift; waiting for R9PF_KEQ.json (or its ZZKEQDONEZZ marker), bound 16:45"
mk "R9-PF KEQ DONE; requesting the gpu lock"
rm -f $D/logs/lock6.json
$PY $LOCKPY acquire --job $JOB --pid $$ --max-wait-s 43200 --out $D/logs/lock6.json >> $D/logs/lock6.log 2>&1
$PY -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" $D/logs/lock6.json || { mk "NOLOCK"; exit 1; }
trap '$PY $LOCKPY release --job $JOB --pid $$ >> $D/logs/lock6.log 2>&1; mk "LOCK RELEASED"' EXIT
mk "LOCKED"
roll fb_s0   fb47a58 0 --rng-capture $D/out/rng_fb_s0.npz
roll tip_s0  4b5f7fb 0 --rng-capture $D/out/rng_tip_s0.npz
roll pre_s0  f6f96d6 0 --rng-capture $D/out/rng_pre_s0.npz
roll c3766_s0 3766dcd 0 --rng-capture $D/out/rng_3766_s0.npz
roll tip_inj_fb 4b5f7fb 0 --rng-inject $D/out/rng_fb_s0.npz
roll fb_inj_tip fb47a58 0 --rng-inject $D/out/rng_tip_s0.npz
roll fb_s1  fb47a58 1
roll tip_s1 4b5f7fb 1
roll fbfix_s0  fb47a58_fix 0
roll cde4_s0 de4e15b 0
roll c947_s0 947b05f 0
roll c3766_inj_fb 3766dcd 0 --rng-inject $D/out/rng_fb_s0.npz
roll pre_inj_fb f6f96d6 0 --rng-inject $D/out/rng_fb_s0.npz
roll c460_s0 460508e 0 --rng-capture $D/out/rng_460_s0.npz
roll c702_s0 70211d6 0
roll fb_s0_rep fb47a58 0
roll tip_s0_rep 4b5f7fb 0
roll tipfix_s0 4b5f7fb_fix 0
mk "CHAIN6 END"
