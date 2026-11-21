#!/bin/bash
# smoke_l7cap_cpu.sh -- P4: CPU smoke of the L7 train-split fan capture on the dev box. ZERO GPU (CUDA_VISIBLE_DEVICES=-1), OMP<=4,
# every process under run_capped.py (commit cap 6 GB, WS kill 3.8 GB: the brief's limit is 4 GB). Markers ZZL7SM<NAME>ZZ in smoke/smoke.log.
# usage: smoke_l7cap_cpu.sh <step...>   steps: W (wrapped train139 s0)  U / U2 (unwrapped train139 s0, two references)  M (wrong-index mutant)  W2 (second wrapped)
#        E (wrapped eval139 s1 os-only, agents+map)  UE (unwrapped eval139 s1 os-only, the K1 reference for E)
P=C:/Users/Admin/refcv8_pf/l7cap; S=$P/smoke
HOLDW=C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
CK=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CFG=E:/refcv8_eval_kit/ckpt/config.json
AGJ=E:/refcv6_eval_kit/data/joins/b1_train_plus_eval_agents.jsonl.xz
SAMEV=E:/refcv6_eval_kit/data/sam3_gt_v3_eval
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=-1 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1
mark() { echo "ZZL7SM$1ZZ $(date +%FT%T%z) $2" >> "$S/smoke.log"; }
CAP="$PY $P/code/run_capped.py --limit-gb 6 --ws-kill-gb 3.8"
for step in "$@"; do
  case $step in
    W|W2|M|U|U2)
      case $step in W) D=$S/wrapped/s0;; W2) D=$S/wrapped2/s0;; M) D=$S/mutant/s0;; U) D=$S/unwrapped/s0;; U2) D=$S/unwrapped2/s0;; esac
      rm -rf $D; mkdir -p $D
      common="--ckpt $CK --config $CFG --seed 0 --dump-dir $D/dump --out-json $D/roll_s0.json --device cpu --max-clips 1 --max-windows-per-clip 3 --smoke-cpu-fp32-trunk"
      mark "${step}_START" ""
      if [ $step = W ] || [ $step = W2 ]; then
        $CAP --tag $step -- "$PY" -X utf8 $P/code/fan_roll_r8_l7cap.py --mode train139 --pf-code $HOLDW/code --fan-meta-json $D/fan_roll_s0.json             --registered-windows $P/TRAIN139_WINDOWS.json --l7-agent-join $AGJ $common > $D/roll.log 2> $D/roll.err
      elif [ $step = M ]; then      # control (c): the writer dumps the WRONG candidate index; the roll-time check and the machinery must both go RED
        L7CAP_MUTATION=wrong_sel_idx $CAP --tag $step -- "$PY" -X utf8 $P/code/fan_roll_r8_l7cap.py --mode train139 --pf-code $HOLDW/code             --fan-meta-json $D/fan_roll_s0.json --registered-windows $P/TRAIN139_WINDOWS.json $common > $D/roll.log 2> $D/roll.err
      else
        $CAP --tag $step -- "$PY" -X utf8 $HOLDW/code/roll_seed_r8_train_pf.py $common > $D/roll.log 2> $D/roll.err
      fi
      mark "${step}_END" "rc=$? $(grep -h ZZCAPPEDZZ $D/roll.err | tail -1)";;
    UE)
      D=$S/eval_unwrapped/s1; rm -rf $D; mkdir -p $D
      mark "UE_START" ""
      $CAP --tag UE -- "$PY" -X utf8 $HOLDW/code/roll_seed_r8_pf.py --ckpt $CK --config $CFG --seed 1 --dump-dir $D/dump           --windows-json C:/Users/Admin/refcv8_pf/picker/smoke/windows_smoke.json --device cpu --out-json $D/roll_s1.json --smoke-cpu-fp32-trunk --os-only > $D/roll.log 2> $D/roll.err
      mark "UE_END" "rc=$? $(grep -h ZZCAPPEDZZ $D/roll.err | tail -1)";;
    E)
      D=$S/eval_wrapped/s1; rm -rf $D; mkdir -p $D
      mark "E_START" ""
      $CAP --tag E -- "$PY" -X utf8 $P/code/fan_roll_r8_l7cap.py --mode eval139 --pf-code $HOLDW/code --ckpt $CK --config $CFG --seed 1 \
          --dump-dir $D/dump --windows-json C:/Users/Admin/refcv8_pf/picker/smoke/windows_smoke.json --out-json $D/roll_s1.json \
          --fan-meta-json $D/fan_roll_s1.json --device cpu --smoke-cpu-fp32-trunk --os-only --l7-agent-join $AGJ --l7-map-gt-root $SAMEV > $D/roll.log 2> $D/roll.err
      mark "E_END" "rc=$? $(grep -h ZZCAPPEDZZ $D/roll.err | tail -1)";;
  esac
done
mark "CHAIN_END" "$*"
