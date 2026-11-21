#!/usr/bin/env bash
# MEASURE one real planner seed (seed 0, full 994 windows, lam 0 hold) on R35_s0 THROUGH the new route (WMACT_CODE_ROOT = optb tree + probe remap), exactly the chain's planner command. Output: tmp/timing/{plan.log,timing.txt}.
export PATH="/usr/bin:/bin:$PATH"
RC="F:/TanitAD-scratch/actrsp/optb/readchain"; PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"; TREE="F:/TanitAD-scratch/actrsp/optb/tree"
W="$RC/tmp/timing"; rm -rf "${W:?}"; mkdir -p "$W/tmp"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 TMP="$W/tmp" TEMP="$W/tmp" OPTB_TREE="$TREE" WMACT_CODE_ROOT="$TREE"
export PYTHONPATH="$(cygpath -w "$TREE/stack");$(cygpath -w "$TREE/taniteval")"
"$PY" F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 120 || exit 6
t0=$(date +%s)
"$PY" "$RC/optb_route.py" F:/TanitAD-scratch/actrsp/lonplan/build/lonplan_run.py plan --windows-src F:/TanitAD-scratch/actsens/read/out/C7_R35_s0/win/windows.npz \
  --windows-sha256 371c753bcc3f28f1052c0e5e7e4898c0bbbed6140477b3338c8133a3632bdaee --declare-lam 0 --declare-seed-mode hold \
  --ckpt-sha256 d920d62c6c391a61291b966a6af1783156b4351d25c7408913e4191daa0215cf -- --tag R35_s0 --ckpt C:/Users/Admin/t2v8f_pulls/ladder_R35_s0/ckpt_last.pt \
  --train-latents F:/TanitAD-scratch/hwm_plan/latents_train_R35_s0.pt --train-tag R35_s0 --seed-mode hold --lam=0 --arms hwm_oracle --seeds 0 --workers 1 --out "$W/plan" > "$W/plan.log" 2>&1
rc=$?
echo "rc=$rc wall_s=$(( $(date +%s) - t0 )) seeds=1 windows=full finished=$(date '+%F %T')" > "$W/timing.txt"
