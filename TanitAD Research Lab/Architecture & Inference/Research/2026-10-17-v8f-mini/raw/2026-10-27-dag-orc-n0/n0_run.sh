#!/bin/bash
# N0 rollout of one seed: DAG nav-logged, argv from argv/N0_s$1.argv, env = dagt1_chain's env.  Usage: n0_run.sh <seed>
S=$1
O=/e/Projects/TanitAD-artifacts/dag-orc-n0
TTm=E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a
TTu=/e/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a
export PATH="/usr/bin:/mingw64/bin:$PATH"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 TANITEVAL_STACK_OVERRIDE=$(cygpath -w "$TTu/stack")
export PYTHONPATH="$TTm/stack;$TTm/taniteval;$TTm/stack/scripts"
export TMP=F:/TanitAD-scratch/n0_tmp TEMP=F:/TanitAD-scratch/n0_tmp
mapfile -t ARGV < <(tr -d '\r' < $O/argv/N0_s$S.argv)
C=$(df -m /c | awk 'NR==2{print $4}'); E=$(df -m /e | awk 'NR==2{print $4}')
echo "$(date '+%m-%d %H:%M:%S') s$S start; C: free $C MB, E: free $E MB" | tee -a $O/logs/n0_chain.out
[ "$C" -ge 2048 ] && [ "$E" -ge 1500 ] || { echo "$(date '+%m-%d %H:%M:%S') s$S STOP: disk guard" | tee -a $O/logs/n0_chain.out; exit 1; }
T0=$(date +%s)
"${ARGV[@]}" > $O/logs/N0DAG_s${S}_m10_causal_heads_navlogged.out 2>&1 < /dev/null
RC=$?
echo "$(date '+%m-%d %H:%M:%S') s$S done rc=$RC wall=$(( $(date +%s) - T0 ))s" | tee -a $O/logs/n0_chain.out
echo "rc=$RC wall_s=$(( $(date +%s) - T0 ))" > $O/done_N0_s${S}.txt
