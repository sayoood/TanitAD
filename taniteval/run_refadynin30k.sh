#!/usr/bin/env bash
# Full milestone eval of refa-dynin-30k on the eval pod. Sequential (GPU shared
# with a possible video-render agent). Continues past any single panel failure.
set -u
cd /root/taniteval
export PYTHONPATH=/root/taniteval:/root/TanitAD/stack
KEY=refa-dynin-30k
LOG=/root/taniteval/results/run_${KEY}.log
: > "$LOG"

step () {  # step "label" cmd...
  local label="$1"; shift
  echo "===== [$label] $(date '+%H:%M:%S') =====" | tee -a "$LOG"
  "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "----- [$label] rc=$rc $(date '+%H:%M:%S') -----" | tee -a "$LOG"
  return 0
}

echo "#### FULL SUITE refa-dynin-30k START $(date) ####" | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$LOG"

step core       python -m taniteval.runner run        --model $KEY --episodes 40
step pathspeed  python -m taniteval.runner pathspeed  --model $KEY --episodes 40
step gen_phys   python -m taniteval.runner generalize --model $KEY --corpus physicalai --episodes 40
step gen_comma  python -m taniteval.runner generalize --model $KEY --corpus comma      --episodes 40
step gen_cosmos python -m taniteval.runner generalize --model $KEY --corpus cosmos     --episodes 40
step imag       python -m taniteval.runner imagination --model $KEY --episodes 12
step hier       python -m taniteval.runner hierarchy   --model $KEY --episodes 40
step plan       python -m taniteval.planning           --model $KEY
step diag       python -m taniteval.bench              --model $KEY --episodes 40
step ab         python -m taniteval.runner ab --a $KEY --b flagship-30k

echo "#### FULL SUITE refa-dynin-30k DONE $(date) ####" | tee -a "$LOG"
echo "== result files ==" | tee -a "$LOG"
ls -la /root/taniteval/results/ | grep -E "${KEY}|ab_${KEY}" | tee -a "$LOG"
