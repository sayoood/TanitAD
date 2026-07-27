#!/bin/bash
# Small validation — the PRIMARY read. Runs ONLY after the training chain is done
# (pod2's cgroup sits at ~53.9/55.0 GB while training; an eval OOM-killed the
# flagship on 2026-07-16 and that must not repeat).
#
# n = 120 val episodes -- the PRE-REGISTERED primary (MDE 0.0059). The episodes
# are an order-preserving PREFIX of the matched 600, so 40 -> 120 -> 600 ADDS
# episodes and RE-SELECTS none: parity holds.
#
# Order = PRIORITY order: the PI's contrast (A, B) and its controls first; C last.
set -u
cd /workspace/smallval/code
export SMALLVAL_STACK=/workspace/TanitAD/stack
export TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack
export PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts:/workspace/TanitAD/taniteval
export OMP_NUM_THREADS=6

RAW_VA=/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11
V2_VA=/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl
ANCH=/workspace/experiments/flagship_v4_anchors_dense.pt
OUT=/workspace/smallval/ps
N=120
mkdir -p $OUT

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[eval] waiting for /workspace/smallval/CHAIN_DONE ... $(stamp)"
while [ ! -f /workspace/smallval/CHAIN_DONE ]; do sleep 120; done
echo "[eval] training chain finished $(stamp); GPU:"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

run_arm () {   # $1=name $2=run-dir $3=corpus $4..=extra flags
  NAME=$1; RUN=$2; CORP=$3; shift 3
  CK=$RUN/ckpt.pt
  if [ ! -f "$CK" ]; then echo "[eval] SKIP $NAME — no $CK"; return 0; fi
  for MODE in sighted blind; do
    ARM=$NAME; BF=""
    if [ "$MODE" = "blind" ]; then ARM=${NAME}_blind; BF="--blind"; fi
    if [ -f "$OUT/pw_$ARM.npz" ]; then echo "[eval] have $ARM, skip"; continue; fi
    echo "[eval] === $ARM ($CORP) $(stamp) ==="
    python3 -u smallval_pseudosim.py --arm "$ARM" --ckpt "$CK" \
      --anchors-dense $ANCH --corpus "$CORP" --episodes $N \
      --stride 8 --horizon 20 --batch 16 --goal-option dropped \
      --device cuda --out-dir $OUT $BF "$@" \
      > /tmp/smallval_ps_$ARM.log 2>&1
    echo "[eval] $ARM rc=$? $(stamp)"
  done
}

# ---- PRIORITY 1: the PI's contrast + its anti-C13 controls ------------------ #
run_arm A_old /workspace/smallval/A_old-256x256  raw --val-dir $RAW_VA
run_arm B_wide /workspace/smallval/B_wide-256x640 v2 \
  --v2-val-cache $V2_VA --v2-lru 64 \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe none
touch /workspace/smallval/PS_AB_DONE

# ---- PRIORITY 2: the rig-fix separator ------------------------------------- #
run_arm C_v5 /workspace/smallval/C_v5-176x624 v2 \
  --v2-val-cache $V2_VA --v2-lru 64 \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624

# ---- combine: panel gate, BOTH progress terms, paired bootstrap, decomposition #
echo "[eval] === COMBINE $(stamp) ==="
CUDA_VISIBLE_DEVICES="" python3 -u smallval_combine.py \
  --in-dir $OUT --out /workspace/smallval/raw/smallval_result.json --n-boot 2000 \
  > /tmp/smallval_combine.log 2>&1
echo "[eval] combine rc=$? $(stamp)"
touch /workspace/smallval/EVAL_DONE
echo "[eval] DONE $(stamp)"
