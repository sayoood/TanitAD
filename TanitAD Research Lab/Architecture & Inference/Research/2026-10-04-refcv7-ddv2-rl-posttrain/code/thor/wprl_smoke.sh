#!/bin/bash
# WP-RL stage-2 smoke on Thor (gate inputs). Each GPU process takes the shared lock IN-PROCESS for its
# GPU phase only (model + dataset are built on the CPU first). The ARTIFACTS are the evidence.
source /home/nvidia/refcv7_post/rl/wprl_env.sh
S=$R/smoke
mkdir -p $S
cd $TREE
W=$R/out/windows_train.json
DRV=stack/scripts/ddv2_rl_refcv7.py
run () {
  local name=$1; shift
  echo "[smoke] $(date -u +%FT%TZ) start $name"
  $PY $DRV "$@" > $S/$name.log 2>&1 < /dev/null
  echo "[smoke] $(date -u +%FT%TZ) end $name rc=$?"
}
TR="--windows $W --ckpt-every 0 --segment-minutes 0 --workers 4 --inproc-lock $LOCK"
# 1. zero-training diagnostics (D1-D6) + cost per micro-batch
run diagnose diagnose --split train --windows $W --n-windows 12 --cost-micro 1 2 4 8 --inproc-lock $LOCK --out $S/diagnose_train.json
# 2. IDENTITY (I-6): an rloff run at lr 0 must export the cold start bitwise
run id_train train --arm rloff --lr 0 --steps 2 --batch 8 --micro 4 --out-dir $S/identity $TR
run id_export export --run-dir $S/identity --out $S/identity/export.pt
run id_check identity --export $S/identity/export.pt --expect identical --out $S/identity/identity.json
# 3. RL at the real recipe: SEGMENTED (2 + 2 steps, resumed) and UNINTERRUPTED (4 steps), same seed
run rl_seg_a train --arm rl --steps 4 --batch 8 --micro 4 --out-dir $S/rl_seg --max-steps-this-segment 2 $TR
run rl_seg_b train --arm rl --steps 4 --batch 8 --micro 4 --out-dir $S/rl_seg $TR
run rl_full train --arm rl --steps 4 --batch 8 --micro 4 --out-dir $S/rl_full $TR
run rl_export export --run-dir $S/rl_full --out $S/rl_full/export.pt
run rl_check identity --export $S/rl_full/export.pt --expect different --out $S/rl_full/identity.json
# 4. RLOFF and RL-SHUF at the real recipe (2 steps each)
run rloff train --arm rloff --steps 2 --batch 8 --micro 4 --out-dir $S/rloff $TR
run shuf train --arm rlshuf --steps 2 --batch 8 --micro 4 --out-dir $S/rlshuf $TR
# 5. cost at the registered batch (32 windows, micro 8): 3 steps
run timing32 train --arm rl --steps 3 --batch 32 --micro 8 --out-dir $S/timing32 $TR
echo "[smoke] $(date -u +%FT%TZ) ALLDONE"
