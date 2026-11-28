#!/bin/bash
# Identity controls I1-I4 on the 74-twin D7 mini bank (PREREG_3C.md).  Held arm (I1-I3) and no-hold mutation arm (I4), NSH shards each, detached by the caller.
set -u
source /workspace/ops/bank3c/code/env_3c.sh
C=/workspace/ops/bank3c/controls; NSH=${NSH:-16}
mkdir -p $C/logs; date -u +%s > $C/started_epoch
cd $PKG/refe
for ((i=0; i<NSH; i++)); do
  s=$(printf "s%04d" $i)
  run3c $PY $CODE3C/reroll_bank_3c.py run --bank $C/bank_i3 --rank0 $RANK0 --out $C/held/shards/$s --mc /workspace/ops/d7fix/mc_final \
     --log-shard $i/$NSH --code-dir $CODE3C --refe-dir $PKG/refe --pkg-code-dir $PKG/code > $C/logs/held_$s.log 2>&1 &
  run3c $PY $CODE3C/reroll_bank_3c.py run --bank $C/bank_i3 --rank0 $RANK0 --out $C/legacy/shards/$s --mc /workspace/ops/d7fix/mc_final \
     --log-shard $i/$NSH --code-dir $CODE3C --refe-dir $PKG/refe --pkg-code-dir $PKG/code --mutate-no-hold > $C/logs/legacy_$s.log 2>&1 &
done
wait
date -u +%s > $C/finished_epoch
echo "ZZ3C_CONTROLS_RUN_DONE" > $C/RUN_DONE
