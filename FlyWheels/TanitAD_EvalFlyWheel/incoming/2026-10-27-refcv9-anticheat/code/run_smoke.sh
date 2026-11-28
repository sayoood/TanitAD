#!/bin/bash
# SMOKE: a 36-window PREFIX of the s0 roll (same sampler offsets as the full roll) in identity then derange mode, under the shared GPU lock.
PKG=C:/Users/Admin/qland/hold_anticheat_r8/code/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-27-refcv9-anticheat
source $PKG/code/env_anticheat.sh
LOGS=$W/logs; mkdir -p $W/smoke $LOGS
echo "ZZSMOKESTARTZZ $(date +%FT%T%z)" >> $LOGS/smoke.log
for mode in identity derange; do
  rm -rf $W/smoke/dump_$mode
  $PY -X utf8 "$LOCKDIR/with_gpu_lock.py" --job refcv9-anticheat-smoke-$mode --log $LOGS/smoke_lock_$mode.log --rec $LOGS/smoke_lockrec_$mode.json --max-wait-s 3600 -- \
    $PY -X utf8 $PKG/code/camblind_roll.py --camera $mode --ztac-out $W/smoke/ztac_$mode.npz --require-seal --smoke-stop-after 36 -- \
      --ckpt $CK --config $CFG --seed 0 --dump-dir $W/smoke/dump_$mode --windows-json $BANKT/windows_s0.json --device cuda --out-json $W/smoke/roll_$mode.json --os-only \
      > $LOGS/smoke_roll_$mode.log 2>&1
  echo "ZZSMOKEDONEZZ $mode $(date +%FT%T%z) npz=$([ -s $W/smoke/ztac_$mode.npz ] && echo yes || echo no)" >> $LOGS/smoke.log
done
