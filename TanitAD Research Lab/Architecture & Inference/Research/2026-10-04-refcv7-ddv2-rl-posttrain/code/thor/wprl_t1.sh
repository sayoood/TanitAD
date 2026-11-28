#!/bin/bash
# WP-RL stage 3, the T1 PRIMARY roll (SPEC_RL sec. 6.1): the refcv7 battery's own roll, os-only,
# on the S2 grid (4,754 windows / 139 episodes), for ONE checkpoint at ONE inference seed.
#   ./wprl_t1.sh <name> <ckpt.pt> <seed>
# One GPU job under the shared lock; the ARTIFACT (roll_<name>_s<seed>.json + its dump) is the evidence.
source /home/nvidia/refcv7_post/rl/wprl_env.sh
NAME=$1; CK=$2; SEED=$3
T=$R/t1
mkdir -p $T
OUT=$T/roll_${NAME}_s${SEED}.json
if [ -s $OUT ]; then echo "[t1] $OUT banked -- skip"; exit 0; fi
cd $R/battery_code
flock $LOCK $PY roll_seed_r7.py --ckpt $CK --config /home/nvidia/refcv7_run/runs/refcv7-r101-s0/config.json \
    --seed $SEED --dump-dir $T/dump_${NAME}_s${SEED} --windows-json $R/battery_win/s2_windows.json \
    --device cuda --out-json $OUT --os-only > $T/roll_${NAME}_s${SEED}.log 2>&1 < /dev/null
echo "[t1] $(date -u +%FT%TZ) $NAME s$SEED rc=$? artifact=$( [ -s $OUT ] && echo yes || echo NO )"
