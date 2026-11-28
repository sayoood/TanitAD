#!/bin/sh
# pull_and_analyze.sh <tag> <seed> <bank_root> <label> -- pull one Thor fan roll (out/<tag>) to the dev box and run fan_control_r8.py (CPU only).
# bank_root = the banked per-frame battery dir holding dump_s<seed>/, s6_s<seed>/, analysis_s<seed>.json.
TAG=${1:?tag}; SEED=${2:?seed}; BANK=${3:?bank}; LABEL=${4:?label}
F=/home/nvidia/r8ns/fanceil; D=C:/Users/Admin/refcv8_pf/fanceil; L=$D/runs/$TAG; mkdir -p $L
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
scp -q -r tanitad-thor:$F/out/$TAG/dump tanitad-thor:$F/out/$TAG/fan_roll_s$SEED.json tanitad-thor:$F/out/$TAG/roll_s$SEED.json tanitad-thor:$F/out/$TAG/fanroll_s$SEED.log $L/
ssh -n tanitad-thor "md5sum $F/out/$TAG/dump/fan_extras.npz" | cut -c1-32 > $L/fan_extras.md5.thor
md5sum $L/dump/fan_extras.npz | cut -c1-32 > $L/fan_extras.md5.dev
cmp -s $L/fan_extras.md5.thor $L/fan_extras.md5.dev && echo "ZZPULLMD5OKZZ $(cat $L/fan_extras.md5.dev)" || { echo "ZZPULLMD5BADZZ"; exit 2; }
"$PY" -X utf8 $D/code/fan_control_r8.py --fan-npz $L/dump/fan_extras.npz --bank-dump $BANK/dump_s$SEED --bank-s6 $BANK/s6_s$SEED \
  --bank-analysis $BANK/analysis_s$SEED.json --roll-dump $L/dump --fan-roll-json $L/fan_roll_s$SEED.json \
  --x8 "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/fan_control_x8.py" \
  --taniteval-root C:/lgs/Lfb47a58376/taniteval --seed $SEED --label "$LABEL" --out-json $D/FANCEIL_R8_$TAG.json ${EXTRA:-}
