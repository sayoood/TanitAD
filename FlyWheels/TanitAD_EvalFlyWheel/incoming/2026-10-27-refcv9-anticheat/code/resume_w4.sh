#!/bin/bash
# RESUME (CPU only): score the W4/W5 contrasts once the GPU rolls are banked, render RESULT.md/json into the package.
#   bash resume_w4.sh [identity]        # `identity` = also read control (a)/(b)/(c) against ztac_identity_s0.npz (required for a non-provisional W4)
# Judged on ARTIFACTS: the bank must hold a FINAL record of 4,754 windows, RESULT_RAW.json must carry the verdicts.
PKG=C:/Users/Admin/qland/hold_anticheat_r8/code/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-27-refcv9-anticheat
source $PKG/code/env_score.sh
BK=$W/bank
fin() { $PY -I -c "import json,sys,numpy as np; z=np.load(sys.argv[1]); m=json.loads(str(z['meta_json'])); print(m['n_windows'] if m.get('final') is True else 'BAD')" "$1" 2>/dev/null || echo NOFILE; }
echo "blind_s0 final_n=$(fin $BK/ztac_blind_s0.npz)  identity_s0 final_n=$(fin $BK/ztac_identity_s0.npz)  blind_s1 final_n=$(fin $BK/ztac_blind_s1.npz)"
[ "$(fin $BK/ztac_blind_s0.npz)" = "4754" ] || { echo "blind_s0 not final: wait for the roll (tail $W/logs/rolls.log)"; exit 3; }
ARGS="--seeds 0 --blind-bank $BK/ztac_blind_s0.npz --out $W/result/final_s0"
# seed 1 (beside) when its blind roll is final too
[ "$(fin $BK/ztac_blind_s1.npz)" = "4754" ] && ARGS="--seeds 0,1 --blind-bank $BK/ztac_blind_s0.npz --blind-bank1 $BK/ztac_blind_s1.npz --out $W/result/final_s0"
[ "$1" = "identity" ] && [ "$(fin $BK/ztac_identity_s0.npz)" = "4754" ] && ARGS="$ARGS --identity-bank $BK/ztac_identity_s0.npz"
mkdir -p $W/result/final_s0
$PY -X utf8 $PKG/code/anticheat_score.py $ARGS > $W/logs/score_final_s0.log 2>&1
[ -s $W/result/final_s0/RESULT_RAW.json ] || { echo "no RESULT_RAW.json: see $W/logs/score_final_s0.log"; exit 4; }
$PY -X utf8 $PKG/code/render_result.py --raw $W/result/final_s0/RESULT_RAW.json --meta $W/result/final_s0/RESULT_META.json --out $PKG
echo "RESULT written to $PKG/RESULT.md"
