#!/bin/bash
# Detached CPU watcher: scores W4/W5 as soon as each blind/identity bank is FINAL (4,754 windows).  Writes ONLY under $W/result (never into the hold).
#   pass 1 (blind_s0 final)            -> result/final_prov   (W4 PROVISIONAL: control (a)/(b)/(c) vs the identity roll pending)
#   pass 2 (identity_s0 also final)    -> result/final_id     (the registered seed-0 read, controls (a)(b)(c) included)
#   pass 3 (blind_s1 also final)       -> result/final_s01    (seed 1 beside)
# Each pass: anticheat_score.py then render_result.py -> result/<pass>/render/RESULT.md.  Markers ZZWATCH...ZZ in logs/watch.log.  Judged on ARTIFACTS.
PKG=C:/Users/Admin/qland/hold_anticheat_r8/code/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-27-refcv9-anticheat
source $PKG/code/env_score.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
BK=$W/bank; LOG=$W/logs/watch.log
mk() { echo "ZZWATCH$1ZZ $(date +%FT%T%z) $2" >> $LOG; }
fin() { $PY -I -c "import json,sys,numpy as np; z=np.load(sys.argv[1]); m=json.loads(str(z['meta_json'])); print(m['n_windows'] if m.get('final') is True else 'BAD')" "$1" 2>/dev/null || echo NOFILE; }
score() {  # $1 name  $2.. extra args
  local name=$1; shift
  mkdir -p $W/result/$name
  $PY -X utf8 $PKG/code/anticheat_score.py "$@" --out $W/result/$name > $W/logs/score_$name.log 2>&1
  if [ -s $W/result/$name/RESULT_RAW.json ]; then
    $PY -X utf8 $PKG/code/render_result.py --raw $W/result/$name/RESULT_RAW.json --meta $W/result/$name/RESULT_META.json --out $W/result/$name/render > /dev/null 2>&1
    mk SCORED "$name $([ -s $W/result/$name/render/RESULT.md ] && echo rendered || echo NO_RENDER)"
  else mk FAIL "$name: no RESULT_RAW.json (see logs/score_$name.log; guard_exit.txt?)"; fi
}
mk START ""
until [ "$(fin $BK/ztac_blind_s0.npz)" = "4754" ]; do sleep 30; done
mk BLIND0FINAL ""
score final_prov --seeds 0 --blind-bank $BK/ztac_blind_s0.npz
until [ "$(fin $BK/ztac_identity_s0.npz)" = "4754" ]; do sleep 60; done
mk IDENTITYFINAL ""
score final_id --seeds 0 --blind-bank $BK/ztac_blind_s0.npz --identity-bank $BK/ztac_identity_s0.npz
until [ "$(fin $BK/ztac_blind_s1.npz)" = "4754" ]; do sleep 60; done
mk BLIND1FINAL ""
score final_s01 --seeds 0,1 --blind-bank $BK/ztac_blind_s0.npz --blind-bank1 $BK/ztac_blind_s1.npz --identity-bank $BK/ztac_identity_s0.npz
mk END ""
