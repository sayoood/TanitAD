#!/bin/bash
# FULL rolls of PREREG_ANTICHEAT_REFCV8.md section 2, sequential, under ONE hold of the shared dev-box GPU lock (with_gpu_lock.py wraps THIS script).
#   stage 1 blind s0 (the binding W4 roll) -> stage 2 identity s0 (control (a), all 4,754 windows) -> stage 3 blind s1 (beside)
# Each stage is skipped when its bank already holds a FINAL record of 4,754 windows.  Judged on the ARTIFACT (final_n), never on an exit code.
PKG=C:/Users/Admin/qland/hold_anticheat_r8/code/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-27-refcv9-anticheat
source $PKG/code/env_anticheat.sh
LOGS=$W/logs; BK=$W/bank; mkdir -p $BK $LOGS
mk() { echo "ZZRL$1ZZ $(date +%FT%T%z) $2" >> $LOGS/rolls.log; }
final_n() { $PY -I -c "import json,sys,numpy as np; z=np.load(sys.argv[1]); m=json.loads(str(z['meta_json'])); print(m['n_windows'] if m.get('final') is True else 'BAD')" "$1" 2>/dev/null || echo NOFILE; }
roll() {   # $1 camera  $2 seed  $3 tag
  local out=$BK/ztac_$3.npz
  if [ "$(final_n $out)" = "4754" ]; then mk HAVE "$3"; return 0; fi
  rm -rf $BK/dump_$3
  mk START "$3"
  $PY -X utf8 $PKG/code/camblind_roll.py --camera $1 --ztac-out $out --require-seal -- \
      --ckpt $CK --config $CFG --seed $2 --dump-dir $BK/dump_$3 --windows-json $BANKT/windows_s$2.json --device cuda --out-json $BK/roll_$3.json --os-only \
      > $LOGS/roll_$3.log 2>&1
  mk DONE "$3 n=$(final_n $out)"
}
roll derange 0 blind_s0
roll identity 0 identity_s0
roll derange 1 blind_s1
mk END ""
