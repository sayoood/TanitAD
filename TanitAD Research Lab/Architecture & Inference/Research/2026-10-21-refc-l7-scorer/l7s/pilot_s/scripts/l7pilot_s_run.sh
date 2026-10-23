#!/bin/bash
# l7pilot_s_run.sh -- H-L7-1 PILOT-S (PREREG_L7_SCORER.AMENDMENT-1.md, sha256 29dd3e84..., registered 2026-10-22 20:19),
# PI GO 2026-10-22 ~21:05 Berlin in chat ("Train PILOT-S now"). l7s/PILOT_S_COMMANDS.md verbatim; code = c04a183 (work copy
# blob-verified). CPU only. Markers ZZL7S<NAME>ZZ in l7pilot_s.log.
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -X utf8"
L7=C:/Users/Admin/refcv8_pf/l7prep; W=C:/Users/Admin/refcv8_pf/l7s_prep/code; RUN="$PY $W/run_isolated.py --stack $L7/stack_min"
B=D:/tanitad-archive/l7cap_thor_1022; T=$L7/targets; CACHE=C:/Users/Admin/refcv8_pf/l7s_prep/scene_cache; LOG=C:/Users/Admin/refcv8_pf/l7pilot_s.log
echo "ZZL7SSTARTZZ $(date +%FT%T%z)" >> $LOG
$RUN $W/scene_bank.py --npz $B/train139_s0/dump/l7cap_scene.npz $B/train139_s1/dump/l7cap_scene.npz --cache $CACHE > C:/Users/Admin/refcv8_pf/l7pilot_s_cache.log 2>&1
echo "ZZL7SCACHEZZ rc=$? $(date +%T)" >> $LOG
for seed in 0 1; do
  O=C:/Users/Admin/refcv8_pf/l7pilot/L7S_ts$seed
  $RUN $W/train_l7.py --fan-npz $B/train139_s0/dump/fan_extras.npz $B/train139_s1/dump/fan_extras.npz \
     --targets-npz $T/train139_s0.npz $T/train139_s1.npz --mode S --scene-cache $CACHE --seed $seed \
     --out-dir $O > C:/Users/Admin/refcv8_pf/l7pilot_L7S_ts$seed.log 2>&1
  echo "ZZL7SRUNZZ ts$seed rc=$? best=$([ -s $O/best.pt ] && echo yes || echo NO) log=$([ -s $O/train_log.json ] && echo yes || echo NO) $(date +%T)" >> $LOG
done
$PY $W/read_pilot_s.py C:/Users/Admin/refcv8_pf/l7pilot/L7S_ts0/train_log.json C:/Users/Admin/refcv8_pf/l7pilot/L7S_ts1/train_log.json > C:/Users/Admin/refcv8_pf/l7pilot_s_read.out 2>&1
echo "ZZL7SREADZZ rc=$? $(date +%T)" >> $LOG
echo "ZZL7SENDZZ $(date +%FT%T%z)" >> $LOG
