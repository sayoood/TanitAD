#!/bin/bash
# smoke_p4_cpu.sh <step...> -- P4 CPU smoke on the r8oracle smoke windows (2 episodes x 3 windows = 6), seed 1, CPU fp32 trunk, ZERO GPU.
# Every process runs under run_capped.py (job-object commit cap) so a too-big CPU forward FAILS instead of pressuring the box.
#   A   compute mode (the real inversion, all arms)             -> smoke/A/{p4_s1.npz,dump_s1}, cache smoke/cache
#   B   read mode + P4_ZERO_DELTA=1 (identity: every arm == BASE, bit-exact) -> smoke/B
#   C   read mode (the cache round trip: arms must equal A's arms bit-exact)  -> smoke/C
#   AN  the smoke analysis (controls + reach ONLY; planner deltas are NOT computed in --smoke mode)
# Markers ZZP4SM<NAME>ZZ in smoke/smoke.log.
P=C:/Users/Admin/refcv8_pf/gtmap_oracle; S=${P4_SMOKE_DIR:-$P/smoke}; CODE=$P/code; T=$P/tree_fb47a58
W=C:/Users/Admin/refcv8_pf/r8oracle/smoke/windows_smoke.json
PLAIN=C:/Users/Admin/refcv8_pf/r8oracle/smoke_final/plain_dump_s1
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
CAP="$PY C:/Users/Admin/refcv8_pf/l7cap/code/run_capped.py --limit-gb ${P4_SMOKE_CAP_GB:-7}"
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=-1 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1
unset P4_ZERO_DELTA
mkdir -p "$S"; LOG=$S/smoke.log
mark() { echo "ZZP4SM$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
CH="--ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json --seed 1 --windows-json $W --device cpu --os-only --smoke-cpu-fp32-trunk"
P4="--map-gt-root E:/refcv6_eval_kit/data/sam3_gt_v3_eval --remap-json $CODE/refcv8_eval_remap_devbox_p4.json --inv-amp off --inv-chunk ${P4_SMOKE_CHUNK:-1} --save-every 2 ${P4_SMOKE_EXTRA:-}"
for step in "$@"; do
  case $step in
    A|B|C)
      D=$S/$step; rm -rf "$D"; mkdir -p "$D"
      case $step in
        A) MODE="--delta-mode compute --delta-cache $S/cache"; rm -rf "$S/cache";;
        B) MODE="--delta-mode read --delta-cache $S/cache"; export P4_ZERO_DELTA=1;;
        C) MODE="--delta-mode read --delta-cache $S/cache"; unset P4_ZERO_DELTA;;
      esac
      mark "${step}_START" ""
      $CAP --tag p4smoke_$step -- "$PY" -X utf8 $CODE/p4_roll.py --p4-out $D/p4_s1.npz $MODE $P4 -- $CH --dump-dir $D/dump_s1 --out-json $D/roll_s1.json \
          > $D/roll.log 2> $D/roll.err
      rc=$?
      unset P4_ZERO_DELTA
      mark "${step}_END" "rc=$rc artifact=$( [ -s $D/p4_s1.npz ] && echo yes || echo NO ) $(grep -h ZZCAPPEDZZ $D/roll.err | tail -1)";;
    AN)
      mark AN_START ""
      cp -f $S/A/p4_s1.npz $S/A/p4_s1.npz 2>/dev/null
      "$PY" -X utf8 $CODE/p4_analyze.py --root $S/A --seeds 1 --smoke --plain-dump $PLAIN --identity-npz $S/B/p4_s1.npz > $S/analyze.log 2>&1
      mark AN_END "rc=$? $(tail -1 $S/analyze.log)";;
  esac
done
