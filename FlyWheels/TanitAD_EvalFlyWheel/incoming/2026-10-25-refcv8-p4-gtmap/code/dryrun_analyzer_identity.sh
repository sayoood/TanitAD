#!/bin/bash
# dryrun_analyzer_identity.sh -- exercises p4_analyze.py's SCORED path end-to-end on IDENTITY data (zero-delta rolls at BOTH seeds on the
# 6 smoke windows): every arm == BASE, so every arm-vs-arm contrast must read EXACTLY 0 -- a known value -- and no oracle effect exists
# to be seen.  CPU only.  Markers ZZP4DRY<NAME>ZZ in smoke_final/dry.log.
P=C:/Users/Admin/refcv8_pf/gtmap_oracle; S=$P/smoke_final; CODE=$P/code; T=$P/tree_fb47a58
W=C:/Users/Admin/refcv8_pf/r8oracle/smoke/windows_smoke.json
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
CAP="$PY C:/Users/Admin/refcv8_pf/l7cap/code/run_capped.py --limit-gb 6"
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=-1 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1
LOG=$S/dry.log; mk() { echo "ZZP4DRY$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
D=$S/B0; rm -rf "$D"; mkdir -p "$D"
mk B0_START ""
P4_ZERO_DELTA=1 $CAP --tag p4dry_B0 -- "$PY" -X utf8 $CODE/p4_roll.py --p4-out $D/p4_s0.npz --delta-mode read --delta-cache $S/cache \
   --map-gt-root E:/refcv6_eval_kit/data/sam3_gt_v3_eval --remap-json $CODE/refcv8_eval_remap_devbox_p4.json -- \
   --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json --seed 0 --windows-json $W --device cpu \
   --os-only --smoke-cpu-fp32-trunk --dump-dir $D/dump_s0 --out-json $D/roll_s0.json > $D/roll.log 2> $D/roll.err
mk B0_END "rc=$? artifact=$( [ -s $D/p4_s0.npz ] && echo yes || echo NO )"
R=$S/DRY; rm -rf "$R"; mkdir -p "$R"
cp -r $S/B0/dump_s0 $R/dump_s0; cp $S/B0/p4_s0.npz $R/p4_s0.npz
cp -r $S/B/dump_s1 $R/dump_s1; cp $S/B/p4_s1.npz $R/p4_s1.npz
"$PY" -X utf8 $CODE/p4_analyze.py --root $R --seeds 0,1 > $R/analyze.log 2>&1
mk AN_END "rc=$? $(tail -1 $R/analyze.log | cut -c1-200)"
