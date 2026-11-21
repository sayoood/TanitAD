#!/bin/sh
# Re-run EVERY measurement against the FIXED module (the stopped-step epsilon). No number
# in the package may come from the pre-fix build: mixing them is exactly the "true
# measurement quoted outside its scope" failure this programme keeps logging.
set -u
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO="/c/Users/Admin/refcv4b_repo"
RAW="/c/Users/Admin/feasdec/raw"
RUN="/c/Users/Admin/feasdec/run"
BASE="/c/Users/Admin/_wp56/dump/refcv3_40284_dump"
NPZ400="C:/Users/Admin/kingate/raw/kingate_bank_drawA_v1_selscore.npz"
NPZ240="C:/Users/Admin/veto_run/raw/fan_bank_base_240w.npz"
BVF="C:/Users/Admin/veto_run/raw/bank_vs_fan_feasibility.json"
export TANITAD_REPO="$REPO" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=3
export PYTHONPATH="$REPO/stack;$REPO/taniteval;$REPO"

echo "ZZSTEP-residual-$(date -u +%H:%M:%S)Z-ZZ"
"$PY" -u "$RAW/stopped_step_residual.py" > "$RAW/stopped_step_residual.log" 2>&1
echo "ZZDONE-residual-rc$?-ZZ"

echo "ZZSTEP-mu-$(date -u +%H:%M:%S)Z-ZZ"
"$PY" -u "$RAW/mu_frontier.py" --npz "$NPZ400" \
  --ckpt C:/Users/Admin/rl_refcv3_min/base/ckpt_step40284_frozen.pt \
  --bank-vs-fan "$BVF" --out "$RAW/mu_frontier.json" > "$RAW/mu_frontier.log" 2>&1
echo "ZZDONE-mu-rc$?-ZZ"

echo "ZZSTEP-p1c-$(date -u +%H:%M:%S)Z-ZZ"
"$PY" -u "$RAW/projected_fan_rank.py" --npz "$NPZ400" --fan-key fan8 \
  --out "$RAW/projected_fan_rank_400w.json" > "$RAW/projected_fan_rank.log" 2>&1
echo "ZZDONE-p1c-rc$?-ZZ"

echo "ZZSTEP-p1-$(date -u +%H:%M:%S)Z-ZZ"
"$PY" -u "$RAW/progress_rank_fix.py" --npz "$NPZ240" --bank-vs-fan "$BVF" \
  --fan-key fan2 --out "$RAW/progress_rank_fix_240w.json" > "$RAW/p1_240w.log" 2>&1
"$PY" -u "$RAW/progress_rank_fix.py" --npz "$NPZ400" --bank-vs-fan "$BVF" \
  --fan-key fan8 --out "$RAW/progress_rank_fix_400w.json" > "$RAW/p1_400w.log" 2>&1
echo "ZZDONE-p1-rc$?-ZZ"

echo "ZZSTEP-derive-$(date -u +%H:%M:%S)Z-ZZ"
"$PY" -u "$RAW/derive_projected_dump.py" --base-dump "$BASE" --out-root "$RUN" \
  --report "$RAW/derived_dump_report.json" > "$RAW/derive.log" 2>&1
echo "ZZDONE-derive-rc$?-ZZ"

for arm in projoff proj07 proj07e; do
  echo "ZZPAIR-START-$arm-$(date -u +%H:%M:%S)Z-ZZ"
  "$PY" -u "$REPO/taniteval/tools/paired_openloop.py" \
     --a-dump "$BASE" --a-name base --a-arm os \
     --b-dump "$RUN/${arm}_dump" --b-name "$arm" --b-arm os \
     --floor ha0 --n-boot 2000 --seed 0 \
     --out "$RAW/paired_${arm}_vs_base.json" \
     --md  "$RAW/paired_${arm}_vs_base.md" > "$RAW/paired_${arm}.log" 2>&1
  echo "ZZPAIR-DONE-$arm-rc$?-$(date -u +%H:%M:%S)Z-ZZ"
done
echo "ZZRERUN-COMPLETE-$(date -u +%H:%M:%S)Z-ZZ"
