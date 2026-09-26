#!/bin/sh
# SPEC A6 (the deployable L2) at step 5000, unattended. The Master Mind's priority order (2026-09-26):
# step 30000 -> A6 -> L3 -> 15k/20k -> FINAL. So this chain waits for the pre-switch chain to bank step
# 30000 (the chain then only waits on Thor, and the GPU is free), rolls the 139 A6 TRAIN clips at inference
# seeds 0 and 1 (each roll in its own process, each behind the dev-box gate), fits w on those rolls ONLY,
# scores the step-5000 EVAL panels with NO refit, and banks. l3_tag.sh waits for this chain's end marker.
# Markers (opaque ZZ...ZZ) in raw/a6/a6_chain.log.
B=/c/Users/Admin/ev6_battery
BW='C:/Users/Admin/ev6_battery'
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
REL=FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval"
export OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
T=step5000
A6D=$B/raw/a6/$T
AL=$B/raw/a6/a6_chain.log
mkdir -p $A6D
cd $B/code || exit 1
echo "ZZA6STARTZZ $(date +%FT%T)" >> $AL
until grep -qE 'ZZBANKED_step30000ZZ|ZZBANKFAIL_step30000ZZ|ZZCHAINMENDZZ' $B/raw/chain.log; do sleep 120; done
if ! grep -q '"verdict": "PASS"' $B/raw/a6/pull_record.json 2>/dev/null; then
  echo "ZZA6FAILZZ train-clip pull not PASS $(date +%FT%T)" >> $AL; exit 1
fi
for s in 0 1; do
  [ -s $A6D/roll_s$s.json ] && continue
  $PY -c "import sys; sys.path.insert(0, r'$BW/code'); import run_battery as RB; RB.gate_wait(r'$BW/raw/a6/$T/gate_s$s.json')" > $A6D/gate_s$s.log 2>&1
  REFCV6_REMAP_OVERRIDES="$BW/raw/a6/remap_overrides_train.json" \
    $PY a6_roll.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_5000.pt --config D:/refcv6_eval_kit/ckpt/config.json \
      --seed $s --episodes D:/refcv6_eval_kit/data/refcv6-b1-416x1024-train-a6 \
      --labels D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz \
      --dump-dir "$BW/raw/a6/$T/dump_s$s" --out-json "$BW/raw/a6/$T/roll_s$s.json" > $A6D/roll_s$s.log 2>&1
  echo "ZZA6ROLL_s${s}ZZ $( [ -s $A6D/roll_s$s.json ] && echo ok || echo NOJSON ) $(date +%FT%T)" >> $AL
done
$PY a6_fit_score.py "$BW/raw/$T" "$BW/raw/a6/$T" > $A6D/fit_score.log 2>&1
if [ -s $A6D/a6.json ]; then
  $PY bank_tag.py "$BW/raw/a6/$T" "$PKG/raw/a6/$T" > $A6D/bank.log 2>&1
  for f in select_record.json train_clips_sha12.json pull_record.json remap_overrides_train.json; do
    $PY sanitize_for_bank.py "$BW/raw/a6/$f" "$PKG/raw/a6" > /dev/null 2>&1
  done
  { echo ""; echo "## $(date +%F) SPEC A6 deployable L2 at $T (a6_chain.sh; sanitized, sha12 ids)"
    for f in $(ls "$PKG/raw/a6/$T"); do [ -f "$PKG/raw/a6/$T/$f" ] && echo "$REL/raw/a6/$T/$f"; done
    for f in select_record.json train_clips_sha12.json pull_record.json remap_overrides_train.json; do echo "$REL/raw/a6/$f"; done
  } >> "$PKG/LANDING_READY.txt"
  echo "ZZA6DONEZZ $(grep -o '"verdict": "[^"]*"' $A6D/a6.json | tail -1) $(date +%FT%T)" >> $AL
else
  echo "ZZA6FAILZZ no a6.json $(date +%FT%T)" >> $AL
fi
