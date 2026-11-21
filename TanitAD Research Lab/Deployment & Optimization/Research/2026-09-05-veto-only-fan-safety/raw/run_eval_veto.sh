#!/bin/sh
# P4 -- the SECONDARY endpoint: the FOUR FAMILIES at T1 for the veto-only arms, paired
# against the BANKED base over the shared `ha0` floor, plus fan safety on each arm's own
# emitted dump so the primary story is told on the same artifact the families come from.
#
# T1 = self-action OPEN LOOP (PI ruling 2026-09-02). Never a closed-loop claim.
# ONE AT A TIME on the shared 4060. The training pod and Thor are never touched.
#
# Usage:  ARMS="s0/veto200 s1/veto200" sh run_eval_veto.sh
set -u
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO="/c/Users/Admin/refcv4b_repo"
OUT="/c/Users/Admin/veto_run/run"
BASE_DUMP="/c/Users/Admin/_wp56/dump/refcv3_40284_dump"
EVAL_EPS="/c/Users/Admin/run_refcv3_ol/data/eval"
EVAL_LABELS="/c/Users/Admin/run_refcv3_ol/data/s2_labels_v7.2_eval.jsonl.gz"
EVAL_LEAD="/c/Users/Admin/tanitad-wt/_lead_b1/b1_eval_lead_block.npz"

export TANITAD_REPO="$REPO" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
export PYTHONPATH="$REPO/stack;$REPO"

STRIDE=5; NBOOT=2000; SEED=0
ARMS="${ARMS:-s0/veto200 s1/veto200}"
mkdir -p "$OUT/eval"

[ -d "$BASE_DUMP" ] || { echo "ZZNOBASE-$BASE_DUMP-ZZ"; exit 2; }

for spec in $ARMS; do
  tag=$(echo "$spec" | tr '/' '-')          # s0-veto200
  CK="$OUT/$spec/ckpt/ckpt_after.pt"
  if [ ! -s "$CK" ]; then echo "ZZNOCKPT-$spec-ZZ"; continue; fi
  if [ ! -s "$OUT/eval/refcv3-40284-$tag.json" ]; then
    echo "ZZEVALSTART-$tag-$(date -u +%H:%M:%S)Z-ZZ"
    "$PY" -u "$REPO/taniteval/tools/openloop_suite.py" --ckpt "$CK" \
       --config "$OUT/$spec/ckpt/config.json" \
       --episodes "$EVAL_EPS" --labels "$EVAL_LABELS" --lead-block "$EVAL_LEAD" \
       --nav-source v72 --grid 2s --action-units steer --with-oracle-sel \
       --window-stride "$STRIDE" --n-boot "$NBOOT" --seed "$SEED" --expect-step 40284 \
       --dump-dir "$OUT/eval/${tag}_dump" --out-dir "$OUT/eval" \
       --tag "refcv3-40284-$tag" \
       --tiers os=T1,os_navshuf=T1,os_navzero=T1,oracle_sel=T0 \
       --corpus "physicalai B1 v7.2 EVAL split, 141 clips" \
       --parity-status "NON-PARITY (B1, v2_parity false)" \
       --train-eval-disjoint "ASSERTED by rl_refcv3_min.py GATE 3 (fit n eval = 0)" \
       --strict > "$OUT/eval_$tag.log" 2>&1
    rc=$?
    echo "ZZEVALDONE-$tag-rc$rc-$(date -u +%H:%M:%S)Z-ZZ"
    # the documented "paid compute, lost analysis" trap: a COMPLETED rollout that died in
    # its analysis step reads like a total failure. Retry --analyze-only before re-rolling.
    if [ ! -s "$OUT/eval/refcv3-40284-$tag.json" ] && [ -s "$OUT/eval/${tag}_dump/manifest.json" ]; then
      echo "ZZANALYZEONLY-$tag-ZZ"
      "$PY" -u "$REPO/taniteval/tools/openloop_suite.py" \
         --analyze-only "$OUT/eval/${tag}_dump" --out-dir "$OUT/eval" \
         --tag "refcv3-40284-$tag" --n-boot "$NBOOT" --seed "$SEED" \
         >> "$OUT/eval_$tag.log" 2>&1
      echo "ZZANALYZEONLY-$tag-rc$?-ZZ"
    fi
  else
    echo "ZZEVALSKIP-$tag-ZZ"
  fi

  if [ -d "$OUT/eval/${tag}_dump" ]; then
    "$PY" -u "$REPO/taniteval/tools/paired_openloop.py" \
       --a-dump "$BASE_DUMP" --a-name base --a-arm os \
       --b-dump "$OUT/eval/${tag}_dump" --b-name "$tag" --b-arm os \
       --floor ha0 --n-boot "$NBOOT" --seed "$SEED" \
       --out "$OUT/paired_${tag}_vs_base.json" --md "$OUT/paired_${tag}_vs_base.md" \
       >> "$OUT/eval_$tag.log" 2>&1
    echo "ZZPAIRED-$tag-rc$?-ZZ"
    "$PY" -u "$REPO/taniteval/tools/fan_safety.py" --dump "$OUT/eval/${tag}_dump" \
       --lead-block "$EVAL_LEAD" --n-boot "$NBOOT" --seed "$SEED" \
       --out "$OUT/eval/fan_safety_${tag}.json" >> "$OUT/eval_$tag.log" 2>&1
    echo "ZZFANSAFE-$tag-rc$?-ZZ"
    # The FOUR-FAMILIES rule is BINDING and machine-checked here, not asserted in
    # prose: a missing family is a WORK ITEM, and the checker NAMES it rather than
    # letting it be absent from a table nobody diffed.
    "$PY" -u "$REPO/tools/criteria_check.py" "$OUT/eval/refcv3-40284-$tag.json" \
       --registry "$REPO/products/P7-TanitEval/CRITERIA_REGISTRY.json" \
       --json "$OUT/eval/criteria_${tag}.json" >> "$OUT/eval_$tag.log" 2>&1
    echo "ZZCRITERIA-$tag-rc$?-ZZ"
  fi
done
echo "ZZEVAL-COMPLETE-$(date -u +%H:%M:%S)Z-ZZ"
