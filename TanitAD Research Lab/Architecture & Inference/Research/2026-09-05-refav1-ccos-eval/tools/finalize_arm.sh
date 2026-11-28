#!/bin/bash
# D-REFAV1-CCOS-EVAL — one command from a landed dump to every number the brief asks for:
#   1. refav1_arm --analyze-only (four families, paired cl-ha0 / cl-ha0_ext, lead block)  [zero GPU]
#   2. openloop_suite --arm-json + --dump-dir (criteria registry check, const0 control)
#   3. tools/criteria_check.py on the suite artifact (the TanitAD_BenchmarkCriteria instrument)
#   4. panel_shape (1b-1d by CONTENT) against the banked cos dump
#   5. paired ccos-vs-cos on the SAME windows (+ HOLD / non-HOLD strata)
#   6. echo gate 1 (ha, ha0_ext, ha0)
# usage: finalize_arm.sh <name> <dump_dir>
set -u
NAME="$1"; DUMP="$2"
C=/c/Users/Admin/ccos_eval; WT="C:/Users/Admin/tanitad-wt"
export PYTHONPATH="C:/Users/Admin/tanitad-wt/stack;C:/Users/Admin/tanitad-wt/taniteval" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
LEAD="C:/Users/Admin/tanitad-wt/TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz"
OUT="$C/final/$NAME"; mkdir -p "$OUT"
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
n=$(ls "$DUMP"/ep*.npz 2>/dev/null | wc -l); [ -f "$DUMP/manifest.json" ] || { stamp "no manifest in $DUMP (arm not finished?) n=$n"; exit 2; }
stamp "finalize $NAME from $DUMP ($n episode files)"
cd "$WT" || exit 1
"$PY" taniteval/tools/refav1_arm.py --analyze-only "$DUMP" --lead-block "$LEAD" --out "$OUT/rec_$NAME.json" --arm "refav1-21109-$NAME" --n-boot 2000 > "$OUT/analyze.log" 2>&1 || { stamp "analyze FAILED"; tail -5 "$OUT/analyze.log"; exit 3; }
"$PY" taniteval/tools/openloop_suite.py --arm-json "$OUT/rec_$NAME.json" --dump-dir "$DUMP" --headline-arm cl --out-dir "$OUT/suite" --tag "suite-refav1-21109-$NAME" > "$OUT/suite.log" 2>&1 || { stamp "suite FAILED"; tail -5 "$OUT/suite.log"; }
grep -E "constant-only|criteria:" "$OUT/suite.log"
"$PY" tools/criteria_check.py "$OUT/suite/suite-refav1-21109-$NAME.json" > "$OUT/criteria_check.log" 2>&1; tail -3 "$OUT/criteria_check.log"
cd "$C" || exit 1
"$PY" tools/panel_shape.py --dump cos_banked=dump_cos_banked/full --dump "$NAME=$DUMP" --stack "C:/Users/Admin/tanitad-wt/stack" --out "$OUT/shape_$NAME.json" > "$OUT/shape.log" 2>&1; cat "$OUT/shape.log"
"$PY" tools/paired_dumps_refav1.py --a-dump dump_cos_banked/full --a-name cos --b-dump "$DUMP" --b-name "$NAME" --panel thor/box_panel_282.json --stack "C:/Users/Admin/tanitad-wt/stack" --taniteval "C:/Users/Admin/tanitad-wt/taniteval" --n-boot 2000 --out "$OUT/paired_${NAME}_vs_cos.json" --md "$OUT/paired_${NAME}_vs_cos.md" > "$OUT/paired.log" 2>&1; tail -4 "$OUT/paired.log"
"$PY" tools/run_echo_gate.py --dump "$DUMP" --arm cl --stack "C:/Users/Admin/tanitad-wt/stack" --taniteval "C:/Users/Admin/tanitad-wt/taniteval" --out "$OUT/echo_gate1_$NAME.json" > "$OUT/echo.log" 2>&1; cat "$OUT/echo.log"
stamp "finalize $NAME done -> $OUT"
