#!/bin/bash
# MM 2026-11-20: whole-file runs of the eval tests the SCORER-FT read + polarity land touches, on swp/q (c98e6114 + both holds), and on swp/p
# (pristine b3545f08 = c98e6114 for these eval paths) for the baseline. One file at a time under the shared pytest mutex.
S=C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/e07efd49-d158-4484-85f3-d8888196bc46/scratchpad
OUT="$S/evaltests"; mkdir -p "$OUT"; : > "$OUT/results.tsv"
LK=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/pytest.lockdir
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
for tree in q p; do
  ROOT=/c/Users/Admin/swp/$tree; EV="$ROOT/$PK/eval"
  for t in test_fam_direction.py p5c/test_p5c_read.py p5c/test_p5c_families.py p5c/test_p5c_tactical.py p5c/test_p5c_render.py p5c/test_p5c_b10b12_wiring.py \
           p5c/test_p5c_determinism.py p5c/test_p5c_seal.py field/test_field_read.py scorerft/test_scorerft_read.py; do
    [ -f "$EV/$t" ] || { printf '%s\t%s\tABSENT\n' "$tree" "$t" >> "$OUT/results.tsv"; continue; }
    until mkdir "$LK" 2>/dev/null; do sleep 15; done
    mkdir -p "${OUT:?}/tmp"; export TMP="$(cygpath -w "${OUT:?}/tmp")" TEMP="$(cygpath -w "${OUT:?}/tmp")"
    cd "$EV/$(dirname "$t")"
    PP="$(cygpath -w "$ROOT/stack");$(cygpath -w "$ROOT/taniteval");$(cygpath -w "$EV");$(cygpath -w "$EV/p5c");$(cygpath -w "$EV/field");$(cygpath -w "$ROOT/$PK/refe");$(cygpath -w "$ROOT")"
    log="$OUT/${tree}_$(echo "$t" | tr '/' '_').log"
    PYTHONPATH="$PP" FIELD_READ_P5C_DIR="$(cygpath -w "$EV/p5c")" CUDA_VISIBLE_DEVICES=-1 timeout 1200 "$PY" -m pytest -q -p no:cacheprovider --basetemp "$OUT/bt" "$(basename "$t")" > "$log" 2>&1; rc=$?
    rm -rf "${OUT:?}/bt" "${OUT:?}/tmp"; rmdir "$LK"
    printf '%s\t%s\t%s\t%s\n' "$tree" "$t" "$rc" "$(tail -n 1 "$log" | tr -d '\r' | cut -c1-160)" >> "$OUT/results.tsv"
  done
done
echo EVALTESTS_DONE >> "$OUT/results.tsv"; cat "$OUT/results.tsv"
