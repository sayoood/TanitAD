#!/usr/bin/env bash
# Dry runs of field_read.py on the P5-C stand-ins (C1 for F, C0 for C). Usage: run_dry.sh <pair|self|selfC0>. Every output is stamped DRY RUN.
set -u
W=C:/Users/Admin/swp/fieldread
PYT=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 FIELD_READ_P5C_DIR="${FIELD_READ_P5C_DIR:-$W/pk/eval/p5c}"
D=E:/Projects/TanitAD/data/refe_navtest; O=E:/Projects/TanitAD-artifacts/p5c_read; N=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/propscore
FR="${FIELD_READ:-$W/land/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/field/field_read.py}"
case "$1" in
  pair) F=C1; C=C0; OUT=dry_C1_vs_C0; LAB="C1 vs C0 as stand-ins for F vs C";;
  self) F=C1; C=C1; OUT=dry_selfread_C1_vs_C1; LAB="CTRL-ID: C1 read against itself";;
  *) echo bad; exit 2;;
esac
lc() { echo "$1" | tr 'A-Z' 'a-z'; }
FAMX=(--fam-ci "$O/out/families/famci_C1_vs_C0.json" --fam-labels C0 C1)
"$PYT" "$FR" --rows "$O/out/pass1/rows_s*.jsonl" --f $F --c $C \
  --seam-f $D/seams/stage2_p5c_$(lc $F).npz --dump-f $D/seams/stage2_p5c_$(lc $F).proposals.npz \
  --seam-c $D/seams/stage2_p5c_$(lc $C).npz --dump-c $D/seams/stage2_p5c_$(lc $C).proposals.npz \
  --csv-f $D/score/refe_stage2_p5c_$(lc $F)/refe_stage2_p5c_$(lc $F).csv --csv-c $D/score/refe_stage2_p5c_$(lc $C)/refe_stage2_p5c_$(lc $C).csv \
  --seam-ledger $W/dry/seam_ledger_dry_$1.json --convert-report $O/plans/convert_report.json \
  --proptable-f "$N/full_${F}_s*of2.jsonl" --proptable-c "$N/full_${C}_s*of2.jsonl" --c3-tokens $W/pk/c3/tokens.json \
  "${FAMX[@]}" --strategic $O/out/strategic_family_C1_C0.json --tactical $O/out/tactical.json \
  --dry-run "$LAB" --out $W/dry/$OUT.json --text $W/dry/$OUT.txt
