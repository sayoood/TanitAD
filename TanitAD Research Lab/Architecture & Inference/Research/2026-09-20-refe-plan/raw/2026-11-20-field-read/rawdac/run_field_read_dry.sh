#!/usr/bin/env bash
set -u
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
H=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_read
PYT=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 FIELD_READ_P5C_DIR="F:/TanitAD_p5cread_tip/$PK/eval/p5c"
D=E:/Projects/TanitAD/data/refe_navtest; O=E:/Projects/TanitAD-artifacts/p5c_read; N=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/propscore
FR="$H/read/land/$PK/eval/field/field_read.py"; R=$H/rawdac
C3="E:/Projects/TanitAD/$PK/raw/2026-10-11-c3-selection-headroom/tokens.json"
case "$1" in
  pair) F=C1; C=C0; OUT=dry_C1_vs_C0_rawdac; LAB="C1 vs C0 as stand-ins for F vs C";;
  self) F=C1; C=C1; OUT=dry_selfread_C1_vs_C1_rawdac; LAB="CTRL-ID: C1 read against itself";;
  *) echo bad; exit 2;;
esac
lc() { echo "$1" | tr 'A-Z' 'a-z'; }
"$PYT" "$FR" --rows "$O/out/pass1/rows_s*.jsonl" --f $F --c $C \
  --seam-f $D/seams/stage2_p5c_$(lc $F).npz --dump-f $D/seams/stage2_p5c_$(lc $F).proposals.npz \
  --seam-c $D/seams/stage2_p5c_$(lc $C).npz --dump-c $D/seams/stage2_p5c_$(lc $C).proposals.npz \
  --csv-f $D/score/refe_stage2_p5c_$(lc $F)/refe_stage2_p5c_$(lc $F).csv --csv-c $D/score/refe_stage2_p5c_$(lc $C)/refe_stage2_p5c_$(lc $C).csv \
  --seam-ledger C:/Users/Admin/swp/rawdac/seam_ledger_dry_$1.json --convert-report $O/plans/convert_report.json \
  --proptable-f "$N/full_${F}_s*of2.jsonl" --proptable-c "$N/full_${C}_s*of2.jsonl" --c3-tokens "$C3" \
  --rawdac-f "$R/rawdac_${F}_s*of2.jsonl" --rawdac-c "$R/rawdac_${C}_s*of2.jsonl" \
  --fam-ci "$O/out/families/famci_C1_vs_C0.json" --fam-labels C0 C1 --strategic $O/out/strategic_family_C1_C0.json --tactical $O/out/tactical.json \
  --dry-run "$LAB" --out $R/$OUT.json --text $R/$OUT.txt
