#!/bin/bash
# MM 2026-11-20: PREREG_P5C_NEXT.md §4 reads (frozen ac0e03f9) after ZZSRULE_TABLES_DONE.
cd /e/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next || exit 1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
S=E:/Projects/TanitAD/data/refe_navtest/seams
C=E:/Projects/TanitAD/data/refe_navtest/score
P1="E:/Projects/TanitAD-artifacts/p5c_read/out/pass1/rows_s*of2.jsonl"
declare -A NPZ=( [C1]=$S/stage2_p5c_c1.proposals.npz [C0]=$S/stage2_p5c_c0.proposals.npz [base]=$S/stage2_p7_full.proposals.npz )
declare -A CSV=( [C1]=$C/refe_stage2_p5c_c1/refe_stage2_p5c_c1.csv [C0]=$C/refe_stage2_p5c_c0/refe_stage2_p5c_c0.csv [base]=$C/refe_stage2_p7_full/refe_stage2_p7_full.csv )
for arm in C1 C0 base; do
  $PY p5c_prop_read_a1.py --rows "propscore/full_${arm}_s*of2.jsonl" --arm $arm --csv "${CSV[$arm]}" --pass1 "$P1" \
    --proposals "${NPZ[$arm]}" --out propscore/full_${arm}_read_a1.json --boot 10000 > propscore/full_${arm}_read_a1.out 2>&1
  echo "ZZREAD $arm rc=$? $(date -u +%H:%MZ)"; tail -n 4 propscore/full_${arm}_read_a1.out | cut -c1-300
done
$PY s_rule_read_a1.py --primary "C1=propscore/full_C1_s*of2.jsonl=${NPZ[C1]}" \
  --replicate "C0=propscore/full_C0_s*of2.jsonl=${NPZ[C0]}" "base=propscore/full_base_s*of2.jsonl=${NPZ[base]}" \
  --out propscore/S_RULE_READ_A1.json > propscore/S_RULE_READ_A1.out 2>&1
echo "ZZSRULE_READ rc=$? $(date -u +%H:%MZ)"; tail -n 30 propscore/S_RULE_READ_A1.out | cut -c1-400
