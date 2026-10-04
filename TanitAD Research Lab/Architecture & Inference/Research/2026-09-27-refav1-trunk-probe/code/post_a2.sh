#!/usr/bin/env bash
# A2 post-processing: verdict per the committed bars + four families on the eval dumps. Artifact-checked.
set -u
cd C:/Users/Admin/refav1_probe || exit 2
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
T=C:/Users/Admin/tipsnap/b3f7ea6f
export PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6
test -s out_te/probe_te_result.json || { echo "ZZA2-NO-RESULTZZ"; exit 3; }
"$PY" verdict_a2.py out_te/probe_te_result.json > out_te/verdict_a2.txt 2>&1
grep -q "VERDICT:" out_te/verdict_a2.txt || { echo "ZZA2-VERDICT-FAILZZ"; exit 4; }
cd out_te && "$PY" "$T/taniteval/tools/refav1_paired_delta.py" --stack "$T/stack" --taniteval "$T/taniteval" \
  --dump P0=dump_P0_kdx --dump P2=dump_P2_kin --dump P4=dump_P4_trunk --dump P6=dump_P6_trunk_sp \
  --dump P7=dump_P7_raw_sp --dump P9=dump_P9_trunk_sp_mlp --dump P5=dump_P5_shuf --dump P8=dump_P8_shuf_sp \
  --pair P6-P2 --pair P4-P2 --pair P9-P2 --pair P6-P7 --pair P9-P6 --pair P2-P0 --pair P5-P2 --pair P8-P2 \
  --out pd_probe_te.json --md pd_probe_te.md > pd_probe_te.log 2>&1
test -s pd_probe_te.md || { echo "ZZA2-PD-FAILZZ"; exit 5; }
echo "ZZA2-POST-OKZZ"; tail -1 verdict_a2.txt
