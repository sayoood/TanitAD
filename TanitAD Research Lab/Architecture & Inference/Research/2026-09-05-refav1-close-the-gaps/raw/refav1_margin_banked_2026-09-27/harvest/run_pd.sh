#!/usr/bin/env bash
set -u
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
T=C:/Users/Admin/tipsnap/b3f7ea6f
P=C:/Users/Admin/refav1_margin/p4out
O=C:/Users/Admin/refav1_harvest
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
D=(); for a in wk15 lonshift lonshift_s1 loncomb3 seambase seamon kamm07 kammshift wk7 best best_seed1 combined combined_seed1 gkappa; do D+=(--dump "$a=$P/dump_$a"); done
"$PY" "$T/taniteval/tools/refav1_paired_delta.py" --stack "$T/stack" --taniteval "$T/taniteval" "${D[@]}" \
  --pair loncomb3-lonshift --pair lonshift_s1-lonshift --pair seamon-seambase --pair kammshift-kamm07 \
  --pair wk7-wk15 --pair best_seed1-best --pair combined_seed1-combined --pair gkappa-wk15 --pair lonshift-wk15 \
  --out "$O/pd_p4.json" --md "$O/pd_p4.md" > "$O/pd_p4.log" 2>&1
[ -s "$O/pd_p4.md" ] && echo "ZZPD-P4-OK" || echo "ZZPD-P4-FAIL"
D=(); for a in ta_wk15_s0 ta_wk15_s1 ta_ccos_s0 ta_ccos_s1; do D+=(--dump "$a=$P/dump_$a"); done
"$PY" "$T/taniteval/tools/refav1_paired_delta.py" --stack "$T/stack" --taniteval "$T/taniteval" "${D[@]}" \
  --pair ta_wk15_s1-ta_wk15_s0 --pair ta_ccos_s1-ta_ccos_s0 --pair ta_wk15_s0-ta_ccos_s0 \
  --out "$O/pd_ta.json" --md "$O/pd_ta.md" > "$O/pd_ta.log" 2>&1
[ -s "$O/pd_ta.md" ] && echo "ZZPD-TA-OK" || echo "ZZPD-TA-FAIL"
