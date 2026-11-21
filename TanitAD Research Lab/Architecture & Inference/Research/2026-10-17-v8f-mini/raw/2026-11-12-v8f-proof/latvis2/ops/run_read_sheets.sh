#!/usr/bin/env bash
# run_read_sheets.sh : card PREREG_V8F_PROOF_ADD38 with notes N1 and N2. Dev box, CPU. Reads G and RR with k5 in parts, the pair, then draws the 18 sheets.
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
H=/f/TanitAD-scratch/v8proof/latvis2; D=$H/dump; R=$H/read_n2
REF=C:/Users/Admin/v8proof_c/ctxroll/LATENTS_CTX_G_s0_R1.npz
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
cd $H || exit 1; mkdir -p "$R"
ts() { date -u +%H:%M:%SZ; }
for A in G RR; do
  "$PY" lv2_read.py read --dump "$D/DUMP_LV2_${A}_s0.npz" "$D/DUMP_LV2_${A}_s1.npz" --arm $A --seeds 0,1 --out-dir "$R/$A" --ref-windows "$REF" \
      --k5-parts ops/k5_probe_${A}_s0.out ops/k5_probe_${A}_s1.out > "$R/$A.log" 2>&1
  echo "[$(ts)] read $A rc $? : $(tail -n 1 "$R/$A.log" | cut -c1-300)"
done
"$PY" lv2_read.py pair --a-dir "$R/RR" --b-dir "$R/G" --out-dir "$R/pair" > "$R/pair.log" 2>&1
echo "[$(ts)] pair RR minus G rc $? : $(tail -n 1 "$R/pair.log" | cut -c1-300)"
"$PY" lv2_sheet.py --windows WINDOWS_LV2.json --out-dir . --tok G_s0="$D/DUMP_LV2_G_s0.npz" --tok RR_s0="$D/DUMP_LV2_RR_s0.npz" --imag LS_s0=../imaghead/dump/DUMP_LS_s0.npz --frames-dir private/frames > "$R/sheets.log" 2>&1
echo "[$(ts)] sheets rc $? : $(tail -n 1 "$R/sheets.log" | cut -c1-200); private $(ls private/sheets/*.png 2>/dev/null | wc -l) public $(ls sheets_public/*.png 2>/dev/null | wc -l)"
echo "ZZ LV2 N2 READS AND SHEETS DONE $(ts)"
