#!/bin/bash
# MM 2026-11-27: L2 step 3 ONLY, for the NULL-HELDOUT case (s0's held-out rule chose step 0 at 17:40Z).
# The card's outcome function (safehead_read.py:89-95) returns INVALID if any control fails and NULL-HELDOUT otherwise,
# so the read is run for its CONTROLS. Families are not run: SH_s0 at step 0 IS C1 (identical picks), so every
# family delta for the decision arm is zero by construction. chain34.sh refuses this case; this replaces it.
export TMP=F:/TanitAD_safehead_tmp TEMP=F:/TanitAD_safehead_tmp CUDA_VISIBLE_DEVICES=-1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TIP=F:/TanitAD_safehead_tip; PKG="$TIP/$PKR"; RUN=E:/Projects/TanitAD-artifacts/safehead/run1; M=F:/safehead_cache/markers
TAB="E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/propscore/full_C1_s*of2.jsonl"
cmf() { powershell -c "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1024)"; }
T0=$(date +%s)
until [ -f $M/navtest.DONE ] || [ -f $M/navtest.FAILED ]; do [ $(( $(date +%s)-T0 )) -gt 86400 ] && { echo giveup > $RUN/READ_BLOCKED.txt; exit 1; }; sleep 60; done
[ -f $M/navtest.FAILED ] && { echo "navtest.FAILED present" > $RUN/READ_BLOCKED.txt; exit 1; }
for a in s0 s1 red; do [ -f $RUN/sh_$a/heldout_select.json ] || { echo "sh_$a missing" > $RUN/READ_BLOCKED.txt; exit 1; }; done
until [ "$(cmf)" -ge 6000 ]; do sleep 30; done
mkdir -p F:/TanitAD_safehead_tmp/adapted_nt
$PY $RUN/scripts/adapt_index.py F:/safehead_cache/navtest/index.csv navtest F:/TanitAD_safehead_tmp/adapted_nt/index.csv || { echo "adapt failed" > $RUN/READ_BLOCKED.txt; exit 1; }
cd "$PKG/refe"; $PY safehead_data.py seal --index F:/TanitAD_safehead_tmp/adapted_nt/index.csv --out "$RUN/nt_sealed.csv" || { echo "seal failed" > $RUN/READ_BLOCKED.txt; exit 1; }
cd "$PKG/raw/2026-11-27-safehead"
$PY safehead_read.py read --table "$TAB" --proposals E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.proposals.npz --seam E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.npz \
  --navtest-index "$RUN/nt_sealed.csv" --train-index "$RUN/index_sealed.csv" --heldout-logs "$PKG/splits/sft_heldout_logs.txt" \
  --run-s0 "$RUN/sh_s0" --run-s1 "$RUN/sh_s1" --run-red "$RUN/sh_red" --eval-dir "$PKG/eval" --refe "$PKG/refe" \
  --card PREREG_SAFEHEAD.md --card-sha256 $(cut -c1-64 PREREG_SAFEHEAD.sha256) --out "$RUN/read" > $RUN/read.log 2>&1 || { echo "read failed rc=$?" > $RUN/READ_BLOCKED.txt; exit 1; }
echo "read done $(date -u +%H:%M:%SZ)" > $RUN/READ.DONE
