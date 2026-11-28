#!/bin/bash
# MM 2026-11-28: SH-MLP controls-only read. All three arms chose step 0 (NULL-HELDOUT); the outcome function returns INVALID if a
# control fails, so the read runs for its controls. No families: SHM_s0 at step 0 IS C1. Runs after the EP-HEAD table read (serial CPU).
export CUDA_VISIBLE_DEVICES=-1 TMP=E:/Projects/TanitAD-artifacts/l2next/tmp TEMP=E:/Projects/TanitAD-artifacts/l2next/tmp TMPDIR=E:/Projects/TanitAD-artifacts/l2next/tmp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TIP=E:/Projects/TanitAD-artifacts/l2next/tip; PKG="$TIP/$PKR"; RUN=E:/Projects/TanitAD-artifacts/l2next/run1; SH=E:/Projects/TanitAD-artifacts/safehead/run1
TAB="E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/propscore/full_C1_s*of2.jsonl"
until [ -f $RUN/read_eph/READ_EPHEAD.json ] || [ -f $RUN/EPH_READ.FAILED ]; do sleep 60; done
cf() { powershell -NoProfile -Command "\$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; [math]::Floor((\$m.CommitLimit-\$m.CommittedBytes)/1MB)" < /dev/null | tr -dc 0-9; }
ok=0; while [ $ok -lt 3 ]; do c=$(cf); [ "${c:-0}" -ge 8000 ] && ok=$((ok+1)) || ok=0; sleep 20; done
cd "$PKG/raw/2026-11-27-l2next"
$PY l2next_read.py read --lever shmlp --table "$TAB" --proposals E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.proposals.npz --seam E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.npz \
  --navtest-index "$SH/nt_sealed.csv" --train-index "$RUN/index_sealed.csv" --heldout-logs "$PKG/splits/sft_heldout_logs.txt" \
  --run-s0 "$RUN/shm_s0" --run-s1 "$RUN/shm_s1" --run-red "$RUN/shm_red" --eval-dir "$PKG/eval" --refe "$PKG/refe" \
  --card PREREG_SHMLP.md --card-sha256 $(cut -c1-64 PREREG_SHMLP.sha256) --out "$RUN/read_shm" > $RUN/read_shm.log 2>&1
echo "rc=$? $(tail -1 $RUN/read_shm.log)" > $RUN/SHM_READ.DONE
