#!/bin/bash
# step2b.sh: step 2 only (gate + train per arm), after step12.sh sealed the index at 63a4af88.
# Why: gate_token.argv_from_string = s.split(); the PKG heldout path has spaces. The heldout list is
# copied to $RUN/sft_heldout_logs.txt (sha256 b814998d... = repo blob at 63a4af88). The gate hashes
# data flags by CONTENT (safehead_gate.py:217), so the path change moves no bound value.
set -u
export TMP=F:/TanitAD_safehead_tmp TEMP=F:/TanitAD_safehead_tmp TMPDIR=F:/TanitAD_safehead_tmp CUDA_VISIBLE_DEVICES=-1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TIP=F:/TanitAD_safehead_tip; PKG="$TIP/$PKR"; RUN=E:/Projects/TanitAD-artifacts/safehead/run1
C1=F:/safehead_cache/ckpt/C1/model_final.pt
HL=$RUN/sft_heldout_logs.txt
[ "$(sha256sum $HL | cut -c1-20)" = "$(sha256sum "$PKG/splits/sft_heldout_logs.txt" | cut -c1-20)" ] || { echo HL_SHA_BAD; exit 1; }
[ -f $RUN/index_sealed.csv ] || { echo NO_SEALED_INDEX; exit 1; }
cmf() { powershell -c "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1024)"; }
cd "$PKG/refe"
A="--cache-index $RUN/index_sealed.csv --init-from $C1 --heldout-logs $HL --navtest-logs $RUN/navtest_logs.txt --steps 2000 --lr 0.0003 --batch 32 --gamma 2.0 --rows 0,1,3,5 --save-steps 0,250,500,1000,2000"
declare -A ARGS=([s0]="$A --out $RUN/sh_s0 --seed 0" [s1]="$A --out $RUN/sh_s1 --seed 1" [red]="$A --out $RUN/sh_red --seed 0 --permute-labels-seed 20261127")
for arm in red s0 s1; do
  until [ "$(cmf)" -ge 6000 ]; do sleep 30; done
  echo "== $arm refusals $(date -u +%H:%M:%SZ)"
  $PY launch_gate.py refusals --profile refe_safehead_$arm --args "${ARGS[$arm]}" | tail -1
  echo "== $arm gate run $(date -u +%H:%M:%SZ)"
  $PY launch_gate.py run --profile refe_safehead_$arm --pkg "$PKG" --args "${ARGS[$arm]}" --out-dir "$RUN/gate_$arm" | tail -3
  TOK=$(ls "$RUN/gate_$arm"/PASS_*.json 2>/dev/null | head -1)
  [ -n "$TOK" ] || { echo "NO_PASS_TOKEN $arm"; exit 1; }
  until [ "$(cmf)" -ge 6000 ]; do sleep 30; done
  echo "== $arm train $(date -u +%H:%M:%SZ)"
  TANITAD_SH_GATE_TOKEN="$TOK" $PY safehead_train.py ${ARGS[$arm]} | tail -2
  echo "== $arm rc=$? $(date -u +%H:%M:%SZ)"
done
grep -o '"chosen_step"[^,}]*' $RUN/sh_s0/heldout_select.json $RUN/sh_s1/heldout_select.json $RUN/sh_red/heldout_select.json
echo STEP2B_DONE
