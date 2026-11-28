#!/bin/bash
# usage: bash step12.sh <LANDED 40-hex that CONTAINS the safehead_data.py memory fix>   (steps 1-2 of the HANDOFF; run ONE heavy process at a time, commit free >= 6000 MB)
set -u; LANDED=$1
export TMP=F:/TanitAD_safehead_tmp TEMP=F:/TanitAD_safehead_tmp TMPDIR=F:/TanitAD_safehead_tmp CUDA_VISIBLE_DEVICES=-1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TIP=F:/TanitAD_safehead_tip; PKG="$TIP/$PKR"; RUN=E:/Projects/TanitAD-artifacts/safehead/run1; AD=F:/TanitAD_safehead_tmp/adapted
C1=F:/safehead_cache/ckpt/C1/model_final.pt
[ "$(sha256sum $C1 | cut -c1-64)" = f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4 ] || { echo C1_SHA_BAD; exit 1; }
cmf() { powershell -c "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1024)"; }
git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive $LANDED "$PKR/refe" "$PKR/splits" "$PKR/code" "$PKR/eval" "$PKR/raw/2026-11-27-safehead" | tar -x -C $TIP
cd "$PKG/refe"
$PY launch_gate.py ship-manifest --pkg "$PKG" --commit $LANDED --git-dir C:/Users/Admin/tanitad-push/.git --repo-prefix "$PKR/" --out "$PKG/SHIP_MANIFEST.json" | tail -1
$PY ../raw/2026-11-27-safehead/safehead_read.py prep --table "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/propscore/full_C1_s*of2.jsonl" --out "$RUN/navtest_logs.txt" | tail -1
mkdir -p $AD; $PY $RUN/scripts/adapt_index.py F:/safehead_cache/index.csv out $AD/index.csv
$PY safehead_data.py check --index $AD/index.csv --heldout-logs ../splits/sft_heldout_logs.txt --navtest-logs "$RUN/navtest_logs.txt" | tail -1
$PY safehead_data.py seal --index $AD/index.csv --out "$RUN/index_sealed.csv" | tail -1
A="--cache-index $RUN/index_sealed.csv --init-from $C1 --heldout-logs $PKG/splits/sft_heldout_logs.txt --navtest-logs $RUN/navtest_logs.txt --steps 2000 --lr 0.0003 --batch 32 --gamma 2.0 --rows 0,1,3,5 --save-steps 0,250,500,1000,2000"
declare -A ARGS=([s0]="$A --out $RUN/sh_s0 --seed 0" [s1]="$A --out $RUN/sh_s1 --seed 1" [red]="$A --out $RUN/sh_red --seed 0 --permute-labels-seed 20261127")
for arm in red s0 s1; do
  until [ "$(cmf)" -ge 6000 ]; do sleep 30; done
  $PY launch_gate.py refusals --profile refe_safehead_$arm --args "${ARGS[$arm]}" | tail -1
  $PY launch_gate.py run --profile refe_safehead_$arm --pkg "$PKG" --args "${ARGS[$arm]}" --out-dir "$RUN/gate_$arm" | tail -2
  TOK=$(ls "$RUN/gate_$arm"/PASS_*.json) || { echo "NO_PASS_TOKEN $arm"; exit 1; }
  until [ "$(cmf)" -ge 6000 ]; do sleep 30; done
  TANITAD_SH_GATE_TOKEN="$TOK" $PY safehead_train.py ${ARGS[$arm]} | tail -1
done
grep -o '"chosen_step"[^,]*' $RUN/sh_s0/heldout_select.json
