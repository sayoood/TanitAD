#!/bin/bash
# MM 2026-11-28: L2-NEXT gate + train, both families, on the LANDED build fb686b3a (cards PREREG_EPHEAD / PREREG_SHMLP; CODE AMENDMENT 1).
# Serial; before each heavy step commit free >= 8,000 MB on 3 consecutive checks (TRK-SCORE's seam runs beside us).
# Data = SAFE-HEAD's sealed 35-log index (run1/index_sealed.csv, 2,272 tokens). Data paths have no spaces (gate_token.argv_from_string = s.split()).
set -u; LANDED=fb686b3aeac40d2bc712618dd492b72c5d0a62fe
export CUDA_VISIBLE_DEVICES=-1 TMP=E:/Projects/TanitAD-artifacts/l2next/tmp TEMP=E:/Projects/TanitAD-artifacts/l2next/tmp TMPDIR=E:/Projects/TanitAD-artifacts/l2next/tmp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TIP=E:/Projects/TanitAD-artifacts/l2next/tip; PKG="$TIP/$PKR"; RUN=E:/Projects/TanitAD-artifacts/l2next/run1; SH=E:/Projects/TanitAD-artifacts/safehead/run1
C1=F:/safehead_cache/ckpt/C1/model_final.pt; mkdir -p $TMP $TIP $RUN
[ "$(sha256sum $C1 | cut -c1-64)" = f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4 ] || { echo C1_SHA_BAD; exit 1; }
cf() { powershell -NoProfile -Command "\$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; [math]::Floor((\$m.CommitLimit-\$m.CommittedBytes)/1MB)" < /dev/null | tr -dc 0-9; }
gate() { local ok=0; while [ $ok -lt 3 ]; do c=$(cf); [ "${c:-0}" -ge 8000 ] && ok=$((ok+1)) || ok=0; sleep 20; done; }
git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive $LANDED "$PKR/refe" "$PKR/splits" "$PKR/code" "$PKR/eval" "$PKR/raw/2026-11-27-safehead" "$PKR/raw/2026-11-27-l2next" | tar -x -C $TIP || { echo EXTRACT_FAIL; exit 1; }
for f in refe/safehead_train.py refe/safehead_gate.py; do grep -q "AMENDMENT 1" "$PKG/$f" || { echo "NOT_AMENDED $f"; exit 1; }; done
cp $SH/index_sealed.csv $SH/navtest_logs.txt $SH/sft_heldout_logs.txt $RUN/
[ "$(sha256sum $RUN/sft_heldout_logs.txt | cut -c1-64)" = "$(sha256sum "$PKG/splits/sft_heldout_logs.txt" | cut -c1-64)" ] || { echo HL_SHA_BAD; exit 1; }
cd "$PKG/refe"
$PY launch_gate.py ship-manifest --pkg "$PKG" --commit $LANDED --git-dir C:/Users/Admin/tanitad-push/.git --repo-prefix "$PKR/" --out "$PKG/SHIP_MANIFEST.json" | tail -1
B="--cache-index $RUN/index_sealed.csv --init-from $C1 --heldout-logs $RUN/sft_heldout_logs.txt --navtest-logs $RUN/navtest_logs.txt --steps 2000 --lr 0.0003 --batch 32 --gamma 2.0 --save-steps 0,250,500,1000,2000"
declare -A FAM=([ephead]="$B --rows 2 --select cv5" [shmlp]="$B --rows 0,1,3,5 --mlp-hidden 64")
declare -A PFX=([ephead]=eph [shmlp]=shm)
for fam in ephead shmlp; do
  for arm in red s0 s1; do
    case $arm in s0) X="--seed 0";; s1) X="--seed 1";; red) X="--seed 0 --permute-labels-seed 20261127";; esac
    ARGS="${FAM[$fam]} --out $RUN/${PFX[$fam]}_$arm $X"
    gate; echo "== $fam $arm refusals $(date -u +%H:%M:%SZ)"
    $PY launch_gate.py refusals --profile refe_${fam}_$arm --args "$ARGS" | tail -1
    echo "== $fam $arm gate run $(date -u +%H:%M:%SZ)"
    $PY launch_gate.py run --profile refe_${fam}_$arm --pkg "$PKG" --args "$ARGS" --out-dir "$RUN/gate_${PFX[$fam]}_$arm" | tail -2
    TOK=$(ls "$RUN/gate_${PFX[$fam]}_$arm"/PASS_*.json 2>/dev/null | head -1); [ -n "$TOK" ] || { echo "NO_PASS_TOKEN $fam $arm"; exit 1; }
    gate; echo "== $fam $arm train $(date -u +%H:%M:%SZ)"
    TANITAD_SH_GATE_TOKEN="$TOK" $PY safehead_train.py $ARGS 2>&1 | grep -v "UserWarning\|Consider using\|rec = " | tail -2
  done
done
for d in eph_s0 eph_s1 eph_red; do echo "$d $(grep -o '"chosen_step"[^,}]*' $RUN/$d/cv_select.json 2>/dev/null)"; done
for d in shm_s0 shm_s1 shm_red; do echo "$d $(grep -o '"chosen_step"[^,}]*' $RUN/$d/heldout_select.json 2>/dev/null)"; done
echo L2NEXT_STEP2_DONE
