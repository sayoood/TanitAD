#!/bin/bash
# MM 2026-11-28: DATA-SCALE step 2 (PREREG_DATASCALE.md 2-5; card sha256 4e52b00e..., frozen). Waits for $RUN/SEAL.DONE (the 185-log index, sealed by seal_185.sh),
# then per lever (A ds_sh rows 0,1,3,5; B ds_ep row 2) and arm (red, s0, s1): launch_gate refusals + run -> PASS token -> safehead_train.py --select cv5m.
# Then M6 per lever (safehead_lc.py --sizes 35,70,110,N under its own PASS token; lever B carries CTRL-PARENT vs EPH_s0). No navtest number is read here.
# Serial; commit free >= 8,000 MB on 3 checks before each heavy step. Markers: $RUN/STEP2.DONE | $RUN/STEP2.FAILED. Launched via WMI (PATH set first).
export PATH="/usr/bin:/bin:/mingw64/bin:/c/Windows/System32:/c/Windows/System32/WindowsPowerShell/v1.0:$PATH"
set -u; LANDED=2b4409c56c092a458130bed6f1f79c8ed2948857
RUN=E:/Projects/TanitAD-artifacts/datascale/run1
export CUDA_VISIBLE_DEVICES=-1 TMP=$RUN/../tmp TEMP=$RUN/../tmp TMPDIR=$RUN/../tmp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TIP=C:/Users/Admin/dsrun_tip; PKG="$TIP/$PKR"; SH=E:/Projects/TanitAD-artifacts/safehead/run1
C1=F:/safehead_cache/ckpt/C1/model_final.pt; BANK=E:/Projects/TanitAD-artifacts/l2next/run1/eph_s0/cv_select.json; PL=$RUN/m6_35/parent_logs.txt
LOG=$RUN/STEP2.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
die() { echo "$1 $(date -u +%FT%TZ)" > $RUN/STEP2.FAILED; say "FAILED $1"; exit 1; }
cf() { powershell -NoProfile -Command "\$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; [math]::Floor((\$m.CommitLimit-\$m.CommittedBytes)/1MB)" < /dev/null | tr -dc 0-9; }
gate() { local ok=0; while [ $ok -lt 3 ]; do c=$(cf); [ "${c:-0}" -ge 8000 ] && ok=$((ok+1)) || ok=0; sleep 20; done; }
mkdir -p $RUN/../tmp
say "start pid $$; waiting for $RUN/SEAL.DONE"
until [ -f $RUN/SEAL.DONE ]; do [ -f $RUN/SEAL.FAILED ] && die "SEAL.FAILED present: $(cat $RUN/SEAL.FAILED)"; sleep 300; done
say "SEAL.DONE seen: $(cat $RUN/SEAL.DONE)"
[ "$(sha256sum $C1 | cut -c1-64)" = f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4 ] || die C1_SHA_BAD
mkdir -p $TIP
git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive $LANDED "$PKR/refe" "$PKR/splits" "$PKR/code" "$PKR/eval" "$PKR/raw/2026-11-27-safehead" "$PKR/raw/2026-11-27-l2next" "$PKR/raw/2026-11-28-datascale" | tar -x -C $TIP || die EXTRACT_FAIL
( cd "$PKG" && sha256sum -c --quiet raw/2026-11-28-datascale/CODE.sha256 ) > $RUN/step2_code_sha.log 2>&1 || die "CODE.sha256 mismatch"
[ "$(cut -c1-64 "$PKG/raw/2026-11-28-datascale/PREREG_DATASCALE.sha256")" = 4e52b00e896e7e2cd1b17e374131c8f838d384ed749e3b59c5246c10a0f6dfbd ] || die CARD_SHA_BAD
cp "$PKG/splits/sft_heldout_logs.txt" $SH/navtest_logs.txt $RUN/
IDX=$RUN/index_185_sealed.csv; [ -f $IDX ] || die "no $IDX"
cd "$PKG/refe"
$PY launch_gate.py ship-manifest --pkg "$PKG" --commit $LANDED --git-dir C:/Users/Admin/tanitad-push/.git --repo-prefix "$PKR/" --out "$PKG/SHIP_MANIFEST.json" >> $LOG 2>&1 || die ship_manifest
B="--cache-index $IDX --init-from $C1 --heldout-logs $RUN/sft_heldout_logs.txt --navtest-logs $RUN/navtest_logs.txt --steps 2000 --lr 0.0003 --batch 32 --gamma 2.0 --save-steps 0,250,500,1000,2000"
declare -A FAM=([sh]="$B --rows 0,1,3,5 --select cv5m" [ep]="$B --rows 2 --select cv5m")
declare -A PFX=([sh]=dsh [ep]=dep)
gate_tok() {   # $1 profile  $2 args  $3 gate dir  -> echoes the PASS token path
  $PY launch_gate.py refusals --profile $1 --args "$2" >> $LOG 2>&1
  $PY launch_gate.py run --profile $1 --pkg "$PKG" --args "$2" --out-dir "$3" > "$3.log" 2>&1
  ls "$3"/PASS_*.json 2>/dev/null | head -1
}
for fam in sh ep; do
  for arm in red s0 s1; do
    [ -f $RUN/${PFX[$fam]}_$arm/run.json ] && { say "skip ${PFX[$fam]}_$arm (trained)"; continue; }
    case $arm in s0) X="--seed 0";; s1) X="--seed 1";; red) X="--seed 0 --permute-labels-seed 20261127";; esac
    ARGS="${FAM[$fam]} --out $RUN/${PFX[$fam]}_$arm $X"
    gate; say "gate refe_ds_${fam}_$arm"
    TOK=$(gate_tok refe_ds_${fam}_$arm "$ARGS" "$RUN/gate_${PFX[$fam]}_$arm"); [ -n "$TOK" ] || die "NO_PASS_TOKEN refe_ds_${fam}_$arm ($(tail -2 $RUN/gate_${PFX[$fam]}_$arm.log | tr '\n' ' ' | cut -c1-300))"
    gate; say "train ${PFX[$fam]}_$arm"
    TANITAD_SH_GATE_TOKEN="$TOK" $PY safehead_train.py $ARGS > $RUN/train_${PFX[$fam]}_$arm.log 2>&1 || die "train ${PFX[$fam]}_$arm rc=$?"
    say "  $(grep -v 'UserWarning\|Consider using\|rec = ' $RUN/train_${PFX[$fam]}_$arm.log | tail -1 | cut -c1-250)"
  done
done
for fam in sh ep; do
  [ -f $RUN/lc_$fam/lc.json ] && { say "skip lc_$fam (done)"; continue; }
  ARGS="${FAM[$fam]} --out $RUN/lc_$fam --seed 0"
  gate; say "gate refe_ds_${fam}_s0 for M6"
  TOK=$(gate_tok refe_ds_${fam}_s0 "$ARGS" "$RUN/gate_lc_$fam"); [ -n "$TOK" ] || die "NO_PASS_TOKEN M6 $fam"
  PC=""; [ $fam = ep ] && PC="--parent-cv-select $BANK"
  gate; say "M6 lc_$fam"
  TANITAD_SH_GATE_TOKEN="$TOK" $PY safehead_lc.py --parent-logs $PL --sizes 35,70,110,N $PC $ARGS > $RUN/lc_$fam.log 2>&1 || die "lc_$fam rc=$?"
  say "  $(grep -v 'UserWarning\|Consider using\|rec = ' $RUN/lc_$fam.log | tail -5 | tr '\n' ' ' | cut -c1-600)"
done
for d in dsh_s0 dsh_s1 dsh_red dep_s0 dep_s1 dep_red; do say "$d chosen $(grep -o '"chosen_step"[^,}]*' $RUN/$d/cv_select.json 2>/dev/null | head -1)"; done
date -u +%FT%TZ > $RUN/STEP2.DONE; say "STEP2 DONE"
