#!/bin/bash
# MM 2026-11-28: DATA-SCALE real-data shakedown = the card's own M6 point n = 35 (PREREG_DATASCALE.md 5 M6 + 6 CTRL-PARENT), run a day early on
# the PARENT's sealed 35-log index (CTRL-INDEX makes those rows bit-identical in index_185). Lever B (row 2, refe_ds_ep_s0 profile), seed 0.
# Gate first (launch_gate.py refusals + run -> PASS token bound to this argv), then safehead_lc.py under the token (CODE AMENDMENT 1).
# CTRL-PARENT must reproduce EPH_s0's banked cv_select.json: no-margin argmax step 500 and OOF PDMS at every step to 1e-9.
# The cv5m choice at n = 35 is descriptive M6, NOT a decision; no navtest number is read.
export PATH="/usr/bin:/bin:/mingw64/bin:/c/Windows/System32:/c/Windows/System32/WindowsPowerShell/v1.0:$PATH"
set -u; LANDED=5850172b11762b854debcc32455016265647469f   # CODE AMENDMENT 2 (attempt 2 refused at G-SUITE: the test minted with a fake commit)
export CUDA_VISIBLE_DEVICES=-1 TMP=C:/Users/Admin/dsm6/tmp TEMP=C:/Users/Admin/dsm6/tmp TMPDIR=C:/Users/Admin/dsm6/tmp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TIP=C:/Users/Admin/dsm6/tip; PKG="$TIP/$PKR"; RUN=E:/Projects/TanitAD-artifacts/datascale/run1/m6_35; SH=E:/Projects/TanitAD-artifacts/safehead/run1
C1=F:/safehead_cache/ckpt/C1/model_final.pt; BANK=E:/Projects/TanitAD-artifacts/l2next/run1/eph_s0/cv_select.json
mkdir -p $TMP $TIP $RUN; for n in 1 2 3 4 5 6; do [ -f $RUN/M6_35.FAILED.attempt$n ] || [ -f $RUN/M6_35.FAILED.attempt${n}_codesha_cwd ] || { [ -f $RUN/M6_35.FAILED ] && mv $RUN/M6_35.FAILED $RUN/M6_35.FAILED.attempt$n; break; }; done; [ -d $RUN/gate ] && mv $RUN/gate $RUN/gate.attempt_$(date -u +%H%M%S)
die() { echo "$1 $(date -u +%FT%TZ)" > $RUN/M6_35.FAILED; echo "ZZM6_FAIL $1"; exit 1; }
[ "$(sha256sum $C1 | cut -c1-64)" = f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4 ] || die C1_SHA_BAD
cf() { powershell -NoProfile -Command "\$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; [math]::Floor((\$m.CommitLimit-\$m.CommittedBytes)/1MB)" < /dev/null | tr -dc 0-9; }
gate() { local ok=0; while [ $ok -lt 3 ]; do c=$(cf); [ "${c:-0}" -ge 8000 ] && ok=$((ok+1)) || ok=0; sleep 20; done; }
git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive $LANDED "$PKR/refe" "$PKR/splits" "$PKR/code" "$PKR/eval" "$PKR/raw/2026-11-27-safehead" "$PKR/raw/2026-11-27-l2next" "$PKR/raw/2026-11-28-datascale" | tar -x -C $TIP || die EXTRACT_FAIL
grep -q "AMENDMENT 1" "$PKG/refe/safehead_lc.py" || die NOT_AMENDED
( cd "$PKG" && sha256sum -c --quiet raw/2026-11-28-datascale/CODE.sha256 ) > $RUN/code_sha.log 2>&1 || die "CODE.sha256 mismatch"
[ "$(cut -c1-64 "$PKG/raw/2026-11-28-datascale/PREREG_DATASCALE.sha256")" = 4e52b00e896e7e2cd1b17e374131c8f838d384ed749e3b59c5246c10a0f6dfbd ] || die CARD_SHA_BAD
cp $SH/index_sealed.csv $SH/navtest_logs.txt $SH/sft_heldout_logs.txt $RUN/
[ "$(sha256sum $RUN/sft_heldout_logs.txt | cut -c1-64)" = "$(sha256sum "$PKG/splits/sft_heldout_logs.txt" | cut -c1-64)" ] || die HL_SHA_BAD
$PY -c "import csv,sys; L=sorted({r['log'] for r in csv.DictReader(open(sys.argv[1]))}); open(sys.argv[2],'w').write(chr(10).join(L)+chr(10)); print(len(L),'parent logs')" $RUN/index_sealed.csv $RUN/parent_logs.txt
cd "$PKG/refe"
$PY launch_gate.py ship-manifest --pkg "$PKG" --commit $LANDED --git-dir C:/Users/Admin/tanitad-push/.git --repo-prefix "$PKR/" --out "$PKG/SHIP_MANIFEST.json" | tail -1
ARGS="--cache-index $RUN/index_sealed.csv --init-from $C1 --heldout-logs $RUN/sft_heldout_logs.txt --navtest-logs $RUN/navtest_logs.txt --steps 2000 --lr 0.0003 --batch 32 --gamma 2.0 --save-steps 0,250,500,1000,2000 --rows 2 --select cv5m --out $RUN/lc --seed 0"
gate; echo "== refusals $(date -u +%H:%M:%SZ)"
$PY launch_gate.py refusals --profile refe_ds_ep_s0 --args "$ARGS" | tail -1
echo "== gate run $(date -u +%H:%M:%SZ)"
$PY launch_gate.py run --profile refe_ds_ep_s0 --pkg "$PKG" --args "$ARGS" --out-dir "$RUN/gate" > $RUN/gate.log 2>&1; tail -2 $RUN/gate.log
TOK=$(ls "$RUN/gate"/PASS_*.json 2>/dev/null | head -1); [ -n "$TOK" ] || die NO_PASS_TOKEN
gate; echo "== lc n=35 $(date -u +%H:%M:%SZ)"
TANITAD_SH_GATE_TOKEN="$TOK" $PY safehead_lc.py --parent-logs $RUN/parent_logs.txt --sizes 35 --parent-cv-select $BANK $ARGS > $RUN/lc.log 2>&1
rc=$?; grep -v "UserWarning\|Consider using\|rec = " $RUN/lc.log | tail -4
[ $rc = 0 ] || die "lc rc=$rc"
date -u +%FT%TZ > $RUN/M6_35.DONE; echo ZZM6_DONE
