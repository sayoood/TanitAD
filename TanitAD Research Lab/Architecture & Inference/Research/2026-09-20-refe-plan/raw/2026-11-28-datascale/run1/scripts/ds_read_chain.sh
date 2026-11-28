#!/bin/bash
# MM 2026-11-28: DATA-SCALE DECISION READ (PREREG_DATASCALE.md, card sha256 4e52b00e..., FROZEN; AMENDMENT 1 plumbing check PASS at bbef7a7a).
# Waits for $RUN/STEP2.DONE (ds_step2.sh: dsh_/dep_{red,s0,s1} trained under PASS tokens + M6). Then per lever, in card order (A ds_sh, B ds_ep):
#   1. ds_read.py read   -> READ_<TAG>.json (every control, M1-M3, the PROVISIONAL outcome) + seams/ for the families chain.
#      A failed control = INVALID: stop and report; never re-read with a changed rule.
#   2. outcome NULL-CV   -> card 4: "the arm is C1 and navtest is not read for a decision". Controls-only; no families, no confirm.
#   3. otherwise         -> run_fam_chain_ds.sh SMOKE=50 (its first run in this parameterised form) -> full -> ds_read.py confirm --table (VERDICT_<TAG>.json).
# CPU only (CUDA_VISIBLE_DEVICES=-1); commit free >= 8,000 MB on 3 checks before every heavy step. One lever at a time.
# Markers: $RUN/DSREAD.DONE | $RUN/DSREAD.FAILED; summary $RUN/DSREAD_SUMMARY.json; log $RUN/DSREAD.log. Launched via WMI (PATH set first).
# Tip export (own temp, re-creatable from git): E:/Projects/TanitAD-artifacts/datascale/readtip -- deleted by literal path on DONE.
export PATH="/usr/bin:/bin:/mingw64/bin:/c/Windows/System32:/c/Windows/System32/WindowsPowerShell/v1.0:$PATH"
set -u; LANDED=151700901f29bb247692794926eb48342fa33377
RUN=E:/Projects/TanitAD-artifacts/datascale/run1
export CUDA_VISIBLE_DEVICES=-1 TMP=$RUN/../tmp TEMP=$RUN/../tmp TMPDIR=$RUN/../tmp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TIP=E:/Projects/TanitAD-artifacts/datascale/readtip; PKG="$TIP/$PKR"; SH=E:/Projects/TanitAD-artifacts/safehead/run1
TAB="E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/propscore/full_C1_s*of2.jsonl"
SEAMS=E:/Projects/TanitAD/data/refe_navtest/seams; LOG=$RUN/DSREAD.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
die() { echo "$1 $(date -u +%FT%TZ)" > $RUN/DSREAD.FAILED; say "FAILED $1"; exit 1; }
cf() { powershell -NoProfile -Command "\$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; [math]::Floor((\$m.CommitLimit-\$m.CommittedBytes)/1MB)" < /dev/null | tr -dc 0-9; }
gate() { local ok=0; while [ $ok -lt 3 ]; do c=$(cf); [ "${c:-0}" -ge 8000 ] && ok=$((ok+1)) || ok=0; sleep 20; done; }
mkdir -p $RUN/../tmp
say "start pid $$; waiting for $RUN/STEP2.DONE"
until [ -f $RUN/STEP2.DONE ]; do [ -f $RUN/STEP2.FAILED ] && die "STEP2.FAILED present: $(cat $RUN/STEP2.FAILED)"; sleep 300; done
say "STEP2.DONE seen: $(cat $RUN/STEP2.DONE)"
for d in dsh_red dsh_s0 dsh_s1 dep_red dep_s0 dep_s1; do [ -f $RUN/$d/run.json ] || die "no $RUN/$d/run.json"; done
[ -f $RUN/index_185_sealed.csv ] || die "no index_185_sealed.csv"
mkdir -p $TIP
git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive $LANDED stack/tanitad taniteval ':(exclude)taniteval/results' \
  "$PKR/refe" "$PKR/splits" "$PKR/code" "$PKR/eval" "$PKR/raw/2026-11-12-instruments/fam_ci" "$PKR/raw/2026-11-27-safehead" "$PKR/raw/2026-11-27-l2next" "$PKR/raw/2026-11-28-datascale" \
  | tar -x -C $TIP || die EXTRACT_FAIL
[ -f "$TIP/taniteval/adapters/navsim.py" ] || die "tip lacks taniteval/adapters/navsim.py"
( cd "$PKG" && sha256sum -c --quiet raw/2026-11-28-datascale/CODE.sha256 ) > $RUN/dsread_code_sha.log 2>&1 || die "CODE.sha256 mismatch"
[ "$(cut -c1-64 "$PKG/raw/2026-11-28-datascale/PREREG_DATASCALE.sha256")" = 4e52b00e896e7e2cd1b17e374131c8f838d384ed749e3b59c5246c10a0f6dfbd ] || die CARD_SHA_BAD
[ "$(cut -c1-64 "$PKG/raw/2026-11-28-datascale/AMENDMENT_1.sha256")" = 09444131e621622c09a575b9dc34a7ef6f25b2fa165fa3c688dc3190783985ab ] || die AMEND_SHA_BAD
FAMSH="$PKG/raw/2026-11-28-datascale/run_fam_chain_ds.sh"
for spec in "ds_sh dsh DSSH DSH0 DSH1" "ds_ep dep DSEP DEP0 DEP1"; do
  set -- $spec; LV=$1; P=$2; TAG=$3; A0=$4; A1=$5; OUT=$RUN/read_$LV
  if [ -f $OUT/LEVER.DONE ]; then say "$LV already read: $(cat $OUT/LEVER.DONE)"; continue; fi
  mkdir -p $OUT; gate; say "read $LV"
  cd "$PKG/raw/2026-11-28-datascale"
  $PY ds_read.py read --lever $LV --table "$TAB" --proposals $SEAMS/stage2_p5c_c1.proposals.npz --seam $SEAMS/stage2_p5c_c1.npz \
    --navtest-index "$SH/nt_sealed.csv" --train-index "$RUN/index_185_sealed.csv" --heldout-logs "$PKG/splits/sft_heldout_logs.txt" \
    --run-s0 "$RUN/${P}_s0" --run-s1 "$RUN/${P}_s1" --run-red "$RUN/${P}_red" --eval-dir "$PKG/eval" --refe "$PKG/refe" \
    --card PREREG_DATASCALE.md --card-sha256 $(cut -c1-64 PREREG_DATASCALE.sha256) --out "$OUT" > $OUT.read.log 2>&1 || die "read $LV rc=$?"
  say "read $LV: $(tail -1 $OUT.read.log | cut -c1-300)"
  oc=$($PY - "$OUT/READ_$TAG.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
bad = sorted(k for k, v in (r.get("controls") or {}).items() if isinstance(v, dict) and v.get("pass") is False)
o = r.get("outcome") or r.get("outcome_provisional") or {}
o = o.get("outcome") if isinstance(o, dict) else o
print("INVALID:" + ",".join(bad) if bad else str(o))
PYEOF
)
  say "$LV provisional outcome: $oc"
  case "$oc" in
    INVALID*) die "$LV $oc (a control failed: the read is INVALID; report, never re-read)";;
    NULL-CV) printf '%s NULL-CV (controls PASS; card 4: the arm is C1, navtest not read for a decision)\n' "$(date -u +%FT%TZ)" > $OUT/LEVER.DONE; continue;;
  esac
  gate; SMOKE=50 bash "$FAMSH" "$OUT" "$TIP" $A0 $A1 > $OUT.fam_smoke.log 2>&1 || die "$LV families smoke ($(tail -1 $OUT.fam_smoke.log | cut -c1-200))"
  say "$LV families smoke done"
  gate; bash "$FAMSH" "$OUT" "$TIP" $A0 $A1 > $OUT.fam_full.log 2>&1 || die "$LV families full ($(tail -1 $OUT.fam_full.log | cut -c1-200))"
  say "$LV families full done"
  F=$OUT/fam; gate
  $PY ds_read.py confirm --read "$OUT/READ_$TAG.json" --table "$TAB" --pass-rows "$F/pass/rows_s*.jsonl" --families "$F/families" \
    --tactical "$F/tactical" --strategic "$F/strategic" --out "$OUT" > $OUT.confirm.log 2>&1 || die "confirm $LV rc=$?"
  say "confirm $LV: $(tail -1 $OUT.confirm.log | cut -c1-300)"
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$(tail -1 $OUT.confirm.log | cut -c1-200)" > $OUT/LEVER.DONE
done
$PY - "$RUN" <<'PYEOF'
import glob, json, os, sys
run = sys.argv[1]; s = {}
for lv, tag in (("ds_sh", "DSSH"), ("ds_ep", "DSEP")):
    d = f"{run}/read_{lv}"
    v = f"{d}/VERDICT_{tag}.json"; r = f"{d}/READ_{tag}.json"
    src = v if os.path.exists(v) else r
    j = json.load(open(src)); o = j.get("outcome") or j.get("outcome_provisional") or {}
    s[lv] = {"source": src, "outcome": o.get("outcome") if isinstance(o, dict) else o, "s0_chosen_step": (j.get("M1_M3") or {}).get("s0", {}).get("chosen_step"),
             "lever_done": open(f"{d}/LEVER.DONE").read().strip()}
json.dump(s, open(f"{run}/DSREAD_SUMMARY.json", "w"), indent=1, default=str)
print(json.dumps(s, default=str)[:600])
PYEOF
date -u +%FT%TZ > $RUN/DSREAD.DONE; say "DONE: $(cat $RUN/DSREAD_SUMMARY.json | tr -d '\n' | cut -c1-400)"
rm -rf "E:/Projects/TanitAD-artifacts/datascale/readtip" && say "DELETED E:/Projects/TanitAD-artifacts/datascale/readtip (own tip export, re-creatable from $LANDED)"
