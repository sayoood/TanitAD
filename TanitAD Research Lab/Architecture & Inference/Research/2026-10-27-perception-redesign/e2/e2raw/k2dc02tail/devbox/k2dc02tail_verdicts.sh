#!/bin/bash
# k2dc02tail_verdicts.sh [DRY] -- run AFTER k2dc02tail_s6c_pull_score.sh ended ZZK2DC02T-S6C-DONEZZ (PREREG_K2DC02TAIL_2026-11-20 + AMENDMENT1). Build agent for MM 4a3d05, 2026-11-20.
# Code = the same PINNED, blob-verified export the S6c step used. CPU only, one python at a time, each behind the 5632 MB commit gate. Output $D/verdicts (never overwritten).
#   (a) AMENDMENT1 sec. 2.5 REPRODUCTION CHECK  families on the RE-ROLLED pair K2DCt (cmp_k2dc) - K0t (cmp), per seed, vs the banked E2-TAIL E2_VERDICT.json (read from the file, not typed):
#       LAT_cross_mae_m within +-1 mm at each seed, else TAIL-VOID (prints it; steps b-d still run so the MM can diagnose, but NO dose claim is made: see TAIL_VOID file)
#       (also r9_e2_distkeep.py on the re-rolled pair when the psync lead block's sha256 equals Thor's, so the LON_dk_* rows can be compared)
#   (b) DOSE verdict  r9_e2_dose_verdict.py --arm K2DC02 --k2dc-pf <stageB/E2_K2_pf = K2DC02t> --k2-pf <cmp_k2dc/E2_K2_pf = K2DCt> --k0-pf <cmp/E2_K0_pf = K0t>
#       --metrics <r9e2-K2DC02 metrics.jsonl, sha256-checked copy> --k2dc-read <K2DC02t read> --k2-read <E2-TAIL's banked read = the K2DCt read: EVAL_K2.json + s6c/>
#       (the tool's flags at the pinned commit; the K2DCt read exists banked, nothing is missing)
#   (c) FOUR-FAMILY table per seed (fam_table.py on the E2 verdict K2DC02t vs K0t; its header text says "K2DC02", read it as K2DC02t - K0t)
#   (d) STRATEGIC, report-only: e2_strategic_navcomp.py (K0t ref; K0t, K2DCt, K2DC02t; seeds 0, 1), then sanitize_navcomp.py -> $D/verdicts/strategic_bank (clip ids removed; bank ONLY that dir)
# Markers: ZZK2DC02T-VERD-DONEZZ | ZZK2DC02T-VERD-DONE-TAILVOIDZZ | ZZK2DC02T-VERD-ABORT-<why>ZZ (+ per-step lines ZZK2DC02T-VERD-STEP-<name> rc N).
# EXIT: 0 all steps ran, repro PASS | 5 all steps ran, TAIL-VOID | 3 a step failed | 6 refused/aborted by a guard.  Dose tool rc is logged: 0 SUPPORTED 1 REFUTED 2 INCONCLUSIVE 3 DOSE-VOID 4 REFUSED.
# DRY: every guard (S6c done, inputs, export, metrics sha, lead-block sha, labels), no python run, no output dir.
MODE=${1:-}
. "$(dirname "$0")/k2dc02tail_common.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=$D/logs/verdicts.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
die() { say "ZZK2DC02T-VERD-ABORT-$1ZZ $2"; exit 6; }
P=$D/pull; MS=$P/stageB; RD=$MS/read; V=$D/verdicts
PFK2=$MS/E2_K2_pf; PFK2DC=$P/cmp_k2dc/E2_K2_pf; PFK0=$P/cmp/E2_K0_pf
METSRC=/f/refcv9_e3/r9e2_read_K2DC02/dose/pull/k2dc02
LABELS=E:/Projects/TanitAD-artifacts/_s2build-copy-20260919/release/v8/s2_labels_v8_eval.jsonl.gz
say "START (pid $$) mode ${MODE:-REAL} tip ${TIP:0:12}"
[ -z "$MODE" ] || [ "$MODE" = DRY ] || die USAGE "usage: k2dc02tail_verdicts.sh [DRY]"
grep -q 'ZZK2DC02T-S6C-DONEZZ' "$D/logs/s6c.log" 2>/dev/null || die NOS6C "no ZZK2DC02T-S6C-DONEZZ in $D/logs/s6c.log: run k2dc02tail_s6c_pull_score.sh first (it needs Thor's AWAIT-S6C)"
[ -s "$RD/E2_VERDICT.json" ] && [ -s "$RD/BOX_PAIRED.json" ] || die NOVERDICT "no E2_VERDICT.json / BOX_PAIRED.json under $RD"
for p in "$PFK2" "$PFK2DC" "$PFK0"; do for s in 0 1; do [ -s "$p/dump_s$s/manifest.json" ] || die NOPF "no $p/dump_s$s/manifest.json"; done; done
[ "$MODE" = DRY ] || { [ -e "$V" ] && die EXISTS "$V exists (a read is never overwritten): MM decision, by literal path"; }
export_code; rc=$?; [ $rc = 0 ] || die EXPORT "export_code rc $rc"
say "code $C: $(cat "$C/.verified")"
[ -s "$C/$E2TREAD/E2_VERDICT.json" ] && [ -s "$C/$E2TREAD/EVAL_K2.json" ] && [ -s "$C/$E2TREAD/s6c/S6C_K2_s0.json" ] || die NOBANKED "banked E2-TAIL read missing in the export ($E2TREAD)"
[ -s "$METSRC/metrics.jsonl" ] && [ "$(sha256sum "$METSRC/metrics.jsonl" | cut -d' ' -f1)" = "$(cut -d' ' -f1 /f/refcv9_e3/r9e2_read_K2DC02/dose/pull/k2dc02_metrics.sha256)" ] || die METRICS "K2DC02 metrics.jsonl copy at $METSRC missing or sha256 != the Thor-side list"
LEADOK=0; [ -s "$LEAD_BLOCK" ] && [ "$(sha256sum "$LEAD_BLOCK" | cut -d' ' -f1)" = "$LEAD_SHA" ] && LEADOK=1
[ "$LEADOK" = 1 ] && say "lead block sha256 == Thor's: distkeep rows will be recomputed" || say "NOTE: lead block missing or sha256 differs: LON_dk_* rows are NOT recomputed (the gate row LAT_cross_mae_m does not need them)"
[ -s "$LABELS" ] || die NOLABELS "$LABELS"
grep -q sel_bank "$C/taniteval/taniteval/nav_compliance.py" || die NOSELBANK "the pinned taniteval lacks the sel_bank fix"
if [ "$MODE" = DRY ]; then say "ZZK2DC02T-VERD-DRY-OKZZ all guards pass; nothing run"; exit 0; fi
mkdir -p "$V/repro" "$V/dose" || die MKDIR "$V"
cp "$METSRC/metrics.jsonl" "$P/metrics_k2dc02.jsonl" || die MKDIR "metrics copy"
export PYTHONPATH="$(W2 "$C/stack");$(W2 "$C/stack/scripts");$(W2 "$C/taniteval")" PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1
SPLW="$(W2 "$C/$SPL")"
FAIL=0; TV=0
step() { say "ZZK2DC02T-VERD-STEP-$1 rc $2"; }
# (a) reproduction check
gate repro || die COMMITLOW "commit free < 5632 MB for 8 h; nothing run yet (rm -r the empty $V by literal path and rerun)"
for S in 0 1; do
  "$PY" -X utf8 "$C/stack/scripts/r9_e2_families.py" --mode families --a-dump "$(W2 "$PFK2DC/dump_s$S")" --b-dump "$(W2 "$PFK0/dump_s$S")" --dev-split "$SPLW" \
      --out "$(W2 "$V/repro/families_s$S.json")" > "$V/repro/families_s$S.log" 2>&1 < /dev/null; rc=$?
  step "families_s$S" $rc; { [ $rc = 0 ] && [ -s "$V/repro/families_s$S.json" ]; } || die REPRO-FAILED "r9_e2_families.py seed $S rc $rc ($V/repro/families_s$S.log)"
  if [ "$LEADOK" = 1 ]; then
    "$PY" -X utf8 "$C/stack/scripts/r9_e2_distkeep.py" --a-dump "$(W2 "$PFK2DC/dump_s$S")" --b-dump "$(W2 "$PFK0/dump_s$S")" --dev-split "$SPLW" --lead-block "$(W2 "$LEAD_BLOCK")" \
        --out "$(W2 "$V/repro/distkeep_s$S.json")" --merge-into "$(W2 "$V/repro/families_s$S.json")" > "$V/repro/distkeep_s$S.log" 2>&1 < /dev/null; rc=$?
    step "distkeep_s$S" $rc; [ $rc = 0 ] || { FAIL=1; say "distkeep seed $S failed (rows LON_dk_* will read one-read-only); see $V/repro/distkeep_s$S.log"; }
  fi
done
"$PY" -I -X utf8 "$HERE/k2dc02tail_repro_check.py" --reroll-dir "$(W2 "$V/repro")" --banked-verdict "$(W2 "$C/$E2TREAD/E2_VERDICT.json")" --out "$(W2 "$V/REPRO_CHECK.json")" > "$V/REPRO_CHECK.txt" 2>&1 < /dev/null; rc=$?
step repro_check $rc; head -4 "$V/REPRO_CHECK.txt" | cut -c1-220 | tee -a "$LOG"
case $rc in 0) ;; 5) TV=1; echo "TAIL-VOID: the re-roll does not reproduce E2-TAIL's K2DCt - K0t LAT_cross_mae_m within 1 mm. NO dose claim (PREREG sec. 5.5). Steps b-d below are diagnostics only." > "$V/TAIL_VOID"; say "ZZK2DC02T-VERD-TAILVOIDZZ";;
  *) die REPRO-REFUSED "k2dc02tail_repro_check.py rc $rc ($V/REPRO_CHECK.txt)";; esac
# (b) dose verdict
gate dose || die COMMITLOW "commit free < 5632 MB for 8 h at step b (a is done: $V/REPRO_CHECK.txt)"
"$PY" -X utf8 "$C/stack/scripts/r9_e2_dose_verdict.py" --arm K2DC02 --k2dc-pf "$(W2 "$PFK2")" --k2-pf "$(W2 "$PFK2DC")" --k0-pf "$(W2 "$PFK0")" --dev-split "$SPLW" \
    --metrics "$(W2 "$P/metrics_k2dc02.jsonl")" --k2dc-read "$(W2 "$RD")" --k2-read "$(W2 "$C/$E2TREAD")" --out-dir "$(W2 "$V/dose")" > "$V/dose/dose_verdict.log" 2>&1 < /dev/null; DRC=$?
step dose $DRC; tail -3 "$V/dose/dose_verdict.log" | cut -c1-240 | tee -a "$LOG"
[ $DRC -le 3 ] || { FAIL=1; say "dose tool REFUSED (rc $DRC): $V/dose/dose_verdict.log"; }
# (c) four-family table per seed
"$PY" -I -X utf8 "$HERE/fam_table.py" "$(W2 "$RD/E2_VERDICT.json")" > "$V/FAM_TABLE.md" 2> "$V/fam_table.err" < /dev/null; rc=$?
step fam_table $rc; [ $rc = 0 ] && [ -s "$V/FAM_TABLE.md" ] || { FAIL=1; say "fam_table failed: $V/fam_table.err"; }
# (d) STRATEGIC, report-only, then sanitize
gate strategic || die COMMITLOW "commit free < 5632 MB for 8 h at step d (a-c are done)"
export PYTHONPATH="$(W2 "$C/taniteval");$(W2 "$C/stack");$(W2 "$C/stack/scripts")"      # the FIXED taniteval first (sel_bank)
"$PY" -X utf8 "$HERE/e2_strategic_navcomp.py" --labels "$LABELS" --out-dir "$(W2 "$V/strategic")" --ref K0t \
    --arm "K0t:0=$(W2 "$PFK0/dump_s0")" --arm "K0t:1=$(W2 "$PFK0/dump_s1")" --arm "K2DCt:0=$(W2 "$PFK2DC/dump_s0")" --arm "K2DCt:1=$(W2 "$PFK2DC/dump_s1")" \
    --arm "K2DC02t:0=$(W2 "$PFK2/dump_s0")" --arm "K2DC02t:1=$(W2 "$PFK2/dump_s1")" --n-boot 2000 > "$V/strategic.log" 2>&1 < /dev/null; rc=$?
step strategic $rc; tail -1 "$V/strategic.log" | cut -c1-240 | tee -a "$LOG"
if [ $rc = 0 ]; then
  "$PY" -I -X utf8 "$HERE/sanitize_navcomp.py" "$(W2 "$V/strategic")" "$(W2 "$V/strategic_bank")" > "$V/sanitize.log" 2>&1 < /dev/null; rc=$?
  step sanitize $rc; [ $rc = 0 ] || { FAIL=1; say "sanitize failed: $V/sanitize.log (do NOT bank $V/strategic: it carries clip ids)"; }
else FAIL=1; say "strategic failed or its REF-REF control did not read 0.0: $V/strategic.log"; fi
E2V=$("$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); print(d.get('verdict'), d.get('failing_clauses'))" "$(W2 "$RD/E2_VERDICT.json")" 2>/dev/null | tr -d '\r')
DOSE=$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8')).get('outcome'))" "$(W2 "$V/dose/DOSE_VERDICT.json")" 2>/dev/null | tr -d '\r')
say "SUMMARY repro=$([ $TV = 1 ] && echo TAIL-VOID || echo PASS) dose=${DOSE:-none}(rc $DRC) e2_verdict_K2DC02t_vs_K0t=${E2V:-none} steps_failed=$FAIL out=$V"
[ $FAIL = 0 ] || { say "ZZK2DC02T-VERD-ABORT-STEPFAILEDZZ see $LOG"; exit 3; }
if [ $TV = 1 ]; then say "ZZK2DC02T-VERD-DONE-TAILVOIDZZ"; exit 5; fi
say "ZZK2DC02T-VERD-DONEZZ"; exit 0
