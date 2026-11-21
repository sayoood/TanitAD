#!/bin/bash
# k2dc02nhtail_verdicts.sh [DRY] -- run AFTER k2dc02nhtail_s6c_pull_score.sh ended ZZK2DC02nhT-S6C-DONEZZ (PREREG_K2DC02NH_2026-11-20 sec. 2 to 4). Template: k2dc02s1tail_verdicts.sh (landed, PREREG_K2DC02S1).
# Code = the same PINNED, blob-verified export the S6c step used. CPU only, one python at a time, each behind the 5632 MB commit gate. Output $D/verdicts (never overwritten).
# Comparators are the PINNED rolls of the K2DC02-tail read, pulled and re-hashed by the S6c step: K0t = $P/pin/cmp/E2_K0_pf, K2DC02t = $P/pin/stageB/E2_K2_pf (the comparator in K2's slot of the lever rule).
#   (v) VALIDITY (PREREG_K2DC02TAIL+AMENDMENT1 sec. 2): items 1 (seed 0 + no HSG, enforced by the Thor build guard), 3 (identity control IDCTL-PASS-TENSOR, tail guard REGRESSION-RED, dump md5 == tail md5 at step 6000),
#       4 (pinned pull), and the ZERO-GRADIENT GATE (no ZEROGRAD_FAIL.json under runs/r9e2-K2DC02nh on Thor), written to $V/VALIDITY.txt; a failed item 3 = TAIL-VOID, a failed zero-gradient gate = ZEROGRAD-VOID (both = sec. 4 item 5 VOID).
#   (a) PREREG_K2DC02S1 sec. 2 item 6 REPRODUCTION CHECK: families on (pinned K2DC02t, PINNED K0t) and on (pinned K2DC02t, RE-ROLLED K0t), inference seed 0: LAT_cross_mae_m within +-1 mm, else REPRO-VOID (steps b-d still run so the MM
#       can diagnose, but the reading is VOID: no lever claim). Also r9_e2_distkeep.py on both when the psync lead block's sha256 equals Thor's.
#   (b) LEVER RULE, tail (sec. 3 + sec. 4)   k2dc02nhtail_lever.py --mode tail: K2DC02nht vs K0t and vs K2DC02t, both inference seeds, the dose rule's constants -> $V/lever/LEVER_READING.json, field "reading" = SUPPORTED | REFUTED | INCONCLUSIVE | VOID
#       (+ "sec4_item"); the map trade-off (nf_swedge_prec_lv_5_20, nf_prec_0_10; nh - K2DC02) is written into it. If the EMA read's E2_VERDICT.json exists (K2T_EMA_E2V) it is passed as the other read's E2 verdict (sec. 4 item 1 = GO on either read).
#   (c) FOUR-FAMILY table per seed (k2dc02nhtail_fam_table.py on the E2 verdict K2DC02nht vs K0t; its header text says "K2DC02nh", read it as K2DC02nht - K0t) = the secondary E2 rule on tail weights (r9_e2_verdict.py ran in the S6c step);
#       the K2DC02nht - K2DC02t family rows are in LEVER_READING.json seeds.<s>.dc_vs_k2_by_family
#   (d) STRATEGIC, report-only (in full): k2dc02nhtail_strategic_navcomp.py (K0t ref; K0t, K2DC02t, K2DC02nht; seeds 0, 1), then k2dc02nhtail_sanitize_navcomp.py -> $D/verdicts/strategic_bank (clip ids removed; bank ONLY that dir)
# GUARDS: the pinned r9_e2_dose_verdict.py must define BAR_M 0.0025 / BAND_M 0.0010 / seed_numbers / decide (lever_tool_guard in the common file); the lever script itself refuses if they differ from its own constants or if its `decide` and the tool's disagree.
# Markers: ZZK2DC02nhT-VERD-DONEZZ | ZZK2DC02nhT-VERD-DONE-VOIDZZ | ZZK2DC02nhT-VERD-ABORT-<why>ZZ (+ per-step lines ZZK2DC02nhT-VERD-STEP-<name> rc N).
# EXIT: 0 all steps ran, no VOID | 5 all steps ran, a VOID | 3 a step failed | 6 refused/aborted by a guard.  Lever script rc: 0 a reading was made, 5 VOID, 4 REFUSED.
# DRY: every guard (S6c done, inputs, export, lever-tool guard, lead-block sha, labels), no python run, no output dir.
MODE=${1:-}
. "$(dirname "$0")/k2dc02nhtail_common.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=$D/logs/verdicts.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
die() { say "ZZK2DC02nhT-VERD-ABORT-$1ZZ $2"; exit 6; }
P=$D/pull; MS=$P/stageB; RD=$MS/read; V=$D/verdicts
PFNH=$MS/E2_K2_pf; PFK0=$P/pin/cmp/E2_K0_pf; PFK2DC02=$P/pin/stageB/E2_K2_pf; PFREPRO=$P/repro/E2_K0_pf
EMA_E2V=${K2T_EMA_E2V:-/f/refcv9_e3/r9e2_read_K2DC02nh/pull/read/E2_VERDICT.json}
LABELS=E:/Projects/TanitAD-artifacts/_s2build-copy-20260919/release/v8/s2_labels_v8_eval.jsonl.gz
say "START (pid $$) mode ${MODE:-REAL} tip ${TIP:0:12}"
[ -z "$MODE" ] || [ "$MODE" = DRY ] || die USAGE "usage: k2dc02nhtail_verdicts.sh [DRY]"
grep -q 'ZZK2DC02nhT-S6C-DONEZZ' "$D/logs/s6c.log" 2>/dev/null || die NOS6C "no ZZK2DC02nhT-S6C-DONEZZ in $D/logs/s6c.log: run k2dc02nhtail_s6c_pull_score.sh first (it needs Thor's AWAIT-S6C)"
[ -s "$RD/E2_VERDICT.json" ] && [ -s "$RD/BOX_PAIRED.json" ] || die NOVERDICT "no E2_VERDICT.json / BOX_PAIRED.json under $RD"
for p in "$PFNH" "$PFK0" "$PFK2DC02"; do for s in 0 1; do [ -s "$p/dump_s$s/manifest.json" ] || die NOPF "no $p/dump_s$s/manifest.json"; done; done
[ -s "$PFREPRO/dump_s0/manifest.json" ] || die NOREPRO "no $PFREPRO/dump_s0/manifest.json (the reproduction roll)"
grep -q 'pinned comparators pulled' "$D/logs/s6c.log" || die NOPIN "the S6c step never logged 'pinned comparators pulled': the comparators were not re-hashed against the pin manifest"
[ "$MODE" = DRY ] || { [ -e "$V" ] && die EXISTS "$V exists (a read is never overwritten): MM decision, by literal path"; }
export_code; rc=$?; [ $rc = 0 ] || die EXPORT "export_code rc $rc"
say "code $C: $(cat "$C/.verified")"
MSG=$(lever_tool_guard) || die LEVERTOOL "$MSG"
[ -s "$C/$K2DC02TREAD/EVAL_K2.json" ] && [ -s "$C/$K2DC02TREAD/s6c/S6C_K2_s0.json" ] && [ -s "$C/$K2DC02TREAD/s6c/S6C_K2_s1.json" ] || die NOBANKED2 "banked K2DC02-tail read (K2DC02t, the comparator in K2's slot) missing in the export ($K2DC02TREAD)"
[ -s "$HERE/k2dc02nhtail_lever.py" ] || die NOLEVER "no $HERE/k2dc02nhtail_lever.py"
LEADOK=0; [ -s "$LEAD_BLOCK" ] && [ "$(sha256sum "$LEAD_BLOCK" | cut -d' ' -f1)" = "$LEAD_SHA" ] && LEADOK=1
[ "$LEADOK" = 1 ] && say "lead block sha256 == Thor's: distkeep rows will be recomputed" || say "NOTE: lead block missing or sha256 differs: LON_dk_* rows are NOT recomputed (the gate row LAT_cross_mae_m does not need them)"
[ -s "$LABELS" ] || die NOLABELS "$LABELS"
grep -q sel_bank "$C/taniteval/taniteval/nav_compliance.py" || die NOSELBANK "the pinned taniteval lacks the sel_bank fix"
if [ "$MODE" = DRY ]; then say "ZZK2DC02nhT-VERD-DRY-OKZZ all guards pass; nothing run"; exit 0; fi
mkdir -p "$V/repro/pinned" "$V/repro/reroll" "$V/lever" || die MKDIR "$V"
export PYTHONPATH="$(W2 "$C/stack");$(W2 "$C/stack/scripts");$(W2 "$C/taniteval")" PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1
SPLW="$(W2 "$C/$SPL")"
FAIL=0; TV=0; RV=0; ZV=0
step() { say "ZZK2DC02nhT-VERD-STEP-$1 rc $2"; }
# (v) validity items that live on Thor / in the pull (a failed item 3 = TAIL-VOID; a failed zero-gradient gate = ZEROGRAD-VOID)
vline() { echo "$1" >> "$V/VALIDITY.txt"; say "validity: $1"; }
vline "item 1 (seed 0, no --r9-hist-supervise-geom, share target 0.02 in config.json): enforced by the Thor build guard; BUILD_STATUS passed it: $(th "grep -c 'START' $W/BUILD_STATUS.txt") START line(s)"
[ "$(th "grep -c 'IDCTL-PASS-TENSOR' $W/BUILD_STATUS.txt")" -ge 1 ] 2>/dev/null && vline "item 3 identity control A1: IDCTL-PASS-TENSOR present" || { TV=1; vline "item 3 identity control A1: NOT PASSED (no IDCTL-PASS-TENSOR in BUILD_STATUS.txt) -> TAIL-VOID"; }
[ "$(th "grep -c 'REGRESSION-RED' $W/stageA/STATUS.txt")" -ge 1 ] 2>/dev/null && vline "item 3 deliberate regression: REGRESSION-RED present (the guard refused a model_ema.pt header)" || { TV=1; vline "item 3 deliberate regression: NOT RED -> TAIL-VOID"; }
"$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); m=open(sys.argv[2]).read().strip(); sys.exit(0 if int(d.get('step',-1))==6000 and d.get('ckpt_md5')==m and m else 1)" "$(W2 "$RD/dump_K2/dump.json")" "$(W2 "$D/sha/tail_md5.txt")" \
  && vline "item 3 tail guard: the K2DC02nht dump records step 6000 and the tail file's md5 $(cat "$D/sha/tail_md5.txt")" || { TV=1; vline "item 3 tail guard: the dump does NOT record step 6000 + the tail md5 -> TAIL-VOID"; }
vline "item 4 pinned comparators: $(grep 'pinned comparators pulled' "$D/logs/s6c.log" | tail -1 | cut -c1-200)"
ZG=$(th "[ -e /home/nvidia/r8ns/r9e2/runs/r9e2-K2DC02nh/ZEROGRAD_FAIL.json ] && echo FAIL; [ -d /home/nvidia/r8ns/r9e2/runs/r9e2-K2DC02nh ] && echo DIR")
case "$ZG" in *FAIL*) ZV=1; vline "zero-gradient gate: ZEROGRAD_FAIL.json EXISTS under runs/r9e2-K2DC02nh -> ZEROGRAD-VOID";; *DIR*) vline "zero-gradient gate: no ZEROGRAD_FAIL.json under runs/r9e2-K2DC02nh";; *) ZV=1; vline "zero-gradient gate: runs/r9e2-K2DC02nh not readable on Thor -> ZEROGRAD-VOID (unread = not passed)";; esac
# (a) reproduction check: families on the two K0t variants against the same pinned K2DC02t, seed 0
gate repro || die COMMITLOW "commit free < 5632 MB for 8 h; nothing run yet (rm -r the empty $V by literal path and rerun)"
S=0
for pair in "pinned:$PFK0" "reroll:$PFREPRO"; do
  tag=${pair%%:*}; b=${pair#*:}
  "$PY" -X utf8 "$C/stack/scripts/r9_e2_families.py" --mode families --a-dump "$(W2 "$PFK2DC02/dump_s$S")" --b-dump "$(W2 "$b/dump_s$S")" --dev-split "$SPLW" \
      --out "$(W2 "$V/repro/$tag/families_s$S.json")" > "$V/repro/$tag/families_s$S.log" 2>&1 < /dev/null; rc=$?
  step "families_${tag}_s$S" $rc; { [ $rc = 0 ] && [ -s "$V/repro/$tag/families_s$S.json" ]; } || die REPRO-FAILED "r9_e2_families.py $tag seed $S rc $rc ($V/repro/$tag/families_s$S.log)"
  if [ "$LEADOK" = 1 ]; then
    "$PY" -X utf8 "$C/stack/scripts/r9_e2_distkeep.py" --a-dump "$(W2 "$PFK2DC02/dump_s$S")" --b-dump "$(W2 "$b/dump_s$S")" --dev-split "$SPLW" --lead-block "$(W2 "$LEAD_BLOCK")" \
        --out "$(W2 "$V/repro/$tag/distkeep_s$S.json")" --merge-into "$(W2 "$V/repro/$tag/families_s$S.json")" > "$V/repro/$tag/distkeep_s$S.log" 2>&1 < /dev/null; rc=$?
    step "distkeep_${tag}_s$S" $rc; [ $rc = 0 ] || { FAIL=1; say "distkeep $tag seed $S failed (rows LON_dk_* will read one-read-only); see $V/repro/$tag/distkeep_s$S.log"; }
  fi
done
"$PY" -I -X utf8 "$HERE/k2dc02nhtail_repro_check.py" --pinned-dir "$(W2 "$V/repro/pinned")" --reroll-dir "$(W2 "$V/repro/reroll")" --out "$(W2 "$V/REPRO_CHECK.json")" > "$V/REPRO_CHECK.txt" 2>&1 < /dev/null; rc=$?
step repro_check $rc; head -4 "$V/REPRO_CHECK.txt" | cut -c1-220 | tee -a "$LOG"
case $rc in 0) ;; 5) RV=1; echo "REPRO-VOID: the re-rolled K0t does not reproduce the pinned K2DC02t - K0t LAT_cross_mae_m within 1 mm at inference seed 0. NO lever claim (sec. 4 item 5). Steps b-d below are diagnostics only." > "$V/REPRO_VOID"; say "ZZK2DC02nhT-VERD-REPROVOIDZZ";;
  *) die REPRO-REFUSED "k2dc02nhtail_repro_check.py rc $rc ($V/REPRO_CHECK.txt)";; esac
# (b) the lever rule on the tail weights
gate lever || die COMMITLOW "commit free < 5632 MB for 8 h at step b (a is done: $V/REPRO_CHECK.txt)"
LARGS=(); [ $TV = 1 ] && LARGS+=(--tail-void); [ $RV = 1 ] && LARGS+=(--repro-void); [ $ZV = 1 ] && LARGS+=(--zerograd-void)
[ -s "$EMA_E2V" ] && { LARGS+=(--other-e2-verdict "$(W2 "$EMA_E2V")"); say "EMA E2 verdict found: $EMA_E2V"; } || say "NOTE: no EMA E2 verdict at $EMA_E2V: sec. 4 item 1 is decided on the tail read's E2 verdict alone"
"$PY" -X utf8 "$HERE/k2dc02nhtail_lever.py" --mode tail --nh-pf "$(W2 "$PFNH")" --k2dc02-pf "$(W2 "$PFK2DC02")" --k0-pf "$(W2 "$PFK0")" --dev-split "$SPLW" \
    --nh-read "$(W2 "$RD")" --k2dc02-read "$(W2 "$C/$K2DC02TREAD")" --out-dir "$(W2 "$V/lever")" "${LARGS[@]}" > "$V/lever/lever.log" 2>&1 < /dev/null; LRC=$?
step lever $LRC; tail -1 "$V/lever/lever.log" | cut -c1-260 | tee -a "$LOG"
[ $LRC = 0 ] || [ $LRC = 5 ] || { FAIL=1; say "lever reading REFUSED (rc $LRC): $V/lever/lever.log"; }
# (c) four-family table per seed
"$PY" -I -X utf8 "$HERE/k2dc02nhtail_fam_table.py" "$(W2 "$RD/E2_VERDICT.json")" > "$V/FAM_TABLE.md" 2> "$V/fam_table.err" < /dev/null; rc=$?
step fam_table $rc; [ $rc = 0 ] && [ -s "$V/FAM_TABLE.md" ] || { FAIL=1; say "fam_table failed: $V/fam_table.err"; }
# (d) STRATEGIC, report-only (in full), then sanitize
gate strategic || die COMMITLOW "commit free < 5632 MB for 8 h at step d (a-c are done)"
export PYTHONPATH="$(W2 "$C/taniteval");$(W2 "$C/stack");$(W2 "$C/stack/scripts")"      # the FIXED taniteval first (sel_bank)
"$PY" -X utf8 "$HERE/k2dc02nhtail_strategic_navcomp.py" --labels "$LABELS" --out-dir "$(W2 "$V/strategic")" --ref K0t \
    --arm "K0t:0=$(W2 "$PFK0/dump_s0")" --arm "K0t:1=$(W2 "$PFK0/dump_s1")" \
    --arm "K2DC02t:0=$(W2 "$PFK2DC02/dump_s0")" --arm "K2DC02t:1=$(W2 "$PFK2DC02/dump_s1")" \
    --arm "K2DC02nht:0=$(W2 "$PFNH/dump_s0")" --arm "K2DC02nht:1=$(W2 "$PFNH/dump_s1")" --n-boot 2000 > "$V/strategic.log" 2>&1 < /dev/null; rc=$?
step strategic $rc; tail -1 "$V/strategic.log" | cut -c1-240 | tee -a "$LOG"
if [ $rc = 0 ]; then
  "$PY" -I -X utf8 "$HERE/k2dc02nhtail_sanitize_navcomp.py" "$(W2 "$V/strategic")" "$(W2 "$V/strategic_bank")" > "$V/sanitize.log" 2>&1 < /dev/null; rc=$?
  step sanitize $rc; [ $rc = 0 ] || { FAIL=1; say "sanitize failed: $V/sanitize.log (do NOT bank $V/strategic: it carries clip ids)"; }
else FAIL=1; say "strategic failed or its REF-REF control did not read 0.0: $V/strategic.log"; fi
E2V=$("$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); print(d.get('verdict'), d.get('failing_clauses'))" "$(W2 "$RD/E2_VERDICT.json")" 2>/dev/null | tr -d '\r')
READ=$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8')).get('reading'))" "$(W2 "$V/lever/LEVER_READING.json")" 2>/dev/null | tr -d '\r')
say "SUMMARY tailvoid=$TV reprovoid=$RV zerogradvoid=$ZV reading=${READ:-none}(rc $LRC) e2_verdict_K2DC02nht_vs_K0t=${E2V:-none} steps_failed=$FAIL out=$V"
[ $FAIL = 0 ] || { say "ZZK2DC02nhT-VERD-ABORT-STEPFAILEDZZ see $LOG"; exit 3; }
if [ $TV = 1 ] || [ $RV = 1 ] || [ $ZV = 1 ] || [ "$LRC" = 5 ]; then say "ZZK2DC02nhT-VERD-DONE-VOIDZZ"; exit 5; fi
say "ZZK2DC02nhT-VERD-DONEZZ"; exit 0
