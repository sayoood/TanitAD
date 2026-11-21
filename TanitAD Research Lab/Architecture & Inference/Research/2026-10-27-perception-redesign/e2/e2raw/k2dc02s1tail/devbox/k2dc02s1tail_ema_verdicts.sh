#!/bin/bash
# k2dc02s1tail_ema_verdicts.sh [DRY] -- DEV BOX, the EMA-weights side of the K2DC02s1 dose reading (PREREG_K2DC02S1_2026-11-20 sec. 3: "read twice: EMA and tail; the EMA outcome is reported beside the tail's").
# Run AFTER the EMA read of K2DC02s1 (stack/ops/r9e2_read/e2read_launch_k2dc02s1.sh) ended on Thor and its S6c/verdict step was run on this box (E2_VERDICT.json under the pulled read dir).
# Same tool and flags as the tail read (k2dc02s1tail_verdicts.sh), with the EMA reads in place of the tail's, and as the K2DC02 EMA dose read invoked it (K2DC in K2's slot, --k2-read = E2_READ_K2DC's read):
#   dose   r9_e2_dose_verdict.py --arm K2DC02s1 --k2dc-pf E2_READ_K2DC02s1/E2_K2_pf --k2-pf E2_READ_K2DC/E2_K2_pf --k0-pf E2_READ_K2DC02s1/E2_K0_pf (K0, hard-linked from E2_READ_K2DC by the read launcher)
#          --metrics runs/r9e2-K2DC02s1/metrics.jsonl --k2dc-read <K2DC02s1 EMA read> --k2-read <E2_READ_K2DC's read>
#   floor  the same with --k2-pf E2_READ_K2DC02/E2_K2_pf and --k2-read <E2_READ_K2DC02's read> (K2DC02 EMA in K2's slot): its rows are the EMA floor, never a verdict
#   then   k2dc02s1tail_combined.py --mode ema: the EMA outcome with the sec. 4 floor-admissibility test written out; the reading is EMA-REPORTED-BESIDE (not a sec. 5 reading: the TAIL outcome decides).
# The dev-box copies of the three EMA reads (EVAL_K2.json + s6c/) are READ from F: (defaults below; override with K2S_EMA_READ_S1 / _K2DC / _K2DC02); the planner dumps are PULLED from Thor and sha256-verified.
# GUARD: the pinned dose tool must know --arm K2DC02s1 (dose_tool_guard): at 48c797f3 (the PREREG commit) it did NOT; the default tip 20179573 (MM land, 11-20) carries the arm. A tip without it REFUSES at DOSETOOL.
# Output $D/verdicts_ema (never overwritten). DRY: every guard, no pull, no python, no output.
# Markers: ZZK2DC02S1EMA-DONEZZ | -DONE-VOIDZZ | -ABORT-<why>ZZ.  EXIT: 0 | 5 VOID | 3 a step failed | 6 refused by a guard.  Dose tool rc: 0 SUPPORTED 1 REFUTED 2 INCONCLUSIVE 3 DOSE-VOID 4 REFUSED.
MODE=${1:-}
K2T_D=${K2T_D:-/c/t/k2dc02s1ema}
. "$(dirname "$0")/k2dc02s1tail_common.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=$D/logs/ema_verdicts.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
die() { say "ZZK2DC02S1EMA-ABORT-$1ZZ $2"; exit 6; }
RUNS=/home/nvidia/r8ns/r9e2/runs
T_S1=$RUNS/E2_READ_K2DC02s1; T_K2DC=$RUNS/E2_READ_K2DC; T_K2DC02=$RUNS/E2_READ_K2DC02
RD_S1=${K2S_EMA_READ_S1:-/f/refcv9_e3/r9e2_read_K2DC02s1/pull/read}
RD_K2DC=${K2S_EMA_READ_K2DC:-/f/refcv9_e3/r9e2_read_K2DC/pull/read}
RD_K2DC02=${K2S_EMA_READ_K2DC02:-/f/refcv9_e3/r9e2_read_K2DC02/pull/read}
P=$D/pull; V=$D/verdicts_ema
METTHOR=$RUNS/r9e2-K2DC02s1/metrics.jsonl
say "START (pid $$) mode ${MODE:-REAL} tip ${TIP:0:12}"
[ -z "$MODE" ] || [ "$MODE" = DRY ] || die USAGE "usage: k2dc02s1tail_ema_verdicts.sh [DRY]"
[ "$(th "echo up")" = up ] || die NOSSH "cannot reach tanitad-thor"
AW=$(th "grep -c 'AWAIT-S6C' $T_S1/ROLL_STATUS.txt"); [ "${AW:-0}" -ge 1 ] 2>/dev/null || die NOAWAIT "no AWAIT-S6C in $T_S1/ROLL_STATUS.txt (the K2DC02s1 EMA read is not done on Thor)"
for p in $T_S1/E2_K2_pf $T_S1/E2_K0_pf $T_K2DC/E2_K2_pf $T_K2DC02/E2_K2_pf; do [ "$(th "[ -d $p ] && echo Y")" = Y ] || die THORIN "missing on Thor: $p"; done
for r in "$RD_S1" "$RD_K2DC" "$RD_K2DC02"; do [ -s "$r/EVAL_K2.json" ] && [ -s "$r/s6c/S6C_K2_s0.json" ] && [ -s "$r/s6c/S6C_K2_s1.json" ] || die NOREAD "no EVAL_K2.json + s6c/S6C_K2_s{0,1}.json under $r"; done
[ -s "$RD_S1/E2_VERDICT.json" ] || die NOVERDICT "no $RD_S1/E2_VERDICT.json: run the S6c / verdict step of the K2DC02s1 EMA read on this box first"
[ "$MODE" = DRY ] || { [ -e "$V" ] && die EXISTS "$V exists (a read is never overwritten): MM decision, by literal path"; [ -e "$P" ] && die EXISTS "$P exists (a previous pull): MM decision, by literal path"; }
export_code; rc=$?; [ $rc = 0 ] || die EXPORT "export_code rc $rc"
say "code $C: $(cat "$C/.verified")"
MSG=$(dose_tool_guard) || die DOSETOOL "$MSG"
METSHA=$(th "sha256sum $METTHOR" | cut -d' ' -f1); [ -n "$METSHA" ] || die METRICS "cannot read the sha256 of $METTHOR on Thor"
if [ "$MODE" = DRY ]; then say "ZZK2DC02S1EMA-DRY-OKZZ all guards pass; nothing pulled, nothing run"; exit 0; fi
FREE=$(df -BM "$D" | awk 'NR==2 {gsub("M","",$4); print $4}'); [ "${FREE:-0}" -gt 2048 ] 2>/dev/null || die CDISK "$D drive free ${FREE} MB < 2048 (the four planner-roll dirs are ~0.2 GB)"
mkdir -p "$P/s1" "$P/k2dc" "$P/k2dc02" "$V/dose" "$V/floor" "$V/combined" || die MKDIR "$P"
pull() {   # pull TAG THORDIR DSTDIR ENTRY
  local tag=$1 src=$2 dst=$3 ent=$4
  TH_TO=1800 th "cd $src && find $ent -type f -print0 | sort -z | xargs -0 sha256sum" > "$D/sha/thor_$tag.sha256"
  [ -s "$D/sha/thor_$tag.sha256" ] || die THORSHA "no sha256 list for $tag"
  timeout 3600 ssh -n -o ConnectTimeout=20 tanitad-thor "cd $src && tar -cf - $ent" | tar --force-local -xf - -C "$dst" || die PULL "tar stream failed ($tag)"
  (cd "$dst" && sha256sum -c --quiet "$D/sha/thor_$tag.sha256") > "$D/sha/check_$tag.txt" 2>&1 || die PULLSHA "$tag sha256 mismatch: $(head -3 "$D/sha/check_$tag.txt" | tr '\n' ' ')"
  [ "$(cd "$dst" && find $ent -type f | wc -l)" = "$(wc -l < "$D/sha/thor_$tag.sha256")" ] || die PULLCOUNT "$tag file count differs from Thor's list"
  say "pulled $tag: $(wc -l < "$D/sha/thor_$tag.sha256") files sha256-equal to Thor"
}
pull s1_k2 "$T_S1" "$P/s1" E2_K2_pf
pull s1_k0 "$T_S1" "$P/s1" E2_K0_pf
pull k2dc "$T_K2DC" "$P/k2dc" E2_K2_pf
pull k2dc02 "$T_K2DC02" "$P/k2dc02" E2_K2_pf
timeout 600 ssh -n -o ConnectTimeout=20 -o BatchMode=yes tanitad-thor "cat $METTHOR" > "$P/metrics_k2dc02s1.jsonl" 2>/dev/null
[ "$(sha256sum "$P/metrics_k2dc02s1.jsonl" | cut -d' ' -f1)" = "$METSHA" ] || die METRICS "the pulled metrics.jsonl sha256 != Thor's $METSHA"
gate ema || die COMMITLOW "commit free < 5632 MB for 8 h; the pull is kept"
export PYTHONPATH="$(W2 "$C/stack");$(W2 "$C/stack/scripts");$(W2 "$C/taniteval")" PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1
SPLW="$(W2 "$C/$SPL")"
PF_S1=$P/s1/E2_K2_pf; PF_K0=$P/s1/E2_K0_pf; PF_K2DC=$P/k2dc/E2_K2_pf; PF_K2DC02=$P/k2dc02/E2_K2_pf
FAIL=0
step() { say "ZZK2DC02S1EMA-STEP-$1 rc $2"; }
"$PY" -X utf8 "$C/stack/scripts/r9_e2_dose_verdict.py" --arm K2DC02s1 --k2dc-pf "$(W2 "$PF_S1")" --k2-pf "$(W2 "$PF_K2DC")" --k0-pf "$(W2 "$PF_K0")" --dev-split "$SPLW" \
    --metrics "$(W2 "$P/metrics_k2dc02s1.jsonl")" --k2dc-read "$(W2 "$RD_S1")" --k2-read "$(W2 "$RD_K2DC")" --out-dir "$(W2 "$V/dose")" > "$V/dose/dose_verdict.log" 2>&1 < /dev/null; DRC=$?
step dose $DRC; tail -3 "$V/dose/dose_verdict.log" | cut -c1-240 | tee -a "$LOG"
[ $DRC -le 3 ] || { FAIL=1; say "dose tool REFUSED (rc $DRC): $V/dose/dose_verdict.log"; }
"$PY" -X utf8 "$C/stack/scripts/r9_e2_dose_verdict.py" --arm K2DC02s1 --k2dc-pf "$(W2 "$PF_S1")" --k2-pf "$(W2 "$PF_K2DC02")" --k0-pf "$(W2 "$PF_K0")" --dev-split "$SPLW" \
    --metrics "$(W2 "$P/metrics_k2dc02s1.jsonl")" --k2dc-read "$(W2 "$RD_S1")" --k2-read "$(W2 "$RD_K2DC02")" --out-dir "$(W2 "$V/floor")" > "$V/floor/floor_verdict.log" 2>&1 < /dev/null; FRC=$?
step floor $FRC; echo "EMA FLOOR READING (K2DC02s1 - K2DC02, identical flags): the tool's outcome label in this directory is NOT a verdict about a lever (PREREG_K2DC02S1 sec. 4)" > "$V/floor/FLOOR_NOT_A_VERDICT.txt"
[ $FRC -le 3 ] || { FAIL=1; say "floor run REFUSED (rc $FRC): $V/floor/floor_verdict.log"; }
"$PY" -I -X utf8 "$HERE/k2dc02s1tail_combined.py" --mode ema --dose-dir "$(W2 "$V/dose")" --floor-dir "$(W2 "$V/floor")" --out-dir "$(W2 "$V/combined")" > "$V/combined/combined.log" 2>&1 < /dev/null; CRC=$?
step combined $CRC; tail -1 "$V/combined/combined.log" | cut -c1-260 | tee -a "$LOG"
[ $CRC = 0 ] || [ $CRC = 5 ] || { FAIL=1; say "combined reading REFUSED (rc $CRC): $V/combined/combined.log"; }
DOSE=$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8')).get('outcome'))" "$(W2 "$V/dose/DOSE_VERDICT.json")" 2>/dev/null | tr -d '\r')
say "SUMMARY ema_dose=${DOSE:-none}(rc $DRC) combined_rc=$CRC steps_failed=$FAIL out=$V"
[ $FAIL = 0 ] || { say "ZZK2DC02S1EMA-ABORT-STEPFAILEDZZ see $LOG"; exit 3; }
if [ "$CRC" = 5 ]; then say "ZZK2DC02S1EMA-DONE-VOIDZZ"; exit 5; fi
say "ZZK2DC02S1EMA-DONEZZ"; exit 0
