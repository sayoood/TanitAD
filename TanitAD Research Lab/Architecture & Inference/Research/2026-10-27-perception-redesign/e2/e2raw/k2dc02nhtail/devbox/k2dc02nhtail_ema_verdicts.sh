#!/bin/bash
# k2dc02nhtail_ema_verdicts.sh [DRY] -- DEV BOX, the EMA-weights side of the K2DC02nh lever reading (PREREG_K2DC02NH_2026-11-20 sec. 2: "EMA read ... reported beside the tail read"). Template: k2dc02s1tail_ema_verdicts.sh.
# Run AFTER the EMA read of K2DC02nh (stack/ops/r9e2_read/e2read_launch_k2dc02nh.sh) ended on Thor and its S6c/verdict step was run on this box (E2_VERDICT.json under the pulled read dir).
#   lever   k2dc02nhtail_lever.py --mode ema --nh-pf E2_READ_K2DC02nh/E2_K2_pf --k2dc02-pf E2_READ_K2DC02/E2_K2_pf (K2DC02 EMA in K2's slot) --k0-pf E2_READ_K2DC02nh/E2_K0_pf (K0, seeded by the read launcher)
#           --nh-read <K2DC02nh EMA read> --k2dc02-read <E2_READ_K2DC02's read>; the reading is EMA-REPORTED-BESIDE (the TAIL read decides the lever, sec. 3); `rule_outcome` carries what the same rule gives on the EMA weights.
# The dev-box copies of the two EMA reads (EVAL_K2.json + s6c/) are READ from F: (defaults below; override with K2S_EMA_READ_NH / _K2DC02); the planner dumps are PULLED from Thor and sha256-verified.
# GUARD: lever_tool_guard (the pinned dose tool defines the constants and functions the lever rule imports). Output $D/verdicts_ema (never overwritten). DRY: every guard, no pull, no python, no output.
# Markers: ZZK2DC02nhEMA-DONEZZ | -ABORT-<why>ZZ.  EXIT: 0 | 3 a step failed | 6 refused by a guard.
MODE=${1:-}
K2T_D=${K2T_D:-/f/refcv9_e3/k2dc02nhema}
. "$(dirname "$0")/k2dc02nhtail_common.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=$D/logs/ema_verdicts.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
die() { say "ZZK2DC02nhEMA-ABORT-$1ZZ $2"; exit 6; }
RUNS=/home/nvidia/r8ns/r9e2/runs
T_NH=$RUNS/E2_READ_K2DC02nh; T_K2DC02=$RUNS/E2_READ_K2DC02
RD_NH=${K2S_EMA_READ_NH:-/f/refcv9_e3/r9e2_read_K2DC02nh/pull/read}
RD_K2DC02=${K2S_EMA_READ_K2DC02:-/f/refcv9_e3/r9e2_read_K2DC02/pull/read}
TAIL_E2V=${K2T_TAIL_E2V:-/f/refcv9_e3/k2dc02nhtail/pull/stageB/read/E2_VERDICT.json}
P=$D/pull; V=$D/verdicts_ema
say "START (pid $$) mode ${MODE:-REAL} tip ${TIP:0:12}"
[ -z "$MODE" ] || [ "$MODE" = DRY ] || die USAGE "usage: k2dc02nhtail_ema_verdicts.sh [DRY]"
[ "$(th "echo up")" = up ] || die NOSSH "cannot reach tanitad-thor"
AW=$(th "grep -c 'AWAIT-S6C' $T_NH/ROLL_STATUS.txt"); [ "${AW:-0}" -ge 1 ] 2>/dev/null || die NOAWAIT "no AWAIT-S6C in $T_NH/ROLL_STATUS.txt (the K2DC02nh EMA read is not done on Thor)"
for p in $T_NH/E2_K2_pf $T_NH/E2_K0_pf $T_K2DC02/E2_K2_pf; do [ "$(th "[ -d $p ] && echo Y")" = Y ] || die THORIN "missing on Thor: $p"; done
for r in "$RD_NH" "$RD_K2DC02"; do [ -s "$r/EVAL_K2.json" ] && [ -s "$r/s6c/S6C_K2_s0.json" ] && [ -s "$r/s6c/S6C_K2_s1.json" ] || die NOREAD "no EVAL_K2.json + s6c/S6C_K2_s{0,1}.json under $r"; done
[ -s "$RD_NH/E2_VERDICT.json" ] || die NOVERDICT "no $RD_NH/E2_VERDICT.json: run the S6c / verdict step of the K2DC02nh EMA read on this box first"
[ "$MODE" = DRY ] || { [ -e "$V" ] && die EXISTS "$V exists (a read is never overwritten): MM decision, by literal path"; [ -e "$P" ] && die EXISTS "$P exists (a previous pull): MM decision, by literal path"; }
export_code; rc=$?; [ $rc = 0 ] || die EXPORT "export_code rc $rc"
say "code $C: $(cat "$C/.verified")"
MSG=$(lever_tool_guard) || die LEVERTOOL "$MSG"
[ -s "$HERE/k2dc02nhtail_lever.py" ] || die NOLEVER "no $HERE/k2dc02nhtail_lever.py"
if [ "$MODE" = DRY ]; then say "ZZK2DC02nhEMA-DRY-OKZZ all guards pass; nothing pulled, nothing run"; exit 0; fi
FREE=$(df -BM "$D" | awk 'NR==2 {gsub("M","",$4); print $4}'); [ "${FREE:-0}" -gt 2048 ] 2>/dev/null || die CDISK "$D drive free ${FREE} MB < 2048 (the three planner-roll dirs are ~0.2 GB)"
mkdir -p "$P/nh" "$P/k2dc02" "$V/lever" || die MKDIR "$P"
pull() {   # pull TAG THORDIR DSTDIR ENTRY
  local tag=$1 src=$2 dst=$3 ent=$4
  TH_TO=1800 th "cd $src && find $ent -type f -print0 | sort -z | xargs -0 sha256sum" > "$D/sha/thor_$tag.sha256"
  [ -s "$D/sha/thor_$tag.sha256" ] || die THORSHA "no sha256 list for $tag"
  timeout 3600 ssh -n -o ConnectTimeout=20 tanitad-thor "cd $src && tar -cf - $ent" | tar --force-local -xf - -C "$dst" || die PULL "tar stream failed ($tag)"
  (cd "$dst" && sha256sum -c --quiet "$D/sha/thor_$tag.sha256") > "$D/sha/check_$tag.txt" 2>&1 || die PULLSHA "$tag sha256 mismatch: $(head -3 "$D/sha/check_$tag.txt" | tr '\n' ' ')"
  [ "$(cd "$dst" && find $ent -type f | wc -l)" = "$(wc -l < "$D/sha/thor_$tag.sha256")" ] || die PULLCOUNT "$tag file count differs from Thor's list"
  say "pulled $tag: $(wc -l < "$D/sha/thor_$tag.sha256") files sha256-equal to Thor"
}
pull nh_k2 "$T_NH" "$P/nh" E2_K2_pf
pull nh_k0 "$T_NH" "$P/nh" E2_K0_pf
pull k2dc02 "$T_K2DC02" "$P/k2dc02" E2_K2_pf
gate ema || die COMMITLOW "commit free < 5632 MB for 8 h; the pull is kept"
export PYTHONPATH="$(W2 "$C/stack");$(W2 "$C/stack/scripts");$(W2 "$C/taniteval")" PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1
SPLW="$(W2 "$C/$SPL")"
FAIL=0
step() { say "ZZK2DC02nhEMA-STEP-$1 rc $2"; }
LARGS=(); [ -s "$TAIL_E2V" ] && LARGS+=(--other-e2-verdict "$(W2 "$TAIL_E2V")")
"$PY" -X utf8 "$HERE/k2dc02nhtail_lever.py" --mode ema --nh-pf "$(W2 "$P/nh/E2_K2_pf")" --k2dc02-pf "$(W2 "$P/k2dc02/E2_K2_pf")" --k0-pf "$(W2 "$P/nh/E2_K0_pf")" --dev-split "$SPLW" \
    --nh-read "$(W2 "$RD_NH")" --k2dc02-read "$(W2 "$RD_K2DC02")" --out-dir "$(W2 "$V/lever")" "${LARGS[@]}" > "$V/lever/lever.log" 2>&1 < /dev/null; LRC=$?
step lever $LRC; tail -1 "$V/lever/lever.log" | cut -c1-260 | tee -a "$LOG"
[ $LRC = 0 ] || [ $LRC = 5 ] || { FAIL=1; say "lever reading REFUSED (rc $LRC): $V/lever/lever.log"; }
OUTC=$("$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); print(d.get('reading'), d.get('rule_outcome'))" "$(W2 "$V/lever/LEVER_READING.json")" 2>/dev/null | tr -d '\r')
say "SUMMARY ema_reading_and_rule_outcome=${OUTC:-none} lever_rc=$LRC steps_failed=$FAIL out=$V"
[ $FAIL = 0 ] || { say "ZZK2DC02nhEMA-ABORT-STEPFAILEDZZ see $LOG"; exit 3; }
say "ZZK2DC02nhEMA-DONEZZ"; exit 0
