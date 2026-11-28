#!/usr/bin/env bash
# dose2_chain.sh -- the STOPBAL SECOND DOSE on box 2 (theory/stopbal/AMENDMENT_3_DOSE2.md, frozen, sha256 460fd548...; code of record 715ef8fc): BAL-G2_s0, BAL-L2_s0, BAL-G2_s1, BAL-L2_s1.
# The first dose's chain (stopbal_chain.sh) with ONLY these changes: it waits for the SLOWWM chain to end (its state file reads DONE, or STOP whose head is not a busy refusal: the rule of slowwm_chain.sh's own wait on OBD seed 3)
# instead of the OBD chain; its own state dir; its own sentinels (ZZ STOPBAL2-...); the pilot is PILOT_BAL-G2_s0; no NULL-parity step (the NULL arms are the first dose's banks, not retrained). The stage scripts, the jobs and the checker
# are the first dose's, reached through dose2/stages (wrappers and two minimal copies) and dose2/stopbal_check.py (a table layer over the first dose's checker).
# Fail closed: no STOPBAL_COMMIT (40 lowercase hex) -> refuse; STOPBAL_SSH / STOPBAL_SFTP unset, relative, missing or of the wrong kind -> refuse; on ANY refusal one line goes to $STATE/STOP.txt, exit non-zero.
# Gate / pin / token refusal of one run (stage rc 11) writes ARM_REFUSED_<tag>.txt and the chain goes on with the next independent run, ending with STOP.txt (rc 4), never DONE.txt; every other failure stops the chain.
# Sentinels in their own files only (never in chain.log): $STATE/STOP.txt ("ZZ STOPBAL2-STOP"), $STATE/DONE.txt ("ZZ STOPBAL2-DONE"), $STATE/ARM_REFUSED_<tag>.txt ("ZZ STOPBAL2-ARM-REFUSED").
# Launch:  STOPBAL_COMMIT=715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf STOPBAL_SSH=$(command -v ssh) STOPBAL_SFTP=$(command -v sftp) [STOPBAL_AMENDMENT=<file>] bash dose2_chain.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; HERE="$(cygpath -m "$HERE" 2>/dev/null || echo "$HERE")"
export STOPBAL_STATE="${STOPBAL_STATE:-C:/Users/Admin/stopbal_dose2_state}"; STATE="$STOPBAL_STATE"
STAGES_DIR="${STOPBAL_STAGES:-$HERE/stages}"; PYV="${STOPBAL_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
PULLS_DEV="${STOPBAL_PULLS:-C:/Users/Admin/t2v8f_pulls}"
BUSY_DEV="${STOPBAL_BUSY_DEV:-C:/Users/Admin/box2_ship/BOX2_BUSY.txt}"; ASIDE_DEV="${STOPBAL_ASIDE_DEV:-C:/Users/Admin/box2_ship/busy_aside}"     # the SAME busy marker the other box-2 chains claim: they exclude each other
PRED_STATE_DEV="${STOPBAL_PRED_STATE:-C:/Users/Admin/slowwm_state}"                                                                     # where the SLOWWM chain writes DONE.txt / STOP.txt (read only here)
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATE/chain.log"; }
# the state dir must be this chain's own: never the first dose's (stopbal_state) nor SLOWWM's (slowwm_state) nor the predecessor dir
for other in C:/Users/Admin/stopbal_state C:/Users/Admin/slowwm_state "$PRED_STATE_DEV"; do
  [ "$(cygpath -u "$STATE" 2>/dev/null || echo "$STATE")" != "$(cygpath -u "$other" 2>/dev/null || echo "$other")" ] || { echo "REFUSED: STOPBAL_STATE $STATE collides with $other" >&2; exit 2; }
done
mkdir -p "$STATE/done" "$ASIDE_DEV" || exit 2
refuse_early() { echo "ZZ STOPBAL2-STOP start $1 rc=$2" > "$STATE/STOP.txt"; echo "REFUSED: $1" >&2; exit "$2"; }
case "${STOPBAL_COMMIT:-}" in *[!0-9a-f]*|'') refuse_early "commit-missing (STOPBAL_COMMIT unset or not lowercase hex)" 8;; esac
[ "${#STOPBAL_COMMIT}" -eq 40 ] || refuse_early "commit-missing (STOPBAL_COMMIT is not a full 40-hex sha)" 8
export STOPBAL_COMMIT
( source "$STAGES_DIR/stopbal_common.sh" ) > /dev/null 2> "$STATE/transport.err" || refuse_early "transport-not-fail-closed ($(tr '\n' ' ' < "$STATE/transport.err" | head -c 300))" 16
if [ -e "$STATE/STOP.txt" ] && [ "${STOPBAL_RESUME:-0}" != "1" ]; then echo "REFUSED: $STATE/STOP.txt exists ($(cat "$STATE/STOP.txt")); inspect, then rerun with STOPBAL_RESUME=1" >&2; exit 2; fi
[ "${STOPBAL_RESUME:-0}" = "1" ] && [ -e "$STATE/STOP.txt" ] && mv "$STATE/STOP.txt" "$STATE/STOP.$(date +%Y%m%d_%H%M%S).txt"
for f in null_banks.py stopbal_check.py frozen/argvs_dose2.json stages/stopbal_common.sh stages/dose2_ship.sh stages/dose2_run.sh jobs/stopbal_stage_job.cmd jobs/stopbal_ship_job.cmd; do
  [ -e "$HERE/$f" ] || refuse_early "package-incomplete ($f missing in $HERE)" 8
done
chk() { "$PYV" -I -c "import sys; sys.path.insert(0, sys.argv[1]); import stopbal_check as C; print($1)" "$HERE"; }
RUN_ORDER="$(chk "' '.join(C.RUN_ORDER)")" || refuse_early "package-incomplete (the checker does not import)" 8
# ---- the banked NULL arms (reference and floor of the read) must be whole BEFORE any box-2 time is spent: bank files present, ckpt sha256 equal to the manifest and to RUN.done, NULL levers read 1.0. Zero box calls.
"$PYV" -I "$HERE/null_banks.py" --pulls "$PULLS_DEV" > "$STATE/null_banks.txt" 2>&1 || refuse_early "null-bank-missing-or-changed ($(tr '\n' ' ' < "$STATE/null_banks.txt" | head -c 400))" 8
# ---- wait for the SLOWWM chain to end, and for nothing to hold box 2.
#   ended = DONE.txt (first bytes ZZ SLOWWM-DONE), or a non-empty STOP.txt whose head (the text before " :: ") does NOT end in rc=12. A STOP with rc=12 (busy / GPU) means the predecessor never ran its stage: it may be resumed, so this chain keeps waiting.
pred_state() {   # prints DONE | STOP | STOP-BUSY | RUNNING   (the rule of slowwm_chain.sh)
  local f="$PRED_STATE_DEV/STOP.txt" head1
  if [ -s "$f" ]; then
    head1="$(head -n 1 "$f" | sed 's/ :: .*//')"
    if printf '%s' "$head1" | grep -Eq ' rc=12 *$'; then echo STOP-BUSY; else echo STOP; fi; return
  fi
  if [ -s "$PRED_STATE_DEV/DONE.txt" ] && head -c 40 "$PRED_STATE_DEV/DONE.txt" | grep -q '^ZZ SLOWWM-DONE'; then echo DONE; return; fi
  echo RUNNING
}
n=0; while :; do
  ps_="$(pred_state)"
  if { [ "$ps_" = DONE ] || [ "$ps_" = STOP ]; } && [ ! -e "$BUSY_DEV" ]; then log "the SLOWWM chain ended ($ps_)"; break; fi
  [ "$n" -lt "${STOPBAL_WAIT_POLLS:-1152}" ] || refuse_early "predecessor-not-ended (no SLOWWM DONE and no non-busy SLOWWM STOP in $PRED_STATE_DEV, or $BUSY_DEV exists, after $n polls; last state $ps_)" 12
  n=$((n+1)); log "waiting for the SLOWWM chain to end (poll $n, state $ps_)"; sleep "${STOPBAL_WAIT_POLL_S:-300}"
done
if ! ( set -o noclobber; echo "stopbal dose2 chain $$ $(date '+%F %T')" > "$BUSY_DEV" ) 2>/dev/null; then echo "REFUSED: $BUSY_DEV exists (box 2 is busy)" >&2; exit 12; fi
OWN_MARK="$(cat "$BUSY_DEV")"
release_box() { if [ -e "$BUSY_DEV" ] && [ "$(cat "$BUSY_DEV" 2>/dev/null)" = "$OWN_MARK" ]; then mv "$BUSY_DEV" "$ASIDE_DEV/BOX2_BUSY.stopbal_dose2.$(date +%Y%m%d_%H%M%S).txt"; fi; }
trap 'release_box' EXIT
stop() {                   # stop <stage> <reason> <rc>
  local detail=""; [ -s "$STATE/stage_detail.txt" ] && detail=" :: $(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
  echo "ZZ STOPBAL2-STOP $1 $2 rc=$3$detail" > "$STATE/STOP.txt"; log "stop at stage $1: $2 (rc $3)"; release_box; exit "$3"
}
dispatch() { rm -f "$STATE/stage_detail.txt"; case "$1" in ship) bash "$STAGES_DIR/dose2_ship.sh";; pilot) bash "$STAGES_DIR/dose2_run.sh" PILOT_BAL-G2_s0;; *) bash "$STAGES_DIR/dose2_run.sh" "$1";; esac; }
map_rc() {   # <stage> <rc>: a non-zero rc stops the chain, except an arm-specific gate refusal of a full run (handled by the caller)
  case "$2" in
    11) stop "$1" "gate-or-pin-refused" 4;; 12) stop "$1" "job-or-transfer-failed" 5;; 13) stop "$1" "disk-low" 6;; 14) stop "$1" "box2-busy-or-gpu-not-idle" 12;; 15) stop "$1" "pull-sha-mismatch" 7;;
    16) stop "$1" "package-commit-or-transport" 8;; 17) stop "$1" "ship-refused" 9;; 18) stop "$1" "pilot-unreadable" 10;; 19) stop "$1" "box2-timeout-or-unreadable" 11;; 23) stop "$1" "blocked-tag" 13;;
    *) stop "$1" "stage-crashed" 5;;
  esac
}
run_stage() {   # run_stage <name>: ship | pilot
  local s="$1" rc
  if [ -e "$STATE/done/$s.done" ]; then log "skip $s (done)"; return 0; fi
  log "start $s"; dispatch "$s"; rc=$?
  if [ "$rc" -eq 0 ]; then echo "$s $(date '+%F %T')" > "$STATE/done/$s.done"; log "done $s"; else map_rc "$s" "$rc"; fi
}
run_stage ship
run_stage pilot
P="$STATE/pilot/PILOT.verified.json"; [ -s "$P" ] || stop pilot "no-verified-pilot ($P missing)" 10
"$PYV" -I "$HERE/stopbal_check.py" permit --pilot "$P" --commit "$STOPBAL_COMMIT" ${STOPBAL_AMENDMENT:+--amendment "$STOPBAL_AMENDMENT"} > "$STATE/permit.txt" 2>&1 || { echo "$(tr '\n' ' ' < "$STATE/permit.txt" | head -c 400)" > "$STATE/stage_detail.txt"; stop permit "12h-rule-refused" 3; }
log "$(cat "$STATE/permit.txt")"
ORDER=($RUN_ORDER)
REFUSED=""
i=0
while [ "$i" -lt "${#ORDER[@]}" ]; do
  t="${ORDER[$i]}"; i=$((i+1))
  if [ -e "$STATE/done/$t.done" ]; then log "skip $t (done)"; continue; fi
  log "start $t"; rm -f "$STATE/stage_detail.txt"; bash "$STAGES_DIR/dose2_run.sh" "$t"; rc=$?
  if [ "$rc" -eq 0 ]; then echo "$t $(date '+%F %T')" > "$STATE/done/$t.done"; log "done $t"
  elif [ "$rc" -eq 11 ]; then
    d=""; [ -s "$STATE/stage_detail.txt" ] && d="$(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
    echo "ZZ STOPBAL2-ARM-REFUSED $t gate-or-pin-refused rc=4 :: $d" > "$STATE/ARM_REFUSED_$t.txt"; log "arm $t refused at its gate; the next independent arm goes on"; REFUSED="$REFUSED $t"
  else map_rc "$t" "$rc"; fi
done
if [ -n "$REFUSED" ]; then echo "ZZ STOPBAL2-STOP runs arms-refused:$REFUSED rc=4" > "$STATE/STOP.txt"; log "finished with refused arms:$REFUSED"; release_box; exit 4; fi
echo "ZZ STOPBAL2-DONE $(date '+%F %T') commit=$STOPBAL_COMMIT runs=${ORDER[*]}" > "$STATE/DONE.txt"; log "all runs banked"
exit 0
