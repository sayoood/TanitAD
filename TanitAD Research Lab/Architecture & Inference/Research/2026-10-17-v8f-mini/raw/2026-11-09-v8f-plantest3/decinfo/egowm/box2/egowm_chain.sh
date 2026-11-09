#!/usr/bin/env bash
# egowm_chain.sh -- the O/T band-split retrain on box 2, in the card's order (theory/PREREG_V8F_EGOWM.md):
#   wait until the OTBAND chain has ended (DONE; a STOP stops this chain too) and box 2 is free -> ship (own tree) -> PILOT (200 steps of E-KIN_s0, projected total, 72 h rule)
#   -> E-KIN_s0, E-NOW_s0, E-KIN_s1, E-NOW_s1, E-KIN_rep_s2.  Each run: argv pin, gate (selfcheck + BUILT-lever assertions + per-run token), training under the token, post check, bank, cleanup.
# Fail closed: no EGOWM_COMMIT (40 lowercase hex) -> refuse; EGOWM_SSH / EGOWM_SFTP unset, relative, missing or of the wrong kind -> refuse (stages/egowm_common.sh); on ANY refusal one line goes to $STATE/STOP.txt, exit non-zero.
# Failure behaviour: a GATE / PIN / TOKEN refusal of one run (stage rc 11) writes $STATE/ARM_REFUSED_<tag>.txt and the chain goes on with the next run (the arms are independent, nothing has used the GPU);
#   every other failure (ship, pilot, 72 h rule, training, post check, disk, pull, transport, timeout, box busy) stops the chain with $STATE/STOP.txt. A chain with a refused arm ends with STOP.txt, never DONE.txt.
# Sentinels live in their own files only (never in chain.log): $STATE/STOP.txt (prefix "ZZ EGOWM-STOP"), $STATE/DONE.txt (prefix "ZZ EGOWM-DONE"), $STATE/ARM_REFUSED_<tag>.txt (prefix "ZZ EGOWM-ARM-REFUSED").
# Launch (README has the full line):  EGOWM_COMMIT=<40 hex> EGOWM_SSH=$(command -v ssh) EGOWM_SFTP=$(command -v sftp) [EGOWM_AMENDMENT=<file>] bash egowm_chain.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; HERE="$(cygpath -m "$HERE" 2>/dev/null || echo "$HERE")"
export EGOWM_STATE="${EGOWM_STATE:-C:/Users/Admin/egowm_state}"; STATE="$EGOWM_STATE"
STAGES_DIR="${EGOWM_STAGES:-$HERE/stages}"; PYV="${EGOWM_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
BUSY_DEV="${EGOWM_BUSY_DEV:-C:/Users/Admin/box2_ship/BOX2_BUSY.txt}"; ASIDE_DEV="${EGOWM_ASIDE_DEV:-C:/Users/Admin/box2_ship/busy_aside}"     # the SAME busy marker the other chains claim: the two chains exclude each other
OTBAND_STATE_DEV="${EGOWM_OTBAND_STATE:-C:/Users/Admin/otband_state}"                                                                     # where the OTBAND chain writes DONE.txt / STOP.txt (read only here)
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATE/chain.log"; }
mkdir -p "$STATE/done" "$ASIDE_DEV" || exit 2
refuse_early() { echo "ZZ EGOWM-STOP start $1 rc=$2" > "$STATE/STOP.txt"; echo "REFUSED: $1" >&2; exit "$2"; }
case "${EGOWM_COMMIT:-}" in *[!0-9a-f]*|'') refuse_early "commit-missing (EGOWM_COMMIT unset or not lowercase hex)" 8;; esac
[ "${#EGOWM_COMMIT}" -eq 40 ] || refuse_early "commit-missing (EGOWM_COMMIT is not a full 40-hex sha)" 8
export EGOWM_COMMIT
( source "$STAGES_DIR/egowm_common.sh" ) > /dev/null 2> "$STATE/transport.err" || refuse_early "transport-not-fail-closed ($(tr '\n' ' ' < "$STATE/transport.err" | head -c 300))" 16
if [ -e "$STATE/STOP.txt" ] && [ "${EGOWM_RESUME:-0}" != "1" ]; then echo "REFUSED: $STATE/STOP.txt exists ($(cat "$STATE/STOP.txt")); inspect, then rerun with EGOWM_RESUME=1" >&2; exit 2; fi
[ "${EGOWM_RESUME:-0}" = "1" ] && [ -e "$STATE/STOP.txt" ] && mv "$STATE/STOP.txt" "$STATE/STOP.$(date +%Y%m%d_%H%M%S).txt"
for f in egowm_check.py egowm_ship_tree.py egowm_refresh.py frozen/argvs_egowm.json frozen/export_list_egowm.txt stages/egowm_common.sh stages/egowm_ship.sh stages/egowm_run.sh jobs/egowm_stage_job.cmd jobs/egowm_ship_job.cmd; do
  [ -e "$HERE/$f" ] || refuse_early "package-incomplete ($f missing in $HERE)" 8
done
RUN_ORDER="$("$PYV" -I -c "import sys; sys.path.insert(0, sys.argv[1]); import egowm_check as C; print(' '.join(C.RUN_ORDER))" "$HERE")" || refuse_early "package-incomplete (the checker does not import)" 8
# ---- wait for the predecessor (the OTBAND chain) to end, and for nothing to hold box 2; a predecessor that ended in STOP stops this chain too (it must not take the box)
pred_state() {   # prints DONE | STOP | RUNNING
  if [ -s "$OTBAND_STATE_DEV/STOP.txt" ] && head -c 40 "$OTBAND_STATE_DEV/STOP.txt" | grep -q '^ZZ OTBAND-STOP'; then echo STOP; return; fi
  if [ -s "$OTBAND_STATE_DEV/DONE.txt" ] && head -c 40 "$OTBAND_STATE_DEV/DONE.txt" | grep -q '^ZZ OTBAND-DONE'; then echo DONE; return; fi
  echo RUNNING
}
n=0; while :; do
  ps_="$(pred_state)"
  [ "$ps_" != STOP ] || refuse_early "predecessor-stopped (the OTBAND chain ended in STOP: $(head -c 200 "$OTBAND_STATE_DEV/STOP.txt" | tr '\r\n' '  '); this chain does not take box 2)" 12
  if [ "$ps_" = DONE ] && [ ! -e "$BUSY_DEV" ]; then break; fi
  [ "$n" -lt "${EGOWM_WAIT_POLLS:-1152}" ] || refuse_early "predecessor-not-ended (no ZZ OTBAND-DONE / ZZ OTBAND-STOP in $OTBAND_STATE_DEV, or $BUSY_DEV exists, after $n polls)" 12
  n=$((n+1)); log "waiting for the OTBAND chain to end (poll $n, state $ps_)"; sleep "${EGOWM_WAIT_POLL_S:-300}"
done
if ! ( set -o noclobber; echo "egowm chain $$ $(date '+%F %T')" > "$BUSY_DEV" ) 2>/dev/null; then echo "REFUSED: $BUSY_DEV exists (box 2 is busy)" >&2; exit 12; fi
OWN_MARK="$(cat "$BUSY_DEV")"
release_box() { if [ -e "$BUSY_DEV" ] && [ "$(cat "$BUSY_DEV" 2>/dev/null)" = "$OWN_MARK" ]; then mv "$BUSY_DEV" "$ASIDE_DEV/BOX2_BUSY.egowm.$(date +%Y%m%d_%H%M%S).txt"; fi; }
trap 'release_box' EXIT
stop() {                   # stop <stage> <reason> <rc>
  local detail=""; [ -s "$STATE/stage_detail.txt" ] && detail=" :: $(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
  echo "ZZ EGOWM-STOP $1 $2 rc=$3$detail" > "$STATE/STOP.txt"; log "stop at stage $1: $2 (rc $3)"; release_box; exit "$3"
}
dispatch() { rm -f "$STATE/stage_detail.txt"; case "$1" in ship) bash "$STAGES_DIR/egowm_ship.sh";; pilot) bash "$STAGES_DIR/egowm_run.sh" PILOT_E-KIN_s0;; *) bash "$STAGES_DIR/egowm_run.sh" "$1";; esac; }
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
"$PYV" -I "$HERE/egowm_check.py" permit --pilot "$P" --commit "$EGOWM_COMMIT" ${EGOWM_AMENDMENT:+--amendment "$EGOWM_AMENDMENT"} > "$STATE/permit.txt" 2>&1 || { echo "$(tr '\n' ' ' < "$STATE/permit.txt" | head -c 400)" > "$STATE/stage_detail.txt"; stop permit "72h-rule-refused" 3; }
log "$(cat "$STATE/permit.txt")"
REFUSED=""
for t in $RUN_ORDER; do
  if [ -e "$STATE/done/$t.done" ]; then log "skip $t (done)"; continue; fi
  log "start $t"; rm -f "$STATE/stage_detail.txt"; bash "$STAGES_DIR/egowm_run.sh" "$t"; rc=$?
  if [ "$rc" -eq 0 ]; then echo "$t $(date '+%F %T')" > "$STATE/done/$t.done"; log "done $t"
  elif [ "$rc" -eq 11 ]; then
    d=""; [ -s "$STATE/stage_detail.txt" ] && d="$(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
    echo "ZZ EGOWM-ARM-REFUSED $t gate-or-pin-refused rc=4 :: $d" > "$STATE/ARM_REFUSED_$t.txt"; log "arm $t refused at its gate; the next independent arm goes on"; REFUSED="$REFUSED $t"
  else map_rc "$t" "$rc"; fi
done
if [ -n "$REFUSED" ]; then echo "ZZ EGOWM-STOP runs arms-refused:$REFUSED rc=4" > "$STATE/STOP.txt"; log "finished with refused arms:$REFUSED"; release_box; exit 4; fi
echo "ZZ EGOWM-DONE $(date '+%F %T') commit=$EGOWM_COMMIT runs=$RUN_ORDER" > "$STATE/DONE.txt"; log "all runs banked"
exit 0
