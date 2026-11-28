#!/usr/bin/env bash
# otband_chain.sh -- the O/T band-split retrain on box 2, in the card's order (theory/PREREG_V8F_OTBAND.md):
#   wait until the ladder chain has ended and box 2's GPU is idle -> ship (own tree) -> PILOT (200 steps of OB-BAND_s0, projected total, 72 h rule)
#   -> OB-CTL_s0, OB-BAND_s0, OB-CTL_s1, OB-BAND_s1, OB-BAND_rep.  Each run: argv pin, gate (selfcheck + BUILT-lever assertions + per-run token), training under the token, post check, bank, cleanup.
#   OB-BAND-TOK_s0/_s1 are in the tag table but BLOCKED (no stok dump); adding them is a data change in otband_check.py TABLE, not a code change.
# Fail closed: no OTBAND_COMMIT (40 lowercase hex) -> refuse; OTBAND_SSH / OTBAND_SFTP unset, relative, missing or of the wrong kind -> refuse (stages/otband_common.sh); on ANY refusal one line goes to $STATE/STOP.txt, exit non-zero.
# Failure behaviour: a GATE / PIN / TOKEN refusal of one run (stage rc 11) writes $STATE/ARM_REFUSED_<tag>.txt and the chain goes on with the next run (the arms are independent, nothing has used the GPU);
#   every other failure (ship, pilot, 72 h rule, training, post check, disk, pull, transport, timeout, box busy) stops the chain with $STATE/STOP.txt. A chain with a refused arm ends with STOP.txt, never DONE.txt.
# Sentinels live in their own files only (never in chain.log): $STATE/STOP.txt (prefix "ZZ OTBAND-STOP"), $STATE/DONE.txt (prefix "ZZ OTBAND-DONE"), $STATE/ARM_REFUSED_<tag>.txt (prefix "ZZ OTBAND-ARM-REFUSED").
# Launch (README has the full line):  OTBAND_COMMIT=<40 hex> OTBAND_SSH=$(command -v ssh) OTBAND_SFTP=$(command -v sftp) [OTBAND_AMENDMENT=<file>] bash otband_chain.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; HERE="$(cygpath -m "$HERE" 2>/dev/null || echo "$HERE")"
export OTBAND_STATE="${OTBAND_STATE:-C:/Users/Admin/otband_state}"; STATE="$OTBAND_STATE"
STAGES_DIR="${OTBAND_STAGES:-$HERE/stages}"; PYV="${OTBAND_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
BUSY_DEV="${OTBAND_BUSY_DEV:-C:/Users/Admin/box2_ship/BOX2_BUSY.txt}"; ASIDE_DEV="${OTBAND_ASIDE_DEV:-C:/Users/Admin/box2_ship/busy_aside}"     # the SAME busy marker the ladder claims: the two chains exclude each other
LADDER_STATE_DEV="${OTBAND_LADDER_STATE:-C:/Users/Admin/ladder_state}"                                                                       # where ladder_full.sh writes FULL_DONE.txt / FULL_STOP.txt (read only here)
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATE/chain.log"; }
mkdir -p "$STATE/done" "$ASIDE_DEV" || exit 2
refuse_early() { echo "ZZ OTBAND-STOP start $1 rc=$2" > "$STATE/STOP.txt"; echo "REFUSED: $1" >&2; exit "$2"; }
case "${OTBAND_COMMIT:-}" in *[!0-9a-f]*|'') refuse_early "commit-missing (OTBAND_COMMIT unset or not lowercase hex)" 8;; esac
[ "${#OTBAND_COMMIT}" -eq 40 ] || refuse_early "commit-missing (OTBAND_COMMIT is not a full 40-hex sha)" 8
export OTBAND_COMMIT
( source "$STAGES_DIR/otband_common.sh" ) > /dev/null 2> "$STATE/transport.err" || refuse_early "transport-not-fail-closed ($(tr '\n' ' ' < "$STATE/transport.err" | head -c 300))" 16
if [ -e "$STATE/STOP.txt" ] && [ "${OTBAND_RESUME:-0}" != "1" ]; then echo "REFUSED: $STATE/STOP.txt exists ($(cat "$STATE/STOP.txt")); inspect, then rerun with OTBAND_RESUME=1" >&2; exit 2; fi
[ "${OTBAND_RESUME:-0}" = "1" ] && [ -e "$STATE/STOP.txt" ] && mv "$STATE/STOP.txt" "$STATE/STOP.$(date +%Y%m%d_%H%M%S).txt"
for f in otband_check.py otband_ship_tree.py otband_refresh.py frozen/argvs_otband.json stages/otband_common.sh stages/otband_ship.sh stages/otband_run.sh jobs/otband_stage_job.cmd jobs/otband_ship_job.cmd; do
  [ -e "$HERE/$f" ] || refuse_early "package-incomplete ($f missing in $HERE)" 8
done
RUN_ORDER="$("$PYV" -I -c "import sys; sys.path.insert(0, sys.argv[1]); import otband_check as C; print(' '.join(C.RUN_ORDER))" "$HERE")" || refuse_early "package-incomplete (the checker does not import)" 8
# ---- wait for the ladder chain to end (its finished marker) and for nothing to hold box 2; refuse otherwise
ladder_ended() {
  [ -e "$BUSY_DEV" ] && return 1                                                                                                              # a chain (the ladder's, or another) holds box 2
  if [ -s "$LADDER_STATE_DEV/FULL_DONE.txt" ] && head -c 40 "$LADDER_STATE_DEV/FULL_DONE.txt" | grep -q '^ZZ LADDER-FULL-DONE'; then return 0; fi
  if [ -s "$LADDER_STATE_DEV/FULL_STOP.txt" ] && head -c 40 "$LADDER_STATE_DEV/FULL_STOP.txt" | grep -q '^ZZ LADDER-FULL-STOP'; then return 0; fi   # the ladder stopped itself and released box 2
  return 1
}
n=0; while ! ladder_ended; do
  [ "$n" -lt "${OTBAND_WAIT_POLLS:-1152}" ] || refuse_early "ladder-not-ended (no ZZ LADDER-FULL-DONE / ZZ LADDER-FULL-STOP in $LADDER_STATE_DEV, or $BUSY_DEV exists, after $n polls)" 12
  n=$((n+1)); log "waiting for the ladder chain to end (poll $n)"; sleep "${OTBAND_WAIT_POLL_S:-300}"
done
if ! ( set -o noclobber; echo "otband chain $$ $(date '+%F %T')" > "$BUSY_DEV" ) 2>/dev/null; then echo "REFUSED: $BUSY_DEV exists (box 2 is busy)" >&2; exit 12; fi
OWN_MARK="$(cat "$BUSY_DEV")"
release_box() { if [ -e "$BUSY_DEV" ] && [ "$(cat "$BUSY_DEV" 2>/dev/null)" = "$OWN_MARK" ]; then mv "$BUSY_DEV" "$ASIDE_DEV/BOX2_BUSY.otband.$(date +%Y%m%d_%H%M%S).txt"; fi; }
trap 'release_box' EXIT
stop() {                   # stop <stage> <reason> <rc>
  local detail=""; [ -s "$STATE/stage_detail.txt" ] && detail=" :: $(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
  echo "ZZ OTBAND-STOP $1 $2 rc=$3$detail" > "$STATE/STOP.txt"; log "stop at stage $1: $2 (rc $3)"; release_box; exit "$3"
}
dispatch() { rm -f "$STATE/stage_detail.txt"; case "$1" in ship) bash "$STAGES_DIR/otband_ship.sh";; pilot) bash "$STAGES_DIR/otband_run.sh" PILOT_OB-BAND_s0;; *) bash "$STAGES_DIR/otband_run.sh" "$1";; esac; }
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
"$PYV" -I "$HERE/otband_check.py" permit --pilot "$P" --commit "$OTBAND_COMMIT" ${OTBAND_AMENDMENT:+--amendment "$OTBAND_AMENDMENT"} > "$STATE/permit.txt" 2>&1 || { echo "$(tr '\n' ' ' < "$STATE/permit.txt" | head -c 400)" > "$STATE/stage_detail.txt"; stop permit "72h-rule-refused" 3; }
log "$(cat "$STATE/permit.txt")"
REFUSED=""
for t in $RUN_ORDER; do
  if [ -e "$STATE/done/$t.done" ]; then log "skip $t (done)"; continue; fi
  log "start $t"; rm -f "$STATE/stage_detail.txt"; bash "$STAGES_DIR/otband_run.sh" "$t"; rc=$?
  if [ "$rc" -eq 0 ]; then echo "$t $(date '+%F %T')" > "$STATE/done/$t.done"; log "done $t"
  elif [ "$rc" -eq 11 ]; then
    d=""; [ -s "$STATE/stage_detail.txt" ] && d="$(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
    echo "ZZ OTBAND-ARM-REFUSED $t gate-or-pin-refused rc=4 :: $d" > "$STATE/ARM_REFUSED_$t.txt"; log "arm $t refused at its gate; the next independent arm goes on"; REFUSED="$REFUSED $t"
  else map_rc "$t" "$rc"; fi
done
if [ -n "$REFUSED" ]; then echo "ZZ OTBAND-STOP runs arms-refused:$REFUSED rc=4" > "$STATE/STOP.txt"; log "finished with refused arms:$REFUSED"; release_box; exit 4; fi
echo "ZZ OTBAND-DONE $(date '+%F %T') commit=$OTBAND_COMMIT runs=$RUN_ORDER" > "$STATE/DONE.txt"; log "all runs banked"
exit 0
