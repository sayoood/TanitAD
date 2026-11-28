#!/usr/bin/env bash
# slowwm_chain.sh -- the SLOW / PCA-B basis runs on box 2, in the card's order (theory/slowwm/PREREG_V8F_SLOWWM.md, sha256 7c2775e3...94ca; code of record f7e790ab57e10a9004b04b206195a0ae5a5295b8):
#   wait until the OBD seed-3 chain has ended (DONE, or a STOP that is NOT a busy stop) and the shared busy marker is free, claim it -> ship (own tree + slow_basis.py + clips8.json) -> BASIS (slow_basis.py fit, cache slow+pca x train+eval, check3 = card control 3; refuses below 6 GB free)
#   -> PILOT (200 steps of SLOW_s0, gates all five full argvs, projected total, 72 h rule) -> SLOW_s0, PCA-B_s0, SLOW_s1, PCA-B_s1, SLOW_rep.  Each run: argv pin, gate (selfcheck + BUILT-lever assertions incl. wm_basis_readback + per-run token), training under the token, post check, bank, cleanup.
# Fail closed: no SLOWWM_COMMIT (40 lowercase hex) -> refuse; SLOWWM_SSH / SLOWWM_SFTP unset, relative, missing or of the wrong kind -> refuse (stages/slowwm_common.sh); on ANY refusal one line goes to $STATE/STOP.txt, exit non-zero.
# Failure behaviour: a GATE / PIN / TOKEN refusal of one run (stage rc 11) writes $STATE/ARM_REFUSED_<tag>.txt and the chain goes on with the next run (the arms are independent, nothing has used the GPU);
#   every other failure (ship, basis, pilot, 72 h rule, training, post check, disk, pull, transport, timeout, box busy) stops the chain with $STATE/STOP.txt. A chain with a refused arm ends with STOP.txt, never DONE.txt.
# Sentinels live in their own files only (never in chain.log): $STATE/STOP.txt, $STATE/DONE.txt, $STATE/ARM_REFUSED_<tag>.txt.
# Launch (README has the full line):  SLOWWM_COMMIT=<40 hex> SLOWWM_SSH=$(command -v ssh) SLOWWM_SFTP=$(command -v sftp) [SLOWWM_AMENDMENT=<file>] bash slowwm_chain.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; HERE="$(cygpath -m "$HERE" 2>/dev/null || echo "$HERE")"
export SLOWWM_STATE="${SLOWWM_STATE:-C:/Users/Admin/slowwm_state}"; STATE="$SLOWWM_STATE"
STAGES_DIR="${SLOWWM_STAGES:-$HERE/stages}"; PYV="${SLOWWM_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
BUSY_DEV="${SLOWWM_BUSY_DEV:-C:/Users/Admin/box2_ship/BOX2_BUSY.txt}"; ASIDE_DEV="${SLOWWM_ASIDE_DEV:-C:/Users/Admin/box2_ship/busy_aside}"     # the SAME busy marker the other chains claim: the chains exclude each other
PRED_STATE_DEV="${SLOWWM_PRED_STATE:-C:/Users/Admin/obd_seed3_state}"                                                                      # where the predecessor chain writes DONE.txt / STOP.txt (read only here)
PRED_DONE_PREFIX="${SLOWWM_PRED_DONE_PREFIX:-ZZ OBD3-DONE}"                                                                                 # first bytes of the predecessor's DONE.txt (STOPBAL second dose: ZZ STOPBAL2-DONE)
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATE/chain.log"; }
mkdir -p "$STATE/done" "$ASIDE_DEV" || exit 2
refuse_early() { echo "ZZ SLOWWM-STOP start $1 rc=$2" > "$STATE/STOP.txt"; echo "REFUSED: $1" >&2; exit "$2"; }
case "${SLOWWM_COMMIT:-}" in *[!0-9a-f]*|'') refuse_early "commit-missing (SLOWWM_COMMIT unset or not lowercase hex)" 8;; esac
[ "${#SLOWWM_COMMIT}" -eq 40 ] || refuse_early "commit-missing (SLOWWM_COMMIT is not a full 40-hex sha)" 8
export SLOWWM_COMMIT
for hook in SLOWWM_TEST_COUNTS SLOWWM_TEST_BASIS SLOWWM_FAKE_FREE_GB; do         # test hooks of the checker: only with the stub transport (SLOWWM_DRY=1); a real launch with one set is refused
  [ -z "${!hook:-}" ] || [ "${SLOWWM_DRY:-0}" = 1 ] || refuse_early "test-hook-in-real-run ($hook is set without SLOWWM_DRY=1)" 8
done
( source "$STAGES_DIR/slowwm_common.sh" ) > /dev/null 2> "$STATE/transport.err" || refuse_early "transport-not-fail-closed ($(tr '\n' ' ' < "$STATE/transport.err" | head -c 300))" 16
if [ -e "$STATE/STOP.txt" ] && [ "${SLOWWM_RESUME:-0}" != "1" ]; then echo "REFUSED: $STATE/STOP.txt exists ($(cat "$STATE/STOP.txt")); inspect, then rerun with SLOWWM_RESUME=1" >&2; exit 2; fi
[ "${SLOWWM_RESUME:-0}" = "1" ] && [ -e "$STATE/STOP.txt" ] && mv "$STATE/STOP.txt" "$STATE/STOP.$(date +%Y%m%d_%H%M%S).txt"
for f in slowwm_check.py slowwm_ship_tree.py slowwm_refresh.py frozen/argvs_slowwm.json frozen/export_list_slowwm.txt stages/slowwm_common.sh stages/slowwm_ship.sh stages/slowwm_run.sh stages/slowwm_basis.sh jobs/slowwm_stage_job.cmd jobs/slowwm_ship_job.cmd; do
  [ -e "$HERE/$f" ] || refuse_early "package-incomplete ($f missing in $HERE)" 8
done
for f in "$HERE/../../slowwm/slow_basis.py" "$HERE/../../ladder/clips8.json"; do
  [ -e "$f" ] || refuse_early "package-incomplete ($f missing: the basis code is shipped from theory/)" 8
done
RUN_ORDER="$("$PYV" -I -c "import sys; sys.path.insert(0, sys.argv[1]); import slowwm_check as C; print(' '.join(C.RUN_ORDER))" "$HERE")" || refuse_early "package-incomplete (the checker does not import)" 8
# ---- wait for the predecessor (the OBD seed-3 chain) to end, and for nothing to hold box 2.
#   (the predecessor is SLOWWM_PRED_STATE + SLOWWM_PRED_DONE_PREFIX; defaults: the OBD seed-3 chain, ZZ OBD3-DONE; for the STOPBAL second dose: C:/Users/Admin/stopbal_dose2_state and "ZZ STOPBAL2-DONE")
#   ended = DONE.txt (first bytes = the DONE prefix), or a non-empty STOP.txt whose head (the text before " :: ") does NOT end in rc=12. A STOP with rc=12 (busy / GPU) means the predecessor never ran its stage: it may be resumed, so this chain keeps waiting.
pred_state() {   # prints DONE | STOP | STOP-BUSY | RUNNING
  local f="$PRED_STATE_DEV/STOP.txt" head1
  if [ -s "$f" ]; then
    head1="$(head -n 1 "$f" | sed 's/ :: .*//')"
    if printf '%s' "$head1" | grep -Eq ' rc=12 *$'; then echo STOP-BUSY; else echo STOP; fi; return
  fi
  if [ -s "$PRED_STATE_DEV/DONE.txt" ] && [ "$(head -c "${#PRED_DONE_PREFIX}" "$PRED_STATE_DEV/DONE.txt")" = "$PRED_DONE_PREFIX" ]; then echo DONE; return; fi
  echo RUNNING
}
n=0; while :; do
  ps_="$(pred_state)"
  if { [ "$ps_" = DONE ] || [ "$ps_" = STOP ]; } && [ ! -e "$BUSY_DEV" ]; then log "the predecessor chain ended ($ps_)"; break; fi
  [ "$n" -lt "${SLOWWM_WAIT_POLLS:-1152}" ] || refuse_early "predecessor-not-ended (no predecessor DONE and no non-busy predecessor STOP in $PRED_STATE_DEV, or $BUSY_DEV exists, after $n polls; last state $ps_)" 12
  n=$((n+1)); log "waiting for the predecessor chain to end (poll $n, state $ps_)"; sleep "${SLOWWM_WAIT_POLL_S:-300}"
done
if ! ( set -o noclobber; echo "slowwm chain $$ $(date '+%F %T')" > "$BUSY_DEV" ) 2>/dev/null; then echo "REFUSED: $BUSY_DEV exists (box 2 is busy)" >&2; exit 12; fi
OWN_MARK="$(cat "$BUSY_DEV")"
release_box() { if [ -e "$BUSY_DEV" ] && [ "$(cat "$BUSY_DEV" 2>/dev/null)" = "$OWN_MARK" ]; then mv "$BUSY_DEV" "$ASIDE_DEV/BOX2_BUSY.slowwm.$(date +%Y%m%d_%H%M%S).txt"; fi; }
trap 'release_box' EXIT
stop() {                   # stop <stage> <reason> <rc>
  local detail=""; [ -s "$STATE/stage_detail.txt" ] && detail=" :: $(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
  echo "ZZ SLOWWM-STOP $1 $2 rc=$3$detail" > "$STATE/STOP.txt"; log "stop at stage $1: $2 (rc $3)"; release_box; exit "$3"
}
dispatch() { rm -f "$STATE/stage_detail.txt"; case "$1" in ship) bash "$STAGES_DIR/slowwm_ship.sh";; pilot) bash "$STAGES_DIR/slowwm_run.sh" PILOT_SLOW_s0;; basis) bash "$STAGES_DIR/slowwm_basis.sh";; *) bash "$STAGES_DIR/slowwm_run.sh" "$1";; esac; }
map_rc() {   # <stage> <rc>: a non-zero rc stops the chain, except an arm-specific gate refusal of a full run (handled by the caller)
  case "$2" in
    11) stop "$1" "gate-or-pin-refused" 4;; 12) stop "$1" "job-or-transfer-failed" 5;; 13) stop "$1" "disk-low" 6;; 14) stop "$1" "box2-busy-or-gpu-not-idle" 12;; 15) stop "$1" "pull-sha-mismatch" 7;;
    16) stop "$1" "package-commit-or-transport" 8;; 17) stop "$1" "ship-refused" 9;; 18) stop "$1" "pilot-unreadable" 10;; 19) stop "$1" "box2-timeout-or-unreadable" 11;; 23) stop "$1" "blocked-tag" 13;;
    *) stop "$1" "stage-crashed" 5;;
  esac
}
run_stage() {   # run_stage <name>: ship | basis | pilot
  local s="$1" rc
  local ck; ck="$(sha256sum "$HERE/slowwm_check.py" | cut -d' ' -f1)"
  # the ship stage puts the checker on box 2: a done marker written for another checker (or by an older chain, which wrote none) does not count, so an amended package is shipped again before the basis stage re-hashes it there. Nothing is deleted: the marker is rewritten.
  if [ -e "$STATE/done/$s.done" ] && { [ "$s" != ship ] || grep -q "checker=$ck" "$STATE/done/$s.done"; }; then log "skip $s (done)"; return 0; fi
  [ ! -e "$STATE/done/$s.done" ] || log "$s was done for another checker: shipping again"
  log "start $s"; dispatch "$s"; rc=$?
  if [ "$rc" -eq 0 ]; then echo "$s $(date '+%F %T') checker=$ck" > "$STATE/done/$s.done"; log "done $s"; else map_rc "$s" "$rc"; fi
}
run_stage ship
run_stage basis
run_stage pilot
P="$STATE/pilot/PILOT.verified.json"; [ -s "$P" ] || stop pilot "no-verified-pilot ($P missing)" 10
"$PYV" -I "$HERE/slowwm_check.py" permit --pilot "$P" --commit "$SLOWWM_COMMIT" ${SLOWWM_AMENDMENT:+--amendment "$SLOWWM_AMENDMENT"} > "$STATE/permit.txt" 2>&1 || { echo "$(tr '\n' ' ' < "$STATE/permit.txt" | head -c 400)" > "$STATE/stage_detail.txt"; stop permit "72h-rule-refused" 3; }
log "$(cat "$STATE/permit.txt")"
REFUSED=""
for t in $RUN_ORDER; do
  if [ -e "$STATE/done/$t.done" ]; then log "skip $t (done)"; continue; fi
  log "start $t"; rm -f "$STATE/stage_detail.txt"; bash "$STAGES_DIR/slowwm_run.sh" "$t"; rc=$?
  if [ "$rc" -eq 0 ]; then echo "$t $(date '+%F %T')" > "$STATE/done/$t.done"; log "done $t"
  elif [ "$rc" -eq 11 ]; then
    d=""; [ -s "$STATE/stage_detail.txt" ] && d="$(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
    echo "ZZ SLOWWM-ARM-REFUSED $t gate-or-pin-refused rc=4 :: $d" > "$STATE/ARM_REFUSED_$t.txt"; log "arm $t refused at its gate; the next independent arm goes on"; REFUSED="$REFUSED $t"
  else map_rc "$t" "$rc"; fi
done
if [ -n "$REFUSED" ]; then echo "ZZ SLOWWM-STOP runs arms-refused:$REFUSED rc=4" > "$STATE/STOP.txt"; log "finished with refused arms:$REFUSED"; release_box; exit 4; fi
echo "ZZ SLOWWM-DONE $(date '+%F %T') commit=$SLOWWM_COMMIT runs=$RUN_ORDER" > "$STATE/DONE.txt"; log "all runs banked"
exit 0
