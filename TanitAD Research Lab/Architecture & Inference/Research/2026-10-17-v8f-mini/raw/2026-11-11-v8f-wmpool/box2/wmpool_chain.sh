#!/usr/bin/env bash
# wmpool_chain.sh -- the P-ATT attention-pool runs on box 2, in the card's order (theory/wmpool/PREREG_V8F_WMPOOL.md, frozen 2026-11-11; code of record abc01481):
#   wait until the STOPBAL chain has ended (DONE, or a STOP that is NOT a busy / GPU stop) and the shared busy marker is free, claim it -> ship (own tree + zcells code) -> CELLS (zcells.py build, manifest, check3 = card control 3; refuses below cache + 5 GB free)
#   -> PILOT (200 steps of P-ATT_s0, gates all three full argvs, projected total, 72 h rule) -> P-ATT_s0, P-ATT_s1, P-ATT_rep_s2.  Each run: argv pin, gate (selfcheck + BUILT-lever assertions + per-run token), training under the token, post check, bank, cleanup.
# Fail closed: no WMPOOL_COMMIT (40 lowercase hex) -> refuse; WMPOOL_SSH / WMPOOL_SFTP unset, relative, missing or of the wrong kind -> refuse (stages/wmpool_common.sh); on ANY refusal one line goes to $STATE/STOP.txt, exit non-zero.
# Failure behaviour: a GATE / PIN / TOKEN refusal of one run (stage rc 11) writes $STATE/ARM_REFUSED_<tag>.txt and the chain goes on with the next run (the arms are independent, nothing has used the GPU);
#   every other failure (ship, pilot, 72 h rule, training, post check, disk, pull, transport, timeout, box busy) stops the chain with $STATE/STOP.txt. A chain with a refused arm ends with STOP.txt, never DONE.txt.
# Sentinels live in their own files only (never in chain.log): $STATE/STOP.txt (prefix "ZZ WMPOOL-STOP"), $STATE/DONE.txt (prefix "ZZ WMPOOL-DONE"), $STATE/ARM_REFUSED_<tag>.txt (prefix "ZZ WMPOOL-ARM-REFUSED").
# Launch (README has the full line):  WMPOOL_COMMIT=<40 hex> WMPOOL_SSH=$(command -v ssh) WMPOOL_SFTP=$(command -v sftp) [WMPOOL_AMENDMENT=<file>] bash wmpool_chain.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; HERE="$(cygpath -m "$HERE" 2>/dev/null || echo "$HERE")"
export WMPOOL_STATE="${WMPOOL_STATE:-C:/Users/Admin/wmpool_state}"; STATE="$WMPOOL_STATE"
STAGES_DIR="${WMPOOL_STAGES:-$HERE/stages}"; PYV="${WMPOOL_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
BUSY_DEV="${WMPOOL_BUSY_DEV:-C:/Users/Admin/box2_ship/BOX2_BUSY.txt}"; ASIDE_DEV="${WMPOOL_ASIDE_DEV:-C:/Users/Admin/box2_ship/busy_aside}"     # the SAME busy marker the other chains claim: the two chains exclude each other
STOPBAL_STATE_DEV="${WMPOOL_STOPBAL_STATE:-C:/Users/Admin/stopbal_state}"                                                                 # where the STOPBAL chain writes DONE.txt / STOP.txt (read only here)
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATE/chain.log"; }
mkdir -p "$STATE/done" "$ASIDE_DEV" || exit 2
refuse_early() { echo "ZZ WMPOOL-STOP start $1 rc=$2" > "$STATE/STOP.txt"; echo "REFUSED: $1" >&2; exit "$2"; }
case "${WMPOOL_COMMIT:-}" in *[!0-9a-f]*|'') refuse_early "commit-missing (WMPOOL_COMMIT unset or not lowercase hex)" 8;; esac
[ "${#WMPOOL_COMMIT}" -eq 40 ] || refuse_early "commit-missing (WMPOOL_COMMIT is not a full 40-hex sha)" 8
export WMPOOL_COMMIT
for hook in WMPOOL_TEST_CELL_COUNTS WMPOOL_TEST_ZCELLS WMPOOL_FAKE_FREE_GB; do         # test hooks of the checker: only with the stub transport (WMPOOL_DRY=1); a real launch with one set is refused
  [ -z "${!hook:-}" ] || [ "${WMPOOL_DRY:-0}" = 1 ] || refuse_early "test-hook-in-real-run ($hook is set without WMPOOL_DRY=1)" 8
done
( source "$STAGES_DIR/wmpool_common.sh" ) > /dev/null 2> "$STATE/transport.err" || refuse_early "transport-not-fail-closed ($(tr '\n' ' ' < "$STATE/transport.err" | head -c 300))" 16
if [ -e "$STATE/STOP.txt" ] && [ "${WMPOOL_RESUME:-0}" != "1" ]; then echo "REFUSED: $STATE/STOP.txt exists ($(cat "$STATE/STOP.txt")); inspect, then rerun with WMPOOL_RESUME=1" >&2; exit 2; fi
[ "${WMPOOL_RESUME:-0}" = "1" ] && [ -e "$STATE/STOP.txt" ] && mv "$STATE/STOP.txt" "$STATE/STOP.$(date +%Y%m%d_%H%M%S).txt"
for f in wmpool_check.py wmpool_ship_tree.py wmpool_refresh.py frozen/argvs_wmpool.json frozen/export_list_wmpool.txt frozen/zcells.py frozen/zcache.py frozen/clips8.json stages/wmpool_common.sh stages/wmpool_ship.sh stages/wmpool_run.sh stages/wmpool_cells.sh jobs/wmpool_stage_job.cmd jobs/wmpool_ship_job.cmd; do
  [ -e "$HERE/$f" ] || refuse_early "package-incomplete ($f missing in $HERE)" 8
done
RUN_ORDER="$("$PYV" -I -c "import sys; sys.path.insert(0, sys.argv[1]); import wmpool_check as C; print(' '.join(C.RUN_ORDER))" "$HERE")" || refuse_early "package-incomplete (the checker does not import)" 8
# ---- wait for the predecessor (the STOPBAL chain) to end, and for nothing to hold box 2.
#   ended = DONE.txt (first bytes ZZ STOPBAL-DONE), or STOP.txt (ZZ STOPBAL-STOP) whose reason is NOT a busy / GPU stop. A STOPBAL STOP with rc=12 (box-2-busy-or-gpu-not-idle, predecessor-not-ended) means it never ran its stage: it may be resumed, so this chain keeps waiting.
pred_state() {   # prints DONE | STOP | STOP-BUSY | RUNNING
  local f="$STOPBAL_STATE_DEV/STOP.txt" head1
  if [ -s "$f" ] && head -c 40 "$f" | grep -q '^ZZ STOPBAL-STOP'; then
    head1="$(head -n 1 "$f" | sed 's/ :: .*//')"
    if printf '%s' "$head1" | grep -Eq ' rc=12 *$'; then echo STOP-BUSY; else echo STOP; fi; return
  fi
  if [ -s "$STOPBAL_STATE_DEV/DONE.txt" ] && head -c 40 "$STOPBAL_STATE_DEV/DONE.txt" | grep -q '^ZZ STOPBAL-DONE'; then echo DONE; return; fi
  echo RUNNING
}
n=0; while :; do
  ps_="$(pred_state)"
  if { [ "$ps_" = DONE ] || [ "$ps_" = STOP ]; } && [ ! -e "$BUSY_DEV" ]; then log "the STOPBAL chain ended ($ps_)"; break; fi
  [ "$n" -lt "${WMPOOL_WAIT_POLLS:-1152}" ] || refuse_early "predecessor-not-ended (no ZZ STOPBAL-DONE and no non-busy ZZ STOPBAL-STOP in $STOPBAL_STATE_DEV, or $BUSY_DEV exists, after $n polls; last state $ps_)" 12
  n=$((n+1)); log "waiting for the STOPBAL chain to end (poll $n, state $ps_)"; sleep "${WMPOOL_WAIT_POLL_S:-300}"
done
if ! ( set -o noclobber; echo "wmpool chain $$ $(date '+%F %T')" > "$BUSY_DEV" ) 2>/dev/null; then echo "REFUSED: $BUSY_DEV exists (box 2 is busy)" >&2; exit 12; fi
OWN_MARK="$(cat "$BUSY_DEV")"
release_box() { if [ -e "$BUSY_DEV" ] && [ "$(cat "$BUSY_DEV" 2>/dev/null)" = "$OWN_MARK" ]; then mv "$BUSY_DEV" "$ASIDE_DEV/BOX2_BUSY.wmpool.$(date +%Y%m%d_%H%M%S).txt"; fi; }
trap 'release_box' EXIT
stop() {                   # stop <stage> <reason> <rc>
  local detail=""; [ -s "$STATE/stage_detail.txt" ] && detail=" :: $(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
  echo "ZZ WMPOOL-STOP $1 $2 rc=$3$detail" > "$STATE/STOP.txt"; log "stop at stage $1: $2 (rc $3)"; release_box; exit "$3"
}
dispatch() { rm -f "$STATE/stage_detail.txt"; case "$1" in ship) bash "$STAGES_DIR/wmpool_ship.sh";; pilot) bash "$STAGES_DIR/wmpool_run.sh" PILOT_P-ATT_s0;; cells) bash "$STAGES_DIR/wmpool_cells.sh";; *) bash "$STAGES_DIR/wmpool_run.sh" "$1";; esac; }
map_rc() {   # <stage> <rc>: a non-zero rc stops the chain, except an arm-specific gate refusal of a full run (handled by the caller)
  case "$2" in
    11) stop "$1" "gate-or-pin-refused" 4;; 12) stop "$1" "job-or-transfer-failed" 5;; 13) stop "$1" "disk-low" 6;; 14) stop "$1" "box2-busy-or-gpu-not-idle" 12;; 15) stop "$1" "pull-sha-mismatch" 7;;
    16) stop "$1" "package-commit-or-transport" 8;; 17) stop "$1" "ship-refused" 9;; 18) stop "$1" "pilot-unreadable" 10;; 19) stop "$1" "box2-timeout-or-unreadable" 11;; 23) stop "$1" "blocked-tag" 13;;
    *) stop "$1" "stage-crashed" 5;;
  esac
}
run_stage() {   # run_stage <name>: ship | cells | pilot
  local s="$1" rc
  if [ -e "$STATE/done/$s.done" ]; then log "skip $s (done)"; return 0; fi
  log "start $s"; dispatch "$s"; rc=$?
  if [ "$rc" -eq 0 ]; then echo "$s $(date '+%F %T')" > "$STATE/done/$s.done"; log "done $s"; else map_rc "$s" "$rc"; fi
}
run_stage ship
run_stage cells
run_stage pilot
P="$STATE/pilot/PILOT.verified.json"; [ -s "$P" ] || stop pilot "no-verified-pilot ($P missing)" 10
"$PYV" -I "$HERE/wmpool_check.py" permit --pilot "$P" --commit "$WMPOOL_COMMIT" ${WMPOOL_AMENDMENT:+--amendment "$WMPOOL_AMENDMENT"} > "$STATE/permit.txt" 2>&1 || { echo "$(tr '\n' ' ' < "$STATE/permit.txt" | head -c 400)" > "$STATE/stage_detail.txt"; stop permit "72h-rule-refused" 3; }
log "$(cat "$STATE/permit.txt")"
REFUSED=""
for t in $RUN_ORDER; do
  if [ -e "$STATE/done/$t.done" ]; then log "skip $t (done)"; continue; fi
  log "start $t"; rm -f "$STATE/stage_detail.txt"; bash "$STAGES_DIR/wmpool_run.sh" "$t"; rc=$?
  if [ "$rc" -eq 0 ]; then echo "$t $(date '+%F %T')" > "$STATE/done/$t.done"; log "done $t"
  elif [ "$rc" -eq 11 ]; then
    d=""; [ -s "$STATE/stage_detail.txt" ] && d="$(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
    echo "ZZ WMPOOL-ARM-REFUSED $t gate-or-pin-refused rc=4 :: $d" > "$STATE/ARM_REFUSED_$t.txt"; log "arm $t refused at its gate; the next independent arm goes on"; REFUSED="$REFUSED $t"
  else map_rc "$t" "$rc"; fi
done
if [ -n "$REFUSED" ]; then echo "ZZ WMPOOL-STOP runs arms-refused:$REFUSED rc=4" > "$STATE/STOP.txt"; log "finished with refused arms:$REFUSED"; release_box; exit 4; fi
echo "ZZ WMPOOL-DONE $(date '+%F %T') commit=$WMPOOL_COMMIT runs=$RUN_ORDER" > "$STATE/DONE.txt"; log "all runs banked"
exit 0
