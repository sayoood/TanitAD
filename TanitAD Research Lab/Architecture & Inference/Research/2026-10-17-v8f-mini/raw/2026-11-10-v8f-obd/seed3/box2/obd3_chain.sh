#!/usr/bin/env bash
# obd3_chain.sh -- the four OBD third-seed retrains on box 2 (theory/obd/AMEND_OBD_SEED3.md, sha256 a8a5848f...; code of record cdbff971fa6dad4d20b937fe8abda6cb9f622adb):
#   wait until the STOPBAL chain has ended (DONE, or a STOP that is NOT a busy / GPU stop) and the shared busy marker is free, claim it -> ship (proof of the tree the OBD chain already shipped; fallback: ship it the same way, same path)
#   -> OBD-CTL_s10, OBD-BAND_s10, OBD-CTL_s11, OBD-BAND_s11 (a stopped chain still yields one complete pair).  Each run: disk, GPU idle, no other python.exe, argv pin, gate (selfcheck + BUILT-lever assertions + per-run token),
#   training under the token, post check, bank (sha256 against box 2's copy) into $PULLS/obd_<tag>/, cleanup of the run's own gate dir only after the bank is proven.  No pilot, no NULL arms, no NULL parity, no 12 h rule.
# Fail closed: no OBD3_COMMIT (40 lowercase hex, equal to the pinned code of record) -> refuse; OBD3_SSH / OBD3_SFTP unset, relative, missing or of the wrong kind -> refuse (stages/obd3_common.sh);
#   on ANY refusal one line goes to $STATE/STOP.txt, exit non-zero.
# Failure behaviour: a GATE / PIN / TOKEN refusal of one run (stage rc 11) writes $STATE/ARM_REFUSED_<tag>.txt and the chain goes on with the next run (the arms are independent, nothing has used the GPU);
#   every other failure (ship, training, post check, disk, pull, transport, timeout, box busy) stops the chain with $STATE/STOP.txt. A chain with a refused arm ends with STOP.txt, never DONE.txt.
# Sentinels live in their own files only (never in chain.log): $STATE/STOP.txt (prefix "ZZ OBD3-STOP"), $STATE/DONE.txt (prefix "ZZ OBD3-DONE"), $STATE/ARM_REFUSED_<tag>.txt (prefix "ZZ OBD3-ARM-REFUSED").
# Launch (README has the full line):  OBD3_COMMIT=cdbff971fa6dad4d20b937fe8abda6cb9f622adb OBD3_SSH=$(command -v ssh) OBD3_SFTP=$(command -v sftp) bash obd3_chain.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; HERE="$(cygpath -m "$HERE" 2>/dev/null || echo "$HERE")"
export OBD3_STATE="${OBD3_STATE:-C:/Users/Admin/obd_seed3_state}"; STATE="$OBD3_STATE"
STAGES_DIR="${OBD3_STAGES:-$HERE/stages}"; PYV="${OBD3_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
BUSY_DEV="${OBD3_BUSY_DEV:-C:/Users/Admin/box2_ship/BOX2_BUSY.txt}"; ASIDE_DEV="${OBD3_ASIDE_DEV:-C:/Users/Admin/box2_ship/busy_aside}"     # the SAME busy marker the other box-2 chains claim: they exclude each other
STOPBAL_STATE_DEV="${OBD3_STOPBAL_STATE:-C:/Users/Admin/stopbal_state}"                                                                     # where the STOPBAL chain writes DONE.txt / STOP.txt (read only here)
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATE/chain.log"; }
mkdir -p "$STATE/done" "$ASIDE_DEV" || exit 2
refuse_early() { echo "ZZ OBD3-STOP start $1 rc=$2" > "$STATE/STOP.txt"; echo "REFUSED: $1" >&2; exit "$2"; }
case "${OBD3_COMMIT:-}" in *[!0-9a-f]*|'') refuse_early "commit-missing (OBD3_COMMIT unset or not lowercase hex)" 8;; esac
[ "${#OBD3_COMMIT}" -eq 40 ] || refuse_early "commit-missing (OBD3_COMMIT is not a full 40-hex sha)" 8
export OBD3_COMMIT
[ -z "${OBD3_CODE_COMMIT:-}" ] || [ "${OBD3_DRY:-0}" = 1 ] || refuse_early "test-hook-in-real-run (OBD3_CODE_COMMIT is set without OBD3_DRY=1)" 8
( source "$STAGES_DIR/obd3_common.sh" ) > /dev/null 2> "$STATE/transport.err" || refuse_early "transport-not-fail-closed ($(tr '\n' ' ' < "$STATE/transport.err" | head -c 300))" 16
if [ -e "$STATE/STOP.txt" ] && [ "${OBD3_RESUME:-0}" != "1" ]; then echo "REFUSED: $STATE/STOP.txt exists ($(cat "$STATE/STOP.txt")); inspect, then rerun with OBD3_RESUME=1" >&2; exit 2; fi
[ "${OBD3_RESUME:-0}" = "1" ] && [ -e "$STATE/STOP.txt" ] && mv "$STATE/STOP.txt" "$STATE/STOP.$(date +%Y%m%d_%H%M%S).txt"
for f in obd3_check.py make_argvs_seed3.py frozen/argvs_obd_seed3.json stages/obd3_common.sh stages/obd3_ship.sh stages/obd3_run.sh jobs/obd3_stage_job.cmd jobs/obd3_ship_job.cmd; do
  [ -e "$HERE/$f" ] || refuse_early "package-incomplete ($f missing in $HERE)" 8
done
chk() { "$PYV" -I -c "import sys; sys.path.insert(0, sys.argv[1]); import obd3_check as C; print($1)" "$HERE"; }
RUN_ORDER="$(chk "' '.join(C.RUN_ORDER)")" || refuse_early "package-incomplete (the checker does not import: the pinned OBD chain files are missing or changed)" 8
PINNED="$(chk "C.CODE_COMMIT")" || refuse_early "package-incomplete (the checker does not import)" 8
[ "${OBD3_DRY:-0}" = 1 ] || [ "$OBD3_COMMIT" = "$PINNED" ] || refuse_early "commit-not-the-code-of-record (OBD3_COMMIT is not $PINNED)" 8
# ---- wait for the predecessor (the STOPBAL chain) to end, and for nothing to hold box 2.
#   ended = DONE.txt (first bytes: the STOPBAL done marker), or STOP.txt (the STOPBAL stop marker) whose reason is NOT a busy / GPU stop. A STOPBAL STOP whose head (the text before " :: ") ends in rc=12
#   (box2-busy-or-gpu-not-idle, predecessor-not-ended) means it never ran its stage: it may be resumed, so this chain keeps waiting.
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
  [ "$n" -lt "${OBD3_WAIT_POLLS:-1152}" ] || refuse_early "predecessor-not-ended (no done marker and no non-busy stop marker of STOPBAL in $STOPBAL_STATE_DEV, or $BUSY_DEV exists, after $n polls; last state $ps_)" 12
  n=$((n+1)); log "waiting for the STOPBAL chain to end (poll $n, state $ps_)"; sleep "${OBD3_WAIT_POLL_S:-300}"
done
if ! ( set -o noclobber; echo "obd3 chain $$ $(date '+%F %T')" > "$BUSY_DEV" ) 2>/dev/null; then echo "REFUSED: $BUSY_DEV exists (box 2 is busy)" >&2; exit 12; fi
OWN_MARK="$(cat "$BUSY_DEV")"
release_box() { if [ -e "$BUSY_DEV" ] && [ "$(cat "$BUSY_DEV" 2>/dev/null)" = "$OWN_MARK" ]; then mv "$BUSY_DEV" "$ASIDE_DEV/BOX2_BUSY.obd3.$(date +%Y%m%d_%H%M%S).txt"; fi; }
trap 'release_box' EXIT
stop() {                   # stop <stage> <reason> <rc>
  local detail=""; [ -s "$STATE/stage_detail.txt" ] && detail=" :: $(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
  echo "ZZ OBD3-STOP $1 $2 rc=$3$detail" > "$STATE/STOP.txt"; log "stop at stage $1: $2 (rc $3)"; release_box; exit "$3"
}
map_rc() {   # <stage> <rc>: a non-zero rc stops the chain, except an arm-specific gate refusal of a full run (handled by the caller)
  case "$2" in
    11) stop "$1" "gate-or-pin-refused" 4;; 12) stop "$1" "job-or-transfer-failed" 5;; 13) stop "$1" "disk-low" 6;; 14) stop "$1" "box2-busy-or-gpu-not-idle" 12;; 15) stop "$1" "pull-sha-mismatch" 7;;
    16) stop "$1" "package-commit-or-transport" 8;; 17) stop "$1" "ship-refused" 9;; 19) stop "$1" "box2-timeout-or-unreadable" 11;; 23) stop "$1" "blocked-tag" 13;;
    *) stop "$1" "stage-crashed" 5;;
  esac
}
if [ -e "$STATE/done/ship.done" ]; then log "skip ship (done)"; else
  log "start ship"; rm -f "$STATE/stage_detail.txt"; bash "$STAGES_DIR/obd3_ship.sh"; rc=$?
  if [ "$rc" -eq 0 ]; then echo "ship $(date '+%F %T')" > "$STATE/done/ship.done"; log "done ship"; else map_rc ship "$rc"; fi
fi
REFUSED=""
for t in $RUN_ORDER; do
  if [ -e "$STATE/done/$t.done" ]; then log "skip $t (done)"; continue; fi
  log "start $t"; rm -f "$STATE/stage_detail.txt"; bash "$STAGES_DIR/obd3_run.sh" "$t"; rc=$?
  if [ "$rc" -eq 0 ]; then echo "$t $(date '+%F %T')" > "$STATE/done/$t.done"; log "done $t"
  elif [ "$rc" -eq 11 ]; then
    d=""; [ -s "$STATE/stage_detail.txt" ] && d="$(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
    echo "ZZ OBD3-ARM-REFUSED $t gate-or-pin-refused rc=4 :: $d" > "$STATE/ARM_REFUSED_$t.txt"; log "arm $t refused at its gate; the next independent arm goes on"; REFUSED="$REFUSED $t"
  else map_rc "$t" "$rc"; fi
done
if [ -n "$REFUSED" ]; then echo "ZZ OBD3-STOP runs arms-refused:$REFUSED rc=4" > "$STATE/STOP.txt"; log "finished with refused arms:$REFUSED"; release_box; exit 4; fi
echo "ZZ OBD3-DONE $(date '+%F %T') commit=$OBD3_COMMIT runs=$RUN_ORDER" > "$STATE/DONE.txt"; log "all runs banked"
exit 0
