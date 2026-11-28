#!/usr/bin/env bash
# stopbal_chain.sh -- the STOPBAL retrains on box 2, in the card's order (theory/stopbal/PREREG_V8F_STOPBAL.md, FROZEN 2026-11-10; code of record 2bca1c52):
#   wait until the OBD chain has ended (ZZ OBD-DONE / ZZ OBD-STOP in its state dir) and the shared busy marker is free, claim it -> GPU idle -> ship (own tree) -> PILOT (200 steps of BAL-G_s0, projected total, 12 h rule)
#   -> BAL-NULL_s0, BAL-NULL_rep, BAL-G_s0, BAL-L_s0, BAL-NULL_s1, BAL-G_s1, BAL-L_s1.  Each run: argv pin, gate (selfcheck + BUILT-lever assertions + per-run token), training under the token, post check, bank, cleanup.
#   NULL parity (card, Beside: no bar): after BAL-NULL_s0 is banked, its checkpoint is compared with DAG-OACC s0's (tensor basis; SHA256SUMS.txt pinned) -> $STATE/NULL_PARITY.json. COMPUTED AND REPORTED, NEVER A STOP: an unreadable
#   parity input writes status not-computed and the chain goes on; no arm is appended or skipped on its result.
# Fail closed: no STOPBAL_COMMIT (40 lowercase hex) -> refuse; STOPBAL_SSH / STOPBAL_SFTP unset, relative, missing or of the wrong kind -> refuse (stages/stopbal_common.sh); on ANY refusal one line goes to $STATE/STOP.txt, exit non-zero.
# Failure behaviour: a GATE / PIN / TOKEN refusal of one run (stage rc 11) writes $STATE/ARM_REFUSED_<tag>.txt and the chain goes on with the next run (the arms are independent, nothing has used the GPU);
#   every other failure (ship, pilot, 12 h rule, training, post check, disk, pull, transport, timeout, box busy) stops the chain with $STATE/STOP.txt. A chain with a refused arm ends with STOP.txt, never DONE.txt.
# Sentinels live in their own files only (never in chain.log): $STATE/STOP.txt (prefix "ZZ STOPBAL-STOP"), $STATE/DONE.txt (prefix "ZZ STOPBAL-DONE"), $STATE/ARM_REFUSED_<tag>.txt (prefix "ZZ STOPBAL-ARM-REFUSED").
# Launch (README has the full line):  STOPBAL_COMMIT=<40 hex> STOPBAL_SSH=$(command -v ssh) STOPBAL_SFTP=$(command -v sftp) [STOPBAL_AMENDMENT=<file>] bash stopbal_chain.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; HERE="$(cygpath -m "$HERE" 2>/dev/null || echo "$HERE")"
export STOPBAL_STATE="${STOPBAL_STATE:-C:/Users/Admin/stopbal_state}"; STATE="$STOPBAL_STATE"
STAGES_DIR="${STOPBAL_STAGES:-$HERE/stages}"; PYV="${STOPBAL_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
PULLS_DEV="${STOPBAL_PULLS:-C:/Users/Admin/t2v8f_pulls}"
BUSY_DEV="${STOPBAL_BUSY_DEV:-C:/Users/Admin/box2_ship/BOX2_BUSY.txt}"; ASIDE_DEV="${STOPBAL_ASIDE_DEV:-C:/Users/Admin/box2_ship/busy_aside}"     # the SAME busy marker the other box-2 chains claim: they exclude each other
OBD_STATE_DEV="${STOPBAL_OBD_STATE:-C:/Users/Admin/obd_state}"                                                                            # where the OBD chain writes DONE.txt / STOP.txt (read only here)
DAGOACC_DIR="${STOPBAL_DAGOACC_DIR:-E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s0_dagoacc}"                                                  # SHA256SUMS.txt (the reference sha256) and, if present, ckpt_last.pt (tensor-level parity beside it)
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATE/chain.log"; }
mkdir -p "$STATE/done" "$ASIDE_DEV" || exit 2
refuse_early() { echo "ZZ STOPBAL-STOP start $1 rc=$2" > "$STATE/STOP.txt"; echo "REFUSED: $1" >&2; exit "$2"; }
case "${STOPBAL_COMMIT:-}" in *[!0-9a-f]*|'') refuse_early "commit-missing (STOPBAL_COMMIT unset or not lowercase hex)" 8;; esac
[ "${#STOPBAL_COMMIT}" -eq 40 ] || refuse_early "commit-missing (STOPBAL_COMMIT is not a full 40-hex sha)" 8
export STOPBAL_COMMIT
( source "$STAGES_DIR/stopbal_common.sh" ) > /dev/null 2> "$STATE/transport.err" || refuse_early "transport-not-fail-closed ($(tr '\n' ' ' < "$STATE/transport.err" | head -c 300))" 16
if [ -e "$STATE/STOP.txt" ] && [ "${STOPBAL_RESUME:-0}" != "1" ]; then echo "REFUSED: $STATE/STOP.txt exists ($(cat "$STATE/STOP.txt")); inspect, then rerun with STOPBAL_RESUME=1" >&2; exit 2; fi
[ "${STOPBAL_RESUME:-0}" = "1" ] && [ -e "$STATE/STOP.txt" ] && mv "$STATE/STOP.txt" "$STATE/STOP.$(date +%Y%m%d_%H%M%S).txt"
for f in stopbal_check.py stopbal_ship_tree.py stopbal_refresh.py frozen/argvs_stopbal.json stages/stopbal_common.sh stages/stopbal_ship.sh stages/stopbal_run.sh jobs/stopbal_stage_job.cmd jobs/stopbal_ship_job.cmd; do
  [ -e "$HERE/$f" ] || refuse_early "package-incomplete ($f missing in $HERE)" 8
done
chk() { "$PYV" -I -c "import sys; sys.path.insert(0, sys.argv[1]); import stopbal_check as C; print($1)" "$HERE"; }
RUN_ORDER="$(chk "' '.join(C.RUN_ORDER)")" || refuse_early "package-incomplete (the checker does not import)" 8
PARITY_TAG="$(chk "C.PARITY_TAG")" || refuse_early "package-incomplete (the checker does not import)" 8
# ---- wait for the OBD chain to end (its finished marker) and for nothing to hold box 2; refuse otherwise
obd_ended() {
  [ -e "$BUSY_DEV" ] && return 1                                                                                                              # a chain (OBD's, or another) holds box 2
  if [ -s "$OBD_STATE_DEV/DONE.txt" ] && head -c 40 "$OBD_STATE_DEV/DONE.txt" | grep -q '^ZZ OBD-DONE'; then return 0; fi
  if [ -s "$OBD_STATE_DEV/STOP.txt" ] && head -c 40 "$OBD_STATE_DEV/STOP.txt" | grep -q '^ZZ OBD-STOP'; then return 0; fi              # the OBD chain stopped itself and released box 2
  return 1
}
n=0; while ! obd_ended; do
  [ "$n" -lt "${STOPBAL_WAIT_POLLS:-1152}" ] || refuse_early "obd-not-ended (no ZZ OBD-DONE / ZZ OBD-STOP in $OBD_STATE_DEV, or $BUSY_DEV exists, after $n polls)" 12
  n=$((n+1)); log "waiting for the OBD chain to end (poll $n)"; sleep "${STOPBAL_WAIT_POLL_S:-300}"
done
if ! ( set -o noclobber; echo "stopbal chain $$ $(date '+%F %T')" > "$BUSY_DEV" ) 2>/dev/null; then echo "REFUSED: $BUSY_DEV exists (box 2 is busy)" >&2; exit 12; fi
OWN_MARK="$(cat "$BUSY_DEV")"
release_box() { if [ -e "$BUSY_DEV" ] && [ "$(cat "$BUSY_DEV" 2>/dev/null)" = "$OWN_MARK" ]; then mv "$BUSY_DEV" "$ASIDE_DEV/BOX2_BUSY.stopbal.$(date +%Y%m%d_%H%M%S).txt"; fi; }
trap 'release_box' EXIT
stop() {                   # stop <stage> <reason> <rc>
  local detail=""; [ -s "$STATE/stage_detail.txt" ] && detail=" :: $(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
  echo "ZZ STOPBAL-STOP $1 $2 rc=$3$detail" > "$STATE/STOP.txt"; log "stop at stage $1: $2 (rc $3)"; release_box; exit "$3"
}
dispatch() { rm -f "$STATE/stage_detail.txt"; case "$1" in ship) bash "$STAGES_DIR/stopbal_ship.sh";; pilot) bash "$STAGES_DIR/stopbal_run.sh" PILOT_BAL-G_s0;; *) bash "$STAGES_DIR/stopbal_run.sh" "$1";; esac; }
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
parity_step() {   # parity_step <tag>: after BAL-NULL_s0 is banked (now or on an earlier pass), compare its checkpoint with DAG-OACC s0's once. Reported, never a stop; a failure of the comparison is logged and the chain goes on.
  [ "$1" = "$PARITY_TAG" ] || return 0
  [ -e "$STATE/done/$PARITY_TAG.done" ] || return 0
  [ -s "$STATE/NULL_PARITY.json" ] && return 0
  local dag=(); [ -f "$DAGOACC_DIR/ckpt_last.pt" ] && dag=(--dagoacc-ckpt "$DAGOACC_DIR/ckpt_last.pt")
  "$PYV" -I "$HERE/stopbal_check.py" parity --bank-ckpt "$PULLS_DEV/stopbal_$PARITY_TAG/ckpt_last.pt" --sums "$DAGOACC_DIR/SHA256SUMS.txt" --out "$STATE/NULL_PARITY.json" "${dag[@]}" --basis "${STOPBAL_PARITY_BASIS:-tensors}" > "$STATE/parity.txt" 2>&1
  local prc=$?
  log "NULL parity (reported, never a stop; rc $prc): $(tr '\n' ' ' < "$STATE/parity.txt" | head -c 300)"
  [ -s "$STATE/NULL_PARITY.json" ] || echo '{"status": "not-computed", "reason": "the parity command wrote no file", "null_parity": "UNKNOWN"}' > "$STATE/NULL_PARITY.json"
}
REFUSED=""
i=0
while [ "$i" -lt "${#ORDER[@]}" ]; do
  t="${ORDER[$i]}"; i=$((i+1))
  if [ -e "$STATE/done/$t.done" ]; then log "skip $t (done)"; parity_step "$t"; continue; fi
  log "start $t"; rm -f "$STATE/stage_detail.txt"; bash "$STAGES_DIR/stopbal_run.sh" "$t"; rc=$?
  if [ "$rc" -eq 0 ]; then echo "$t $(date '+%F %T')" > "$STATE/done/$t.done"; log "done $t"; parity_step "$t"
  elif [ "$rc" -eq 11 ]; then
    d=""; [ -s "$STATE/stage_detail.txt" ] && d="$(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
    echo "ZZ STOPBAL-ARM-REFUSED $t gate-or-pin-refused rc=4 :: $d" > "$STATE/ARM_REFUSED_$t.txt"; log "arm $t refused at its gate; the next independent arm goes on"; REFUSED="$REFUSED $t"
  else map_rc "$t" "$rc"; fi
done
if [ -n "$REFUSED" ]; then echo "ZZ STOPBAL-STOP runs arms-refused:$REFUSED rc=4" > "$STATE/STOP.txt"; log "finished with refused arms:$REFUSED"; release_box; exit 4; fi
echo "ZZ STOPBAL-DONE $(date '+%F %T') commit=$STOPBAL_COMMIT runs=${ORDER[*]}" > "$STATE/DONE.txt"; log "all runs banked"
exit 0
