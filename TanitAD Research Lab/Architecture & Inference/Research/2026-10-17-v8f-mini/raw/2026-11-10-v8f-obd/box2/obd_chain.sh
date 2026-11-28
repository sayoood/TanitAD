#!/usr/bin/env bash
# obd_chain.sh -- the OBD retrains on box 2, in the card's order (theory/obd/PREREG_V8F_OBD_DRAFT.md):
#   wait until the EGOWM chain has ended and the shared busy marker is free, claim it -> GPU idle -> ship (own tree) -> PILOT (200 steps of OBD-BAND_s0, projected total, 12 h rule)
#   -> OBD-NULL_s0, OBD-NULL_rep, OBD-CTL_s0, OBD-BAND_s0, OBD-CTL_s1, OBD-BAND_s1, OBD-BAND_rep.  Each run: argv pin, gate (selfcheck + BUILT-lever assertions + per-run token), training under the token, post check, bank, cleanup.
#   NULL parity: after OBD-NULL_s0 is banked, its ckpt_last.pt sha256 is compared with DAG-OACC s0's (SHA256SUMS.txt) -> $STATE/NULL_PARITY.json; if they differ the conditional OBD-NULL_s1 is appended to the run order.
# Fail closed: no OBD_COMMIT (40 lowercase hex) -> refuse; OBD_SSH / OBD_SFTP unset, relative, missing or of the wrong kind -> refuse (stages/obd_common.sh); on ANY refusal one line goes to $STATE/STOP.txt, exit non-zero.
# Failure behaviour: a GATE / PIN / TOKEN refusal of one run (stage rc 11) writes $STATE/ARM_REFUSED_<tag>.txt and the chain goes on with the next run (the arms are independent, nothing has used the GPU);
#   every other failure (ship, pilot, 12 h rule, training, post check, parity input, disk, pull, transport, timeout, box busy) stops the chain with $STATE/STOP.txt. A chain with a refused arm ends with STOP.txt, never DONE.txt.
#   NOTE: while the held trainer's purity probe does not read the acceleration term (PURITY_ACC_FINDING.md) the PILOT's own gate refuses (rc 4) and the chain stops before any run.
# Sentinels live in their own files only (never in chain.log): $STATE/STOP.txt (prefix "ZZ OBD-STOP"), $STATE/DONE.txt (prefix "ZZ OBD-DONE"), $STATE/ARM_REFUSED_<tag>.txt (prefix "ZZ OBD-ARM-REFUSED").
# Launch (README has the full line):  OBD_COMMIT=<40 hex> OBD_SSH=$(command -v ssh) OBD_SFTP=$(command -v sftp) [OBD_AMENDMENT=<file>] bash obd_chain.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; HERE="$(cygpath -m "$HERE" 2>/dev/null || echo "$HERE")"
export OBD_STATE="${OBD_STATE:-C:/Users/Admin/obd_state}"; STATE="$OBD_STATE"
STAGES_DIR="${OBD_STAGES:-$HERE/stages}"; PYV="${OBD_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
PULLS_DEV="${OBD_PULLS:-C:/Users/Admin/t2v8f_pulls}"
BUSY_DEV="${OBD_BUSY_DEV:-C:/Users/Admin/box2_ship/BOX2_BUSY.txt}"; ASIDE_DEV="${OBD_ASIDE_DEV:-C:/Users/Admin/box2_ship/busy_aside}"     # the SAME busy marker the other box-2 chains claim: they exclude each other
EGOWM_STATE_DEV="${OBD_EGOWM_STATE:-C:/Users/Admin/egowm_state}"                                                                            # where the EGOWM chain writes DONE.txt / STOP.txt (read only here)
DAGOACC_DIR="${OBD_DAGOACC_DIR:-E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s0_dagoacc}"                                                  # SHA256SUMS.txt (the reference sha256) and, if present, ckpt_last.pt (tensor-level parity beside it)
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATE/chain.log"; }
mkdir -p "$STATE/done" "$ASIDE_DEV" || exit 2
refuse_early() { echo "ZZ OBD-STOP start $1 rc=$2" > "$STATE/STOP.txt"; echo "REFUSED: $1" >&2; exit "$2"; }
case "${OBD_COMMIT:-}" in *[!0-9a-f]*|'') refuse_early "commit-missing (OBD_COMMIT unset or not lowercase hex)" 8;; esac
[ "${#OBD_COMMIT}" -eq 40 ] || refuse_early "commit-missing (OBD_COMMIT is not a full 40-hex sha)" 8
export OBD_COMMIT
( source "$STAGES_DIR/obd_common.sh" ) > /dev/null 2> "$STATE/transport.err" || refuse_early "transport-not-fail-closed ($(tr '\n' ' ' < "$STATE/transport.err" | head -c 300))" 16
if [ -e "$STATE/STOP.txt" ] && [ "${OBD_RESUME:-0}" != "1" ]; then echo "REFUSED: $STATE/STOP.txt exists ($(cat "$STATE/STOP.txt")); inspect, then rerun with OBD_RESUME=1" >&2; exit 2; fi
[ "${OBD_RESUME:-0}" = "1" ] && [ -e "$STATE/STOP.txt" ] && mv "$STATE/STOP.txt" "$STATE/STOP.$(date +%Y%m%d_%H%M%S).txt"
for f in obd_check.py obd_ship_tree.py obd_refresh.py frozen/argvs_obd.json stages/obd_common.sh stages/obd_ship.sh stages/obd_run.sh jobs/obd_stage_job.cmd jobs/obd_ship_job.cmd; do
  [ -e "$HERE/$f" ] || refuse_early "package-incomplete ($f missing in $HERE)" 8
done
chk() { "$PYV" -I -c "import sys; sys.path.insert(0, sys.argv[1]); import obd_check as C; print($1)" "$HERE"; }
RUN_ORDER="$(chk "' '.join(C.RUN_ORDER)")" || refuse_early "package-incomplete (the checker does not import)" 8
CONDITIONAL_TAG="$(chk "C.CONDITIONAL_TAG")" || refuse_early "package-incomplete (the checker does not import)" 8
PARITY_TAG="$(chk "C.PARITY_TAG")" || refuse_early "package-incomplete (the checker does not import)" 8
# ---- wait for the EGOWM chain to end (its finished marker) and for nothing to hold box 2; refuse otherwise
egowm_ended() {
  [ -e "$BUSY_DEV" ] && return 1                                                                                                              # a chain (EGOWM's, or another) holds box 2
  if [ -s "$EGOWM_STATE_DEV/DONE.txt" ] && head -c 40 "$EGOWM_STATE_DEV/DONE.txt" | grep -q '^ZZ EGOWM-DONE'; then return 0; fi
  if [ -s "$EGOWM_STATE_DEV/STOP.txt" ] && head -c 40 "$EGOWM_STATE_DEV/STOP.txt" | grep -q '^ZZ EGOWM-STOP'; then return 0; fi              # the EGOWM chain stopped itself and released box 2
  return 1
}
n=0; while ! egowm_ended; do
  [ "$n" -lt "${OBD_WAIT_POLLS:-1152}" ] || refuse_early "egowm-not-ended (no ZZ EGOWM-DONE / ZZ EGOWM-STOP in $EGOWM_STATE_DEV, or $BUSY_DEV exists, after $n polls)" 12
  n=$((n+1)); log "waiting for the EGOWM chain to end (poll $n)"; sleep "${OBD_WAIT_POLL_S:-300}"
done
if ! ( set -o noclobber; echo "obd chain $$ $(date '+%F %T')" > "$BUSY_DEV" ) 2>/dev/null; then echo "REFUSED: $BUSY_DEV exists (box 2 is busy)" >&2; exit 12; fi
OWN_MARK="$(cat "$BUSY_DEV")"
release_box() { if [ -e "$BUSY_DEV" ] && [ "$(cat "$BUSY_DEV" 2>/dev/null)" = "$OWN_MARK" ]; then mv "$BUSY_DEV" "$ASIDE_DEV/BOX2_BUSY.obd.$(date +%Y%m%d_%H%M%S).txt"; fi; }
trap 'release_box' EXIT
stop() {                   # stop <stage> <reason> <rc>
  local detail=""; [ -s "$STATE/stage_detail.txt" ] && detail=" :: $(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
  echo "ZZ OBD-STOP $1 $2 rc=$3$detail" > "$STATE/STOP.txt"; log "stop at stage $1: $2 (rc $3)"; release_box; exit "$3"
}
dispatch() { rm -f "$STATE/stage_detail.txt"; case "$1" in ship) bash "$STAGES_DIR/obd_ship.sh";; pilot) bash "$STAGES_DIR/obd_run.sh" PILOT_OBD-BAND_s0;; *) bash "$STAGES_DIR/obd_run.sh" "$1";; esac; }
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
"$PYV" -I "$HERE/obd_check.py" permit --pilot "$P" --commit "$OBD_COMMIT" ${OBD_AMENDMENT:+--amendment "$OBD_AMENDMENT"} > "$STATE/permit.txt" 2>&1 || { echo "$(tr '\n' ' ' < "$STATE/permit.txt" | head -c 400)" > "$STATE/stage_detail.txt"; stop permit "12h-rule-refused" 3; }
log "$(cat "$STATE/permit.txt")"
ORDER=($RUN_ORDER)
parity_step() {   # parity_step <tag>: after OBD-NULL_s0 is banked (now or on an earlier pass), compare its checkpoint with DAG-OACC s0's once; append the conditional arm when they differ
  [ "$1" = "$PARITY_TAG" ] || return 0
  [ -e "$STATE/done/$PARITY_TAG.done" ] || return 0
  if [ ! -s "$STATE/NULL_PARITY.json" ]; then
    local dag=(); [ -f "$DAGOACC_DIR/ckpt_last.pt" ] && dag=(--dagoacc-ckpt "$DAGOACC_DIR/ckpt_last.pt")
    "$PYV" -I "$HERE/obd_check.py" parity --bank-ckpt "$PULLS_DEV/obd_$PARITY_TAG/ckpt_last.pt" --sums "$DAGOACC_DIR/SHA256SUMS.txt" --out "$STATE/NULL_PARITY.json" "${dag[@]}" --basis "${OBD_PARITY_BASIS:-file}" > "$STATE/parity.txt" 2>&1 \
      || { cp "$STATE/parity.txt" "$STATE/stage_detail.txt"; stop parity "null-parity-input-unreadable" 14; }
    log "$(cat "$STATE/parity.txt")"
  fi
  local app; app="$("$PYV" -I -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['append_obd_null_s1'])" "$STATE/NULL_PARITY.json")" || stop parity "null-parity-json-unreadable" 14
  if [ "$app" = "True" ]; then
    case " ${ORDER[*]} " in *" $CONDITIONAL_TAG "*) ;; *) ORDER+=("$CONDITIONAL_TAG"); log "NULL parity failed: $CONDITIONAL_TAG appended to the run order";; esac
  else log "NULL parity holds: $CONDITIONAL_TAG is not trained"; fi
}
REFUSED=""
i=0
while [ "$i" -lt "${#ORDER[@]}" ]; do
  t="${ORDER[$i]}"; i=$((i+1))
  if [ -e "$STATE/done/$t.done" ]; then log "skip $t (done)"; parity_step "$t"; continue; fi
  log "start $t"; rm -f "$STATE/stage_detail.txt"; bash "$STAGES_DIR/obd_run.sh" "$t"; rc=$?
  if [ "$rc" -eq 0 ]; then echo "$t $(date '+%F %T')" > "$STATE/done/$t.done"; log "done $t"; parity_step "$t"
  elif [ "$rc" -eq 11 ]; then
    d=""; [ -s "$STATE/stage_detail.txt" ] && d="$(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"
    echo "ZZ OBD-ARM-REFUSED $t gate-or-pin-refused rc=4 :: $d" > "$STATE/ARM_REFUSED_$t.txt"; log "arm $t refused at its gate; the next independent arm goes on"; REFUSED="$REFUSED $t"
  else map_rc "$t" "$rc"; fi
done
if [ -n "$REFUSED" ]; then echo "ZZ OBD-STOP runs arms-refused:$REFUSED rc=4" > "$STATE/STOP.txt"; log "finished with refused arms:$REFUSED"; release_box; exit 4; fi
echo "ZZ OBD-DONE $(date '+%F %T') commit=$OBD_COMMIT runs=${ORDER[*]}" > "$STATE/DONE.txt"; log "all runs banked"
exit 0
