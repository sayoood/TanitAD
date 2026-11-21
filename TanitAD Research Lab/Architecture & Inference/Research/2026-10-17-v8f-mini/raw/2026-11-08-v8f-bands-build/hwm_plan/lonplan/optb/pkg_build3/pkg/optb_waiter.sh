#!/usr/bin/env bash
# optb_waiter.sh: sentinel waiter for the Option B chain. Polls by script (no model turns); wakes on the FIRST sentinel that is NEW relative to a baseline taken BEFORE the chain launches.
#   bash optb_waiter.sh snap [STATE]            take the baseline (run it before launching the chain; it refuses to overwrite one)
#   bash optb_waiter.sh wait [STATE] [POLL_S]   block until: a new STOP.txt / DONE.txt / ARM_REFUSED_<tag>.txt / HF_BANK_PENDING_<tag>.txt, or the chain DRIVER process died (seen alive, then absent twice in a row), or MAX_POLLS ran out
# Sentinel text is read only from the sentinel files (prefix "ZZ OPTB-"), never from chain.log. The wait step prints exactly one line "WAKE <kind> <file>" and exits 0 for DONE, 10 for STOP / refusal / pending, 11 for driver-dead, 12 for timeout.
# The driver check matches the bash.exe command line that holds the chain script name (pattern built from parts so this script never matches itself) and excludes powershell.exe.
set -u
CMD="${1:?snap|wait}"; STATE="${2:-${OPTB_STATE:-C:/Users/Admin/optb_state}}"; POLL="${3:-60}"; BASE="$STATE/waiter_baseline.txt"
snapshot() { local f; for f in STOP.txt DONE.txt ARM_REFUSED_* HF_BANK_PENDING_*; do for g in "$STATE"/$f; do [ -e "$g" ] && echo "$(basename "$g") $(stat -c '%Y %s' "$g")"; done; done | sort; }
case "$CMD" in
  snap) mkdir -p "$STATE"; [ ! -e "$BASE" ] || { echo "REFUSED: baseline $BASE exists (move it aside to take a new one)" >&2; exit 2; }; snapshot > "$BASE"; echo "baseline taken: $(wc -l < "$BASE") sentinel files already present"; exit 0;;
  wait) [ -e "$BASE" ] || { echo "REFUSED: no baseline in $STATE (run snap BEFORE launching the chain)" >&2; exit 2; };;
  *) echo "usage: snap|wait" >&2; exit 2;;
esac
P1="optb_"; P2="chain.sh"
driver_alive() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*${P1}${P2}*' }).Count" 2>/dev/null | tr -d '\r' | tail -1; }
seen=0; miss=0; n=0
while :; do
  new="$(snapshot | grep -vxFf "$BASE" | head -1)"
  if [ -n "$new" ]; then
    f="${new%% *}"; head -c 8 "$STATE/$f" | grep -q "^ZZ OPTB-" || { sleep "$POLL"; continue; }
    case "$f" in DONE.txt) echo "WAKE done $f"; exit 0;; *) echo "WAKE sentinel $f"; exit 10;; esac
  fi
  a="$(driver_alive)"; case "$a" in ''|*[!0-9]*) a=1;; esac
  if [ "$a" -gt 0 ]; then seen=1; miss=0; elif [ "$seen" = 1 ]; then miss=$((miss+1)); [ "$miss" -lt 2 ] || { echo "WAKE driver-dead (no chain process, no new sentinel)"; exit 11; }; fi
  n=$((n+1)); [ "$n" -lt "${OPTB_WAITER_MAX_POLLS:-1500}" ] || { echo "WAKE timeout after $n polls"; exit 12; }
  sleep "$POLL"
done
