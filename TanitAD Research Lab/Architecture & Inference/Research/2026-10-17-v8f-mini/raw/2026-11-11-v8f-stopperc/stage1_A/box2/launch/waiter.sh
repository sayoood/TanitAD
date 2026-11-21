#!/bin/bash
# waiter.sh (MM 2026-11-12 ~20:40): launches the STOPPERC stage-1 chain (arm A + NULLs, box 2) after the PI's go ("Yes, run arm A", 2026-11-12 ~20:30).
# Launch only when ALL hold: (1) MM_DRYRUN_OK.json (the MM's reviewed copy of the dry-run pass file) reads PASS and its sha256 values cover the CURRENT sha256 of the 10 package files below
# (a fix made after the dry run, or an unpassed package, keeps it waiting); (2) the SLOWWM chain has ended (its DONE.txt or STOP.txt exists); (3) box 2's busy marker is absent.
# Then it runs stopperc_chain.sh in the foreground with the real ssh/sftp. A start refusal for box2-busy (another owner claimed box 2 first) goes back to waiting with
# STOPPERC_RESUME=1; any other end is final and is left for the MM. Gives up at 2026-11-13 14:00 (WAITER_STOP.txt). Sentinels go to their own files, never to waiter.log.
set -u
D=C:/Users/Admin/stopperc_launch
LOG=$D/waiter.log
BUSY=C:/Users/Admin/box2_ship/BOX2_BUSY.txt
PASS=C:/Users/Admin/stopperc_launch/MM_DRYRUN_OK.json   # written by the MM after reviewing the agent's C:/Users/Admin/stopperc_dry/DRYRUN_PASS.json (same schema)
SLOW=C:/Users/Admin/slowwm_state
ST=C:/Users/Admin/stopperc_state
TH=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory
PKG=$TH/stopperc/stage1_A/box2
BAL=$TH/stopbal/box2
PYV=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
DEADLINE=$(date -d '2026-11-13 14:00' +%s)
lg() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
last=""
note() { [ "$1" != "$last" ] && lg "$1"; last="$1"; }
pass_ok() {
  "$PYV" -I - "$PASS" "$PKG/stopperc_chain.sh" "$PKG/stopperc_check.py" "$PKG/argvs_stopperc.txt" "$PKG/make_argvs_stopperc.py" "$PKG/jobs/stopperc_ship_job.cmd" "$PKG/jobs/stopperc_stage_job.cmd" \
    "$BAL/stages/stopbal_common.sh" "$BAL/stopbal_check.py" "$BAL/stopbal_ship_tree.py" "$BAL/stopbal_refresh.py" << 'PYEOF'
import hashlib, json, os, sys
p = sys.argv[1]
if not os.path.exists(p): print("no pass file"); sys.exit(1)
j = json.load(open(p, encoding="utf-8"))
if j.get("result") != "PASS" or j.get("failed", 0) != 0: print("pass file does not read PASS"); sys.exit(1)
have = {str(v).lower() for v in (j.get("sha256") or {}).values()}
miss = [f for f in sys.argv[2:] if hashlib.sha256(open(f, "rb").read()).hexdigest() not in have]
if miss: print("files changed since the dry run: " + " ".join(os.path.basename(m) for m in miss)); sys.exit(1)
print("pass ok"); sys.exit(0)
PYEOF
}
lg "waiter start pid $$ (PI go 2026-11-12 ~20:30)"
tries=0
while :; do
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "ZZ STOPPERC-WAITER-STOP deadline 2026-11-13 14:00 passed; last state: $last" > "$D/WAITER_STOP.txt"; lg "deadline passed"; exit 3; }
  if [ -e "$ST/DONE.txt" ]; then lg "the chain already wrote DONE.txt: nothing to launch"; exit 0; fi
  if [ -e "$ST/STOP.txt" ] && ! grep -q '^ZZ STOPPERC-STOP start box2-busy' "$ST/STOP.txt"; then echo "ZZ STOPPERC-WAITER-STOP the chain stopped: $(head -c 300 "$ST/STOP.txt")" > "$D/WAITER_STOP.txt"; lg "chain STOP present: left for the MM"; exit 4; fi
  r="$(pass_ok 2>&1 | tail -1)"; if [ "$r" != "pass ok" ]; then note "waiting: dry run ($r)"; sleep 120; continue; fi
  if [ ! -e "$SLOW/DONE.txt" ] && [ ! -e "$SLOW/STOP.txt" ]; then note "waiting: SLOWWM chain not ended (dry run ok)"; sleep 120; continue; fi
  if [ -e "$BUSY" ]; then note "waiting: box 2 busy ($(head -c 80 "$BUSY"))"; sleep 120; continue; fi
  tries=$((tries+1)); [ "$tries" -le 20 ] || { echo "ZZ STOPPERC-WAITER-STOP 20 busy refusals" > "$D/WAITER_STOP.txt"; exit 5; }
  RES=0; [ -e "$ST/STOP.txt" ] && RES=1
  lg "launch try $tries: SLOWWM ended ($(ls "$SLOW" | grep -E '^(DONE|STOP)\.txt$' | tr '\n' ' ')), box 2 free, dry run ok; RESUME=$RES"
  STOPPERC_SSH="$(command -v ssh)" STOPPERC_SFTP="$(command -v sftp)" STOPPERC_RESUME=$RES bash "$PKG/stopperc_chain.sh" >> "$D/chain_console.log" 2>&1; rc=$?
  lg "chain exited rc=$rc"
  if [ "$rc" -eq 12 ] && grep -q '^ZZ STOPPERC-STOP start box2-busy' "$ST/STOP.txt" 2>/dev/null; then lg "another owner claimed box 2 first: back to waiting"; sleep 120; continue; fi
  echo "ZZ STOPPERC-WAITER-END chain rc=$rc $(date '+%F %T')" > "$D/WAITER_END.txt"; exit 0
done
