#!/bin/bash
# adopt_and_relaunch.sh (MM 2026-11-12): the dev box restarted at 01:32:46 (Windows Update, event 1074) while the SLOWWM chain (pid 146542) waited on the
# pilot stage job 20261112_011153. That job finished DONE on box 2 at 01:58:43 ("SLOWWM-OK PILOT_SLOW_s0 mean 0.4704 s/step, 12.5 h planned").
# 1 move the dead chain's busy marker aside (only if it is exactly that chain's mark and its pid is gone), 2 claim box 2 (noclobber),
# 3 run the chain's own post-DONE pilot steps on that job (adopt_pilot_run.sh: pull + sha verify, project, verify-token, cleanup job),
# 4 write done/pilot.done in the chain's own format, 5 release box 2, 6 exec the normal relaunch (the chain then skips ship/basis/pilot and runs permit + the five runs).
# On any failure: one line in $LOG_DIR/ADOPT_STOP.txt, box 2 released, the chain is NOT started.
set -u
export SLOWWM_COMMIT=7296df50017cd3f4e9b226cd7acca30bda5fa3ff
export SLOWWM_SSH="$(command -v ssh)" SLOWWM_SFTP="$(command -v sftp)"
export SLOWWM_STATE=C:/Users/Admin/slowwm_state
LOG_DIR=C:/Users/Admin/slowwm_relaunch
BUSY=C:/Users/Admin/box2_ship/BOX2_BUSY.txt; ASIDE=C:/Users/Admin/box2_ship/busy_aside
PKG=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/slowwm/box2
STALE="slowwm chain 146542 2026-11-12 01:11:18"
lg() { echo "[$(date '+%F %T')] adopt: $*" | tee -a "$LOG_DIR/adopt.log"; }
OWN=""
release() { if [ -n "$OWN" ] && [ -e "$BUSY" ] && [ "$(cat "$BUSY" 2>/dev/null)" = "$OWN" ]; then mv "$BUSY" "$ASIDE/BOX2_BUSY.slowwm_adopt.$(date +%Y%m%d_%H%M%S).txt"; lg "released box 2"; fi; OWN=""; }
die() { echo "ZZ SLOWWM-ADOPT-STOP $* $(date '+%F %T')" > "$LOG_DIR/ADOPT_STOP.txt"; lg "STOP: $*"; release; exit 1; }
trap release EXIT
lg "start ssh=$SLOWWM_SSH"
[ -e "$SLOWWM_STATE/done/pilot.done" ] && die "pilot.done already exists"
[ -e "$SLOWWM_STATE/STOP.txt" ] && die "a SLOWWM STOP.txt exists: inspect first"
if [ -e "$BUSY" ]; then
  [ "$(cat "$BUSY")" = "$STALE" ] || die "busy marker is not the dead chain's mark: $(cat "$BUSY")"
  ps -p 146542 > /dev/null 2>&1 && die "pid 146542 is alive"
  mv "$BUSY" "$ASIDE/BOX2_BUSY.slowwm_stale_146542.$(date +%Y%m%d_%H%M%S).txt" || die "cannot move the stale marker aside"
  lg "moved aside the dead chain's busy marker ($STALE)"
fi
( set -o noclobber; echo "slowwm adopt $$ $(date '+%F %T')" > "$BUSY" ) 2>/dev/null || die "cannot claim $BUSY"
OWN="$(cat "$BUSY")"; lg "claimed box 2 ($OWN)"
bash "$LOG_DIR/adopt_pilot_run.sh" PILOT_SLOW_s0 >> "$LOG_DIR/adopt.log" 2>&1; rc=$?
[ "$rc" -eq 0 ] || die "adopt_pilot_run.sh rc=$rc (see $LOG_DIR/adopt.log)"
[ -s "$SLOWWM_STATE/pilot/PILOT.verified.json" ] || die "no PILOT.verified.json after the adopt"
ck="$(sha256sum "$PKG/slowwm_check.py" | cut -d' ' -f1)"
echo "pilot $(date '+%F %T') checker=$ck adopted-gid=20261112_011153" > "$SLOWWM_STATE/done/pilot.done"
lg "pilot adopted: $(head -c 300 "$SLOWWM_STATE/pilot/PILOT.verified.json" | tr '\n' ' ')"
release
trap - EXIT
lg "exec the normal relaunch"
exec bash C:/Users/Admin/slowwm_relaunch/launch.sh
