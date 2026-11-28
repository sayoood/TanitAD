#!/bin/bash
# adopt_rep_and_relaunch.sh (MM 2026-11-13): the SLOWWM driver chain died at ~20:34 on 2026-11-12 (its console log ends in "^C", 7 min after it launched
# SLOW_rep on box 2; no STOP.txt, no EXIT trap ran, so BOX2_BUSY.txt kept its mark). The box-2 job ran to the end: STAGE_20261112_202710.done,
# "SLOWWM-OK SLOW_rep", SLOWWM-JOB-END rc=0 at 23:46:53.
# 1 move the dead chain's busy marker aside (only if it is exactly that chain's mark and no slowwm_chain process runs), 2 claim box 2 (noclobber),
# 3 run the chain's own post-DONE steps on that job (adopt_rep_run.sh: pull + sha verify, verify-token, bank manifest, cleanup job),
# 4 write done/SLOW_rep.done in the chain's own format, 5 release box 2, 6 exec the normal relaunch (the chain then skips every stage and run,
# writes its own DONE.txt and releases box 2). On any failure: one line in $LOG_DIR/ADOPT_STOP.txt, box 2 released, the chain is NOT started.
set -u
export SLOWWM_COMMIT=7296df50017cd3f4e9b226cd7acca30bda5fa3ff
export SLOWWM_SSH="$(command -v ssh)" SLOWWM_SFTP="$(command -v sftp)"
export SLOWWM_STATE=C:/Users/Admin/slowwm_state
LOG_DIR=C:/Users/Admin/slowwm_relaunch
BUSY=C:/Users/Admin/box2_ship/BOX2_BUSY.txt; ASIDE=C:/Users/Admin/box2_ship/busy_aside
STALE="slowwm chain 834 2026-11-12 07:02:31"
lg() { echo "[$(date '+%F %T')] adopt-rep: $*" | tee -a "$LOG_DIR/adopt_rep.log"; }
OWN=""
release() { if [ -n "$OWN" ] && [ -e "$BUSY" ] && [ "$(cat "$BUSY" 2>/dev/null)" = "$OWN" ]; then mv "$BUSY" "$ASIDE/BOX2_BUSY.slowwm_adopt_rep.$(date +%Y%m%d_%H%M%S).txt"; lg "released box 2"; fi; OWN=""; }
die() { echo "ZZ SLOWWM-ADOPT-STOP $* $(date '+%F %T')" > "$LOG_DIR/ADOPT_STOP.txt"; lg "STOP: $*"; release; exit 1; }
trap release EXIT
lg "start ssh=$SLOWWM_SSH"
[ -e "$SLOWWM_STATE/done/SLOW_rep.done" ] && die "SLOW_rep.done already exists"
[ -e "$SLOWWM_STATE/STOP.txt" ] && die "a SLOWWM STOP.txt exists: inspect first"
[ -e "$SLOWWM_STATE/DONE.txt" ] && die "a SLOWWM DONE.txt exists"
for t in SLOW_s0 PCA-B_s0 SLOW_s1 PCA-B_s1; do [ -e "$SLOWWM_STATE/done/$t.done" ] || die "done/$t.done missing: the relaunch would train $t"; done
n="$(powershell -NoProfile -Command "\$p='slowwm_'+'chain'; @(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like \"*\$p*\" -and \$_.Name -ne 'powershell.exe' }).Count" | tr -d '\r')"
[ "$n" = 0 ] || die "a slowwm_chain process is running ($n)"
if [ -e "$BUSY" ]; then
  [ "$(cat "$BUSY")" = "$STALE" ] || die "busy marker is not the dead chain's mark: $(cat "$BUSY")"
  mv "$BUSY" "$ASIDE/BOX2_BUSY.slowwm_stale_834.$(date +%Y%m%d_%H%M%S).txt" || die "cannot move the stale marker aside"
  lg "moved aside the dead chain's busy marker ($STALE)"
fi
( set -o noclobber; echo "slowwm adopt-rep $$ $(date '+%F %T')" > "$BUSY" ) 2>/dev/null || die "cannot claim $BUSY"
OWN="$(cat "$BUSY")"; lg "claimed box 2 ($OWN)"
bash "$LOG_DIR/adopt_rep_run.sh" SLOW_rep >> "$LOG_DIR/adopt_rep.log" 2>&1; rc=$?
[ "$rc" -eq 0 ] || die "adopt_rep_run.sh rc=$rc (see $LOG_DIR/adopt_rep.log)"
[ -s "C:/Users/Admin/t2v8f_pulls/slowwm_SLOW_rep/BANK.done" ] || die "no BANK.done after the adopt"
echo "SLOW_rep $(date '+%F %T') adopted-gid=20261112_202710" > "$SLOWWM_STATE/done/SLOW_rep.done"
lg "SLOW_rep adopted and banked"
release
trap - EXIT
lg "exec the normal relaunch"
exec bash C:/Users/Admin/slowwm_relaunch/launch.sh
