#!/usr/bin/env bash
# PREREG_RA1_FULL: the full-corpus build on box 2 (code/ra1_full_box2.py as ONE job, SAFE MODE), watched by the boot-time crash guard.
# Starts only when (1) the input transfer reported ZZSHIP-DONE and (2) box 2 is idle: no other chain_ra1_*.sh runs on the dev box and
# no python runs on box 2, at two checks 60 s apart. Stops on a reboot (waitjob exit 2) or any nonzero rc. Script polling, not model turns.
B=/c/Users/Admin/box2_ship; L=$B/logs
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
echo "$(date '+%m-%d %H:%M') waiting for the transfer"
until grep -q ZZSHIP-DONE $L/ship_trainall.log 2>/dev/null; do
  grep -q ZZSHIP-FAIL $L/ship_trainall.log 2>/dev/null && { echo "ZZRA1FULL-STOP the transfer failed: $(grep ZZSHIP-FAIL $L/ship_trainall.log)"; exit 1; }
  sleep 60
done
echo "$(date '+%m-%d %H:%M') transfer done: $(grep ZZSHIP-DONE $L/ship_trainall.log)"
idle=0
for i in $(seq 1 720); do
  nl=$(ps -ef 2>/dev/null | grep -E 'bash [^ ]*chain_ra1_[A-Za-z0-9_]*\.sh' | grep -v chain_ra1_full | grep -v grep | wc -l)
  nb=$($SSH "powershell -NoProfile -Command \"@(Get-Process python -ErrorAction SilentlyContinue).Count\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$')
  if [ "$nl" = 0 ] && [ "$nb" = 0 ]; then idle=$((idle + 1)); else idle=0; fi
  [ $idle -ge 2 ] && break
  [ $((i % 30)) = 1 ] && echo "$(date '+%m-%d %H:%M') box 2 busy (other chains $nl, python on box 2 '${nb:-?}')"
  sleep 60
done
[ $idle -ge 2 ] || { echo "ZZRA1FULL-STOP box 2 never idle in 12 h"; exit 1; }
BOOT0=$(boot); [ -n "$BOOT0" ] || { echo "ZZRA1FULL-STOP box 2 unreachable"; exit 1; }
echo "$(date '+%m-%d %H:%M') box 2 idle, boot0=$BOOT0"
$PY "$(cygpath -m $B)/mk_full_build.py" >/dev/null || { echo "ZZRA1FULL-STOP job files"; exit 1; }
scp -q -o BatchMode=yes -o ConnectTimeout=20 $B/ra1full_build.cmd $B/telemetry_long.cmd tanitad-box2:D:/tanitad/jobs/ && \
  scp -q -o BatchMode=yes -o ConnectTimeout=20 $B/telemetry_long.ps1 tanitad-box2:D:/tanitad/ || { echo "ZZRA1FULL-STOP ship job files"; exit 1; }
echo "ra1full_build (session bfe78918, PREREG_RA1_FULL) launched $(date '+%m-%d %H:%M'), ~22 h. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch (the build goes on without it)"
launch ra1full_build || { echo "ZZRA1FULL-STOP launch"; exit 1; }
bash $B/waitjob.sh ra1full_build 4320 "$BOOT0"; w=$?
# the job is over only on a clean end or a reboot; on 3 (unreachable) / 4 (timeout) it may still run, so the marker stays
{ [ $w = 0 ] || [ $w = 2 ]; } && rm -f $B/BOX2_BUSY.txt
$SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
mkdir -p $L/ra1_full
for f in data/trainall_stage/ctl_result.json data/ra1_full_state.json logs/telemetry_full.csv; do
  scp -q -o BatchMode=yes -o ConnectTimeout=20 "tanitad-box2:D:/tanitad/$f" $L/ra1_full/ 2>/dev/null
done
[ $w = 0 ] || { echo "ZZRA1FULL-STOP build wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"; exit 1; }
rc=$(tr '\r' '\n' < $L/ra1full_build.log | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
tr '\r' '\n' < $L/ra1full_build.log | grep -a -E 'ZZCONTROL|ZZCHUNK-DONE|ZZ[A-Z0-9-]*(FAIL|MISMATCH|GUARD|INCOMPLETE|MISSING)|ZZRA1FULL-DONE|ZZVERIFY' | tail -20
echo "$(date '+%m-%d %H:%M') ra1full_build rc=$rc"
[ "$rc" = 0 ] || { echo "ZZRA1FULL-STOP build rc=$rc (10 control, 11 disk, 12 mp4 sha, 13 build, 14 encode, 15 verify)"; exit 1; }
echo "ZZRA1FULL-BUILD-DONE $(date '+%m-%d %H:%M') boot unchanged ($BOOT0)"
