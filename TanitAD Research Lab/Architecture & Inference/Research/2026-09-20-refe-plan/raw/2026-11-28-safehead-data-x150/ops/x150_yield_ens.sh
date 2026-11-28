#!/bin/bash
# MM 2026-11-28: x150 YIELDS to ENS-SCORE, the successor of x150_yield.sh (which covers TRK only and exits when TRK's chain ends).
# Same logic: ENS waiting at its card-pinned memgate (avail >= 4,500 MB x5 + commit >= 6 GB) and x150 alive -> stop x150 top-down
# (resumable, atomic writes; set aside a GPU lock left by a dead x150 pid); ENS past its gate for 5 min and x150 down -> relaunch x150.
# Phase 0 waits for TRK's chain.DONE|FAILED (x150_yield.sh owns that window). Ends when ENS's chain.DONE|FAILED exists (x150 left running).
# Launched via WMI; Git usr/bin is NOT on a WMI bash's PATH, so it is set first. Log: mm/x150_yield_ens.log
export PATH="/usr/bin:/bin:/mingw64/bin:/c/Windows/System32:/c/Windows/System32/WindowsPowerShell/v1.0:$PATH"
TM=E:/Projects/TanitAD-artifacts/trkscore/work/markers
EL=E:/Projects/TanitAD-artifacts/ensscore/work/CHAIN.log; EM=E:/Projects/TanitAD-artifacts/ensscore/work/markers
LOG=E:/Projects/TanitAD-artifacts/mm/x150_yield_ens.log; LK=C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
ps1() { powershell -NoProfile -Command "$1" < /dev/null | tr -d '\r'; }
x150_root() { ps1 "Get-CimInstance Win32_Process -Filter \"Name='cmd.exe'\" | Where-Object { \$_.CommandLine -like '*run_x150_chain.cmd*' } | ForEach-Object { \$_.ProcessId }" | head -1; }
stop_x150() {
  local r=$(x150_root); [ -z "$r" ] && return 0
  ps1 "\$all = Get-CimInstance Win32_Process; \$ids = @($r); \$i = 0; while (\$i -lt \$ids.Count) { \$p = \$ids[\$i]; \$all | Where-Object { \$_.ParentProcessId -eq \$p } | ForEach-Object { \$ids += \$_.ProcessId }; \$i++ }; foreach (\$id in \$ids) { & taskkill /PID \$id /F 2>&1 | Out-Null; \"killed \$id\" }" | tr '\n' ' ' >> $LOG; echo >> $LOG
  sleep 5
  if [ -f $LK ] && grep -q '"job": "refe-safehead-dump"' $LK; then
    local lp=$(grep -oE '"pid": [0-9]+' $LK | grep -oE '[0-9]+')
    [ "$(ps1 "if (Get-Process -Id $lp -ErrorAction SilentlyContinue) { 'ALIVE' } else { 'DEAD' }")" = DEAD ] && mv $LK $LK.stale_refe-safehead-dump_$lp && say "set aside stale x150 GPU lock (pid $lp)"
  fi
}
start_x150() { say "relaunch x150: $(ps1 "\$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags = [uint32]0x08000200}; \$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = 'cmd.exe /c F:\safehead_cache\code_x150\run_x150_chain.cmd'; ProcessStartupInformation = \$si}; 'rv {0} pid {1}' -f \$r.ReturnValue, \$r.ProcessId")"; }
say "start pid $$; phase 0: waiting for TRK chain end"
until [ -f $TM/chain.DONE ] || [ -f $TM/chain.FAILED ]; do sleep 120; done
say "TRK chain ended ($(ls $TM | tr '\n' ' ')); phase 1: yield to ENS"
sleep 180   # let x150_yield.sh finish its own exit (it relaunches x150 if down) before this loop acts
T0=$(date +%s); open_since=0
while [ $(( $(date +%s) - T0 )) -lt 172800 ]; do
  if [ -f $EM/chain.DONE ] || [ -f $EM/chain.FAILED ]; then
    [ -z "$(x150_root)" ] && [ ! -f F:/safehead_cache/x150/markers/chain.DONE ] && start_x150
    say "ENS chain ended ($(ls $EM | tr '\n' ' ')); exit"; exit 0
  fi
  last=$(grep "ENS " $EL | tail -1)
  if echo "$last" | grep -qE "memgate .*ok [0-4]/5"; then
    open_since=0
    [ -n "$(x150_root)" ] && { say "ENS waiting: '$last' -> stop x150"; stop_x150; }
  else
    [ $open_since = 0 ] && open_since=$(date +%s)
    if [ -z "$(x150_root)" ] && [ ! -f F:/safehead_cache/x150/markers/chain.DONE ] && [ $(( $(date +%s) - open_since )) -ge 300 ]; then say "ENS past its gate: '$last'"; start_x150; fi
  fi
  sleep 120
done
say "48 h cap reached; exit"
