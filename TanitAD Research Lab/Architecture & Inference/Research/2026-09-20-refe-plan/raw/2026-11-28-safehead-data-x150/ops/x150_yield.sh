#!/bin/bash
# MM 2026-11-27: x150 YIELDS to TRK-SCORE. TRK's card-pinned memgate (avail >= 4,500 MB x5 + commit >= 6 GB) cannot open while x150's
# dump/labels run beside the PI's apps. Loop (2 min, <= 36 h): TRK waiting at a memgate and x150 alive -> stop x150 top-down (resumable,
# atomic writes) and set aside a GPU lock left by a dead x150 pid; TRK past its gate for 5 min and x150 down -> relaunch x150.
# Ends when TRK's markers/chain.DONE or chain.FAILED exists (x150 is then left running / relaunched). Log: mm/x150_yield.log
TL=E:/Projects/TanitAD-artifacts/trkscore/work/CHAIN.log; TM=E:/Projects/TanitAD-artifacts/trkscore/work/markers
LOG=E:/Projects/TanitAD-artifacts/mm/x150_yield.log; LK=C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock
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
say "start (x150 root: $(x150_root))"
T0=$(date +%s); open_since=0
while [ $(( $(date +%s) - T0 )) -lt 129600 ]; do
  if [ -f $TM/chain.DONE ] || [ -f $TM/chain.FAILED ]; then
    [ -z "$(x150_root)" ] && start_x150; say "TRK chain ended ($(ls $TM | tr '\n' ' ')); exit"; exit 0
  fi
  last=$(grep "TRK " $TL | tail -1)
  if echo "$last" | grep -qE "memgate .*ok [0-4]/5"; then
    open_since=0
    [ -n "$(x150_root)" ] && { say "TRK waiting: '$last' -> stop x150"; stop_x150; }
  else
    [ $open_since = 0 ] && open_since=$(date +%s)
    if [ -z "$(x150_root)" ] && [ $(( $(date +%s) - open_since )) -ge 300 ]; then say "TRK past its gate: '$last'"; start_x150; fi
  fi
  sleep 120
done
say "36 h cap reached; exit"
