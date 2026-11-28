#!/bin/bash
# MM 2026-11-26: P's seam score aborted on the harness RAM guard (W3_RAM_GUARD_ABORT, available < 3,000 MB; the desktop's own apps hold
# most of the 32 GB). Wait until available RAM >= 4,500 MB on 5 consecutive 1-min samples, then (1) relaunch P's seam chain (resumes at
# score: chain.FAILED is set aside by the chain itself), (2) 90 s later relaunch the read waiter (it sets its read.FAILED aside itself).
# Both detached (WMI) through no-argument .cmd files (no quoting). Give-up 24 h. Log: E:/Projects/TanitAD-artifacts/scorerft_read/relaunch_p.log
LOG=E:/Projects/TanitAD-artifacts/scorerft_read/relaunch_p.log
SEAM_CMD='cmd.exe /c E:\Projects\TanitAD-artifacts\scorer_gym_20261107\scorerft_read\run_sft_seam_p.cmd'
READ_CMD='cmd.exe /c E:\Projects\TanitAD-artifacts\scorer_gym_20261107\scorerft_read\run_sft_read_p.cmd'
say() { echo "[$(date -u +%FT%TZ)] RELAUNCHP $*" >> "$LOG"; }
avail() { powershell -NoProfile -Command "(Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes" 2>/dev/null | tr -d '\r\n '; }
wmi() { powershell -NoProfile -Command "\$si = (New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }); \$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = '$1'; ProcessStartupInformation = \$si }; 'ReturnValue=' + \$r.ReturnValue + ' ProcessId=' + \$r.ProcessId" 2>&1 | tr -d '\r'; }
say "start pid $$; waiting for available RAM >= 4500 MB x5"
ok=0
for i in $(seq 1 1440); do
  a=$(avail); case "$a" in ''|*[!0-9]*) a=0;; esac
  if [ "$a" -ge 4500 ]; then ok=$((ok+1)); else ok=0; fi
  [ $((i % 30)) = 1 ] && say "available ${a} MB (ok ${ok}/5)"
  [ $ok -ge 5 ] && break
  sleep 60
done
[ $ok -ge 5 ] || { say "gave up after 24 h without 4500 MB available"; exit 2; }
say "memory OK (${a} MB): launching P's seam chain: $(wmi "$SEAM_CMD")"
sleep 90
say "launching the read waiter: $(wmi "$READ_CMD")"
say "done"
