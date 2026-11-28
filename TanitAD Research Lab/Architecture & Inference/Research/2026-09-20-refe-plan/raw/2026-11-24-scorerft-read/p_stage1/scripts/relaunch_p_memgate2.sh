#!/bin/bash
# MM 2026-11-26 ~21:55Z: P's det step (the rep seam) died at 2,413 s on numpy _ArrayMemoryError (5.62 MiB): the dev box hit its COMMIT
# limit (Chrome ~21.7 GB commit), not the physical-RAM guard. Gate on BOTH: available RAM >= 4,500 MB AND commit free >= 8,000 MB, on 5
# consecutive 1-min samples; then relaunch P's seam chain (resumes at det; it sets chain.FAILED aside itself) and, 90 s later, the read
# waiter (it sets read.FAILED aside itself). Detached (WMI) via the same no-argument .cmd files. Give-up 24 h. Log: relaunch_p2.log
LOG=E:/Projects/TanitAD-artifacts/scorerft_read/relaunch_p2.log
SEAM_CMD='cmd.exe /c E:\Projects\TanitAD-artifacts\scorer_gym_20261107\scorerft_read\run_sft_seam_p.cmd'
READ_CMD='cmd.exe /c E:\Projects\TanitAD-artifacts\scorer_gym_20261107\scorerft_read\run_sft_read_p.cmd'
say() { echo "[$(date -u +%FT%TZ)] RELAUNCHP2 $*" >> "$LOG"; }
avail() { powershell -NoProfile -Command "(Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes" 2>/dev/null | tr -d '\r\n '; }
cfree() { powershell -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1024)" 2>/dev/null | tr -d '\r\n '; }
wmi() { powershell -NoProfile -Command "\$si = (New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }); \$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = '$1'; ProcessStartupInformation = \$si }; 'ReturnValue=' + \$r.ReturnValue + ' ProcessId=' + \$r.ProcessId" 2>&1 | tr -d '\r'; }
say "start pid $$; waiting for available RAM >= 4500 MB and commit free >= 8000 MB x5"
ok=0
for i in $(seq 1 1440); do
  a=$(avail); case "$a" in ''|*[!0-9]*) a=0;; esac
  c=$(cfree); case "$c" in ''|*[!0-9]*) c=0;; esac
  if [ "$a" -ge 4500 ] && [ "$c" -ge 8000 ]; then ok=$((ok+1)); else ok=0; fi
  [ $((i % 30)) = 1 ] && say "available ${a} MB, commit free ${c} MB (ok ${ok}/5)"
  [ $ok -ge 5 ] && break
  sleep 60
done
[ $ok -ge 5 ] || { say "gave up after 24 h"; exit 2; }
say "memory OK (avail ${a} MB, commit free ${c} MB): launching P's seam chain: $(wmi "$SEAM_CMD")"
sleep 90
say "launching the read waiter: $(wmi "$READ_CMD")"
say "done"
