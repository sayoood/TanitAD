#!/bin/bash
# MM 2026-11-28: ENS-SCORE relaunch after seam_ENS1 died rc 3221225477 (0xC0000005, native access violation, no WER record) at ~1,300/12,146 tokens.
# 1. Wait until x150 has DUMPED every already-extracted log (GPU ~76 min), so ENS's 8.4 h seam does not strand them; cap 4 h.
# 2. Relaunch run_ens_chain.cmd (the chain is marker-resumable: it keeps chain.FAILED as attempt_N and re-runs only the seam onward).
# 3. Every 60 s log commit free + the ENS seam process's private MB, to diagnose a second crash (commit exhaustion vs a true native fault).
# 4. If the chain FAILS again at seam_ENS1 and the trace shows commit free never below 1,500 MB, relaunch ONCE more; otherwise stop and report.
# Launched via WMI; Git usr/bin is NOT on a WMI bash's PATH, so it is set first. Log: mm/ens_relaunch.log, trace: mm/ens_memtrace.tsv
export PATH="/usr/bin:/bin:/mingw64/bin:/c/Windows/System32:/c/Windows/System32/WindowsPowerShell/v1.0:$PATH"
LOG=E:/Projects/TanitAD-artifacts/mm/ens_relaunch.log; TR=E:/Projects/TanitAD-artifacts/mm/ens_memtrace.tsv
EM=E:/Projects/TanitAD-artifacts/ensscore/work/markers; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
ps1() { powershell -NoProfile -Command "$1" < /dev/null | tr -d '\r'; }
launch() { say "launch ENS chain: $(ps1 "\$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags = [uint32]0x08000200}; \$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = 'cmd.exe /c E:\Projects\TanitAD-artifacts\ensscore\run_ens_chain.cmd'; ProcessStartupInformation = \$si}; 'rv {0} pid {1}' -f \$r.ReturnValue, \$r.ProcessId")"; }
sample() { ps1 "\$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; \$cf=[math]::Floor((\$m.CommitLimit-\$m.CommittedBytes)/1MB); \$p=Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*ens_select.py*' } | Select-Object -First 1; \$pm=0; if (\$p) { \$pm=[math]::Round((Get-Process -Id \$p.ProcessId -ErrorAction SilentlyContinue).PrivateMemorySize64/1MB) }; '{0} {1}' -f \$cf, \$pm" | tr ' ' '\t'; }
say "start pid $$"
T0=$(date +%s)
until $PY -I E:/Projects/TanitAD-artifacts/mm/x150_dumped_all_fetched.py > /dev/null 2>&1 || [ $(( $(date +%s) - T0 )) -ge 14400 ]; do sleep 120; done
say "x150 dump state: $($PY -I E:/Projects/TanitAD-artifacts/mm/x150_dumped_all_fetched.py)"
[ -f $TR ] || printf 'utc\tcommit_free_mb\tens_private_mb\n' > $TR
for attempt in 2 3; do
  launch
  w=0; while [ -f $EM/chain.FAILED ] && [ $w -lt 30 ]; do sleep 10; w=$((w+1)); done   # the chain sets the OLD chain.FAILED aside first; never read it as new
  [ -f $EM/chain.FAILED ] && { say "stop: the old chain.FAILED was not set aside within 5 min (did the chain start?)"; exit 1; }
  until [ -f $EM/chain.DONE ] || [ -f $EM/chain.FAILED ]; do printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$(sample)" >> $TR; sleep 60; done
  [ -f $EM/chain.DONE ] && { say "ENS chain DONE ($(ls $EM | tr '\n' ' '))"; exit 0; }
  f=$(cat $EM/chain.FAILED); say "ENS chain FAILED again: $f"
  minc=$(awk -F'\t' -v t="$(date -u -d "-12 hours" +%FT%TZ)" 'NR > 1 && $3 > 0 && $1 > t { if (m == "" || $2 < m) m = $2 } END { print m + 0 }' $TR)
  say "min commit free while the ENS seam was alive: ${minc} MB"
  if [ $attempt = 2 ] && echo "$f" | grep -q "stage=seam_ENS1" && [ "${minc:-0}" -ge 1500 ]; then say "commit never exhausted: relaunching once more"; sleep 120; continue; fi
  say "stop: not retried (attempt $attempt; min commit ${minc} MB)"; exit 1
done
