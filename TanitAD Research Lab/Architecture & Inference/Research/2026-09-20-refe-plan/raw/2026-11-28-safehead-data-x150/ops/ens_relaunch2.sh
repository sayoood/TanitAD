#!/bin/bash
# MM 2026-11-28 (v2): ENS-SCORE relaunch with the seam run ALONE. Why: this box's commit is exhausted by other sessions' processes
# (x150's dump hit cv::OutOfMemoryError at 10:57Z; ENS seam_ENS1 died 0xC0000005 at 08:49Z), and the 8.4 h seam is NOT resumable
# while x150 is resumable per log. So:
# 1. Wait until x150 has DONE every log in mm/ens_wait_logs.txt (the 49 logs of already-extracted archives, snapshot 11:40Z); cap 3 h.
# 2. Stop x150 top-down (resumable), relaunch run_ens_chain.cmd (marker-resumable: re-runs the seam onward).
# 3. Every 60 s: commit free + the seam's private MB -> mm/ens_memtrace.tsv. When ENS1.seam.DONE exists (seam finished; ctrl/score are CPU)
#    or the chain ends, relaunch x150.
# 4. Chain FAILED again at seam_ENS1 with commit never < 1,500 MB -> relaunch ONCE more; else stop and report.
# Launched via WMI; Git usr/bin is NOT on a WMI bash's PATH, so it is set first. Log: mm/ens_relaunch.log
export PATH="/usr/bin:/bin:/mingw64/bin:/c/Windows/System32:/c/Windows/System32/WindowsPowerShell/v1.0:$PATH"
LOG=E:/Projects/TanitAD-artifacts/mm/ens_relaunch.log; TR=E:/Projects/TanitAD-artifacts/mm/ens_memtrace.tsv
EM=E:/Projects/TanitAD-artifacts/ensscore/work/markers; WL=E:/Projects/TanitAD-artifacts/mm/ens_wait_logs.txt
LK=C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock; X=F:/safehead_cache/x150
say() { echo "[$(date -u +%FT%TZ)] $*" >> $LOG; }
source <(sed -n '/^ps1()/,/^start_x150()/p' E:/Projects/TanitAD-artifacts/mm/x150_yield_ens.sh)   # ps1, x150_root, stop_x150, start_x150 (proven 07:34Z/07:59Z)
launch() { say "launch ENS chain: $(ps1 "\$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags = [uint32]0x08000200}; \$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = 'cmd.exe /c E:\Projects\TanitAD-artifacts\ensscore\run_ens_chain.cmd'; ProcessStartupInformation = \$si}; 'rv {0} pid {1}' -f \$r.ReturnValue, \$r.ProcessId")"; }
sample() { ps1 "\$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; \$cf=[math]::Floor((\$m.CommitLimit-\$m.CommittedBytes)/1MB); \$p=Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*ens_select.py*' } | Select-Object -First 1; \$pm=0; if (\$p) { \$pm=[math]::Round((Get-Process -Id \$p.ProcessId -ErrorAction SilentlyContinue).PrivateMemorySize64/1MB) }; '{0} {1}' -f \$cf, \$pm" | tr ' ' '\t'; }
pending() { local n=0; while read -r l; do [ -n "$l" ] && [ ! -f "$X/markers/log_$l.DONE" ] && n=$((n+1)); done < $WL; echo $n; }
say "v2 start pid $$; waiting for x150 to finish $(wc -l < $WL) logs ($(pending) pending)"
T0=$(date +%s)
until [ "$(pending)" = 0 ] || [ $(( $(date +%s) - T0 )) -ge 10800 ] || [ -f $X/markers/chain.FAILED ]; do sleep 120; done
say "x150 wait over: $(pending) of the set pending$( [ -f $X/markers/chain.FAILED ] && echo '; x150 chain.FAILED present')"
[ -f $TR ] || printf 'utc\tcommit_free_mb\tens_private_mb\n' > $TR
for attempt in 2 3; do
  stop_x150; say "x150 stopped for the ENS seam (root now: '$(x150_root)')"
  launch
  w=0; while [ -f $EM/chain.FAILED ] && [ $w -lt 30 ]; do sleep 10; w=$((w+1)); done   # the chain sets the OLD chain.FAILED aside first
  [ -f $EM/chain.FAILED ] && { say "stop: the old chain.FAILED was not set aside within 5 min"; start_x150; exit 1; }
  resumed=0
  until [ -f $EM/chain.DONE ] || [ -f $EM/chain.FAILED ]; do
    printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$(sample)" >> $TR
    if [ $resumed = 0 ] && [ -f $EM/ENS1.seam.DONE ]; then say "ENS1 seam DONE ($(cat $EM/ENS1.seam.DONE)); relaunching x150 beside the CPU ctrl/score"; start_x150; resumed=1; fi
    sleep 60
  done
  [ $resumed = 0 ] && [ -z "$(x150_root)" ] && [ ! -f $X/markers/chain.DONE ] && start_x150
  [ -f $EM/chain.DONE ] && { say "ENS chain DONE ($(ls $EM | tr '\n' ' '))"; exit 0; }
  f=$(cat $EM/chain.FAILED); say "ENS chain FAILED again: $f"
  minc=$(awk -F'\t' -v t="$(date -u -d "-12 hours" +%FT%TZ)" 'NR > 1 && $3 > 0 && $1 > t { if (m == "" || $2 < m) m = $2 } END { print m + 0 }' $TR)
  say "min commit free while the ENS seam was alive: ${minc} MB"
  if [ $attempt = 2 ] && echo "$f" | grep -q "stage=seam_ENS1" && [ "${minc:-0}" -ge 1500 ]; then say "commit never exhausted: relaunching once more"; sleep 120; continue; fi
  say "stop: not retried (attempt $attempt; min commit ${minc} MB)"; exit 1
done
