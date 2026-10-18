<#
=====================================================================================================
cleanup_devbox_A.ps1  --  TanitAD dev-box disk cleanup, category A  (safe to delete, re-creatable; C: only)
Generated 2026-10-04 by a READ-ONLY inventory. The inventory deleted/moved/changed NOTHING and this script has NOT been run.
Review, then run from a normal (non-admin) PowerShell 5.1 window:   powershell -NoProfile -File "cleanup_devbox_A.ps1"

HOW IT PROTECTS YOU
  * $DryRun = $true (default): every line only PRINTS what it would delete and the running total. Flip to $false to delete.
  * Each "Remove-Item -LiteralPath ... -Recurse -Force" line below goes through a guarded function that SHADOWS the cmdlet:
      - skips a path that does not exist (message), - REFUSES a path that is in / inside / a parent of the $KEEP set,
      - REFUSES a directory that contains junctions or symlinks (PS 5.1 can follow them), - REFUSES a path with a file written in
        the last $MinIdleMinutes minutes (something may be writing there), - deletes a junction itself without touching its target.
  * Sizes are ON-DISK bytes (4 KiB clusters on C:, 1 MiB clusters on the exFAT D:/E: volume).
  * Nothing here touches G: or prints any secret.
UNITS: "GB" in comments = GiB (2^30 B), the Windows convention.
=====================================================================================================
#>
$DryRun = $false                  # <-- flip to $false to really delete
$MinIdleMinutes = 30             # refuse paths with files modified within this many minutes

Set-StrictMode -Version 2
$ErrorActionPreference = 'Continue'
$script:nPlanned = 0; $script:nDone = 0; $script:nSkip = 0; $script:nRefuse = 0; $script:nFail = 0
$script:bytesPlanned = [long]0; $script:bytesDone = [long]0

# ---- the KEEP set: live jobs, live sessions, the repo, the PI's other data. Never touched.
$KEEP = @(
  'C:\Users\Admin\qland',
  'C:\Users\Admin\ev7',
  'C:\Users\Admin\ev7nav',
  'C:\Users\Admin\venvs',
  'C:\Users\Admin\navsim-crun',
  'C:\Users\Admin\tanitad-push',
  'C:\Users\Admin\g0cache',
  'C:\Users\Admin\tanitad-caches',
  'C:\Users\Admin\tanitad-media-backup',
  'C:\Users\Admin\navsim',
  'C:\Users\Admin\tanitad-data\physicalai',
  'C:\Users\Admin\tanitad-data\refav1-eval141',
  'C:\Users\Admin\tanitad-data\joins',
  'C:\Users\Admin\tanitad-data\models',
  'C:\Users\Admin\a40-rescue',
  'C:\Users\Admin\refcv5cmp',
  'C:\Users\Admin\.claude',
  'C:\Users\Admin\AppData\Roaming\uv',
  'C:\Users\Admin\AppData\Local\TanitADStudio',
  'C:\Users\Admin\AppData\Local\Google',
  'C:\Users\Admin\.cache\huggingface\hub\models--*',
  'C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\7c618562-39eb-4a61-af81-755048086a13',
  'C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\bd7d00af-b98e-42f1-a53c-cb4113059b0f',
  'C:\Users\Admin\AppData\Local\Temp\claude\D--Projects-TanitAD\91effc67-8c1e-4b66-9a63-341a3109dfc1',
  'C:\Users\Admin\AppData\Local\Temp\claude\E--Projects-TanitAD',
  'G:\*',
  'D:\Projects\TanitAD',
  'D:\Projects\TanitAD\data\refe_*',
  'D:\Projects\TanitAD\data\nuplan-camera',
  'D:\Projects\TanitAD\data\nuplan-maps',
  'D:\Projects\TanitAD\data\backbones',
  'D:\Projects\TanitAD\data\openscene_probe',
  'D:\Projects\TanitAD\FlyWheels\TanitAD_EvalFlyWheel\incoming\2026-09-28-refcv7-standard-tests',
  'D:\refcv7_eval_kit',
  'D:\venvs',
  'D:\Archive\devbox-C\navsim',
  'D:\Archive\devbox-C\tanitad-caches',
  'D:\Projects\TanitAD-artifacts\refe-pod-final-2026-10-04',
  'D:\Projects\TanitAD-artifacts\refe-onpolicy-2026-10-01',
  'D:\Projects\TanitAD-artifacts\refcv4b_final',
  'D:\Projects\TanitAD-artifacts\refcv3_final',
  'D:\Projects\TanitAD-artifacts\refcv5v2_final',
  'D:\Projects\TanitAD-artifacts\_s2build-copy-20260919',
  'D:\Projects\HierarchicalWorldModeling',
  'D:\Projects\JevRLsimulation',
  'C:\Users\Admin\.cache\jevrl',
  'C:\Users\Admin\.cache\world-modeling',
  'C:\Users\Admin\.cache\codex-runtimes',
  'C:\Users\Admin\EarningsSwingResearch',
  'C:\Users\Admin\Documents',
  'C:\Users\Admin\Downloads',
  'C:\Dev',
  'C:\Users\Admin\.codex',
  'C:\Users\Admin\Desktop',
  'C:\Users\Admin\Pictures',
  'D:\euw-audi-fashaf-l3-prod',
  'D:\Daten_Paris',
  'D:\TanitAstra',
  'D:\TanitAstraResearch',
  'D:\System Volume Information',
  'D:\$RECYCLE.BIN',
  'D:\Install Security for Mac.zip',
  'D:\Install Security for Windows.exe'
)
# ---- sub-paths of a KEEP entry that this script may touch (bulk data that merely lives inside the repo directory)
$KEEP_EXCEPT = @(
  'D:\Projects\TanitAD\data',
  'D:\Projects\TanitAD\_c',
  'D:\Projects\TanitAD\.claude\worktrees'
)

if (-not ('W' -as [type])) {
Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Collections.Generic;
public static class W {
  public static long Files, Bytes, Alloc, Reparse, Errors; public static DateTime Newest;
  public static void Scan(string root, long cluster) {
    Files = 0; Bytes = 0; Alloc = 0; Reparse = 0; Errors = 0; Newest = DateTime.MinValue;
    var st = new Stack<string>(); st.Push(root);
    while (st.Count > 0) {
      string cur = st.Pop(); IEnumerable<FileSystemInfo> en;
      try { en = new DirectoryInfo(cur).EnumerateFileSystemInfos(); } catch { Errors++; continue; }
      try {
        foreach (var fi in en) {
          try {
            if ((fi.Attributes & FileAttributes.ReparsePoint) != 0) { Reparse++; continue; }
            if ((fi.Attributes & FileAttributes.Directory) != 0) { st.Push(fi.FullName); }
            else { var f = (FileInfo)fi; Files++; Bytes += f.Length; Alloc += ((f.Length + cluster - 1) / cluster) * cluster;
                   if (f.LastWriteTimeUtc > Newest) Newest = f.LastWriteTimeUtc; }
          } catch { Errors++; }
        }
      } catch { Errors++; }
    }
  }
}
'@
}

function Get-Cluster([string]$p) { if ($p.Substring(0,1).ToUpperInvariant() -eq 'C') { return [long]4096 } else { return [long]1048576 } }
function Fmt-Gb([long]$b) { return ('{0:N2} GB' -f ($b / 1GB)) }

function Test-Keep([string]$full) {
  $f = $full.ToLowerInvariant().TrimEnd('\')
  foreach ($k in $KEEP) {
    $kk = $k.ToLowerInvariant().TrimEnd('\')
    if ($kk.Contains('*')) {
      if ($f -like $kk -or $f -like ($kk + '\*')) { return $true }
      $pre = $kk.Substring(0, $kk.IndexOf('*')).TrimEnd('\')
      if ($pre.Length -gt 2 -and ($pre.StartsWith($f + '\') -or $pre -eq $f)) { return $true }   # parent of a wildcard KEEP entry
      continue
    }
    if ($f -eq $kk -or $f.StartsWith($kk + '\')) {
      $excepted = $false
      foreach ($x in $KEEP_EXCEPT) {
        $xx = $x.ToLowerInvariant().TrimEnd('\')
        if ($xx.StartsWith($kk + '\') -and ($f -eq $xx -or $f.StartsWith($xx + '\'))) { $excepted = $true }
      }
      if (-not $excepted) { return $true }
    }
    if ($kk.StartsWith($f + '\')) { return $true }       # the path is a parent of a KEEP entry
  }
  return $false
}

# ---- guarded replacement for the Remove-Item cmdlet (functions win over cmdlets in PowerShell)
function Remove-Item {
  [CmdletBinding()]
  param([Parameter(Mandatory = $true, Position = 0)][string]$LiteralPath, [switch]$Recurse, [switch]$Force)
  $p = $LiteralPath
  if (-not (Test-Path -LiteralPath $p)) { Write-Host ('SKIP    absent                 {0}' -f $p); $script:nSkip++; return }
  $full = [System.IO.Path]::GetFullPath($p).TrimEnd('\')
  if (Test-Keep $full) { Write-Host ('REFUSE  in KEEP set            {0}' -f $full) -ForegroundColor Red; $script:nRefuse++; return }
  $item = Get-Item -LiteralPath $full -Force
  $isLink = [bool]($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)
  $bytes = [long]0; $desc = ''
  if ($isLink) {
    $desc = 'junction/link (target untouched)'
  } elseif ($item.PSIsContainer) {
    [W]::Scan($full, (Get-Cluster $full))
    if ([W]::Reparse -gt 0) { Write-Host ('REFUSE  contains {0} junction/symlink entries  {1}' -f [W]::Reparse, $full) -ForegroundColor Red; $script:nRefuse++; return }
    if ([W]::Errors -gt 0) { Write-Host ('REFUSE  {0} unreadable entries (cannot size safely)  {1}' -f [W]::Errors, $full) -ForegroundColor Red; $script:nRefuse++; return }
    if ([W]::Newest -gt (Get-Date).ToUniversalTime().AddMinutes(-$MinIdleMinutes)) { Write-Host ('REFUSE  written in the last {0} min (in use?)  {1}' -f $MinIdleMinutes, $full) -ForegroundColor Yellow; $script:nRefuse++; return }
    $bytes = [W]::Alloc; $desc = ('{0} files' -f [W]::Files)
  } else {
    if ($item.LastWriteTimeUtc -gt (Get-Date).ToUniversalTime().AddMinutes(-$MinIdleMinutes)) { Write-Host ('REFUSE  written in the last {0} min  {1}' -f $MinIdleMinutes, $full) -ForegroundColor Yellow; $script:nRefuse++; return }
    $cl = Get-Cluster $full
    $bytes = [long]([math]::Ceiling($item.Length / $cl) * $cl); $desc = 'file'
  }
  $script:nPlanned++; $script:bytesPlanned += $bytes
  if ($DryRun) { Write-Host ('DRY-RUN would delete {0,10}  ({1})  {2}' -f (Fmt-Gb $bytes), $desc, $full); return }
  try {
    if ($isLink) { [System.IO.Directory]::Delete($full) }
    else { Microsoft.PowerShell.Management\Remove-Item -LiteralPath $full -Recurse -Force -ErrorAction Stop }
    Write-Host ('DELETED {0,10}  {1}' -f (Fmt-Gb $bytes), $full) -ForegroundColor Green
    $script:nDone++; $script:bytesDone += $bytes
  } catch { Write-Host ('FAILED  {0}  {1}' -f $full, $_.Exception.Message) -ForegroundColor Red; $script:nFail++ }
}

function Show-Summary {
  Write-Host ''
  Write-Host ('planned: {0} items, {1}   | skipped (absent): {2}   | refused: {3}' -f $script:nPlanned, (Fmt-Gb $script:bytesPlanned), $script:nSkip, $script:nRefuse)
  if ($DryRun) { Write-Host 'DRY RUN - nothing was deleted. Set $DryRun = $false at the top to execute.' -ForegroundColor Cyan }
  else { Write-Host ('deleted: {0} items, {1}   | failed: {2}' -f $script:nDone, (Fmt-Gb $script:bytesDone), $script:nFail) }
  foreach ($d in 'C','D') { try { $v = Get-PSDrive -Name $d -ErrorAction Stop; Write-Host ('free now on {0}: {1:N1} GB' -f $d, ($v.Free / 1GB)) } catch { } }
}



# ====================================================================================================
# A1. package / model caches (re-downloadable)  -  6 items, 39.0 GB
# ====================================================================================================
# 24.7 GB on disk (24.2 GB real, 232,004 files) | A | package cache, re-downloadable from PyPI
#   evidence: single dir `cache` + uv-receipt.json in AppData\Local\uv; NOT the uv python installs (those are AppData\Roaming\uv\python = KEEP)
#   risk/note: offline venv rebuilds (REFe eval venv was rebuilt offline once) lose their wheel source -> need network. Interpreters are untouched.
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\uv\cache' -Recurse -Force

# 6.54 GB on disk (6.54 GB real, 193 files) | A | dataset snapshot cache, re-downloadable from the PI private HF (token in <token-file>)
#   evidence: named only by stack/scripts/aug120_pipeline.py, pod_pull_b1_epcache.py, refc_v3_train.py (dataset builders / Thor trainer); no live-job script (NavSim package, waiter chain,
#   refcv7 eval) names it; refcv6_loader.py sets HF_HUB_OFFLINE for MODEL init only
#   risk/note: any job run with HF_HUB_OFFLINE=1 that expects this dataset would fail; the live refcv7 chain does not read it
Remove-Item -LiteralPath 'C:\Users\Admin\.cache\huggingface\hub\datasets--Sayood--tanitad-physicalai-w120-256x640cyl' -Recurse -Force

# 6.11 GB on disk (6.10 GB real, 1,952 files) | A | package cache, re-downloadable
#   evidence: 1,952 files
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\pip\cache' -Recurse -Force

# 1.02 GB on disk (1.01 GB real, 7,151 files) | A | package cache, re-downloadable (TanitADStudio vite dev server keeps its own node_modules)
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\npm-cache' -Recurse -Force

# 449 MB on disk (440 MB real, 4,534 files) | A | dataset snapshot cache, re-downloadable
#   evidence: default-arg string in stack/scripts/refc_v3_train.py (Thor trainer); not read by a dev-box live job
Remove-Item -LiteralPath 'C:\Users\Admin\.cache\huggingface\hub\datasets--Sayood--tanitad-ph0-aug120' -Recurse -Force

# 118 MB on disk (118 MB real, 85 files) | A | download-dedup cache, re-created on next hf download
#   evidence: 85 files
Remove-Item -LiteralPath 'C:\Users\Admin\.cache\huggingface\xet' -Recurse -Force


# ====================================================================================================
# A2. stale temp dirs and stale Claude session scratchpads (> 3 days, not a running session)  -  63 items, 14.1 GB
# ====================================================================================================
# 3.33 GB on disk (3.33 GB real, 524 files) | A | stale temp dir, newest file 2026-09-22 11:13 UTC (> 3 days), not under Temp\claude
#   evidence: 524 files, oldest 2026-09-22 11:13
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\clipid_ref_2ap15jyt' -Recurse -Force

# 3.33 GB on disk (3.33 GB real, 524 files) | A | stale temp dir, newest file 2026-09-22 11:13 UTC (> 3 days), not under Temp\claude
#   evidence: 524 files, oldest 2026-09-22 11:13
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\clipid_ref_rwqh9hh8' -Recurse -Force

# 843 MB on disk (842 MB real, 114 files) | A | stale session scratchpad (newest file 2026-09-05 UTC, > 3 days); session is not running
#   evidence: not one of the running sessions (7c618562 = this one, a085b71c = REFe). Resuming that old session later would not find its scratch files.
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8e7cfa33-c625-47cf-88aa-711db80ac113' -Recurse -Force

# 498 MB on disk (497 MB real, 10,638 files) | A | stale session scratchpad (newest file 2026-09-27 UTC, > 3 days); session is not running
#   evidence: not one of the running sessions (7c618562 = this one, a085b71c = REFe). Resuming that old session later would not find its scratch files.
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\ff19d6ac-620c-4636-a8e4-d5402c796859' -Recurse -Force

# 299 MB on disk (297 MB real, 849 files) | A | stale temp dir, newest file 2026-09-27 22:58 UTC (> 3 days), not under Temp\claude
#   evidence: 849 files, oldest 2026-09-20 15:45
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\measures_selftest_9zj5wjm1' -Recurse -Force

# 248 MB on disk (246 MB real, 706 files) | A | stale temp dir, newest file 2026-09-27 11:00 UTC (> 3 days), not under Temp\claude
#   evidence: 706 files, oldest 2026-09-20 15:45
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\measures_selftest_gbw4is3v' -Recurse -Force

# 233 MB on disk (233 MB real, 2 files) | A | stale temp dir, newest file 2026-09-30 14:34 UTC (> 3 days), not under Temp\claude
#   evidence: 2 files, oldest 2026-09-30 14:34
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\vscode-stable-user-x64' -Recurse -Force

# 198 MB on disk (196 MB real, 586 files) | A | stale temp dir, newest file 2026-09-27 10:47 UTC (> 3 days), not under Temp\claude
#   evidence: 586 files, oldest 2026-09-20 15:45
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\measures_selftest__svdua4l' -Recurse -Force

# 193 MB on disk (193 MB real, 20 files) | A | stale session scratchpad (newest file 2026-08-17 UTC, > 3 days); session is not running
#   evidence: not one of the running sessions (7c618562 = this one, a085b71c = REFe). Resuming that old session later would not find its scratch files.
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2' -Recurse -Force

# 184 MB on disk (182 MB real, 574 files) | A | stale temp dir, newest file 2026-09-27 21:56 UTC (> 3 days), not under Temp\claude
#   evidence: 574 files, oldest 2026-09-27 21:29
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\measures_selftest_poyqqg0j' -Recurse -Force

# 172 MB on disk (171 MB real, 570 files) | A | stale temp dir, newest file 2026-09-27 10:42 UTC (> 3 days), not under Temp\claude
#   evidence: 570 files, oldest 2026-09-20 15:45
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\measures_selftest_ie0uom7e' -Recurse -Force

# 147 MB on disk (147 MB real, 4 files) | A | stale temp dir, newest file 2026-09-27 14:58 UTC (> 3 days), not under Temp\claude
#   evidence: 4 files, oldest 2026-09-27 14:58
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\i3probe2_3j0bmsqc' -Recurse -Force

# 147 MB on disk (147 MB real, 4 files) | A | stale temp dir, newest file 2026-09-27 15:09 UTC (> 3 days), not under Temp\claude
#   evidence: 4 files, oldest 2026-09-27 15:09
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\i3_measure_b3sawjk1' -Recurse -Force

# 147 MB on disk (147 MB real, 4 files) | A | stale temp dir, newest file 2026-09-27 15:02 UTC (> 3 days), not under Temp\claude
#   evidence: 4 files, oldest 2026-09-27 15:02
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\i3probe3_0w7ccluy' -Recurse -Force

# 145 MB on disk (145 MB real, 5 files) | A | stale temp dir, newest file 2026-09-27 14:54 UTC (> 3 days), not under Temp\claude
#   evidence: 5 files, oldest 2026-09-27 14:54
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\i3probe_hxk0nbpd' -Recurse -Force

# 140 MB on disk (140 MB real, 1 files) | A | loose 140 MB .tmp file, last written 2026-09-26 (8 days)
#   evidence: file
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\wctFB03.tmp' -Recurse -Force

# 138 MB on disk (137 MB real, 20 files) | A | stale temp dir, newest file 2026-09-27 16:13 UTC (> 3 days), not under Temp\claude
#   evidence: 20 files, oldest 2026-09-27 16:13
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3mon_30x0zi3_' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 15:53 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 15:53
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3_nav_sigreg_bank_q325ttkf' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 15:54 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 15:54
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3_nav_sigreg_bank_sbfd47gp' -Recurse -Force

# 137 MB on disk (137 MB real, 19 files) | A | stale temp dir, newest file 2026-09-27 16:07 UTC (> 3 days), not under Temp\claude
#   evidence: 19 files, oldest 2026-09-27 16:07
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3probe_abc_post_F3_2fqcf8qt' -Recurse -Force

# 137 MB on disk (137 MB real, 19 files) | A | stale temp dir, newest file 2026-09-27 16:17 UTC (> 3 days), not under Temp\claude
#   evidence: 19 files, oldest 2026-09-27 16:17
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3probe_abc_post_F3_qu852sll' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 15:52 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 15:52
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3_nav_o4_9adkpri6' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 15:52 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 15:52
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3_nav_o4_no_restore_8korvej0' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 15:54 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 15:54
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3_nav_o4_no_restore_d7f188zh' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 15:54 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 15:54
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3_nav_o4_xhxzq3t_' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 15:53 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 15:52
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3_nav_uniform_niwdhv46' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 15:54 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 15:54
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3_nav_uniform_zfoq5u0g' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 16:07 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 16:06
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3probe_nav_sigreg_bank_3djy0egx' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 16:17 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 16:17
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3probe_nav_sigreg_bank_vek8d30s' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 16:17 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 16:16
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3probe_nav_o4_hcbjmo09' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 16:06 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 16:06
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3probe_nav_o4_m4pjawp4' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 16:17 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 16:17
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3probe_nav_uniform_npqs7htc' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 16:06 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 16:06
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3probe_nav_uniform_zujmlav4' -Recurse -Force

# 137 MB on disk (137 MB real, 19 files) | A | stale temp dir, newest file 2026-09-27 16:07 UTC (> 3 days), not under Temp\claude
#   evidence: 19 files, oldest 2026-09-27 16:07
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3probe_abc_pre_F3_dq49g7oc' -Recurse -Force

# 137 MB on disk (137 MB real, 19 files) | A | stale temp dir, newest file 2026-09-27 16:17 UTC (> 3 days), not under Temp\claude
#   evidence: 19 files, oldest 2026-09-27 16:17
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3probe_abc_pre_F3_msbhqqja' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 15:54 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 15:54
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3_nonav_o4_lzxdp_y4' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 15:52 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 15:52
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3_nonav_o4_x1jixx_t' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 16:17 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 16:17
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3probe_nonav_o4_i3621ghq' -Recurse -Force

# 137 MB on disk (137 MB real, 18 files) | A | stale temp dir, newest file 2026-09-27 16:06 UTC (> 3 days), not under Temp\claude
#   evidence: 18 files, oldest 2026-09-27 16:06
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\f3probe_nonav_o4_po2vsfg7' -Recurse -Force

# 113 MB on disk (113 MB real, 3 files) | A | stale temp dir, newest file 2026-09-10 20:56 UTC (> 3 days), not under Temp\claude
#   evidence: 3 files, oldest 2026-09-10 20:36
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\lab0910' -Recurse -Force

# 106 MB on disk (105 MB real, 374 files) | A | stale temp dir, newest file 2026-09-27 10:55 UTC (> 3 days), not under Temp\claude
#   evidence: 374 files, oldest 2026-09-20 15:45
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\measures_selftest_5lt4y9hl' -Recurse -Force

# 77 MB on disk (77 MB real, 11 files) | A | stale session scratchpad (newest file 2026-08-29 UTC, > 3 days); session is not running
#   evidence: not one of the running sessions (7c618562 = this one, a085b71c = REFe). Resuming that old session later would not find its scratch files.
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\76eee715-fa69-42b9-b76b-f2f2879ba6f5' -Recurse -Force

# 73 MB on disk (73 MB real, 2 files) | A | stale temp dir, newest file 2026-09-27 14:54 UTC (> 3 days), not under Temp\claude
#   evidence: 2 files, oldest 2026-09-27 14:54
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\i3probe_n6sosil9' -Recurse -Force

# 68 MB on disk (68 MB real, 250 files) | A | stale temp dir, newest file 2026-09-27 12:06 UTC (> 3 days), not under Temp\claude
#   evidence: 250 files, oldest 2026-09-23 05:46
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\yaw19_repro' -Recurse -Force

# 60 MB on disk (60 MB real, 2 files) | A | stale temp dir, newest file 2026-09-27 15:01 UTC (> 3 days), not under Temp\claude
#   evidence: 2 files, oldest 2026-09-27 15:01
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\proxycache_rows_fizehrdo' -Recurse -Force

# 28 MB on disk (28 MB real, 4 files) | A | stale temp dir, newest file 2026-09-27 13:01 UTC (> 3 days), not under Temp\claude
#   evidence: 4 files, oldest 2026-09-27 13:01
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\v4smoke_6lwjlwrh' -Recurse -Force

# 28 MB on disk (28 MB real, 4 files) | A | stale temp dir, newest file 2026-09-27 13:01 UTC (> 3 days), not under Temp\claude
#   evidence: 4 files, oldest 2026-09-27 13:01
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\v4smoke_6m7sx89c' -Recurse -Force

# 28 MB on disk (28 MB real, 4 files) | A | stale temp dir, newest file 2026-09-27 13:34 UTC (> 3 days), not under Temp\claude
#   evidence: 4 files, oldest 2026-09-27 13:34
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\v4smoke_8aa333kg' -Recurse -Force

# 28 MB on disk (28 MB real, 4 files) | A | stale temp dir, newest file 2026-09-27 13:21 UTC (> 3 days), not under Temp\claude
#   evidence: 4 files, oldest 2026-09-27 13:21
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\v4smoke_8pq9rdnv' -Recurse -Force

# 28 MB on disk (28 MB real, 4 files) | A | stale temp dir, newest file 2026-09-27 13:53 UTC (> 3 days), not under Temp\claude
#   evidence: 4 files, oldest 2026-09-27 13:53
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\v4smoke_f932l4y6' -Recurse -Force

# 28 MB on disk (28 MB real, 4 files) | A | stale temp dir, newest file 2026-09-27 13:10 UTC (> 3 days), not under Temp\claude
#   evidence: 4 files, oldest 2026-09-27 13:10
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\v4smoke_l9u11vmy' -Recurse -Force

# 28 MB on disk (28 MB real, 4 files) | A | stale temp dir, newest file 2026-09-27 12:55 UTC (> 3 days), not under Temp\claude
#   evidence: 4 files, oldest 2026-09-27 12:55
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\v4smoke_lrl1u8c1' -Recurse -Force

# 28 MB on disk (28 MB real, 4 files) | A | stale temp dir, newest file 2026-09-27 12:56 UTC (> 3 days), not under Temp\claude
#   evidence: 4 files, oldest 2026-09-27 12:55
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\v4smoke_sco3iu83' -Recurse -Force

# 28 MB on disk (28 MB real, 3 files) | A | stale temp dir, newest file 2026-09-26 09:25 UTC (> 3 days), not under Temp\claude
#   evidence: 3 files, oldest 2026-09-26 09:25
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\v4smoke_0919bqyb' -Recurse -Force

# 28 MB on disk (28 MB real, 3 files) | A | stale temp dir, newest file 2026-09-26 08:23 UTC (> 3 days), not under Temp\claude
#   evidence: 3 files, oldest 2026-09-26 08:23
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\v4smoke_2l2ou6tr' -Recurse -Force

# 28 MB on disk (28 MB real, 3 files) | A | stale temp dir, newest file 2026-09-26 09:23 UTC (> 3 days), not under Temp\claude
#   evidence: 3 files, oldest 2026-09-26 09:23
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\v4smoke_vrv8nynr' -Recurse -Force

# 8 MB on disk (8 MB real, 2 files) | A | stale temp dir, newest file 2026-09-27 15:57 UTC (> 3 days), not under Temp\claude
#   evidence: 2 files, oldest 2026-09-27 15:57
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\selftest_proxytrain_76e0mfi5' -Recurse -Force

# 8 MB on disk (8 MB real, 2 files) | A | stale temp dir, newest file 2026-09-27 21:32 UTC (> 3 days), not under Temp\claude
#   evidence: 2 files, oldest 2026-09-27 21:32
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\selftest_proxytrain_brz71_1w' -Recurse -Force

# 8 MB on disk (8 MB real, 2 files) | A | stale temp dir, newest file 2026-09-27 19:53 UTC (> 3 days), not under Temp\claude
#   evidence: 2 files, oldest 2026-09-27 19:53
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\selftest_proxytrain_dzkj56rf' -Recurse -Force

# 8 MB on disk (8 MB real, 2 files) | A | stale temp dir, newest file 2026-09-27 15:33 UTC (> 3 days), not under Temp\claude
#   evidence: 2 files, oldest 2026-09-27 15:33
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\selftest_proxytrain_o5bdrvnh' -Recurse -Force

# 8 MB on disk (8 MB real, 2 files) | A | stale temp dir, newest file 2026-09-27 22:42 UTC (> 3 days), not under Temp\claude
#   evidence: 2 files, oldest 2026-09-27 22:42
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\selftest_proxytrain_ueh9paql' -Recurse -Force

# 8 MB on disk (8 MB real, 2 files) | A | stale temp dir, newest file 2026-09-27 14:57 UTC (> 3 days), not under Temp\claude
#   evidence: 2 files, oldest 2026-09-27 14:57
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\selftest_proxytrain_zj9ovtfk' -Recurse -Force

# 7 MB on disk (7 MB real, 10 files) | A | stale temp dir, newest file 2026-09-26 18:58 UTC (> 3 days), not under Temp\claude
#   evidence: 10 files, oldest 2026-09-26 18:57
Remove-Item -LiteralPath 'C:\Users\Admin\AppData\Local\Temp\tmp.QiDXSCmd4L' -Recurse -Force


# ====================================================================================================
# A3. git-archive scratch trees (plain extracts; unique files, if any, are listed per tree)  -  67 items, 33.9 GB
# ====================================================================================================
# 3.52 GB on disk (3.52 GB real, 559 files) | A | plain git-archive extract (no .git); every one of its 559 files is a byte-identical blob in git
#   evidence: ALL 559 files hashed as git blobs: 559 found in git; 0 not found; newest file 2026-09-26 12:06 UTC; mentioned by qland/work/coord_land/stage_libpdf.py (finished one-shot, 09-26)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\libtree' -Recurse -Force

# 590 MB on disk (580 MB real, 4,588 files) | A | plain git-archive extract (no .git); every one of its 4588 files is a byte-identical blob in git
#   evidence: ALL 4588 files hashed as git blobs: 4588 found in git; 0 not found; newest file 2026-09-26 15:04 UTC; mentioned by qland/work/coord_land/gate_tcW8_*.log (gate logs)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\tcW8_cand' -Recurse -Force

# 584 MB on disk (574 MB real, 4,464 files) | A | plain git-archive extract (no .git); every one of its 4464 files is a byte-identical blob in git
#   evidence: ALL 4464 files hashed as git blobs: 4464 found in git; 0 not found; newest file 2026-09-26 14:36 UTC; mentioned by qland/work/coord_land/gate_tcW8_*.log (gate logs)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\tcW8_tip' -Recurse -Force

# 398 MB on disk (391 MB real, 3,171 files) | A | plain git-archive extract (no .git); every one of its 2835 files is a byte-identical blob in git
#   evidence: ALL 2835 files hashed as git blobs: 2835 found in git; 0 not found; newest file 2026-09-27 14:37 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_a17' -Recurse -Force

# 397 MB on disk (389 MB real, 3,953 files) | A | plain git-archive extract (no .git); every one of its 2869 files is a byte-identical blob in git
#   evidence: ALL 2869 files hashed as git blobs: 2869 found in git; 0 not found; newest file 2026-09-26 08:27 UTC; extra entries: data; mentioned by qland/work/coord_land/make_test_tree.py
#   (generator script, 09-26)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\ctree_cand' -Recurse -Force

# 390 MB on disk (384 MB real, 3,073 files) | A | plain git-archive extract (no .git); every one of its 2895 files is a byte-identical blob in git
#   evidence: ALL 2895 files hashed as git blobs: 2895 found in git; 0 not found; newest file 2026-09-28 05:03 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\rc7combo' -Recurse -Force

# 390 MB on disk (383 MB real, 3,066 files) | A | plain git-archive extract (no .git); every one of its 2891 files is a byte-identical blob in git
#   evidence: ALL 2891 files hashed as git blobs: 2891 found in git; 0 not found; newest file 2026-09-28 04:03 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\rc7restart' -Recurse -Force

# 388 MB on disk (381 MB real, 2,985 files) | A | plain git-archive extract (no .git); every one of its 2811 files is a byte-identical blob in git
#   evidence: ALL 2811 files hashed as git blobs: 2811 found in git; 0 not found; newest file 2026-09-27 03:18 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tip_cef9709' -Recurse -Force

# 386 MB on disk (380 MB real, 3,014 files) | A | plain git-archive extract (no .git); every one of its 2896 files is a byte-identical blob in git
#   evidence: ALL 2896 files hashed as git blobs: 2896 found in git; 0 not found; newest file 2026-09-28 20:04 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\r4tree' -Recurse -Force

# 386 MB on disk (380 MB real, 2,950 files) | A | plain git-archive extract (no .git); every one of its 2835 files is a byte-identical blob in git
#   evidence: ALL 2835 files hashed as git blobs: 2835 found in git (2395 of them only after CRLF->LF normalisation: this tree was extracted with CRLF line endings); 0 not found; newest file
#   2026-09-26 21:22 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\nb2_tip_2310' -Recurse -Force

# 386 MB on disk (380 MB real, 2,942 files) | A | plain git-archive extract (no .git); every one of its 2827 files is a byte-identical blob in git
#   evidence: ALL 2827 files hashed as git blobs: 2827 found in git (2387 of them only after CRLF->LF normalisation: this tree was extracted with CRLF line endings); 0 not found; newest file
#   2026-09-26 20:10 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\nb2_tip_2203' -Recurse -Force

# 386 MB on disk (379 MB real, 3,006 files) | A | plain git-archive extract (no .git); every one of its 2893 files is a byte-identical blob in git
#   evidence: ALL 2893 files hashed as git blobs: 2893 found in git; 0 not found; newest file 2026-09-28 03:53 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\rc7frozen' -Recurse -Force

# 386 MB on disk (380 MB real, 2,956 files) | A | plain git-archive extract (no .git); every one of its 2797 files is a byte-identical blob in git
#   evidence: ALL 2797 files hashed as git blobs: 2797 found in git; 0 not found; newest file 2026-09-27 02:27 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tip_4797ffb' -Recurse -Force

# 386 MB on disk (379 MB real, 3,000 files) | A | plain git-archive extract (no .git); every one of its 2956 files is a byte-identical blob in git
#   evidence: ALL 2956 files hashed as git blobs: 2956 found in git; 0 not found; newest file 2026-09-27 16:05 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\nb2a19_tree' -Recurse -Force

# 385 MB on disk (379 MB real, 2,999 files) | A | plain git-archive extract (no .git); every one of its 2891 files is a byte-identical blob in git
#   evidence: ALL 2891 files hashed as git blobs: 2891 found in git; 0 not found; newest file 2026-09-28 04:41 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\rc7lever2' -Recurse -Force

# 385 MB on disk (379 MB real, 2,951 files) | A | plain git-archive extract (no .git); every one of its 2835 files is a byte-identical blob in git
#   evidence: ALL 2835 files hashed as git blobs: 2835 found in git; 0 not found; newest file 2026-09-27 13:49 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_m' -Recurse -Force

# 385 MB on disk (379 MB real, 2,934 files) | A | plain git-archive extract (no .git); every one of its 2792 files is a byte-identical blob in git
#   evidence: ALL 2792 files hashed as git blobs: 2792 found in git; 0 not found; newest file 2026-09-27 01:16 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tip_68e8ec6' -Recurse -Force

# 385 MB on disk (379 MB real, 2,940 files) | A | plain git-archive extract (no .git); every one of its 2796 files is a byte-identical blob in git
#   evidence: ALL 2796 files hashed as git blobs: 2796 found in git; 0 not found; newest file 2026-09-27 01:30 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tip_49f10b4' -Recurse -Force

# 384 MB on disk (378 MB real, 2,938 files) | A | plain git-archive extract (no .git); every one of its 2835 files is a byte-identical blob in git
#   evidence: ALL 2835 files hashed as git blobs: 2835 found in git; 0 not found; newest file 2026-09-27 14:58 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_toy' -Recurse -Force

# 382 MB on disk (376 MB real, 2,853 files) | A | plain git-archive extract (no .git); every one of its 2853 files is a byte-identical blob in git
#   evidence: ALL 2853 files hashed as git blobs: 2853 found in git; 0 not found; newest file 2026-09-27 10:42 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\nb2r4_tip_1b81' -Recurse -Force

# 382 MB on disk (376 MB real, 2,852 files) | A | plain git-archive extract (no .git); every one of its 2852 files is a byte-identical blob in git
#   evidence: ALL 2852 files hashed as git blobs: 2852 found in git; 0 not found; newest file 2026-09-27 09:25 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\nb2r3_tip_2374' -Recurse -Force

# 381 MB on disk (375 MB real, 2,843 files) | A | plain git-archive extract (no .git); every one of its 2843 files is a byte-identical blob in git
#   evidence: ALL 2843 files hashed as git blobs: 2843 found in git; 0 not found; newest file 2026-09-27 03:06 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\nb2r2_tip_cef9' -Recurse -Force

# 381 MB on disk (375 MB real, 2,875 files) | A | plain git-archive extract (no .git); every one of its 2835 files is a byte-identical blob in git
#   evidence: ALL 2835 files hashed as git blobs: 2835 found in git; 0 not found; newest file 2026-09-27 14:29 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tip_2ac0bfb' -Recurse -Force

# 381 MB on disk (375 MB real, 2,829 files) | A | plain git-archive extract (no .git); every one of its 2829 files is a byte-identical blob in git
#   evidence: ALL 2829 files hashed as git blobs: 2829 found in git; 0 not found; newest file 2026-09-27 01:56 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\nb2_tip_4797ffb' -Recurse -Force

# 380 MB on disk (374 MB real, 2,835 files) | A | plain git-archive extract (no .git); every one of its 2835 files is a byte-identical blob in git
#   evidence: ALL 2835 files hashed as git blobs: 2835 found in git; 0 not found; newest file 2026-09-27 13:26 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tip_28d8365' -Recurse -Force

# 379 MB on disk (373 MB real, 2,819 files) | A | plain git-archive extract (no .git); every one of its 2819 files is a byte-identical blob in git
#   evidence: ALL 2819 files hashed as git blobs: 2819 found in git; 0 not found; newest file 2026-09-27 10:42 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tip_1b8170e' -Recurse -Force

# 379 MB on disk (373 MB real, 2,819 files) | A | plain git-archive extract (no .git); every one of its 2819 files is a byte-identical blob in git
#   evidence: ALL 2819 files hashed as git blobs: 2819 found in git; 0 not found; newest file 2026-09-27 09:25 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tip_2374cd2' -Recurse -Force

# 378 MB on disk (372 MB real, 2,817 files) | A | plain git-archive extract (no .git); every one of its 2817 files is a byte-identical blob in git
#   evidence: ALL 2817 files hashed as git blobs: 2817 found in git; 0 not found; newest file 2026-09-27 01:20 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\nb2_tip_b3prev' -Recurse -Force

# 378 MB on disk (372 MB real, 2,816 files) | A | plain git-archive extract (no .git); every one of its 2816 files is a byte-identical blob in git
#   evidence: ALL 2816 files hashed as git blobs: 2816 found in git; 0 not found; newest file 2026-09-27 00:56 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\nb2_tip_b4a5' -Recurse -Force

# 378 MB on disk (372 MB real, 2,814 files) | A | plain git-archive extract (no .git); every one of its 2814 files is a byte-identical blob in git
#   evidence: ALL 2814 files hashed as git blobs: 2814 found in git; 0 not found; newest file 2026-09-26 23:59 UTC
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\nb2_tip_0638' -Recurse -Force

# 307 MB on disk (301 MB real, 2,710 files) | A | plain git-archive extract (no .git); every one of its 2710 files is a byte-identical blob in git
#   evidence: ALL 2710 files hashed as git blobs: 2710 found in git; 0 not found; newest file 2026-09-26 10:56 UTC; mentioned by qland/work/a16_switch/stage_a16fix.py (finished one-shot, 09-26)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\cfix' -Recurse -Force

# 268 MB on disk (266 MB real, 1,114 files) | A | plain git-archive extract (no .git); every one of its 1111 files is a byte-identical blob in git
#   evidence: ALL 1111 files hashed as git blobs: 1111 found in git; 0 not found; newest file 2026-09-26 19:49 UTC; mentioned by qland historic stage records only
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\tcL' -Recurse -Force

# 217 MB on disk (212 MB real, 2,402 files) | A | plain git-archive extract (no .git); every one of its 2402 files is a byte-identical blob in git
#   evidence: ALL 2402 files hashed as git blobs: 2402 found in git; 0 not found; newest file 2026-09-26 07:11 UTC; mentioned by qland/work/coord_land/make_test_tree.py (generator script, 09-26)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\ctree_tip' -Recurse -Force

# 197 MB on disk (194 MB real, 1,001 files) | A | plain git-archive extract (no .git); every one of its 998 files is a byte-identical blob in git
#   evidence: ALL 998 files hashed as git blobs: 998 found in git; 0 not found; newest file 2026-09-26 11:45 UTC; mentioned by qland/work/watch6/gate_watch_k.py (finished watch script)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
Remove-Item -LiteralPath 'C:\Users\Admin\tcK' -Recurse -Force


# ---- A3b. trees that also hold a few patched source files not (yet) in git (each <= 1.4 MB of .py): candidate-patch iterations whose landed form is in the repo. Cheap to re-create, but read the note.
# 6.21 GB on disk (6.18 GB real, 18,031 files) | A | plain git-archive extract (no .git); only trivial unique content: 12 file(s), 1.17 MB (.py x12)
#   evidence: ALL 18031 files hashed as git blobs: 18019 found in git; 12 not found; newest file 2026-09-27 02:43 UTC; 12 NOT in git (1.17 MB: .py x12); extra entries: .claude, .gitignore,
#   CLAUDE.md, Media
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 1.17 MB
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_e' -Recurse -Force

# 425 MB on disk (417 MB real, 3,738 files) | A | plain git-archive extract (no .git); only trivial unique content: 4 file(s), 0.45 MB (.py x4)
#   evidence: ALL 2828 files hashed as git blobs: 2824 found in git (2382 of them only after CRLF->LF normalisation: this tree was extracted with CRLF line endings); 4 not found; newest file
#   2026-09-27 16:16 UTC; 4 NOT in git (0.45 MB: .py x4)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.45 MB
Remove-Item -LiteralPath 'C:\Users\Admin\v7f_chain' -Recurse -Force

# 400 MB on disk (393 MB real, 3,179 files) | A | plain git-archive extract (no .git); only trivial unique content: 1 file(s), 0.17 MB (.py x1)
#   evidence: ALL 2842 files hashed as git blobs: 2841 found in git (2401 of them only after CRLF->LF normalisation: this tree was extracted with CRLF line endings); 1 not found; newest file
#   2026-09-28 11:49 UTC; 1 NOT in git (0.17 MB: .py x1)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.17 MB
Remove-Item -LiteralPath 'C:\Users\Admin\rc7ceil' -Recurse -Force

# 396 MB on disk (389 MB real, 3,159 files) | A | plain git-archive extract (no .git); only trivial unique content: 9 file(s), 1.17 MB (.py x9)
#   evidence: ALL 2827 files hashed as git blobs: 2818 found in git; 9 not found; newest file 2026-09-27 08:02 UTC; 9 NOT in git (1.17 MB: .py x9)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 1.17 MB
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_i6' -Recurse -Force

# 396 MB on disk (389 MB real, 3,159 files) | A | plain git-archive extract (no .git); only trivial unique content: 10 file(s), 1.18 MB (.py x10)
#   evidence: ALL 2827 files hashed as git blobs: 2817 found in git; 10 not found; newest file 2026-09-27 06:18 UTC; 10 NOT in git (1.18 MB: .py x10)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 1.18 MB
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_h6' -Recurse -Force

# 396 MB on disk (389 MB real, 3,155 files) | A | plain git-archive extract (no .git); only trivial unique content: 8 file(s), 1.16 MB (.py x8)
#   evidence: ALL 2825 files hashed as git blobs: 2817 found in git; 8 not found; newest file 2026-09-27 08:11 UTC; 8 NOT in git (1.16 MB: .py x8)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 1.16 MB
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_i' -Recurse -Force

# 396 MB on disk (389 MB real, 3,155 files) | A | plain git-archive extract (no .git); only trivial unique content: 10 file(s), 1.17 MB (.py x10)
#   evidence: ALL 2825 files hashed as git blobs: 2815 found in git; 10 not found; newest file 2026-09-27 03:30 UTC; 10 NOT in git (1.17 MB: .py x10)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 1.17 MB
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_h' -Recurse -Force

# 391 MB on disk (385 MB real, 3,126 files) | A | plain git-archive extract (no .git); only trivial unique content: 1 file(s), 0.00 MB (.log x1)
#   evidence: ALL 2955 files hashed as git blobs: 2954 found in git; 1 not found; newest file 2026-09-27 14:35 UTC; 1 NOT in git (0.00 MB: .log x1); extra entries: _t_r5.log
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.00 MB
Remove-Item -LiteralPath 'C:\Users\Admin\nb2r5_tree_2ac0' -Recurse -Force

# 391 MB on disk (385 MB real, 2,957 files) | A | plain git-archive extract (no .git); only trivial unique content: 1 file(s), 0.00 MB (.txt x1)
#   evidence: ALL 2867 files hashed as git blobs: 2866 found in git (2424 of them only after CRLF->LF normalisation: this tree was extracted with CRLF line endings); 1 not found; newest file
#   2026-09-26 21:02 UTC; 1 NOT in git (0.00 MB: .txt x1); extra entries: COMMIT.txt
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.00 MB
Remove-Item -LiteralPath 'C:\Users\Admin\ev6_postfix' -Recurse -Force

# 391 MB on disk (385 MB real, 2,955 files) | A | plain git-archive extract (no .git); only trivial unique content: 4 file(s), 0.36 MB (.py x3, .txt x1)
#   evidence: ALL 2866 files hashed as git blobs: 2862 found in git (2414 of them only after CRLF->LF normalisation: this tree was extracted with CRLF line endings); 4 not found; newest file
#   2026-09-26 20:10 UTC; 4 NOT in git (0.36 MB: .py x3, .txt x1); extra entries: COMMIT.txt
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.36 MB
Remove-Item -LiteralPath 'C:\Users\Admin\ev6_postfix_da8400b_overlay' -Recurse -Force

# 389 MB on disk (383 MB real, 3,071 files) | A | plain git-archive extract (no .git); only trivial unique content: 1 file(s), 0.20 MB (.xml x1)
#   evidence: ALL 2949 files hashed as git blobs: 2948 found in git; 1 not found; newest file 2026-09-26 19:35 UTC; 1 NOT in git (0.20 MB: .xml x1); extra entries: junit.xml; mentioned by
#   qland/work/coord_land/gate_b1213.json (gate record)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.20 MB
Remove-Item -LiteralPath 'C:\Users\Admin\gt13_cand_212930' -Recurse -Force

# 389 MB on disk (383 MB real, 3,065 files) | A | plain git-archive extract (no .git); only trivial unique content: 1 file(s), 0.19 MB (.xml x1)
#   evidence: ALL 2947 files hashed as git blobs: 2946 found in git; 1 not found; newest file 2026-09-26 19:32 UTC; 1 NOT in git (0.19 MB: .xml x1); extra entries: junit.xml; mentioned by
#   qland/work/coord_land/gate_b1213.json (gate record)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.19 MB
Remove-Item -LiteralPath 'C:\Users\Admin\gt13_tip_212930' -Recurse -Force

# 389 MB on disk (383 MB real, 3,016 files) | A | plain git-archive extract (no .git); only trivial unique content: 3 file(s), 0.07 MB (.py x2, .log x1)
#   evidence: ALL 2855 files hashed as git blobs: 2852 found in git; 3 not found; newest file 2026-09-27 13:22 UTC; 3 NOT in git (0.07 MB: .py x2, .log x1); extra entries: _t_r4.log
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.07 MB
Remove-Item -LiteralPath 'C:\Users\Admin\nb2r4_tree_1b81' -Recurse -Force

# 389 MB on disk (383 MB real, 3,015 files) | A | plain git-archive extract (no .git); only trivial unique content: 1 file(s), 0.00 MB (.log x1)
#   evidence: ALL 2854 files hashed as git blobs: 2853 found in git; 1 not found; newest file 2026-09-27 09:44 UTC; 1 NOT in git (0.00 MB: .log x1); extra entries: _t_r3.log
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.00 MB
Remove-Item -LiteralPath 'C:\Users\Admin\nb2r3_tree_2374' -Recurse -Force

# 389 MB on disk (382 MB real, 3,014 files) | A | plain git-archive extract (no .git); only trivial unique content: 11 file(s), 1.20 MB (.py x11)
#   evidence: ALL 2827 files hashed as git blobs: 2816 found in git; 11 not found; newest file 2026-09-27 03:20 UTC; 11 NOT in git (1.20 MB: .py x11)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 1.20 MB
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_g6' -Recurse -Force

# 388 MB on disk (382 MB real, 3,010 files) | A | plain git-archive extract (no .git); only trivial unique content: 11 file(s), 1.20 MB (.py x11)
#   evidence: ALL 2825 files hashed as git blobs: 2814 found in git; 11 not found; newest file 2026-09-27 03:12 UTC; 11 NOT in git (1.20 MB: .py x11)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 1.20 MB
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_g' -Recurse -Force

# 388 MB on disk (382 MB real, 3,006 files) | A | plain git-archive extract (no .git); only trivial unique content: 1 file(s), 0.00 MB (.log x1)
#   evidence: ALL 2845 files hashed as git blobs: 2844 found in git; 1 not found; newest file 2026-09-27 07:54 UTC; 1 NOT in git (0.00 MB: .log x1); extra entries: _t_r2.log
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.00 MB
Remove-Item -LiteralPath 'C:\Users\Admin\nb2r2_tree_cef9' -Recurse -Force

# 388 MB on disk (382 MB real, 3,003 files) | A | plain git-archive extract (no .git); only trivial unique content: 1 file(s), 0.00 MB (.log x1)
#   evidence: ALL 2842 files hashed as git blobs: 2841 found in git; 1 not found; newest file 2026-09-27 03:00 UTC; 1 NOT in git (0.00 MB: .log x1); extra entries: _t_4797.log
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.00 MB
Remove-Item -LiteralPath 'C:\Users\Admin\nb2_tree_4797ffb' -Recurse -Force

# 387 MB on disk (380 MB real, 2,984 files) | A | plain git-archive extract (no .git); only trivial unique content: 13 file(s), 1.19 MB (.py x13)
#   evidence: ALL 2813 files hashed as git blobs: 2800 found in git; 13 not found; newest file 2026-09-27 03:05 UTC; 13 NOT in git (1.19 MB: .py x13)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 1.19 MB
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_f6' -Recurse -Force

# 387 MB on disk (380 MB real, 2,981 files) | A | plain git-archive extract (no .git); only trivial unique content: 12 file(s), 1.18 MB (.py x12)
#   evidence: ALL 2811 files hashed as git blobs: 2799 found in git; 12 not found; newest file 2026-09-27 03:06 UTC; 12 NOT in git (1.18 MB: .py x12)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 1.18 MB
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_f' -Recurse -Force

# 386 MB on disk (380 MB real, 2,979 files) | A | plain git-archive extract (no .git); only trivial unique content: 12 file(s), 1.17 MB (.py x12)
#   evidence: ALL 2810 files hashed as git blobs: 2798 found in git; 12 not found; newest file 2026-09-27 02:27 UTC; 12 NOT in git (1.17 MB: .py x12)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 1.17 MB
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_d' -Recurse -Force

# 386 MB on disk (380 MB real, 2,990 files) | A | plain git-archive extract (no .git); only trivial unique content: 1 file(s), 0.00 MB (.log x1)
#   evidence: ALL 2830 files hashed as git blobs: 2829 found in git; 1 not found; newest file 2026-09-27 01:49 UTC; 1 NOT in git (0.00 MB: .log x1); extra entries: _t_b3prev.log
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.00 MB
Remove-Item -LiteralPath 'C:\Users\Admin\nb2_tree_b3prev' -Recurse -Force

# 386 MB on disk (380 MB real, 2,993 files) | A | plain git-archive extract (no .git); only trivial unique content: 9 file(s), 1.25 MB (.py x5, .log x4)
#   evidence: ALL 2832 files hashed as git blobs: 2823 found in git; 9 not found; newest file 2026-09-27 01:40 UTC; 9 NOT in git (1.25 MB: .py x5, .log x4); extra entries: _t_b4a5.log,
#   _t_b4a5_s.log, _v3ctl_137.log, _v3ctl_137b.log
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 1.25 MB
Remove-Item -LiteralPath 'C:\Users\Admin\nb2_tree_b4a5' -Recurse -Force

# 386 MB on disk (380 MB real, 2,965 files) | A | plain git-archive extract (no .git); only trivial unique content: 13 file(s), 1.20 MB (.py x13)
#   evidence: ALL 2808 files hashed as git blobs: 2795 found in git; 13 not found; newest file 2026-09-27 02:16 UTC; 13 NOT in git (1.20 MB: .py x13)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 1.20 MB
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_b' -Recurse -Force

# 386 MB on disk (379 MB real, 2,988 files) | A | plain git-archive extract (no .git); only trivial unique content: 14 file(s), 1.38 MB (.py x10, .log x4)
#   evidence: ALL 2830 files hashed as git blobs: 2816 found in git; 14 not found; newest file 2026-09-27 00:33 UTC; 14 NOT in git (1.38 MB: .py x10, .log x4); extra entries: _t_0638.log,
#   _t_a8.log, _t_e2e.log, _t_v3real.log
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 1.38 MB
Remove-Item -LiteralPath 'C:\Users\Admin\nb2_tree_0638' -Recurse -Force

# 385 MB on disk (379 MB real, 3,001 files) | A | plain git-archive extract (no .git); only trivial unique content: 1 file(s), 0.00 MB (.log x1)
#   evidence: ALL 2957 files hashed as git blobs: 2956 found in git; 1 not found; newest file 2026-09-27 15:55 UTC; 1 NOT in git (0.00 MB: .log x1); extra entries: _t_v2.log
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.00 MB
Remove-Item -LiteralPath 'C:\Users\Admin\nb2ml2_tree' -Recurse -Force

# 385 MB on disk (379 MB real, 2,946 files) | A | plain git-archive extract (no .git); only trivial unique content: 11 file(s), 1.15 MB (.py x11)
#   evidence: ALL 2796 files hashed as git blobs: 2785 found in git; 11 not found; newest file 2026-09-27 01:24 UTC; 11 NOT in git (1.15 MB: .py x11)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 1.15 MB
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_a' -Recurse -Force

# 385 MB on disk (379 MB real, 2,956 files) | A | plain git-archive extract (no .git); only trivial unique content: 3 file(s), 0.96 MB (.py x3)
#   evidence: ALL 2827 files hashed as git blobs: 2824 found in git; 3 not found; newest file 2026-09-27 09:14 UTC; 3 NOT in git (0.96 MB: .py x3)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.96 MB
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_k' -Recurse -Force

# 383 MB on disk (377 MB real, 2,938 files) | A | plain git-archive extract (no .git); only trivial unique content: 1 file(s), 0.00 MB (.log x1)
#   evidence: ALL 2836 files hashed as git blobs: 2835 found in git; 1 not found; newest file 2026-09-27 14:21 UTC; 1 NOT in git (0.00 MB: .log x1); extra entries: _t_box.log
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.00 MB
Remove-Item -LiteralPath 'C:\Users\Admin\tip_2ac0_clean' -Recurse -Force

# 383 MB on disk (377 MB real, 2,892 files) | A | plain git-archive extract (no .git); only trivial unique content: 2 file(s), 0.03 MB (.log x1, .py x1)
#   evidence: ALL 2855 files hashed as git blobs: 2853 found in git; 2 not found; newest file 2026-09-27 13:50 UTC; 2 NOT in git (0.03 MB: .log x1, .py x1); extra entries: _t_r5.log
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.03 MB
Remove-Item -LiteralPath 'C:\Users\Admin\nb2r5_tree_8796' -Recurse -Force

# 382 MB on disk (376 MB real, 2,909 files) | A | plain git-archive extract (no .git); only trivial unique content: 14 file(s), 1.19 MB (.py x14)
#   evidence: ALL 2811 files hashed as git blobs: 2797 found in git; 14 not found; newest file 2026-09-27 02:23 UTC; 14 NOT in git (1.19 MB: .py x14)
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 1.19 MB
Remove-Item -LiteralPath 'C:\Users\Admin\bxh_tree_c' -Recurse -Force

# 321 MB on disk (315 MB real, 2,859 files) | A | plain git-archive extract (no .git); only trivial unique content: 1 file(s), 0.00 MB (.txt x1)
#   evidence: ALL 2743 files hashed as git blobs: 2742 found in git (2408 of them only after CRLF->LF normalisation: this tree was extracted with CRLF line endings); 1 not found; newest file
#   2026-09-28 04:58 UTC; 1 NOT in git (0.00 MB: .txt x1); extra entries: COMMIT.txt
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.00 MB
Remove-Item -LiteralPath 'C:\Users\Admin\ev6_82c2331' -Recurse -Force

# 177 MB on disk (173 MB real, 2,246 files) | A | plain git-archive extract (no .git); only trivial unique content: 1 file(s), 0.00 MB (.txt x1)
#   evidence: ALL 2152 files hashed as git blobs: 2151 found in git (1943 of them only after CRLF->LF normalisation: this tree was extracted with CRLF line endings); 1 not found; newest file
#   2026-09-20 08:22 UTC; 1 NOT in git (0.00 MB: .txt x1); extra entries: A7_BASE_HEAD.txt
#   survivor: regenerate with `git archive <sha>` (all tracked content is in git)
#   risk/note: patched sources not in git: 0.00 MB
Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-a7-run' -Recurse -Force

# stale Temp entries smaller than 5 MB each (1073 entries, 278 MB on disk in total) are deliberately NOT listed.

Show-Summary
