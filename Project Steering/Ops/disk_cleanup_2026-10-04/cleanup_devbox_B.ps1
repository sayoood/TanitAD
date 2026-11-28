<#
=====================================================================================================
cleanup_devbox_B.ps1  --  TanitAD dev-box disk cleanup, category B  (redundant copies; survivors named per line)
Generated 2026-10-04 by a READ-ONLY inventory. The inventory deleted/moved/changed NOTHING and this script has NOT been run.
Review, then run from a normal (non-admin) PowerShell 5.1 window:   powershell -NoProfile -File "cleanup_devbox_B.ps1"

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
$DryRun = $true                  # <-- flip to $false to really delete
$MinIdleMinutes = 30             # refuse paths with files modified within this many minutes
$IncludeExtractedArchives = $false   # B-derived: source zips whose full content is extracted next to them (see INVENTORY). Off by default.

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


$VerifyMd5 = $false              # $true = also md5 both copies (files <= 2,000,000,000 B) right before deleting (slow on the USB drive)

function Get-Md5([string]$f) { return (Get-FileHash -LiteralPath $f -Algorithm MD5).Hash }

# every file under $Source must exist under $Survivor with the same relative path and size (and the same md5 if $VerifyMd5)
function Test-CopyOf([string]$Source, [string]$Survivor) {
  if (-not (Test-Path -LiteralPath $Survivor)) { Write-Host ('REFUSE  survivor missing: {0}' -f $Survivor) -ForegroundColor Red; $script:nRefuse++; return $false }
  if (-not (Test-Path -LiteralPath $Source)) { return $true }          # nothing to delete; the Remove-Item line will print SKIP
  $n = 0
  if ((Get-Item -LiteralPath $Source -Force).PSIsContainer) {
    $files = Get-ChildItem -LiteralPath $Source -Recurse -File -Force
    $srcRoot = (Get-Item -LiteralPath $Source -Force).FullName.TrimEnd('\')
    foreach ($f in $files) {
      $rel = $f.FullName.Substring($srcRoot.Length).TrimStart('\')
      $s = Join-Path $Survivor $rel
      if (-not (Test-Path -LiteralPath $s -PathType Leaf)) { Write-Host ('REFUSE  survivor lacks {0}' -f $s) -ForegroundColor Red; $script:nRefuse++; return $false }
      if ((Get-Item -LiteralPath $s -Force).Length -ne $f.Length) { Write-Host ('REFUSE  size differs {0}' -f $s) -ForegroundColor Red; $script:nRefuse++; return $false }
      if ($VerifyMd5 -and $f.Length -le 2000000000) { if ((Get-Md5 $f.FullName) -ne (Get-Md5 $s)) { Write-Host ('REFUSE  md5 differs {0}' -f $s) -ForegroundColor Red; $script:nRefuse++; return $false } }
      $n++
    }
  } else {
    $f = Get-Item -LiteralPath $Source -Force
    if ((Get-Item -LiteralPath $Survivor -Force).Length -ne $f.Length) { Write-Host ('REFUSE  size differs {0}' -f $Survivor) -ForegroundColor Red; $script:nRefuse++; return $false }
    if ($VerifyMd5 -and $f.Length -le 2000000000) { if ((Get-Md5 $f.FullName) -ne (Get-Md5 $Survivor)) { Write-Host ('REFUSE  md5 differs {0}' -f $Survivor) -ForegroundColor Red; $script:nRefuse++; return $false } }
    $n = 1
  }
  Write-Host ('SURVIVOR OK ({0} file(s) present with equal size{1}): {2}' -f $n, $(if ($VerifyMd5) { ' + md5' } else { '' }), $Survivor)
  return $true
}

# a zip is redundant only if every member is present, with the same name and size, as an extracted file
function Test-ZipExtracted([string]$Zip, [string[]]$ExtractedRoots, [switch]$BarToUnderscore) {
  if (-not (Test-Path -LiteralPath $Zip)) { return $true }
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $idx = New-Object 'System.Collections.Generic.HashSet[string]'
  foreach ($root in $ExtractedRoots) {
    if (-not (Test-Path -LiteralPath $root)) { Write-Host ('REFUSE  extracted root missing: {0}' -f $root) -ForegroundColor Red; $script:nRefuse++; return $false }
    $rootFull = (Get-Item -LiteralPath $root -Force).FullName.TrimEnd('\')
    foreach ($f in [System.IO.Directory]::EnumerateFiles($rootFull, '*', [System.IO.SearchOption]::AllDirectories)) {
      $fi = New-Object System.IO.FileInfo $f
      if ($BarToUnderscore) { $rel = $f.Substring($rootFull.Length).TrimStart('\').Replace('\','/').ToLowerInvariant(); [void]$idx.Add($rel + '|' + $fi.Length) }
      else { [void]$idx.Add($fi.Name.ToLowerInvariant() + '|' + $fi.Length) }
    }
  }
  $z = [System.IO.Compression.ZipFile]::OpenRead($Zip)
  $miss = 0; $tot = 0
  try {
    foreach ($e in $z.Entries) {
      if ($e.FullName.EndsWith('/')) { continue }
      $tot++
      if ($BarToUnderscore) { $nm = $e.FullName.Replace('|','_').ToLowerInvariant(); $k = $nm + '|' + $e.Length; $k2 = $nm.Substring($nm.IndexOf('/') + 1) + '|' + $e.Length; $ok = ($idx.Contains($k) -or $idx.Contains($k2)) }
      else { $ok = $idx.Contains([System.IO.Path]::GetFileName($e.FullName).ToLowerInvariant() + '|' + $e.Length) }
      if (-not $ok -and [System.IO.Path]::GetFileName($e.FullName) -ne 'LICENSE') { $miss++ }
    }
  } finally { $z.Dispose() }
  # 2026-10-04 (Master Mind): a zip that yields ZERO members verified NOTHING -- 'all 0 present' is a vacuous pass
  # (it let comma2k19 Chunk_1.zip through; the inventory had verified it offline, 7,333/7,333). Refuse it.
  if ($tot -eq 0) { Write-Host ('REFUSE  0 zip members could be read -- nothing verified: {0}' -f $Zip) -ForegroundColor Red; $script:nRefuse++; return $false }
  if ($miss -gt 0) { Write-Host ('REFUSE  {0} of {1} zip members are NOT extracted: {2}' -f $miss, $tot, $Zip) -ForegroundColor Red; $script:nRefuse++; return $false }
  Write-Host ('EXTRACTED OK (all {0} members present by name+size): {1}' -f $tot, $Zip)
  return $true
}


# ---- D: is a subst of the USB drive E: (per logon session). If it vanished, stop instead of deleting through a wrong mapping.
if (-not (Test-Path -LiteralPath 'D:\Projects\TanitAD\.git')) { throw 'D:\Projects\TanitAD\.git not found - D: is not mapped. Run  subst D: E:\  first, then re-run.' }


# ====================================================================================================
# B1. identical copies (md5-verified by the inventory; the script re-checks that the survivor exists with equal sizes before each delete)
# ====================================================================================================
# 3.68 GB on disk (3.68 GB real, 7 files) | B-copy | identical copy exists on D: (7 files)
#   evidence: md5 of both copies equal for 6 of 7 files (1.88 GB read-verified); the 1 file > 2 GB (onpolicy_sets_logs_full.tar.zst, 2,070,541,308 B) has equal size and the same recorded md5
#   4fc97993... in both MD5SUMS.txt (identical manifests) - NOT read-hashed
#   survivor: D:\Projects\TanitAD-artifacts\refe-pod-final-2026-10-04 (the REFe session is uploading it to HF right now: a third copy)
#   risk/note: the C: copy was written 2026-10-04 01:19 local (apparently a backup copy made around the HF upload); deleting it leaves one dev-box copy + the HF upload in progress. Wait until
#   the REFe upload finishes if you want 2 independent copies.
if (Test-CopyOf 'C:\Users\Admin\tanitad-backup\refe-pod-final-2026-10-04' 'D:\Projects\TanitAD-artifacts\refe-pod-final-2026-10-04') { Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-backup\refe-pod-final-2026-10-04' -Recurse -Force }

# 1.62 GB on disk (1.60 GB real, 25 files) | B-copy | identical copy on the same volume
#   evidence: md5 equal for 25 of 25 files (1.72 GB read-verified on both sides), same relative paths and sizes
#   survivor: D:\Projects\TanitAD-artifacts\refcv4b_final
#   risk/note: same physical disk as the survivor (no failure-domain gain). C:\Users\Admin\refcv4b_final is a junction into the deleted dir; the script removes the link too.
if (Test-CopyOf 'D:\Archive\devbox-C\refcv4b_final' 'D:\Projects\TanitAD-artifacts\refcv4b_final') {
  Remove-Item -LiteralPath 'C:\Users\Admin\refcv4b_final' -Force            # the junction first (link only; its target is the dir below)
  Remove-Item -LiteralPath 'D:\Archive\devbox-C\refcv4b_final' -Recurse -Force
}

# 1.60 GB on disk (1.60 GB real, 7 files) | B-copy | identical copy on the same volume
#   evidence: md5 equal for 7 of 7 files (1.71 GB read-verified on both sides)
#   survivor: D:\Projects\TanitAD-artifacts\refcv3_final
#   risk/note: same physical disk as the survivor. Junction C:\Users\Admin\refcv3_final is removed too.
if (Test-CopyOf 'D:\Archive\devbox-C\refcv3_final' 'D:\Projects\TanitAD-artifacts\refcv3_final') {
  Remove-Item -LiteralPath 'C:\Users\Admin\refcv3_final' -Force            # the junction first (link only; its target is the dir below)
  Remove-Item -LiteralPath 'D:\Archive\devbox-C\refcv3_final' -Recurse -Force
}

# 1.10 GB on disk (1.10 GB real, 1 files) | B-copy | identical file exists at D:\refcv6_eval_kit\ckpt_final\ckpt_step38000.pt
#   evidence: md5 of both = 5a2e7222a9f5f8c7aa7bf38ef4698d8a = the recorded md5 in ckpt\MD5SUMS and STOPPED_BY_PI.json (1,177,897,033 B each); read-hashed 2026-10-04
#   survivor: D:\refcv6_eval_kit\ckpt_final\ckpt_step38000.pt (+ HF private Sayood/tanitad-refcv6-r101-s0, upload "verified in a follow-up" per MODEL_REGISTRY line 5273 - NOT re-checked by me)
#   risk/note: same disk. The MD5SUMS manifest still lists this file name afterwards (a stale manifest line, harmless).
if (Test-CopyOf 'D:\refcv6_eval_kit\ckpt\ckpt_step38000_stopped.pt' 'D:\refcv6_eval_kit\ckpt_final\ckpt_step38000.pt') { Remove-Item -LiteralPath 'D:\refcv6_eval_kit\ckpt\ckpt_step38000_stopped.pt' -Recurse -Force }

# ====================================================================================================
# B2. DERIVED copies: a zip whose every member is extracted next to it. OFF by default ($IncludeExtractedArchives). Re-download cost applies.
# ====================================================================================================
if ($IncludeExtractedArchives) {
  # 90.3 GB on disk (90.3 GB real, 1 files) | B-derived | derived copy: every member is extracted next to it (CRC-checked at extraction)
  #   evidence: zip central directory read (a few MB, no content read): 1381 of 1381 .db members have an extracted file with identical name+size in nuplan-v1.1/ (test/trainval; the mini members
  #   in nuplan-v1.1/splits and dblinks/) (LICENSE is at nuplan/LICENSE); the extract logs (extract_test.log / val_extract.json) record "extracting ... with per-member CRC". Whole-zip md5 NOT
  #   computed (> 2 GB).
  #   survivor: D:\Projects\TanitAD\data\nuplan\nuplan-v1.1\ (extracted) - and the zips are re-downloadable (public nuPlan S3 objects: INHERITED from memory note nuplan-s3-objects-are-public;
  #   pull_splits.log shows test/val were curl-downloaded from it)
  #   risk/note: frees the most space on D: but the extraction is on the SAME disk; if the dbs are ever damaged the 201 GB of zips must be re-downloaded: ~35 h at the dev-box rate of 1.6 MB/s
  #   (INHERITED rate, memory refe-training-data-on-a40-pod).
  if (Test-ZipExtracted 'D:\Projects\TanitAD\data\nuplan\nuplan-v1.1_val.zip' @('D:\Projects\TanitAD\data\nuplan\nuplan-v1.1', 'D:\Projects\TanitAD\data\nuplan\dblinks')) { Remove-Item -LiteralPath 'D:\Projects\TanitAD\data\nuplan\nuplan-v1.1_val.zip' -Force }

  # 89.3 GB on disk (89.3 GB real, 1 files) | B-derived | derived copy: every member is extracted next to it (CRC-checked at extraction)
  #   evidence: zip central directory read (a few MB, no content read): 1349 of 1349 .db members have an extracted file with identical name+size in nuplan-v1.1/ (test/trainval; the mini members
  #   in nuplan-v1.1/splits and dblinks/) (LICENSE is at nuplan/LICENSE); the extract logs (extract_test.log / val_extract.json) record "extracting ... with per-member CRC". Whole-zip md5 NOT
  #   computed (> 2 GB).
  #   survivor: D:\Projects\TanitAD\data\nuplan\nuplan-v1.1\ (extracted) - and the zips are re-downloadable (public nuPlan S3 objects: INHERITED from memory note nuplan-s3-objects-are-public;
  #   pull_splits.log shows test/val were curl-downloaded from it)
  #   risk/note: frees the most space on D: but the extraction is on the SAME disk; if the dbs are ever damaged the 201 GB of zips must be re-downloaded: ~35 h at the dev-box rate of 1.6 MB/s
  #   (INHERITED rate, memory refe-training-data-on-a40-pod).
  if (Test-ZipExtracted 'D:\Projects\TanitAD\data\nuplan\nuplan-v1.1_test.zip' @('D:\Projects\TanitAD\data\nuplan\nuplan-v1.1', 'D:\Projects\TanitAD\data\nuplan\dblinks')) { Remove-Item -LiteralPath 'D:\Projects\TanitAD\data\nuplan\nuplan-v1.1_test.zip' -Force }

  # 8.13 GB on disk (8.13 GB real, 1 files) | B-derived | derived copy: all members extracted next to it
  #   evidence: zip central directory vs tanitad-data/comma2k19/extracted/Chunk_1: 7,333 of 7,333 members present with identical size (| in the name is _ on disk); whole-zip md5 not computed (8.1
  #   GB, > 2 GB)
  #   survivor: C:\Users\Admin\tanitad-data\comma2k19\extracted (and the public comma2k19 dataset; HF Sayood/tanitad-comma2k19-episodes)
  #   risk/note: re-download of Chunk_1 is ~8 GB
  if (Test-ZipExtracted 'C:\Users\Admin\tanitad-data\comma2k19\Chunk_1.zip' 'C:\Users\Admin\tanitad-data\comma2k19\extracted' -BarToUnderscore) { Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-data\comma2k19\Chunk_1.zip' -Force }

  # 7.96 GB on disk (7.96 GB real, 1 files) | B-derived | derived copy: every member is extracted next to it (CRC-checked at extraction)
  #   evidence: zip central directory read (a few MB, no content read): 64 of 64 .db members have an extracted file with identical name+size in nuplan-v1.1/ (test/trainval; the mini members in
  #   nuplan-v1.1/splits and dblinks/) (LICENSE is at nuplan/LICENSE); the extract logs (extract_test.log / val_extract.json) record "extracting ... with per-member CRC". Whole-zip md5 NOT
  #   computed (> 2 GB).
  #   survivor: D:\Projects\TanitAD\data\nuplan\nuplan-v1.1\ (extracted) - and the zips are re-downloadable (public nuPlan S3 objects: INHERITED from memory note nuplan-s3-objects-are-public;
  #   pull_splits.log shows test/val were curl-downloaded from it)
  #   risk/note: frees the most space on D: but the extraction is on the SAME disk; if the dbs are ever damaged the 201 GB of zips must be re-downloaded: ~35 h at the dev-box rate of 1.6 MB/s
  #   (INHERITED rate, memory refe-training-data-on-a40-pod).
  if (Test-ZipExtracted 'D:\Projects\TanitAD\data\nuplan\nuplan-v1.1_mini.zip' @('D:\Projects\TanitAD\data\nuplan\nuplan-v1.1', 'D:\Projects\TanitAD\data\nuplan\dblinks')) { Remove-Item -LiteralPath 'D:\Projects\TanitAD\data\nuplan\nuplan-v1.1_mini.zip' -Force }

}

# ====================================================================================================
# B3. HELD: identical copies, but KEEP-area scripts hard-code the C: path. Repoint the scripts to the survivor, THEN un-comment.
# ====================================================================================================
# 1.33 GB on disk (1.33 GB real, 19 files) | B-held | identical copy exists on D:
#   evidence: md5 equal for 19 of 19 files (1.43 GB read-verified on both sides)
#   survivor: D:\Projects\TanitAD-artifacts\refcv5v2_final
#   risk/note: HELD: qland/t1_rolls.sh, qland/t1_s1_only.sh and qland/pkgrl/code/t1_rolls.sh hard-code BASE_CK=/c/Users/Admin/refcv5v2_final/ckpt.pt (+config.json) and PREREG_REFCV6.md cites
#   the C: path as MEASURED evidence. Repoint those scripts to the D: copy first, then un-comment the line.
# if (Test-CopyOf 'C:\Users\Admin\refcv5v2_final' 'D:\Projects\TanitAD-artifacts\refcv5v2_final') { Remove-Item -LiteralPath 'C:\Users\Admin\refcv5v2_final' -Recurse -Force }

# 3.25 GB on disk (3.25 GB real, 962 files) | B-held | identical copy exists on D: (962 files)
#   evidence: md5 equal for 962 of 962 files (3.49 GB read-verified on both sides)
#   survivor: D:\Projects\TanitAD-artifacts\_s2build-copy-20260919
#   risk/note: HELD: 238 KEEP-area text files (qland a2_harness_smoke.sh, a3_*, a8_occupancy_5k.sh ...) use LAB=/c/Users/Admin/tanitad-wt/_s2build/release/v8/s2_labels_v8_eval.jsonl.gz as an
#   input. Repoint to the D: copy, then un-comment.
# if (Test-CopyOf 'C:\Users\Admin\tanitad-wt\_s2build' 'D:\Projects\TanitAD-artifacts\_s2build-copy-20260919') { Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-wt\_s2build' -Recurse -Force }


Show-Summary
