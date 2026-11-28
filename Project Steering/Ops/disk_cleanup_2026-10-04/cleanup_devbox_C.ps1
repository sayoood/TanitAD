<#
=====================================================================================================
cleanup_devbox_C.ps1  --  TanitAD dev-box disk cleanup, category C  (probably obsolete - NEEDS PI DECISION; all removals commented out)
Generated 2026-10-04 by a READ-ONLY inventory. The inventory deleted/moved/changed NOTHING and this script has NOT been run.
Review, then run from a normal (non-admin) PowerShell 5.1 window:   powershell -NoProfile -File "cleanup_devbox_C.ps1"

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


# ---- D: is a subst of the USB drive E: (per logon session). If it vanished, stop instead of deleting through a wrong mapping.
if (-not (Test-Path -LiteralPath 'D:\Projects\TanitAD\.git')) { throw 'D:\Projects\TanitAD\.git not found - D: is not mapped. Run  subst D: E:\  first, then re-run.' }


# EVERY removal line below is COMMENTED OUT. Un-comment only the ones you approve (and read the risk note first).
# These are "probably obsolete, needs PI decision" items. Several are the ONLY copy of something (see the risk notes).

# ====================================================================================================
# C1. experiment outputs on C:  -  12 items, 157 GB
# ====================================================================================================
# 59.7 GB on disk (59.7 GB real, 1,811 files) | C | refav1 trunk-probe working set; re-creatable from Thor via refav1_trainprobe/pull_encode.py (scp + fp8 encode); refav1 planner eliminated
# on the full grid (memory 2026-09-27)
#   evidence: only mention: memory note fp8-encoder-reads-keys-from-g (how it was built)
#   risk/note: re-creation = scp the 600 v2ep from Thor (~16 min at the INHERITED 63 MB/s LAN rate) + re-run the fp8 encode on the dev-box GPU (time not measured)
# Remove-Item -LiteralPath 'C:\Users\Admin\refav1_trainprobe' -Recurse -Force

# 56.4 GB on disk (56.3 GB real, 19,577 files) | C | obsolete experiment outputs: rc7cost_runs holds 47 per-run ckpt.pt files (51.4 GB; 26 of them exactly 1,172,814,977 B, the rest 1.17-1.18
# GB; same SIZE only - content not compared, different runs) of the refcv7 restart cost study; the rest is gate/dry-run tree extracts and tars
#   evidence: 480 files in rc7cost_runs, 47 of them ckpt.pt; no mention in MODEL_REGISTRY / GOALS / RETRACTION_LOG; only qland docs mention C:/lgt/dry37086(_out)
#   risk/note: per-run result JSONs are inside rc7cost_runs; if a cost-study claim cites them, bank them first. Re-creation = re-running the cost probes.
# Remove-Item -LiteralPath 'C:\lgt' -Recurse -Force

# 12.4 GB on disk (12.3 GB real, 36,152 files) | C | READINESS_LOOP.md lists "clean C:/lgs gate trees" as a post-launch to-do; launch happened 2026-09-28
#   evidence: 2 trees of ~6 GB (19k + 16k files) + 8 tiny gate_devbox_<sha> result dirs (73 files each)
#   risk/note: gate_devbox_* hold the per-commit gate verdicts (tiny) - keep those 8 dirs (~1 MB) if you delete only the two trees
# Remove-Item -LiteralPath 'C:\lgs' -Recurse -Force

# 10.6 GB on disk (10.6 GB real, 3,934 files) | C | refcv7 restart study 2026-09-28, superseded by the 50,400-step run
#   evidence: 103 files in ab/; 3 files in liveckpt
#   risk/note: liveckpt = pulled snapshots of the then-live run (HF holds 5000/20000/30000/50400 only)
# Remove-Item -LiteralPath 'C:\Users\Admin\rc7restart_work' -Recurse -Force

# 7.79 GB on disk (7.79 GB real, 652 files) | C | DZ experiment outputs; superseded runs
#   evidence: registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 2 / RETRACTION_LOG 2 mentions; RETRACTION_LOG cites C:/dzo/m-nr-n and m-nr-r1 as rollout dirs of a retracted number
#   risk/note: raw rollouts behind a retracted claim - keep m-nr-n / m-nr-r1 if the retraction evidence matters
# Remove-Item -LiteralPath 'C:\dzo' -Recurse -Force

# 7.09 GB on disk (7.05 GB real, 20,627 files) | C | lander/Thor shipping scratch 2026-09-26/27; referenced only by finished readiness scripts
#   evidence: 12,058 files differ from the tree mtime; unique content not in git; run_dry_devbox.sh names IN=C:/Users/Admin/lg0926/inputs
#   risk/note: inputs/ is read by qland/work/refcv7/probe_cost/run_dry_devbox.sh (cost probe, finished) - keep inputs/ if you re-run it
# Remove-Item -LiteralPath 'C:\Users\Admin\lg0926' -Recurse -Force

# 978 MB on disk (977 MB real, 294 files) | C | v7F gate checkpoints (09-27) - checkpoint copies, not re-creatable from git
#   evidence: files=294
#   risk/note: experiment outputs; confirm the registry/RESULT docs do not cite them as the only raw artifact
# Remove-Item -LiteralPath 'C:\Users\Admin\v7f_ckpt_gate' -Recurse -Force

# 842 MB on disk (841 MB real, 599 files) | C | refav1 margin experiment outputs
#   evidence: files=599
#   risk/note: experiment outputs; confirm the registry/RESULT docs do not cite them as the only raw artifact
# Remove-Item -LiteralPath 'C:\Users\Admin\refav1_margin' -Recurse -Force

# 795 MB on disk (785 MB real, 4,942 files) | C | refav1 R1-R6 experiment outputs
#   evidence: files=4942
#   risk/note: experiment outputs; confirm the registry/RESULT docs do not cite them as the only raw artifact
# Remove-Item -LiteralPath 'C:\Users\Admin\refav1_r1r6' -Recurse -Force

# 405 MB on disk (399 MB real, 3,214 files) | C | v7F checkpoint staging (09-27) - checkpoint copies
#   evidence: files=3214
#   risk/note: experiment outputs; confirm the registry/RESULT docs do not cite them as the only raw artifact
# Remove-Item -LiteralPath 'C:\Users\Admin\v7f_ckpt' -Recurse -Force

# 265 MB on disk (265 MB real, 9 files) | C | bench files kept off-repo (not scanned)
#   evidence: files=9
#   risk/note: experiment outputs; confirm the registry/RESULT docs do not cite them as the only raw artifact
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-bench-offrepo' -Recurse -Force

# 252 MB on disk (252 MB real, 41 files) | C | refcv4b visualisation outputs (not scanned file-by-file)
#   evidence: files=41
#   risk/note: experiment outputs; confirm the registry/RESULT docs do not cite them as the only raw artifact
# Remove-Item -LiteralPath 'C:\Users\Admin\refcv4b_viz' -Recurse -Force

# ====================================================================================================
# C2. old datasets / staging on C:  -  4 items, 88.6 GB
# ====================================================================================================
# 33.0 GB on disk (32.9 GB real, 25,097 files) | C | mixed old eval inputs (2026-07-07/08)
#   evidence: v1_speedjerk_ckpt.pt = the deployed flagship v1 (registry: flagship4b-speedjerk-30k) and ckpt_full.pt = the no-speed control: HF copies exist per memory
#   (Sayood/tanitad-flagship-*), NOT re-verified
#   risk/note: delete only the two .pt files after confirming HF; comma_val.tgz is probably the compressed twin of comma2k19-val-61c46fca8f7f (not hash-checked)
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-data\eval' -Recurse -Force

# 29.7 GB on disk (29.7 GB real, 106 files) | C | staging for HF dataset push (2026-07-21..25); the three dirs are HARDLINKED: apparent 44.5 GB, real 29.7 GB on disk
#   evidence: hardlinks.py: tanitdataset logical 44.5 GB vs unique inodes 29.7 GB (28 multi-link files). Deleting only one of the three frees ~0
#   risk/note: confirm the HF dataset it staged is published (Sayood/tanitad-comma2k19-episodes) before deleting all three
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-data\tanitdataset' -Recurse -Force

# 15.3 GB on disk (15.3 GB real, 372 files) | C | RL pilot sweeps, superseded by ddv2rl
#   evidence: registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 1 mentions; same-size ckpt_after.pt x33 (different runs)
#   risk/note: rl-pilot/obstacle.offline.chunks (0.68 GB) is an input of the obstacle join - keep that subdir
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-data\rl-pilot' -Recurse -Force

# 10.7 GB on disk (10.7 GB real, 7,378 files) | C | raw comma2k19 chunk; registry cites comma2k19 13x (it is training data provenance)
#   evidence: registers: MODEL_REGISTRY 13 / GOALS_AND_CLAIMS 1 / RETRACTION_LOG 16 mentions
#   risk/note: the zip is listed separately as B-derived; the extracted copy is the working data
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-data\comma2k19' -Recurse -Force

# ====================================================================================================
# C3. scratch trees with unique content (C:)  -  24 items, 40.3 GB
# ====================================================================================================
# 5.83 GB on disk (5.80 GB real, 15,726 files) | C | scratch tree with unique content - unique data outputs not in git: .csv x31, .jsonl x23, .json x15, .npz x1 (all unique files incl.
# sources total 277.2 MB)
#   evidence: files=15726; newest 2026-09-26 23:09 UTC; 779 files differ from the extract mtime, 655 of them are byte-identical blobs in git; 124 NOT in git (277.18 MB: .log x43, .csv x31,
#   .jsonl x23, .json x15, .sh x3); extra entries: .claude, .gitignore, CLAUDE.md, Media
#   risk/note: qland/work/evalfw_restart/fallback_*.md, make_fallbacks.py and work/coord_land/watch_navhard30k.sh name C:/Users/Admin/ev6 as the CODE tree of the refcv6 eval fallback (fallback
#   docs of 2026-09-2x). Confirm the fallback is retired before deleting.
# Remove-Item -LiteralPath 'C:\Users\Admin\ev6' -Recurse -Force

# 4.80 GB on disk (4.77 GB real, 14,159 files) | C | scratch tree with unique content - top-level entries that are not part of the repo layout: .pytest_cache; unique data outputs not in git:
# .json x85, .jsonl x9 (all unique files incl. sources total 11.1 MB)
#   evidence: files=14159; newest 2026-09-17 08:26 UTC; 265 files differ from the extract mtime, 21 of them are byte-identical blobs in git; 244 NOT in git (11.12 MB: .py x89, .json x85, .md
#   x25, .log x15, .sh x12); extra entries: .claude, .gitignore, CLAUDE.md, Media
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-wt-join3d' -Recurse -Force

# 4.79 GB on disk (4.76 GB real, 13,845 files) | C | scratch tree with unique content - unique data outputs not in git: .json x2 (all unique files incl. sources total 1.6 MB)
#   evidence: files=13845; newest 2026-09-16 22:20 UTC; 47 files differ from the extract mtime, 9 of them are byte-identical blobs in git; 38 NOT in git (1.60 MB: .py x27, .md x8, .json x2,
#   .patch x1); extra entries: .claude, .gitignore, CLAUDE.md, Media
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-wt-percep-v6' -Recurse -Force

# 3.24 GB on disk (3.21 GB real, 16,435 files) | C | scratch tree with unique content - .git present (scratch `git init` repo, single synthetic commit not found in D:/Projects/TanitAD/.git or
# tanitad-push)
#   evidence: files=16435; has a .git dir (~1.2 GB); newest file 2026-09-26 12:40 UTC; 3 mtime-anomalous files, 3 of them byte-identical blobs in git; extra entries: .claude, .gitignore,
#   CLAUDE.md, Media; mentioned by qland/work/yawfix/build_ygate.py (generator script)
# Remove-Item -LiteralPath 'C:\Users\Admin\ygate_fix' -Recurse -Force

# 3.24 GB on disk (3.21 GB real, 16,434 files) | C | scratch tree with unique content - .git present (scratch `git init` repo, single synthetic commit not found in D:/Projects/TanitAD/.git or
# tanitad-push)
#   evidence: files=16434; has a .git dir (~1.2 GB); newest file 2026-09-26 12:35 UTC; 0 mtime-anomalous files, 0 of them byte-identical blobs in git; extra entries: .claude, .gitignore,
#   CLAUDE.md, Media; mentioned by qland/work/yawfix/build_ygate.py (generator script)
# Remove-Item -LiteralPath 'C:\Users\Admin\ygate_tip' -Recurse -Force

# 3.22 GB on disk (3.19 GB real, 17,214 files) | C | scratch tree with unique content - .git present (scratch `git init` repo, single synthetic commit not found in D:/Projects/TanitAD/.git or
# tanitad-push); unique data outputs not in git: .json x1 (all unique files incl. sources total 0.2 MB)
#   evidence: files=17214; has a .git dir (~1.2 GB); newest file 2026-09-26 11:58 UTC; 513 mtime-anomalous files, 508 of them byte-identical blobs in git; 5 NOT in git (0.21 MB: .py x4, .json
#   x1); extra entries: .claude, .gitignore, CLAUDE.md, Media; mentioned by qland/work/coord_land/compare_stack.json (log record)
# Remove-Item -LiteralPath 'C:\Users\Admin\cfull_cand' -Recurse -Force

# 3.07 GB on disk (3.03 GB real, 16,672 files) | C | scratch tree with unique content - .git present (scratch `git init` repo, single synthetic commit not found in D:/Projects/TanitAD/.git or
# tanitad-push)
#   evidence: files=16672; has a .git dir (~1.2 GB); newest file 2026-09-26 09:26 UTC; 0 mtime-anomalous files, 0 of them byte-identical blobs in git; extra entries: .claude, .gitignore,
#   CLAUDE.md, Media; mentioned by qland/work/coord_land/compare_*.json (log record)
# Remove-Item -LiteralPath 'C:\Users\Admin\cfull_tip' -Recurse -Force

# 2.75 GB on disk (2.74 GB real, 1,885 files) | C | scratch tree with unique content - top-level entries that are not part of the repo layout: a17, bench, code, diag, hqs, overlay; unique
# data outputs not in git: .parquet x42, .tar x27, .md5 x14, .json x12, .list x2, .done x2 (all unique files incl. sources total 2887.4 MB)
#   evidence: files=1885; newest 2026-09-27 15:03 UTC; 559 files differ from the extract mtime, 291 of them are byte-identical blobs in git; 268 NOT in git (2887.43 MB: .py x109, .parquet x42,
#   .txt x29, .tar x27, .log x18)
# Remove-Item -LiteralPath 'C:\Users\Admin\bxh_work' -Recurse -Force

# 2.66 GB on disk (2.65 GB real, 6,385 files) | C | scratch tree with unique content - top-level entries that are not part of the repo layout: argv, gate_runs, raw, scratch, tmp_keys, tree;
# unique data outputs not in git: .json x1014, .pt x105, .jsonl x67, .csv x29, .svg x19 (all unique files incl. sources total 2172.4 MB)
#   evidence: files=6385; newest 2026-09-27 15:35 UTC; 3470 files differ from the extract mtime, 459 of them are byte-identical blobs in git; 3011 NOT in git (2172.43 MB: .py x1504, .json
#   x1014, .pt x105, .sh x82, .jsonl x67)
# Remove-Item -LiteralPath 'C:\Users\Admin\v7f_gate' -Recurse -Force

# 915 MB on disk (898 MB real, 8,156 files) | C | scratch tree with unique content - top-level entries that are not part of the repo layout: baseline, deliv, mut, raw, smoke; unique data
# outputs not in git: .json x690, .jsonl x8 (all unique files incl. sources total 71.9 MB)
#   evidence: files=8156; newest 2026-09-27 13:07 UTC; 3496 files differ from the extract mtime, 285 of them are byte-identical blobs in git; 3211 NOT in git (71.88 MB: .py x2236, .json x690,
#   .sh x146, .md x56, .log x21)
# Remove-Item -LiteralPath 'C:\Users\Admin\v7f_r1r4' -Recurse -Force

# 833 MB on disk (827 MB real, 3,252 files) | C | scratch tree with unique content - .git present (scratch `git init` repo, single synthetic commit not found in D:/Projects/TanitAD/.git or
# tanitad-push); top-level entries that are not part of the repo layout: deliverable, env.sh, proof, raw; unique data outputs not in git: .json x55, .sample x14, .pt x11, .jsonl x3 (all
# unique files incl. sources total 456.4 MB)
#   evidence: files=3252; has a .git dir (~1.2 GB); newest file 2026-09-27 13:41 UTC; 161 mtime-anomalous files, 20 of them byte-identical blobs in git; 141 NOT in git (456.40 MB: .json x55,
#   .log x23, .txt x22, .sample x14, .pt x11)
# Remove-Item -LiteralPath 'C:\Users\Admin\v7f_r3' -Recurse -Force

# 670 MB on disk (656 MB real, 6,727 files) | C | scratch tree with unique content - top-level entries that are not part of the repo layout: b3f7ea6f, c36b6ddd; unique data outputs not in
# git: .json x645, .csv x29, .jsonl x20, .svg x19 (all unique files incl. sources total 121.4 MB)
#   evidence: files=6727; newest 2026-09-27 12:33 UTC; 2869 files differ from the extract mtime, 440 of them are byte-identical blobs in git; 2429 NOT in git (121.44 MB: .py x1482, .json x645,
#   .sh x82, .md x51, .log x49)
# Remove-Item -LiteralPath 'C:\Users\Admin\tipsnap' -Recurse -Force

# 428 MB on disk (420 MB real, 3,785 files) | C | scratch tree with unique content - top-level entries that are not part of the repo layout: _tip_eff.py, apply_f2_gate_test.py,
# apply_f2_strict.py, apply_f7_f8.py, apply_navcond_map.py, bank_merge.py; unique data outputs not in git: .json x328, .csv x29, .svg x19, .jsonl x15, .html x7 (all unique files incl. sources
# total 89.2 MB)
#   evidence: files=3785; newest 2026-09-27 15:52 UTC; 1171 files differ from the extract mtime, 328 of them are byte-identical blobs in git; 843 NOT in git (89.21 MB: .py x342, .json x328,
#   .log x42, .csv x29, .txt x26)
# Remove-Item -LiteralPath 'C:\Users\Admin\v7f_merge' -Recurse -Force

# 409 MB on disk (397 MB real, 6,528 files) | C | scratch tree with unique content - top-level entries that are not part of the repo layout: RESULT.md, RESULT_DRAFT_METHOD.md, SPEC.md, code,
# raw; not a git-archive extract (6195 mtime-anomalous files)
#   evidence: files=6528; newest 2026-09-27 00:49 UTC; 6195 files differ from the extract mtime, 0 of them are byte-identical blobs in git
# Remove-Item -LiteralPath 'C:\Users\Admin\ev6_battery' -Recurse -Force

# 402 MB on disk (396 MB real, 3,324 files) | C | scratch tree with unique content - top-level entries that are not part of the repo layout: _pkg
#   evidence: files=3324; newest 2026-09-27 16:40 UTC; 232 files differ from the extract mtime, 188 of them are byte-identical blobs in git; 44 NOT in git (2.97 MB: .py x27, .log x14, .txt x2,
#   .diff x1); extra entries: _regress_final_cand.log, _t_all_a6.log, _t_bt.log, _t_f_a6.log
# Remove-Item -LiteralPath 'C:\Users\Admin\nb2_tree_2301' -Recurse -Force

# 398 MB on disk (392 MB real, 3,135 files) | C | scratch tree with unique content - top-level entries that are not part of the repo layout: _out, _pkg; unique data outputs not in git: .json
# x3, .npz x1 (all unique files incl. sources total 5.8 MB)
#   evidence: files=3135; newest 2026-09-26 21:20 UTC; 64 files differ from the extract mtime, 12 of them are byte-identical blobs in git; 52 NOT in git (5.76 MB: .py x39, .json x3, .log x3,
#   .txt x3, .npz x1)
# Remove-Item -LiteralPath 'C:\Users\Admin\nb2_tree_2128' -Recurse -Force

# 397 MB on disk (390 MB real, 3,093 files) | C | scratch tree with unique content - unique data outputs not in git: .json x1 (all unique files incl. sources total 0.1 MB)
#   evidence: files=3093; newest 2026-09-27 13:00 UTC; 2 files differ from the extract mtime, 0 of them are byte-identical blobs in git; 2 NOT in git (0.07 MB: .md x1, .json x1)
# Remove-Item -LiteralPath 'C:\Users\Admin\v7f_r3_base' -Recurse -Force

# 385 MB on disk (379 MB real, 2,998 files) | C | scratch tree with unique content - unique data outputs not in git: .json x1 (all unique files incl. sources total 0.5 MB)
#   evidence: files=2998; newest 2026-09-27 14:45 UTC; 9 files differ from the extract mtime, 4 of them are byte-identical blobs in git; 5 NOT in git (0.51 MB: .py x3, .log x1, .json x1); extra
#   entries: _t_gate.log
# Remove-Item -LiteralPath 'C:\Users\Admin\nb2ml_tree_3708' -Recurse -Force

# 385 MB on disk (379 MB real, 2,909 files) | C | scratch tree with unique content - modified within the last 72 h (newest file 2026-10-04 01:44 UTC): may still be in use by the lander
#   evidence: files=2909; newest 2026-10-04 01:44 UTC; 16 files differ from the extract mtime, 14 of them are byte-identical blobs in git; 2 NOT in git (0.01 MB: .txt x2); extra entries:
#   relevant_tests.txt, relevant_tests_edited.txt
# Remove-Item -LiteralPath 'C:\Users\Admin\fixtree' -Recurse -Force

# 385 MB on disk (379 MB real, 2,907 files) | C | scratch tree with unique content - modified within the last 72 h (newest file 2026-10-04 02:38 UTC): may still be in use by the lander
#   evidence: files=2907; newest 2026-10-04 02:38 UTC; 14 files differ from the extract mtime, 14 of them are byte-identical blobs in git
# Remove-Item -LiteralPath 'C:\Users\Admin\ygate_fixes' -Recurse -Force

# 384 MB on disk (378 MB real, 2,902 files) | C | scratch tree with unique content - modified within the last 72 h (newest file 2026-10-04 01:02 UTC): may still be in use by the lander
#   evidence: files=2902; newest 2026-10-04 01:02 UTC; 3 files differ from the extract mtime, 0 of them are byte-identical blobs in git; 3 NOT in git (0.00 MB: .log x1, .txt x1, .sh x1); extra
#   entries: baseline_stack.log, relevant_tests.txt, run_baseline.sh
# Remove-Item -LiteralPath 'C:\Users\Admin\fixtree_tip' -Recurse -Force

# 345 MB on disk (328 MB real, 8,283 files) | C | repo snapshot of 2026-09-04 (refcv4b landing) with result files and a <token-file> copy
#   evidence: files=8283; newest 2026-09-06 11:35 UTC; 7614 files differ from the extract mtime, 0 of them are byte-identical blobs in git; extra entries: CLAUDE.md, Media, README.md
#   risk/note: CONTAINS A COPY OF <token-file> (secrets file, seen as a top-level entry; contents NOT read). Delete this dir for credential hygiene regardless of the rest; never commit/zip it. Not
#   in git (7,614 mtime-anomalous files).
# Remove-Item -LiteralPath 'C:\Users\Admin\refcv4b_repo' -Recurse -Force

# 278 MB on disk (275 MB real, 1,417 files) | C | scratch tree with unique content - modified within the last 72 h (newest file 2026-10-03 23:59 UTC): may still be in use by the lander
#   evidence: files=1417; newest 2026-10-03 23:59 UTC; 3 files differ from the extract mtime, 3 of them are byte-identical blobs in git
# Remove-Item -LiteralPath 'C:\Users\Admin\ygate_video' -Recurse -Force

# 205 MB on disk (203 MB real, 1,006 files) | C | scratch tree with unique content - .git present (scratch `git init` repo, single synthetic commit not found in D:/Projects/TanitAD/.git or
# tanitad-push); top-level entries that are not part of the repo layout: DriveZero, dblinks, out; unique data outputs not in git: .tsv x11, .temp x5, .xz x4, .nuboard x3, .yaml x3 (all unique
# files incl. sources total 96.7 MB)
#   evidence: files=1006; has a .git dir (~1.2 GB); newest file 2026-09-26 09:56 UTC; 131 mtime-anomalous files, 17 of them byte-identical blobs in git; 114 NOT in git (96.70 MB: .log x77, .tsv
#   x11, .txt x8, .temp x5, .xz x4)
# Remove-Item -LiteralPath 'C:\Users\Admin\dz' -Recurse -Force

# ====================================================================================================
# C4. old worktrees / mirrors  -  4 items, 42.5 GB
# ====================================================================================================
# 21.0 GB on disk (5.94 GB real, 16,342 files) | C | git worktrees registered in the repo (`git worktree list`); 16,342 files; stale since 2026-09-20
#   evidence: upbeat-lamport is at the same commit as the repo tip 37645fcc
#   risk/note: use `git worktree remove` (not Remove-Item) so .git/worktrees is cleaned; check no session is attached
# git -c safe.directory=* -C "D:\Projects\TanitAD" worktree remove --force <worktree path>   # preferred over Remove-Item for registered worktrees
# Remove-Item -LiteralPath 'D:\Projects\TanitAD\.claude\worktrees' -Recurse -Force

# 15.7 GB on disk (4.69 GB real, 12,048 files) | C | registered git worktree, stale since 2026-09-16
#   risk/note: use `git worktree remove`; confirm branch agent/dataeng-256x1024-20260916 is merged/pushed
# git -c safe.directory=* -C "D:\Projects\TanitAD" worktree remove --force <worktree path>   # preferred over Remove-Item for registered worktrees
# Remove-Item -LiteralPath 'D:\Projects\TanitAD-wt-1024' -Recurse -Force

# 3.09 GB on disk (3.08 GB real, 4,705 files) | C | old worktree (2026-07/08) incl. stack/experiments/*/model.pt, _s2build (B-held), v3b1smoke
#   evidence: registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 2 / RETRACTION_LOG 4 mentions; 3,602 mtime-anomalous files, 1,941 byte-identical to git, 1,661 not (6.5 GB)
#   risk/note: _s2build inside is the input of live qland harness scripts (see B-held); delete tanitad-wt only after repointing them
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-wt' -Recurse -Force

# 2.67 GB on disk (2.66 GB real, 8,346 files) | C | mirror; not synced since 2026-08-29
#   evidence: registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 1 / RETRACTION_LOG 8 mentions; 614 files / 2.1 GB not in git (model.pt p0-sA01/sB00 ... same 1.05 GB files also exist x12 in
#   devbox-C\tanitad-lwbase etc.)
#   risk/note: RETRACTION_LOG cites it 8x as a comparison tree (v6.py md5 table) - a retraction record, not live code
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-mirror' -Recurse -Force

# ====================================================================================================
# C6. D:\Archive\devbox-C - junction-backed dirs moved from C: on 2026-09-13  -  31 items, 264 GB
# ====================================================================================================
# 53.2 GB on disk (25.3 GB real, 31,420 files) | C | CARLA 0.9 sim install + data (2025-06..2026-07)
#   evidence: newest file 2026-07-08; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\carla' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\carla' -Recurse -Force

# 31.0 GB on disk (30.1 GB real, 989 files) | C | wpd probe banks (tok_D0/D1/D2.npy 2.98 GB each x2 copies)
#   evidence: newest file 2026-09-10; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\wpd-probe' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\wpd-probe' -Recurse -Force

# 18.8 GB on disk (15.9 GB real, 3,318 files) | C | dkhead run outputs
#   evidence: newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\dkhead_run' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\dkhead_run' -Recurse -Force

# 14.0 GB on disk (13.8 GB real, 285 files) | C | refav1 full-eval episodes (13.8 GB); same size as C:\tanitad-data\refav1-eval141 (13.8 GB) - not hash-compared
#   evidence: newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\refav1_eval_full' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\refav1_eval_full' -Recurse -Force

# 11.8 GB on disk (5.91 GB real, 6,204 files) | C | refav1 probe runs
#   evidence: newest file 2026-09-27; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\refav1_probe' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\refav1_probe' -Recurse -Force

# 11.0 GB on disk (5.83 GB real, 5,576 files) | C | refcv4 run (2026-09-04)
#   evidence: newest file 2026-09-04; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\run_refcv4' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\run_refcv4' -Recurse -Force

# 10.5 GB on disk (8.24 GB real, 2,462 files) | C | wpa readout banks (tok.npy 2.2 GB x2)
#   evidence: newest file 2026-09-08; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\wpa-readout' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\wpa-readout' -Recurse -Force

# 8.31 GB on disk (7.83 GB real, 514 files) | C | refav1 eval slice
#   evidence: newest file 2026-09-04; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 1 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\refav1_eval_slice' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\refav1_eval_slice' -Recurse -Force

# 7.93 GB on disk (6.23 GB real, 1,849 files) | C | NavSim comparison runs
#   evidence: newest file 2026-09-05; registers: MODEL_REGISTRY 2 / GOALS_AND_CLAIMS 2 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\navcomp' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\navcomp' -Recurse -Force

# 7.13 GB on disk (6.99 GB real, 215 files) | C | P1 gate runs
#   evidence: newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\p1gate' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\p1gate' -Recurse -Force

# 7.12 GB on disk (6.98 GB real, 243 files) | C | RL gate 2 runs
#   evidence: newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\rlgate2' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\rlgate2' -Recurse -Force

# 6.71 GB on disk (5.38 GB real, 1,459 files) | C | refcv3 open-loop run
#   evidence: newest file 2026-09-03; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\run_refcv3_ol' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\run_refcv3_ol' -Recurse -Force

# 5.97 GB on disk (4.03 GB real, 2,084 files) | C | refcv4 variant run
#   evidence: newest file 2026-09-04; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 2 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\run_refcv4v' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\run_refcv4v' -Recurse -Force

# 5.63 GB on disk (3.44 GB real, 2,883 files) | C | bench worktree (refav1_eval_21109 ckpts)
#   evidence: newest file 2026-09-04; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-bench' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\tanitad-bench' -Recurse -Force

# 5.17 GB on disk (4.99 GB real, 279 files) | C | RL refcv3-min
#   evidence: newest file 2026-09-05; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 9 / RETRACTION_LOG 3 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\rl_refcv3_min' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\rl_refcv3_min' -Recurse -Force

# 5.03 GB on disk (2.31 GB real, 2,955 files) | C | label-wire base worktree (12 x 1.05 GB model.pt copies)
#   evidence: newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-lwbase' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\tanitad-lwbase' -Recurse -Force

# 5.03 GB on disk (2.31 GB real, 2,948 files) | C | label-wire worktree (same model.pt copies)
#   evidence: newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-labelwire' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\tanitad-labelwire' -Recurse -Force

# 4.81 GB on disk (2.23 GB real, 2,772 files) | C | refcv4b analysis worktree
#   evidence: newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-refcv4b-analyze' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\tanitad-refcv4b-analyze' -Recurse -Force

# 4.72 GB on disk (2.10 GB real, 2,791 files) | C | evtool worktree
#   evidence: newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-evtool-20260906' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\tanitad-evtool-20260906' -Recurse -Force

# 4.47 GB on disk (2.09 GB real, 2,532 files) | C | per-batch worktree
#   evidence: newest file 2026-09-07; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-perbatch' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\tanitad-perbatch' -Recurse -Force

# 4.45 GB on disk (2.13 GB real, 2,480 files) | C | paired runs
#   evidence: newest file 2026-09-03; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\run_paired' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\run_paired' -Recurse -Force

# 4.30 GB on disk (2.08 GB real, 2,363 files) | C | ctg worktree
#   evidence: newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 1 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-ctg' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\tanitad-ctg' -Recurse -Force

# 4.29 GB on disk (3.66 GB real, 681 files) | C | veto runs
#   evidence: newest file 2026-09-05; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 2 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\veto_run' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\veto_run' -Recurse -Force

# 4.19 GB on disk (1.26 GB real, 3,608 files) | C | refcv3 viz run
#   evidence: newest file 2026-09-04; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\run_refcv3_viz' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\run_refcv3_viz' -Recurse -Force

# 4.11 GB on disk (3.20 GB real, 953 files) | C | tsc2 runs
#   evidence: newest file 2026-09-02; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\tsc2' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\tsc2' -Recurse -Force

# 3.91 GB on disk (2.54 GB real, 1,465 files) | C | weight-bank runs
#   evidence: newest file 2026-09-05; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\run_wbank' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\run_wbank' -Recurse -Force

# 2.88 GB on disk (2.86 GB real, 22 files) | C | off-repo, one copy, re-fetchable (qland r_d.md: "accepted_terms_by: Sayed")
#   evidence: no junction
#   risk/note: re-fetch needs the nuScenes licence acceptance again
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\nuscenes' -Recurse -Force

# 2.73 GB on disk (1.63 GB real, 1,158 files) | C | refcv4b ego-drop runs
#   evidence: newest file 2026-09-05; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 1 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\refcv4b_egodrop' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\refcv4b_egodrop' -Recurse -Force

# 1.99 GB on disk (1.65 GB real, 369 files) | C | RL rescope
#   evidence: newest file 2026-09-05; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\rl_rescope' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\rl_rescope' -Recurse -Force

# 1.56 GB on disk (1.40 GB real, 178 files) | C | refav1 drive outputs
#   evidence: newest file 2026-09-05; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 3 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\refav1_drive' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\refav1_drive' -Recurse -Force

# 1.23 GB on disk (1.23 GB real, 4 files) | C | ego-drop checkpoint
#   evidence: newest file 2026-09-04; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions
#   risk/note: finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it)
# Remove-Item -LiteralPath 'C:\Users\Admin\tanitad-egodrop' -Force                 # junction first (link only)
# Remove-Item -LiteralPath 'D:\Archive\devbox-C\tanitad-egodrop' -Recurse -Force

# ====================================================================================================
# C7. archives of removed originals (thor_archive, _c)  -  2 items, 286 GB
# ====================================================================================================
# 175 GB on disk (108 GB real, 82,695 files) | C | Thor content archived to the dev box and then REMOVED FROM THOR on 2026-09-14 (sam3-only-road-map RESULT.md: md5-verified, then deleted on
# Thor)
#   evidence: this may be the ONLY copy of those Thor files; alpasim = 60,998 files / 2.8 GB real (57 GB is 1 MiB-cluster slack)
#   risk/note: irreversible if Thor no longer has them. Cheaper first step: pack alpasim/ into one tar (frees ~57 GB, loses nothing)
# Remove-Item -LiteralPath 'D:\thor_archive' -Recurse -Force

# 111 GB on disk (111 GB real, 209 files) | C | archives of dirs that were deleted from C: after sha256+CRC verification (MANIFEST.tsv, MOVE_LOG.txt)
#   evidence: each archive has a sha256 in _c\MANIFEST.tsv; the originals no longer exist
#   risk/note: these ARE the only copies of those old worktrees (git history is in .git, so tracked content is recoverable; untracked outputs inside are not)
# Remove-Item -LiteralPath 'D:\Projects\TanitAD\_c' -Recurse -Force

# ====================================================================================================
# C8. D:\Projects\TanitAD-artifacts  -  12 items, 71.3 GB
# ====================================================================================================
# 10.8 GB on disk (10.7 GB real, 141 files) | C | eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); eval episode cache 416x1024 cylindrical; re-creatable from
# the Thor v2ep + build scripts
#   evidence: registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 1 / RETRACTION_LOG 0 mentions; named as an input by qland/a3_halfA_maphead.sh, a3_heldout_read.sh, a8_occupancy_5k.sh (one-shots of
#   2026-09-19)
#   risk/note: KEEP-area one-shot scripts name it - delete only if the A-series (A1-A9) panels will not be re-run
# Remove-Item -LiteralPath 'D:\Projects\TanitAD-artifacts\v2ep-eval139-416x1024cyl' -Recurse -Force

# 10.7 GB on disk (10.6 GB real, 140 files) | C | eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); eval episode cache 408x1024; re-creatable from the Thor v2ep
# + build scripts
#   evidence: registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions; not named by any KEEP-area script
#   risk/note: low
# Remove-Item -LiteralPath 'D:\Projects\TanitAD-artifacts\v2ep-eval139-408x1024cyl' -Recurse -Force

# 8.33 GB on disk (5.59 GB real, 2,971 files) | C | refcv5 comparison outputs (2026-09-07); 2,971 files
#   evidence: the live NavSim package mentions refcv5cmp (C:\Users\Admin\refcv5cmp is the working copy)
#   risk/note: separate from C:\Users\Admin\refcv5cmp (KEEP, referenced)
# Remove-Item -LiteralPath 'D:\Projects\TanitAD-artifacts\refcv5cmp' -Recurse -Force

# 7.59 GB on disk (7.52 GB real, 141 files) | C | eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); eval episode cache 256x1024; re-creatable from the Thor v2ep
# + build scripts
#   evidence: registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions; not named by any KEEP-area script
#   risk/note: low
# Remove-Item -LiteralPath 'D:\Projects\TanitAD-artifacts\v2ep-eval139-256x1024cyl' -Recurse -Force

# 5.58 GB on disk (5.55 GB real, 71 files) | C | eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); split A of the 416x1024 cache; re-creatable from the Thor v2ep
# + build scripts
#   evidence: no register mention; named only by qland/a0_clean124_split.py (the generator of the halves)
#   risk/note: low
# Remove-Item -LiteralPath 'D:\Projects\TanitAD-artifacts\v2ep-eval139-416x1024cyl-splitA' -Recurse -Force

# 5.45 GB on disk (4.86 GB real, 1,128 files) | C | staging dir of the 2026-09-15 augmented-corpus HF push (1,128 files)
#   evidence: HF push assumed done (not re-verified)
#   risk/note: verify the HF dataset before deleting
# Remove-Item -LiteralPath 'D:\Projects\TanitAD-artifacts\hf-corpus-aug-20260915' -Recurse -Force

# 5.23 GB on disk (5.20 GB real, 70 files) | C | eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); split B of the 416x1024 cache; re-creatable from the Thor v2ep
# + build scripts
#   evidence: no register mention; named only by qland/a0_clean124_split.py (the generator of the halves)
#   risk/note: low
# Remove-Item -LiteralPath 'D:\Projects\TanitAD-artifacts\v2ep-eval139-416x1024cyl-splitB' -Recurse -Force

# 4.87 GB on disk (4.84 GB real, 64 files) | C | eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); clean-124 half B; re-creatable from the Thor v2ep + build
# scripts
#   evidence: no register mention; named as an input by qland/a1_per_half_floor.py, a2_harness_smoke.sh, a3_*.sh, a8_occupancy_5k.sh (one-shots of 2026-09-19)
#   risk/note: KEEP-area one-shot scripts name it - delete only if the A-series (A1-A9) panels will not be re-run
# Remove-Item -LiteralPath 'D:\Projects\TanitAD-artifacts\v2ep-eval124clean-416x1024cyl-halfB' -Recurse -Force

# 4.83 GB on disk (4.80 GB real, 64 files) | C | eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); clean-124 half A; re-creatable from the Thor v2ep + build
# scripts
#   evidence: no register mention; named as an input by qland/a1_per_half_floor.py, a2_harness_smoke.sh, a3_*.sh, a8_occupancy_5k.sh (one-shots of 2026-09-19)
#   risk/note: KEEP-area one-shot scripts name it - delete only if the A-series (A1-A9) panels will not be re-run
# Remove-Item -LiteralPath 'D:\Projects\TanitAD-artifacts\v2ep-eval124clean-416x1024cyl-halfA' -Recurse -Force

# 4.04 GB on disk (3.83 GB real, 288 files) | C | eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); build workspace of the eval139 caches; re-creatable from the
# Thor v2ep + build scripts
#   evidence: no register mention; not named by any KEEP-area script
#   risk/note: low
# Remove-Item -LiteralPath 'D:\Projects\TanitAD-artifacts\v2ep-eval139-build' -Recurse -Force

# 2.69 GB on disk (2.38 GB real, 531 files) | C | REFe final model bundle of 2026-10-01 (model_final.pt 1.29 GB also at data\refe_runs_eval\model_final.pt)
#   evidence: not named in the live upload script (it names refe-pod-final-2026-10-04 and refe-onpolicy-2026-10-01)
#   risk/note: REFe session is active - ask it before touching anything named refe-*
# Remove-Item -LiteralPath 'D:\Projects\TanitAD-artifacts\refe-final-2026-10-01' -Recurse -Force

# 1.22 GB on disk (1.21 GB real, 11 files) | C | HF push staging (also packed in _c\hf-refcv5v2.tar.zst)
#   evidence: overlaps refcv5v2_final/ckpt.pt (1.3 GB)
# Remove-Item -LiteralPath 'D:\Projects\TanitAD-artifacts\hf-refcv5v2' -Recurse -Force

# ====================================================================================================
# C9. eval kits  -  2 items, 30.3 GB
# ====================================================================================================
# 28.5 GB on disk (28.3 GB real, 594 files) | C | refcv6 stopped 2026-09-26; eval kit data (refcv6-b1-416x1024-eval139 caches) + step ckpts
#   evidence: registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions; qland hfpush_refcv6.py + a16_switch meta name it; ckpt_step38000 md5 5a2e7222... verified = recorded;
#   HF private Sayood/tanitad-refcv6-r101-s0 (not re-checked)
#   risk/note: delete data/ + ckpt/ first; keep ckpt_final/ (the stopped checkpoint) until the HF copy is verified
# Remove-Item -LiteralPath 'D:\refcv6_eval_kit' -Recurse -Force

# 1.81 GB on disk (1.76 GB real, 192 files) | C | SAM3 map LiDAR spin decodes (decode_sweeps.py output, 2026-09-13)
#   evidence: only mention: sam3-only-road-map decode_sweeps.py
#   risk/note: SAM3 map GT is a PI-led direction (memory 2026-09-13); small
# Remove-Item -LiteralPath 'D:\sam3map_lidar' -Recurse -Force


Show-Summary
