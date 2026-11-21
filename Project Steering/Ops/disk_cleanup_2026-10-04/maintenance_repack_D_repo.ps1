<#
maintenance_repack_D_repo.ps1  --  NOT a delete script. Generated 2026-10-04, NOT run.
Why: D:\Projects\TanitAD\.git holds 42,842 LOOSE objects (5.37 GB of content) plus 16 packs (2.02 GB). On the exFAT volume every loose
object occupies a whole 1 MiB cluster, so the object store costs 48.1 GB on disk for ~7.4 GB of content. `git prune-packed` removes the
3,374 loose objects that are ALREADY inside a pack (zero information loss); `git repack -a -d` then folds the rest into packs.
Expected gain: ~3.3 GB from prune-packed alone, ~36-40 GB after the repack. No object is deleted that is not stored in a pack (no `gc --prune`).
Run it when no agent/lander is committing (the lander commits via tanitad-push, a separate repo; agents commit via mm_commit.py/scoped_commit.py
into this one). The repack reads ~7.4 GB and writes ~2-5 GB on the USB drive: schedule it off-peak and keep the drive awake.
#>
$DryRun = $true
$Repo = 'D:\Projects\TanitAD'
if (-not (Test-Path -LiteralPath (Join-Path $Repo '.git'))) { throw 'D: not mapped (subst D: E:\ )' }
$git = { param([string[]]$a) & git -c safe.directory=* -C $Repo @a }
Write-Host 'before:'; & $git @('count-objects','-vH')
if ($DryRun) { Write-Host 'DRY RUN: would run  git prune-packed  then  git repack -a -d --no-write-bitmap-index  then  git prune-packed  then  git count-objects -vH'; return }
if (Get-Process git -ErrorAction SilentlyContinue) { throw 'a git process is running - wait for it to finish' }
& $git @('prune-packed')
& $git @('repack','-a','-d','--no-write-bitmap-index')
& $git @('prune-packed')
Write-Host 'after:'; & $git @('count-objects','-vH')
& $git @('fsck','--connectivity-only','--no-dangling')
