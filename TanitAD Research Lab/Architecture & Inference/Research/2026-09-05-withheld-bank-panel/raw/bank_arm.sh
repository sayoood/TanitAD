#!/bin/sh
# Bank ONE arm's outputs into the research package, md5-verified per file.
# ⛔ Never bank a checkpoint (ckpt.pt is large and the repo is not a model store); bank
# the CONFIG, the METRICS and the LOG -- what a reader needs to re-derive the numbers and
# to check the arm actually ran the lever it claims.
#
# ⚠️⚠️ THREE WRITE TRAPS ON THIS MOUNT, ALL MEASURED 2026-09-05, ALL FALSE-FAILURE
# GENERATORS -- each reports something alarming about the DATA when the fault is the
# ACCESS PATH:
#   1. The MSYS `/g/…` path form does not resolve for writes: `mkdir -p` says "File
#      exists" while a redirect into the same path says "No such file or directory".
#      Use the drive-letter form.
#   2. `cp` can fail with "cannot create regular file: File exists" for a path that `ls`
#      proves does NOT exist.
#   3. ⛔ THE ONE THAT COST THE MOST: into a FRESHLY CREATED directory, EVERY MSYS write
#      fails ("Invalid argument" / "No such file or directory") even though `mkdir`
#      returned 0 and `test -d` says EXISTS -- while the NATIVE Windows API writes there
#      fine. A0/A1/A2 banked through MSYS; A3's new directory refused all four files
#      through MSYS and accepted all four through PowerShell `Copy-Item` immediately.
# ⇒ copy through PowerShell (native API), then verify the md5 from the shell side.
# This is the same family as the `git add` fresh-inode trap already in CLAUDE.md.
#
# usage: sh bank_arm.sh <arm-name>
set -u
A=$1
W=/c/Users/Admin/run_wbank
PKGW="G:\\Meine Ablage\\SayBouBase\\raw\\Projects\\TanitAD\\TanitAD Research Lab\\Architecture & Inference\\Research\\2026-09-05-withheld-bank-panel\\raw"
PKG="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-withheld-bank-panel/raw"

# native-API copy of the whole arm, one PowerShell call
powershell.exe -NoProfile -NonInteractive -Command "
\$src = 'C:\\Users\\Admin\\run_wbank\\arms\\$A'
\$dst = '$PKGW\\$A'
if (-not (Test-Path \$dst)) { New-Item -ItemType Directory -Force -Path \$dst | Out-Null }
foreach (\$f in 'config.json','metrics.jsonl','summary.json') {
  \$s = Join-Path \$src \$f
  if (Test-Path \$s) { try { Copy-Item -LiteralPath \$s -Destination (Join-Path \$dst \$f) -Force -ErrorAction Stop } catch { Write-Output ('FAIL ' + \$f + ' :: ' + \$_.Exception.Message) } }
  else { Write-Output ('MISSING ' + \$f) }
}
\$lg = 'C:\\Users\\Admin\\run_wbank\\arms\\$A.log'
if (Test-Path \$lg) { try { Copy-Item -LiteralPath \$lg -Destination (Join-Path \$dst 'train.log') -Force -ErrorAction Stop } catch { Write-Output ('FAIL train.log :: ' + \$_.Exception.Message) } }
" 2>&1 | sed 's/^/  ps: /'

# ⭐ verify by CONTENT from the shell side -- a copy tool reporting success is not
# evidence its output is right (skill section 5).
ok=0; bad=0
check () {
  s=$1; d=$2
  a=$(md5sum < "$s" 2>/dev/null | cut -d' ' -f1)
  b=$(md5sum < "$PKG/$A/$d" 2>/dev/null | cut -d' ' -f1)
  if [ -n "$a" ] && [ "$a" = "$b" ]; then
    echo "OK   $d  md5=$a  bytes=$(wc -c < "$PKG/$A/$d")"; ok=$((ok+1))
  else
    echo "⛔ BAD $d  (src=${a:-ERR} dst=${b:-ERR})"; bad=$((bad+1))
  fi
}
check "$W/arms/$A/config.json"   config.json
check "$W/arms/$A/metrics.jsonl" metrics.jsonl
check "$W/arms/$A/summary.json"  summary.json
check "$W/arms/$A.log"           train.log

n=$(grep -c '' "$PKG/$A/metrics.jsonl" 2>/dev/null || echo 0)
echo "ZZBANK-$A-ok${ok}-bad${bad}-rows${n}-ZZ"
[ "$n" -ge 10 ] || echo "⛔ $A: banked metrics.jsonl has $n rows -- do not quote this arm"
