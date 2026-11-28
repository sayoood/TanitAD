#!/usr/bin/env bash
# Stage 5 (HANDOFF_S5.md): ONE composite arm on box 2, the way stage4_chain.sh ran C0C. NOT RUN by its author (AGENT R): the Master Mind launches it.
# usage: bash stage5_chain.sh <RUN>        RUN = EGAIN_a10 | EGAIN_a05  (the folder under t2v8f/stage5/ that make_stage5.py built)
# Order: claim the box ATOMICALLY (refuse if BOX2_BUSY.txt exists) -> preconditions on box 2 (stages 2/3/4 done and not running, scene roots, stage-4 files, no earlier pull5_<RUN>, D: free) ->
#        put + verify the overlay tar -> extract + verify every extracted file -> launch -> canary -> wait on the EXACT marker names -> pull to C:/Users/Admin/t2v8f_pulls/stage5_<RUN>/ -> move the busy marker aside.
# Every abort AFTER the claim moves the marker aside (never deletes it). Nothing on box 2 is deleted by this script.
set -u
RUN="${1:?usage: stage5_chain.sh <RUN, e.g. EGAIN_a10>}"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
T=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f
S=$T/stage5/$RUN
TARF=$S/t2v8f_stage5_overlay_$RUN.tar
BR=$S/BUILD_RESULT5_$RUN.json
B=C:/Users/Admin/box2_ship/BOX2_BUSY.txt
PT=C:/Users/Admin/t2v8f_pulls/stage5_$RUN
F='post-quantum|store now|upgraded|pq.html'
R5=D:/t2v8f/pull5_$RUN
ps1() { $SSH "powershell -NoProfile -Command \"$1\"" 2>&1 | grep -vE "$F" | tr -d '\r'; }
aside() { mkdir -p C:/Users/Admin/box2_ship/busy_aside && mv "$B" "C:/Users/Admin/box2_ship/busy_aside/BOX2_BUSY.t2v8f_stage5_${RUN}.$1.$(date +%Y%m%d_%H%M%S).txt"; }
[ -f "$TARF" ] && [ -f "$BR" ] || { echo "ZZ STAGE5-NO-PACKAGE $S ZZ"; exit 2; }
# ---- expected values, from the build's own record (BUILD_RESULT5), never typed by hand
read -r WANT_TAR NEED_BOTH NEED_DEC < <($PY -I -c "
import json,sys
b=json.load(open('$BR'));f=b['footprint']
print(b['tar_sha256'].upper(), f['D_free_gib_needed_to_run_both_passes'], f['D_free_gib_needed_for_deciding_pass_only'])")
# the overlay's files in tar order: 'relpath SHA256UPPER size' lines, MANIFEST5.json last
$PY -I -c "
import json
b=json.load(open('$BR'))
for e in b['overlay_files']: print(e['path'], e['sha256'].upper(), e['size'])
print('MANIFEST5.json', b['manifest5_sha256'].upper(), 0)" | tr -d '\r' > "$S/expect_files.txt"
# ---- 1. claim the box atomically (noclobber: the write FAILS if the file exists)
mkdir -p C:/Users/Admin/box2_ship
( set -C; printf 'owner=T2V8F stage5 %s\nstart=%s\neta=1.8-2.7 h\n' "$RUN" "$(date '+%F %T')" > "$B" ) 2>/dev/null || { echo "ZZ STAGE5-BUSY: $B exists, STOP (read it, do not delete it) ZZ"; cat "$B"; exit 3; }
echo "[$(date +%T)] box claimed for stage 5 $RUN"
# ---- 2. preconditions on box 2
for n in 2 3 4; do
  st=$(ps1 "if (Test-Path D:/t2v8f/pull$n/STAGE$n.done -PathType Leaf) {'MK_DONE'} elseif (Test-Path D:/t2v8f/pull$n/STAGE$n.fail -PathType Leaf) {'MK_FAIL'} else {'NOT_FINISHED'}" | grep -xE 'MK_DONE|MK_FAIL|NOT_FINISHED' | head -1)
  rn=$(ps1 "Test-Path D:/t2v8f/pull$n/STAGE$n.running" | head -1)
  echo "stage$n marker=$st running=$rn"
  [ "$rn" = "False" ] && { [ "$st" = MK_DONE ] || [ "$n" = 2 ]; } || { echo "ZZ STAGE5-NOT-STARTED stage$n marker=$st running=$rn ZZ"; aside precheck; exit 4; }
done
roots=$(ps1 "Test-Path C:/t2v8f/scenes3, D:/t2v8f/scenes2, $R5" | tr '\n' ' ')
[ "$roots" = "True True False " ] || { echo "ZZ STAGE5-NOT-STARTED roots(scenes3,scenes2,pull5_$RUN)=$roots (want True True False; an existing pull5_$RUN would be RESUMED, never mix two runs) ZZ"; aside precheck; exit 4; }
fr=$(ps1 "[math]::Round((Get-PSDrive D).Free/1GB,2)" | tr ',' '.' | grep -E '^[0-9.]+$' | head -1)
echo "free D=$fr GiB (need >= $NEED_BOTH for both passes, >= $NEED_DEC for the deciding pass alone)"
awk -v d="${fr:-0}" -v n="$NEED_BOTH" 'BEGIN{exit !(d>=n)}' || { echo "ZZ STAGE5-DISK-LOW free=$fr need=$NEED_BOTH ZZ"; aside disklow; exit 4; }
# ---- 3. put + verify the tar, extract, verify every extracted file (and the stage-4 files it requires)
sftp -b - tanitad-box2 <<EOS 2>&1 | grep -vE "$F"
put $TARF /D:/t2v8f/t2v8f_stage5_overlay_$RUN.tar
EOS
got=$($SSH "certutil -hashfile D:\t2v8f\t2v8f_stage5_overlay_$RUN.tar SHA256" 2>&1 | grep -vE "$F" | tr -d '\r ' | sed -n 2p | tr 'a-f' 'A-F')
echo "tar want=$WANT_TAR got=$got"
[ -n "$WANT_TAR" ] && [ "$WANT_TAR" = "$got" ] || { echo "ZZ STAGE5-TAR-MISMATCH ZZ"; aside tarmismatch; exit 5; }
$SSH "tar -xf D:\t2v8f\t2v8f_stage5_overlay_$RUN.tar -C D:\t2v8f" 2>&1 | grep -vE "$F"
cmdline=""; want=""
while read -r rel sha size; do
  cmdline="$cmdline (Get-FileHash D:/t2v8f/pkg/$rel).Hash;"; want="$want$sha "
done < "$S/expect_files.txt"
for rel in code/t2v8f_c0c.py code/t2v8f_stage4.py refs/tool/v8f_t1_eval_ri.py code/t2v8f_stage2.py; do       # what stage 5 REQUIRES from stages 2 and 4 (its preflight pins them again)
  cmdline="$cmdline (Get-FileHash D:/t2v8f/pkg/$rel).Hash;"
  want="$want$($PY -I -c "import json;b=json.load(open('$BR'));print(b['requires']['$rel']['sha256'].upper())" | tr -d '\r') "
done
v=$(ps1 "$cmdline" | tr '\n' ' ')
echo "verify want: $want"; echo "verify got : $v"
[ "$v" = "$want" ] || { echo "ZZ STAGE5-VERIFY-MISMATCH ZZ"; aside verify; exit 6; }
# ---- 4. launch (the preflight sha256-checks 186 scene files, the 2 checkpoints, the pinned files; refuses by name)
$SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\t2v8f\pkg\jobs\stage5_$RUN.cmd -Log D:\t2v8f\logs\stage5_$RUN.log" 2>&1 | grep -vE "$F" | tail -2
echo "[$(date +%T)] ZZ STAGE5 LAUNCHED $RUN ZZ"
can=""
for i in $(seq 1 60); do                                          # the preflight hashes ~33 GiB first (~15 min ESTIMATED, as stage 4)
  sleep 60
  pf=$(ps1 "if (Test-Path $R5/preflight.fail -PathType Leaf) {'PF_FAIL'}" | grep -x PF_FAIL)
  [ -n "$pf" ] && { echo "ZZ STAGE5-PREFLIGHT-FAIL ZZ"; ps1 "Get-Content $R5/preflight.fail" | head -20; break; }
  can=$(ps1 "Get-Content $R5/STAGE5.running -ErrorAction SilentlyContinue" | grep -oE 'last=[0-9a-f]{8}[|][^ ]*' | head -1)
  [ -n "$can" ] && break
done
echo "[$(date +%T)] canary: ${can:-NONE} (a FAIL here does not stop the waiter; the read applies R-SCN)"
st=""
for i in $(seq 1 360); do                                         # EXACT marker names: the driver renames an old STAGE5.done to STAGE5.done.prev, a listing pattern would match it
  st=$(ps1 "if (Test-Path $R5/STAGE5.done -PathType Leaf) {'MK_DONE'} elseif (Test-Path $R5/STAGE5.fail -PathType Leaf) {'MK_FAIL'}" | grep -xE 'MK_(DONE|FAIL)' | head -1)
  [ -n "$st" ] && break
  sleep 60
done
echo "[$(date +%T)] stage 5 marker=${st:-TIMEOUT}"
[ -n "$st" ] || { echo "ZZ STAGE5-WAIT-TIMEOUT (the box stays CLAIMED: a job may still be running) ZZ"; exit 9; }
# ---- 5. pull into its OWN folder, then release the box
[ -e "$PT" ] && { echo "ZZ PULL TARGET EXISTS $PT (box stays claimed) ZZ"; exit 3; }
mkdir -p "$PT"
sftp -b - tanitad-box2 <<EOS 2>&1 | grep -vE "$F" | tail -2
get -r /D:/t2v8f/pull5_$RUN $PT
get /D:/t2v8f/logs/stage5_$RUN.log $PT/stage5_$RUN.log
EOS
aside pulled
echo "ZZ STAGE5 PULLED $st $RUN -> $PT ZZ"
