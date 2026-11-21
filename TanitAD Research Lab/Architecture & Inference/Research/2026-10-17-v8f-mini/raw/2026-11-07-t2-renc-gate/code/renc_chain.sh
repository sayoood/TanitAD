#!/usr/bin/env bash
# RENC (PREREG_T2_RENC.md) on box 2: modes data | train | gate, run in order under ONE claim of the box. Modelled on gap48/gap48_chain.sh (same two busy markers, same refusals, exact marker waits).
# usage: RENC_MODES=data bash renc_chain.sh          (default: data)     RENC_MODES=train,gate  /  RENC_MODES=data,train,gate
# env:   RENC_BUILD=<build dir>, RENC_RESUME=1 (continue an existing data/out folder; a leftover .fail marker is moved aside on box 2), RENC_SSH / RENC_SFTP / RENC_BUSY_MARKER / RENC_ASIDE_DIR / RENC_POLL (tests only)
# train is ARMED by a file on the dev box (renc/GO_TRAIN.txt, written by hand once the CPU tests are green). If data finishes and the file is absent, the box is released and the chain stops (exit 7).
# Every abort AFTER the claim moves OUR marker(s) aside (never deletes; never touches a marker this script did not create). Nothing on box 2 is deleted by this script.
# Exit codes: 2 no package, 3 busy, 4 precondition, 5 hash, 6 verify, 7 train not armed, 8 a job failed (small files pulled), 9 wait timeout (the box stays CLAIMED: a job may still be running).
set -u
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
SSH="${RENC_SSH:-ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2}"
SFTP="${RENC_SFTP:-sftp -b - tanitad-box2}"
T=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f
RD=$T/renc
BD="${RENC_BUILD:-$RD/build}"
TARF=$BD/t2v8f_renc.tar
BR=$BD/BUILD_RESULT_RENC.json
B="${RENC_BUSY_MARKER:-C:/Users/Admin/box2_ship/BOX2_BUSY.txt}"
ASIDE="${RENC_ASIDE_DIR:-C:/Users/Admin/box2_ship/busy_aside}"
B2=D:/t2v8f/BOX2_BUSY.txt
DATA=D:/t2v8f/renc/data
OUT=D:/t2v8f/renc/out
POLL="${RENC_POLL:-30}"
MODES="${RENC_MODES:-data}"
F='post-quantum|store now|upgraded|pq.html'
BS='\'
B2OWN=0
TS=$(date +%Y%m%d_%H%M%S)
ps1() { $SSH "powershell -NoProfile -Command \"$1\"" 2>&1 | grep -vE "$F" | tr -d '\r'; }
aside() {   # $1 = reason. Moves ONLY the markers this run created.
  mkdir -p "$ASIDE" && mv "$B" "$ASIDE/BOX2_BUSY.t2v8f_renc.$1.$(date +%Y%m%d_%H%M%S).txt"
  if [ "$B2OWN" = 1 ]; then ps1 "New-Item -ItemType Directory -Force D:/t2v8f/busy_aside | Out-Null; Move-Item $B2 D:/t2v8f/busy_aside/BOX2_BUSY.renc.$1.$TS.txt" >/dev/null; fi
}
[ -f "$TARF" ] && [ -f "$BR" ] && [ -f "$BD/expect_files.txt" ] && [ -f "$BD/expect_external.txt" ] || { echo "ZZ RENC-NO-PACKAGE $BD ZZ"; exit 2; }
for M in ${MODES//,/ }; do case $M in data|train|gate) ;; *) echo "ZZ RENC-BAD-MODE $M ZZ"; exit 2;; esac; done
read -r WANT_TAR < <($PY -I -c "
import json
print(json.load(open('$BR'))['tar_sha256'].upper())" | tr -d '\r')                      # Windows Python ends the line with CRLF: strip it before comparing
FIRST=${MODES%%,*}
# ---- 1. claim the box atomically (noclobber: the write FAILS if the file exists)
mkdir -p "$(dirname "$B")" "$RD/logs"
( set -C; printf 'owner=T2V8F renc %s\nstart=%s\neta=data 0.6-0.9 h; train 3-9 h; gate 0.2 h\n' "$MODES" "$(date '+%F %T')" > "$B" ) 2>/dev/null || { echo "ZZ RENC-BUSY: $B exists, STOP (read it, do not delete it) ZZ"; cat "$B"; exit 3; }
echo "[$(date +%T)] box claimed for renc $MODES (dev-box marker)"
c2=$(ps1 "try { New-Item -ItemType File -Path $B2 -ErrorAction Stop | Out-Null; Set-Content -Path $B2 -Value 'owner=T2V8F renc'; 'B2CLAIMED' } catch { 'B2EXISTS' }" | grep -xE 'B2CLAIMED|B2EXISTS' | head -1)
if [ "$c2" = B2CLAIMED ]; then B2OWN=1; else echo "ZZ RENC-BUSY: $B2 exists on box 2 (or the claim could not be made: '$c2'), STOP ZZ"; aside box2busy; exit 3; fi
# ---- 2. preconditions on box 2
run=$(ps1 "Get-ChildItem D:/t2v8f/pull*/*.running, D:/t2v8f/renc/*/*.running -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName" | grep -c running)
[ "$run" = "0" ] || { echo "ZZ RENC-NOT-STARTED: $run *.running marker(s) under D:/t2v8f/pull* or renc/* (another job may be alive) ZZ"; ps1 "Get-ChildItem D:/t2v8f/pull*/*.running, D:/t2v8f/renc/*/*.running | Select-Object -ExpandProperty FullName"; aside precheck; exit 4; }
roots=$(ps1 "Test-Path C:/t2v8f/scenes3, D:/t2v8f/scenes2, D:/t2v8f/pkg/code/t2v8f_stage1.py, D:/t2a/box2_kit/scenes, D:/tanitad/launch.ps1, D:/t2v8f/gap48/code/t2v8f_gap48.py, D:/t2v8f/gap48/MANIFEST_GAP48.json, $DATA" | tr '\n' ' ')
want="True True True True True True True False "
{ [ "$FIRST" != data ] || [ "${RENC_RESUME:-0}" = 1 ]; } && want="True True True True True True True True "
[ "$roots" = "$want" ] || { echo "ZZ RENC-NOT-STARTED roots(scenes3,scenes2,stage1 code,kit scenes,launch.ps1,gap48 code,gap48 manifest,renc/data)=$roots (want $want; an existing renc/data is only resumed with RENC_RESUME=1) ZZ"; aside precheck; exit 4; }
if [ "$FIRST" = train ]; then
  d=$(ps1 "if (Test-Path $DATA/RENC_DATA.done -PathType Leaf) {'DATA_DONE'} else {'DATA_MISSING'}" | grep -xE 'DATA_DONE|DATA_MISSING' | head -1)
  [ "$d" = DATA_DONE ] || { echo "ZZ RENC-NOT-STARTED: $DATA/RENC_DATA.done is missing, train has no data ZZ"; aside nodata; exit 4; }
fi
NEED=6.0; [ "$FIRST" = data ] || NEED=3.0
fr=$(ps1 "[math]::Round((Get-PSDrive D).Free/1GB,2)" | tr ',' '.' | grep -E '^[0-9.]+$' | head -1)
echo "free D=$fr GiB (need >= $NEED)"
awk -v d="${fr:-0}" -v n="$NEED" 'BEGIN{exit !(d>=n)}' || { echo "ZZ RENC-DISK-LOW free=$fr need=$NEED ZZ"; aside disklow; exit 4; }
# ---- 3. put + verify the code tar, extract, verify its files and the GAP48 files it imports
$SFTP <<EOS 2>&1 | grep -vE "$F"
put $TARF /D:/t2v8f/t2v8f_renc.tar
EOS
got=$($SSH "certutil -hashfile D:${BS}t2v8f${BS}t2v8f_renc.tar SHA256" 2>&1 | grep -vE "$F" | tr -d '\r ' | sed -n 2p | tr 'a-f' 'A-F')
echo "tar want=$WANT_TAR got=$got"
[ -n "$WANT_TAR" ] && [ "$WANT_TAR" = "$got" ] || { echo "ZZ RENC-TAR-MISMATCH ZZ"; aside tarmismatch; exit 5; }
$SSH "tar -xf D:${BS}t2v8f${BS}t2v8f_renc.tar -C D:${BS}t2v8f" 2>&1 | grep -vE "$F"
verify() {  # $1 = list file, $2 = box-2 base dir
  local cmdline="" want="" rel sha size
  while read -r rel sha size; do cmdline="$cmdline (Get-FileHash $2/$rel).Hash;"; want="$want$sha "; done < "$1"
  local v; v=$(ps1 "$cmdline" | tr '\n' ' ')
  echo "verify want: $want"; echo "verify got : $v"
  [ "$v" = "$want" ]
}
verify "$BD/expect_files.txt" D:/t2v8f/renc || { echo "ZZ RENC-VERIFY-MISMATCH (shipped files) ZZ"; aside verify; exit 6; }
verify "$BD/expect_external.txt" D:/t2v8f || { echo "ZZ RENC-VERIFY-MISMATCH (GAP48 files already on box 2) ZZ"; aside verifyext; exit 6; }
# ---- 4. the modes, in order
pull_small() {   # $1 = mode; into $PT (set below)
  case $1 in
    data)  $SFTP <<EOS 2>&1 | grep -vE "$F" | tail -3
-get /$DATA/RENC_DATA_RESULT.json $PT/RENC_DATA_RESULT.json
-get /$DATA/RENC_DATA.done $PT/RENC_DATA.done
-get /$DATA/RENC_DATA.fail $PT/RENC_DATA.fail
EOS
    ;;
    train) $SFTP <<EOS 2>&1 | grep -vE "$F" | tail -3
-get /$OUT/TRAIN_RESULT_A.json $PT/TRAIN_RESULT_A.json
-get /$OUT/TRAIN_RESULT_B.json $PT/TRAIN_RESULT_B.json
-get /$OUT/RENC_TRAIN.done $PT/RENC_TRAIN.done
-get /$OUT/RENC_TRAIN.fail $PT/RENC_TRAIN.fail
-get /$OUT/E_A/E_A_lora_blob.pt $PT/E_A_lora_blob.pt
-get /$OUT/E_B/E_B_lora_blob.pt $PT/E_B_lora_blob.pt
EOS
    ;;
    gate)  mkdir -p "$PT/data_meta"; $SFTP <<EOS 2>&1 | grep -vE "$F" | tail -3
-get /$OUT/G_ID0.json $PT/G_ID0.json
-get /$OUT/RENC_GATE_ENCODE.done $PT/RENC_GATE_ENCODE.done
-get /$OUT/RENC_GATE_ENCODE.fail $PT/RENC_GATE_ENCODE.fail
-get -r /$OUT/gate_tokens $PT/gate_tokens
-get /$DATA/*.meta.pt $PT/data_meta/
EOS
    ;;
  esac
}
PB=${RENC_PULL_BASE:-C:/Users/Admin/t2v8f_pulls/renc}
cfree=$($PY -I -c "import shutil;print(int(shutil.disk_usage('C:/').free//10**6))" | tr -d '\r')
[ "$cfree" -ge 1500 ] || PB=F:/TanitAD-scratch/t2v8f_pulls/renc
PT=$PB; [ -e "$PT" ] && PT="${PB}_$TS"
mkdir -p "$PT"
for M in ${MODES//,/ }; do
  case $M in
    data)  JOB=renc_data;  DIR=$DATA; NAME=RENC_DATA;        POLLS=${RENC_POLLS_DATA:-240};;
    train) JOB=renc_train; DIR=$OUT;  NAME=RENC_TRAIN;       POLLS=${RENC_POLLS_TRAIN:-1200}
           if [ ! -f "$RD/GO_TRAIN.txt" ]; then echo "ZZ RENC-TRAIN-NOT-ARMED: $RD/GO_TRAIN.txt is absent (written once the CPU tests are green); the box is released ZZ"; aside notarmed; exit 7; fi;;
    gate)  JOB=renc_gate;  DIR=$OUT;  NAME=RENC_GATE_ENCODE; POLLS=${RENC_POLLS_GATE:-120}
           d=$(ps1 "if (Test-Path $OUT/RENC_TRAIN.done -PathType Leaf) {'TRAIN_DONE'} else {'TRAIN_MISSING'}" | grep -xE 'TRAIN_DONE|TRAIN_MISSING' | head -1)
           [ "$d" = TRAIN_DONE ] || { echo "ZZ RENC-GATE-NOT-STARTED: $OUT/RENC_TRAIN.done is missing ZZ"; aside notrain; exit 4; };;
  esac
  if [ "${RENC_RESUME:-0}" = 1 ]; then ps1 "if (Test-Path $DIR/$NAME.fail -PathType Leaf) { Move-Item $DIR/$NAME.fail $DIR/$NAME.fail.aside_$TS }" >/dev/null; fi
  LOG="D:${BS}t2v8f${BS}logs${BS}${JOB}_$TS.log"
  $SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:${BS}tanitad${BS}launch.ps1 -Job D:${BS}t2v8f${BS}renc${BS}jobs${BS}$JOB.cmd -Log $LOG" 2>&1 | grep -vE "$F" | tail -2
  echo "[$(date +%T)] ZZ RENC $M LAUNCHED ZZ  log=$LOG"
  st=""
  for i in $(seq 1 "$POLLS"); do                                    # EXACT marker names: the driver renames an old marker to <name>.prev, a listing pattern would match it
    st=$(ps1 "if (Test-Path $DIR/$NAME.done -PathType Leaf) {'MK_DONE'} elseif (Test-Path $DIR/$NAME.fail -PathType Leaf) {'MK_FAIL'} elseif (Select-String -Path $LOG -Pattern ZZJOB-END -Quiet -ErrorAction SilentlyContinue) {'MK_FAIL'}" | grep -xE 'MK_(DONE|FAIL)' | head -1)
    [ -n "$st" ] && break
    if [ $((i % 10)) = 0 ] && [ "$M" = data ]; then echo "[$(date +%T)] data: $(ps1 "(Get-ChildItem $DATA -Filter *.meta.pt).Count" | grep -E '^[0-9]+$' | head -1)/48 scenes"; fi
    sleep "$POLL"
  done
  echo "[$(date +%T)] $M marker=${st:-TIMEOUT}"
  [ -n "$st" ] || { echo "ZZ RENC-WAIT-TIMEOUT mode=$M (the box stays CLAIMED: a job may still be running) ZZ"; exit 9; }
  pull_small "$M"
  echo "$M $st $PT" > "$RD/logs/CHAIN_RENC_$M.$st"
  if [ "$st" != MK_DONE ]; then echo "ZZ RENC-JOB-FAILED mode=$M, small files pulled to $PT ZZ"; aside "jobfail_$M"; exit 8; fi
done
aside pulled
echo "ZZ RENC PULLED modes=$MODES -> $PT ZZ"
