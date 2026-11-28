#!/usr/bin/env bash
# PREREG_RA1_FULL Amendment 1 on box 2 (the K-PASS lever), one job at a time under SAFE MODE. ARMED ONLY ON THE PI's GO.
#   0. wait for the rung-beta chain's ZZBETA-DONE (stop on its ZZBETA-STOP); AFTER_OUT/AFTER_DONE/AFTER_STOP override the predecessor
#   1. refresh box 2's code from the REMOTE tip (stack/, taniteval/, the package); sha-check the key files against the tip's blobs
#   2. full4 3000 0.08 50: A0 x2, A2 x2, A1x seed 1 at 3000 (in st3000, beside phase 1/2's runs) + the queue verdict over 9 runs
#   3. pull the 5 new runs + full4_verdict.json; code/ra1_full_amend1.py reads Amendment 1 -> amend1_final.json
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4) or any nonzero rc. Script polling, not model turns.
# Temperature: the chain restarts telemetry, STOPS if no fresh sample arrives, and starts ra1runs_guard.sh (RA1_CHAIN_OUT = this log)
# itself, as the v8F-mini and beta chains do. PI go 2026-10-19: "after rung beta finishes, same safe mode and stop rules, with the
# temperature guard".
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/ra1_full; mkdir -p $E/runs
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
PKGP="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
S=3000; BETA=0.08; GAMMA=50
AFTER_OUT=${AFTER_OUT:-$L/chain_beta.out}; AFTER_DONE=${AFTER_DONE:-ZZBETA-DONE}; AFTER_STOP=${AFTER_STOP:-ZZBETA-STOP}
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZRA1A1-STOP $*"; rm -f $B/BOX2_BUSY.txt; exit 1; }
waitrc() {   # waitrc <job> <polls>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; stop "$1 rc=$rc"; }
}
pullrun() {  # pullrun <run dir under runs\ra1_yaw_full>
  local r="D:/tanitad/runs/ra1_yaw_full/$1" d="$E/runs/$1"; mkdir -p "$d/eval"
  for f in launch_check.json meta.json train_log.jsonl train.log eval.log; do $SCP "tanitad-box2:$r/$f" "$d/" || stop "pull $r/$f"; done
  for f in eval.json perwindow.npz; do $SCP "tanitad-box2:$r/eval/$f" "$d/eval/" || stop "pull $r/eval/$f"; done
}
echo "$(date '+%m-%d %H:%M') waiting for $AFTER_DONE in $AFTER_OUT"
until grep -q "$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do
  grep -q "$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZRA1A1-STOP the predecessor stopped: $(grep "$AFTER_STOP" "$AFTER_OUT" | tail -1)"; exit 1; }
  sleep 300
done
[ -f $B/BOX2_BUSY.txt ] && { echo "ZZRA1A1-STOP busy marker present: $(cat $B/BOX2_BUSY.txt)"; exit 1; }
echo "ra1_full Amendment 1 (PREREG_RA1_FULL, the K-PASS lever) since $(date '+%m-%d %H:%M'), ~4 h. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_full4_tip.tar $T -- stack taniteval "$PKGP" || stop "archive"
$SCP $B/repo_full4_tip.tar tanitad-box2:D:/tanitad/repo_full4_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_full4_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_FULL4.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/ra1full4_refresh.cmd
$SCP $B/ra1full4_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch ra1full4_refresh || stop "launch refresh"
waitrc ra1full4_refresh 40
for f in stack/scripts/ra1_hjepa_train.py stack/scripts/ra1_hjepa_eval.py stack/tanitad/models/ra1_hjepa.py stack/tanitad/data/ra1_pooled.py \
         "$PKGP/code/ra1_box2_queue.cmd" "$PKGP/raw/p10_yaw4512/p10_prior.pt"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  r=$($SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm SHA256 -LiteralPath 'D:/tanitad/repo/$f').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{64}$')
  [ "$l" = "$r" ] || stop "box 2 $f != tip $T"
done
git -C $MIRROR show "$T:$PKGP/code/ra1_box2_queue.cmd" | grep -q '^:full4' || stop "the tip's queue has no full4 stage"
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (6 key files sha-checked)"
# the readers run HERE at the same tip: taniteval + the package's code dir, from the archive box 2 got (the reader's parents[5] fallback)
rm -rf $B/tip4 && mkdir -p $B/tip4 && tar -xf $B/repo_full4_tip.tar -C $B/tip4 taniteval "$PKGP/code" || stop "extract the readers"
RD="$(cygpath -m "$B/tip4/$PKGP/code")"; [ -f "$RD/ra1_full_amend1.py" ] || stop "the tip has no ra1_full_amend1.py"
# ---- telemetry + the USB-HDD temperature guard, exactly as the v8F-mini and beta chains do (beta's end leaves telemetry STOPPED):
#      no fresh samples => STOP, because the guard would be blind
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
RA1_CHAIN_OUT=$L/chain_ra1_full4.out bash $B/ra1runs_guard.sh > $L/ra1a1_guard.out 2>&1 &
# ---- 2. Amendment 1's runs + the queue verdict
$PY "$(cygpath -m $B)/mk_ra1_job.py" ra1full_p4 full4 $S $BETA $GAMMA >/dev/null && $SCP $B/ra1full_p4.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship ra1full_p4"
launch ra1full_p4 || stop "launch ra1full_p4"
waitrc ra1full_p4 900
for a in A0_b${BETA}_s0 A0_b${BETA}_s1 A2_b${BETA}g${GAMMA}_s0 A2_b${BETA}g${GAMMA}_s1 A1x_b${BETA}g${GAMMA}_s1; do pullrun "st$S/$a"; done
$SCP "tanitad-box2:D:/tanitad/runs/ra1_yaw_full/st$S/full4_verdict.json" $E/ || stop "pull full4_verdict"
# ---- 3. the readings
PYTHONIOENCODING=utf-8 $PY "$RD/ra1_full_amend1.py" --root "$(cygpath -m $E)" --verdict "$(cygpath -m $E/full4_verdict.json)" \
  --S $S --out "$(cygpath -m $E/amend1_final.json)" || stop "amend1 read"
$SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
$SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_amend1.csv 2>/dev/null
rm -f $B/BOX2_BUSY.txt
echo "ZZRA1A1-DONE $(date '+%m-%d %H:%M') tip $T: $(PYTHONIOENCODING=utf-8 $PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))['reading'])" "$(cygpath -m $E/amend1_final.json)") (boot unchanged $BOOT0)"
