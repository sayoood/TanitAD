#!/usr/bin/env bash
# PREREG_RA1_FULL Amendment 2 on box 2 (IDM at gamma 200, the committed IDM-FAIL lever), one job at a time under SAFE MODE.
# ARMED ONLY ON THE PI's GO: 2026-10-20, "Continue with the ra-1's committed IDM."
#   0. wait for Amendment 1's ZZRA1A1-DONE (stop on its ZZRA1A1-STOP); AFTER_OUT/AFTER_DONE/AFTER_STOP override the predecessor
#   1. refresh box 2's code from the REMOTE tip (stack/, taniteval/, the package); sha-check the key files against the tip's blobs
#   2. full5 3000 0.08 200: A1 x2 at level-1 gamma 200 (gamma_up the trainer default 50), into st3000 beside Amendment 1's runs
#   3. pull the 2 new runs; code/ra1_full_amend2.py reads IDM-200 over them and the A0 / A1 (gamma 50) runs already pulled -> amend2_final.json
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4) or any nonzero rc. Script polling, not model turns.
# Temperature: the chain restarts telemetry, STOPS if no fresh sample arrives, and starts ra1runs_guard.sh (RA1_CHAIN_OUT = this log)
# itself, exactly as Amendment 1's chain did; the guard knows the ZZRA1A2-DONE / ZZRA1A2-STOP markers.
# Same as chain_ra1_full4.sh except: the predecessor, the markers, the stage, the readers' extract (stack/ too: ra1_full_amend2.py
# imports stack/scripts/ra1_hjepa_eval.py for the paired family block) and a fresh extract directory instead of rm -rf.
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/ra1_full; mkdir -p $E/runs
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
PKGP="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
S=3000; BETA=0.08; GAMMA=200
AFTER_OUT=${AFTER_OUT:-$L/chain_ra1_full4.out}; AFTER_DONE=${AFTER_DONE:-ZZRA1A1-DONE}; AFTER_STOP=${AFTER_STOP:-ZZRA1A1-STOP}
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZRA1A2-STOP $*"; rm -f $B/BOX2_BUSY.txt; exit 1; }
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
  grep -q "$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZRA1A2-STOP the predecessor stopped: $(grep "$AFTER_STOP" "$AFTER_OUT" | tail -1)"; exit 1; }
  sleep 300
done
[ -f $B/BOX2_BUSY.txt ] && { echo "ZZRA1A2-STOP busy marker present: $(cat $B/BOX2_BUSY.txt)"; exit 1; }
for a in A0_b${BETA}_s0 A0_b${BETA}_s1 A1_b${BETA}g50_s0 A1_b${BETA}g50_s1; do
  [ -f "$E/runs/st$S/$a/eval/perwindow.npz" ] || { echo "ZZRA1A2-STOP the reader's pulled run $a is missing locally"; exit 1; }
done
echo "ra1_full Amendment 2 (PREREG_RA1_FULL, IDM at gamma 200) since $(date '+%m-%d %H:%M'), ~2 h. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
git -C $MIRROR show "$T:$PKGP/code/ra1_box2_queue.cmd" | grep -q '^:full5' || stop "the tip's queue has no full5 stage"
git -C $MIRROR cat-file -e "$T:$PKGP/code/ra1_full_amend2.py" || stop "the tip has no ra1_full_amend2.py"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_full5_tip.tar $T -- stack taniteval "$PKGP" || stop "archive"
$SCP $B/repo_full5_tip.tar tanitad-box2:D:/tanitad/repo_full5_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_full5_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_FULL5.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/ra1full5_refresh.cmd
$SCP $B/ra1full5_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch ra1full5_refresh || stop "launch refresh"
waitrc ra1full5_refresh 40
for f in stack/scripts/ra1_hjepa_train.py stack/scripts/ra1_hjepa_eval.py stack/tanitad/models/ra1_hjepa.py stack/tanitad/data/ra1_pooled.py \
         "$PKGP/code/ra1_box2_queue.cmd" "$PKGP/raw/p10_yaw4512/p10_prior.pt"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  r=$($SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm SHA256 -LiteralPath 'D:/tanitad/repo/$f').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{64}$')
  [ "$l" = "$r" ] || stop "box 2 $f != tip $T"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (6 key files sha-checked)"
# the reader runs HERE at the same tip: stack + taniteval + the package's code dir, from the archive box 2 got (its parents[5] root)
X=$B/tip5_$(date +%Y%m%d%H%M%S); mkdir -p $X && tar -xf $B/repo_full5_tip.tar -C $X stack taniteval "$PKGP/code" || stop "extract the reader"
RD="$(cygpath -m "$X/$PKGP/code")"; [ -f "$RD/ra1_full_amend2.py" ] || stop "the extract has no ra1_full_amend2.py"
# ---- telemetry + the USB-HDD temperature guard, exactly as Amendment 1's chain (its end left telemetry STOPPED):
#      no fresh samples => STOP, because the guard would be blind
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF '|ZZRA1A2)-(DONE|STOP)' $B/ra1runs_guard.sh || stop "ra1runs_guard.sh does not know this chain's markers (anchored): it would never exit"
RA1_CHAIN_OUT=$L/chain_ra1_full5.out bash $B/ra1runs_guard.sh > $L/ra1a2_guard.out 2>&1 &
# ---- 2. Amendment 2's runs (no queue verdict)
$PY "$(cygpath -m $B)/mk_ra1_job.py" ra1full_p5 full5 $S $BETA $GAMMA >/dev/null && $SCP $B/ra1full_p5.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship ra1full_p5"
launch ra1full_p5 || stop "launch ra1full_p5"
waitrc ra1full_p5 600
for a in A1_b${BETA}g${GAMMA}_s0 A1_b${BETA}g${GAMMA}_s1; do pullrun "st$S/$a"; done
# ---- 3. the reading
PYTHONIOENCODING=utf-8 $PY "$RD/ra1_full_amend2.py" --root "$(cygpath -m $E)" --S $S --out "$(cygpath -m $E/amend2_final.json)" || stop "amend2 read"
$SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
$SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_amend2.csv 2>/dev/null
rm -f $B/BOX2_BUSY.txt
echo "ZZRA1A2-DONE $(date '+%m-%d %H:%M') tip $T: $(PYTHONIOENCODING=utf-8 $PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))['reading'])" "$(cygpath -m $E/amend2_final.json)") (boot unchanged $BOOT0)"
