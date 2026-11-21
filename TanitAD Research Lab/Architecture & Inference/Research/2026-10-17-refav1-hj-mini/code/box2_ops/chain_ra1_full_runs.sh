#!/usr/bin/env bash
# PREREG_RA1_FULL phases 1-3 on box 2, run automatically after the build (Rule Zero), one job at a time under SAFE MODE:
#   0. wait for chain_ra1_full.sh's ZZRA1FULL-BUILD-DONE (stop on its ZZRA1FULL-STOP)
#   1. refresh box 2's code from the REMOTE tip (stack/, taniteval/, the two packages): every run is attributable to a landed commit;
#      key files are then sha-checked against the tip's blobs
#   2. phase 1 (full) -> holdout selection -> at most one edge extension per direction (then select --extended)
#   3. phase 2 (full2) -> read: the A1x gate, then K
#   4. K FAIL -> phase 3 (full3, --wd 3.0) -> read --phase3-root
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4) or any nonzero rc. Script polling, not model turns.
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/ra1_full; mkdir -p $E/runs
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
PKGP="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
GRID="500 1000 3000 6000"
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZRA1FULLRUNS-STOP $*"; exit 1; }
waitrc() {   # waitrc <job> <polls>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; stop "$1 rc=$rc"; }
}
queue() {    # queue <job> <polls> <queue args...>
  local j=$1 m=$2; shift 2
  $PY "$(cygpath -m $B)/mk_ra1_job.py" $j "$@" >/dev/null && $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship $j"
  launch $j || stop "launch $j"
  waitrc $j $m
}
pullrun() {  # pullrun <box-2 run dir under runs\ra1_yaw_full> <local dir>
  local r="D:/tanitad/runs/ra1_yaw_full/$1" d="$E/runs/$2"; mkdir -p "$d/eval"
  for f in launch_check.json meta.json train_log.jsonl train.log eval.log DONE; do $SCP "tanitad-box2:$r/$f" "$d/" || stop "pull $r/$f"; done
  for f in eval.json perwindow.npz; do $SCP "tanitad-box2:$r/eval/$f" "$d/eval/" || stop "pull $r/eval/$f"; done
}
pullgrid() { # pullgrid "<S list>"
  for S in $1; do for a in R0_s0 R0_s1 A1_b0.08g50_s0 A1_b0.08g50_s1; do pullrun "st$S/$a" "st$S/$a"; done; done
}
echo "$(date '+%m-%d %H:%M') waiting for the build chain"
until grep -q ZZRA1FULL-BUILD-DONE $L/chain_ra1_full.out 2>/dev/null; do
  grep -q ZZRA1FULL-STOP $L/chain_ra1_full.out 2>/dev/null && stop "the build chain stopped: $(grep ZZRA1FULL-STOP $L/chain_ra1_full.out)"
  sleep 120
done
[ -f $B/BOX2_BUSY.txt ] && stop "busy marker present: $(cat $B/BOX2_BUSY.txt)"
echo "ra1_full runs (session bfe78918, PREREG_RA1_FULL phases 1-3) since $(date '+%m-%d %H:%M'), ~6-10 h. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_full_tip.tar $T -- stack taniteval "$PKGP" \
  "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini" || stop "archive"
$SCP $B/repo_full_tip.tar tanitad-box2:D:/tanitad/repo_full_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_full_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_FULL.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/ra1full_refresh.cmd
$SCP $B/ra1full_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch ra1full_refresh || stop "launch refresh"
waitrc ra1full_refresh 40
for f in stack/scripts/ra1_hjepa_train.py stack/scripts/ra1_hjepa_eval.py stack/tanitad/models/ra1_hjepa.py stack/tanitad/data/ra1_pooled.py \
         "$PKGP/code/ra1_box2_queue.cmd" "$PKGP/raw/p10_yaw4512/p10_prior.pt"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  r=$($SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm SHA256 -LiteralPath 'D:/tanitad/repo/$f').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{64}$')
  [ "$l" = "$r" ] || stop "box 2 $f != tip $T"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (6 key files sha-checked)"
git -C $MIRROR show "$T:$PKGP/code/ra1_full_verdict.py" > $E/ra1_full_verdict.py || stop "verdict script"
V="$(cygpath -m $E/ra1_full_verdict.py)"
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch (the runs go on without it)"
# ---- 2. phase 1 + selection
queue ra1full_p1 1800 full "\"$GRID\"" 0.08 50
pullgrid "$GRID"
$PY "$V" select --root "$(cygpath -m $E/runs)" --budgets $GRID --out "$(cygpath -m $E/full1.json)" || stop "select"
NEXT=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['next'])" "$(cygpath -m $E/full1.json)")
echo "$(date '+%m-%d %H:%M') phase 1: $NEXT"
if [[ $NEXT == EXTEND* ]]; then
  EXT=$(echo "$NEXT" | tr ' ' '\n' | grep ':' | cut -d: -f2 | sort -un | tr '\n' ' ')
  queue ra1full_p1x 1800 full "\"${EXT% }\"" 0.08 50
  pullgrid "$EXT"
  $PY "$V" select --root "$(cygpath -m $E/runs)" --budgets $GRID $EXT --extended --out "$(cygpath -m $E/full1.json)" || stop "select --extended"
  NEXT=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['next'])" "$(cygpath -m $E/full1.json)")
  echo "$(date '+%m-%d %H:%M') phase 1 extended ($EXT): $NEXT"
fi
[[ $NEXT == full2* ]] || stop "phase 1 did not reach phase 2: $NEXT"
set -- $NEXT; SA=$2; SR=$3
# ---- 3. phase 2: the gate, then K
queue ra1full_p2 600 full2 $SA $SR 0.08 50
pullrun "st$SA/A1x_b0.08g50_s0" "st$SA/A1x_b0.08g50_s0"
$SCP "tanitad-box2:D:/tanitad/runs/ra1_yaw_full/st$SA/full2_verdict.json" $E/ || stop "pull full2_verdict"
$PY "$V" read --full1 "$(cygpath -m $E/full1.json)" --verdict "$(cygpath -m $E/full2_verdict.json)" --out "$(cygpath -m $E/full_final.json)" || stop "read"
OUT=$($PY -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['outcome'])" "$(cygpath -m $E/full_final.json)")
echo "$(date '+%m-%d %H:%M') phase 2: $OUT"
# ---- 4. K FAIL => full3 (--wd 3.0), read under the same readings
if [ "$OUT" = K_FAIL ]; then
  queue ra1full_p3 600 full3 $SA $SR 0.08 50 3.0
  for a in A1_b0.08g50_s0 A1_b0.08g50_s1 A1x_b0.08g50_s0; do pullrun "st${SA}_wd3.0/$a" "st${SA}_wd3.0/$a"; done
  $SCP "tanitad-box2:D:/tanitad/runs/ra1_yaw_full/st${SA}_wd3.0/full3_verdict.json" $E/ || stop "pull full3_verdict"
  $PY "$V" read --full1 "$(cygpath -m $E/full1.json)" --verdict "$(cygpath -m $E/full3_verdict.json)" \
    --phase3-root "$(cygpath -m $E/runs/st${SA}_wd3.0)" --out "$(cygpath -m $E/full_final3.json)" || stop "read phase 3"
  OUT="$OUT; phase3 $($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['outcome'])" "$(cygpath -m $E/full_final3.json)")"
fi
$SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
$SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_runs.csv 2>/dev/null
rm -f $B/BOX2_BUSY.txt
echo "ZZRA1FULLRUNS-DONE $(date '+%m-%d %H:%M') tip $T outcome: $OUT (boot unchanged $BOOT0)"
