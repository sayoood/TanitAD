#!/usr/bin/env bash
# PREREG_RA1_ES Addendum D: A1a b0.08g50 seed 0 at S1 (250 steps) then S2 (1000 steps, wd 3.0) -- the a1a stage, box 2, SAFE MODE,
# one job at a time, stop on a reboot (waitjob boot guard) or any nonzero rc. Pulls everything except the .pt checkpoints.
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/a1a; mkdir -p $E
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
ship() { $PY "$(cygpath -m $B)/mk_ra1_job.py" "$@" >/dev/null && scp -q -o BatchMode=yes -o ConnectTimeout=20 $B/$1.cmd tanitad-box2:D:/tanitad/jobs/; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date +%H:%M) $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
run() {
  local j=$1 m=$2; shift 2
  ship $j "$@" || { echo "ZZRA1A1A-STOP ship $j"; exit 1; }
  launch $j || { echo "ZZRA1A1A-STOP launch $j"; exit 1; }
  bash $B/waitjob.sh $j $m "$BOOT0"; local w=$?
  [ $w = 0 ] || { echo "ZZRA1A1A-STOP $j wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"; exit 1; }
  local rc=$(clean $j | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date +%H:%M) $j rc=$rc"
  [ "$rc" = 0 ] || { echo "ZZRA1A1A-STOP $j rc=$rc"; clean $j | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -5; exit 1; }
}
pull() { mkdir -p "$E/$2"; scp -q -o BatchMode=yes -o ConnectTimeout=20 "tanitad-box2:$1" "$E/$2/" || { echo "ZZRA1A1A-STOP pull $1"; exit 1; }; }
BOOT0=$(boot); [ -n "$BOOT0" ] || { echo "ZZRA1A1A-STOP box 2 unreachable"; exit 1; }
echo "$(date +%H:%M) boot0=$BOOT0"
for spec in "st250 ra1a1a_s250 a1a 250 0.08 50" "st1000_wd3.0 ra1a1a_s1000wd3 a1a 1000 0.08 50 3.0"; do
  set -- $spec; root=$1; job=$2; shift 2
  run $job 90 "$@"
  R=D:/tanitad/runs/ra1_yaw_es/$root/A1a_b0.08g50_s0
  for f in launch_check.json meta.json train_log.jsonl train.log eval.log DONE; do pull "$R/$f" "$root/A1a_b0.08g50_s0"; done
  for f in eval.json perwindow.npz; do pull "$R/eval/$f" "$root/A1a_b0.08g50_s0/eval"; done
  grep -a -o '"act_shuffle_[a-z_]*": [^,]*' "$E/$root/A1a_b0.08g50_s0/launch_check.json" | tr '\n' ' '; echo
  echo "$(date +%H:%M) a1a $root pulled"
done
echo "ZZRA1A1A-DONE $(date +%H:%M) boot unchanged ($BOOT0)"
