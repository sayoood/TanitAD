#!/usr/bin/env bash
# PREREG_V8F_TAC Amendment 1 on box 2 (EVAL ONLY, zero training): WHICH part of v8F-mini's tactical code carries the oracle's headroom?
# One job at a time under SAFE MODE. ARMED ONLY ON THE PI's GO. Derived from chain_v8f_tac.sh; the differences are the predecessor,
# the stage (tacgrp: 4 checkpoints x 8 group oracles = 32 evals), the pulls and the reader. Its markers are ZZV8FTAC-DONE / -STOP in
# ITS OWN log (chain_v8f_tacgrp.out), so v8f_guard.sh's anchored end test already knows them.
#   0. wait for the predecessor (default: PREREG_V8F_TAC's chain, ZZV8FTAC-DONE; stop on its ZZV8FTAC-STOP); AFTER_* override it
#   1. refresh box 2's code from the REMOTE tip (stack/, taniteval/, the v8F-mini package, refav1's P10-yaw-4512); sha-check key files
#   2. `v8f_mini_box2_queue.cmd tacgrp`: M1 / M2b x seeds 0, 1 x 8 group oracles = 32 evals of ~25 s each
#   3. pull them + PREREG_V8F_TAC's tacpred / tacorc runs; code/v8f_tac_groups_read.py -> tac_groups_read.json
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4), any nonzero rc, or a guard stop. Script polling, not model turns.
# Temperature: telemetry restarted (no fresh sample => STOP), and v8f_guard.sh (it stops only v8f_mini_train / v8f_mini_eval) watches
# this log; its end test is ANCHORED (^ZZV8FTAC-DONE|STOP), so the waiting line below cannot switch it off.
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_tacgrp; mkdir -p $E/eval
GROUPS="goal latdec londec latcons lonspeed stoplead decisions constraints"
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
HJ="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/raw/p10_yaw4512"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
EVR=D:/tanitad/runs/v8f_mini_full/eval
AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_tac.out}; AFTER_DONE=${AFTER_DONE:-ZZV8FTAC-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FTAC-STOP}
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZV8FTAC-STOP $*"; rm -f $B/BOX2_BUSY.txt; exit 1; }
waitrc() {   # waitrc <job> <polls>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; grep ZZGUARD-STOP $L/v8ftacgrp_guard.out 2>/dev/null; stop "$1 rc=$rc"; }
}
echo "$(date '+%m-%d %H:%M') waiting for the predecessor chain ($AFTER_OUT)"
until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do
  grep -q "^$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZV8FTAC-STOP the predecessor stopped: $(grep "^$AFTER_STOP" "$AFTER_OUT" | tail -1)"; exit 1; }
  sleep 300
done
[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FTAC-STOP busy marker present: $(cat $B/BOX2_BUSY.txt)"; exit 1; }
echo "v8F-mini per-group oracle (PREREG_V8F_TAC Amendment 1, eval only) since $(date '+%m-%d %H:%M'), ~30 min. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" | grep -q '^:tacgrp' || stop "the tip's queue has no tacgrp stage"
git -C $MIRROR cat-file -e "$T:$PK/code/v8f_tac_groups_read.py" || stop "the tip has no v8f_tac_groups_read.py"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8ftacgrp_tip.tar $T -- stack taniteval "$PK" "$HJ" || stop "archive"
$SCP $B/repo_v8ftacgrp_tip.tar tanitad-box2:D:/tanitad/repo_v8ftacgrp_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8ftacgrp_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8FTACGRP.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8ftacgrp_refresh.cmd
$SCP $B/v8ftacgrp_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8ftacgrp_refresh || stop "launch refresh"
waitrc v8ftacgrp_refresh 40
for f in stack/tanitad/models/v8f_mini.py stack/scripts/v8f_mini_eval.py stack/tanitad/data/v8f_mini_data.py stack/tanitad/models/hjepa.py \
         "$PK/code/v8f_mini_box2_queue.cmd" "$HJ/p10_prior.pt"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  r=$($SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm SHA256 -LiteralPath 'D:/tanitad/repo/$f').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{64}$')
  [ "$l" = "$r" ] || stop "box 2 $f != tip $T"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (6 key files sha-checked)"
for c in M1_s0 M1_s1 M2b_s0 M2b_s1; do
  $SSH "powershell -NoProfile -Command \"Test-Path D:/tanitad/runs/v8f_mini_full/$c/ckpt_last.pt\"" 2>/dev/null | tr -d '\r' | grep -q True || stop "box 2 has no $c checkpoint"
done
# the reader runs HERE at the same tip, from the archive box 2 got (a fresh directory, never rm -rf)
X=$B/v8ftacgrp_tip_$(date +%Y%m%d%H%M%S); mkdir -p $X && tar -xf $B/repo_v8ftacgrp_tip.tar -C $X || stop "extract the reader"
RD="$(cygpath -m "$X/$PK/code")"; [ -f "$RD/v8f_tac_groups_read.py" ] || stop "the extract has no v8f_tac_groups_read.py"
[ -f $L/v8f_tac/tac_read.json ] || stop "PREREG_V8F_TAC's tac_read.json is missing locally"
# ---- telemetry + the v8F guard (temperature rule unchanged; the time rule is inert: no trainer runs)
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF '|ZZV8FTAC)-(DONE|STOP)' $B/v8f_guard.sh || stop "v8f_guard.sh does not know this chain's markers (anchored): it would never exit"
: > $L/v8ftacgrp_current.txt
V8F_CHAIN_OUT=$L/chain_v8f_tacgrp.out V8F_CURRENT=$L/v8ftacgrp_current.txt bash $B/v8f_guard.sh > $L/v8ftacgrp_guard.out 2>&1 &
# ---- 2. the 32 group-oracle evals
$PY "$(cygpath -m $B)/mk_v8f_job.py" v8ftacgrp_run tacgrp >/dev/null && $SCP $B/v8ftacgrp_run.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship v8ftacgrp_run"
launch v8ftacgrp_run || stop "launch v8ftacgrp_run"
waitrc v8ftacgrp_run 300
for c in M1_s0 M1_s1 M2b_s0 M2b_s1; do
  for s in tacpred tacorc; do $SCP "tanitad-box2:$EVR/eval_$c-$s.npz" $E/eval/ || stop "pull eval_$c-$s.npz"; done
  for g in $GROUPS; do s=tacorc-$g
    for x in npz json; do $SCP "tanitad-box2:$EVR/eval_$c-$s.$x" $E/eval/ || stop "pull eval_$c-$s.$x"; done
  done
done
# ---- 3. the reading
PYTHONIOENCODING=utf-8 $PY "$RD/v8f_tac_groups_read.py" --eval-dir "$(cygpath -m $E/eval)" --tac-read "$(cygpath -m $L/v8f_tac/tac_read.json)" \
  --out "$(cygpath -m $E/tac_groups_read.json)" || stop "tac groups read"
$SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
$SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_v8ftacgrp.csv 2>/dev/null
rm -f $B/BOX2_BUSY.txt
echo "ZZV8FTAC-DONE $(date '+%m-%d %H:%M') tip $T: $(PYTHONIOENCODING=utf-8 $PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))['reading'])" "$(cygpath -m $E/tac_groups_read.json)") (boot unchanged $BOOT0)"
