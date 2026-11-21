#!/usr/bin/env bash
# PREREG_V8F_CZ on box 2: T's ABSENT constraint slots. ARMED ONLY ON THE PI's GO. One job at a time under SAFE MODE (mk_v8f_job.py).
# Derived from chain_v8f_mini.sh (the chain that trained M1 / M2b); markers ZZV8FMINI-DONE / -STOP in ITS OWN log (chain_v8f_cz.out),
# which v8f_guard.sh's anchored end test already knows.
#   0. wait for the predecessor (default: PREREG_V8F_TAC Amendment 2's chain, ZZV8FTAC-DONE; stop on its ZZV8FTAC-STOP)
#   1. refresh box 2's code from the REMOTE tip; sha-check 8 key files, md5-check both v9 releases; telemetry + the v8F guard
#   2. STAGE 1 (eval only): `tacabs` -- T's own code with the GT-absent constraint slots at 0, 4 checkpoints; code/v8f_cz_gate_read.py.
#      GATE FAIL -> DONE, no training (a committed outcome, not a stop)
#   3. STAGE 2 (on GATE PASS): the in-process launch gate for M2b-cz, pulled and asserted HERE (4,512 clips, the RA-1 holdout withheld,
#      M2b's 19,164 steps, cons_absent zero); then M2b-cz s0 -> s1, each with its eval and its absent-slot oracle (the mechanism check)
#   4. code/v8f_cz_read.py against the banked M2b runs -> cz_read.json
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4), any nonzero rc, or a guard stop. Script polling, not model turns.
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_cz; mkdir -p $E/runs $E/eval
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
HJ="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
R=D:/tanitad/runs/v8f_mini_full
HOLDOUT_SHA=f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9
M2B_STEPS=19164
TAG=$(date '+%m%d%H%M')
CUR=$L/v8fcz_current.txt
AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_tacgrp2.out}; AFTER_DONE=${AFTER_DONE:-ZZV8FTAC-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FTAC-STOP}
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; exit 1; }
waitrc() {   # waitrc <job> <polls of 30 s>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; grep ZZGUARD-STOP $L/v8fcz_guard.out 2>/dev/null; stop "$1 rc=$rc"; }
}
queue() {    # queue <job> <polls> <queue args...>: every job name carries this chain's TAG, so an old log's ZZJOB-END can never satisfy a new wait
  local j="v8fcz${TAG}_$1" m=$2; shift 2
  $PY "$(cygpath -m $B)/mk_v8f_job.py" $j "$@" >/dev/null && $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship $j"
  rm -f $L/$j.log
  launch $j || stop "launch $j"
  waitrc $j $m
}
rsha() { $SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm $2 -LiteralPath '$1').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{32,64}$'; }
pull() {     # pull <box-2 path under R> <local dir>
  mkdir -p "$2"; $SCP "tanitad-box2:$R/$1" "$2/" || stop "pull $R/$1"
}
finish() {   # finish <summary>: the committed end of the chain (also after a GATE FAIL)
  $SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
  $SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_v8fcz.csv 2>/dev/null
  echo "" > $CUR
  rm -f $B/BOX2_BUSY.txt
  echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)"
  exit 0
}
echo "$(date '+%m-%d %H:%M') waiting for the predecessor chain ($AFTER_OUT)"
until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do
  grep -q "^$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZV8FMINI-STOP the predecessor stopped: $(grep "^$AFTER_STOP" "$AFTER_OUT" | tail -1)"; exit 1; }
  sleep 300
done
[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present: $(cat $B/BOX2_BUSY.txt)"; exit 1; }
echo "v8F-mini PREREG_V8F_CZ (absent-slot oracle, then M2b-cz x 2 seeds on GATE PASS) since $(date '+%m-%d %H:%M'), ~9 h. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" | grep -q '^:tacabsarm' || stop "the tip's queue has no tacabsarm stage"
for f in v8f_cz_gate_read.py v8f_cz_read.py; do git -C $MIRROR cat-file -e "$T:$PK/code/$f" || stop "the tip has no $f"; done
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8fcz_tip.tar $T -- stack taniteval "$HJ" "$PK" || stop "archive"
$SCP $B/repo_v8fcz_tip.tar tanitad-box2:D:/tanitad/repo_v8fcz_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8fcz_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8FCZ.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8fcz${TAG}_refresh.cmd
$SCP $B/v8fcz${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8fcz${TAG}_refresh || stop "launch refresh"
waitrc v8fcz${TAG}_refresh 40
for f in stack/scripts/v8f_mini_train.py stack/scripts/v8f_mini_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py \
         stack/tanitad/models/hjepa.py "$PK/code/v8f_mini_box2_queue.cmd" "$HJ/raw/p10_yaw4512/p10_prior.pt" "$HJ/raw/p10_yaw4512/p10_eval_pred6.npy"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  [ "$l" = "$(rsha "D:/tanitad/repo/$f" SHA256)" ] || stop "box 2 $f != tip $T"
done
for kv in v9_labels_train.npz=83e30bd3791f0767707eade5f4671759 v9_labels_eval139.npz=6b5c7f207cffc3b7eb3cd527fd433599; do
  [ "${kv#*=}" = "$(rsha "D:/tanitad/data/v9/${kv%%=*}" MD5)" ] || stop "box 2 v9 ${kv%%=*} md5 != ${kv#*=}"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (8 key files sha-checked; both v9 releases md5-checked)"
# the readers run HERE at the same tip, from the archive box 2 got (a fresh directory, never rm -rf)
X=$B/v8fcz_tip_$(date +%Y%m%d%H%M%S); mkdir -p $X && tar -xf $B/repo_v8fcz_tip.tar -C $X || stop "extract the readers"
RD="$(cygpath -m "$X/$PK/code")"
[ -f $L/v8f_tac/tac_read.json ] || stop "PREREG_V8F_TAC's tac_read.json is missing locally"
for s in 0 1; do
  for f in runs/M2b_s$s/summary.json eval/eval_M2b_s$s.npz eval/eval_M2b_s$s.json; do [ -f "$L/v8f_mini/$f" ] || stop "the banked M2b file $f is missing locally"; done
done
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF "'^(ZZV8FMINI|" $B/v8f_guard.sh || stop "v8f_guard.sh does not know this chain's markers (anchored): it would never exit"
echo "" > $CUR
V8F_CHAIN_OUT=$L/chain_v8f_cz.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh > $L/v8fcz_guard.out 2>&1 &
# ---- 2. STAGE 1: the absent-slot oracle on the four banked checkpoints
queue tacabs 120 tacabs
for c in M1_s0 M1_s1 M2b_s0 M2b_s1; do
  for s in tacpred tacorc tacabs; do pull "eval/eval_$c-$s.npz" "$E/eval"; done
  pull "eval/eval_$c-tacabs.json" "$E/eval"
done
PYTHONIOENCODING=utf-8 $PY "$RD/v8f_cz_gate_read.py" --eval-dir "$(cygpath -m $E/eval)" --tac-read "$(cygpath -m $L/v8f_tac/tac_read.json)" \
  --out "$(cygpath -m $E/cz_gate.json)" || stop "cz gate read"
GATE=$($PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))['GATE'])" "$(cygpath -m $E/cz_gate.json)")
G1=$($PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))['reading'])" "$(cygpath -m $E/cz_gate.json)")
echo "$(date '+%m-%d %H:%M') $G1"
[ "$GATE" = PASS ] || finish "$G1 -- no training (committed)"
# ---- 3. STAGE 2: the launch gate for M2b-cz, asserted here as well as in-process
queue gatearm 120 gatearm M2b-cz
pull "gate_M2b-cz/selfcheck.json" "$E/gate_M2b-cz"
$PY - "$(cygpath -m $E/gate_M2b-cz/selfcheck.json)" $HOLDOUT_SHA $M2B_STEPS "$(cygpath -m $L/v8f_mini/runs/M2b_s0/summary.json)" <<'EOF' || stop "gate M2b-cz assertions"
import json, sys
sc, hold, steps, m2b = json.load(open(sys.argv[1])), sys.argv[2], int(sys.argv[3]), json.load(open(sys.argv[4]))
i, lv = sc["info"], sc["levers"]
bad = []
if sc.get("declared_vs_built") != "PASS": bad.append("declared_vs_built")
if not sc["live"].get("pass") or sc["live"].get("control_prior_grad_sum") != 0.0: bad.append(f"live {sc['live'].get('pass')}")
if (i["n_clips"], i["n_excluded"], i["n_excluded_declared"], i["excluded_in_train"]) != (4512, 60, 60, 0): bad.append("clips / holdout counts")
if i["excluded_sha256"] != hold: bad.append("the excluded set is not the RA-1 holdout")
if i["train_sha256"] != m2b["train_sha256"]: bad.append(f"TRAIN list {i['train_sha256'][:12]} != M2b's {m2b['train_sha256'][:12]}")
if i["kin_mode"] != "yaw": bad.append("kin_mode")
if (lv["wm"], lv["navonly"], lv["consistency"], lv.get("cons_absent")) != ("b", False, True, "zero"): bad.append(f"levers {lv}")
if sc["steps_total"] != steps: bad.append(f"steps {sc['steps_total']} != M2b's {steps}")
print("M2b-cz gate:", "PASS" if not bad else bad, "| clips", i["n_clips"], "steps", sc["steps_total"], "| levers", lv.get("cons_absent"))
sys.exit(1 if bad else 0)
EOF
# ---- the two runs, each with its eval and its absent-slot oracle
for s in 0 1; do
  echo "M2b-cz_s$s" > $CUR
  queue "t_M2b-cz_s$s" 1500 train M2b-cz $s
  echo "" > $CUR
  queue "e_M2b-cz_s$s" 360 eval M2b-cz $s
  queue "a_M2b-cz_s$s" 120 tacabsarm M2b-cz $s
  for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull "M2b-cz_s$s/$f" "$E/runs/M2b-cz_s$s"; done
  for f in eval_M2b-cz_s$s.json eval_M2b-cz_s$s.npz eval_M2b-cz_s$s.log eval_M2b-cz_s$s-tacabs.json eval_M2b-cz_s$s-tacabs.npz; do pull "eval/$f" "$E/eval"; done
  grep ZZGUARD-STOP $L/v8fcz_guard.out 2>/dev/null && stop "guard stop"
done
# ---- 4. the stage-2 reading, against the banked M2b runs
for s in 0 1; do
  mkdir -p $E/runs/M2b_s$s && cp $L/v8f_mini/runs/M2b_s$s/summary.json $E/runs/M2b_s$s/ || stop "stage M2b_s$s summary"
  cp $L/v8f_mini/eval/eval_M2b_s$s.npz $L/v8f_mini/eval/eval_M2b_s$s.json $E/eval/ || stop "stage eval_M2b_s$s"
done
PYTHONIOENCODING=utf-8 $PY "$RD/v8f_cz_read.py" --eval-dir "$(cygpath -m $E/eval)" --runs-dir "$(cygpath -m $E/runs)" \
  --out "$(cygpath -m $E/cz_read.json)" || stop "cz read"
finish "$G1 | $($PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))['reading'])" "$(cygpath -m $E/cz_read.json)")"
