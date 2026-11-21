#!/usr/bin/env bash
# PREREG_V8F_WC on box 2: T's PRESENT constraint slots, by the loss weight. ARMED ONLY ON THE PI's GO. One job at a time under SAFE MODE
# (mk_v8f_job.py). Derived from chain_v8f_cz.sh; markers ZZV8FMINI-DONE / -STOP in ITS OWN log (chain_v8f_wc.out), which v8f_guard.sh's
# anchored end test already knows.
#   0. wait for the predecessor (default: PREREG_V8F_CZ's chain, ZZV8FMINI-DONE; stop on its ZZV8FMINI-STOP)
#   1. refresh box 2's code from the REMOTE tip; sha-check 8 key files, md5-check both v9 releases; telemetry + the v8F guard
#   2. the in-process launch gate for M2b-wc (`gatewc`), pulled and asserted HERE: 4,512 clips, the RA-1 holdout withheld, M2b's clip
#      list and 19,164 steps, M2b's levers, and the built w_cons 4.0
#   3. STAGE 1: M2b-wc s0 -> its eval, absmask and GT oracle (`evalwc`); code/v8f_wc_read.py --stage s0 against the banked M2b runs.
#      STOP -> DONE with the seed-0 verdict (one failing seed already fails a both-seeds criterion: a committed outcome, not a stop)
#   4. STAGE 2 (on CONTINUE): M2b-wc s1 -> evalwc; code/v8f_wc_read.py --stage final -> wc_read.json
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4), any nonzero rc, or a guard stop. Script polling, not model turns.
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_wc; mkdir -p $E/runs $E/eval
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
HJ="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
R=D:/tanitad/runs/v8f_mini_full
HOLDOUT_SHA=f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9
M2B_STEPS=19164
WC=4.0
TAG=$(date '+%m%d%H%M')
CUR=$L/v8fwc_current.txt
AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_cz.out}; AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; exit 1; }
waitrc() {   # waitrc <job> <polls of 30 s>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; grep ZZGUARD-STOP $L/v8fwc_guard.out 2>/dev/null; stop "$1 rc=$rc"; }
}
queue() {    # queue <job> <polls> <queue args...>: every job name carries this chain's TAG, so an old log's ZZJOB-END can never satisfy a new wait
  local j="v8fwc${TAG}_$1" m=$2; shift 2
  $PY "$(cygpath -m $B)/mk_v8f_job.py" $j "$@" >/dev/null && $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship $j"
  rm -f $L/$j.log
  launch $j || stop "launch $j"
  waitrc $j $m
}
rsha() { $SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm $2 -LiteralPath '$1').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{32,64}$'; }
pull() {     # pull <box-2 path under R> <local dir>
  mkdir -p "$2"; $SCP "tanitad-box2:$R/$1" "$2/" || stop "pull $R/$1"
}
finish() {   # finish <summary>: the committed end of the chain (also after a stage-1 STOP)
  $SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
  $SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_v8fwc.csv 2>/dev/null
  echo "" > $CUR
  rm -f $B/BOX2_BUSY.txt
  echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)"
  exit 0
}
runone() {   # runone <seed>: train, eval (+ absmask + GT oracle), pull
  local s=$1
  echo "M2b-wc_s$s" > $CUR
  queue "t_M2b-wc_s$s" 1500 trainwc $s
  echo "" > $CUR
  queue "e_M2b-wc_s$s" 480 evalwc $s
  for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull "M2b-wc_s$s/$f" "$E/runs/M2b-wc_s$s"; done
  for f in "" -tacabs -tacorc; do for x in json npz log; do pull "eval/eval_M2b-wc_s$s$f.$x" "$E/eval"; done; done
  grep ZZGUARD-STOP $L/v8fwc_guard.out 2>/dev/null && stop "guard stop"
}
wcread() {   # wcread <stage> <out>
  PYTHONIOENCODING=utf-8 $PY "$RD/v8f_wc_read.py" --stage $1 --eval-dir "$(cygpath -m $E/eval)" --runs-dir "$(cygpath -m $E/runs)" \
    --out "$(cygpath -m $E/$2)" || stop "wc read $1"
}
jget() { $PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))[sys.argv[2]])" "$(cygpath -m $E/$1)" $2; }
echo "$(date '+%m-%d %H:%M') waiting for the predecessor chain ($AFTER_OUT)"
until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do
  grep -q "^$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZV8FMINI-STOP the predecessor stopped: $(grep "^$AFTER_STOP" "$AFTER_OUT" | tail -1)"; exit 1; }
  sleep 300
done
[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present: $(cat $B/BOX2_BUSY.txt)"; exit 1; }
echo "v8F-mini PREREG_V8F_WC (M2b-wc = M2b + w_cons $WC; seed 0, then seed 1 on CONTINUE) since $(date '+%m-%d %H:%M'), ~5-10 h. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" | grep -q '^:trainwc' || stop "the tip's queue has no trainwc stage"
git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" | grep -q "^set \"WC=$WC\"" || stop "the tip's queue does not set WC=$WC"
git -C $MIRROR cat-file -e "$T:$PK/code/v8f_wc_read.py" || stop "the tip has no v8f_wc_read.py"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"w_cons": cfg.w_cons' || stop "the tip's built_levers does not record w_cons"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8fwc_tip.tar $T -- stack taniteval "$HJ" "$PK" || stop "archive"
$SCP $B/repo_v8fwc_tip.tar tanitad-box2:D:/tanitad/repo_v8fwc_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8fwc_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8FWC.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8fwc${TAG}_refresh.cmd
$SCP $B/v8fwc${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8fwc${TAG}_refresh || stop "launch refresh"
waitrc v8fwc${TAG}_refresh 40
for f in stack/scripts/v8f_mini_train.py stack/scripts/v8f_mini_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py \
         stack/tanitad/models/hjepa.py "$PK/code/v8f_mini_box2_queue.cmd" "$HJ/raw/p10_yaw4512/p10_prior.pt" "$HJ/raw/p10_yaw4512/p10_eval_pred6.npy"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  [ "$l" = "$(rsha "D:/tanitad/repo/$f" SHA256)" ] || stop "box 2 $f != tip $T"
done
for kv in v9_labels_train.npz=83e30bd3791f0767707eade5f4671759 v9_labels_eval139.npz=6b5c7f207cffc3b7eb3cd527fd433599; do
  [ "${kv#*=}" = "$(rsha "D:/tanitad/data/v9/${kv%%=*}" MD5)" ] || stop "box 2 v9 ${kv%%=*} md5 != ${kv#*=}"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (8 key files sha-checked; both v9 releases md5-checked)"
# the reader runs HERE at the same tip, from the archive box 2 got (a fresh directory, never rm -rf)
X=$B/v8fwc_tip_$(date +%Y%m%d%H%M%S); mkdir -p $X && tar -xf $B/repo_v8fwc_tip.tar -C $X || stop "extract the reader"
RD="$(cygpath -m "$X/$PK/code")"
# the banked M2b side, staged into this chain's dirs BEFORE any training (a missing file stops the chain while box 2 is still idle)
for s in 0 1; do
  mkdir -p $E/runs/M2b_s$s
  for f in summary.json argv.json selfcheck.json; do cp "$L/v8f_mini/runs/M2b_s$s/$f" $E/runs/M2b_s$s/ || stop "stage the banked M2b_s$s/$f"; done
  cp $L/v8f_mini/eval/eval_M2b_s$s.npz $L/v8f_mini/eval/eval_M2b_s$s.json $E/eval/ || stop "stage eval_M2b_s$s"
  for f in eval_M2b_s$s-tacabs.npz eval_M2b_s$s-tacabs.json eval_M2b_s$s-tacorc.npz; do cp "$L/v8f_cz/eval/$f" $E/eval/ || stop "stage $f"; done
  # PREREG_V8F_WC Amendment 1: M2b's tacorc predates tac_code; its GT constraints come from Amendment 2's all-goal oracle
  cp "$L/v8f_tacgrp2/eval/eval_M2b_s$s-tacorc-all-goal.npz" $E/eval/ || stop "stage eval_M2b_s$s-tacorc-all-goal.npz"
done
# Amendment 1: the reader must load the REAL banked M2b side BEFORE any training (it was tested only on synthetic inputs and crashed)
PYTHONIOENCODING=utf-8 $PY "$RD/v8f_wc_read.py" --check-inputs --eval-dir "$(cygpath -m $E/eval)" --runs-dir "$(cygpath -m $E/runs)"   --out "$(cygpath -m $E/check_inputs.json)" || stop "the reader cannot use the banked M2b side (check-inputs)"
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF "'^(ZZV8FMINI|" $B/v8f_guard.sh || stop "v8f_guard.sh does not know this chain's markers (anchored): it would never exit"
echo "" > $CUR
V8F_CHAIN_OUT=$L/chain_v8f_wc.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh > $L/v8fwc_guard.out 2>&1 &
# ---- 2. the launch gate for M2b-wc, asserted here as well as in-process
queue gatewc 120 gatewc
pull "gate_M2b-wc/selfcheck.json" "$E/gate_M2b-wc"
$PY - "$(cygpath -m $E/gate_M2b-wc/selfcheck.json)" $HOLDOUT_SHA $M2B_STEPS "$(cygpath -m $L/v8f_mini/runs/M2b_s0/summary.json)" $WC <<'EOF' || stop "gate M2b-wc assertions"
import json, sys
sc, hold, steps, m2b, wc = json.load(open(sys.argv[1])), sys.argv[2], int(sys.argv[3]), json.load(open(sys.argv[4])), float(sys.argv[5])
i, lv = sc["info"], sc["levers"]
bad = []
if sc.get("declared_vs_built") != "PASS": bad.append("declared_vs_built")
if not sc["live"].get("pass") or sc["live"].get("control_prior_grad_sum") != 0.0: bad.append(f"live {sc['live'].get('pass')}")
if (i["n_clips"], i["n_excluded"], i["n_excluded_declared"], i["excluded_in_train"]) != (4512, 60, 60, 0): bad.append("clips / holdout counts")
if i["excluded_sha256"] != hold: bad.append("the excluded set is not the RA-1 holdout")
if i["train_sha256"] != m2b["train_sha256"]: bad.append(f"TRAIN list {i['train_sha256'][:12]} != M2b's {m2b['train_sha256'][:12]}")
if i["kin_mode"] != "yaw": bad.append("kin_mode")
if (lv["wm"], lv["navonly"], lv["consistency"], lv.get("cons_absent")) != ("b", False, True, "mask"): bad.append(f"levers {lv}")
w = lv.get("w") or {}
if w.get("w_cons") != wc or w.get("w_tac") != 1.0: bad.append(f"built w_cons {w.get('w_cons')} / w_tac {w.get('w_tac')}, expected {wc} / 1.0")
for k in ("w_pen", "w_pen_self", "w_wm"):
    if w.get(k) != m2b["levers"]["w"][k]: bad.append(f"{k} {w.get(k)} != M2b's {m2b['levers']['w'][k]}")
if sc["steps_total"] != steps: bad.append(f"steps {sc['steps_total']} != M2b's {steps}")
print("M2b-wc gate:", "PASS" if not bad else bad, "| clips", i["n_clips"], "steps", sc["steps_total"], "| w_cons", w.get("w_cons"))
sys.exit(1 if bad else 0)
EOF
# ---- 3. STAGE 1: seed 0, read against the banked M2b runs
runone 0
wcread s0 wc_read_s0.json
G1=$(jget wc_read_s0.json reading); echo "$(date '+%m-%d %H:%M') $G1"
[ "$(jget wc_read_s0.json DECISION)" = CONTINUE ] || finish "$G1 -- seed 1 not trained (committed)"
# ---- 4. STAGE 2: seed 1, the final reading
runone 1
wcread final wc_read.json
finish "$G1 | $(jget wc_read.json reading)"
