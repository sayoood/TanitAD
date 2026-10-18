#!/usr/bin/env bash
# PREREG_V8F_MINI Amendment 4 on box 2. PI go (2026-10-18, verbatim): "Yes, queue v8F-mini M1 vs M2b on box 2 after RA-1 full finishes,
# on cache_full with a PREREG amendment, safe mode, stop on a crash." One job at a time under SAFE MODE (mk_v8f_job.py):
#   0. wait for chain_ra1_full_runs.sh's ZZRA1FULLRUNS-DONE; its ZZRA1FULLRUNS-STOP stops this chain too (box 2 is left for a human)
#   1. refresh box 2's code from the REMOTE tip; sha-check the key files against the tip's blobs, md5-check the v9 inputs
#   2. the trainer's in-process launch gate (G-DVB + G-LIVE + admissibility) for M1 and M2b; the selfchecks are pulled and asserted
#      HERE: 4,512 TRAIN clips, 60 excluded = the RA-1 holdout (sha f6a1dee0...), the TRAIN list = RA-1 full's, the live check passes
#   3. M1 s0 -> M2b s0 -> M1 s1 -> M2b s1 (seed-major), each followed by its eval; v8f_guard.sh watches temperature + projected time
#   4. floors -> compare -> pull everything -> v8f_mini_full_verdict.py (K-1 and WM-1 as committed in Amendment 4)
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4), any nonzero rc, or a guard stop. Script polling, not model turns.
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_mini; mkdir -p $E/runs $E/eval
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
HJ="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
R=D:/tanitad/runs/v8f_mini_full
HOLDOUT_SHA=f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9
TAG=$(date '+%m%d%H%M')
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $L/v8f_current.txt; exit 1; }
waitrc() {   # waitrc <job> <polls of 30 s>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; grep ZZGUARD-STOP $L/v8f_guard.out 2>/dev/null; stop "$1 rc=$rc"; }
}
queue() {    # queue <job> <polls> <queue args...>: every job name carries this chain's TAG, so an old log's ZZJOB-END can never satisfy a new wait
  local j="v8fm${TAG}_$1" m=$2; shift 2
  $PY "$(cygpath -m $B)/mk_v8f_job.py" $j "$@" >/dev/null && $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship $j"
  rm -f $L/$j.log
  launch $j || stop "launch $j"
  waitrc $j $m
}
rsha() { $SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm $2 -LiteralPath '$1').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{32,64}$'; }
pull() {     # pull <box-2 path under R> <local dir>
  mkdir -p "$2"; $SCP "tanitad-box2:$R/$1" "$2/" || stop "pull $R/$1"
}
echo "$(date '+%m-%d %H:%M') waiting for the RA-1 full runs chain"
until grep -q ZZRA1FULLRUNS-DONE $L/chain_ra1_full_runs.out 2>/dev/null; do
  grep -q ZZRA1FULLRUNS-STOP $L/chain_ra1_full_runs.out 2>/dev/null && stop "the RA-1 runs chain stopped: $(grep ZZRA1FULLRUNS-STOP $L/chain_ra1_full_runs.out)"
  sleep 300
done
[ -f $B/BOX2_BUSY.txt ] && stop "busy marker present: $(cat $B/BOX2_BUSY.txt)"
echo "v8f_mini full (session bfe78918, PREREG_V8F_MINI Amendment 4: M1 vs M2b, 2 seeds) since $(date '+%m-%d %H:%M'), ~10-30 h. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8f_tip.tar $T -- stack taniteval "$HJ" "$PK" || stop "archive"
$SCP $B/repo_v8f_tip.tar tanitad-box2:D:/tanitad/repo_v8f_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8f_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8F.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8fm${TAG}_refresh.cmd
$SCP $B/v8fm${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8fm${TAG}_refresh || stop "launch refresh"
waitrc v8fm${TAG}_refresh 40
for f in stack/scripts/v8f_mini_train.py stack/scripts/v8f_mini_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py \
         stack/tanitad/models/hjepa.py "$PK/code/v8f_mini_box2_queue.cmd" "$HJ/raw/p10_yaw4512/p10_prior.pt" "$HJ/raw/p10_yaw4512/p10_eval_pred6.npy"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  [ "$l" = "$(rsha "D:/tanitad/repo/$f" SHA256)" ] || stop "box 2 $f != tip $T"
done
for kv in v9_labels_train.npz=83e30bd3791f0767707eade5f4671759 v9_labels_eval139.npz=6b5c7f207cffc3b7eb3cd527fd433599; do
  [ "${kv#*=}" = "$(rsha "D:/tanitad/data/v9/${kv%%=*}" MD5)" ] || stop "box 2 v9 ${kv%%=*} md5 != ${kv#*=}"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (8 key files sha-checked; both v9 releases md5-checked)"
git -C $MIRROR show "$T:$PK/code/v8f_mini_full_verdict.py" > $E/v8f_mini_full_verdict.py || stop "verdict script"
rm -rf $E/tip && mkdir -p $E/tip && git -C $MIRROR -c core.autocrlf=false archive --format=tar $T -- taniteval | tar -x -C $E/tip || stop "taniteval for the verdict"
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch (the guard's TEMPERATURE rule is then blind: stopping)"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
echo "" > $L/v8f_current.txt
bash $B/v8f_guard.sh > $L/v8f_guard.out 2>&1 &
# ---- 2. the launch gate, asserted here as well as in-process
queue gate 120 gate
for a in M1 M2b; do
  pull "gate_$a/selfcheck.json" "$E/gate_$a"
  $PY - "$(cygpath -m $E/gate_$a/selfcheck.json)" $a $HOLDOUT_SHA "$(cygpath -m $L/ra1_full)" <<'EOF' || stop "gate $a assertions"
import glob, json, os, sys
sc, arm, hold, ra1 = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], sys.argv[4]
i = sc["info"]
bad = []
if sc.get("declared_vs_built") != "PASS": bad.append("declared_vs_built")
if not sc["live"].get("pass") or sc["live"].get("control_prior_grad_sum") != 0.0: bad.append(f"live {sc['live'].get('pass')} control {sc['live'].get('control_prior_grad_sum')}")
if (i["n_clips"], i["n_excluded"], i["n_excluded_declared"], i["excluded_in_train"]) != (4512, 60, 60, 0): bad.append(f"clips {i['n_clips']} excluded {i['n_excluded']}/{i['n_excluded_declared']} leak {i['excluded_in_train']}")
if i["excluded_sha256"] != hold: bad.append(f"excluded sha {i['excluded_sha256']} is not the RA-1 holdout")
if i["kin_mode"] != "yaw": bad.append("kin_mode")
lv = sc["levers"]
if (lv["wm"], lv["navonly"], lv["consistency"]) != ({"M1": "off", "M2b": "b"}[arm], False, True): bad.append(f"levers {lv}")
ra1_train = set()
for m in glob.glob(os.path.join(ra1, "runs", "st*", "*", "meta.json")):
    t = json.load(open(m)).get("train", {}).get("sha256")
    if t: ra1_train.add(t)
if ra1_train and ra1_train != {i["train_sha256"]}: bad.append(f"TRAIN list {i['train_sha256'][:12]} != RA-1 full's {sorted(ra1_train)}")
print(arm, "gate:", "PASS" if not bad else bad, "| clips", i["n_clips"], "windows", i["n_windows_all_clips"], "steps", sc["steps_total"],
      "| RA-1 train list", "matched" if ra1_train else "not banked locally (not asserted)")
sys.exit(1 if bad else 0)
EOF
done
# ---- 3. the four runs, seed-major, each with its eval
for rs in "M1 0" "M2b 0" "M1 1" "M2b 1"; do
  set -- $rs
  echo "$1_s$2" > $L/v8f_current.txt
  queue "t_$1_s$2" 1500 train $1 $2
  echo "" > $L/v8f_current.txt
  queue "e_$1_s$2" 360 eval $1 $2
  for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull "$1_s$2/$f" "$E/runs/$1_s$2"; done
  for f in eval_$1_s$2.json eval_$1_s$2.npz eval_$1_s$2_proj.npz eval_$1_s$2.log; do pull "eval/$f" "$E/eval"; done
  [ $1 = M1 ] && for f in eval_M1-navshuf_s$2.json eval_M1-navshuf_s$2.npz; do pull "eval/$f" "$E/eval"; done
  grep ZZGUARD-STOP $L/v8f_guard.out 2>/dev/null && stop "guard stop"
done
# ---- 4. floors, compare, the verdict
queue floors 240 floors
queue compare 240 compare
for f in floors_report.json floors_perwindow.npz floors.log compare.json compare.log; do pull "eval/$f" "$E/eval"; done
PYTHONPATH="$(cygpath -w $E/tip/taniteval)" $PY "$(cygpath -m $E/v8f_mini_full_verdict.py)" --eval-dir "$(cygpath -m $E/eval)" --runs-dir "$(cygpath -m $E/runs)" \
  --out "$(cygpath -m $E/verdict.json)" || stop "verdict"
OUT=$($PY -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['summary'])" "$(cygpath -m $E/verdict.json)")
$SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
$SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_v8f.csv 2>/dev/null
rm -f $B/BOX2_BUSY.txt
echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $OUT (boot unchanged $BOOT0)"
