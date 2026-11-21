#!/usr/bin/env bash
# PREREG_V8F_BETA Amendment 1 on box 2: rung beta's EVAL-ONLY gate on the v8F-mini M2b checkpoints. PI go (2026-10-18, verbatim):
# "Yes, queue rung beta's eval-only gate on box 2 after the v8F-mini chain finishes (M2b seeds 0 and 1, plus its shuffled-goal
# control), same safe mode and stop rules." One job at a time under SAFE MODE (mk_beta_job.py):
#   0. wait for chain_v8f_mini.sh's ZZV8FMINI-DONE; its ZZV8FMINI-STOP stops this chain too (box 2 is left for a human)
#   1. refresh box 2's code from the REMOTE tip; sha-check the key files; check both M2b runs finished on ONE clip list with the holdout out
#   2. run --plan beta on M2b s0, then s1 (M2 itself, beta, the shuffled-goal control, the oracle-goal diagnostic), then compare-beta
#   3. pull everything; assert the scorer actually ran (plan_beta.score_calls.candidates > 0, else the read is VOID)
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4: our eval processes are stopped first), any nonzero rc, or a guard stop.
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_beta; mkdir -p $E
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
HJ="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
BK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-18-v8f-beta"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
R=D:/tanitad/runs/v8f_mini_full
HOLDOUT_SHA=f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9
TAG=$(date '+%m%d%H%M')
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
kill_ours() {
  $SSH "powershell -NoProfile -Command \"\$p = @(Get-CimInstance Win32_Process -Filter \\\"Name = 'python.exe'\\\"); \$ours = @(\$p | Where-Object { \$_.CommandLine -match 'v8f_mini_eval' }); \$ids = @(\$ours | ForEach-Object { \$_.ProcessId }); \$kids = @(\$p | Where-Object { \$ids -contains \$_.ParentProcessId }); (\$ours + \$kids) | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue; 'stopped ' + \$_.ProcessId }\"" 2>/dev/null | tr -d '\r'
}
stop() { echo "ZZBETA-STOP $*"; exit 1; }
waitrc() {   # waitrc <job> <polls of 30 s>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 4 ] && kill_ours
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout: our eval processes were stopped)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; grep ZZGUARD-STOP $L/beta_guard.out 2>/dev/null; stop "$1 rc=$rc"; }
}
queue() {    # every job name carries this chain's TAG, so an old log's ZZJOB-END can never satisfy a new wait
  local j="v8fb${TAG}_$1" m=$2; shift 2
  $PY "$(cygpath -m $B)/mk_beta_job.py" $j "$@" >/dev/null && $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship $j"
  rm -f $L/$j.log
  launch $j || stop "launch $j"
  waitrc $j $m
}
rsha() { $SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm SHA256 -LiteralPath '$1').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{64}$'; }
pull() { mkdir -p "$2"; $SCP "tanitad-box2:$R/$1" "$2/" || stop "pull $R/$1"; }
echo "$(date '+%m-%d %H:%M') waiting for the v8F-mini chain"
until grep -q ZZV8FMINI-DONE $L/chain_v8f_mini.out 2>/dev/null; do
  grep -q ZZV8FMINI-STOP $L/chain_v8f_mini.out 2>/dev/null && stop "the v8F-mini chain stopped: $(grep ZZV8FMINI-STOP $L/chain_v8f_mini.out)"
  sleep 300
done
[ -f $B/BOX2_BUSY.txt ] && stop "busy marker present: $(cat $B/BOX2_BUSY.txt)"
echo "v8f beta gate (session bfe78918, PREREG_V8F_BETA Amendment 1: eval only, M2b s0/s1) since $(date '+%m-%d %H:%M'). Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip, and the inputs' provenance
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_beta_tip.tar $T -- stack taniteval "$HJ" "$PK" "$BK" || stop "archive"
$SCP $B/repo_beta_tip.tar tanitad-box2:D:/tanitad/repo_beta_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_beta_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_BETA.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8fb${TAG}_refresh.cmd
$SCP $B/v8fb${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8fb${TAG}_refresh || stop "launch refresh"
waitrc v8fb${TAG}_refresh 40
for f in stack/scripts/v8f_mini_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/models/v8f_beta.py stack/tanitad/models/hjepa.py \
         taniteval/taniteval/v8f_beta_eval.py "$BK/code/v8f_beta_box2_queue.cmd"; do
  [ "$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)" = "$(rsha "D:/tanitad/repo/$f")" ] || stop "box 2 $f != tip $T"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (6 key files sha-checked)"
for s in 0 1; do pull "M2b_s$s/summary.json" "$E/M2b_s$s"; done
$PY - "$(cygpath -m $E)" $HOLDOUT_SHA <<'EOF' || stop "the M2b checkpoints' provenance"
import json, sys
S = [json.load(open(f"{sys.argv[1]}/M2b_s{s}/summary.json")) for s in (0, 1)]
bad = [f"s{s}: {k}" for s, d in enumerate(S) for k, ok in (("done", d.get("done") is True), ("arm M2b", d.get("arm") == "M2b"),
       ("holdout withheld", d.get("excluded_sha256") == sys.argv[2]), ("4,512 clips", d.get("n_clips") == 4512)) if not ok]
if S[0].get("train_sha256") != S[1].get("train_sha256"): bad.append("the two seeds trained on different clip lists")
print("M2b checkpoints:", "OK" if not bad else bad)
sys.exit(1 if bad else 0)
EOF
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
: > $L/beta_current.txt
V8F_CHAIN_OUT=$L/chain_beta.out V8F_CURRENT=$L/beta_current.txt bash $B/v8f_guard.sh > $L/beta_guard.out 2>&1 &
# ---- 2. the gate (the scorer's full-scale runtime was never measured: 6 h per seed, then our processes are stopped)
queue run_s0 720 run 0
queue run_s1 720 run 1
queue compare 120 compare
for f in eval_M2b_s0.json eval_M2b_s1.json eval_M2b_s0_beta.npz eval_M2b_s1_beta.npz eval_M2b_s0.log eval_M2b_s1.log verdict_M2b.json compare_beta.log; do
  pull "eval_beta/$f" "$E/eval_beta"
done
# ---- 3. the read is VOID unless the scorer ran
OUT=$($PY - "$(cygpath -m $E/eval_beta)" <<'EOF'
import json, sys
d = sys.argv[1]
calls = [json.load(open(f"{d}/eval_M2b_s{s}.json")).get("plan_beta", {}).get("score_calls", {}).get("candidates", 0) for s in (0, 1)]
v = json.load(open(f"{d}/verdict_M2b.json"))
print(("VOID (the scorer never ran: candidates calls %s)" % calls) if min(calls) <= 0 else f"{v.get('verdict')} | next: {v.get('next')}")
EOF
) || stop "read the verdict"
$SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
$SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_beta.csv 2>/dev/null
rm -f $B/BOX2_BUSY.txt
echo "ZZBETA-DONE $(date '+%m-%d %H:%M') tip $T: $OUT (boot unchanged $BOOT0)"
