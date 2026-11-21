#!/usr/bin/env bash
# PREREG_V8F_KIND on box 2 (EVAL ONLY, zero training): which KIND of T's constraint cells is worth ADE6 to O? The two banked checkpoints M2b_s0 and M2b_s1, 15 evals each:
# own + all (the two identity controls), 6 single-kind, 6 leave-one-out, the union = 30 evals of ~13 s. ARMED ONLY ON THE PI's go for box 2. One job at a time under SAFE MODE (mk_v8f_job.py).
# Derived from chain_v8f_dscale.sh (its refresh, its sha checks, its telemetry + temperature guard, its markers) and chain_v8f_tacgrp2.sh (an eval-only chain). Markers ZZV8FMINI-DONE / -STOP
# in ITS OWN log (chain_v8f_kind.out), which v8f_guard.sh's anchored end test already knows.
#   chain_v8f_kind.sh --dry-run   PRINTS every command the chain runs against box 2 and the reader -- the 30 evals are the queue's own `kind` stage lines, from the tip's queue or $V8F_QUEUE -- and exits;
#                                 nothing is shipped, launched or waited for. The line printed IS the line that runs.
#   0. wait for the predecessor (default: PREREG_V8F_DSCALE's chain, chain_v8f_dscale.out, ZZV8FMINI-DONE; stop on its ZZV8FMINI-STOP): box 2 is not shared with a training run
#   1. refresh box 2's code from the REMOTE tip; sha-check 8 key files, md5-check the eval v9 labels; extract the reader HERE from the same archive
#   2. assert box 2's two checkpoints ARE the banked runs' (sha256 of M2b_s0 / M2b_s1 ckpt_last.pt, raw/full/MANIFEST.md): a different file stops the chain while box 2 is idle
#   3. stage the banked B side (runs, standard evals, -tacorc, -tacabs, all-goal, all-latcons, floors) and run `v8f_kind_read.py --check-inputs` on it BEFORE any box-2 eval
#   4. `v8f_mini_box2_queue.cmd kind`: the 30 evals into box 2's eval dir (each skipped once its npz exists); the checkpoints' sha256 is measured AGAIN afterwards and must be unchanged
#   5. pull them; `v8f_kind_read.py` -> kind_read.json (ONE KIND CARRIES / DIFFUSE / INTERACTIVE / MIXED, committed; an identity control that misses exactly +0 [0, 0] is a refusal) -> DONE
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4), any nonzero rc, or a guard stop. Script polling, not model turns.
# TIMEOUTS: the eval wait is 240 polls of 30 s = 2 h for a job the PREREG ESTIMATES at under 1 h. The guard's time rule is inert (no trainer runs: the current-run file stays empty); its
# temperature rule (USB HDD >= 4 of 20 samples >= 61 C, or one >= 63 C) stops our python and the chain then stops on the failed job.
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
R=D:/tanitad/runs/v8f_mini_full
# the banked checkpoints (raw/full/MANIFEST.md); the reader pins the same two values (BANKED_CKPT_SHA) and a test pins that they agree
CKPT_SHA=(601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8 6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6)
KINDS="speed angle tte timegap geom leadgap"
SUFFIXES="kind-own kind-all"; for k in $KINDS; do SUFFIXES="$SUFFIXES kind-only-$k"; done; for k in $KINDS; do SUFFIXES="$SUFFIXES kind-loo-$k"; done; SUFFIXES="$SUFFIXES kind-union"
dry_stage() {   # dry_stage <queue file> <stage args...>: the line(s) the queue prints for a stage under V8F_DRYRUN. cmd.exe cannot be handed `"<path with an ampersand>" args`
                # through MSYS quoting (it strips the quotes it needs): a one-line wrapper batch file can
  local qf=$1; shift
  local w=${TMPDIR:-/tmp}/v8fkind_dry_$$.cmd
  printf '@echo off\r\nset V8F_DRYRUN=1\r\ncall "%s" %s\r\n' "$(cygpath -w "$qf")" "$*" > "$w"
  MSYS_NO_PATHCONV=1 cmd /c "$(cygpath -w "$w")" </dev/null | tr -d '\r'
}
reader_args() {   # reader_args <eval dir> <ckpt sha json> <out json>: the one argument string of the read, printed by --dry-run and run by the chain
  echo "--eval-dir $1 --ckpt-sha $2 --out $3"
}
if [ "${1:-}" = "--dry-run" ]; then
  QF=${V8F_QUEUE:-}
  if [ -z "$QF" ]; then
    MIRROR=${MIRROR:-/c/Users/Admin/tanitad-push}; BR=${BR:-agent/arch-inf-20260803}
    T=$(git -C $MIRROR ls-remote origin $BR | cut -f1); QF=${TMPDIR:-/tmp}/v8fkind_queue_dry_$T.cmd
    git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > "$QF" || { echo "ZZV8FMINI-STOP no queue at the tip"; exit 1; }
  fi
  echo "== predecessor"; echo "wait for ${AFTER_OUT:-<logs>/chain_v8f_dscale.out}: ^${AFTER_DONE:-ZZV8FMINI-DONE} (stop on ^${AFTER_STOP:-ZZV8FMINI-STOP})"
  echo "== checkpoint sha256 (asserted on box 2 before and after the evals)"
  for s in 0 1; do echo "M2b_s$s $R/M2b_s$s/ckpt_last.pt ${CKPT_SHA[$s]}"; done
  echo "== kind"; dry_stage "$QF" kind || exit 1
  echo "== read"; echo "v8f_kind_read.py $(reader_args '<eval>' '<eval>/../ckpt_sha256.json' '<eval>/../kind_read.json')"
  exit 0
fi
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_kind; mkdir -p $E/runs $E/eval
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
HJ="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
B1="TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
EVAL_POLLS=240   # 2 h of 30 s polls for a job the PREREG estimates at under 1 h
TAG=$(date '+%m%d%H%M')
CUR=$L/v8fkind_current.txt
AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_dscale.out}; AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; rm -f $B/BOX2_BUSY.txt; exit 1; }
waitrc() {   # waitrc <job> <polls of 30 s>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; grep ZZGUARD-STOP $L/v8fkind_guard.out 2>/dev/null; stop "$1 rc=$rc"; }
}
queue() {    # queue <job> <polls> <queue args...>: every job name carries this chain's TAG, so an old log's ZZJOB-END can never satisfy a new wait
  local j="v8fkind${TAG}_$1" m=$2; shift 2
  $PY "$(cygpath -m $B)/mk_v8f_job.py" $j "$@" >/dev/null && $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship $j"
  rm -f $L/$j.log
  launch $j || stop "launch $j"
  waitrc $j $m
}
rsha() { $SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm $2 -LiteralPath '$1').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{32,64}$'; }
pull() {     # pull <box-2 path under R> <local dir>
  mkdir -p "$2"; $SCP "tanitad-box2:$R/$1" "$2/" || stop "pull $R/$1"
}
ckptsha() {  # ckptsha <phase>: box 2's two checkpoints ARE the banked runs' (raw/full/MANIFEST.md); writes the measured values for the reader
  local s g0 g1
  for s in 0 1; do
    g=$(rsha "$R/M2b_s$s/ckpt_last.pt" SHA256); [ -n "$g" ] || stop "$1: cannot hash box 2's M2b_s$s checkpoint"
    [ "$g" = "${CKPT_SHA[$s]}" ] || stop "$1: box 2's M2b_s$s checkpoint sha256 $g != the banked run's ${CKPT_SHA[$s]}"
    [ $s = 0 ] && g0=$g || g1=$g
  done
  printf '{"M2b_s0": "%s", "M2b_s1": "%s"}\n' "$g0" "$g1" > $E/ckpt_sha256.json
  echo "$(date '+%m-%d %H:%M') $1: both checkpoints are the banked runs' (sha256 asserted)"
}
finish() {   # finish <summary>: the committed end of the chain
  $SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
  $SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_v8fkind.csv 2>/dev/null
  echo "" > $CUR
  rm -f $B/BOX2_BUSY.txt
  echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)"
  exit 0
}
jget() { $PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))[sys.argv[2]])" "$(cygpath -m $E/$1)" $2; }
echo "$(date '+%m-%d %H:%M') waiting for the predecessor chain ($AFTER_OUT)"
until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do
  grep -q "^$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZV8FMINI-STOP the predecessor stopped: $(grep "^$AFTER_STOP" "$AFTER_OUT" | tail -1)"; exit 1; }
  sleep 300
done
[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present: $(cat $B/BOX2_BUSY.txt)"; exit 1; }
# the marker is taken ATOMICALLY (noclobber): two chains that wait on the same predecessor (this one and the LoRA AR-2 chain both default to DSCALE's) cannot both pass the test above and both write it
( set -C; echo "v8F-mini PREREG_V8F_KIND (eval only: 30 oracle evals on the banked M2b_s0 / M2b_s1) since $(date '+%m-%d %H:%M'), ~1 h. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt ) 2>/dev/null   || { echo "ZZV8FMINI-STOP busy marker taken while arming: $(cat $B/BOX2_BUSY.txt 2>/dev/null)"; exit 1; }
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
QT=$B/v8fkind_queue_$TAG.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" | tr -d '\r' > $QT
for st in kind kindseed kindown kindorc; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done
for f in v8f_kind_read.py v8f_pgt_read.py v8f_dk_read.py v8f_wc_read.py v8f_tac_groups_read.py v8f_mini_full_report.py; do
  git -C $MIRROR cat-file -e "$T:$PK/code/$f" || stop "the tip has no $f"
done
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '^CODE_KINDS = {' || stop "the tip's v8f_mini.py has no CODE_KINDS partition"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_eval.py" | grep -q 'M.code_group_index(a.tac_groups)' || stop "the tip's eval does not route --tac-groups through code_group_index"
git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8fkind_tip.tar $T -- stack taniteval "$HJ" "$PK" "$B1" || stop "archive"
$SCP $B/repo_v8fkind_tip.tar tanitad-box2:D:/tanitad/repo_v8fkind_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8fkind_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8FKIND.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8fkind${TAG}_refresh.cmd
$SCP $B/v8fkind${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8fkind${TAG}_refresh || stop "launch refresh"
waitrc v8fkind${TAG}_refresh 40
for f in stack/scripts/v8f_mini_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py stack/tanitad/models/hjepa.py \
         "$PK/code/v8f_mini_box2_queue.cmd" "$HJ/raw/p10_yaw4512/p10_prior.pt" "$HJ/raw/p10_yaw4512/p10_eval_pred6.npy"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  [ "$l" = "$(rsha "D:/tanitad/repo/$f" SHA256)" ] || stop "box 2 $f != tip $T"
done
[ "6b5c7f207cffc3b7eb3cd527fd433599" = "$(rsha "D:/tanitad/data/v9/v9_labels_eval139.npz" MD5)" ] || stop "box 2 v9 v9_labels_eval139.npz md5 != 6b5c7f207cffc3b7eb3cd527fd433599"
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (7 key files sha-checked; the eval v9 labels md5-checked)"
# the reader runs HERE at the same tip, from the archive box 2 got (a fresh directory, never rm -rf)
X=$B/v8fkind_tip_$(date +%Y%m%d%H%M%S); mkdir -p $X && tar -xf $B/repo_v8fkind_tip.tar -C $X || stop "extract the reader"
RD="$(cygpath -m "$X/$PK/code")"
# ---- 2. the checkpoints ARE the banked runs' (box 2 is idle: a different file costs nothing)
ckptsha "before the evals"
# ---- 3. the B side, staged into this chain's dirs BEFORE any eval (a missing file stops the chain while box 2 is still idle)
for s in 0 1; do
  mkdir -p $E/runs/M2b_s$s
  for f in summary.json argv.json selfcheck.json; do cp "$L/v8f_mini/runs/M2b_s$s/$f" $E/runs/M2b_s$s/ || stop "stage the banked M2b_s$s/$f"; done
  cp "$L/v8f_mini/runs/M2b_s$s/train_log.jsonl" $E/runs/M2b_s$s/ 2>/dev/null || echo "$(date '+%m-%d %H:%M') NOTE: no banked M2b_s$s/train_log.jsonl: not needed by the kind reader"
  cp $L/v8f_mini/eval/eval_M2b_s$s.npz $L/v8f_mini/eval/eval_M2b_s$s.json $E/eval/ || stop "stage eval_M2b_s$s"
  for f in eval_M2b_s$s-tacabs.npz eval_M2b_s$s-tacabs.json eval_M2b_s$s-tacorc.npz; do cp "$L/v8f_cz/eval/$f" $E/eval/ || stop "stage $f"; done
  cp "$L/v8f_cz/eval/eval_M2b_s$s-tacorc.json" $E/eval/ 2>/dev/null || echo "$(date '+%m-%d %H:%M') NOTE: no banked eval_M2b_s$s-tacorc.json (its mode is then unchecked)"
  # M2b's tacorc predates tac_code (WC Amendment 1): its GT constraints come from the all-goal oracle; the LAMBDA known value needs the all-latcons oracle (same dir)
  for f in eval_M2b_s$s-tacorc-all-goal.npz eval_M2b_s$s-tacorc-all-latcons.npz; do cp "$L/v8f_tacgrp2/eval/$f" $E/eval/ || stop "stage $f"; done
  cp "$L/v8f_tacgrp2/eval/eval_M2b_s$s-tacorc-all-latcons.json" $E/eval/ 2>/dev/null
done
cp "$L/v8f_mini/eval/floors_perwindow.npz" $E/eval/ || stop "stage floors_perwindow.npz (the P10 reproduction guard)"
# the reader must load the REAL B side BEFORE any eval: runs, evals, F / LAMBDA / S5 known values, the committed window counts, the identity control, the partition
PYTHONIOENCODING=utf-8 $PY "$RD/v8f_kind_read.py" --check-inputs --eval-dir "$(cygpath -m $E/eval)" --runs-dir "$(cygpath -m $E/runs)" --out "$(cygpath -m $E/check_inputs.json)" \
  || stop "the reader cannot use the M2b side (check-inputs)"
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF "'^(ZZV8FMINI|" $B/v8f_guard.sh || stop "v8f_guard.sh does not know this chain's markers (anchored): it would never exit"
echo "" > $CUR
V8F_CHAIN_OUT=$L/chain_v8f_kind.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh > $L/v8fkind_guard.out 2>&1 &
# ---- 4. the 30 evals (eval only: no trainer runs, so the guard's time rule has nothing to read)
queue kindrun $EVAL_POLLS kind
ckptsha "after the evals"
# ---- 5. pull, read
for s in 0 1; do
  for suf in $SUFFIXES; do for x in json npz log; do pull "eval/eval_M2b_s$s-$suf.$x" "$E/eval"; done; done
done
grep ZZGUARD-STOP $L/v8fkind_guard.out 2>/dev/null && stop "guard stop"
PYTHONIOENCODING=utf-8 $PY "$RD/v8f_kind_read.py" $(reader_args "$(cygpath -m $E/eval)" "$(cygpath -m $E/ckpt_sha256.json)" "$(cygpath -m $E/kind_read.json)") || stop "kind read"
D1=$(jget kind_read.json reading); echo "$(date '+%m-%d %H:%M') $D1"
finish "$D1 | $(jget kind_read.json next)"
