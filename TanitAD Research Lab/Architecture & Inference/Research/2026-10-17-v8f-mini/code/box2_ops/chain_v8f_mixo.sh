#!/usr/bin/env bash
# PREREG_V8F_MIXO on box 2: does training O on codes mixed PER UNIT teach it to use a partly right T code? (X = M2b + --code-mix kind, ONE lever: the granularity of O's teacher forcing; B = M2b,
# fixed in the PREREG.) CONDITIONAL (PREREG_V8F_KIND must have read DIFFUSE, INTERACTIVE or MIXED -- it read INTERACTIVE) and ARMED ONLY ON THE PI's go for box 2. One job at a time under SAFE MODE
# (mk_v8f_job.py). Derived from chain_v8f_hist.sh (the training chain: gate, seed 0, read, seed 1 on CONTINUE) and chain_v8f_kind.sh (the predecessor wait with an atomic busy marker, the checkpoint
# sha256 assertions). Markers ZZV8FMINI-DONE / -STOP in ITS OWN log (chain_v8f_mixo.out), which v8f_guard.sh's anchored end test already knows.
#   chain_v8f_mixo.sh --dry-run   PRINTS the exact command every stage runs against box 2 (the queue's own `gatemixo` / `trainmixo` / `evalmixo` lines, from the tip's queue or $V8F_QUEUE), the
#                                 checkpoint sha256 values it asserts and the reader's arguments, and exits; nothing is shipped, launched or waited for. The line printed IS the line that runs.
#   0. wait for the predecessor (default: PREREG_V8F_KIND's chain, chain_v8f_kind.out, ZZV8FMINI-DONE; stop on its ZZV8FMINI-STOP): box 2 is not shared with another job. AFTER_OUT / AFTER_DONE /
#      AFTER_STOP override it; the busy marker is then taken ATOMICALLY (noclobber)
#   1. refresh box 2's code from the REMOTE tip; sha-check 8 key files, md5-check both v9 releases; extract the reader HERE from the same archive
#   2. B = M2b is fixed in the PREREG: box 2's two M2b checkpoints ARE the banked runs' (sha256, raw/full/MANIFEST.md); a different file stops the chain while box 2 is idle
#   3. the B side is staged from the banked M2b files (the standard evals, absmask, the GT-code and all-latcons oracles, and PREREG_V8F_TAC Amendments 1 and 2's tacpred / constraints / alpha 0.50
#      files) and `v8f_mixo_read.py --check-inputs` runs on it BEFORE any training: runs, evals, the PREREG's known values, the committed window counts (2,399 / 141), B's r(0.5) = 0.36 / 0.21
#      REPRODUCED from the banked dose files, the identity control (+0 [0, 0], the dose contrasts included)
#   4. the in-process launch gate for X (`gatemixo`), pulled and asserted HERE: the PASS token (declared_vs_built PASS, G-LIVE pass with a frozen-prior control of exactly 0.0, the code_mix probe
#      that read a PARTLY-GT code), 4,512 clips, the RA-1 holdout withheld, B's clip list and WINDOW SET (306,566 windows) and 19,164 steps, the BUILT code_mix kind, B's levers / weights / p_gt
#      schedule (1.0 -> 0.2) / hist 4 / hmax 16 / window start 3, and X's parameter count = B's EXACTLY (the lever adds none). The queue's own printed line carries `--code-mix kind` exactly once and none of
#      the other levers' flags. A token record (gate_token.json: tip, queue / selfcheck / key-file sha256, the data manifests) is written beside it. No PASS => no training (the trainer's own in-process
#      gate refuses as well, ZZDVB-REFUSEDZZ)
#   5. STAGE 1: X s0 (`trainmixo 0`) -> its seven evals (`evalmixo 0`: the standard eval, absmask, the GT-code oracle, the all-latcons oracle, T's own code, the constraints oracle at alpha 1 and 0.50);
#      the checkpoint sha256 is measured after training AND after the evals (one value); v8f_mixo_read.py --stage s0 against B's runs. STOP -> DONE with the seed-0 verdict (committed)
#   6. STAGE 2 (on CONTINUE = a seed-0 PRIMARY PASS): X s1 -> evalmixo; v8f_mixo_read.py --stage final -> mixo_read.json
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4), any nonzero rc, or a guard stop. Script polling, not model turns.
# TIMEOUTS: the training wait is 1,320 polls of 30 s = 11 h, one hour above the guard's 10 h cap (V8F_MAX_RUN_H); X trains M2b's graph (the lever adds no parameter and one mask draw per step),
# so a seed is ESTIMATED at B's own wall-clock (banked summary.json `seconds`), and v8f_guard.sh stops any run it PROJECTS above the cap, so no run the guard lets live can outlast the wait.
# The seven evals are ~2-4 min each (the wait is 600 polls = 5 h, far above). The PREREG fixes steps, not wall-clock.
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
R=D:/tanitad/runs/v8f_mini_full
# the banked checkpoints (raw/full/MANIFEST.md); the reader pins the same two values (BANKED_CKPT_SHA) and a test pins that they agree
CKPT_SHA=(601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8 6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6)
# the seven evals of evalmixo: the standard eval (empty suffix) and six tagged ones; EXPECT_EVALS pins the count (the reader expects the same seven)
X_SUFFIXES="-tacabs -tacorc -tacorc-all-latcons -tacpred -tacorc-constraints -tacorc-constraints-a50"
dry_stage() {   # dry_stage <queue file> <stage args...>: the line(s) the queue prints for a stage under V8F_DRYRUN. cmd.exe cannot be handed `"<path with an ampersand>" args`
                # through MSYS quoting (it strips the quotes it needs): a one-line wrapper batch file can
  local qf=$1; shift
  local w=${TMPDIR:-/tmp}/v8fmixo_dry_$$.cmd
  printf '@echo off\r\nset V8F_DRYRUN=1\r\ncall "%s" %s\r\n' "$(cygpath -w "$qf")" "$*" > "$w"
  MSYS_NO_PATHCONV=1 cmd /c "$(cygpath -w "$w")" </dev/null | tr -d '\r'
}
reader_args() {   # reader_args <eval dir> <runs dir> <ckpt sha json> <out json>: the one argument string of the read, printed by --dry-run and run by the chain
  echo "--eval-dir $1 --runs-dir $2 --ckpt-sha $3 --out $4"
}
if [ "${1:-}" = "--dry-run" ]; then
  QF=${V8F_QUEUE:-}
  if [ -z "$QF" ]; then
    MIRROR=${MIRROR:-/c/Users/Admin/tanitad-push}; BR=${BR:-agent/arch-inf-20260803}
    T=$(git -C $MIRROR ls-remote origin $BR | cut -f1); QF=${TMPDIR:-/tmp}/v8fmixo_queue_dry_$T.cmd
    git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > "$QF" || { echo "ZZV8FMINI-STOP no queue at the tip"; exit 1; }
  fi
  echo "== predecessor"; echo "wait for ${AFTER_OUT:-<logs>/chain_v8f_kind.out}: ^${AFTER_DONE:-ZZV8FMINI-DONE} (stop on ^${AFTER_STOP:-ZZV8FMINI-STOP})"
  echo "== checkpoint sha256 (asserted on box 2 before the gate; X's measured after training and after the evals)"
  for s in 0 1; do echo "M2b_s$s $R/M2b_s$s/ckpt_last.pt ${CKPT_SHA[$s]}"; done
  for st in "gatemixo" "trainmixo 0" "trainmixo 1" "evalmixo 0" "evalmixo 1"; do
    echo "== $st"; dry_stage "$QF" $st || exit 1
  done
  echo "== read"
  echo "v8f_mixo_read.py --check-inputs $(reader_args '<eval>' '<runs>' '<eval>/../ckpt_sha256.json' '<eval>/../check_inputs.json')"
  echo "v8f_mixo_read.py --stage s0 $(reader_args '<eval>' '<runs>' '<eval>/../ckpt_sha256.json' '<eval>/../mixo_read_s0.json')"
  echo "v8f_mixo_read.py --stage final $(reader_args '<eval>' '<runs>' '<eval>/../ckpt_sha256.json' '<eval>/../mixo_read.json')"
  exit 0
fi
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_mixo; mkdir -p $E/runs $E/eval
CK=$E/ckpt_sha256.json
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
HJ="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
B1="TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
HOLDOUT_SHA=f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9
M2B_STEPS=19164
M2B_WINDOWS=306566   # B's n_windows_all_clips (banked summary.json); X trains on B's window LIST (the lever is the code mask only)
TRAIN_POLLS=1320; EVAL_POLLS=600   # 11 h: one hour above the guard cap, so the GUARD (a committed rule), never the wait, ends a slow run
TAG=$(date '+%m%d%H%M')
CUR=$L/v8fmixo_current.txt
AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_kind.out}; AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; rm -f $B/BOX2_BUSY.txt; exit 1; }
waitrc() {   # waitrc <job> <polls of 30 s>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; grep ZZGUARD-STOP $L/v8fmixo_guard.out 2>/dev/null; stop "$1 rc=$rc"; }
}
queue() {    # queue <job> <polls> <queue args...>: every job name carries this chain's TAG, so an old log's ZZJOB-END can never satisfy a new wait
  local j="v8fmixo${TAG}_$1" m=$2; shift 2
  $PY "$(cygpath -m $B)/mk_v8f_job.py" $j "$@" >/dev/null && $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship $j"
  rm -f $L/$j.log
  launch $j || stop "launch $j"
  waitrc $j $m
}
rsha() { $SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm $2 -LiteralPath '$1').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{32,64}$'; }
pull() {     # pull <box-2 path under R> <local dir>
  mkdir -p "$2"; $SCP "tanitad-box2:$R/$1" "$2/" || stop "pull $R/$1"
}
ckset() {    # ckset <key> <sha256>: one measured value into the json the reader takes as --ckpt-sha
  $PY -c "import json,os,sys;p=sys.argv[1];d=json.load(open(p,encoding='utf-8')) if os.path.exists(p) else {};d[sys.argv[2]]=sys.argv[3];json.dump(d,open(p,'w',encoding='utf-8'),indent=1)" "$(cygpath -m $CK)" "$1" "$2"
}
ckptsha_b() {   # ckptsha_b <phase>: box 2's two M2b checkpoints ARE the banked runs' (raw/full/MANIFEST.md)
  local s g
  for s in 0 1; do
    g=$(rsha "$R/M2b_s$s/ckpt_last.pt" SHA256); [ -n "$g" ] || stop "$1: cannot hash box 2's M2b_s$s checkpoint"
    [ "$g" = "${CKPT_SHA[$s]}" ] || stop "$1: box 2's M2b_s$s checkpoint sha256 $g != the banked run's ${CKPT_SHA[$s]}"
    ckset "M2b_s$s" "$g"
  done
  echo "$(date '+%m-%d %H:%M') $1: both M2b checkpoints are the banked runs' (sha256 asserted)"
}
ckptsha_x() {   # ckptsha_x <seed> <trained|after_evals>: X's checkpoint, measured on box 2; after the evals it must be the value measured after training (the scored checkpoint IS the trained one)
  local g; g=$(rsha "$R/M2b_s${1}_mixo/ckpt_last.pt" SHA256); [ -n "$g" ] || stop "cannot hash box 2's M2b_s${1}_mixo checkpoint ($2)"
  if [ "$2" = after_evals ]; then
    local g0; g0=$($PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))[sys.argv[2]])" "$(cygpath -m $CK)" "M2b-mixo_s${1}_trained")
    [ "$g" = "$g0" ] || stop "M2b_s${1}_mixo's checkpoint changed during the evals: $g != $g0"
  fi
  ckset "M2b-mixo_s${1}_$2" "$g"
  echo "$(date '+%m-%d %H:%M') M2b_s${1}_mixo checkpoint sha256 ($2) ${g:0:16}..."
}
finish() {   # finish <summary>: the committed end of the chain (also after a stage-1 STOP)
  $SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
  $SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_v8fmixo.csv 2>/dev/null
  echo "" > $CUR
  rm -f $B/BOX2_BUSY.txt
  echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)"
  exit 0
}
runone() {   # runone <seed>: train X, eval (seven evals), pull, record the checkpoint sha256
  local s=$1 g="M2b_s${1}_mixo" tg="M2b-mixo_s${1}" f x
  echo "$g" > $CUR
  queue "t_$g" $TRAIN_POLLS trainmixo $s
  echo "" > $CUR
  ckptsha_x $s trained
  queue "e_$g" $EVAL_POLLS evalmixo $s
  ckptsha_x $s after_evals
  for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull "$g/$f" "$E/runs/$g"; done
  for f in "" $X_SUFFIXES; do for x in json npz log; do pull "eval/eval_$tg$f.$x" "$E/eval"; done; done
  grep ZZGUARD-STOP $L/v8fmixo_guard.out 2>/dev/null && stop "guard stop"
}
mixoread() {   # mixoread <stage> <out>
  PYTHONIOENCODING=utf-8 $PY "$RD/v8f_mixo_read.py" --stage $1 $(reader_args "$(cygpath -m $E/eval)" "$(cygpath -m $E/runs)" "$(cygpath -m $CK)" "$(cygpath -m $E/$2)") || stop "mixo read $1"
}
jget() { $PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))[sys.argv[2]])" "$(cygpath -m $E/$1)" $2; }
echo "$(date '+%m-%d %H:%M') waiting for the predecessor chain ($AFTER_OUT)"
until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do
  grep -q "^$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZV8FMINI-STOP the predecessor stopped: $(grep "^$AFTER_STOP" "$AFTER_OUT" | tail -1)"; exit 1; }
  sleep 300
done
[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present: $(cat $B/BOX2_BUSY.txt)"; exit 1; }
# the marker is taken ATOMICALLY (noclobber): two chains that wait on the same predecessor cannot both pass the test above and both write it
( set -C; echo "v8F-mini PREREG_V8F_MIXO (X = M2b + --code-mix kind, B = M2b; seed 0, then seed 1 on a seed-0 PRIMARY PASS) since $(date '+%m-%d %H:%M'), ~B's wall-clock per seed plus 7 evals. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt ) 2>/dev/null \
  || { echo "ZZV8FMINI-STOP busy marker taken while arming: $(cat $B/BOX2_BUSY.txt 2>/dev/null)"; exit 1; }
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
QT=$B/v8fmixo_queue_$TAG.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" | tr -d '\r' > $QT
QC=$B/v8fmixo_queue_$TAG.crlf.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > $QC      # the committed bytes (CRLF): what cmd.exe can run for the dry-run line
for st in gatemixo trainmixo evalmixo; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done
for f in v8f_mixo_read.py v8f_grid_read.py v8f_pgt_read.py v8f_dk_read.py v8f_wc_read.py v8f_tac_groups_read.py v8f_mini_full_report.py; do
  git -C $MIRROR cat-file -e "$T:$PK/code/$f" || stop "the tip has no $f"
done
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"code_mix": cfg.code_mix' || stop "the tip's built_levers does not record code_mix"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '^CODE_MIX_UNITS = {' || stop "the tip's v8f_mini.py has no CODE_MIX_UNITS partition"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"p_gt_start": cfg.p_gt_start' || stop "the tip's built_levers does not record p_gt_start"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"p_gt_end": cfg.p_gt_end' || stop "the tip's built_levers does not record p_gt_end"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q -- '"--code-mix"' || stop "the tip's trainer has no --code-mix flag"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q '"code_mix": "lever:code_mix"' || stop "the tip's trainer has no G-DVB entry (FLAG_CHECKS) for --code-mix"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q 'code_mix_probe(model, probe, cfg)' || stop "the tip's trainer does not run the code_mix probe in its in-process gate"
git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8fmixo_tip.tar $T -- stack taniteval "$HJ" "$PK" "$B1" || stop "archive"
$SCP $B/repo_v8fmixo_tip.tar tanitad-box2:D:/tanitad/repo_v8fmixo_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8fmixo_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8FMIXO.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8fmixo${TAG}_refresh.cmd
$SCP $B/v8fmixo${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8fmixo${TAG}_refresh || stop "launch refresh"
waitrc v8fmixo${TAG}_refresh 40
KEYSHA=""
for f in stack/scripts/v8f_mini_train.py stack/scripts/v8f_mini_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py \
         stack/tanitad/models/hjepa.py "$PK/code/v8f_mini_box2_queue.cmd" "$HJ/raw/p10_yaw4512/p10_prior.pt" "$HJ/raw/p10_yaw4512/p10_eval_pred6.npy"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  [ "$l" = "$(rsha "D:/tanitad/repo/$f" SHA256)" ] || stop "box 2 $f != tip $T"
  KEYSHA="$KEYSHA\"$f\": \"$l\", "
done
for kv in v9_labels_train.npz=83e30bd3791f0767707eade5f4671759 v9_labels_eval139.npz=6b5c7f207cffc3b7eb3cd527fd433599; do
  [ "${kv#*=}" = "$(rsha "D:/tanitad/data/v9/${kv%%=*}" MD5)" ] || stop "box 2 v9 ${kv%%=*} md5 != ${kv#*=}"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (8 key files sha-checked; both v9 releases md5-checked)"
# the reader runs HERE at the same tip, from the archive box 2 got (a fresh directory, never rm -rf)
X=$B/v8fmixo_tip_$(date +%Y%m%d%H%M%S); mkdir -p $X && tar -xf $B/repo_v8fmixo_tip.tar -C $X || stop "extract the reader"
RD="$(cygpath -m "$X/$PK/code")"
# ---- 2. the base is fixed (B = M2b): its checkpoints ARE the banked runs' (box 2 is idle: a different file costs nothing)
rm -f $CK
ckptsha_b "before the gate"
# ---- 3. the B side, staged into this chain's dirs BEFORE any training (a missing file stops the chain while box 2 is still idle)
for s in 0 1; do
  mkdir -p $E/runs/M2b_s$s
  for f in summary.json argv.json selfcheck.json; do cp "$L/v8f_mini/runs/M2b_s$s/$f" $E/runs/M2b_s$s/ || stop "stage the banked M2b_s$s/$f"; done
  cp "$L/v8f_mini/runs/M2b_s$s/train_log.jsonl" $E/runs/M2b_s$s/ 2>/dev/null || echo "$(date '+%m-%d %H:%M') NOTE: no banked M2b_s$s/train_log.jsonl: its p_gt schedule is NOT-CHECKED (by construction 1.0 -> 0.2)"
  cp $L/v8f_mini/eval/eval_M2b_s$s.npz $L/v8f_mini/eval/eval_M2b_s$s.json $E/eval/ || stop "stage eval_M2b_s$s"
  for f in eval_M2b_s$s-tacabs.npz eval_M2b_s$s-tacabs.json eval_M2b_s$s-tacorc.npz; do cp "$L/v8f_cz/eval/$f" $E/eval/ || stop "stage $f"; done
  cp "$L/v8f_cz/eval/eval_M2b_s$s-tacorc.json" $E/eval/ 2>/dev/null || echo "$(date '+%m-%d %H:%M') NOTE: no banked eval_M2b_s$s-tacorc.json (its mode is then unchecked)"
  # M2b's tacorc predates tac_code (WC Amendment 1): its GT constraints come from the all-goal oracle; M-LEAK needs the all-latcons oracle (same dir)
  for f in eval_M2b_s$s-tacorc-all-goal.npz eval_M2b_s$s-tacorc-all-latcons.npz; do cp "$L/v8f_tacgrp2/eval/$f" $E/eval/ || stop "stage $f"; done
  cp "$L/v8f_tacgrp2/eval/eval_M2b_s$s-tacorc-all-latcons.json" $E/eval/ 2>/dev/null
  # TAC_A2's dose chain on B (the same files RESULT_V8F_TAC_A2.md read): T's own code, the constraints oracle at alpha 1 (Amendment 1) and at alpha 0.50 (Amendment 2)
  for f in eval_M2b_s$s-tacpred.npz eval_M2b_s$s-tacorc-constraints.npz eval_M2b_s$s-tacorc-constraints.json eval_M2b_s$s-tacorc-constraints-a50.npz eval_M2b_s$s-tacorc-constraints-a50.json; do
    cp "$L/v8f_tacgrp2/eval/$f" $E/eval/ || stop "stage $f (the banked dose chain of B)"
  done
done
cp "$L/v8f_mini/eval/floors_perwindow.npz" $E/eval/ || stop "stage floors_perwindow.npz (the P10 reproduction guard)"
# the reader must load the REAL B side BEFORE any training: runs, evals, the PREREG's known values, the committed window counts, B's r(0.5), the identity control
PYTHONIOENCODING=utf-8 $PY "$RD/v8f_mixo_read.py" --check-inputs --eval-dir "$(cygpath -m $E/eval)" --runs-dir "$(cygpath -m $E/runs)" --ckpt-sha "$(cygpath -m $CK)" \
  --out "$(cygpath -m $E/check_inputs.json)" || stop "the reader cannot use the M2b side (check-inputs)"
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF "'^(ZZV8FMINI|" $B/v8f_guard.sh || stop "v8f_guard.sh does not know this chain's markers (anchored): it would never exit"
echo "" > $CUR
V8F_CHAIN_OUT=$L/chain_v8f_mixo.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh > $L/v8fmixo_guard.out 2>&1 &
# ---- 4. the launch gate for X, asserted here as well as in-process. GLINE is the line the queue PRINTS for the stage: a static test pins it to the line the stage RUNS.
GLINE=$(dry_stage "$QC" gatemixo | grep -a 'v8f_mini_train.py'); [ -n "$GLINE" ] || stop "the tip's queue prints no gatemixo line"
queue gatemixo 120 gatemixo
pull "gate_M2b-mixo/selfcheck.json" "$E/gate_M2b-mixo"
$PY - "$(cygpath -m $E/gate_M2b-mixo/selfcheck.json)" $HOLDOUT_SHA $M2B_STEPS "$(cygpath -m $E/runs/M2b_s0/summary.json)" "$GLINE" $M2B_WINDOWS <<'EOF' || stop "gate X assertions"
import json, sys
sc, hold, steps, b, line, windows = json.load(open(sys.argv[1])), sys.argv[2], int(sys.argv[3]), json.load(open(sys.argv[4])), sys.argv[5], int(sys.argv[6])
i, lv, bl = sc["info"], sc["levers"], b["levers"]
bad = []
# THE PASS TOKEN: the trainer's in-process gate wrote these three; without all three there is no token
if sc.get("declared_vs_built") != "PASS": bad.append("declared_vs_built")
if not sc["live"].get("pass") or sc["live"].get("control_prior_grad_sum") != 0.0: bad.append(f"live {sc['live'].get('pass')} / control {sc['live'].get('control_prior_grad_sum')}")
pr = sc.get("code_mix_probe")
if not isinstance(pr, dict) or pr.get("pass") is not True: bad.append(f"code_mix_probe {pr}")
else:
    if pr.get("code_mix") != "kind": bad.append(f"the code_mix probe was run for {pr.get('code_mix')!r}")
    if not (isinstance(pr.get("n_partial"), int) and pr["n_partial"] > 0): bad.append(f"the code_mix probe saw no PARTLY-GT code ({pr.get('n_partial')}): the mix did not reach the code O reads")
if (i["n_clips"], i["n_excluded"], i["n_excluded_declared"], i["excluded_in_train"]) != (4512, 60, 60, 0): bad.append("clips / holdout counts")
if i["excluded_sha256"] != hold: bad.append("the excluded set is not the RA-1 holdout")
if i["train_sha256"] != b["train_sha256"]: bad.append(f"TRAIN list {i['train_sha256'][:12]} != B's {b['train_sha256'][:12]}")
if i["kin_mode"] != "yaw": bad.append("kin_mode")
if (lv["wm"], lv["navonly"], lv["consistency"], lv.get("cons_absent")) != ("b", False, True, "mask"): bad.append(f"levers {lv}")
# THE LEVER, from the BUILT model: code_mix kind
if lv.get("code_mix") != "kind": bad.append(f"built code_mix {lv.get('code_mix')!r} != 'kind'")
# B's schedule is X's: M2b's is the trainer's default 1.0 -> 0.2; X moves the granularity of the mask and nothing else
if (lv.get("p_gt_start"), lv.get("p_gt_end")) != (1.0, 0.2): bad.append(f"built p_gt {lv.get('p_gt_start')} -> {lv.get('p_gt_end')}, expected 1.0 -> 0.2")
w = lv.get("w") or {}
for k in ("w_pen", "w_pen_self", "w_wm"):
    if w.get(k) != bl["w"][k]: bad.append(f"{k} {w.get(k)} != B's {bl['w'][k]}")
if (w.get("w_tac"), w.get("w_cons")) != (1.0, 0.5): bad.append(f"w_tac / w_cons {w.get('w_tac')} / {w.get('w_cons')} != B's 1.0 / 0.5 (a second lever moved)")
# THE PREREG'S CLAIMS: "19,164 steps" and "the windows ... [are] B's"
if i.get("n_windows_all_clips") != b.get("n_windows"): bad.append(f"the window set is not B's: n_windows_all_clips {i.get('n_windows_all_clips')} != B's {b.get('n_windows')}")
if i.get("n_windows_all_clips") != windows: bad.append(f"n_windows_all_clips {i.get('n_windows_all_clips')} != the PREREG's {windows}")
if sc["steps_total"] != steps: bad.append(f"steps {sc['steps_total']} != B's {steps}")
# B's history and window start are X's (the lever touches neither): hist 4, the arm's auto hmax 16, effective window start 3 = hist - 1
if lv.get("hist") != 4: bad.append(f"built hist {lv.get('hist')} != 4 (B's)")
if lv.get("hmax") != 16: bad.append(f"built hmax {lv.get('hmax')} != 16 (the arm's auto value, B's)")
if lv.get("window_start") != 3: bad.append(f"built window_start {lv.get('window_start')} != 3 (B's effective first window, hist - 1)")
if (i.get("window_start"), i.get("window_t_min"), i.get("ds_window_start")) != (3, 3, 3): bad.append(f"window start (asked / first t of the built window list / dataset clamp origin) {(i.get('window_start'), i.get('window_t_min'), i.get('ds_window_start'))} != (3, 3, 3)")
# THE LEVER ADDS NO PARAMETER: X's count equals B's in every group and in total
pg, pb = lv["params"], bl["params"]
if lv["n_params_trainable"] != bl["n_params_trainable"]: bad.append(f"n_params_trainable {lv['n_params_trainable']} != B's {bl['n_params_trainable']}")
for k in sorted(set(pg) | set(pb)):
    if pg.get(k, 0) != pb.get(k, 0): bad.append(f"parameter group {k}: {pg.get(k)} vs B's {pb.get(k)}")
# THE QUEUE'S OWN PRINTED LINE: --code-mix kind exactly once, --arm M2b, and none of the other levers' flags (X differs from B by this one flag)
toks = line.split()
if toks.count("--code-mix") != 1 or toks[toks.index("--code-mix") + 1] != "kind": bad.append("the queue's gatemixo line does not carry --code-mix kind exactly once")
if toks.count("--arm") != 1 or toks[toks.index("--arm") + 1] != "M2b": bad.append("the queue's gatemixo line is not --arm M2b")
for fl in ("--hist", "--hmax", "--window-start", "--p-gt-start", "--p-gt-end", "--grid", "--w-tac", "--w-cons"):
    if fl in toks: bad.append(f"the queue's gatemixo line sets {fl}: a second lever beside --code-mix")
if not i.get("lru"): bad.append("info.lru is not recorded in the selfcheck (the pipeline sizing must be on the record)")
print("X gate:", "PASS" if not bad else bad, "| clips", i["n_clips"], "steps", sc["steps_total"], "windows", i.get("n_windows_all_clips"), "| built code_mix", lv.get("code_mix"), "| built p_gt", lv.get("p_gt_start"), "->", lv.get("p_gt_end"),
      "| probe partial", (pr or {}).get("n_partial"), "of", (pr or {}).get("n_draws_samples"), "| params", lv["n_params_trainable"], "(B", bl["n_params_trainable"], ") | lru", i.get("lru"))
sys.exit(1 if bad else 0)
EOF
# the token record: the exact commit, the sha256 of the queue and of the gate's selfcheck, the key files box 2 was checked against, and the data manifests the gate read (written by the chain, never by hand)
$PY - "$(cygpath -m $E/gate_M2b-mixo/selfcheck.json)" "$(cygpath -m $E/gate_token.json)" "$T" "$(sha256sum $QT | cut -c1-64)" "{${KEYSHA%, }}" <<'EOF' || stop "write the gate token"
import hashlib, json, sys
sc = json.load(open(sys.argv[1]))
tok = {"prereg": "PREREG_V8F_MIXO", "gate": "PASS", "commit": sys.argv[3], "queue_sha256": sys.argv[4], "selfcheck_sha256": hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest(),
       "argv_sha": sc.get("argv_sha"), "key_files_sha256": json.loads(sys.argv[5]), "train_sha256": sc["info"]["train_sha256"], "excluded_sha256": sc["info"]["excluded_sha256"],
       "v9_md5": {"v9_labels_train.npz": "83e30bd3791f0767707eade5f4671759", "v9_labels_eval139.npz": "6b5c7f207cffc3b7eb3cd527fd433599"}}
json.dump(tok, open(sys.argv[2], "w"), indent=1)
print("gate token written for commit", tok["commit"])
EOF
# ---- 5. STAGE 1: seed 0, read against B
runone 0
mixoread s0 mixo_read_s0.json
M1=$(jget mixo_read_s0.json reading); echo "$(date '+%m-%d %H:%M') $M1"
[ "$(jget mixo_read_s0.json DECISION)" = CONTINUE ] || finish "$M1 -- seed 1 not trained (committed): $(jget mixo_read_s0.json next)"
# ---- 6. STAGE 2: seed 1, the final reading
runone 1
mixoread final mixo_read.json
finish "$M1 | $(jget mixo_read.json reading) | $(jget mixo_read.json next)"
