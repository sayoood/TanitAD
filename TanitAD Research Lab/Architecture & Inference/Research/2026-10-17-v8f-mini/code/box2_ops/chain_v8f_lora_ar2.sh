#!/usr/bin/env bash
# PREREG_V8F_LORA on box 2: Amendment 2's SAME-DEVICE identity control, then the AR-2 smoke (proof 4). Instruments only: no training, ONE short GPU job at a time under SAFE MODE (mk_v8f_job.py).
# ARMED ONLY AFTER PREREG_V8F_DSCALE's chain ends, and only on the PI's box-2 go for AR-2. Derived from chain_v8f_dscale.sh; markers ZZV8FMINI-DONE / -STOP in ITS OWN log (chain_v8f_lora_ar2.out),
# which v8f_guard.sh's anchored end test already knows.
#   chain_v8f_lora_ar2.sh --dry-run   PRINTS every command the chain would run -- the dev-side frame build, the ship, the preflight, the two queue stages' lines (from the tip's queue or $V8F_QUEUE) and
#                                     the pulls -- and exits; nothing is built, shipped, launched or waited for. The line printed IS the line that runs.
#   0. wait for the predecessor (default: PREREG_V8F_DSCALE's chain, logs/chain_v8f_dscale.out, ZZV8FMINI-DONE; stop on its ZZV8FMINI-STOP); BOX2_BUSY marker; boot check
#   1. refresh box 2's code from the REMOTE tip; sha-check the key files; extract the tip HERE (the dev-side builder, the driver and the reader all run from the SAME archive box 2 got)
#   2. pull box 2's cache manifests (train, eval); PREFLIGHT on them, no GPU: the declared train clips are the first 4 of B's list in sha12 order, the declared eval clips the first 2 of the 141, and every
#      declared row reads source=local256 / enc_dtype=bf16 (v8f_lora_identity_dev.py --manifest-only); the queue's printed lines carry the same ids, --max-clips 4 and NO --amp
#   3. DEV-SIDE: build the declared clips' .v2ep.pt with the PRODUCTION builder (pod_build_b1_epcache.py, the route raw/2026-10-25-lora-route/ROUTE.md reproduced byte for byte; CPU only, <= 8 clips, ~37 MB each),
#      check them tensor by tensor against the banked production frames, then ship them to box 2 (%LORAFRAMES% = D:\tanitad\data\lora_ar2_frames\{train,eval}) and check every sha256 on the far side
#   4. identitylora (the same-device control + its mutant) -> pull -> v8f_lora_ar2_read.py --identity: STOP unless the control PASSED (qualifying) and the mutant FAILED
#   5. smokelora (200 steps at batch 1, bf16, --amp OFF, on the 4 declared train clips) -> pull into logs/v8f_lora_ar2/ -> v8f_lora_ar2_read.py --smoke: step time, peak memory, the projected hours of one
#      19,164-step seed at effective batch 64 (ESTIMATED) and the venue rule (<= 36 h per seed => box 2) -> DONE
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4), any nonzero rc, or a guard stop. Script polling, not model turns.
# v8f_guard.sh's kill_ours matches v8f_mini_train|v8f_mini_eval|v8f_lora_smoke|v8f_lora_identity_dev (widened by the MM 2026-10-25, pinned by
# test_the_guard_can_kill_every_python_script_the_ar2_stages_launch), so a TEMPERATURE stop kills either job mid-run; the chain also greps the guard's log after each job.
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
dry_stage() {   # dry_stage <queue file> <stage args...>: the line(s) the queue prints for a stage under V8F_DRYRUN. cmd.exe cannot be handed `"<path with an ampersand>" args`
                # through MSYS quoting (it strips the quotes it needs): a one-line wrapper batch file can
  local qf=$1; shift
  local w=${TMPDIR:-/tmp}/v8floraar2_dry_$$.cmd
  printf '@echo off\r\nset V8F_DRYRUN=1\r\ncall "%s" %s\r\n' "$(cygpath -w "$qf")" "$*" > "$w"
  MSYS_NO_PATHCONV=1 cmd /c "$(cygpath -w "$w")" </dev/null | tr -d '\r'
}
# the DECLARED clips: the first 4 train clips of B's list in sha12 order (--exclude 540:600 applied; all inside the first 600, whose production frames exist) and the first 2 of the 141 eval clips.
# The queue (:loraenv) carries the same lists; the preflight below proves both equal the manifest's.
TRAIN_CLIPS="8f387977-4086-466d-bba4-b04f8cdc0743 bf5ec76e-e8b5-4f0d-a9cc-4fe1687d3d3a 0e1526b3-d2b6-46b2-95fa-2903bb34e2ec 83e32d0f-ed19-4116-8c1c-9e4c843de993"
EVAL_CLIPS="dca157a0-20a9-464b-a74c-084b678e7be1 e6cb7cd4-8349-4b9d-9674-e3ebf789e611"
N_TRAIN=4
PY=${PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
# dev-side inputs: the staged roots the production frames were built from (mp4 + timestamps + egomotion zip + calibration) and the banked production frames themselves
STAGE_TRAIN=${STAGE_TRAIN:-C:/Users/Admin/tanitad-data/refav1-train600-256_stage}
STAGE_EVAL=${STAGE_EVAL:-C:/Users/Admin/tanitad-data/refav1-eval141-256rb_stage}
BANKED_TRAIN=${BANKED_TRAIN:-C:/Users/Admin/tanitad-data/refav1-train600-256}
BANKED_EVAL=${BANKED_EVAL:-C:/Users/Admin/tanitad-data/refav1-eval141-256rb}
RFR=D:/tanitad/data/lora_ar2_frames          # on box 2; the queue's %LORAFRAMES% default is the same directory
R=D:/tanitad/runs/v8f_mini_full
devbuild_cmd() {   # devbuild_cmd <tip dir> <stage root> <out dir> <ids file>: the PRODUCTION builder, CPU only, exactly the S4b / S4c recipe (launch_build_train600.ps1, launch_build_eval141.ps1) but from the tip archive
  echo "env CUDA_VISIBLE_DEVICES=-1 TANITAD_STACK=$1/stack PYTHONPATH=$1/stack PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 V2_TORCH_THREADS=1 PAI_DECODE_THREADS=1 $PY -u $1/stack/scripts/pod_build_b1_epcache.py --out $3 --root $2 --workers 4 --clips $4"
}
pycmd() {   # pycmd <script dir> <script> <args...>: the python line of a reader / driver run, the script path QUOTED (the repo path carries spaces and an ampersand). The dry run prints it, the chain evals it
  local d=$1 s=$2; shift 2
  echo "PYTHONIOENCODING=utf-8 $PY \"$d/$s\" $*"
}
shipcmds() {   # shipcmds <local frames dir>: the scp lines that put the declared clips on box 2
  local c
  for c in $TRAIN_CLIPS; do echo "scp -q -o BatchMode=yes -o ConnectTimeout=20 $1/train/$c.v2ep.pt tanitad-box2:$RFR/train/"; done
  for c in $EVAL_CLIPS; do echo "scp -q -o BatchMode=yes -o ConnectTimeout=20 $1/eval/$c.v2ep.pt tanitad-box2:$RFR/eval/"; done
}
PULL_IDENT="identity_lora.json identity_lora_mutant.json identity_lora.log identity_lora_mutant.log"
PULL_SMOKE="smoke_report.json smoke_steps.jsonl smoke.log run/selfcheck.json run/summary.json run/argv.json run/train_log.jsonl"
if [ "${1:-}" = "--dry-run" ]; then
  QF=${V8F_QUEUE:-}
  if [ -z "$QF" ]; then
    MIRROR=${MIRROR:-/c/Users/Admin/tanitad-push}; BR=${BR:-agent/arch-inf-20260803}
    T=$(git -C $MIRROR ls-remote origin $BR | cut -f1); QF=${TMPDIR:-/tmp}/v8floraar2_queue_dry_$T.cmd
    git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > "$QF" || { echo "ZZV8FMINI-STOP no queue at the tip"; exit 1; }
  fi
  X=${X:-<TIPDIR>}; FR=${FR:-<FRAMES>}
  echo "== dev-side build, train ($N_TRAIN clips)"; devbuild_cmd "$X" "$STAGE_TRAIN" "$FR/train" "$FR/train_ids.txt"
  echo "== dev-side build, eval (2 clips)"; devbuild_cmd "$X" "$STAGE_EVAL" "$FR/eval" "$FR/eval_ids.txt"
  echo "== check the built frames against the banked production frames"
  pycmd "$X/$PK/code" v8f_lora_ar2_read.py --compare-frames --built "$FR" --banked "$BANKED_TRAIN" "$BANKED_EVAL" --train-clips $TRAIN_CLIPS --eval-clips $EVAL_CLIPS --out "<E>/frames_vs_banked.json"
  echo "== ship the frames to box 2"; shipcmds "$FR"
  echo "== preflight (no GPU): the declared clips vs box 2's manifests, and the driver's --manifest-only"
  pycmd "$X/$PK/code" v8f_lora_ar2_read.py --check-declared --manifest-train "<E>/cache/train/manifest.jsonl" --manifest-eval "<E>/cache/eval/manifest.jsonl" --train-clips $TRAIN_CLIPS --eval-clips $EVAL_CLIPS --n-train $N_TRAIN --exclude 540:600
  pycmd "$X/$PK/code" v8f_lora_identity_dev.py --manifest-only --cache-root "<E>/cache" --train-clips $TRAIN_CLIPS --eval-clips $EVAL_CLIPS --out "<E>/preflight_manifest_only.json"
  for st in "identitylora" "smokelora"; do
    echo "== $st"; dry_stage "$QF" $st || exit 1
  done
  echo "== pull identity: $PULL_IDENT"; echo "== pull smoke: $PULL_SMOKE"
  echo "== read"
  pycmd "$X/$PK/code" v8f_lora_ar2_read.py --identity "<E>/identity" --train-clips $TRAIN_CLIPS --eval-clips $EVAL_CLIPS --expect-driver-sha "<sha256 of the tip's v8f_lora_identity_dev.py>" --out "<E>/identity_read.json"
  pycmd "$X/$PK/code" v8f_lora_ar2_read.py --identity "<E>/identity" --smoke "<E>/smoke" --train-clips $TRAIN_CLIPS --eval-clips $EVAL_CLIPS --expect-driver-sha "<sha256 of the tip's v8f_lora_identity_dev.py>" --out "<E>/ar2_read.json"
  exit 0
fi
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_lora_ar2; mkdir -p $E/cache/train $E/cache/eval $E/identity $E/smoke/run
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
HJ="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
B1="TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
HOLDOUT_SHA=f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9
IDENT_POLLS=120; SMOKE_POLLS=240   # 1 h and 2 h of 30 s polls: the identity control is ~minutes (two model loads + 6 clips x 2 readings), the smoke ~30 min ESTIMATED
TAG=$(date '+%m%d%H%M')
CUR=$L/v8floraar2_current.txt
AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_dscale.out}; AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; exit 1; }
waitrc() {   # waitrc <job> <polls of 30 s>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; grep ZZGUARD-STOP $L/v8floraar2_guard.out 2>/dev/null; stop "$1 rc=$rc"; }
}
queue() {    # queue <job> <polls> <queue args...>: every job name carries this chain's TAG, so an old log's ZZJOB-END can never satisfy a new wait
  local j="v8flora${TAG}_$1" m=$2; shift 2
  $PY "$(cygpath -m $B)/mk_v8f_job.py" $j "$@" >/dev/null && $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship $j"
  rm -f $L/$j.log
  launch $j || stop "launch $j"
  waitrc $j $m
}
rsha() { $SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm $2 -LiteralPath '$1').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{32,64}$'; }
pull() {     # pull <box-2 path under R> <local dir>
  mkdir -p "$2"; $SCP "tanitad-box2:$R/$1" "$2/" || stop "pull $R/$1"
}
finish() {   # finish <summary>: the committed end of the chain
  $SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
  $SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_v8floraar2.csv 2>/dev/null
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
echo "v8F-mini PREREG_V8F_LORA AR-2 (Amendment 2's same-device identity control, then the 200-step smoke; no training) since $(date '+%m-%d %H:%M'), ~1 h. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
QT=$B/v8floraar2_queue_$TAG.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" | tr -d '\r' > $QT
QC=$B/v8floraar2_queue_$TAG.crlf.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > $QC      # the committed bytes (CRLF): what cmd.exe can run for the dry-run line
for st in identitylora smokelora loraenv; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done
for f in v8f_lora_identity_dev.py v8f_lora_smoke.py v8f_lora_ar2_read.py; do
  git -C $MIRROR cat-file -e "$T:$PK/code/$f" || stop "the tip has no $f"
done
for f in stack/tanitad/models/v8f_lora_encoder.py stack/scripts/dinov3_fp8_encode_ship.py stack/scripts/pod_build_b1_epcache.py "$HJ/code/pull_encode_pooled.py"; do
  git -C $MIRROR cat-file -e "$T:$f" || stop "the tip has no $f (the on-the-fly encoder imports the cache builder by file path)"
done
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q -- '"--encoder"' || stop "the tip's trainer has no --encoder flag"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q -- '"--frames-root"' || stop "the tip's trainer has no --frames-root flag"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q -- '"--max-clips"' || stop "the tip's trainer has no --max-clips flag"
git -C $MIRROR show "$T:$PK/code/v8f_lora_smoke.py" | grep -q -- '"--amp", action="store_true"' || stop "the tip's smoke script still appends --amp (Amendment 2 pins it OFF)"
git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8floraar2_tip.tar $T -- stack taniteval "$HJ" "$PK" "$B1" || stop "archive"
$SCP $B/repo_v8floraar2_tip.tar tanitad-box2:D:/tanitad/repo_v8floraar2_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8floraar2_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8FLORAAR2.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8flora${TAG}_refresh.cmd
$SCP $B/v8flora${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8flora${TAG}_refresh || stop "launch refresh"
waitrc v8flora${TAG}_refresh 40
for f in stack/scripts/v8f_mini_train.py stack/scripts/v8f_mini_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py stack/tanitad/models/hjepa.py \
         stack/tanitad/models/v8f_lora_encoder.py stack/scripts/dinov3_fp8_encode_ship.py "$HJ/code/pull_encode_pooled.py" \
         "$PK/code/v8f_mini_box2_queue.cmd" "$PK/code/v8f_lora_identity_dev.py" "$PK/code/v8f_lora_smoke.py" "$HJ/raw/p10_yaw4512/p10_prior.pt" "$HJ/raw/p10_yaw4512/p10_eval_pred6.npy"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  [ "$l" = "$(rsha "D:/tanitad/repo/$f" SHA256)" ] || stop "box 2 $f != tip $T"
done
for kv in v9_labels_train.npz=83e30bd3791f0767707eade5f4671759 v9_labels_eval139.npz=6b5c7f207cffc3b7eb3cd527fd433599; do
  [ "${kv#*=}" = "$(rsha "D:/tanitad/data/v9/${kv%%=*}" MD5)" ] || stop "box 2 v9 ${kv%%=*} md5 != ${kv#*=}"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (13 key files sha-checked; both v9 releases md5-checked)"
# the dev-side builder, the driver and the reader run HERE from the same archive (a fresh directory, never rm -rf)
X=$B/v8floraar2_tip_$(date +%Y%m%d%H%M%S); mkdir -p $X && tar -xf $B/repo_v8floraar2_tip.tar -C $X || stop "extract the tip"
RD="$(cygpath -m "$X/$PK/code")"
DRV_SHA=$(sha256sum "$X/$PK/code/v8f_lora_identity_dev.py" | cut -c1-64)
# ---- 2. the preflight, no GPU: box 2's manifests (the lists the cache and B were built on)
$SCP tanitad-box2:D:/tanitad/cache_full/train/manifest.jsonl $E/cache/train/manifest.jsonl || stop "pull box 2's cache train manifest"
$SCP tanitad-box2:D:/tanitad/cache_full/eval/manifest.jsonl $E/cache/eval/manifest.jsonl || stop "pull box 2's cache eval manifest"
EM=$(cygpath -m $E)
eval "$(pycmd "$RD" v8f_lora_ar2_read.py --check-declared --manifest-train "$EM/cache/train/manifest.jsonl" --manifest-eval "$EM/cache/eval/manifest.jsonl" --train-clips $TRAIN_CLIPS --eval-clips $EVAL_CLIPS --n-train $N_TRAIN --exclude 540:600)" \
  || stop "the declared clips are not the manifest's first (check-declared)"
eval "$(pycmd "$RD" v8f_lora_identity_dev.py --manifest-only --cache-root "$EM/cache" --train-clips $TRAIN_CLIPS --eval-clips $EVAL_CLIPS --out "$EM/preflight_manifest_only.json")" \
  || stop "a declared manifest row is not a production row (the driver's --manifest-only)"
# the queue's own printed lines carry the declared ids, the smoke's --max-clips and NO --amp: what is PRINTED is what RUNS (a static test pins that)
ILINES=$(dry_stage "$QC" identitylora | grep -a 'v8f_lora_identity_dev.py'); SLINE=$(dry_stage "$QC" smokelora | grep -a 'v8f_lora_smoke.py')
[ -n "$ILINES" ] && [ -n "$SLINE" ] || stop "the tip's queue prints no identitylora / smokelora line"
for c in $TRAIN_CLIPS $EVAL_CLIPS; do echo "$ILINES" | grep -q -- "$c" || stop "the queue's identitylora line does not carry $c"; done
[ "$(echo "$SLINE" | grep -c -- "--max-clips $N_TRAIN ")" = 1 ] || stop "the queue's smokelora line does not carry --max-clips $N_TRAIN exactly once"
echo "$SLINE" | grep -q -- "--amp" && stop "the queue's smokelora line carries --amp (Amendment 2 pins it OFF)"
echo "$SLINE" | grep -q -- "--exclude 540:600" || stop "the queue's smokelora line lacks B's --exclude 540:600"
# ---- 3. DEV-SIDE: the declared clips' frames by the PRODUCTION builder, checked against the banked production frames, then shipped
FR=$B/v8floraar2_frames_$TAG; mkdir -p $FR/train $FR/eval
printf '%s\n' $TRAIN_CLIPS > $FR/train_ids.txt; printf '%s\n' $EVAL_CLIPS > $FR/eval_ids.txt
$(devbuild_cmd "$(cygpath -m $X)" "$STAGE_TRAIN" "$(cygpath -m $FR/train)" "$(cygpath -m $FR/train_ids.txt)") > $FR/build_train.log 2>&1 || { tail -5 $FR/build_train.log; stop "the production builder failed (train)"; }
$(devbuild_cmd "$(cygpath -m $X)" "$STAGE_EVAL" "$(cygpath -m $FR/eval)" "$(cygpath -m $FR/eval_ids.txt)") > $FR/build_eval.log 2>&1 || { tail -5 $FR/build_eval.log; stop "the production builder failed (eval)"; }
for c in $TRAIN_CLIPS; do [ -s $FR/train/$c.v2ep.pt ] || stop "the builder wrote no frames for train clip $c"; done
for c in $EVAL_CLIPS; do [ -s $FR/eval/$c.v2ep.pt ] || stop "the builder wrote no frames for eval clip $c"; done
eval "$(pycmd "$RD" v8f_lora_ar2_read.py --compare-frames --built "$(cygpath -m $FR)" --banked "$BANKED_TRAIN" "$BANKED_EVAL" --train-clips $TRAIN_CLIPS --eval-clips $EVAL_CLIPS --out "$EM/frames_vs_banked.json")" \
  || stop "the rebuilt frames differ from the banked production frames (compare-frames)"
(cd $FR && sha256sum train/*.v2ep.pt eval/*.v2ep.pt) > $E/frames_sha256.txt
cp $E/frames_sha256.txt $FR/frames_sha256.txt
$SSH "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force $RFR/train, $RFR/eval | Out-Null\"" 2>/dev/null
for c in $TRAIN_CLIPS; do $SCP $FR/train/$c.v2ep.pt tanitad-box2:$RFR/train/ || stop "ship train frames $c"; done
for c in $EVAL_CLIPS; do $SCP $FR/eval/$c.v2ep.pt tanitad-box2:$RFR/eval/ || stop "ship eval frames $c"; done
for c in $TRAIN_CLIPS; do [ "$(sha256sum $FR/train/$c.v2ep.pt | cut -c1-64)" = "$(rsha "$RFR/train/$c.v2ep.pt" SHA256)" ] || stop "box 2's train frames $c != the local build (sha256)"; done
for c in $EVAL_CLIPS; do [ "$(sha256sum $FR/eval/$c.v2ep.pt | cut -c1-64)" = "$(rsha "$RFR/eval/$c.v2ep.pt" SHA256)" ] || stop "box 2's eval frames $c != the local build (sha256)"; done
echo "$(date '+%m-%d %H:%M') frames built by the production builder, equal to the banked ones tensor by tensor, shipped and sha256-checked ($(echo $TRAIN_CLIPS $EVAL_CLIPS | wc -w) clips)"
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF "'^(ZZV8FMINI|" $B/v8f_guard.sh || stop "v8f_guard.sh does not know this chain's markers (anchored): it would never exit"
echo "" > $CUR
V8F_CHAIN_OUT=$L/chain_v8f_lora_ar2.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh > $L/v8floraar2_guard.out 2>&1 &
# ---- 4. the same-device identity control (+ its mutant): pulled even when it FAILS (the numbers are the evidence), judged HERE
queue identitylora $IDENT_POLLS identitylora
for f in $PULL_IDENT; do pull "lora_ar2/identity/$f" "$E/identity"; done
grep ZZGUARD-STOP $L/v8floraar2_guard.out 2>/dev/null && stop "guard stop"
eval "$(pycmd "$RD" v8f_lora_ar2_read.py --identity "$EM/identity" --train-clips $TRAIN_CLIPS --eval-clips $EVAL_CLIPS --expect-driver-sha $DRV_SHA --out "$EM/identity_read.json")" \
  || stop "the same-device identity control did not PASS (or its mutant was not caught): see $E/identity_read.json -- the arm is TWO levers until it does (Amendment 2's fallback applies)"
echo "$(date '+%m-%d %H:%M') identity control PASS, mutant caught"
# ---- 5. the AR-2 smoke
queue smokelora $SMOKE_POLLS smokelora
for f in $PULL_SMOKE; do pull "lora_ar2/smoke/$f" "$E/smoke/$(dirname $f | sed 's#^\.$##')"; done
grep ZZGUARD-STOP $L/v8floraar2_guard.out 2>/dev/null && stop "guard stop"
eval "$(pycmd "$RD" v8f_lora_ar2_read.py --identity "$EM/identity" --smoke "$EM/smoke" --train-clips $TRAIN_CLIPS --eval-clips $EVAL_CLIPS --expect-driver-sha $DRV_SHA --out "$EM/ar2_read.json")" \
  > $E/ar2_read.out || { cat $E/ar2_read.out; stop "the AR-2 smoke did not pass its checks: see $E/ar2_read.json"; }
S1=$(head -1 $E/ar2_read.out); echo "$(date '+%m-%d %H:%M') $S1"
finish "$S1"
