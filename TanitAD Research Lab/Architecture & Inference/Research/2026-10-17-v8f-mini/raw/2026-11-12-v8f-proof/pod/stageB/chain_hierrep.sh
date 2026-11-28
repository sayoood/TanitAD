#!/usr/bin/env bash
# chain_hierrep.sh -- Stage B of PREREG_V8F_PROOF_ADD11 (HIERREP): gate, train, bank, T1-kin-dump 24 arms (HIER FLAT DAG FLATDAG x seeds 0..5), then read.  Runs DETACHED on the pod.
# Order: r1 (harness) and r2 (deliberate-regression gate) once, then per seed s = 0..5 the WHOLE seed before the next one (HIER_s, FLAT_s, bank of each, DAG_s, FLATDAG_s, a T1-kin dump
# after every arm), so a killed chain leaves complete seeds (control r5).  Resumable: an arm whose token matches, whose summary says done at the final step and whose dump is complete is skipped.
# Fail closed: any non-zero rc, a missing final step or a token / argv mismatch writes STAGEB_STOP.txt and exits 1.  Launches nothing but its own jobs, one at a time.
#   chain_hierrep.sh             run     |   chain_hierrep.sh --dry-run   print every gate and train argv (no file touched, no wait) and exit 0
# Files beside it ($SB): PREREG_V8F_PROOF_ADD11.md (the card), gate_token.py, r2_regress_gate.py, hierrep_read.py, HIERREP_DEFS.md, r1_reference.json (optional: absent = r1 PENDING).
set -u
W=${HR_W:-/workspace/v8proof}
PY=${HR_PY:-/workspace/venv/bin/python}
TREE=${HR_TREE:-$W/tree}
HR=${HR_DIR:-$W/hierrep}
SB=${HR_SB:-$HR/stageB}
RUN=${HR_RUN:-$W/runs/hierrep}
T1=$RUN/t1
BANK=$RUN/bank
GATE=$RUN/gate
ST=$HR/stageB_state
LOG=$HR/stageB.log
POLL=${HR_POLL_S:-60}
SEEDS=${HR_SEEDS:-"0 1 2 3 4 5"}
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
CARD_SHA=94647519b965e5d0ac5b6decc0475719ed16a57757decd6a1994ebb25ee29d1c
EXPECT_COMMIT_PREFIX=3e264bbf   # the pod tree is this commit (lead's statement); the four key-file sha256 below are the git blobs at it
# pinned key files of the commit (sha256 of the git blobs at 3e264bbf, read from the mirror)
PIN_TRAIN=465a7f08078103c4b07c0cb311c096e6602ceda9f32c37669d65f1a7d5180cef
PIN_T1=61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397
PIN_QTOOL=a2b70d425c7b2a8b3cbcd5474b5d1eb63c05d330cb130caa0a4eb9d783874ddd
PIN_BANK=893dd6547b361c34fd2307f9efaf0b9bf822a1476506d3aa65498d7dbba53da3
# inputs
CACHE=$W/data/cache_pod
V9=$W/data/v9
INP=$W/data/inputs
TRACKS=$INP/native_tracks.pt
P10=$INP/p10_prior.pt
JOINT=$INP/join_train.npz
EVALC=$W/art/sfresh_inputs/cache          # box-2-equal eval cache (uploaded): control r1 and the reference side of r6
EVALC_ARM=${HR_EVALC_ARM:-$CACHE}         # pod-built eval cache: every arm of the card (note N1)
EVR="$W/art/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
PINS=$HR/pins
GRID=$PINS/eval_grid_windows.json
JOINE=$PINS/join_eval141.npz
REFCK=$W/pulls/stopbal_BAL-NULL_s0

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[stageB $(ts)] $*" | tee -a "$LOG"; }
DONE_ARMS=""
status() { # status <state> <last>
  local arms; arms=$(echo "$DONE_ARMS" | sed 's/^,//; s/[A-Za-z0-9_]\+/"&"/g')
  printf '{"updated":"%s","state":"%s","last":"%s","arms_done":[%s],"n_done":%s}\n' "$(ts)" "$1" "$2" "$arms" "$(echo "$DONE_ARMS" | tr ',' '\n' | grep -c .)" > "$HR/STAGEB_STATUS.json"
}
stop() { # stop <step> <detail>
  log "STOP $1: ${2:-}"
  echo "$1: ${2:-} ($(ts))" > "$HR/STAGEB_STOP.txt"
  status STOPPED "$1"
  exit 1
}
sha_of() { sha256sum "$1" | cut -c1-64; }
# test-only escape (stub runs off the pod): honoured ONLY when the work root is not under /workspace
nopin() { [ "${HR_TEST_NOPIN:-0}" = 1 ] && [ "${W#/workspace}" = "$W" ]; }
pin_ok() { [ "$(sha_of "$1")" = "$2" ] || nopin; }        # pin_ok <file> <full sha256>
pre_ok() { case $(sha_of "$1") in $2*) return 0;; esac; nopin; }   # pre_ok <file> <sha256 prefix>

# ---------------------------------------------------------------- argv builders (the box-2 queue's lines, pod paths)
A=(--cache-dir "$CACHE" --v9-train "$V9/v9_labels_train.npz" --v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train "$V9/lead_train4572.npz"
   --speed-train "$V9/refcv6_speed_max_v8_train.jsonl" --join-train "$JOINT" --p10 "$P10" --kin-mode yaw --tracks "$TRACKS" --exclude 540:600 --batch 64 --steps-mult 4 --device cuda)
BANKSHA_DRY=BANK_SHA256_OF_BANK_JSON
build_argv() { # build_argv <KIND> <seed> <out>  -> ARGV
  local kind=$1 s=$2 out=$3 cut=() dagf=() sha parent
  [ "$kind" = FLAT ] && cut=(--o-interface cut)
  ARGV=(--arm M2b --seed "$s" --grid pool4x10 ${cut[@]+"${cut[@]}"} --out "$out" "${A[@]}")
  case $kind in
    DAG|FLATDAG)
      if [ "$kind" = DAG ]; then parent=$RUN/HIER_s$s/ckpt_last.pt; else parent=$RUN/FLAT_s$s/ckpt_last.pt; fi
      if [ "$DRY" = 1 ]; then sha=$BANKSHA_DRY; else sha=$(tr -d '\r\n ' < "$BANK/$([ "$kind" = DAG ] && echo HIER || echo FLAT)_s$s/bank_sha256.txt" 2>/dev/null); fi
      [ "$DRY" = 1 ] || echo "$sha" | grep -Eq '^[0-9a-f]{64}$' || return 1
      dagf=(--init-ckpt "$parent" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0)
      [ "$kind" = FLATDAG ] && dagf+=(--o-interface cut)
      dagf+=(--dagger-p 0.1 --dagger-bank "$BANK/$([ "$kind" = DAG ] && echo HIER || echo FLAT)_s$s" --dagger-bank-sha256 "$sha" --dagger-seed "$s")
      ARGV+=("${dagf[@]}") ;;
  esac
}
steps_of() { case $1 in HIER|FLAT) echo 19164;; *) echo 3000;; esac; }

if [ "$DRY" = 1 ]; then
  for s in $SEEDS; do for k in HIER FLAT DAG FLATDAG; do
    build_argv $k $s "$RUN/${k}_s$s"; echo "TRAIN ${k}_s$s: $PY $TREE/stack/scripts/v8f_mini_train.py ${ARGV[*]}"
    build_argv $k $s "$GATE/${k}_s$s/out"; echo "GATE  ${k}_s$s: $PY $TREE/stack/scripts/v8f_mini_train.py ${ARGV[*]} --selfcheck-only"
  done; done
  echo "T1    : $PY $TREE/stack/scripts/v8f_t1_eval.py run --ckpt <ckpt_last.pt> --tag <ARM>_s<S>_m10_causal_heads --tracks $TRACKS --eval-cache $EVALC --eval-grid $GRID --eval-join $JOINE --out-dir $T1/<ARM>_s<S> --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks $T1/<ARM>_s<S>/ticks --dump-heads --device cuda"
  echo "BANK  : $PY $TREE/stack/scripts/v8f_dagger_queue_tool.py bank --n-shards 4 --out-dir $BANK/<HIER|FLAT>_s<S> -- --ckpt <parent>/ckpt_last.pt --m2b-argv <parent>/argv.json --cache-dir $CACHE --tracks $TRACKS --join-train $JOINT --eval-grid $GRID --batch 32 --threads 1 --device cuda --expect-clips 4512 --expect-windows 306566"
  exit 0
fi

mkdir -p "$HR" "$RUN" "$T1" "$BANK" "$GATE" "$ST" "$PINS" || { echo "cannot create $HR"; exit 1; }
rm -f "$HR/STAGEB_STOP.txt" 2>/dev/null   # a re-run after the lead resolved a STOP starts clean; STAGEB_DONE.txt is never removed here
if command -v flock >/dev/null 2>&1; then exec 9>"$HR/stageB.lock"; flock -n 9 || { echo "another chain holds the lock"; exit 1; }; fi
export PYTHONPATH=$TREE/stack:$TREE/taniteval
export OMP_NUM_THREADS=${HR_THREADS:-6} MKL_NUM_THREADS=${HR_THREADS:-6} OPENBLAS_NUM_THREADS=${HR_THREADS:-6}   # 7.65-core quota: 6 for the main process, the rest for loader workers
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TREE
status RUNNING start
log "chain start; seeds: $SEEDS; tree $TREE; runs $RUN"

# ---------------------------------------------------------------- A. wait for stage A, verify the card, the tree and the inputs
while [ ! -f "$HR/STAGEA_DONE.txt" ]; do
  [ -f "$HR/STAGEA_STOP.txt" ] && stop A-WAIT "stage A stopped: $(head -c 300 "$HR/STAGEA_STOP.txt")"
  sleep "$POLL"
done
log "STAGEA_DONE.txt present"
[ -f "$ST/done_A" ] || {
  [ -f "$SB/PREREG_V8F_PROOF_ADD11.md" ] || stop A-CARD "the card copy is not next to the chain ($SB)"
  [ "$(sha_of "$SB/PREREG_V8F_PROOF_ADD11.md")" = "$CARD_SHA" ] || stop A-CARD "card sha256 != $CARD_SHA"
  for f in gate_token.py r2_regress_gate.py hierrep_read.py; do [ -f "$SB/$f" ] || stop A-FILES "missing $SB/$f"; done
  pin_ok "$TREE/stack/scripts/v8f_mini_train.py" "$PIN_TRAIN" || stop A-TREE "v8f_mini_train.py differs from commit $EXPECT_COMMIT_PREFIX"
  pin_ok "$TREE/stack/scripts/v8f_t1_eval.py" "$PIN_T1" || stop A-TREE "v8f_t1_eval.py differs from commit $EXPECT_COMMIT_PREFIX"
  pin_ok "$TREE/stack/scripts/v8f_dagger_queue_tool.py" "$PIN_QTOOL" || stop A-TREE "v8f_dagger_queue_tool.py differs from commit $EXPECT_COMMIT_PREFIX"
  pin_ok "$TREE/stack/scripts/v8f_dagger_bank.py" "$PIN_BANK" || stop A-TREE "v8f_dagger_bank.py differs from commit $EXPECT_COMMIT_PREFIX"
  COMMIT=""; COMMIT_OK=false
  [ -f "$TREE/COMMIT" ] && COMMIT=$(head -1 "$TREE/COMMIT" | tr -d '\r\n ')
  [ -z "$COMMIT" ] && [ -e "$TREE/.git" ] && COMMIT=$(git -C "$TREE" rev-parse HEAD 2>/dev/null)
  if [ -n "$COMMIT" ]; then
    case $COMMIT in $EXPECT_COMMIT_PREFIX*) COMMIT_OK=true;; *) stop A-TREE "tree commit $COMMIT is not $EXPECT_COMMIT_PREFIX";; esac
  else COMMIT="${EXPECT_COMMIT_PREFIX} (unverified: the tree names no commit; the four key files equal the pinned blobs)"; log "WARNING commit not named by the tree; key-file sha256 pins hold"; fi
  echo "$COMMIT" > "$ST/commit.txt"; echo "$COMMIT_OK" > "$ST/commit_ok.txt"
  if [ -f "$V9/v9_labels_train.npz" ]; then { [ "$(md5sum "$V9/v9_labels_train.npz" | cut -c1-32)" = 83e30bd3791f0767707eade5f4671759 ] || nopin; } || stop A-INPUTS "v9_labels_train.npz md5 differs"; fi
  pre_ok "$JOINT" 978f41bf || stop A-INPUTS "join_train.npz sha256 != 978f41bf..."
  pre_ok "$TRACKS" 1db431f1 || stop A-INPUTS "native_tracks.pt sha256 != 1db431f1..."
  pre_ok "$P10" 14a062a9 || stop A-INPUTS "p10_prior.pt sha256 != 14a062a9..."
  for f in "$V9/v9_labels_train.npz" "$V9/lead_train4572.npz" "$V9/refcv6_speed_max_v8_train.jsonl"; do
    if [ -f "$f" ]; then echo "$(sha_of "$f")  $f" >> "$HR/INPUTS_SHA256.txt"
    else echo "ABSENT-NAMED-ONLY  $f  (named in the argv, not shipped, not opened while --join-train exists; note N1)" >> "$HR/INPUTS_SHA256.txt"; log "label file named only, not on the pod: $f"; fi
  done
  [ -f "$CACHE/train/manifest.jsonl" ] || stop A-INPUTS "no train manifest in $CACHE"
  [ -f "$EVALC/eval/manifest.jsonl" ] || stop A-INPUTS "no eval manifest in $EVALC"
  [ -f "$EVALC_ARM/eval/manifest.jsonl" ] || stop A-INPUTS "no pod-built eval manifest in $EVALC_ARM"
  [ -f "$SB/cache_identity.py" ] || stop A-FILES "missing $SB/cache_identity.py"
  cp -f "$EVR/eval_grid_windows.json" "$GRID" && cp -f "$EVR/join_eval141.npz" "$JOINE" || stop A-INPUTS "eval grid / join did not copy from $EVR"
  echo "$(sha_of "$GRID")  eval_grid_windows.json" >> "$HR/INPUTS_SHA256.txt"; echo "$(sha_of "$JOINE")  join_eval141.npz" >> "$HR/INPUTS_SHA256.txt"
  touch "$ST/done_A"; log "A ok: card, key files, inputs"
}
COMMIT=$(cat "$ST/commit.txt"); COMMIT_OK=$(cat "$ST/commit_ok.txt")

# ---------------------------------------------------------------- helpers
t1dump() { # t1dump <ckpt> <name> [eval cache root, default the pod-built one]   -> $T1/<name>/dump_<name>_m10_causal_heads with 141 ep files
  local ck=$1 name=$2 out=$T1/$2 tag=${2}_m10_causal_heads tk=()
  [ -f "$out/T1_DONE" ] && [ "$(ls "$out/dump_$tag" 2>/dev/null | grep -c '^ep.*npz$')" = 141 ] && { log "[skip] T1 dump $name"; return 0; }
  mkdir -p "$out"; rm -f "$out/T1_DONE"
  [ "${HR_TICKS:-1}" = 1 ] && { mkdir -p "$out/ticks"; tk=(--dump-ticks "$out/ticks" --dump-heads); }
  log "T1-kin dump $name"
  "$PY" "$TREE/stack/scripts/v8f_t1_eval.py" run --ckpt "$ck" --tag "$tag" --tracks "$TRACKS" --eval-cache "${3:-$EVALC_ARM}" --eval-grid "$GRID" --eval-join "$JOINE" --out-dir "$out" \
     --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal ${tk[@]+"${tk[@]}"} --device cuda >> "$out/t1.log" 2>&1
  local rc=$?; [ $rc = 0 ] || stop "T1-$name" "v8f_t1_eval.py rc=$rc (see $out/t1.log)"
  [ "$(ls "$out/dump_$tag" 2>/dev/null | grep -c '^ep.*npz$')" = 141 ] || stop "T1-$name" "dump holds $(ls "$out/dump_$tag" 2>/dev/null | grep -c '^ep.*npz$') ep files, expected 141"
  [ -f "$out/t1v8f_$tag.json" ] || stop "T1-$name" "no t1v8f_$tag.json"
  touch "$out/T1_DONE"
}
gate_arm() { # gate_arm <KIND> <seed> : selfcheck, token, argv equality.  Needs the bank for DAG kinds.
  local kind=$1 s=$2 name=${1}_s$2 g=$GATE/${1}_s$2
  mkdir -p "$g"
  build_argv "$kind" "$s" "$RUN/$name" || stop "GATE-$name" "no bank sha256 for $name"
  printf '%s\n' "${ARGV[@]}" > "$g/train_argv.txt"
  build_argv "$kind" "$s" "$g/out"; printf '%s\n' "${ARGV[@]}" --selfcheck-only > "$g/gate_argv.txt"
  if [ -f "$g/token.json" ] && "$PY" "$SB/gate_token.py" verify --token "$g/token.json" --argv-file "$g/train_argv.txt" >/dev/null 2>&1; then log "[skip] gate $name"; return 0; fi
  rm -f "$g/token.json"
  local bsha=""; case $kind in DAG) bsha=$(tr -d '\r\n ' < "$BANK/HIER_s$s/bank_sha256.txt");; FLATDAG) bsha=$(tr -d '\r\n ' < "$BANK/FLAT_s$s/bank_sha256.txt");; esac
  log "gate $name (selfcheck-only)"
  mapfile -t GA < "$g/gate_argv.txt"
  "$PY" "$TREE/stack/scripts/v8f_mini_train.py" "${GA[@]}" > "$g/gate.log" 2>&1
  local rc=$?; [ $rc = 0 ] || stop "GATE-$name" "selfcheck rc=$rc: $(grep -a -m1 -E 'ZZDVB|REFUSED|Error' "$g/gate.log" | head -c 300)"
  grep -aq "SELFCHECK PASS" "$g/gate.log" || stop "GATE-$name" "rc 0 but no 'SELFCHECK PASS' line"
  "$PY" "$SB/gate_token.py" make --selfcheck "$g/out/selfcheck.json" --argv-file "$g/gate_argv.txt" --out "$g/token.json" --kind "$kind" --commit "$COMMIT" --commit-verified "$COMMIT_OK" ${bsha:+--bank-sha256 "$bsha"} >> "$g/gate.log" 2>&1 \
    || stop "GATE-$name" "token refused: $(tail -c 400 "$g/gate.log")"
  "$PY" "$SB/gate_token.py" verify --token "$g/token.json" --argv-file "$g/train_argv.txt" >> "$g/gate.log" 2>&1 || stop "GATE-$name" "training argv differs from the gated argv"
}
train_arm() { # train_arm <KIND> <seed>
  local kind=$1 s=$2 name=${1}_s$2 g=$GATE/${1}_s$2 out=$RUN/${1}_s$2 n; n=$(steps_of "$kind")
  if [ -f "$out/summary.json" ] && "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('done') is True and d.get('steps')==int(sys.argv[2]) else 1)" "$out/summary.json" "$n" && [ -f "$out/ckpt_last.pt" ]; then
    "$PY" "$SB/gate_token.py" verify-run --token "$g/token.json" --argv-json "$out/argv.json" >/dev/null 2>&1 || stop "TRAIN-$name" "finished run's argv.json differs from the token"
    log "[skip] train $name"; return 0
  fi
  "$PY" "$SB/gate_token.py" verify --token "$g/token.json" --argv-file "$g/train_argv.txt" >/dev/null 2>&1 || stop "TRAIN-$name" "no matching token"
  mkdir -p "$out"; mapfile -t TA < "$g/train_argv.txt"
  log "train $name ($n steps)"
  "$PY" "$TREE/stack/scripts/v8f_mini_train.py" "${TA[@]}" >> "$out/train.log" 2>&1
  local rc=$?; [ $rc = 0 ] || stop "TRAIN-$name" "trainer rc=$rc (see $out/train.log)"
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('done') is True and d.get('steps')==int(sys.argv[2]) else 1)" "$out/summary.json" "$n" 2>/dev/null || stop "TRAIN-$name" "rc 0 but no summary.json done at step $n"
  [ -f "$out/ckpt_last.pt" ] || stop "TRAIN-$name" "no ckpt_last.pt"
  "$PY" "$SB/gate_token.py" verify-run --token "$g/token.json" --argv-json "$out/argv.json" >> "$g/gate.log" 2>&1 || stop "TRAIN-$name" "the trainer's argv.json differs from the gated argv"
}
bank_arm() { # bank_arm <HIER|FLAT> <seed>: own-state bank of the FROZEN stage-1 parent, TRAIN clips only
  local k=$1 s=$2 p=$RUN/${1}_s$2 b=$BANK/${1}_s$2
  mkdir -p "$b"
  if [ -f "$b/bank.json" ] && [ -f "$b/PROVENANCE.json" ] && [ -f "$b/bank_sha256.txt" ]; then log "[skip] bank $k s$s"; return 0; fi
  [ -f "$p/summary.json" ] && [ -f "$p/ckpt_last.pt" ] && [ -f "$p/argv.json" ] || stop "BANK-${k}_s$s" "parent $p is not finished"
  log "bank ${k}_s$s from the frozen parent"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$TREE/stack/scripts/v8f_dagger_queue_tool.py" bank --n-shards 4 --out-dir "$b" -- --ckpt "$p/ckpt_last.pt" --m2b-argv "$p/argv.json" \
    --cache-dir "$CACHE" --tracks "$TRACKS" --join-train "$JOINT" --eval-grid "$GRID" --batch 32 --threads 1 --device cuda --expect-clips 4512 --expect-windows 306566 >> "$b/bank_stage.log" 2>&1
  local rc=$?; [ $rc = 0 ] || stop "BANK-${k}_s$s" "bank tool rc=$rc (see $b/bank_stage.log)"
  [ -f "$b/bank.json" ] && [ -f "$b/PROVENANCE.json" ] || stop "BANK-${k}_s$s" "rc 0 without bank.json / PROVENANCE.json"
  "$PY" "$TREE/stack/scripts/v8f_dagger_queue_tool.py" sha "$b/bank.json" > "$b/bank_sha256.txt" || stop "BANK-${k}_s$s" "could not hash bank.json"
}
arm_done() { DONE_ARMS="$DONE_ARMS,$1"; status RUNNING "$1 done"; log "ARM DONE $1"; }

# ---------------------------------------------------------------- r1 harness control, r2 deliberate-regression arm (once)
if [ ! -f "$ST/done_r1" ]; then
  [ -f "$REFCK/ckpt_last.pt" ] || stop r1 "reference checkpoint $REFCK/ckpt_last.pt is missing"
  mkdir -p "$T1/r1"; t1dump "$REFCK/ckpt_last.pt" r1_BAL-NULL_s0 "$EVALC"
  [ -f "$SB/r1_reference.json" ] && cp -f "$SB/r1_reference.json" "$HR/r1_reference.json"
  "$PY" "$SB/hierrep_read.py" r1 --dump "$T1/r1_BAL-NULL_s0/dump_r1_BAL-NULL_s0_m10_causal_heads" --ref "$HR/r1_reference.json" --out "$HR/r1_result.json" >> "$LOG" 2>&1
  rc=$?
  case $rc in 0) log "r1 PASS";; 4) log "WARNING r1 PENDING: no r1_reference.json; recorded as PENDING, not a pass";; *) stop r1 "harness control FAILED (rc=$rc): $(head -c 400 "$HR/r1_result.json")";; esac
  touch "$ST/done_r1"
fi
if [ ! -f "$ST/done_r6" ]; then
  log "r6 cache identity: pod-built eval cache against the uploaded box-2-equal eval cache"
  "$PY" "$SB/cache_identity.py" --pod "$EVALC_ARM" --ref "$EVALC" --out "$HR/r6_result.json" >> "$LOG" 2>&1; rc=$?
  [ $rc = 0 ] || stop r6 "cache identity control did not pass (rc=$rc): $(head -c 300 "$HR/r6_result.json" 2>/dev/null)"
  log "r6 PASS"; touch "$ST/done_r6"
fi
if [ ! -f "$ST/done_r2" ]; then
  g=$GATE/r2_regress; mkdir -p "$g"; build_argv HIER 0 "$g/out"
  log "r2 deliberate-regression arm: argv declares full, model built with cut; the gate must REFUSE"
  "$PY" "$SB/r2_regress_gate.py" "${ARGV[@]}" --selfcheck-only > "$g/regress.log" 2>&1; rc=$?
  { [ $rc != 0 ] && grep -aq "ZZDVB-REFUSEDZZ" "$g/regress.log" && grep -aq "o_interface" "$g/regress.log"; } || stop r2 "the deliberately wrong lever was NOT refused by the declared-vs-built gate (rc=$rc): $(tail -c 400 "$g/regress.log")"
  echo "{\"r2_regression\":\"REFUSED as required\",\"rc\":$rc,\"line\":\"$(grep -a -m1 ZZDVB-REFUSEDZZ "$g/regress.log" | head -c 300 | tr -d '"\\')\"}" > "$HR/r2_regression.json"
  touch "$ST/done_r2"; log "r2 regression arm: refused as required"
fi

# ---------------------------------------------------------------- per seed: the whole seed before the next
for s in $SEEDS; do
  for k in HIER FLAT; do
    gate_arm $k $s; train_arm $k $s; t1dump "$RUN/${k}_s$s/ckpt_last.pt" "${k}_s$s"; arm_done "${k}_s$s"
  done
  for k in HIER FLAT; do bank_arm $k $s; done
  gate_arm DAG $s; train_arm DAG $s; t1dump "$RUN/DAG_s$s/ckpt_last.pt" "DAG_s$s"; arm_done "DAG_s$s"
  gate_arm FLATDAG $s; train_arm FLATDAG $s; t1dump "$RUN/FLATDAG_s$s/ckpt_last.pt" "FLATDAG_s$s"; arm_done "FLATDAG_s$s"
  log "SEED $s COMPLETE (all four arms trained and dumped)"
done

# ---------------------------------------------------------------- read and finish
ntok=$(ls "$GATE"/*/token.json 2>/dev/null | wc -l)
echo "{\"r2_gate\":{\"tokens\":$ntok,\"expected\":$(( $(echo $SEEDS | wc -w) * 4 )),\"each_token_requires\":\"declared_vs_built PASS, G-LIVE pass, frozen-prior control exactly 0.0\"}}" > "$HR/r2_gate.json"
mkdir -p "$HR/read"
"$PY" "$SB/hierrep_read.py" read --t1-root "$T1" --runs-root "$RUN" --out-dir "$HR/read" --r1-result "$HR/r1_result.json" >> "$LOG" 2>&1
rrc=$?
log "reader rc=$rrc (0 = controls pass, 2 = a control failed: READ NOT VALID, files still written)"
echo "stageB done $(ts); tokens $ntok; reader rc $rrc; $HR/read/VERDICT_HIERREP.md" > "$HR/STAGEB_DONE.txt"
status DONE "reader rc=$rrc"
exit 0
