#!/usr/bin/env bash
# chain_arms.sh <chain name> <arms file> [--dry-run]
# Generic gated arm queue for the pod: SPATIAL stage 1 (PREREG_V8F_PROOF_ADD12) and DATAEFF (ADD13). One arm at a time.
# Waits for a predecessor marker and for an idle GPU (never adds load to a training pod). Fail closed: any miss writes <chain>_STOP.txt and exits 1. Resumable.
# Arms file, one arm per line (lines starting with # are comments):
#   ARM|<name>|<tree>|<HIER or FLAT>|<seed>|<exclude A:B or NONE>|<cache root>|<extra argv, space separated>|<expectations for gate_extra.py, space separated>
#   PIN|<tree>|<path relative to the tree>|<sha256>          every PIN must hold before the first arm
# Reuses, unchanged, from the HIERREP chain: gate_token.py (kinds HIER / FLAT: built o_interface, 19,164 steps, live check, frozen-prior control), the eval pins and inputs.
set -u
NAME=${1:?chain name}; ARMS=${2:?arms file}
DRY=0; [ "${3:-}" = "--dry-run" ] && DRY=1
W=${HR_W:-/workspace/v8proof}
PY=${HR_PY:-/workspace/venv/bin/python}
HR=$W/hierrep
SB=$HR/stageB
C2=${C2_DIR:-$W/chain2}
CH=$W/chains/$NAME
RUN=$W/runs/$NAME
T1=$RUN/t1; GATE=$RUN/gate
LOG=$CH/chain.log
CARD=${C2_CARD:?card name, for example PREREG_V8F_PROOF_ADD12}
WAIT_FOR=${C2_WAIT_FOR:-$HR/STAGEB_DONE.txt}
COMMIT=${C2_COMMIT:-3e264bbf}
IDLE_MIB=${C2_IDLE_MIB:-2000}
V9=$W/data/v9; INP=$W/data/inputs
TRACKS=$INP/native_tracks.pt; JOINT=$INP/join_train.npz
P10=$INP/p10_prior.pt
PINS=$HR/pins; GRID=$PINS/eval_grid_windows.json; JOINE=$PINS/join_eval141.npz
EVALC=$W/data/cache_pod
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$NAME $(ts)] $*" | tee -a "$LOG"; }
stop() { log "STOP at $1: ${2:-}"; echo "$1: ${2:-} ($(ts))" > "$CH/${NAME}_STOP.txt"; exit 1; }
sha_of() { sha256sum "$1" | cut -c1-64; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

build_argv() { # build_argv <seed> <kind> <exclude> <cache> <out> <extra...>  -> ARGV
  local s=$1 kind=$2 ex=$3 cache=$4 out=$5; shift 5
  local cut=() exf=()
  [ "$kind" = FLAT ] && cut=(--o-interface cut)
  [ "$ex" = NONE ] || exf=(--exclude "$ex")
  ARGV=(--arm M2b --seed "$s" --grid pool4x10 ${cut[@]+"${cut[@]}"} "$@" --out "$out" --cache-dir "$cache" --v9-train "$V9/v9_labels_train.npz" --v9-train-md5 83e30bd3791f0767707eade5f4671759
        --lead-train "$V9/lead_train4572.npz" --speed-train "$V9/refcv6_speed_max_v8_train.jsonl" --join-train "$JOINT" --p10 "$P10" --kin-mode yaw --tracks "$TRACKS"
        ${exf[@]+"${exf[@]}"} --batch 64 --steps-mult 4 --device cuda)
}

mapfile -t LINES < <(grep -v -E '^\s*(#|$)' "$ARMS" | tr -d '\r')
if [ "$DRY" = 1 ]; then
  for ln in "${LINES[@]}"; do
    IFS='|' read -r typ a b c d e f g h <<< "$ln"
    if [ "$typ" = PIN ]; then echo "PIN  $a/$b $c"; continue; fi
    read -r -a EX <<< "$g"
    build_argv "$d" "$c" "$e" "$f" "$RUN/$a" ${EX[@]+"${EX[@]}"}
    echo "TRAIN $a: $PY $b/stack/scripts/v8f_mini_train.py ${ARGV[*]}"
    echo "EXPECT $a: $h"
  done
  exit 0
fi

mkdir -p "$CH" "$RUN" "$T1" "$GATE" || { echo "cannot create $CH"; exit 1; }
rm -f "$CH/${NAME}_STOP.txt" 2>/dev/null
if command -v flock >/dev/null 2>&1; then exec 8>"$CH/chain.lock"; flock -n 8 || { echo "another $NAME chain holds the lock"; exit 1; }; fi
log "chain $NAME start; card $CARD; arms file sha256 $(sha_of "$ARMS"); waits for $WAIT_FOR"

# ---- wait for the predecessor (bounded: 72 h), then for an idle GPU before every arm (bounded: 6 h)
n=0; while [ ! -f "$WAIT_FOR" ]; do n=$((n+1)); [ $n -gt 4320 ] && stop WAIT "predecessor marker $WAIT_FOR absent after 72 h"; sleep 60; done
log "predecessor marker present"
wait_idle() { local k=0 u; while :; do u=$(gpu_used); [ -n "$u" ] && [ "$u" -lt "$IDLE_MIB" ] && return 0; k=$((k+1)); [ $k -gt 360 ] && stop IDLE "GPU not below $IDLE_MIB MiB after 6 h (used $u)"; sleep 60; done; }

# ---- pins of every tree
for ln in "${LINES[@]}"; do
  IFS='|' read -r typ a b c rest <<< "$ln"
  [ "$typ" = PIN ] || continue
  [ -f "$a/$b" ] || stop PIN "absent $a/$b"
  [ "$(sha_of "$a/$b")" = "$c" ] || stop PIN "$a/$b sha256 $(sha_of "$a/$b") != $c"
done
log "pins hold"
[ -f "$SB/gate_token.py" ] && [ -f "$C2/gate_extra.py" ] || stop TOOLS "gate_token.py or gate_extra.py absent"
[ -f "$GRID" ] && [ -f "$JOINE" ] && [ -f "$JOINT" ] && [ -f "$TRACKS" ] && [ -f "$P10" ] || stop INPUTS "an eval pin or a training input is absent"

t1dump() { # t1dump <tree> <ckpt> <name>
  local tree=$1 ck=$2 name=$3 out=$T1/$3 tag=${3}_m10_causal_heads
  [ -f "$out/T1_DONE" ] && [ "$(ls "$out/dump_$tag" 2>/dev/null | grep -c '^ep.*npz$')" = 141 ] && { log "[skip] T1 dump $name"; return 0; }
  mkdir -p "$out/ticks"; rm -f "$out/T1_DONE"
  log "T1-kin dump $name"
  "$PY" "$tree/stack/scripts/v8f_t1_eval.py" run --ckpt "$ck" --tag "$tag" --tracks "$TRACKS" --eval-cache "$EVALC" --eval-grid "$GRID" --eval-join "$JOINE" --out-dir "$out" \
     --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks "$out/ticks" --dump-heads --device cuda >> "$out/t1.log" 2>&1
  local rc=$?; [ $rc = 0 ] || stop "T1-$name" "v8f_t1_eval.py rc=$rc (see $out/t1.log)"
  [ "$(ls "$out/dump_$tag" 2>/dev/null | grep -c '^ep.*npz$')" = 141 ] || stop "T1-$name" "dump does not hold 141 ep files"
  touch "$out/T1_DONE"
}

for ln in "${LINES[@]}"; do
  IFS='|' read -r typ name tree kind seed ex cache extra expect <<< "$ln"
  [ "$typ" = ARM ] || continue
  g=$GATE/$name; out=$RUN/$name; mkdir -p "$g"
  read -r -a EX <<< "$extra"; read -r -a EXP <<< "$expect"
  [ -f "$tree/stack/scripts/v8f_mini_train.py" ] || stop "ARM-$name" "no trainer in $tree"
  build_argv "$seed" "$kind" "$ex" "$cache" "$out" ${EX[@]+"${EX[@]}"}; printf '%s\n' "${ARGV[@]}" > "$g/train_argv.txt"
  build_argv "$seed" "$kind" "$ex" "$cache" "$g/out" ${EX[@]+"${EX[@]}"}; printf '%s\n' "${ARGV[@]}" --selfcheck-only > "$g/gate_argv.txt"
  if [ -f "$out/summary.json" ] && [ -f "$out/ckpt_last.pt" ] && [ -f "$g/token.json" ] && [ -f "$g/extra_token.json" ] \
     && "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('done') is True and d.get('steps')==19164 else 1)" "$out/summary.json" \
     && "$PY" "$SB/gate_token.py" verify-run --token "$g/token.json" --argv-json "$out/argv.json" >/dev/null 2>&1; then
    log "[skip] train $name"
  else
    wait_idle
    # gate: selfcheck on the exact argv, token (HIERREP judge), extra token (the levers this card moves)
    rm -f "$g/token.json" "$g/extra_token.json"
    log "gate $name (selfcheck-only)"
    mapfile -t GA < "$g/gate_argv.txt"
    "$PY" "$tree/stack/scripts/v8f_mini_train.py" "${GA[@]}" > "$g/gate.log" 2>&1
    rc=$?; [ $rc = 0 ] || stop "GATE-$name" "selfcheck rc=$rc: $(grep -a -m1 -E 'ZZDVB|REFUSED|Error' "$g/gate.log" | head -c 300)"
    grep -aq "SELFCHECK PASS" "$g/gate.log" || stop "GATE-$name" "rc 0 but no SELFCHECK PASS line"
    "$PY" "$SB/gate_token.py" make --selfcheck "$g/out/selfcheck.json" --argv-file "$g/gate_argv.txt" --out "$g/token.json" --kind "$kind" --commit "$COMMIT" --commit-verified true >> "$g/gate.log" 2>&1 \
      || stop "GATE-$name" "token refused: $(tail -c 300 "$g/gate.log")"
    "$PY" "$SB/gate_token.py" verify --token "$g/token.json" --argv-file "$g/train_argv.txt" >> "$g/gate.log" 2>&1 || stop "GATE-$name" "training argv differs from the gated argv"
    "$PY" "$C2/gate_extra.py" check --selfcheck "$g/out/selfcheck.json" --card "$CARD" --out "$g/extra_token.json" --expect "${EXP[@]}" >> "$g/gate.log" 2>&1 \
      || stop "GATE-$name" "extra gate refused: $(grep -a -m1 REFUSED "$g/gate.log" | head -c 300)"
    # deliberate regression of the extra gate, once per arm: a wrong expectation must be refused
    if "$PY" "$C2/gate_extra.py" check --selfcheck "$g/out/selfcheck.json" --card "$CARD" --out "$g/extra_regress.json" --expect "info.n_clips=-1" > "$g/extra_regress.log" 2>&1; then
      stop "GATE-$name" "the extra gate accepted a wrong expectation"; fi
    rm -f "$g/extra_regress.json"
    mkdir -p "$out"; mapfile -t TA < "$g/train_argv.txt"
    log "train $name (19164 steps)"
    "$PY" "$tree/stack/scripts/v8f_mini_train.py" "${TA[@]}" >> "$out/train.log" 2>&1
    rc=$?; [ $rc = 0 ] || stop "TRAIN-$name" "trainer rc=$rc (see $out/train.log)"
    "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('done') is True and d.get('steps')==19164 else 1)" "$out/summary.json" 2>/dev/null || stop "TRAIN-$name" "no summary.json done at step 19164"
    [ -f "$out/ckpt_last.pt" ] || stop "TRAIN-$name" "no ckpt_last.pt"
    "$PY" "$SB/gate_token.py" verify-run --token "$g/token.json" --argv-json "$out/argv.json" >> "$g/gate.log" 2>&1 || stop "TRAIN-$name" "the trainer argv.json differs from the gated argv"
  fi
  t1dump "$tree" "$out/ckpt_last.pt" "$name"
  log "ARM DONE $name"
done
echo "chain $NAME done $(ts)" > "$CH/${NAME}_DONE.txt"
log "ZZ CHAIN $NAME DONE ZZ"
exit 0
