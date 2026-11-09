#!/usr/bin/env bash
# p6_labels_chain.sh -- PDM labels for the P6 scorer arms (PREREG_SCORER1P6 sections 2 and 7).  Runs ON THE POD, with P7 paused (GPU free).
#   Job A (GPU): fan dump of P6's model_final.pt, command-mode dataset, over three scene lists: heldout (24 logs), fit (extra whole logs), train.
#   Job B (CPU): PDM labelling, started as soon as the draw exists, in passes over the growing set files, WORKERS processes at most.
#   Ends with labels_manifest.json and exactly one of $OUT/labels.DONE / $OUT/labels.FAILED.
# Launch (survives ssh drops):  setsid nohup bash p6_labels_chain.sh > $OUT/chain.stdout 2>&1 < /dev/null &  echo $! > $OUT/chain.pid
# Resumable: stage markers in $OUT/stages/; re-running continues.  FORCE=1 re-runs everything.  Success line: P6LABELS_ALL_OK.  Failure line: P6LABELS_ABORT.
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="${PKG:-/workspace/refe-plan}"
REFE="${REFE:-$PKG/refe}"
STACK="${STACK-/workspace/TanitAD/stack}"
if [ -n "$STACK" ]; then export PYTHONPATH="$STACK${PYTHONPATH:+:$PYTHONPATH}"; fi
PY="${PY:-python}"
NAVSIM_PY="${NAVSIM_PY:-$PY}"
NVSMI="${NVSMI:-nvidia-smi}"
SHA256="${SHA256:-sha256sum}"
MODEL="${MODEL:-/workspace/data/refe_p6/model_final.pt}"
EXPECT_SHA="${P6L_TEST_EXPECT_SHA:-968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1}"
TARGETS="${TARGETS:-/workspace/data/refe_navtrain/train_grow_cmd_3c}"
IMAGES="${IMAGES:-/workspace/data/navtrain_pixels}"
CALIB="${CALIB:-/workspace/data/refe_navtrain/train_grow/calib_table.json}"
HELDOUT_LOGS="${HELDOUT_LOGS:-$PKG/splits/sft_heldout_logs.txt}"
OUT="${OUT:-/workspace/data/p6base_labels}"
N_TRAIN="${N_TRAIN:-60000}"
WORKERS="${WORKERS:-6}"
FIT_TARGET="${FIT_TARGET:-8000}"
SEED="${SEED:-20261109}"
BACKBONE="${BACKBONE:-vitl16}"
DUMP_WORKERS="${DUMP_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
EXPECT_HELDOUT="${EXPECT_HELDOUT:-3137}"
MIN_TRAIN="${MIN_TRAIN:-30000}"
POLL_S="${POLL_S:-30}"
SKIP_PLAN="${SKIP_PLAN:-0}"
FORCE="${FORCE:-0}"
DUMP_PY="${DUMP_PY:-$HERE/p6_fan_dump.py}"
TOOL="$HERE/p6_labels_tool.py"

LOG="$OUT/chain.log"
LPID=""
FINISHED=0

say() { mkdir -p "$OUT" 2>/dev/null; echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

stop_loop() {
    if [ -n "$LPID" ] && kill -0 "$LPID" 2>/dev/null; then
        kill -TERM -- "-$LPID" 2>/dev/null || kill -TERM "$LPID" 2>/dev/null
        sleep 1
        kill -KILL -- "-$LPID" 2>/dev/null || kill -KILL "$LPID" 2>/dev/null
    fi
}

fail() {   # fail <stage> <short reason: never tool output>
    say "P6LABELS_ABORT stage=$1 reason=$2"
    rm -f "$OUT/labels.DONE"
    printf 'stage=%s reason=%s\n' "$1" "$2" > "$OUT/labels.FAILED"
    : > "$OUT/abort.flag"
    stop_loop
    FINISHED=1
    exit 1
}

on_exit() {
    if [ "$FINISHED" != "1" ]; then
        rm -f "$OUT/labels.DONE"
        [ -f "$OUT/labels.FAILED" ] || printf 'stage=unknown reason=unexpected_exit\n' > "$OUT/labels.FAILED"
        : > "$OUT/abort.flag"
        stop_loop
        echo "P6LABELS_ABORT stage=unknown reason=unexpected_exit" >> "$LOG"
    fi
}
mkdir -p "$OUT/stages" "$OUT/logs" "$OUT/reports"
stage_done() { [ "$FORCE" != "1" ] && [ -f "$OUT/stages/$1.done" ]; }
mark() { date -u +%FT%TZ > "$OUT/stages/$1.done"; }

# one dir after the other; for each: passes over the growing set files until a pass that STARTED after the dump finished has completed.
label_dir() {   # label_dir <name> <sets_dir> <pdm_dir> <expect>
    local name="$1" sets="$2" pdm="$3" expect="$4" k=0 complete last
    if stage_done "label_$name"; then return 0; fi
    mkdir -p "$pdm"
    while true; do
        [ -f "$OUT/abort.flag" ] && return 9
        complete=0; [ -f "$OUT/dump_$name.flag" ] && complete=1
        if ls "$sets"/onpolicy_rlfan_*.jsonl >/dev/null 2>&1; then
            k=$((k + 1)); last="$OUT/logs/relabel_${name}_pass$k.log"
            (cd "$REFE" && "$NAVSIM_PY" "${RELABEL_SCRIPT:-onpolicy_relabel_pdm.py}" --sets "$sets" --out "$pdm" --workers "$WORKERS" --served-only) > "$last" 2>&1 || return 1
            if [ "$complete" = "1" ]; then break; fi
        fi
        sleep "$POLL_S"
    done
    cp "$last" "$OUT/logs/relabel_${name}_final.log"
    "$PY" "$REFE/pdm_label_complete.py" --log "$OUT/logs/relabel_${name}_final.log" --out "$pdm" --sets "$sets" --expect "$expect" \
        --json "$OUT/reports/$name.json" > "$OUT/logs/complete_$name.log" 2>&1 || return 2
    mark "label_$name"
}
label_loop() {
    label_dir heldout "$OUT/heldout_sets" "$OUT/heldout_pdm" "$NH" || { echo "$?" > "$OUT/label_loop.rc"; exit 1; }
    label_dir fit "$OUT/fit_sets" "$OUT/fit_pdm" "$NF" || { echo "$?" > "$OUT/label_loop.rc"; exit 1; }
    label_dir train "$OUT/train_sets" "$OUT/train_pdm" "$NT" || { echo "$?" > "$OUT/label_loop.rc"; exit 1; }
    echo 0 > "$OUT/label_loop.rc"
}

if [ "${1:-}" = "__label_loop" ]; then label_loop; exit $?; fi

trap on_exit EXIT
trap 'exit 143' TERM INT HUP

jget() { "$PY" -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))[sys.argv[2]])' "$OUT/draw.json" "$1" | tr -d '\r'; }

if [ "$FORCE" = "1" ]; then rm -f "$OUT"/stages/*.done "$OUT"/dump_*.flag "$OUT/labels.DONE" "$OUT/label_loop.rc"; FORCE=0; fi
if [ -f "$OUT/labels.DONE" ] && [ "$FORCE" != "1" ] && [ -z "${P6L_RECHECK:-}" ]; then
    FINISHED=1; say "P6LABELS_ALL_OK already finished (labels.DONE present)"; exit 0
fi
rm -f "$OUT/labels.FAILED" "$OUT/abort.flag" "$OUT/label_loop.rc"
say "chain start: N_TRAIN=$N_TRAIN WORKERS=$WORKERS FIT_TARGET=$FIT_TARGET SEED=$SEED OUT=$OUT"

# ---------------------------------------------------------------------------------------------------------- 0 preconditions
case "$WORKERS" in ''|*[!0-9]*) fail precond workers_not_integer;; esac
[ "$WORKERS" -ge 1 ] && [ "$WORKERS" -le 14 ] || fail precond workers_out_of_range_1_14
if [ "$N_TRAIN" != "all" ]; then
    case "$N_TRAIN" in ''|*[!0-9]*) fail precond n_train_not_integer_or_all;; esac
fi
[ -f "$MODEL" ] || fail precond model_file_missing
got="$("$SHA256" "$MODEL" | cut -d' ' -f1 | tr -d '\r')"
[ "$got" = "$EXPECT_SHA" ] || fail precond model_sha256_mismatch
pending_gpu=0
for s in dump_heldout dump_fit dump_train; do stage_done "$s" || pending_gpu=1; done
if [ "$pending_gpu" = "1" ]; then
    apps="$("$NVSMI" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"; rc=$?
    [ "$rc" = "0" ] || fail precond nvidia_smi_unavailable
    apps="$(printf '%s' "$apps" | tr -d ' \r\n')"
    [ -z "$apps" ] || fail precond gpu_has_compute_process
fi
for p in "$REFE/rl_fan_dump.py" "$REFE/onpolicy_relabel_pdm.py" "$REFE/pdm_label_complete.py" "$REFE/ckpt_io.py" "$HELDOUT_LOGS" "$TOOL" "$DUMP_PY"; do
    [ -f "$p" ] || fail precond "missing_file_$(basename "$p")"
done
[ -d "$TARGETS" ] || fail precond targets_dir_missing
say "P6LABELS_PRECOND_OK model sha256 ${got:0:16}"
mark precond

# ---------------------------------------------------------------------------------------------------------- 1 draw
if ! stage_done draw; then
    "$PY" "$TOOL" draw --targets "$TARGETS" --heldout-logs "$HELDOUT_LOGS" --out "$OUT" --seed "$SEED" --n-train "$N_TRAIN" \
        --fit-target "$FIT_TARGET" --expect-heldout "$EXPECT_HELDOUT" --min-train "$MIN_TRAIN" > "$OUT/logs/draw.log" 2>&1 || fail draw tool_refused_see_logs_draw
    mark draw
fi
NH="$(jget n_heldout)"; NF="$(jget n_fit_extra)"; NT="$(jget n_train)"
say "draw: heldout $NH, fit_extra $NF, train $NT"

# ---------------------------------------------------------------------------------------------------------- 2 plan (coverage, no GPU work)
if [ "$SKIP_PLAN" != "1" ] && ! stage_done plan; then
    "$PY" "$DUMP_PY" --refe-dir "$REFE" --plan-only --ckpt "$MODEL" --keys "$OUT/keys/heldout.tsv" "$OUT/keys/fit_extra.tsv" "$OUT/keys/train.tsv" \
        --targets "$TARGETS" --images "$IMAGES" --calib "$CALIB" --backbone "$BACKBONE" --out "$OUT/plan" --tag plan > "$OUT/logs/plan.log" 2>&1 \
        || fail plan coverage_gap_or_refused_see_logs_plan
    mark plan
fi

# ---------------------------------------------------------------------------------------------------------- Job B: the labelling loop (background, own process group)
NEED_LABEL=0
for s in label_heldout label_fit label_train; do stage_done "$s" || NEED_LABEL=1; done
if [ "$NEED_LABEL" = "1" ]; then
    for n in heldout fit train; do stage_done "dump_$n" && : > "$OUT/dump_$n.flag"; done
    NH="$NH" NF="$NF" NT="$NT" OUT="$OUT" WORKERS="$WORKERS" REFE="$REFE" PY="$PY" NAVSIM_PY="$NAVSIM_PY" POLL_S="$POLL_S" STACK="$STACK" \
        P6L_TEST_EXPECT_SHA="$EXPECT_SHA" $( command -v setsid >/dev/null 2>&1 && echo setsid ) bash "$HERE/p6_labels_chain.sh" __label_loop > "$OUT/logs/label_loop.log" 2>&1 < /dev/null &
    LPID=$!
    echo "$LPID" > "$OUT/label_loop.pid"
    say "labelling loop started (pid $LPID, $WORKERS workers)"
fi
check_loop() {
    if [ -f "$OUT/label_loop.rc" ] && [ "$(cat "$OUT/label_loop.rc")" != "0" ]; then fail labelling "label_loop_rc_$(cat "$OUT/label_loop.rc")"; fi
}

# ---------------------------------------------------------------------------------------------------------- 3 Job A: dumps (GPU)
dump_one() {   # dump_one <name> <tag> <keys.tsv> <sets_dir> <expect> [extra args]
    local name="$1" tag="$2" keys="$3" sets="$4" expect="$5"; shift 5
    if stage_done "dump_$name"; then return 0; fi
    "$PY" "$DUMP_PY" --refe-dir "$REFE" --ckpt "$MODEL" --expect-ckpt-sha256 "$EXPECT_SHA" --keys "$keys" --targets "$TARGETS" --images "$IMAGES" \
        --calib "$CALIB" --backbone "$BACKBONE" --out "$sets" --tag "$tag" --workers "$DUMP_WORKERS" --device "$DEVICE" "$@" > "$OUT/logs/dump_$name.log" 2>&1 \
        || fail "dump_$name" dump_exit_nonzero_see_logs
    "$PY" "$TOOL" check-sets --dir "$sets" --keys "$keys" --expect "$expect" > "$OUT/logs/checksets_$name.log" 2>&1 || fail "dump_$name" dumped_sets_do_not_match_draw
    mark "dump_$name"; : > "$OUT/dump_$name.flag"
    say "dump $name finished ($expect sets)"
    check_loop
}
dump_one heldout p6h "$OUT/keys/heldout.tsv" "$OUT/heldout_sets" "$NH" --save-logits
dump_one fit p6f "$OUT/keys/fit_extra.tsv" "$OUT/fit_sets" "$NF" --save-logits
dump_one train p6t "$OUT/keys/train.tsv" "$OUT/train_sets" "$NT"

# ---------------------------------------------------------------------------------------------------------- 4 wait for Job B
if [ -n "$LPID" ]; then
    while kill -0 "$LPID" 2>/dev/null; do sleep "$POLL_S"; done
    wait "$LPID" 2>/dev/null
fi
[ -f "$OUT/label_loop.rc" ] || { for s in label_heldout label_fit label_train; do stage_done "$s" || fail labelling label_loop_ended_without_result; done; }
check_loop
for s in label_heldout label_fit label_train; do stage_done "$s" || fail labelling "stage_marker_missing_$s"; done

# ---------------------------------------------------------------------------------------------------------- 5 pack, manifest, DONE
if ! stage_done pack; then
    "$PY" "$TOOL" pack --sets "$OUT/heldout_sets" "$OUT/fit_sets" --pdm "$OUT/heldout_pdm" "$OUT/fit_pdm" --out "$OUT/fit_pack/fit_pack.npz" > "$OUT/logs/pack.log" 2>&1 \
        || fail pack pack_refused_see_logs
    mark pack
fi
if ! stage_done manifest; then
    "$PY" "$TOOL" manifest --root "$OUT" --model-sha256 "$EXPECT_SHA" --targets "$TARGETS" > "$OUT/logs/manifest.log" 2>&1 || fail manifest manifest_refused_see_logs
    mark manifest
fi
[ -f "$OUT/labels_manifest.json" ] || fail manifest manifest_file_missing
date -u +%FT%TZ > "$OUT/labels.DONE"
FINISHED=1
say "P6LABELS_ALL_OK train=$NT heldout=$NH fit_extra=$NF manifest=$OUT/labels_manifest.json"
exit 0
