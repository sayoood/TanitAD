#!/usr/bin/env bash
# rehearse_repl_read.sh -- lead's rehearsal of the reader of card ADD121 (REPL10K) on BANKED runs only, before any model of the second pair exists.
# A stand-in second pair is built from links, SWAPPED so that a read of the wrong folder shows: model seed 2 = the banked runs of model seed 1, model seed 3 = those of model seed 0 (dumps and tick
# files linked under the new names; the first line of each run text rewritten to the form this queue writes: new arm name, model seed m, reader seed m - 2, the wrapper hash of the file, and for a
# line of an earlier queue form the fields pre-roll, exec, nav and grid; the checkpoint hash stays that of the linked data, which is the file linked as the model of seed m). Nothing of a run of the card is read; every value this rehearsal can print is a value of rows 179 to 186.
# Purpose: the engine paths of the reader (three seed sets, the strategic grid with the spread run at seeds 2 and 3, the eight controls, the names stage) run once end to end.
# Expected: pooled levels and pair means of the stand-in pair equal those of the first pair, the levels per seed are those of the first pair in the other order. Writes only under $RH. CPU only. It deletes, moves and stops nothing.
# Usage: bash rehearse_repl_read.sh <reader kit folder> <new rehearsal folder> <read test output>
set -u
K=${1:?reader kit folder}; RH=${2:?rehearsal folder}; TR=${3:?read test output}
V=/workspace/v8proof; W=$V/integ10k; GE=$V/repl; P=$V/podread; RD=$P/root/integ10k; PL=$RD/pull; CODE=$P/code
GFM=$V/gapmargin/gapfiles; CS=$V/cropscale; CHK=$V/visrule10k/check_v
MODEL=$CS/kit_read/read_cropscale10k.py; GAPF=$CS/gapfiles/VGAPLE.npz; PINNED=$V/hierrep/pins/eval_grid_windows.json
PY=/workspace/venv/bin/python; WRAP=$W/build/v8f_t1_eval_reader_rule_rzgfhqetjv.py
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ REHEARSE-REPL STOP $*"; exit 3; }
[ -e "$RH" ] && stop "EXISTS $RH"
[ -f "$K/read_repl10k.py" ] || stop "NO-READER"
[ -f "$TR" ] || stop "NO-TEST-OUTPUT"
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz SR_ESTREAD_DIR=$RD
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"; fi
mkdir -p "$RH/pull" "$RH/w/ckpt/UE_s2/A" "$RH/w/ckpt/UE_s3/A" "$RH/out" "$RH/models"
ln -s "$W/ckpt/UE_s1/A/ckpt_last.pt" "$RH/w/ckpt/UE_s2/A/ckpt_last.pt"
ln -s "$W/ckpt/UE_s0/A/ckpt_last.pt" "$RH/w/ckpt/UE_s3/A/ckpt_last.pt"
c2=$(sha256sum < "$W/ckpt/UE_s1/A/ckpt_last.pt" | cut -c1-64); c3=$(sha256sum < "$W/ckpt/UE_s0/A/ckpt_last.pt" | cut -c1-64); wsha=$(sha256sum < "$WRAP" | cut -c1-64)
for e in "$PL"/*; do b=$(basename "$e"); case "$b" in *_s2|*_s3) continue;; esac; ln -s "$(readlink -f "$e")" "$RH/pull/$b"; done
RUNS="I10RLcHb20 I10RQ50Hb20 I10RQ50Hb20gGEc I10RQ50Hb20gA90 I10R0Hb20 I10RLcKF5Hb20 I10RQ50KF5Hb20 I10RQ50KF5Hb20gGEc I10RQ50KF5Hb20gA90 I10RLcKF5P30Hb20 I10RQ50KF5P30Hb20 I10RQ50KF5P30Hb20gGEc I10RQ50KF5P30Hb20gA90"
n=0
for x in $RUNS; do
  for s in 0 1; do
    [ -f "$W/t1/${x}_s$s/T1_DONE" ] || stop "MISSING-RUN ${x}_s$s"
    [ -e "$RH/pull/${x}_s$s" ] || ln -s "$W/t1/${x}_s$s" "$RH/pull/${x}_s$s"
  done
  for m in 2 3; do
    s=$((3 - m)); src=$W/t1/${x}_s$s; d=$RH/pull/${x}_s$m
    mkdir -p "$d/ticks"
    ln -s "$src/T1_DONE" "$d/T1_DONE"
    ln -s "$src/dump_${x}_s${s}_m10_causal_heads" "$d/dump_${x}_s${m}_m10_causal_heads"
    for f in "$src"/ticks/*; do
      b=$(basename "$f"); nb=${b//${x}_s$s/${x}_s$m}
      if [ "$b" = "run_${x}_s$s.txt" ]; then
        sed -e "1s/^arm ${x}_s$s /arm ${x}_s$m /" -e "1s/ model seed $s reader seed $s / model seed $m reader seed $((m - 2)) /" -e "1s/wrapper sha256 [0-9a-f]*/wrapper sha256 $wsha/" \n            -e "1{/ preroll /!s/ model seed / preroll 0 exec plan nav deadreckon grid eval_grid_windows.json model seed /}" "$f" > "$d/ticks/$nb"
      else
        ln -s "$f" "$d/ticks/$nb"
      fi
    done
    n=$((n + 1))
  done
done
# the strategic grid reads its tick files under <w>/t1: a stand-in w with links to the banked entries and to the stand-in runs
mkdir -p "$RH/w/t1"
for e in "$W"/*; do b=$(basename "$e"); case "$b" in t1|ckpt) continue;; esac; ln -s "$e" "$RH/w/$b"; done
for s in 0 1; do ln -s "$W/ckpt/UE_s$s" "$RH/w/ckpt/UE_s$s"; done
for e in "$W"/t1/*; do b=$(basename "$e"); case "$b" in *_s2|*_s3) continue;; esac; ln -s "$e" "$RH/w/t1/$b"; done
for x in $RUNS; do for m in 2 3; do ln -s "$RH/pull/${x}_s$m" "$RH/w/t1/${x}_s$m"; done; done
echo "stand-in runs built: $n"
printf '{"models": {"UE_s2": {"ckpt_sha256": "%s", "gate_pass": true, "recipe_same_but_seed": true, "record": "REHEARSAL: a stand-in built from banked files, not a model record"}, "UE_s3": {"ckpt_sha256": "%s", "gate_pass": true, "recipe_same_but_seed": true, "record": "REHEARSAL: a stand-in built from banked files, not a model record"}}, "note_sha256": "%s"}\n' "$c2" "$c3" "$(printf 'rehearsal' | sha256sum | cut -c1-64)" > "$RH/models/REPL_MODELS_RECORD.json"
QS=$(date +%s); CSTART=$QS
echo "ZZ REPL10K DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) all steps done" > "$RH/REPL10K_DONE.txt"
MQ=$(stat -c %Y "$GE/lead_tests/lead_pytest_queue.out"); MR=$(stat -c %Y "$TR")
echo "controls stage start $(ts)"
( cd "$RD" && "$PY" "$K/read_repl10k.py" --stage controls --model-reader "$MODEL" --pull "$RH/pull" --code "$CODE" --grid-dir "$W/longroll" --grid-pinned "$PINNED" --w "$RH/w" --tree "$W/tree" --wrapper-file "$WRAP" \
    --gap-file "$GAPF" --margin-dir "$GFM" --margin-record "$GFM/GAPMARGIN_GAPFILES_RECORD.json" --models-record "$RH/models/REPL_MODELS_RECORD.json" \
    --wrapper-check "$CHK/CHECK_WRAPPER_V_DONE.txt" --wrapper-hashes "$CHK/CHECK_WRAPPER_V.hashes" \
    --tests-queue "$GE/lead_tests/lead_pytest_queue.out" --tests-read "$TR" --tests-mtime-queue "$MQ" --tests-mtime-read "$MR" --queue-start "$QS" --controls-start "$CSTART" \
    --done-marker "$RH/REPL10K_DONE.txt" --out "$RH/out/REPL10K_CONTROLS.json" > "$RH/out/read_controls.log" 2>&1 ); rc=$?
grep -E '^e[0-7] (PASS|FAIL)|^ZZ READ-REPL10K' "$RH/out/read_controls.log" | cut -c1-200 | head -n 10
echo "ZZ REHEARSE-REPL controls DONE rc $rc $(ts)"
[ "$rc" = 0 ] || exit 1
( cd "$RD" && "$PY" "$K/read_repl10k.py" --stage names --model-reader "$MODEL" --controls "$RH/out/REPL10K_CONTROLS.json" --out "$RH/out/REPL10K_RESULT.json" --log "$RH/out/read_names.log" > "$RH/out/read_names.out" 2>&1 ); rc=$?
tail -n 1 "$RH/out/read_names.out" | cut -c1-200
echo "ZZ REHEARSE-REPL names DONE rc $rc $(ts)"
