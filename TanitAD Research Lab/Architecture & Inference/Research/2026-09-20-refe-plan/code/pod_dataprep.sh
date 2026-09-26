#!/bin/bash
# REFe navtrain DATA PREP on a Linux pod -- DBs, pixels, teacher targets, augmentation, scorer, gates.
#
# Prerequisite: code/pod_teacher_env.sh has run and written $WORK/teacher_env.sh.
# Every stage is RESUMABLE and idempotent: re-running this script after any interruption is always
# safe, because each bank file IS its own progress record (keys: log_name, token, step).
#
#   STAGES (select with STAGES="dbs calib pix rank0 aug score assemble gate", default all, in order)
#     dbs    range-fetch the 1,192 navtrain nuPlan DBs out of the train+val zips
#            MEASURED 2026-09-23: 97.57 GiB transit -> 163.82 GiB on disk (vs 1,037.7 GiB whole zips)
#     calib  each log's own camera rig from its DB (R22: PETR's embedding is per-sample; seconds)
#     pix    OpenScene navtrain_current_{1..32}.tgz -> keep ONLY the 4 REFe cameras
#            MEASURED on shard 25: REFe's cameras are 27.5 % of the bytes (LiDAR 43.8 %) =>
#            304.5 GB transit, ~84 GB on disk. navtrain_history is NOT needed.
#     rank0  teacher rollouts, rank 0, self-healing shards
#     aug    K=2 diversity search over the rank-0 bank (TAU = min divergence, a PI decision)
#     score  PDM scorer targets for rank 0 AND the augmented half (per-row aug, R18)
#     assemble  the ONE training dir: both ranks' targets + scorer targets + the camera rigs
#     gate   consumer conformance + signal consistency on the TRAINING dir, both ranks, ASSERTED
#
# ⛔ Knobs, all printed at start: SHARDS (default = nvidia-visible cores / 6), THREADS per shard,
#   TAU (default 0.3 -- PI decision 2026-09-23: 71 % yield in 1,418 shard-h vs 52 % in 1,629 at 0.5;
#   Pareto-better on yield AND cost, and halves the Singapore gap. See D-REFE-AUGSEARCH-1).
set -u
WORK="${WORK:-/workspace}"
PKG="${PKG:-$WORK/refe-plan}"
DATA="${DATA:-$WORK/data}"
DBS="$DATA/navtrain_dbs"; PIX="$DATA/navtrain_pixels"; BANK="$DATA/refe_navtrain"
STAGES="${STAGES:-dbs calib pix rank0 aug score assemble gate}"
# ⛔ CORES FROM THE CGROUP QUOTA, NOT nproc. MEASURED 2026-09-23 on the first REFe pod: `nproc` said 96
# (the host) while cpu.max allowed 765000/100000 = 7.65 CPUs -- `nproc / 6` would have started 16
# shards x 6 threads on under 8 real cores and thrashed (the N=8 thrash the dev box measured).
CORES="$(nproc)"
if read -r q per < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != "max" ] && [ "${per:-0}" -gt 0 ]; then
  CORES=$(( q / per > 0 ? q / per : 1 ))
fi
SHARDS="${SHARDS:-$(( CORES / 6 > 0 ? CORES / 6 : 1 ))}"
THREADS="${THREADS:-$(( CORES / SHARDS > 0 ? CORES / SHARDS : 1 ))}"
TAU="${TAU:-0.3}"; MAXR="${MAXR:-20}"   # PI DECISION 2026-09-23: 0.3 m (71 % yield vs 52 % at 0.5)
HF="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim"

[ -f "$WORK/teacher_env.sh" ] || { echo "ZZFAIL no teacher_env.sh -- run code/pod_teacher_env.sh first ZZ"; exit 1; }
# shellcheck disable=SC1091
source "$WORK/teacher_env.sh"
export NUPLAN_DATA_ROOT="$DATA" REFE_NUPLAN_DB_ROOT="$DBS"
export REFE_NAVTRAIN_YAML="$PKG/splits/navtrain.yaml" REFE_BACKBONE_ROOT="$DATA/backbones"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export PYTHONUNBUFFERED=1   # shard logs readable while they run (redirected stdout is block-buffered)
# ⭐ SIMULATION RATE of every teacher rollout (refe/navtrain_scenarios.py SIM_HZ), 2026-09-24. nuPlan's
# own sim subsamples the 20 Hz DB to 10 Hz -- the teacher's PUBLISHED operating point -- while the first
# navtrain rows were rolled out at 20 Hz. MEASURED (diag_sim_rate.py + diag_sim_rate_pdm.py, 73 frames):
# 10 Hz is 1.64x cheaper, moves targets ADE 0.23 m, and scores no worse on the PDM calculators.
# Rows carry `sim_hz`; the builders REFUSE to extend a bank of another rate (exit 5).
export REFE_SIM_HZ="${REFE_SIM_HZ:-10}"
# ⭐ THE TEACHER RUNS ON THE CPU (2026-09-24). MEASURED on the pod with 7 shards live: the same 6
# frames took 28.4 s on CPU vs 36.4 s on CUDA -- seven processes time-slicing one GPU make every
# small forward wait -- and CPU rollouts leave the A40 to the trainer (train.py --grow runs beside
# this). Outputs differ from CUDA by float noise only (0 - 7 cm/frame); rows record teacher_device.
TEACHER_DEVICE="${TEACHER_DEVICE:-cpu}"
PY="$DRIVERL_EVAL_PYTHON"
mkdir -p "$DBS" "$PIX" "$BANK"
echo "ZZSTART stages=[$STAGES] shards=$SHARDS threads=$THREADS cores=$CORES tau=$TAU sim_hz=$REFE_SIM_HZ device=$TEACHER_DEVICE $(date -u +%FT%TZ) ZZ"
has () { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }

# run N shards of a command in bounded self-healing loops; $1 = tag, rest = command.
# ⛔ EVERY SHARD WRITES ITS OWN DIRECTORY. `@I@` in any argument is replaced by the shard index, and
# callers must put it in --out: N processes appending to ONE bank file interleave and tear rows. The
# first draft of this script did exactly that; the dev-box runner never did (it merged after).
shards () {
  local tag=$1; shift
  for ((i=0; i<SHARDS; i++)); do
    (
      args=(); for x in "$@"; do args+=("${x//@I@/$i}"); done
      r=0
      until "${args[@]}" --log-shard "$i/$SHARDS" >> "$BANK/log_${tag}_s$i.txt" 2>&1; do
        r=$((r+1)); echo "ZZRESTART $tag shard=$i n=$r $(date -u +%T)" >> "$BANK/restarts.log"
        [ "$r" -ge "$MAXR" ] && { echo "ZZGAVEUP $tag shard=$i" >> "$BANK/restarts.log"; break; }
        sleep 15
      done
    ) &
  done
  wait
}

# merge per-shard bank files into one; $1 = glob of shard files, $2 = destination, $3 = key fields.
# ⛔ NOT `cat`: every line is parsed, torn lines (a killed shard) and duplicate keys (a changed shard
# count) are dropped and COUNTED -- code/merge_bank.py. Atomic via .tmp.
merge () { "$PY" "$PKG/code/merge_bank.py" --key "${3:-log_name,token,step,rank}" --out "$2" "$1" \
             || { echo "ZZFAIL merge $2 ZZ"; exit 1; }; }

if has dbs; then
  echo "=== dbs ==="
  ( cd "$PKG/code" && "$PY" fetch_navtrain_dbs.py --out "$DBS" --held-root /nonexistent --jobs "${DB_JOBS:-8}" ) \
    || { echo "ZZFAIL dbs ZZ"; exit 1; }
  echo "ZZOK dbs $(ls "$DBS"/*.db 2>/dev/null | wc -l) files ZZ"
fi

if has calib; then
  echo "=== calib: each log's own camera rig (R22) ==="
  ( cd "$PKG/refe" && "$PY" calib_table.py --db-root "$DBS" --out "$BANK/calib_table.json" ) \
      > "$BANK/log_calib.txt" 2>&1
  tail -3 "$BANK/log_calib.txt"
  # the artifact decides: the table's own OK marker, never the exit status
  grep -q "ZZCALIB_TABLE_OKZZ" "$BANK/log_calib.txt" || { echo "ZZFAIL calib -- see $BANK/log_calib.txt ZZ"; exit 1; }
  echo "ZZOK calib ZZ"
fi

if has pix; then
  echo "=== pix: 4 REFe cameras only ==="
  for n in $(seq 1 32); do
    [ -f "$PIX/.done_$n" ] && continue
    tgz="$PIX/navtrain_current_$n.tgz"
    curl -sSL -C - --retry 10 --retry-all-errors -o "$tgz" "$HF/navtrain_current_$n.tgz" || { echo "ZZFAIL pix dl $n ZZ"; continue; }
    # extraction is the integrity check: a truncated gzip stream fails here and leaves no .done marker
    if tar -xzf "$tgz" -C "$PIX" --wildcards '*/CAM_F0/*' '*/CAM_B0/*' '*/CAM_L0/*' '*/CAM_R0/*'; then
      touch "$PIX/.done_$n"; rm -f "$tgz"; echo "ZZOK pix shard $n ZZ"
    else
      echo "ZZFAIL pix extract $n ZZ"; rm -f "$tgz"
    fi
  done
  echo "ZZOK pix $(ls "$PIX"/.done_* 2>/dev/null | wc -l)/32 shards ZZ"
fi

cd "$PKG/refe" || exit 1

if has rank0; then
  echo "=== rank0 ==="
  # --resume-glob: a row done by ANY shard dir (incl. rows uploaded from the dev box into r0_sDEV/) is
  # skipped, so SHARDS can change between launches without redoing or duplicating work
  shards A_r0 "$PY" build_teacher_rollouts.py --source navtrain --out "$BANK/r0_s@I@" --rank 0 --resume --device "$TEACHER_DEVICE" \
      --resume-glob "$BANK/r0_s*/targets_rank0.jsonl" ${LOGS_FILE:+--logs-file "$LOGS_FILE"}
  merge "$BANK/r0_s*/targets_rank0.jsonl" "$BANK/r0/targets_rank0.jsonl"
  echo "ZZOK rank0 $(wc -l < "$BANK/r0/targets_rank0.jsonl") rows ZZ"
fi

if has aug; then
  echo "=== aug (tau=$TAU) ==="
  shards AUG "$PY" augment_search.py --bank "$BANK/r0" --out "$BANK/aug_s@I@" --tau "$TAU" --resume --device "$TEACHER_DEVICE" \
      --resume-glob "$BANK/aug_s*" ${LOGS_FILE:+--logs-file "$LOGS_FILE"}
  merge "$BANK/aug_s*/targets_aug.jsonl" "$BANK/aug/targets_aug.jsonl"
  merge "$BANK/aug_s*/aug_stats.jsonl"   "$BANK/aug/aug_stats.jsonl"   log_name,token
  echo "ZZOK aug $(wc -l < "$BANK/aug/targets_aug.jsonl") rows ZZ"
fi

if has score; then
  echo "=== score: rank 0, then the augmented half under its own per-row route (R18) ==="
  shards SC_r0 "$PY" build_scorer_targets.py --source navtrain --perframe-bank "$BANK/r0" \
      --out "$BANK/sc_r0_s@I@" --rank 0 --frame-stride 1 --resume \
      --resume-glob "$BANK/sc_r0_s*/scorer_targets.jsonl" ${LOGS_FILE:+--logs-file "$LOGS_FILE"}
  merge "$BANK/sc_r0_s*/scorer_targets.jsonl" "$BANK/sc_r0/scorer_targets.jsonl" log_name,token,step,rank,candidate
  shards SC_aug "$PY" build_scorer_targets.py --source navtrain --perframe-bank "$BANK/aug" \
      --perframe-file "$BANK/aug/targets_aug.jsonl" --out "$BANK/sc_aug_s@I@" --rank 1 --frame-stride 1 \
      --resume --resume-glob "$BANK/sc_aug_s*/scorer_targets_rank1.jsonl" ${LOGS_FILE:+--logs-file "$LOGS_FILE"}
  merge "$BANK/sc_aug_s*/scorer_targets_rank1.jsonl" "$BANK/sc_aug/scorer_targets_rank1.jsonl" log_name,token,step,rank,candidate
  echo "ZZOK score r0=$(wc -l < "$BANK/sc_r0/scorer_targets.jsonl") aug=$(wc -l < "$BANK/sc_aug/scorer_targets_rank1.jsonl") ZZ"
fi

if has assemble; then
  # ⛔ ONE TRAINING DIRECTORY, RANK-NAMED. train.py globs `targets_rank*.jsonl` and `scorer_targets*.jsonl`
  # in ONE dir. Left as `aug/targets_aug.jsonl` the augmented half was INVISIBLE to the trainer (it
  # now refuses a stray targets_*.jsonl rather than dropping it). Augmented rows carry rank 1.
  # Its own stage (it lived inside `score`): cheap to re-run, e.g. when a new input such as the
  # camera rigs (R22) is added, without recomputing a single scorer target.
  echo "=== assemble: the ONE directory the trainer reads ==="
  for f in "$BANK/r0/targets_rank0.jsonl" "$BANK/aug/targets_aug.jsonl" "$BANK/sc_r0/scorer_targets.jsonl" \
           "$BANK/sc_aug/scorer_targets_rank1.jsonl" "$BANK/calib_table.json"; do
    [ -s "$f" ] || { echo "ZZFAIL assemble: missing or empty $f ZZ"; exit 1; }
  done
  mkdir -p "$BANK/train"
  cp -f "$BANK/r0/targets_rank0.jsonl"            "$BANK/train/targets_rank0.jsonl"
  cp -f "$BANK/aug/targets_aug.jsonl"             "$BANK/train/targets_rank1.jsonl"
  cp -f "$BANK/sc_r0/scorer_targets.jsonl"        "$BANK/train/scorer_targets.jsonl"
  cp -f "$BANK/sc_aug/scorer_targets_rank1.jsonl" "$BANK/train/scorer_targets_rank1.jsonl"
  cp -f "$BANK/calib_table.json"                  "$BANK/train/calib_table.json"      # R22
  echo "ZZOK train dir $BANK/train: $(cat "$BANK"/train/targets_rank*.jsonl | wc -l) target rows + calib ZZ"
fi

if has gate; then
  # ⛔ ASSERTED, NOT PRINTED. This stage used to pipe both gates into `tail` and move on, so a
  # DIVERGED verdict scrolled past and the script still ended ZZDONE. And it gated `r0/`, not the
  # `train/` directory the trainer actually consumes (R23: the conformance gate reads BOTH ranks).
  echo "=== gate: the TRAINING directory, both ranks ==="
  "$PY" diag_consumer_conformance.py --targets "$BANK/train" > "$BANK/gate_conformance.txt" 2>&1
  tail -2 "$BANK/gate_conformance.txt"
  grep -q "^CONSUMERS_CONFORM" "$BANK/gate_conformance.txt" \
    || { echo "ZZFAIL gate conformance -- see $BANK/gate_conformance.txt ZZ"; exit 1; }
  for rk in 0 1; do
    sc="$BANK/train/scorer_targets.jsonl"; [ "$rk" = 1 ] && sc="$BANK/train/scorer_targets_rank1.jsonl"
    "$PY" diag_signal_consistency.py --bank "$BANK/train" --scorer "$sc" --rank "$rk" \
        > "$BANK/gate_signals_r$rk.txt" 2>&1
    tail -2 "$BANK/gate_signals_r$rk.txt"
    grep -q "SIGNALS_CONSISTENT" "$BANK/gate_signals_r$rk.txt" \
      || { echo "ZZFAIL gate signals rank $rk -- see $BANK/gate_signals_r$rk.txt ZZ"; exit 1; }
  done
  echo "ZZOK gate ZZ"
fi
echo "ZZDONE $(date -u +%FT%TZ) ZZ"
