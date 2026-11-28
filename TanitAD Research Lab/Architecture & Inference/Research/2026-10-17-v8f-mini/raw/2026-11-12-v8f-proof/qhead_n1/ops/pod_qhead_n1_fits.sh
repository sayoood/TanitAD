#!/bin/bash
# pod_qhead_n1_fits.sh -- runs ON THE POD (nohup), CPU only. Note PREREG_V8F_PROOF_ADD42_N1: the four fits of the note (qhead_fit_n1.py: fit 800 clips, early
# stopping on 200 select clips, constant shift, check on the 200 held-out clips of the card). One fit at a time; each starts only with 20 GB of the container
# memory limit free. No GPU job, no process test. Deletes nothing. A second start is refused by a marker. Last line: ZZ QHEAD-N1 FITS DONE <n written> of 4.
set -u
W=/workspace/v8proof; D=$W/distrule; ST=$W/probe/states; RUNF=$W/qhead_run_n1; HEADS=$RUNF/heads
PY=/workspace/venv/bin/python
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
mkdir -p "$HEADS" || exit 9
( set -o noclobber; echo "started $(ts)" > "$RUNF/FITS_STARTED.txt" ) 2>/dev/null || { echo "[$(ts)] REFUSED: $RUNF/FITS_STARTED.txt exists"; exit 3; }
freegb() { local a m; a=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat); m=$(cat /sys/fs/cgroup/memory.max); echo $(( (m - a) / 1073741824 )); }
echo "[$(ts)] tool sha256 fit $(sha256sum $D/qhead_fit_n1.py | cut -c1-16) model $(sha256sum $D/qhead_model.py | cut -c1-16) pack $(sha256sum $D/train_pack.npz | cut -c1-16)"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
n=0
for line in "LS_s0 0" "LS_s1 1" "LSD_s0 0" "LSD_s1 1"; do
  set -- $line; BASE=$1; SEED=$2
  k=0; while [ "$(freegb)" -lt 20 ]; do k=$((k+1)); [ $k -gt 60 ] && { echo "[$(ts)] ZZ QHEAD-N1 FITS STOP: under 20 GB free for 1 h"; exit 8; }; sleep 60; done
  [ -f "$ST/${BASE}_train.npz" ] || { echo "[$(ts)] no train states for $BASE"; continue; }
  EXTRA=()
  [ -f "$ST/${BASE}_train_lead.npz" ] && EXTRA+=(--lead "$ST/${BASE}_train_lead.npz")
  [ -f "$ST/${BASE}_train_lead.npz.head.npz" ] && EXTRA+=(--banked-head "$ST/${BASE}_train_lead.npz.head.npz")
  "$PY" "$D/qhead_fit_n1.py" --arm "$BASE" --seed "$SEED" --states "$ST/${BASE}_train.npz" --pack "$D/train_pack.npz" --out-dir "$HEADS" --device cpu ${EXTRA[@]+"${EXTRA[@]}"} > "$RUNF/fit_$BASE.out" 2>&1
  rc=$?
  echo "[$(ts)] fit $BASE rc $rc : $(tail -n 1 "$RUNF/fit_$BASE.out" | cut -c1-260)"
  [ -f "$HEADS/$BASE.qhead.json" ] && n=$((n+1))
done
echo "[$(ts)] ZZ QHEAD-N1 FITS DONE $n of 4"
