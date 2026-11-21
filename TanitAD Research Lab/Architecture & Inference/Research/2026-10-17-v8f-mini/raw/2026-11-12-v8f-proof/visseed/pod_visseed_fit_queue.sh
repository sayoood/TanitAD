#!/usr/bin/env bash
# pod_visseed_fit_queue.sh -- pod queue of card PREREG_V8F_PROOF_ADD104 (VISSEED10K), section 2: the three fits of the head D-H5 of card ADD99 with fit seeds 2, 3 and 4 (crop_fit.py, unchanged, from the kit folder of card ADD99),
# at most 3 at a time on the one GPU, and then the three dumps (crop_dump.py, unchanged) on the rows of the row template folder. Tier T0. Derived from pod_cropread_fit_queue.sh (card ADD99) and pod_steer_fit_queue.sh (card ADD100).
# What it builds (never writes into the run root of card ADD99, $R1):
#   $FR (the fit root, default $GP/fitroot): tok -> $R1/tok (one link to the folder of the tokens), lists/{train_final.txt,eval141.txt,LISTS_RECORD.json} -> the same files of $R1/lists, and the real folders fits/ and dump/
#   $TPL (the row template folder, default $GP/present_tpl): reader_s2_score.npz, reader_s3_score.npz, reader_s4_score.npz -> $PRESENT/reader_s0_score.npz (the present reader's dump of seed 0)
# What the tools read and write under --root (checked in crop_fit.py and crop_dump.py): crop_fit.py reads <root>/lists/train_final.txt, <root>/lists/eval141.txt, <root>/tok/D/<clip>.npy and the train join; it writes
#   <root>/fits/D_H5_s<S>/head.pt and FIT_RECORD.json (and refuses when that FIT_RECORD.json exists). crop_dump.py reads <root>/fits/D_H5_s<S>/{head.pt,FIT_RECORD.json}, <root>/tok/D/<clip>.npy and
#   <present-dump>/reader_s<S>_score.npz; it writes <root>/dump/D_H5/reader_s<S>_score.npz and reader_s<S>_meta.json (and refuses when the score file exists).
# Markers in $GP: START_FITS.txt (a line per start, appended), DONE_FITS.txt (all six steps done), STOP_FITS.txt (a step failed; new steps stop; RESUME=1 continues). Per step in $GP/fits_q/jobs: fit_s<S>.START/.DONE/.STOP,
#   dump_s<S>.START/.DONE/.STOP; logs in $GP/fits_q/logs. Resumable: a fit with its FIT_RECORD.json skips the fit, a dump with its score file and meta file skips the dump. Never deletes, never moves, never kills.
# Usage: bash pod_visseed_fit_queue.sh      (DRY=1 prints every command and runs nothing, not even a folder or a link)
set -u
V=/workspace/v8proof
R1=${VS_RUN1:-$V/cropread/run1}
GP=${VS_ROOT:-$V/visseed10k}
FR=${VS_FITROOT:-$GP/fitroot}
TPL=${VS_PRESENT_TPL:-$GP/present_tpl}
K=${VS_KIT:-$V/cropread/kit/fit}
Q=$GP/fits_q
PY=${VS_PY:-/workspace/venv/bin/python}
PRESENT=${VS_PRESENT:-$V/closeread/dump/both}
JOIN=${VS_JOIN_TRAIN:-$V/data/inputs/join_train.npz}
MAXJ=${VS_MAX_JOBS:-3}
MINGB=${VS_MIN_FREE_GB:-28}
DRY=${DRY:-0}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONIOENCODING=utf-8
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
SEEDS=(2 3 4)
LISTS="train_final.txt eval141.txt LISTS_RECORD.json"
fitcmd() { echo "$PY $K/crop_fit.py --arm D --head H5 --seed $1 --root $FR --join-train $JOIN --device cuda"; }
dumpcmd() { echo "$PY $K/crop_dump.py --arm D --head H5 --seed $1 --root $FR --present-dump $TPL --device cuda"; }
if [ "$DRY" = 1 ]; then
  echo "CMD: memory free > $MINGB GB (memory.max minus anon), else STOP"
  echo "CMD: mkdir -p $GP $Q/jobs $Q/logs $FR/lists $FR/fits $FR/dump $TPL"
  echo "CMD: ln -s $R1/tok $FR/tok"
  for f in $LISTS; do echo "CMD: ln -s $R1/lists/$f $FR/lists/$f"; done
  for s in "${SEEDS[@]}"; do echo "CMD: ln -s $PRESENT/reader_s0_score.npz $TPL/reader_s${s}_score.npz"; done
  for s in "${SEEDS[@]}"; do echo "JOB fit_s$s:"; echo "  CMD: $(fitcmd "$s")"; done
  for s in "${SEEDS[@]}"; do echo "JOB dump_s$s (after the three fits):"; echo "  CMD: $(dumpcmd "$s")"; done
  echo "DRY run complete: at most $MAXJ jobs at a time"; exit 0
fi
stop() { mkdir -p "$GP"; echo "$(ts) $*" >> "$GP/STOP_FITS.txt"; echo "STOP $*"; exit 3; }
[ -d "$K" ] || { echo "REFUSED: no $K"; exit 2; }
[ -f "$GP/DONE_FITS.txt" ] && { echo "ALREADY-DONE"; exit 0; }
if [ -f "$GP/STOP_FITS.txt" ] && [ "${RESUME:-0}" != 1 ]; then echo "REFUSED: $GP/STOP_FITS.txt exists; RESUME=1 to continue"; exit 2; fi
need=("$K/crop_fit.py" "$K/crop_dump.py" "$JOIN" "$PRESENT/reader_s0_score.npz" "$R1/tok/D/MANIFEST.json")
for f in $LISTS; do need+=("$R1/lists/$f"); done
for f in "${need[@]}"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
mmax=$(tr -d ' \r\n' < /sys/fs/cgroup/memory.max); anon=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat | head -n 1)
if [[ "$mmax" =~ ^[0-9]+$ ]] && [[ "$anon" =~ ^[0-9]+$ ]]; then [ $(( (mmax - anon) / 1073741824 )) -gt "$MINGB" ] || stop "MEMORY-FREE-TOO-LOW"; fi
mkdir -p "$GP" "$Q/jobs" "$Q/logs" "$FR/lists" "$FR/fits" "$FR/dump" "$TPL" || stop "CANNOT-MAKE-FOLDERS"
mklink() { # TARGET LINK: makes the link, or accepts the one that is already there when it points at TARGET
  if [ -L "$2" ] || [ -e "$2" ]; then [ "$(readlink "$2")" = "$1" ] || stop "LINK-IS-NOT-AS-EXPECTED $2"; else ln -s "$1" "$2" || stop "LINK-FAILED $2"; fi
}
mklink "$R1/tok" "$FR/tok"
for f in $LISTS; do mklink "$R1/lists/$f" "$FR/lists/$f"; done
for s in "${SEEDS[@]}"; do mklink "$PRESENT/reader_s0_score.npz" "$TPL/reader_s${s}_score.npz"; done
echo "$(ts) start pid $$ resume ${RESUME:-0}" >> "$GP/START_FITS.txt"
run_fit() { # seed
  local s=$1
  echo "$(ts) START fit_s$s" > "$Q/jobs/fit_s$s.START"
  if [ ! -f "$FR/fits/D_H5_s$s/FIT_RECORD.json" ]; then
    $(fitcmd "$s") > "$Q/logs/fit_s$s.log" 2>&1 || { echo "$(ts) fit rc $?" > "$Q/jobs/fit_s$s.STOP"; return 1; }
  fi
  echo "$(ts) DONE fit_s$s" > "$Q/jobs/fit_s$s.DONE"
}
run_dump() { # seed
  local s=$1 d="$FR/dump/D_H5"
  echo "$(ts) START dump_s$s" > "$Q/jobs/dump_s$s.START"
  if [ -f "$d/reader_s${s}_score.npz" ] && [ ! -f "$d/reader_s${s}_meta.json" ]; then echo "$(ts) half dump" > "$Q/jobs/dump_s$s.STOP"; return 1; fi
  if [ ! -f "$d/reader_s${s}_score.npz" ]; then
    $(dumpcmd "$s") > "$Q/logs/dump_s$s.log" 2>&1 || { echo "$(ts) dump rc $?" > "$Q/jobs/dump_s$s.STOP"; return 1; }
  fi
  echo "$(ts) DONE dump_s$s" > "$Q/jobs/dump_s$s.DONE"
}
blocked() { [ -f "$GP/STOP_FITS.txt" ] && [ "${RESUME:-0}" != 1 ]; }
for s in "${SEEDS[@]}"; do
  [ -f "$Q/jobs/fit_s$s.DONE" ] && continue
  [ -f "$Q/jobs/fit_s$s.STOP" ] && echo "$(ts) retry fit_s$s (an old .STOP stays on disk)" >> "$GP/START_FITS.txt"
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n || true; done
  blocked && break
  ( run_fit "$s" || echo "$(ts) job fit_s$s failed" >> "$GP/STOP_FITS.txt" ) &
done
wait
blocked && { echo "STOPPED: see $GP/STOP_FITS.txt"; exit 3; }
for s in "${SEEDS[@]}"; do [ -f "$Q/jobs/fit_s$s.DONE" ] || { echo "ZZ VISSEED-FITQ fits INCOMPLETE"; exit 3; }; done
for s in "${SEEDS[@]}"; do
  [ -f "$Q/jobs/dump_s$s.DONE" ] && continue
  [ -f "$Q/jobs/dump_s$s.STOP" ] && echo "$(ts) retry dump_s$s (an old .STOP stays on disk)" >> "$GP/START_FITS.txt"
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n || true; done
  blocked && break
  ( run_dump "$s" || echo "$(ts) job dump_s$s failed" >> "$GP/STOP_FITS.txt" ) &
done
wait
blocked && { echo "STOPPED: see $GP/STOP_FITS.txt"; exit 3; }
ok=1
for s in "${SEEDS[@]}"; do [ -f "$Q/jobs/fit_s$s.DONE" ] && [ -f "$Q/jobs/dump_s$s.DONE" ] || ok=0; done
if [ "$ok" = 1 ]; then echo "$(ts) all six steps done" > "$GP/DONE_FITS.txt"; echo "ZZ VISSEED-FITQ DONE"; else echo "ZZ VISSEED-FITQ INCOMPLETE"; exit 3; fi
