#!/usr/bin/env bash
# pod_cropscale_fit_queue.sh -- pod queue of card PREREG_V8F_PROOF_ADD106 (CROPSCALE10K), section 2: the fits of the head D-H5 of card ADD99 on the large fit set (crop_fit.py and crop_dump.py, unchanged, from the kit
# folder of card ADD99) and their dumps on the rows of the row template folder. Tier T0. Derived from pod_visseed_fit_queue.sh (card ADD104).
# Stage FITS (default): fit seeds 2, 3, 4 (at most 3 at a time on the one GPU), then the three dumps.   Stage PERM=1: the permuted-label fit of seed 2 (--permute-labels) and its dump only (control g10); it writes
#   fits/D_H5_s2_perm and dump/D_H5_perm (the tools name them so, crop_fit.py:48-53), never a file of the real fit of seed 2 (fits/D_H5_s2, dump/D_H5), and has its own markers and job folder.
# Admission (both stages): ENC_CONTROLS.json reads g0 PASS and g1 PASS (the form {"g0": {"PASS": true}} or {"controls": {"g0": {"PASS": true}}}), memory free above 28 GB (memory.max minus anon), the lists and tools exist,
#   the large list starts with the pilot's list, every clip of the large list and of the evaluation list has a token file in the pilot's folder or in the new folder.
# What it builds (never writes into the run root of card ADD99, $R1, and never into the encode output):
#   $FR (the fit root, default $GP/fitroot): tok/D/ is a REAL folder with one link per clip (every clip of the large list and of the evaluation list) to $R1/tok/D/<clip>.npy when the pilot has it, else to
#   $NEWTOK/<clip>.npy, a clip found in neither is a STOP; lists/train_final.txt -> the large list, lists/eval141.txt -> the pilot's; real folders fits/ and dump/.
#   $TPL (the row template folder, default $GP/present_tpl): reader_s2_score.npz, reader_s3_score.npz, reader_s4_score.npz -> $PRESENT/reader_s0_score.npz (the present reader's dump of seed 0), as in card ADD104.
# A link that already exists must point to the same target, else STOP. Markers in $GP: START_<STAGE>.txt (a line per start), DONE_<STAGE>.txt, STOP_<STAGE>.txt (RESUME=1 continues), STAGE = FITS or PERM. Per step
#   in $GP/<fits_q|perm_q>/jobs: fit_s<S>[_perm].START/.DONE/.STOP, dump_s<S>[_perm].START/.DONE/.STOP; logs in .../logs. Resumable: a fit with its FIT_RECORD.json skips the fit, a dump with its score file and meta
#   file skips the dump; a score file without its meta file is a STOP. Never deletes, never moves, never kills.
# Usage: bash pod_cropscale_fit_queue.sh            (stage FITS)
#        PERM=1 bash pod_cropscale_fit_queue.sh     (stage PERM)
#        DRY=1 [PERM=1] bash pod_cropscale_fit_queue.sh     prints every command and makes nothing, not even a folder or a link
set -u
V=/workspace/v8proof
R1=${CS_RUN1:-$V/cropread/run1}
GP=${CS_ROOT:-$V/cropscale}
FR=${CS_FITROOT:-$GP/fitroot}
TPL=${CS_PRESENT_TPL:-$GP/present_tpl}
K=${CS_KIT:-$V/cropread/kit/fit}
LARGE=${CS_LARGE_LIST:-$GP/lists/train_final.txt}
NEWTOK=${CS_NEWTOK:-$GP/enc_out/tok/D}
ENCC=${CS_ENC_CONTROLS:-$GP/ENC_CONTROLS.json}
PY=${CS_PY:-/workspace/venv/bin/python}
PRESENT=${CS_PRESENT:-$V/closeread/dump/both}
JOIN=${CS_JOIN_TRAIN:-$V/data/inputs/join_train.npz}
MAXJ=${CS_MAX_JOBS:-3}
MINGB=${CS_MIN_FREE_GB:-28}
DRY=${DRY:-0}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONIOENCODING=utf-8
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
if [ "${PERM:-0}" = 1 ]; then
  STAGE=PERM; SEEDS=(2); SUF=_perm; PFLAG=--permute-labels; Q=$GP/perm_q; MAXJ=1
else
  STAGE=FITS; SEEDS=(2 3 4); SUF=""; PFLAG=""; Q=$GP/fits_q
fi
TEMPLATES=(2 3 4); [ "$STAGE" = PERM ] && TEMPLATES=(2)
fitcmd() { echo "$PY $K/crop_fit.py --arm D --head H5 --seed $1${PFLAG:+ $PFLAG} --root $FR --join-train $JOIN --device cuda"; }
dumpcmd() { echo "$PY $K/crop_dump.py --arm D --head H5 --seed $1${PFLAG:+ $PFLAG} --root $FR --present-dump $TPL --device cuda"; }
if [ "$DRY" = 1 ]; then
  echo "CMD: stage $STAGE; admission: $ENCC reads g0 PASS and g1 PASS, else STOP"
  echo "CMD: memory free > $MINGB GB (memory.max minus anon), else STOP"
  echo "CMD: the first lines of $LARGE (as many as $R1/lists/train_final.txt has, 500 on record) equal $R1/lists/train_final.txt, else STOP"
  echo "CMD: mkdir -p $GP $Q/jobs $Q/logs $FR/lists $FR/fits $FR/dump $FR/tok/D $TPL"
  echo "CMD: ln -s $LARGE $FR/lists/train_final.txt"
  echo "CMD: ln -s $R1/lists/eval141.txt $FR/lists/eval141.txt"
  echo "CMD: for every clip of $LARGE and of $R1/lists/eval141.txt: ln -s ($R1/tok/D|$NEWTOK)/<clip>.npy $FR/tok/D/<clip>.npy   (the pilot's file when it exists, else the new one, else STOP)"
  for s in "${TEMPLATES[@]}"; do echo "CMD: ln -s $PRESENT/reader_s0_score.npz $TPL/reader_s${s}_score.npz"; done
  for s in "${SEEDS[@]}"; do echo "JOB fit_s$s$SUF:"; echo "  CMD: $(fitcmd "$s")"; done
  for s in "${SEEDS[@]}"; do echo "JOB dump_s$s$SUF (after the fits):"; echo "  CMD: $(dumpcmd "$s")"; done
  echo "DRY run complete: stage $STAGE, at most $MAXJ jobs at a time"; exit 0
fi
stop() { mkdir -p "$GP"; echo "$(ts) $*" >> "$GP/STOP_$STAGE.txt"; echo "STOP $*"; exit 3; }
[ -d "$K" ] || { echo "REFUSED: no $K"; exit 2; }
[ -f "$GP/DONE_$STAGE.txt" ] && { echo "ALREADY-DONE"; exit 0; }
if [ -f "$GP/STOP_$STAGE.txt" ] && [ "${RESUME:-0}" != 1 ]; then echo "REFUSED: $GP/STOP_$STAGE.txt exists; RESUME=1 to continue"; exit 2; fi
need=("$K/crop_fit.py" "$K/crop_dump.py" "$JOIN" "$PRESENT/reader_s0_score.npz" "$LARGE" "$R1/lists/train_final.txt" "$R1/lists/eval141.txt" "$ENCC")
for f in "${need[@]}"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
adm=$("$PY" - "$ENCC" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError) as ex:
    print("UNREADABLE %s" % type(ex).__name__); sys.exit(1)
c = d.get("controls") if isinstance(d, dict) and isinstance(d.get("controls"), dict) else d
flag = lambda g: g.get("pass") if "pass" in g else g.get("PASS")      # the encode kit (ctl_g0g1.py) writes the key in lower case, and pass_g0, pass_g1 at the top of the record
top = lambda k: (not isinstance(d, dict)) or ("pass_" + k) not in d or d["pass_" + k] is True
bad = [k for k in ("g0", "g1") if not (isinstance(c, dict) and isinstance(c.get(k), dict) and flag(c[k]) is True and top(k))]
if bad:
    print("NOT-PASS " + ",".join(bad)); sys.exit(1)
print("ADMIT-OK g0 g1")
PYEOF
) || stop "ENC-CONTROLS $adm"
echo "$adm"
mmax=$(tr -d ' \r\n' < /sys/fs/cgroup/memory.max); anon=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat | head -n 1)
if [[ "$mmax" =~ ^[0-9]+$ ]] && [[ "$anon" =~ ^[0-9]+$ ]]; then [ $(( (mmax - anon) / 1073741824 )) -gt "$MINGB" ] || stop "MEMORY-FREE-TOO-LOW"; fi
npil=$(tr -d '\r' < "$R1/lists/train_final.txt" | grep -c .)
[ "$npil" -gt 0 ] && [ "$(head -n "$npil" "$LARGE" | tr -d '\r')" = "$(tr -d '\r' < "$R1/lists/train_final.txt")" ] || stop "LARGE-LIST-DOES-NOT-START-WITH-THE-PILOT-LIST"
mkdir -p "$GP" "$Q/jobs" "$Q/logs" "$FR/lists" "$FR/fits" "$FR/dump" "$FR/tok/D" "$TPL" || stop "CANNOT-MAKE-FOLDERS"
mklink() { # TARGET LINK: makes the link, or accepts the one that is already there when it points at TARGET
  if [ -L "$2" ] || [ -e "$2" ]; then [ "$(readlink "$2")" = "$1" ] || stop "LINK-IS-NOT-AS-EXPECTED $2"
  else ln -s "$1" "$2" 2>/dev/null || { [ "$(readlink "$2" 2>/dev/null)" = "$1" ] || stop "LINK-FAILED $2"; }; fi
}
mklink "$LARGE" "$FR/lists/train_final.txt"
mklink "$R1/lists/eval141.txt" "$FR/lists/eval141.txt"
nlink=0
for lst in "$LARGE" "$R1/lists/eval141.txt"; do
  while read -r c rest || [ -n "${c:-}" ]; do
    c=${c%$'\r'}; [ -n "$c" ] || continue
    if [ -e "$R1/tok/D/$c.npy" ]; then t="$R1/tok/D/$c.npy"
    elif [ -e "$NEWTOK/$c.npy" ]; then t="$NEWTOK/$c.npy"
    else stop "NO-TOKEN-FOR-CLIP $c"; fi
    mklink "$t" "$FR/tok/D/$c.npy"; nlink=$((nlink + 1))
  done < "$lst"
done
for s in "${TEMPLATES[@]}"; do mklink "$PRESENT/reader_s0_score.npz" "$TPL/reader_s${s}_score.npz"; done
echo "$(ts) start pid $$ resume ${RESUME:-0} token links $nlink" >> "$GP/START_$STAGE.txt"
run_fit() { # seed
  local s=$1 j="fit_s$1$SUF"
  echo "$(ts) START $j" > "$Q/jobs/$j.START"
  if [ ! -f "$FR/fits/D_H5_s$s$SUF/FIT_RECORD.json" ]; then
    $(fitcmd "$s") > "$Q/logs/$j.log" 2>&1 || { echo "$(ts) fit rc $?" > "$Q/jobs/$j.STOP"; return 1; }
  fi
  echo "$(ts) DONE $j" > "$Q/jobs/$j.DONE"
}
run_dump() { # seed
  local s=$1 j="dump_s$1$SUF" d="$FR/dump/D_H5$SUF"
  echo "$(ts) START $j" > "$Q/jobs/$j.START"
  if [ -f "$d/reader_s${s}_score.npz" ] && [ ! -f "$d/reader_s${s}_meta.json" ]; then echo "$(ts) half dump" > "$Q/jobs/$j.STOP"; return 1; fi
  if [ ! -f "$d/reader_s${s}_score.npz" ]; then
    $(dumpcmd "$s") > "$Q/logs/$j.log" 2>&1 || { echo "$(ts) dump rc $?" > "$Q/jobs/$j.STOP"; return 1; }
  fi
  echo "$(ts) DONE $j" > "$Q/jobs/$j.DONE"
}
blocked() { [ -f "$GP/STOP_$STAGE.txt" ] && [ "${RESUME:-0}" != 1 ]; }
for s in "${SEEDS[@]}"; do
  [ -f "$Q/jobs/fit_s$s$SUF.DONE" ] && continue
  [ -f "$Q/jobs/fit_s$s$SUF.STOP" ] && echo "$(ts) retry fit_s$s$SUF (an old .STOP stays on disk)" >> "$GP/START_$STAGE.txt"
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n || true; done
  blocked && break
  ( run_fit "$s" || echo "$(ts) job fit_s$s$SUF failed" >> "$GP/STOP_$STAGE.txt" ) &
done
wait
blocked && { echo "STOPPED: see $GP/STOP_$STAGE.txt"; exit 3; }
for s in "${SEEDS[@]}"; do [ -f "$Q/jobs/fit_s$s$SUF.DONE" ] || { echo "ZZ CROPSCALE-$STAGE fits INCOMPLETE"; exit 3; }; done
for s in "${SEEDS[@]}"; do
  [ -f "$Q/jobs/dump_s$s$SUF.DONE" ] && continue
  [ -f "$Q/jobs/dump_s$s$SUF.STOP" ] && echo "$(ts) retry dump_s$s$SUF (an old .STOP stays on disk)" >> "$GP/START_$STAGE.txt"
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n || true; done
  blocked && break
  ( run_dump "$s" || echo "$(ts) job dump_s$s$SUF failed" >> "$GP/STOP_$STAGE.txt" ) &
done
wait
blocked && { echo "STOPPED: see $GP/STOP_$STAGE.txt"; exit 3; }
ok=1
for s in "${SEEDS[@]}"; do [ -f "$Q/jobs/fit_s$s$SUF.DONE" ] && [ -f "$Q/jobs/dump_s$s$SUF.DONE" ] || ok=0; done
if [ "$ok" = 1 ]; then echo "$(ts) all steps done" > "$GP/DONE_$STAGE.txt"; echo "ZZ CROPSCALE-$STAGE DONE"; else echo "ZZ CROPSCALE-$STAGE INCOMPLETE"; exit 3; fi
