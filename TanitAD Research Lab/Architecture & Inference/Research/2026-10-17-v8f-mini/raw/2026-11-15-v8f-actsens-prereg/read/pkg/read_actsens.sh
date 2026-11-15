#!/bin/bash
# ACTSENS READ of ONE banked arm (card section 4). Tier T0 (oracle goal: a world-model diagnostic, never driving performance). Dev-box CPU only; no training; never writes to box 2 (read-only sftp of the 141 TRAIN clips).
# usage:  bash read_actsens.sh <tag> <chassis-seed 0|1>        e.g.  bash read_actsens.sh IDM_s0 0     NULL_s2 0     NULL_s3 1     DIV_s1 1
# env (optional):  ARM_RUN_DIR (default C:/Users/Admin/t2v8f_pulls/actsens_<tag>)  ARM_LATENTS (reuse an existing latents file keyed by <tag>)  ARM_C7=1 (control C7 extras)  HWM_WORKERS (default 3)
#                  OUTROOT (default F:/TanitAD-scratch/actsens/read/out)  SKIP_DECODE=1 (do not run decode_check)
# Fails closed on any missing input and never overwrites a finished arm: a finished arm has ARM_DONE_<tag>.txt and arm_<tag>.json. A killed run resumes (banked windows / plans / latents are skipped).
# Output: $OUTROOT/<tag>/ (all intermediates), $OUTROOT/arm_<tag>.json (the one JSON per arm, input of verdict_actsens.py), $OUTROOT/<tag>/ARM_DONE_<tag>.txt (own done marker file).
# C7DIAG 2026-11-15 (original kept as read_actsens.sh.orig_c7diag_20261115, sha256 7b379fa8): the windows and plan stages run at torch intra-op threads 1, as phase 1 did
# (phase 1: common.sh MKL_NUM_THREADS=1, phase1_R35.sh raised only OMP to 2; torch reads MKL_NUM_THREADS after OMP_NUM_THREADS, so it ran at 1 thread). At 2 threads the
# HWM plans of R35_s0 are not bit-equal to the banked ones (MEASURED, c7diag/HANDOFF). A probe before step 4 fails closed unless torch.get_num_threads() == 1.
set -u
TAG=${1:?usage: read_actsens.sh <tag> <chassis-seed>}; CH=${2:?chassis seed 0 or 1}
case "$CH" in 0|1) ;; *) echo "FAIL-CLOSED: chassis seed must be 0 or 1"; exit 2;; esac
source /f/TanitAD-scratch/actsens/read/env.sh
RUN_DIR=${ARM_RUN_DIR:-$PULLS/actsens_$TAG}
OUT=$OUTROOT/$TAG; WD=$OUT/win; PD=$WD; LOG=$OUT/read.log
W=${HWM_WORKERS:-3}; KEY=${ARM_KEY:-$TAG}
fail() { echo "$(date +%T) FAIL-CLOSED: $*" | tee -a "$LOG"; echo "rc=1 $*" > "$OUT/ARM_FAILED_$TAG.txt"; exit 1; }
mkdir -p "$OUT" "$WD" "$OUT/decode_check"
if [ -f "$OUT/ARM_DONE_$TAG.txt" ] || [ -f "$OUTROOT/arm_$TAG.json" ]; then echo "REFUSED: arm $TAG already read (outputs are never overwritten)"; exit 4; fi
rm -f "$OUT/ARM_FAILED_$TAG.txt"
echo "$(date +%T) read $TAG chassis $CH run_dir $RUN_DIR workers $W" | tee -a "$LOG"

# ---- inputs, fail closed
for f in ckpt_last.pt argv.txt; do [ -s "$RUN_DIR/$f" ] || fail "missing $RUN_DIR/$f (arm not banked)"; done
[ -f "$RUN_DIR/BANK.done" ] || [ -f "$RUN_DIR/RUN.done" ] || fail "no BANK.done / RUN.done in $RUN_DIR"
[ -s "$R35REF/s$CH/windows.npz" ] && [ -s "$R35REF/s$CH/plans_hwm_oracle_s0.npz" ] || fail "R35 phase-1 reference missing (run hf_fetch.py)"
$PY $RD/check_tree.py </dev/null >> "$LOG" 2>&1 || fail "check_tree.py (tree / code-of-record shas / R35 reference shas)"
$PY -c "
import torch,sys
ck=torch.load('$RUN_DIR/ckpt_last.pt',map_location='cpu',weights_only=False)
print('ckpt step',ck.get('step'),'gamma levels',ck['cfg'].get('wm_gamma_levels'))" </dev/null >> "$LOG" 2>&1 || fail "checkpoint does not load under the ACTSENS tree"
FREE=$(df -m /f | awk 'NR==2{print $4}'); [ "$FREE" -gt 1500 ] || fail "F: has only $FREE MB free"

# ---- RAM guard (as the phase-1 waiter: 6 GB free physical), wait up to 2 h
for i in $(seq 1 24); do
  FKB=$(powershell -NoProfile -c "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" </dev/null | tr -dc '0-9')
  [ "${FKB:-0}" -ge 6000000 ] && break
  echo "$(date +%T) waiting for 6 GB free RAM (have ${FKB:-?} KB)" >> "$LOG"; sleep 300
done
[ "${FKB:-0}" -ge 6000000 ] || fail "RAM below 6 GB for 2 h"

# ---- 1  IA-3 / ladder read of the arm (card 4.1), ~5.7 min
if [ ! -f "$OUT/ia_read.json" ]; then
  echo "$(date +%T) IA read" | tee -a "$LOG"
  OMP_NUM_THREADS=3 $PY -u $RD/ia_read.py --name "$TAG" --run-dir "$RUN_DIR" --chassis-seed "$CH" --out "$OUT/ia_read.json" </dev/null >> "$LOG" 2>&1 || fail "ia_read.py"
fi

# ---- 2  windows of the arm (its own adapter + proj) and C6 against R35's phase-1 windows
if [ ! -f "$WD/windows.npz" ]; then
  echo "$(date +%T) windows" | tee -a "$LOG"
  cd $H
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 $PY -u run_hwm.py windows --tag $KEY --ckpt $RUN_DIR/ckpt_last.pt --train-latents $HWMTOOLS/latents_train_R35_s0.pt --train-tag R35_s0 --seed-mode prior --lam 0 --out $WD </dev/null >> "$LOG" 2>&1 || fail "run_hwm windows"
  $PY -u $HWMTOOLS/subset_score.py $WD </dev/null >> "$LOG" 2>&1 || fail "subset_score"
fi
$PY $RD/c6_windows.py $WD $CH $OUT/c6.json </dev/null >> "$LOG" 2>&1 || fail "C6 WINDOWS-EQUAL failed: the arm's SCORE windows differ from R35's (read is INVALID)"

# ---- 3  the arm's own TRAIN latents (box 2 read-only sftp, CPU; 141 clips x 17 starts)
LAT=${ARM_LATENTS:-$OUT/latents_train_$TAG.pt}
if [ ! -f "$LAT" ]; then
  echo "$(date +%T) collect latents" | tee -a "$LOG"
  mkdir -p $OUT/ts/cache/train/clips; cp -n $RD/ts/clips141.json $OUT/ts/
  COL_OUT_DIR=$OUT COL_TAG=$KEY COL_TS=$OUT/ts COL_CKPT=$RUN_DIR/ckpt_last.pt OMP_NUM_THREADS=4 $PY -u $RD/collect_latents_arm.py </dev/null >> "$OUT/collect.log" 2>&1 || fail "latent collection (see collect.log)"
fi
[ -s "$LAT" ] || fail "no latents file $LAT"

# ---- 4  plans, phase 1's levers exactly: --seed-mode prior --lam 0, T3 2 (default K2), plan-ctx 1, stitch decode, vocab off, constraint weight 0 (all defaults of run_hwm.py)
cd $H
OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 $PY -c "import torch,sys; n=torch.get_num_threads(); print('plan env torch threads', n); sys.exit(0 if n == 1 else 1)" </dev/null >> "$LOG" 2>&1 || fail "plan env does not give torch 1 intra-op thread (phase 1 ran at 1)"
COM="--tag $KEY --ckpt $RUN_DIR/ckpt_last.pt --train-latents $LAT --train-tag $KEY --seed-mode prior --lam 0 --out $WD"
for ARM in hwm_oracle flat_oracle; do for SEED in 0 1; do
  if [ ! -f $WD/plans_${ARM}_s$SEED.npz ]; then
    echo "$(date +%T) plan $ARM seed $SEED" | tee -a "$LOG"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 $PY -u run_hwm.py plan $COM --arms $ARM --seeds $SEED --workers $W </dev/null >> "$LOG" 2>&1 || fail "run_hwm plan $ARM $SEED"
  fi
done; done

# ---- 5  contrast_l1full.py (validated reader): arm vs R35_k plans and vs hold; replicates; four families per band
V=$HWMTOOLS/phase2_l1full
for K in hwm flat; do
  if [ ! -f $OUT/contrast_$K.json ]; then
    echo "$(date +%T) contrast $K" | tee -a "$LOG"
    $PY $V/contrast_l1full.py --windows $R35REF/s$CH --a $WD/plans_${K}_oracle_s0.npz --b $R35REF/s$CH/plans_${K}_oracle_s0.npz --rep-a $WD/plans_${K}_oracle_s1.npz --rep-b $R35REF/s$CH/plans_${K}_oracle_s1.npz --out $OUT/contrast_$K.json </dev/null > $OUT/contrast_$K.log 2>&1 || fail "contrast_l1full $K"
  fi
done

# ---- 6  off-image diagnostic (decode_check, code of record with paths from env), 6 batches of 64 planned windows
if [ "${SKIP_DECODE:-0}" != "1" ] && [ ! -f $OUT/decode_check/result_$TAG.json ]; then
  echo "$(date +%T) decode_check" | tee -a "$LOG"
  [ -f $RD/decode_check_arm.py ] || $PY $RD/make_decode_check_arm.py </dev/null >> "$LOG" 2>&1 || fail "make_decode_check_arm"
  cd $RD
  DC_TAG=$KEY DC_OUT_DIR=$OUT/decode_check DC_WIN_DIR=$WD DC_CKPT=$RUN_DIR/ckpt_last.pt DC_LATENTS=$LAT DC_PLANS=$WD/plans_hwm_oracle_s0.npz OMP_NUM_THREADS=2 $PY -u $RD/decode_check_arm.py $TAG 6 </dev/null > $OUT/decode_check/run.log 2>&1 || fail "decode_check"
fi

# ---- 7  control C7 extras (only when ARM_C7=1)
if [ "${ARM_C7:-0}" = "1" ]; then
  echo "$(date +%T) C7 checks" | tee -a "$LOG"
  $PY $V/contrast_l1full.py --windows $R35REF/s$CH --a $WD/plans_hwm_oracle_s0.npz --b $WD/plans_flat_oracle_s0.npz --rep-a $WD/plans_hwm_oracle_s1.npz --rep-b $WD/plans_flat_oracle_s1.npz --out $OUT/c7_check_rows.json --check $R35REF/s$CH/RESULT_HWMPLAN_R35_s$CH.json --check-a hwm_oracle --check-b flat_oracle </dev/null > $OUT/c7_check.log 2>&1
  $PY $RD/c7_checks.py $OUT $CH </dev/null >> $OUT/c7_check.log 2>&1 || echo "C7 FAIL (see c7_check.log)" | tee -a "$LOG"
fi

# ---- 8  assemble the one JSON, then the done marker (its own file)
$PY $RD/assemble_arm.py $OUT $TAG $CH $OUT/arm_$TAG.json </dev/null >> "$LOG" 2>&1 || fail "assemble_arm"
cp $OUT/arm_$TAG.json $OUTROOT/arm_$TAG.json
echo "rc=0 $(date -Iseconds) tier=T0" > $OUT/ARM_DONE_$TAG.txt
echo "$(date +%T) DONE $TAG" | tee -a "$LOG"
