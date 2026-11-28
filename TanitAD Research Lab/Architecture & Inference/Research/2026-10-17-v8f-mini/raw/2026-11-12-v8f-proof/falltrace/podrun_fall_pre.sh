#!/bin/bash
# podrun_fall_pre.sh -- pod side, CPU only: runs fall_pre.py (second count, definitions FALLPRE_DEFS.md) on the banked T1-kin rollouts whose links the first count made.
# Usage: bash podrun_fall_pre.sh        (ARMS=VD5c,F2,F3,F4,E5c by default; OUTNAME=out_pre by default; NOEXPECT=1 for arms with no expected count)
set -u
PY=/workspace/venv/bin/python
V=/workspace/v8proof; W=$V/integ10k; P=$V/podread; RD=$P/root/integ10k; CODE=$P/code
R=$V/falltrace; K=$R/kit; LINKS=$R/pull
OUT=$R/${OUTNAME:-out_pre}
ARMS=${ARMS:-VD5c,F2,F3,F4,E5c}
JOIN=$P/data/join_eval141.npz
[ -e "$OUT" ] && { echo "ZZ PODRUN-FALLPRE STOP $OUT exists"; exit 3; }
[ -d "$LINKS" ] || { echo "ZZ PODRUN-FALLPRE STOP no links folder"; exit 3; }
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev HEADWAY_JOIN=$JOIN HEADWAY_BANK=$P/data/frames.npz SR_ESTREAD_DIR=$RD
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
EX=""; [ "${NOEXPECT:-0}" = 1 ] && EX="--no-expect"
( cd "$RD" && "$PY" "$K/fall_pre.py" --selftest ) | tail -n 1
( cd "$RD" && "$PY" "$K/fall_pre.py" --pull "$LINKS" --grid-dir "$W/longroll" --join "$JOIN" --code "$CODE" --out "$OUT" --arms "$ARMS" $EX > "$R/fall_pre_${OUTNAME:-out_pre}.log" 2>&1 ); rc=$?
tail -n 1 "$R/fall_pre_${OUTNAME:-out_pre}.log" | cut -c1-200
echo "ZZ PODRUN-FALLPRE DONE rc $rc"
