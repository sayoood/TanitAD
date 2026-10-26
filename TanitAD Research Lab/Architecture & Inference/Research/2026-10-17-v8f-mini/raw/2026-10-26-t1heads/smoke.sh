#!/bin/bash
# OPTIONAL real-data smoke (24 windows, seed 0, CPU, replan 10): the tip tool vs the new tool with flags OFF vs the new tool with --dump-ticks --dump-heads.
# Inputs exactly as hlat/ct_smoke.sh (already on disk, no download). Wall time per stage = the whole process (model load + rollout + analysis).
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
T=C:/Users/Admin/t1heads; H=C:/Users/Admin/hlat; NW=24; TH=4
SC=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
CK=E:/Projects/TanitAD-artifacts/v8f-mini-hacc/M2b_s0_hacc.pt
GRID="$H/inputs/eval_grid_windows.json"; JOIN="$H/inputs/join_eval141.npz"
[ "$(md5sum < "$GRID" | cut -c1-32)" = 34ed8ac56365eeff72e42581bde2f5c4 ] || { echo "STOP grid md5"; exit 1; }
[ "$(md5sum < "$JOIN" | cut -c1-32)" = 532fcd644c0e0835e847b2f1cee011ad ] || { echo "STOP join md5"; exit 1; }
W=$T/smoke; rm -rf "$W"; mkdir -p "$W/ticks"
COMMON="run --ckpt $CK --tag SM --tracks $SC/sfresh/inputs/native_tracks.pt --eval-cache $SC/sfresh/cache --eval-grid $GRID --eval-join $JOIN --replan-steps 10 --device cpu --max-windows $NW --n-boot 50"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH PYTHONIOENCODING=utf-8
R=$T/tree; export PYTHONPATH="$R/stack;$R/taniteval;$R/stack/scripts"
cp $T/ref/v8f_t1_eval.tip.lf.py $R/stack/scripts/_v8f_t1_eval_tip.py          # the tip tool beside the new one (REPO resolves to the same tree); removed below
stage() { local L=$1 S=$2; shift 2; local s=$(date +%s); $PY $R/stack/scripts/$S $COMMON --out-dir $W/$L "$@" > $W/$L.out 2>&1 || { echo "STOP $L failed"; tail -5 $W/$L.out; rm -f $R/stack/scripts/_v8f_t1_eval_tip.py; exit 1; }; echo "$L wall_s=$(( $(date +%s) - s ))  rollout: $(grep -E '^\[24/24\]' $W/$L.out)"; }
stage tip   _v8f_t1_eval_tip.py
stage off   v8f_t1_eval.py
stage heads v8f_t1_eval.py --dump-ticks $W/ticks --dump-heads
stage tip2  _v8f_t1_eval_tip.py          # a second tip run, to see the run-to-run spread of the box (the other session's chain shares the CPU)
rm -f $R/stack/scripts/_v8f_t1_eval_tip.py
echo "SMOKE-DONE"
