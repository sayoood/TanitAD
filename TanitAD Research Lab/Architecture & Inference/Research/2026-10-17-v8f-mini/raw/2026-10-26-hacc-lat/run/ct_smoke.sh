#!/bin/bash
# C-T / C-T2 smoke (PREREG_V8F_HACC_LAT section 3): the ORIGINAL tool (the landed-tip export, read-only) and the COPY with the flag ABSENT must write bit-identical output; with --dump-ticks
# the tick-0 plans must equal the dump's `ol` and the run's output must still equal the original's. Code from $CODE (default hlat/code; the chain sets the FROZEN copy). Usage: ct_smoke.sh <seed 0|1> <max-windows> <threads> <tag>
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
S=${1:-0}; NW=${2:-24}; TH=${3:-4}; TAG=${4:-ctsmoke}
H=C:/Users/Admin/hlat; CODE=${CODE:-C:/Users/Admin/hlat/code}; O=C:/Users/Admin/t1k; SC=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
CK=E:/Projects/TanitAD-artifacts/v8f-mini-hacc/M2b_s${S}_hacc.pt
GRID="$H/inputs/eval_grid_windows.json"; JOIN="$H/inputs/join_eval141.npz"
[ "$(md5sum < "$GRID" | cut -c1-32)" = 34ed8ac56365eeff72e42581bde2f5c4 ] || { echo "ZZCT-STOP grid md5"; exit 1; }
[ "$(md5sum < "$JOIN" | cut -c1-32)" = 532fcd644c0e0835e847b2f1cee011ad ] || { echo "ZZCT-STOP join md5"; exit 1; }
W=$H/controls/$TAG; rm -rf "$W"; mkdir -p "$W/orig" "$W/new" "$W/new_ticks" "$W/ticks"
COMMON="run --ckpt $CK --tag CT --tracks $SC/sfresh/inputs/native_tracks.pt --eval-cache $SC/sfresh/cache --eval-grid $GRID --eval-join $JOIN --replan-steps 10 --device cpu --max-windows $NW --n-boot 50"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH PYTHONIOENCODING=utf-8
echo "== original tool"
PYTHONPATH="$O/stack;$O/taniteval;$O/stack/scripts" $PY $O/stack/scripts/v8f_t1_eval.py $COMMON --out-dir $W/orig > $W/orig.out 2>&1 || { echo "ZZCT-STOP original run failed"; tail -5 $W/orig.out; exit 1; }
echo "== copy, flag absent"
PYTHONPATH="$CODE/stack;$CODE/taniteval;$CODE/stack/scripts" $PY $CODE/stack/scripts/v8f_t1_eval.py $COMMON --out-dir $W/new > $W/new.out 2>&1 || { echo "ZZCT-STOP copy run failed"; tail -5 $W/new.out; exit 1; }
echo "== copy, --dump-ticks"
PYTHONPATH="$CODE/stack;$CODE/taniteval;$CODE/stack/scripts" $PY $CODE/stack/scripts/v8f_t1_eval.py $COMMON --out-dir $W/new_ticks --dump-ticks $W/ticks > $W/new_ticks.out 2>&1 || { echo "ZZCT-STOP ticks run failed"; tail -5 $W/new_ticks.out; exit 1; }
PYTHONPATH="$CODE/stack;$CODE/taniteval;$CODE/stack/scripts" $PY $CODE/stack/scripts/v8f_hlat.py check-ct --a $W/orig --b $W/new --out $W/C-T.json || { echo "ZZCT-STOP C-T failed"; exit 1; }
$PY $H/ct_compare_ticks.py $W || { echo "ZZCT-STOP ticks compare failed"; exit 1; }
echo "ZZCT-DONE"
