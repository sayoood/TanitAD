#!/bin/bash
# The producer->pixels->trainer seam, on the REAL pod layout, with arms that must fail (R21).
# Inputs (dev-box scratch, 2026-09-23): TARGETS = the real producer's 61 rank-0 rows for the one
# navtrain log held both as a DB and inside OpenScene shard 25; PIX = that log's 4 cameras
# extracted from navtrain_current_25.tgz with the tarball's own prefix (as pod_dataprep.sh does);
# PIX_M1 = PIX minus one CAM_L0 frame; PIX_M2 = PIX minus the whole CAM_B0 directory.
set -u
PY="${PY:?}"; TARGETS="${TARGETS:?}"; PIX="${PIX:?}"; PIX_M1="${PIX_M1:?}"; PIX_M2="${PIX_M2:?}"
cd "${REFE_DIR:?}"
run () { OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 "$PY" train.py --backbone vits16 --cpu \
           --targets "$TARGETS" --scorer-targets "$TARGETS" --steps 2 --batch 1 --log-every 1 "$@" 2>&1 \
         | grep -E "images:|DROPPED|TargetBank:|bank:|TRAIN_" | sed "s#$PIX_M1#<PIX_M1>#; s#$PIX_M2#<PIX_M2>#; s#$PIX#<PIX>#"; }
echo "=== REAL (pod layout, all frames)            expect: 61/61, TRAIN_DONE"; run --images "$PIX"
echo "=== M1a one frame missing, default 0.1 %     expect: REFUSED"; run --images "$PIX_M1"
echo "=== M1b one frame missing, limit 5 %         expect: DROPPED 1 (named), TRAIN_DONE"; run --images "$PIX_M1" --max-missing-images 0.05
echo "=== M2  a whole camera missing               expect: REFUSED, 0/61"; run --images "$PIX_M2"
