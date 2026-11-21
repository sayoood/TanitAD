#!/bin/bash
# Rebuild the v2ep of the two train clips (+ the eval clip dca157a0) from the LOCAL mp4s into $WORK, by build_v2ep.py (read, not edited) of the identity control:
#   --frame 256 : native 256x640 cylinder  (= pod_build_b1_epcache.build_one's call; the PRODUCTION route)
#   --frame 416 : 416x1024 cylinder -> resize416_to_256 (= the dev-box 'thor416' TEST route)
# WORK is a SHORT path: the export path + work/build_root/r0/camera_front_wide exceeds Windows MAX_PATH (260).
# R: is `subst R: <this directory>` (removed at the end of the session), so the files still live in this directory only.
set -e
HERE="$(cygpath -m "$(cd "$(dirname "$0")" && pwd)")"
WORK="${WORK:-R:/work}"
EXPORT="$(cygpath -m "$(cd "$(dirname "$0")/../../../../../.." && pwd)")"
PK="$EXPORT/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-lora"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="$EXPORT/stack;$EXPORT/taniteval" PYTHONIOENCODING=utf-8
CLIPS="8f387977-4086-466d-bba4-b04f8cdc0743 bf5ec76e-e8b5-4f0d-a9cc-4fe1687d3d3a dca157a0-20a9-464b-a74c-084b678e7be1"
$PY "$PK/build_v2ep.py" --out "$WORK/v2ep_256" --root "$WORK/build_root" --frame 256 --clips $CLIPS
$PY "$PK/build_v2ep.py" --out "$WORK/v2ep_416" --root "$WORK/build_root" --frame 416 --clips $CLIPS
echo BUILD-DONE
