#!/usr/bin/env bash
# pod_boxfuse10k_det.sh -- card PREREG_V8F_PROOF_ADD131, section 2: the detector pass over EVERY cache frame of the 141 evaluation clips, wide window.
# The detector tool of card ADD127 (box_detect.py) is called unchanged, WITHOUT --only-lead-frames and WITHOUT --box-dir: the pass reads no label.
# Writes the call into DET_CALL.txt (control j0), sets the shared state file to BUSY for the pass and to IDLE after it, and writes BOXFUSE10K_DET_DONE.txt
# or BOXFUSE10K_DET_STOP.txt. It deletes, moves and stops nothing. One GPU job. Refuses when the out folder exists.
set -u
ROOT=/workspace/v8proof/boxfuse10k
OUT=$ROOT/det_out
PY=/workspace/venv/bin/python
DET=/workspace/v8proof/boxread/kit/det/box_detect.py
WEIGHTS=/workspace/v8proof/boxread/torch_home/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
LIST_EVAL=/workspace/v8proof/cropread/run1/lists/eval141.txt
STAGE=/workspace/v8proof/cropread/stage
STATE=/workspace/FORK_STATE.txt
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
[ -e "$OUT" ] && { echo "REFUSED: $OUT exists"; exit 2; }
mm=$(tr -d ' \r\n' < /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 ))
[ "$free_gb" -ge 28 ] || { echo "REFUSED: memory free ${free_gb} GB"; exit 2; }
ngpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
[ "$ngpu" -le 2 ] || { echo "REFUSED: $ngpu GPU jobs on the pod"; exit 2; }
for f in "$DET" "$WEIGHTS" "$LIST_EVAL"; do [ -f "$f" ] || { echo "REFUSED: missing $f"; exit 2; }; done
mkdir -p "$OUT" || exit 2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PAI_DECODE_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TANITAD_STACK=/workspace/v8proof/tree/stack PYTHONPATH=/workspace/v8proof/tree/stack
CALL=("$PY" "$DET" --camera wide --list "$LIST_EVAL" --root "$STAGE" --weights "$WEIGHTS" --weights-sha-prefix dd69338a --out-dir "$OUT"
      --win-size-deg 32 --win-px 768 --az-left-deg=0 --el-up-deg=-6 --score-keep 0.3 --batch 8 --device cuda --workers 2)
printf '%s\n' "${CALL[*]}" > "$OUT/DET_CALL.txt"
sha256sum "$DET" "$WEIGHTS" "$LIST_EVAL" > "$OUT/DET_INPUT_SHA.txt"
echo "$(date -u +%s) $(ts) pid $$" > "$ROOT/BOXFUSE10K_DET_START.txt"
echo "BUSY card ADD131 detector pass of the v8proof fork $(ts)" >> "$STATE"
"${CALL[@]}" > "$OUT/det.log" 2>&1; rc=$?
echo "IDLE $(ts)" >> "$STATE"
if [ "$rc" -eq 0 ]; then echo "ZZ BOXFUSE10K-DET DONE rc 0 $(ts)" > "$ROOT/BOXFUSE10K_DET_DONE.txt"; else echo "ZZ BOXFUSE10K-DET STOP rc $rc $(ts)" > "$ROOT/BOXFUSE10K_DET_STOP.txt"; fi
echo "ZZ BOXFUSE10K-DET END rc $rc $(ts)"
