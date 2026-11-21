#!/bin/bash
# usage: run_eval.sh <seed> <ages: cycle|fresh>
# The banked KIND own-code eval's argv (queue :kindown = `run --ckpt <ckpt> --tag <tag> %EC% %ER% --project --families`), VERBATIM except the paths (box 2 D:\tanitad -> OUT)
# and the ONE lever of PREREG_V8F_SFRESH: --ages. CPU only, torch capped at 8 threads.
set -u
O="C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/sfresh"
S=$1; AGES=$2
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
TAG=M2b_s${S}-sfresh-${AGES}
ARGV=(run --ckpt "$O/inputs/M2b_s${S}/ckpt_last.pt" --tag "$TAG" --ages "$AGES"
      --v9-eval "$O/inputs/v9/v9_labels_eval139.npz" --lead-eval "$O/inputs/v9/lead_eval139.npz" --speed-eval "$O/inputs/v9/refcv6_speed_max_v8_eval.jsonl"
      --kin-mode yaw --p10 "$O/inputs/p10/p10_prior.pt" --out-dir "$O/eval"
      --eval-cache "$O/cache" --tracks "$O/inputs/native_tracks.pt" --project --families)
printf '%s\n' "$PY" "$O/code/stack/scripts/v8f_mini_eval.py" "${ARGV[@]}" > "$O/eval/argv_${TAG}.txt"
echo "START $(date '+%H:%M:%S') $TAG"
"$PY" -I "$O/code/stack/scripts/v8f_mini_eval.py" "${ARGV[@]}" > "$O/eval/eval_${TAG}.log" 2>&1
echo "END $(date '+%H:%M:%S') $TAG rc=$?"
