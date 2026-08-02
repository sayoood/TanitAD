#!/bin/bash
# Self-retrying range-stream fill of the comma train cache (pod3).
# Resumes from a proportional tar offset until all 400 episodes exist.
URL=https://huggingface.co/datasets/Sayood/tanitad-comma2k19-episodes/resolve/main/comma_train.tar
SIZE=72501070656
DIR=/workspace/data/comma2k19-train-b40a21eb5216
for i in $(seq 1 20); do
  N=$(ls "$DIR" | wc -l)
  if [ "$N" -ge 400 ]; then echo "FILL_COMPLETE $N"; exit 0; fi
  OFF=$(( SIZE * N / 400 - 2147483648 ))
  [ "$OFF" -lt 0 ] && OFF=0
  echo "[fill] attempt $i: $N files, streaming from byte $OFF"
  curl -sL --retry 3 -r "$OFF"- "$URL" | tar -x --ignore-zeros --skip-old-files -f - -C /workspace/data
  echo "[fill] attempt $i ended (tar exit $?), have $(ls "$DIR" | wc -l)"
  sleep 10
done
echo "FILL_GAVE_UP after 20 attempts: $(ls "$DIR" | wc -l) files"
