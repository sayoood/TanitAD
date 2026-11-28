#!/usr/bin/env bash
# push_q.sh <file> [<file> ...] : pushes files of imagplan_q/ into the pod folder /workspace/v8proof/imagplan_q/ (and nothing else), sha256-compared on both sides. A file that is on the pod with the same sha256 is not written again; another sha256 is overwritten only with PUSH_OVERWRITE=1. Deletes nothing.
P=/c/Users/Admin/v8pod/pssh.sh
H=/f/TanitAD-scratch/v8proof/imagplan_q; D=/workspace/v8proof/imagplan_q
timeout 60 bash "$P" "mkdir -p $D" 2>/dev/null | sed '/hf_/d'
for f in "$@"; do
  b=$(sha256sum "$H/$f" | cut -d' ' -f1)
  a=$(timeout 60 bash "$P" "[ -f $D/$f ] && sha256sum $D/$f | cut -d' ' -f1" 2>/dev/null | sed '/hf_/d' | tr -cd 'a-f0-9')
  if [ "$a" = "$b" ]; then echo "same $f ${b:0:12}"; continue; fi
  if [ -n "$a" ] && [ "${PUSH_OVERWRITE:-0}" != 1 ]; then echo "PUSH-STOP $f is on the pod with another sha256"; exit 3; fi
  timeout 180 bash "$P" "cat > $D/$f" < "$H/$f" 2>/dev/null | sed '/hf_/d'
  a=$(timeout 60 bash "$P" "sha256sum $D/$f | cut -d' ' -f1" 2>/dev/null | sed '/hf_/d' | tr -cd 'a-f0-9')
  [ "$a" = "$b" ] && echo "pushed $f ${b:0:12}" || { echo "PUSH-MISMATCH $f"; exit 3; }
done
