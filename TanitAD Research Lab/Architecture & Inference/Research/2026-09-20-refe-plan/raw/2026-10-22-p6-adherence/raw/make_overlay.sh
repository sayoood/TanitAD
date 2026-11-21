#!/usr/bin/env bash
# Rebuilds work/overlay = the mirror tip a9bc000 (unchanged for these paths through a8f2c89) restricted to what the instrument and its tests import:
#   taniteval/taniteval/*.py (top level) + taniteval/adapters, stack/tanitad, PK/refe, PK/eval (non-raw .py/.json)
# then `bash raw/sync_overlay.sh` puts the package's two NEW files on top (exactly the MM's "overlay of the mirror tip, with the package's files on top").
set -e
M=C:/Users/Admin/tanitad-push
OV=C:/Users/Admin/par18/p6adh/work/overlay
TIP=a9bc000
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
mkdir -p "$OV"; cd "$M"
git archive $TIP taniteval/adapters stack/tanitad | tar -x -C "$OV"
git ls-tree --name-only $TIP taniteval/taniteval/ | grep '\.py$' | tr '\n' '\0' | xargs -0 git archive $TIP | tar -x -C "$OV"
git archive $TIP "$PK/refe" | tar -x -C "$OV"
git ls-tree -r --name-only $TIP -- "$PK/eval" | grep -v '/raw/' | grep -E '\.(py|json)$' | tr '\n' '\0' | xargs -0 git archive $TIP | tar -x -C "$OV"
echo "overlay built at $TIP"
