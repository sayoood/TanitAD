#!/usr/bin/env bash
# Builds the overlay tree run_speedsub.sh needs: a COPY of SPEEDREAD's read-only run tree (git archive of f3971165: stack/, taniteval/, the eval inputs) with the two staged files laid over it.
# Never writes under the source tree. Usage: bash run/make_tree.sh [DST]   (default F:/TanitAD-scratch/speedsub_tmp/tree)
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC=F:/TanitAD-scratch/speedread/tree_f3971165; DST=${1:-F:/TanitAD-scratch/speedsub_tmp/tree}
[ -d "$SRC" ] || { echo "REFUSED: $SRC missing"; exit 3; }
[ -e "$DST" ] && { echo "REFUSED: $DST exists"; exit 3; }
cp -r "$SRC" "$DST"
cp "$HERE/stage/stack/scripts/v8f_mini_eval_speedread.py" "$DST/stack/scripts/v8f_mini_eval_speedread.py"
cp "$HERE/families/families_longlat_sub.py" "$DST/stack/scripts/families_longlat_sub.py"   # MM 2026-10-28: a run-time script, banked in the package as SPEEDREAD's families_longlat.py is (it parses argv at import, so it is not a stack/ module)
sha256sum "$DST/stack/scripts/v8f_mini_eval_speedread.py" "$DST/stack/scripts/families_longlat_sub.py"
