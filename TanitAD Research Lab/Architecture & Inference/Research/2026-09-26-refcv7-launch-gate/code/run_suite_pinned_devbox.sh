#!/usr/bin/env bash
# G-SUITE-PINNED on the dev box (Master Mind 2026-09-27): the pinned files on a FULL-commit archive,
# ONE compute process, started at >= 7.5 GB free on 3 samples, aborted below 4.0 GB.
#   TREE=<launched tree: commit + gate files> COMMIT=<sha> ARGV=<argv json> OUT=<gate dir> bash run_suite_pinned_devbox.sh
set -u
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' PYTHONIOENCODING=utf-8
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
: "${TREE:?}" "${COMMIT:?}" "${ARGV:?}" "${OUT:?}"
"$PY" "$TREE/stack/scripts/launch_gate.py" run --profile refcv7 --checks G-SUITE-PINNED \
  --tree "$TREE" --commit "$COMMIT" --argv-file "$ARGV" --out-dir "$OUT" \
  --git-dir C:/Users/Admin/tanitad-push/.git --suite-work C:/lgs \
  --cpu-only --omp 4 --min-free-gb 7.5 --abort-free-gb 4.0 --ram-wait-s 21600 \
  --key-file C:/Users/Admin/lg0926/keys/rehearsal.key > "$OUT.out" 2>&1
echo "ZZPINNED-DONE-$(date -u +%H%M%S)ZZ" >> "$OUT.out"
