#!/bin/bash
# make_deliverables.sh <PRISTINE PK> <PATCHED PK> <OUT package dir>
# Mirrors the SAFEOBJ build into the package dir: deliverables/<pkg-relative path> (the patched files, LF text), diffs/PATCH_NN_<file>.diff (unified, pristine -> patched, -p1 from the
# refe-plan directory), and the file/blob manifest.  Nothing is added to git; the Master Mind lands these.
set -e
BASE="$1"; NEW="$2"; OUT="$3"
[ -d "$BASE/refe" ] && [ -d "$NEW/refe" ] && [ -d "$OUT" ] || { echo "usage: make_deliverables.sh <pristine PK> <patched PK> <out dir>"; exit 2; }
mkdir -p "$OUT/deliverables/refe" "$OUT/deliverables/eval" "$OUT/diffs"
i=0
for f in refe/scorer_finetune.py refe/launch_gate.py refe/gate_checks.py refe/rls1_checks.py refe/rls1_pod.sh \
         refe/test_rl_s1b.py refe/test_rl_r0c.py refe/test_rl_epo.py refe/test_launch_gate.py refe/test_rls1_chain.py refe/test_rls1_queue.py refe/test_dump_scores.py refe/test_aux_s1_wiring.py; do
  i=$((i + 1)); n=$(printf '%02d' "$i")
  cp "$NEW/$f" "$OUT/deliverables/$f"
  diff -u --label "a/$f" --label "b/$f" "$BASE/$f" "$NEW/$f" > "$OUT/diffs/PATCH_${n}_$(basename "$f").diff" || true
done
# new files (no pristine counterpart)
for f in refe/test_rl_safeobj.py eval/rls1d_arms.json eval/safeobj_read.py eval/test_safeobj_read.py eval/test_safeobj_gate.py; do
  cp "$NEW/$f" "$OUT/deliverables/$f"
  diff -u --label "/dev/null" --label "b/$f" /dev/null "$NEW/$f" > "$OUT/diffs/NEW_$(basename "$f").diff" || true
done
( cd "$OUT" && find deliverables -type f ! -name BLOBS.txt | sort | while read -r p; do printf '%s  %s  %s\n' "$(git hash-object --no-filters "$p" 2> /dev/null || echo NA)" "$(sha256sum "$p" | cut -c1-16)" "$p"; done ) > "$OUT/deliverables/BLOBS.txt"
echo "wrote $(find "$OUT/deliverables" -type f | wc -l) files and $(ls "$OUT/diffs" | wc -l) diffs under $OUT"
