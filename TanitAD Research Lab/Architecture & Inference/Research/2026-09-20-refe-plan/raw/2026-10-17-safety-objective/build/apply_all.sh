#!/bin/bash
# apply_all.sh <PK> : apply the SAFEOBJ build to a CLEAN checkout of PK (TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan at the tip daa4568, LF text).
# Every patch script uses EXACT anchors (each occurs exactly once; a miss is a hard error) and checks the base blob first, so a tree that is not the tip is refused, never half-patched.
# Order matters: trainer -> gate -> chain -> tests -> arms file -> the new test (generated from the trainer's own edit list) -> the S-reader (two new eval/ files) -> round 3 (the gate verifies the reader's record; edits the files above) -> round 4 (the gate re-reads the arm directory; the pod precheck lists the S-reader files; edits the files above again).
set -e
PK="$1"
[ -d "$PK/refe" ] || { echo "usage: apply_all.sh <PK dir>"; exit 2; }
B="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
cd "$B"
"$PYTHON" patch_trainer.py "$PK"
"$PYTHON" patch_gate.py "$PK"
"$PYTHON" patch_chain.py "$PK"
"$PYTHON" patch_tests.py "$PK"
"$PYTHON" make_arms_json.py "$PK"
"$PYTHON" gen_safeobj_test.py "$PK"
"$PYTHON" add_reader.py "$PK"            # eval/safeobj_read.py + eval/test_safeobj_read.py: the reader of bar S (new files; touches nothing else)
"$PYTHON" patch_genuine.py "$PK"         # round 3: safeobj_gate opens only on a GENUINE reader record (tool, record_sha256, input sha256s) + the tests that fed it hand-written S files + eval/test_safeobj_gate.py
"$PYTHON" patch_reread.py "$PK"         # round 4: the gate RE-READS the arm directory and requires the stored record to equal the fresh one; the pod precheck requires the S-reader and its two tests; the pod header says the run is TWO steps
"$PYTHON" verify_blobs.py "$PK"
