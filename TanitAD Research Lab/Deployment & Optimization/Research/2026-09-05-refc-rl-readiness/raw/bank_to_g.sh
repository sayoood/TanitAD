#!/usr/bin/env bash
# bank_to_g.sh — copy the WP deliverables + code edits + the stranded d_safe-2 artifacts
# onto the G: worktree, verifying EVERY copy by md5 (the mount flaps and silently drops
# writes; three earlier copies "succeeded" and did not land). Retries each file 3x.
# Usage: bash bank_to_g.sh            (prints one line per file: OK / FAIL)
set -u
R="/g/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
WP="/c/Users/Admin/rl_readiness_wp"
CL="/c/Users/Admin/refcv4b_repo"
S="/c/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8e7cfa33-c625-47cf-88aa-711db80ac113/scratchpad"
DST="$R/TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-refc-rl-readiness"
LIB="$R/TanitAD Research Lab/Architecture & Inference/Research/2026-08-29-rl-posttrain-library"
fails=0
put () {  # put <src> <dst>
  local src="$1" dst="$2" i want got
  want=$(md5sum "$src" | cut -d' ' -f1)
  for i in 1 2 3; do
    mkdir -p "$(dirname "$dst")" 2>/dev/null
    cp -f "$src" "$dst" 2>/dev/null
    got=$(md5sum "$dst" 2>/dev/null | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then echo "OK   $dst"; return 0; fi
    sleep 3
  done
  echo "FAIL $dst (want $want got ${got:-none})"; fails=$((fails+1)); return 1
}
# --- the WP ---------------------------------------------------------------------------
put "$WP/RESULT.md" "$DST/RESULT.md"
put "$WP/SPEC.md" "$DST/SPEC.md"
put "$WP/launch_refcv3_rl_min.sh" "$DST/launch_refcv3_rl_min.sh"
for f in "$WP"/raw/*; do put "$f" "$DST/raw/$(basename "$f")"; done
# --- code (repo paths) ------------------------------------------------------------------
put "$CL/stack/scripts/rl_refcv3_min.py" "$R/stack/scripts/rl_refcv3_min.py"
put "$CL/stack/tanitad/rl/rewards.py" "$R/stack/tanitad/rl/rewards.py"
put "$CL/stack/tests/test_rl_rewards.py" "$R/stack/tests/test_rl_rewards.py"
# refcv3_arm.py on G: is NEWER than the clone (another agent moved it): re-apply the
# anchored patch onto a LOCAL copy of the G: file, then put that (never the clone copy).
TMP=$(mktemp -d)
if cp "$R/taniteval/tools/refcv3_arm.py" "$TMP/refcv3_arm.py" \
   && /c/Users/Admin/venvs/tanitad/Scripts/python.exe "$WP/raw/patch_1a_harness_only.py" "$TMP/refcv3_arm.py"; then
  put "$TMP/refcv3_arm.py" "$R/taniteval/tools/refcv3_arm.py"
else
  echo "FAIL refcv3_arm.py patch-on-G:"; fails=$((fails+1))
fi
# --- the predecessor's stranded D-SAFE-CAL-2 readout ------------------------------------
put "$S/RESULT_D_SAFE_CAL_2.md" "$LIB/RESULT_D_SAFE_CAL_2.md"
put "$S/dsafe2_analyze.py" "$LIB/code/dsafe2_analyze.py"
put "$S/dsafe2_result.json" "$LIB/raw/dsafe_cal_2/dsafe2_result.json"
echo "BANK_DONE fails=$fails"
exit $fails
