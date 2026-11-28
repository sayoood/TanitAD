#!/usr/bin/env bash
# commit_retry_A.sh — the register commit (B) landed; this retries ONLY commit A
# (code + WP + d_safe-2), re-copying the three WP files the mount refused last pass.
set -u
R="/g/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
WP="/c/Users/Admin/rl_readiness_wp"
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
WPREL="TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-refc-rl-readiness"
LIBREL="TanitAD Research Lab/Architecture & Inference/Research/2026-08-29-rl-posttrain-library"
wait_mount () {
  local n=0
  until timeout 15 head -c 64 "$R/CLAUDE.md" >/dev/null 2>&1 \
        && timeout 30 git -C "$R" rev-parse --abbrev-ref HEAD >/dev/null 2>&1; do
    n=$((n+1)); [ $n -ge 30 ] && { echo "MOUNT DOWN after ~3 min"; return 1; }; sleep 6
  done
}
put () {
  local src="$1" dst="$2" i want got
  want=$(md5sum "$src" | cut -d' ' -f1)
  for i in 1 2 3 4 5 6 7 8; do
    mkdir -p "$(dirname "$dst")" 2>/dev/null; cp -f "$src" "$dst" 2>/dev/null
    got=$(md5sum "$dst" 2>/dev/null | cut -d' ' -f1)
    [ "$got" = "$want" ] && { echo "OK   $(basename "$dst")"; return 0; }
    sleep 8
  done
  echo "FAIL $dst"; return 1
}
hashable () { local p h; for p in "$@"; do h=$(timeout 60 git -C "$R" hash-object "$R/$p" 2>/dev/null); [ ${#h} -eq 40 ] || { echo "  not hashable: $p"; return 1; }; done; }

PATHS_A=( "stack/scripts/rl_refcv3_min.py" "stack/tanitad/rl/rewards.py" "stack/tests/test_rl_rewards.py"
          "taniteval/tools/refcv3_arm.py"
          "$WPREL/RESULT.md" "$WPREL/SPEC.md" "$WPREL/launch_refcv3_rl_min.sh"
          "$LIBREL/RESULT_D_SAFE_CAL_2.md" "$LIBREL/code/dsafe2_analyze.py" "$LIBREL/raw/dsafe_cal_2/dsafe2_result.json" )
for f in "$WP"/raw/*; do PATHS_A+=( "$WPREL/raw/$(basename "$f")" ); done

wait_mount || exit 2
echo "=== re-copy the three WP files (+ this script into raw/) ==="
put "$WP/RESULT.md" "$R/$WPREL/RESULT.md"; put "$WP/SPEC.md" "$R/$WPREL/SPEC.md"
put "$WP/launch_refcv3_rl_min.sh" "$R/$WPREL/launch_refcv3_rl_min.sh"
put "$WP/raw/commit_retry_A.sh" "$R/$WPREL/raw/commit_retry_A.sh"
put "$WP/raw/commit_msg_register.txt" "$R/$WPREL/raw/commit_msg_register.txt"
echo "=== commit A (retried) ==="
ok=0
for attempt in 1 2 3 4 5 6 7 8; do
  wait_mount || continue
  if ! hashable "${PATHS_A[@]}"; then echo "[A] attempt $attempt: waiting 30 s"; sleep 30; continue; fi
  if "$PY" "$R/stack/scripts/mm_commit.py" "$WP/raw/commit_msg.txt" "${PATHS_A[@]}"; then ok=1; echo "[A] COMMITTED"; break; fi
  echo "[A] attempt $attempt failed; retrying in 30 s"; sleep 30
done
echo "=== VERIFY ==="
wait_mount
echo "HEAD=$(git -C "$R" rev-parse HEAD 2>/dev/null)"; git -C "$R" log -3 --format='%h %s' 2>/dev/null | cut -c1-140
bad=0
for p in "${PATHS_A[@]}" "Project Steering/GOALS_AND_CLAIMS.md"; do
  a=$(timeout 60 git -C "$R" rev-parse --verify -q "HEAD:$p" 2>/dev/null); b=$(timeout 60 git -C "$R" hash-object "$R/$p" 2>/dev/null)
  if [ ${#a} -ne 40 ] || [ ${#b} -ne 40 ]; then echo "VERIFY INCONCLUSIVE $p (a=${#a} b=${#b})"; bad=$((bad+1))
  elif [ "$a" = "$b" ]; then echo "VERIFY OK   $p"
  else echo "VERIFY MISMATCH $p"; bad=$((bad+1)); fi
done
echo "VERIFY_DONE bad=$bad okA=$ok"
