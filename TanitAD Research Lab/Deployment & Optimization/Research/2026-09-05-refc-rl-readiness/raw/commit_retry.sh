#!/usr/bin/env bash
# commit_retry.sh — the register file may be unreadable through git while everything
# else hashes. Commit the code + WP first (unblocked), then the register alone, each
# retried until the mount lets git hash the paths. Ends with the VERIFY table.
set -u
R="/g/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
WP="/c/Users/Admin/rl_readiness_wp"
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
WPREL="TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-refc-rl-readiness"
LIBREL="TanitAD Research Lab/Architecture & Inference/Research/2026-08-29-rl-posttrain-library"
REG="Project Steering/GOALS_AND_CLAIMS.md"

wait_mount () {
  local n=0
  until timeout 15 head -c 64 "$R/CLAUDE.md" >/dev/null 2>&1 \
        && timeout 30 git -C "$R" rev-parse --abbrev-ref HEAD >/dev/null 2>&1; do
    n=$((n+1)); [ $n -ge 30 ] && { echo "MOUNT DOWN after ~3 min"; return 1; }; sleep 6
  done
}
hashable () {  # every named path must hash to 40 chars
  local p h
  for p in "$@"; do h=$(timeout 60 git -C "$R" hash-object "$R/$p" 2>/dev/null); [ ${#h} -eq 40 ] || return 1; done
  return 0
}
PATHS_A=( "stack/scripts/rl_refcv3_min.py" "stack/tanitad/rl/rewards.py" "stack/tests/test_rl_rewards.py"
          "taniteval/tools/refcv3_arm.py"
          "$WPREL/RESULT.md" "$WPREL/SPEC.md" "$WPREL/launch_refcv3_rl_min.sh"
          "$LIBREL/RESULT_D_SAFE_CAL_2.md" "$LIBREL/code/dsafe2_analyze.py" "$LIBREL/raw/dsafe_cal_2/dsafe2_result.json" )
for f in "$WP"/raw/*; do PATHS_A+=( "$WPREL/raw/$(basename "$f")" ); done

# second message for the register-only commit
MSG_B="$WP/raw/commit_msg_register.txt"
cat > "$MSG_B" <<'EOF'
Register D-RL-READY-1, D-RL-REWARD-FLOOR-1, H-RL-MIN-1 (REF-C RL readiness, resumed 2026-09-05)

Companion to the RL-readiness commit (stack/scripts/rl_refcv3_min.py, tanitad/rl/rewards.py,
taniteval/tools/refcv3_arm.py, the 2026-09-05-refc-rl-readiness WP). Three rows INSERTED before
`## D-REFCV4B-EGODROP2`; nothing rewritten. The worktree register may carry other agents'
uncommitted rows — taken as-is, a recorded sweep.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF

commit_with_retry () {  # commit_with_retry <msgfile> <label> <paths...>
  local msg="$1" label="$2"; shift 2
  local attempt
  for attempt in 1 2 3 4 5 6; do
    wait_mount || continue
    if ! hashable "$@"; then echo "[$label] attempt $attempt: a path does not hash yet; waiting 30 s"; sleep 30; continue; fi
    if "$PY" "$R/stack/scripts/mm_commit.py" "$msg" "$@"; then echo "[$label] COMMITTED"; return 0; fi
    echo "[$label] attempt $attempt failed; retrying in 30 s"; sleep 30
  done
  echo "[$label] FAILED after 6 attempts"; return 1
}

echo "=== A. code + WP ==="
commit_with_retry "$WP/raw/commit_msg.txt" "A" "${PATHS_A[@]}"; rcA=$?
echo "=== B. the register ==="
commit_with_retry "$MSG_B" "B" "$REG"; rcB=$?

echo "=== VERIFY (HEAD blob vs worktree blob; both 40 chars or INCONCLUSIVE) ==="
wait_mount
echo "HEAD=$(git -C "$R" rev-parse HEAD 2>/dev/null)"; git -C "$R" log -2 --format='%h %s' 2>/dev/null | cut -c1-150
bad=0
for p in "${PATHS_A[@]}" "$REG"; do
  a=$(timeout 60 git -C "$R" rev-parse "HEAD:$p" 2>/dev/null); b=$(timeout 60 git -C "$R" hash-object "$R/$p" 2>/dev/null)
  if [ ${#a} -ne 40 ] || [ ${#b} -ne 40 ]; then echo "VERIFY INCONCLUSIVE $p (a=${#a} b=${#b})"; bad=$((bad+1))
  elif [ "$a" = "$b" ]; then echo "VERIFY OK   $p"
  else echo "VERIFY MISMATCH $p"; bad=$((bad+1)); fi
done
echo "VERIFY_DONE bad=$bad rcA=$rcA rcB=$rcB"
