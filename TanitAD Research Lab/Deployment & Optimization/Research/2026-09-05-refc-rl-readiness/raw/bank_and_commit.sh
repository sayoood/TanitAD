#!/usr/bin/env bash
# bank_and_commit.sh — the whole banking sequence, re-runnable, for a flapping mount:
#   wait for the mount → md5-verified copies (bank_to_g.sh) → register insert (idempotent)
#   → compile checks on G: → mm_commit.py (retried) → length-guarded blob verification.
# Prints VERIFY lines: OK / MISMATCH / INCONCLUSIVE per committed path.
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
    n=$((n+1)); [ $n -ge 30 ] && { echo "MOUNT DOWN (git+read) after ~3 min"; return 1; }; sleep 6
  done
  echo "mount up (polls=$n)"
}

wait_mount || exit 2
echo "=== 1. copies (md5-verified) ==="
bash "$WP/raw/bank_to_g.sh" || { echo "BANK FAILED — stop"; exit 3; }

echo "=== 2. register insert (re-reads immediately before editing; refuses duplicates) ==="
"$PY" "$WP/raw/insert_register_rows.py" "$R/Project Steering/GOALS_AND_CLAIMS.md" "$WP/raw/register_rows.md" \
  || echo "(register insert refused or failed — see above; continuing so code still lands)"

echo "=== 3. compile checks on the G: copies ==="
for f in stack/scripts/rl_refcv3_min.py stack/tanitad/rl/rewards.py taniteval/tools/refcv3_arm.py stack/tests/test_rl_rewards.py; do
  "$PY" -m py_compile "$R/$f" && echo "compiles $f" || { echo "DOES NOT COMPILE $f — stop"; exit 4; }
done
"$PY" -c "import json,sys; json.load(open(sys.argv[1],encoding='utf-8'))" "$R/$WPREL/raw/preflight.json" && echo "preflight.json parses"

echo "=== 4. mm_commit (retried on transient failures) ==="
PATHS=( "stack/scripts/rl_refcv3_min.py" "stack/tanitad/rl/rewards.py" "stack/tests/test_rl_rewards.py"
        "taniteval/tools/refcv3_arm.py" "Project Steering/GOALS_AND_CLAIMS.md"
        "$WPREL/RESULT.md" "$WPREL/SPEC.md" "$WPREL/launch_refcv3_rl_min.sh"
        "$LIBREL/RESULT_D_SAFE_CAL_2.md" "$LIBREL/code/dsafe2_analyze.py" "$LIBREL/raw/dsafe_cal_2/dsafe2_result.json" )
for f in "$WP"/raw/*; do PATHS+=( "$WPREL/raw/$(basename "$f")" ); done
ok=0
for attempt in 1 2 3; do
  wait_mount || continue
  if "$PY" "$R/stack/scripts/mm_commit.py" "$WP/raw/commit_msg.txt" "${PATHS[@]}"; then ok=1; break; fi
  echo "mm_commit attempt $attempt failed; retrying in 20 s"; sleep 20
done
[ $ok = 1 ] || { echo "MM_COMMIT FAILED after 3 attempts"; exit 5; }

echo "=== 5. length-guarded verification (HEAD blob vs worktree blob, both must be 40 chars) ==="
wait_mount
head=$(git -C "$R" rev-parse HEAD 2>/dev/null); echo "HEAD=$head"
git -C "$R" log -1 --format='%h %s' 2>/dev/null | cut -c1-140
bad=0
for p in "${PATHS[@]}"; do
  a=$(git -C "$R" rev-parse "HEAD:$p" 2>/dev/null); b=$(git -C "$R" hash-object "$R/$p" 2>/dev/null)
  if [ ${#a} -ne 40 ] || [ ${#b} -ne 40 ]; then echo "VERIFY INCONCLUSIVE $p (a=${#a} b=${#b} chars)"; bad=$((bad+1))
  elif [ "$a" = "$b" ]; then echo "VERIFY OK   $p"
  else echo "VERIFY MISMATCH $p ($a vs $b)"; bad=$((bad+1)); fi
done
echo "VERIFY_DONE bad=$bad of ${#PATHS[@]}"
exit $bad
