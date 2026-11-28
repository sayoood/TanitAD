#!/usr/bin/env bash
# pod_telemetry_push.sh — drain a training pod's telemetry to a GitHub branch.
#
# WHY THIS EXISTS. A cloud session's egress policy is HTTPS-CONNECT to an
# allow-list. MEASURED 2026-08-04 from a Claude Code cloud session:
#
#   api.runpod.io:443 / rest.runpod.io / api.runpod.ai  -> 403 (policy denial,
#                                        recorded by the gateway itself)
#   huggingface.co:443 / api.wandb.ai:443               -> 403
#   github.com:443 / raw.githubusercontent.com:443      -> REACHABLE
#   github.com:22 (a host that certainly serves an SSH
#     banner instantly)                                 -> CONNECT answered
#                                        200 locally, then NOTHING. Non-443 TCP
#                                        is silently dropped.
#
# ⚠️ That last line is the one that matters, and it corrects the tempting
# reading. The pods' SSH ports are not individually "denied" — ALL non-443
# egress is dropped, GitHub's own port 22 included. So allow-listing the pod
# IPs would NOT buy a shell, and no SSH key belongs in an environment variable
# to chase one (env vars are readable by anyone using the environment).
#
# ⇒ Do not pull from outside. PUSH from the pod, over the one channel that is
# already open. This script needs no egress-policy change whatsoever.
#
# WHAT IT DOES. Every INTERVAL seconds it writes a snapshot to an ORPHAN branch
# `telemetry/<pod>` and pushes. Orphan = it shares no history with main, so a
# telemetry commit can never touch program code, and the branch can be deleted
# without trace.
#
# USAGE (run once per pod; RunPod's browser terminal is enough, no SSH needed):
#
#   export GITHUB_TOKEN=<fine-grained PAT, contents:write on THIS repo only>
#   export POD_NAME=pod4                      # or tanitad-new
#   export TRAIN_LOG=/workspace/experiments/<run>/train.log
#   export RUN_DIR=/workspace/experiments/<run>
#   nohup nice -n 19 bash stack/ops/pod_telemetry_push.sh >/tmp/telemetry.log 2>&1 &
#
# ⚠️ Use a FINE-GRAINED token scoped to contents:write on this repo. A classic
# `repo`-scope PAT on a shared pod is far more access than a log drain needs.
#
# ⛔ NEVER ADD LOAD TO A TRAINING POD (CLAUDE.md invariant). Everything here is
# `nice -n 19`, reads files, and runs `nvidia-smi --query` (a driver query, not
# a GPU kernel). It imports no torch and allocates no GPU memory.
set -uo pipefail

INTERVAL="${INTERVAL:-300}"
POD_NAME="${POD_NAME:-$(hostname)}"
BRANCH="telemetry/${POD_NAME}"
RUN_DIR="${RUN_DIR:-}"
TRAIN_LOG="${TRAIN_LOG:-}"
TAIL_N="${TAIL_N:-80}"
WORK="${WORK:-/tmp/tanitad-telemetry-${POD_NAME}}"
REPO_SLUG="${REPO_SLUG:-sayoood/TanitAD}"

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "FATAL: GITHUB_TOKEN unset — nothing to push with." >&2
  exit 2
fi

# The token lives in the remote URL only inside this private work clone, never
# in the repo and never echoed. `set -x` is deliberately NOT used anywhere here.
REMOTE="https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO_SLUG}.git"

snapshot() {
  local out="$1"
  {
    echo "# telemetry — ${POD_NAME}"
    echo
    echo "generated_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host: $(hostname)"
    echo "uptime: $(uptime -p 2>/dev/null || true)"
    echo

    echo "## GPU"
    echo '```'
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
               --format=csv 2>&1 || echo "nvidia-smi unavailable"
    echo '```'
    echo

    # ⚠️ CLAUDE.md trap: `memory.usage_in_bytes` COUNTS RECLAIMABLE PAGE CACHE
    # and reads 74-100 % on an IDLE box. `rss` and `failcnt` are the admissible
    # numbers — a cgroup that never hit its limit reports failcnt 0, and that
    # is the fact that settles an "OOM" question. Reporting usage_in_bytes here
    # would re-manufacture the false OOM that cost ~40 min of v5f training.
    echo "## Memory (cgroup — rss and failcnt ONLY, never usage_in_bytes)"
    echo '```'
    for base in /sys/fs/cgroup/memory /sys/fs/cgroup; do
      if [ -r "$base/memory.stat" ]; then
        grep -E '^(rss|cache|total_rss|anon) ' "$base/memory.stat" 2>/dev/null | head -6
        echo "failcnt: $(cat "$base/memory.failcnt" 2>/dev/null || echo n/a)"
        cat "$base/memory.events" 2>/dev/null | head -5
        break
      fi
    done
    echo '```'
    echo

    # ⚠️ CLAUDE.md trap: NEVER judge pod disk with `df` — it reports the 965 TB
    # cluster and hides the per-pod MooseFS quota. A real write test is the only
    # probe that answers the question. 16 MB, once per interval, is negligible.
    echo "## Disk (real write test — \`df\` reports the cluster, not the quota)"
    echo '```'
    if dd if=/dev/zero of="${WORK}/.ddtest" bs=1M count=16 oflag=direct 2>&1 | tail -1; then :; else
      dd if=/dev/zero of="${WORK}/.ddtest" bs=1M count=16 2>&1 | tail -1
    fi
    rm -f "${WORK}/.ddtest"
    echo '```'
    echo

    # ⚠️ CLAUDE.md trap: `pgrep -f <trainer>` SELF-MATCHES the invoking command.
    # `pgrep -a` here is read-only (we never kill from this script — killing is
    # by explicit PID, by a human), but the self-match would still publish a
    # misleading line, so the grep excludes this script by name.
    echo "## Trainer processes"
    echo '```'
    ps -eo pid,etime,rss,stat,cmd 2>/dev/null \
      | grep -E 'train_|supervise_run' \
      | grep -v 'pod_telemetry_push' \
      | grep -v ' grep ' | head -12 || echo "(none matched)"
    echo '```'
    echo

    if [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ]; then
      echo "## Run dir: ${RUN_DIR}"
      echo '```'
      ls -la "$RUN_DIR" 2>&1 | head -25
      echo '```'
      for f in metrics.json config.json; do
        if [ -r "${RUN_DIR}/${f}" ]; then
          echo
          echo "### ${f}"
          echo '```json'
          head -c 4000 "${RUN_DIR}/${f}"
          echo
          echo '```'
        fi
      done
      echo
      echo "### latest checkpoints"
      echo '```'
      ls -lat "${RUN_DIR}"/*.pt 2>/dev/null | head -5 || echo "(no .pt yet)"
      echo '```'
      echo
    fi

    if [ -n "$TRAIN_LOG" ] && [ -r "$TRAIN_LOG" ]; then
      echo "## train.log — last ${TAIL_N} lines"
      echo
      # ⚠️ CLAUDE.md trap: `step_s` in trainer logs is ACCUMULATED over
      # --log-every (divide by 50), NOT per-step. Reading it raw has produced
      # false "training is 430 s/step" alarms. Stated here so whoever reads the
      # drain reads it correctly.
      echo "> \`step_s\` is ACCUMULATED over \`--log-every\` — divide by 50 for per-step."
      echo
      echo '```'
      tail -n "$TAIL_N" "$TRAIN_LOG" 2>&1
      echo '```'
    else
      echo "## train.log"
      echo "(TRAIN_LOG unset or unreadable: '${TRAIN_LOG}')"
    fi
  } > "$out" 2>&1
}

# --- one-time: a private work clone holding ONLY the orphan branch --------- #
rm -rf "$WORK"
mkdir -p "$WORK"
cd "$WORK" || exit 2
git init -q .
git config user.email "telemetry@tanitad.local"
git config user.name  "tanitad-pod-telemetry"
git remote add origin "$REMOTE"
git checkout -q --orphan "$BRANCH" 2>/dev/null || git checkout -q -b "$BRANCH"
# Adopt the branch if it already exists, so history accumulates across restarts
# instead of a force-push throwing away the record.
if git fetch -q origin "$BRANCH" 2>/dev/null; then
  git reset -q --hard FETCH_HEAD
fi

echo "[telemetry] pod=${POD_NAME} branch=${BRANCH} interval=${INTERVAL}s"

while true; do
  snapshot "${WORK}/STATUS.md"
  cp "${WORK}/STATUS.md" "${WORK}/history-$(date -u +%Y%m%dT%H%M%SZ).md" 2>/dev/null || true
  # keep the branch small: only the newest 48 history files (~4 h at 5 min)
  ls -1t "${WORK}"/history-*.md 2>/dev/null | tail -n +49 | xargs -r rm -f

  git add -A -- STATUS.md 'history-*.md' 2>/dev/null || true
  # ⚠️ CLAUDE.md: `git add` exit codes are NOT evidence. Verify the index.
  if [ -z "$(git ls-files --cached)" ]; then
    echo "[telemetry] WARN nothing staged — skipping this tick" >&2
  elif git diff --cached --quiet; then
    : # no change since last push
  else
    git commit -q -m "telemetry ${POD_NAME} $(date -u +%Y-%m-%dT%H:%M:%SZ)" || true
    for i in 1 2 3 4; do
      if git push -q origin "HEAD:${BRANCH}" 2>/dev/null; then break; fi
      echo "[telemetry] push retry ${i}" >&2
      sleep $((2 ** i))
    done
  fi
  sleep "$INTERVAL"
done
