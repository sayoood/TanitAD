#!/usr/bin/env bash
# refcv6 -- write the per-arm manifests, RUN THE GATE, and start one supervisor.
#
#   ./launch_refcv6.sh --arm D --w-agent <W> [--budget cut|full] [--dry-run]
#
# ⛔ THIS SCRIPT REFUSES TO LAUNCH UNTIL prelaunch_gate.py HAS WRITTEN
#    "verdict": "PASS" INTO ITS JSON. The gate's EXIT CODE is never consulted.
#
#    *"The admissible evidence that this gate did not run is the MISSING JSON,
#    never the exit code."* MEASURED 2026-09-07: a 25-minute `timeout` killed a
#    pre-launch gate with zero output and no json, and its wrapper printed
#    `GATE_EXIT=0`. Reading `$?` through `cmd | tail` reports TAIL's status.
#
# ⛔ ONE ARM PER INVOCATION. Chaining is the caller's job (see chain_refcv6.sh),
#    because a chain that launches arm 2 while arm 1 is still training is how a
#    pod ends up with two trainers on one card.

set -u

ARM=""; BUDGET="full"; DRY=0
W_AGENT=""; W_TAC_GOAL=""
ROOT="${REFCV6_ROOT:-/workspace/refcv6}"
STACK="${TANITAD_STACK:-/workspace/TanitAD/stack}"
TRAINER="${STACK}/scripts/refc_v3_train.py"
LABELS="${REFCV6_LABELS:-/workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz}"
JOIN="${REFCV6_JOIN:-/root/data/joins/train2400_agents.jsonl.xz}"
ANCHORS="${REFCV6_ANCHORS:-/root/data/refcv6/anchors.pt}"
HOST="${REFCV6_HOST:-}"
# ⭐ C3 (import closure) can only be measured FROM the box that holds the repo,
#    while C5 (the smoke) can only run ON the GPU box. On a two-box fleet the
#    closure audit is therefore run separately and its ARTIFACT handed to the
#    gate, which re-checks its host, its remote_root, its age and its
#    import_probe before honouring it. See prelaunch_gate._c3_from_artifact.
CLOSURE_JSON="${REFCV6_CLOSURE_JSON:-}"
REMOTE_ROOT="${REFCV6_REMOTE_ROOT:-}"
# When the gate runs ON the box the closure artifact describes, re-hash the
# closure files against the artifact's own remote_md5_lf. That asserts THE
# TREE IS UNCHANGED -- the actual claim -- instead of asking the clock, so a
# 47-hour arm cannot make the NEXT arm's C3 inconclusive by mere elapsed time.
CLOSURE_REVERIFY_ROOT="${REFCV6_CLOSURE_REVERIFY_ROOT:-}"
PY="${REFCV6_PYTHON:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --budget) BUDGET="$2"; shift 2 ;;
    --w-agent) W_AGENT="$2"; shift 2 ;;
    --w-tac-goal) W_TAC_GOAL="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

[ -n "$ARM" ] || { echo "usage: launch_refcv6.sh --arm <ARM> [...]"; exit 2; }

RUNS_DIR="${ROOT}/runs.d"
RAW_DIR="${ROOT}/raw"
mkdir -p "$RUNS_DIR" "$RAW_DIR"

case "$BUDGET" in
  full) STEPS=40284 ;;
  cut)  STEPS=12000 ;;
  *) echo "unknown budget: $BUDGET"; exit 2 ;;
esac

# ⛔ WAS HARDCODED `/workspace/experiments/...` -- a POD path on a machine that no
#    longer exists. Per-box, like every other path in this script.
OUT_ROOT="${REFCV6_OUT_ROOT:-/workspace/experiments}"
OUT_DIR="${OUT_ROOT}/refcv6-${ARM}"
GATE_JSON="${RAW_DIR}/prelaunch_gate_${ARM}.json"

# ⛔⛔ `PYTHONPATH=<tree>/stack` IS REQUIRED OR THE TRAINER DIES
#    `ModuleNotFoundError: No module named 'tanitad'`. It used to be set only as
#    a per-command prefix on the gate invocation below -- i.e. NOT exported, and
#    NOT inherited by the supervisor's trainer. The manifest is the run's record,
#    so the environment belongs IN `TRAIN_CMD`, not in whatever shell happened to
#    start the supervisor. `env` is used rather than an exported variable so the
#    manifest can be read months later and still say what the run actually ran.
# ⛔ OMP_NUM_THREADS=6: torch spawns ~113 threads per process; without this a
#    multi-process box sits at 0-6 % GPU for 50 minutes and looks like a hang.
TRAIN_ENV="${REFCV6_TRAIN_ENV:-env PYTHONPATH=${STACK} OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8}"

# ⚠️ `refc_v3_train.py` has NO `--resume`, so a relaunch RESTARTS the arm from
#    step 0. On a ~46 h arm that is not a recovery, it is a second full run.
#    Default 1 = the supervisor watches and writes the done-marker, but never
#    silently restarts. Override with REFCV6_MAX_RELAUNCH if you mean it.
MAX_RELAUNCH="${REFCV6_MAX_RELAUNCH:-1}"

# ---- 1. build this arm's argv from the SINGLE SOURCE OF TRUTH -------------- #
# ⛔ Never hand-written here. `arms.py` is the only place an arm's flags live, so
#    a launch script and the pre-registration table cannot drift apart.
EXTRA=""
[ -n "$W_AGENT" ]    && EXTRA="$EXTRA --w-agent $W_AGENT"
[ -n "$W_TAC_GOAL" ] && EXTRA="$EXTRA --w-tac-goal $W_TAC_GOAL"

# shellcheck disable=SC2086
ARGV="$(cd "$HERE" && PYTHONPATH="$STACK" "$PY" arms.py "$ARM" \
          --steps "$STEPS" --anchors "$ANCHORS" --agent-join "$JOIN" $EXTRA)"
ARGV_RC=$?
if [ "$ARGV_RC" -ne 0 ] || [ -z "$ARGV" ]; then
  echo "⛔ could not build argv for arm ${ARM} (rc=${ARGV_RC})."
  echo "   If this arm needs --w-agent or --w-tac-goal, arms.py refuses rather"
  echo "   than inventing a value. --w-tac-goal is a PI DECISION (queue item 10)."
  exit 3
fi

# ---- 2. the manifest ------------------------------------------------------- #
# ⚠️ The supervisor SOURCES THIS ONCE, at startup. Editing it under a live
#    supervisor changes nothing: it replays the TRAIN_CMD it captured when it
#    booted, and the relaunch looks successful while running the OLD config.
#    To change a live run: edit here, kill the SUPERVISOR first, then the
#    trainer, then start a fresh supervisor.
MANIFEST="${RUNS_DIR}/${ARM}.env"
{
  echo "# refcv6 arm ${ARM} -- generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by launch_refcv6.sh"
  echo "# ⛔ sourced ONCE at supervisor startup. See the warning in sup_refcv6.sh rule 7."
  echo "ARM=${ARM}"
  echo "STEPS=${STEPS}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "MAX_RELAUNCH=${MAX_RELAUNCH}"
  echo "TRAIN_CMD=\"${TRAIN_ENV} ${PY} ${TRAINER} ${ARGV}\""
} > "$MANIFEST"
echo "[launch] manifest -> ${MANIFEST}"

if [ "$DRY" -eq 1 ]; then
  echo "[launch] --dry-run: manifest written, gate NOT run, nothing launched."
  echo "[launch] TRAIN_CMD would be:"
  echo "  ${PY} ${TRAINER} ${ARGV}"
  exit 0
fi

# ---- 3. THE GATE ----------------------------------------------------------- #
echo "[launch] running the pre-launch gate ..."
# ⛔ No pipe. A pipe would make `$?` report the pipe's last element.
MSYS_NO_PATHCONV=1 PYTHONPATH="$STACK" TANITAD_STACK="$STACK" \
  "$PY" "${HERE}/prelaunch_gate.py" \
    --trainer "$TRAINER" --labels "$LABELS" --agent-join "$JOIN" \
    --anchors "$ANCHORS" --budget "$BUDGET" \
    ${W_AGENT:+--w-agent "$W_AGENT"} ${W_TAC_GOAL:+--w-tac-goal "$W_TAC_GOAL"} \
    ${HOST:+--host "$HOST"} \
    ${CLOSURE_JSON:+--closure-json "$CLOSURE_JSON"} \
    ${CLOSURE_REVERIFY_ROOT:+--closure-reverify-root "$CLOSURE_REVERIFY_ROOT"} \
    ${REMOTE_ROOT:+--remote-root "$REMOTE_ROOT"} \
    --smoke-arm "$ARM" --smoke-out "${RAW_DIR}/smoke-${ARM}" \
    --json "$GATE_JSON" > "${RAW_DIR}/gate_${ARM}.log" 2>&1
GATE_RC=$?

# ⭐ ASSERT ON THE ARTIFACT. The rc above is recorded and NOT trusted.
if [ ! -s "$GATE_JSON" ]; then
  echo "⛔ REFUSING TO LAUNCH: the gate wrote NO json at ${GATE_JSON} (rc=${GATE_RC})."
  echo "   The MISSING ARTIFACT is the evidence that the gate did not run."
  echo "   Log: ${RAW_DIR}/gate_${ARM}.log"
  exit 4
fi
GATE_VERDICT="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8')).get('verdict','ABSENT'))" "$GATE_JSON" 2>/dev/null)"
if [ "$GATE_VERDICT" != "PASS" ]; then
  echo "⛔ REFUSING TO LAUNCH: gate verdict = ${GATE_VERDICT:-UNREADABLE} (rc=${GATE_RC})."
  echo "   ⛔ INCONCLUSIVE IS NOT A PASS -- a check that could not RUN has not passed."
  "$PY" -c "
import json,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
for c in r.get('checks',[]):
    print('   [%-12s] %s: %s' % (c['verdict'], c['check'], str(c.get('why',''))[:140]))
" "$GATE_JSON" 2>/dev/null
  exit 5
fi
echo "[launch] gate PASS (${GATE_JSON})"

# ---- 4. refuse to resurrect a finished run --------------------------------- #
if [ -f "${OUT_DIR}/summary.json" ] && grep -q '"done"[[:space:]]*:[[:space:]]*true' "${OUT_DIR}/summary.json" 2>/dev/null; then
  echo "[launch] ${ARM} already carries a done-marker -- nothing to do."
  exit 0
fi

# ---- 5. start the supervisor ----------------------------------------------- #
mkdir -p "$OUT_DIR"
SUPLOG="${OUT_DIR}/sup_${ARM}.out"
# ⛔ The supervisor takes the lock itself. It is started WITHOUT inheriting any
#    fd from this script that it should not have.
nohup bash "${HERE}/sup_refcv6.sh" "$ARM" "$RUNS_DIR" >> "$SUPLOG" 2>&1 &
SUP_PID=$!
echo "$SUP_PID" > "${RUNS_DIR}/${ARM}.sup.pid"
echo "[launch] supervisor pid ${SUP_PID} (recorded in ${RUNS_DIR}/${ARM}.sup.pid)"

# ---- 6. ASSERT it is actually there ---------------------------------------- #
# ⛔ Every failure in this family REPORTS SUCCESS AND LEAVES NOTHING RUNNING.
#    The assertion reads /proc/<pid>/cmdline -- never `grep -c`, which
#    self-matches the ssh command that carried it.
sleep 5 200>&-
if ! bash "${HERE}/assert_supervisor.sh" "$ARM" "$RUNS_DIR"; then
  echo "⛔ THE SUPERVISOR IS NOT RUNNING despite a clean-looking start."
  echo "   Check ${SUPLOG} -- a lock held by an INHERITED fd (a trainer or a"
  echo "   stray sleep) prints 'another supervisor holds ...' and exits."
  exit 6
fi

echo "[launch] arm ${ARM} is supervised. steps=${STEPS} out=${OUT_DIR}"
echo "[launch] progress marker: grep for ZZ${ARM}-  in ${OUT_DIR}/sup_${ARM}.log"
echo "[launch] ⛔ kill by explicit PID only: supervisor ${SUP_PID}, trainer in ${OUT_DIR}/train.pid"
exit 0
