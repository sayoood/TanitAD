#!/usr/bin/env bash
# refcv7 supervisor -- one arm per supervisor. It REFUSES to start the trainer unless a launch-gate
# PASS token matches the EXACT code tree, argv and data it is about to launch (SPEC_REFCV7 sec. 2).
#
#   ./sup_refcv7.sh <ARM> [RUNS_DIR]
#
# Derived from `sup_refcv6.sh` (TanitAD Research Lab/Architecture & Inference/Research/
# 2026-09-10-refcv6-build/code/sup_refcv6.sh on the tip). Every rule of that script is KEPT; the
# gate rules (G1-G4) are new. Every rule is here because it cost a run.
#
# The manifest `${RUNS_DIR}/${ARM}.env` is sourced ONCE and MUST define:
#   CODE            the shipped code tree (git archive of the gated commit; stack/ taniteval/)
#   PYTHON          the interpreter that runs the trainer
#   GATE_TOKEN      the PASS_<commit12>.json the launch gate wrote for this launch
#   GATE_ARGV_FILE  the JSON argv the gate bound -- the ONLY source of the trainer's argv
# and MAY define: COMMIT (40-char, must equal the token's), GATE_KEY_FILE, MAX_RELAUNCH, POLL_S,
# and exported environment (OMP_NUM_THREADS ...). OUT_DIR and STEPS are READ FROM THE GATED ARGV;
# a manifest that also sets them must agree with it, or the supervisor refuses.
#
# ⛔ G1. NO FREE-FORM LAUNCH LINE. A manifest carrying TRAIN_CMD is REFUSED: a hand-written command
#    line is exactly what the gate binds against, and a supervisor that runs one bypasses it. The
#    trainer is started ONLY as `launch_gate.py exec`, which re-verifies the token and then REPLACES
#    itself with the trainer on the bound argv (the recorded pid becomes the trainer's pid).
# ⛔ G2. VERIFY BEFORE EVERY LAUNCH, RELAUNCHES INCLUDED. The code tree, the argv and every data
#    input are re-measured against the token each time; a relaunch after a crash is a launch.
# ⛔ G3. THE GATE'S VERDICT IS ITS JSON, NEVER ITS EXIT CODE. `verify --json` writes the verdict;
#    the supervisor reads `"verdict": "MATCH"` out of that file. A missing file is a refusal.
# ⛔ G4. A REFUSAL IS FINAL. The supervisor exits 5 and does NOT relaunch -- relaunching a refused
#    launch only repeats the refusal (or, worse, races a fix into a half-verified state).
#
# ⛔ 1. THE LOCK FD IS CLOSED ON EVERY CHILD (`200>&-`), NOT JUST THE TRAINER.
#    `exec 200>"$LOCK"; flock -n 200` puts the lock on fd 200. A child launched as
#    `nohup python3 ... &` INHERITS EVERY OPEN FD, so the trainer ends up holding the supervisor's
#    lock for its entire life -- and once the old supervisor is killed the lock is NEVER released
#    while its trainer lives, so no replacement can ever start.
#    ⛔⛔ AND `200>&-` ON THE TRAINER ALONE IS NOT ENOUGH -- the lock was once held by `sleep 180`,
#    the supervisor's own poll child. EVERY child gets `200>&-`: the sleeps, the python helpers,
#    the gate calls.
# ⛔ 2. THE DONE-MARKER IS WRITTEN ON COMPLETION, AND CHECKED AT STARTUP (never resurrect a
#    finished run).
# ⛔ 3. KILL BY EXPLICIT PID. `pkill -f <trainer>` SELF-MATCHES the ssh command that issued it.
# ⛔ 4. THE PROGRESS MARKER IS DISJOINT FROM ANYTHING GREPPED (an echoing PTY matches its own
#    filter). Counts are computed HERE and emitted as an opaque `ZZ...ZZ` token.
# ⛔ 5. SUCCESS IS ASSERTED ON THE ARTIFACT (metrics.jsonl's last step), NEVER ON AN EXIT CODE.
# ⚠️ 6. NEVER `sed -i` THIS FILE WHILE IT IS RUNNING (bash reads it lazily by byte offset).
# ⚠️ 7. TO CHANGE A LIVE RUN: a new argv means a NEW GATE RUN and a new token. Kill the SUPERVISOR
#    first, then the trainer (explicit pid), then start a fresh supervisor on the new manifest.

set -u  # ⛔ deliberately NOT `set -e`: a supervisor that exits on the first non-zero command is
        # not a supervisor.

ARM="${1:?usage: sup_refcv7.sh <ARM> [RUNS_DIR]}"
RUNS_DIR="${2:-/home/nvidia/refcv7_run/runs.d}"
MANIFEST="${RUNS_DIR}/${ARM}.env"

if [ ! -f "$MANIFEST" ]; then
  echo "[sup:${ARM}] FATAL: no manifest at ${MANIFEST}"
  exit 2
fi

# The manifest is the ONLY source of its variables: nothing is inherited from the caller's
# environment (MEASURED 2026-09-27 on Thor: a runner's exported STEPS=unit,... was read as the
# manifest's step count and the supervisor refused a valid launch).
unset CODE PYTHON GATE_TOKEN GATE_ARGV_FILE GATE_KEY_FILE TRAIN_CMD STEPS OUT_DIR
# Sourced ONCE, on purpose (rule 7).
# shellcheck disable=SC1090
. "$MANIFEST"

: "${CODE:?manifest must define CODE}"
: "${PYTHON:?manifest must define PYTHON}"
: "${GATE_TOKEN:?manifest must define GATE_TOKEN}"
: "${GATE_ARGV_FILE:?manifest must define GATE_ARGV_FILE}"

if [ -n "${TRAIN_CMD:-}" ]; then          # ⛔ G1
  echo "[sup:${ARM}] REFUSED: the manifest defines TRAIN_CMD. sup_refcv7 launches ONLY through"
  echo "[sup:${ARM}]   launch_gate.py exec on the gated argv; a free-form command bypasses the gate."
  exit 5
fi

GATE="${CODE}/stack/scripts/launch_gate.py"
if [ ! -f "$GATE" ]; then
  echo "[sup:${ARM}] FATAL: no launch gate at ${GATE} -- this tree cannot be gated"
  exit 2
fi

# OUT_DIR and STEPS come from the GATED argv -- one source of truth, never a second copy.
ARGV_FACTS="$("$PYTHON" - "$GATE_ARGV_FILE" <<'PYEOF' 200>&-
import json, sys
a = json.load(open(sys.argv[1], encoding="utf-8"))
a = a["argv"] if isinstance(a, dict) else a
def val(flag):
    v = None
    for i, t in enumerate(a):
        if t == flag and i + 1 < len(a):
            v = a[i + 1]
    return v
# no trailing newline: a Windows python writes "\r\n" into $(...), and bash keeps the \r
sys.stdout.write((val("--out") or "") + "\t" + (val("--steps") or ""))
PYEOF
)"
ARGV_OUT="${ARGV_FACTS%%$'\t'*}"
ARGV_STEPS="${ARGV_FACTS##*$'\t'}"
if [ -z "$ARGV_OUT" ] || [ -z "$ARGV_STEPS" ]; then
  echo "[sup:${ARM}] REFUSED: the gated argv ${GATE_ARGV_FILE} carries no --out / --steps"
  exit 5
fi
if [ -n "${OUT_DIR:-}" ] && [ "$OUT_DIR" != "$ARGV_OUT" ]; then
  echo "[sup:${ARM}] REFUSED: manifest OUT_DIR=${OUT_DIR} but the gated argv says --out ${ARGV_OUT}"
  exit 5
fi
if [ -n "${STEPS:-}" ] && [ "$STEPS" != "$ARGV_STEPS" ]; then
  echo "[sup:${ARM}] REFUSED: manifest STEPS=${STEPS} but the gated argv says --steps ${ARGV_STEPS}"
  exit 5
fi
OUT_DIR="$ARGV_OUT"
STEPS="$ARGV_STEPS"

LOCK="${RUNS_DIR}/${ARM}.lock"
LOG="${OUT_DIR}/train.log"
ERRLOG="${OUT_DIR}/train.stderr.log"
SUPLOG="${OUT_DIR}/sup_${ARM}.log"
DONE_MARKER="${OUT_DIR}/summary.json"
MAX_RELAUNCH="${MAX_RELAUNCH:-8}"
POLL_S="${POLL_S:-120}"

GATE_ARGS=(--token "$GATE_TOKEN" --argv-file "$GATE_ARGV_FILE" --tree "$CODE")
if [ -n "${COMMIT:-}" ]; then GATE_ARGS+=(--commit "$COMMIT"); fi
if [ -n "${GATE_KEY_FILE:-}" ]; then GATE_ARGS+=(--key-file "$GATE_KEY_FILE"); fi

mkdir -p "$OUT_DIR" "$RUNS_DIR"

log() { echo "[sup:${ARM}] $(date -u +%Y-%m-%dT%H:%M:%SZ 200>&-) $*" | tee -a "$SUPLOG" 200>&-; }

# ---- rule 2: never resurrect a finished run -------------------------------------------------- #
if [ -f "$DONE_MARKER" ] && grep -q '"done"[[:space:]]*:[[:space:]]*true' "$DONE_MARKER" 2>/dev/null; then
  log "done-marker present at ${DONE_MARKER} -- run is FINISHED. Exiting without launching."
  exit 0
fi

# ---- rule 1: take the lock, and close it on every child -------------------------------------- #
exec 200>"$LOCK"
if ! flock -n 200; then
  log "another supervisor holds ${LOCK} -- exiting."
  log "⚠️ If nothing is actually supervising, the holder may be an INHERITED fd"
  log "   (a trainer or a stray sleep). Name it with:"
  log "   for p in /proc/[0-9]*/fd/*; do [ \"\$(readlink \$p 2>/dev/null)\" = \"${LOCK}\" ] && tr '\\0' ' ' < /proc/\$(echo \$p|cut -d/ -f3)/cmdline && echo; done"
  log "   ⛔ Do NOT kill a live trainer to free it -- point a new supervisor at a FRESH lock path instead."
  exit 3
fi
log "lock acquired on ${LOCK} (fd 200); supervisor pid $$"

# ---- the gate's verdict, read from its JSON (G3) --------------------------------------------- #
gate_verdict() {   # $1 = the JSON the gate wrote -> prints MATCH / REFUSED / MISSING
  "$PYTHON" - "$1" <<'PYEOF' 200>&-
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.stdout.write("MISSING"); raise SystemExit(0)
sys.stdout.write("MATCH" if d.get("verdict") == "MATCH" else "REFUSED")
PYEOF
}

gate_reasons() {   # $1 = the JSON -> its reasons, one per line (for the log)
  "$PYTHON" - "$1" <<'PYEOF' 200>&-
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception as e:
    print(f"(no verdict file: {type(e).__name__})"); raise SystemExit(0)
for r in d.get("reasons") or ["(no reasons recorded)"]:
    print(r[:400])
PYEOF
}

# ---- artifact-based completion test (rule 5) --------------------------------------------------- #
run_is_complete() {
  local mj="${OUT_DIR}/metrics.jsonl"
  [ -s "$mj" ] || return 1
  "$PYTHON" - "$mj" "$STEPS" <<'PYEOF' 200>&-
import json, sys
steps = []
try:
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue            # a torn final line is skipped, never fatal
            if isinstance(r, dict) and isinstance(r.get("step"), int):
                steps.append(r["step"])
except Exception:
    sys.exit(1)
sys.exit(0 if steps and max(steps) >= int(sys.argv[2]) - 1 else 1)
PYEOF
}

write_done_marker() {
  local final_step="$1"
  "$PYTHON" - "$DONE_MARKER" "$ARM" "$final_step" "$STEPS" "$OUT_DIR" "$GATE_TOKEN" <<'PYEOF' 200>&-
import json, sys, time
path, arm, final_step, steps, out_dir, token = sys.argv[1:7]
with open(path, "w", encoding="utf-8") as fh:
    json.dump({
        "done": True, "arm": arm,
        "final_step": int(final_step), "steps_requested": int(steps),
        "out_dir": out_dir, "gate_token": token,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": ("Written by sup_refcv7.sh AFTER metrics.jsonl was read and its step count "
                 "checked. Its presence is the supervisor's OFF SWITCH."),
    }, fh, indent=2)
PYEOF
}

final_step_from_metrics() {
  "$PYTHON" - "${OUT_DIR}/metrics.jsonl" <<'PYEOF' 200>&-
import json, sys
steps = []
try:
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r, dict) and isinstance(r.get("step"), int):
                steps.append(r["step"])
except Exception:
    sys.stdout.write("-1"); raise SystemExit(0)
sys.stdout.write(str(max(steps) if steps else -1))
PYEOF
}

# ---- the supervise loop -------------------------------------------------------------------- #
launch=0
while [ "$launch" -lt "$MAX_RELAUNCH" ]; do
  launch=$((launch + 1))

  if run_is_complete; then
    fs="$(final_step_from_metrics)"
    log "metrics.jsonl already reaches step ${fs} >= ${STEPS} -- writing done-marker."
    write_done_marker "$fs"
    log "done-marker written; exiting cleanly."
    exit 0
  fi

  # ⛔ G2 + G3: re-measure tree, argv and data against the token BEFORE this launch; the
  #    verdict is the JSON file, read back, never the exit status.
  VJSON="${OUT_DIR}/gate_verify_${launch}.json"
  rm -f "$VJSON" 200>&-
  "$PYTHON" "$GATE" verify "${GATE_ARGS[@]}" --json "$VJSON" >> "$SUPLOG" 2>&1 200>&-
  vrc=$?
  verdict="$(gate_verdict "$VJSON")"
  if [ "$vrc" -ne 0 ] || [ "$verdict" != "MATCH" ]; then
    log "⛔ LAUNCH GATE REFUSED launch #${launch} (rc ${vrc}, verdict ${verdict}):"
    gate_reasons "$VJSON" | while IFS= read -r line; do log "   - ${line}"; done
    echo "ZZGATEREFUSED-${ARM}-${launch}ZZ" | tee -a "$SUPLOG" > /dev/null 200>&-
    exit 5                                             # ⛔ G4: final
  fi
  echo "ZZGATEOK-${ARM}-${launch}ZZ" | tee -a "$SUPLOG" > /dev/null 200>&-
  log "gate MATCH for launch #${launch} (${VJSON})"

  EJSON="${OUT_DIR}/gate_exec_${launch}.json"
  rm -f "$EJSON" 200>&-
  log "launch #${launch} of ${MAX_RELAUNCH}: ${PYTHON} ${GATE} exec ${GATE_ARGS[*]}"
  # ⛔ rule 1: `200>&-` closes the lock fd in the child. ⛔ G1: the ONLY launch line; `exec`
  #    re-verifies and then replaces itself with the trainer, so TRAIN_PID is the trainer's pid.
  nohup "$PYTHON" "$GATE" exec "${GATE_ARGS[@]}" --json "$EJSON" >> "$LOG" 2>> "$ERRLOG" 200>&- &
  TRAIN_PID=$!
  echo "$TRAIN_PID" > "${OUT_DIR}/train.pid"
  log "trainer pid ${TRAIN_PID} (recorded in ${OUT_DIR}/train.pid -- kill by THIS pid, never pkill -f)"

  # ---- poll ---------------------------------------------------------------------------------- #
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    # rule 4: compute counts HERE, emit an OPAQUE marker. The client greps `ZZ...ZZ`.
    cur="$(final_step_from_metrics)"
    n_err=0
    if [ -s "$ERRLOG" ]; then
      # ⛔ the searched words are BUILT here, never written literally, so this command line
      #    cannot match itself in an echoing PTY.
      pat="$(printf 'Trace''back|CUDA out of mem''ory|OutOfMemory' 200>&-)"
      # ⛔ NOT `|| echo 0`: grep -c prints its count AND exits 1 when it is zero, so that printed a
      #    SECOND 0 and split this token over two lines (MEASURED 2026-09-26 on refcv6-r101-s0).
      #    Exit 2 means the log could not be READ: that is U (unread), never zero errors.
      rc=0
      n_err="$(grep -Ec "$pat" "$ERRLOG" 2>/dev/null 200>&-)" || rc=$?
      [ "$rc" -le 1 ] || n_err=U
    fi
    echo "ZZ${ARM}-${cur}-${STEPS}-${n_err}-${launch}ZZ" | tee -a "$SUPLOG" > /dev/null 200>&-
    sleep "$POLL_S" 200>&-   # ⛔ rule 1: the sleep gets it too
  done

  wait "$TRAIN_PID"
  rc=$?
  log "trainer pid ${TRAIN_PID} exited rc=${rc} (⚠️ rc is NOT the verdict -- the artifact is)"

  # ⛔ G4: an exec that REFUSED (the token no longer matched between verify and exec) is final
  everdict="$(gate_verdict "$EJSON")"
  if [ "$everdict" != "MATCH" ]; then
    log "⛔ launch_gate.py exec did not MATCH (${everdict}) -- the trainer never started:"
    gate_reasons "$EJSON" | while IFS= read -r line; do log "   - ${line}"; done
    echo "ZZGATEREFUSED-${ARM}-${launch}ZZ" | tee -a "$SUPLOG" > /dev/null 200>&-
    exit 5
  fi

  if run_is_complete; then
    fs="$(final_step_from_metrics)"
    log "ARTIFACT CHECK PASSED: metrics.jsonl reaches step ${fs} >= ${STEPS}."
    write_done_marker "$fs"
    log "done-marker written at ${DONE_MARKER}; supervisor exiting cleanly."
    exit 0
  fi

  cur="$(final_step_from_metrics)"
  log "ARTIFACT CHECK FAILED: metrics.jsonl reaches ${cur}, wanted >= ${STEPS}."
  log "relaunching into the SAME out dir: train() RESUMES from ckpt.pt (model, optimiser,"
  log "   step, data position); the gate re-verifies first. MAX_RELAUNCH=1 forbids it."
  sleep 30 200>&-
done

log "⛔ exhausted ${MAX_RELAUNCH} launches without a complete artifact. NO done-marker written."
log "   The absence of ${DONE_MARKER} is the admissible evidence that this arm did not finish."
exit 4
