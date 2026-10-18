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
# The post-training ZERO-GRADIENT GATE (G5, below) is ON by default and is tuned by these optional
# manifest variables (all read ONCE, at startup -- rule 7 applies):
#   ZEROGRAD_GATE     on (default) | off.  `off` is LOGGED LOUDLY and recorded in the done-marker.
#   ZEROGRAD_ALLOW    regexes (bash array, or one string with one regex per line) of parameter names that are
#                     FROZEN BY DESIGN -- `re.fullmatch` against the gate's parameter names, e.g.
#                     ZEROGRAD_ALLOW=('core\.encoder\..*').  Each one is logged and recorded.
#   ZEROGRAD_CONTROL  regexes at least one TRAINED parameter must fully match (a same-breath positive control).
#   ZEROGRAD_EXCLUDE  extra state_dict entry names that are buffers, not parameters (naming only).
#   ZEROGRAD_CKPT     the checkpoint to gate (default ${OUT_DIR}/ckpt.pt, the trainer's resume checkpoint).
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
#
# ⛔ G5. THE DONE-MARKER IS NOT WRITTEN UNTIL THE ZERO-GRADIENT GATE HAS LOOKED AT THE FINAL CHECKPOINT.
#    CLAUDE.md "BUILT, TESTED, AND UNREACHABLE FROM ITS CALLER": a head wired into the model but not into the
#    loss trains for days and is found only after the arm is trained (refcv5-v2: 18,472 parameters).
#    `stack/scripts/zerograd_gate.py` reads that from the checkpoint's optimizer state. Its VERDICT is its
#    JSON (like G3) -- a traceback also exits 1, so the exit code alone cannot tell a FAIL from a crash:
#      PASS            -> the done-marker is written, with a `zerograd_gate` record.
#      NOT_APPLICABLE  -> the checkpoint carries no optimizer state (error_kind no_optimizer_state): the
#                         done-marker is written WITH that record -- never silently a pass.
#      DISABLED        -> ZEROGRAD_GATE=off: written with that record, and a loud log line.
#      FAIL            -> an untrained parameter, or a failed control: NO done-marker; ZEROGRAD_FAIL.json
#                         (embedding the gate's JSON) is written; the supervisor exits 6.
#      UNEVALUATED     -> the gate could not look (unreadable checkpoint, no gate JSON, JSON and exit code
#                         disagree, gate script missing): treated EXACTLY like FAIL -- a gate that did not run
#                         is not a pass.  `ZEROGRAD_GATE=off` is the explicit, logged way to opt out.
#    ⛔ A ZEROGRAD_FAIL IS FINAL, LIKE G4, AND IT NEVER RELAUNCHES. A relaunch only follows an INCOMPLETE
#    artifact; the gate runs only on a COMPLETE one, and the supervisor `exit`s after refusing. A later
#    supervisor start sees ZEROGRAD_FAIL.json first and exits 6 without launching or gating again. To re-gate
#    (e.g. after adding ZEROGRAD_ALLOW for a frozen-by-design parameter), delete that file deliberately.
#    ⚠️ The TRAINER writes its own summary.json {"done": true} at completion, before this script can gate.
#    That file is the trainer's artifact and is left untouched on FAIL, so a consumer that keys on
#    summary.json alone is NOT protected: it must also require that ZEROGRAD_FAIL.json is absent.
#    A done-marker with no `zerograd_gate` record (the trainer's, or a pre-G5 supervisor's) is therefore not
#    "FINISHED": the supervisor runs the gate on it and finalises WITHOUT relaunching.

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
unset ZEROGRAD_GATE ZEROGRAD_ALLOW ZEROGRAD_CONTROL ZEROGRAD_EXCLUDE ZEROGRAD_CKPT
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

# ⛔ G5: the post-training zero-gradient gate.
ZG_MODE="${ZEROGRAD_GATE:-on}"
case "$ZG_MODE" in
  on|off) ;;
  *) echo "[sup:${ARM}] REFUSED: ZEROGRAD_GATE=${ZG_MODE} -- must be 'on' or 'off' (a typo must not disable the gate)"
     exit 2 ;;
esac
ZG_SCRIPT="${CODE}/stack/scripts/zerograd_gate.py"
ZG_CKPT="${ZEROGRAD_CKPT:-${OUT_DIR}/ckpt.pt}"
ZG_JSON="${OUT_DIR}/zerograd_gate.json"
ZG_TXT="${OUT_DIR}/zerograd_gate.txt"
ZG_SUMMARY="${OUT_DIR}/zerograd_summary.json"
ZG_FAIL_MARKER="${OUT_DIR}/ZEROGRAD_FAIL.json"
ZG_STATUS=""
FINALIZE_ONLY=0

GATE_ARGS=(--token "$GATE_TOKEN" --argv-file "$GATE_ARGV_FILE" --tree "$CODE")
if [ -n "${COMMIT:-}" ]; then GATE_ARGS+=(--commit "$COMMIT"); fi
if [ -n "${GATE_KEY_FILE:-}" ]; then GATE_ARGS+=(--key-file "$GATE_KEY_FILE"); fi

mkdir -p "$OUT_DIR" "$RUNS_DIR"

log() { echo "[sup:${ARM}] $(date -u +%Y-%m-%dT%H:%M:%SZ 200>&-) $*" | tee -a "$SUPLOG" 200>&-; }

# ---- G5: a zero-gradient FAIL is final (like G4) -- never launch, never re-gate -------------- #
if [ -f "$ZG_FAIL_MARKER" ]; then
  log "⛔ ${ZG_FAIL_MARKER} present -- this run FAILED the post-training zero-gradient gate. NOT launching."
  log "   Read the gate report in that file. To re-gate (e.g. after adding ZEROGRAD_ALLOW for a frozen-by-design"
  log "   parameter) delete the file DELIBERATELY and restart; it never clears itself."
  exit 6
fi

# ---- rule 2: never resurrect a finished run -------------------------------------------------- #
if [ -f "$DONE_MARKER" ] && grep -q '"done"[[:space:]]*:[[:space:]]*true' "$DONE_MARKER" 2>/dev/null; then
  if [ "$ZG_MODE" = "off" ] || grep -q '"zerograd_gate"' "$DONE_MARKER" 2>/dev/null; then
    log "done-marker present at ${DONE_MARKER} -- run is FINISHED. Exiting without launching."
    exit 0
  fi
  # The trainer writes {"done": true} itself, before this script can gate (or a pre-G5 supervisor wrote it).
  # Not "FINISHED" yet -- but ALSO never a reason to relaunch: gate it, finalise it, exit.
  FINALIZE_ONLY=1
  log "done-marker present at ${DONE_MARKER} but it carries no zerograd_gate record -- the run is complete;"
  log "   it will be GATED and finalised WITHOUT relaunching."
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
  "$PYTHON" - "$DONE_MARKER" "$ARM" "$final_step" "$STEPS" "$OUT_DIR" "$GATE_TOKEN" "$ZG_SUMMARY" <<'PYEOF' 200>&-
import json, sys, time
path, arm, final_step, steps, out_dir, token, zg_summary = sys.argv[1:8]
try:                                   # G5: the zero-gradient gate's record rides in the marker
    with open(zg_summary, "r", encoding="utf-8") as zfh:
        zg = json.load(zfh)
except Exception:
    zg = {"status": "NOT_RUN", "reason": "no zerograd summary was written before the done-marker"}
with open(path, "w", encoding="utf-8") as fh:
    json.dump({
        "done": True, "arm": arm,
        "final_step": int(final_step), "steps_requested": int(steps),
        "out_dir": out_dir, "gate_token": token,
        "zerograd_gate": zg,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": ("Written by sup_refcv7.sh AFTER metrics.jsonl was read and its step count "
                 "checked. Its presence is the supervisor's OFF SWITCH."),
    }, fh, indent=2)
PYEOF
}

# ---- G5: the post-training zero-gradient gate ------------------------------------------------- #
# $1 = mode, $2 = the gate's exit code, $3 = its JSON, $4 = ckpt, $5 = summary out, $6 = final step,
# $7 = steps requested, then the flat `--allow R --control R --exclude-entry R` list the gate was given.
# Prints ONE word (PASS | FAIL | NOT_APPLICABLE | DISABLED | UNEVALUATED) and writes the summary JSON.
# The verdict is the gate's JSON; the exit code must AGREE with it or the answer is UNEVALUATED (G3).
zg_decide() {
  "$PYTHON" - "$@" <<'PYEOF' 200>&-
import json, sys
mode, rc, jp, ckpt, outp, final_step, steps = sys.argv[1:8]
pats = sys.argv[8:]
def _flags(flag):
    return [pats[i + 1] for i in range(0, len(pats) - 1, 2) if pats[i] == flag]
rc = int(rc)
rep, rerr = None, None
try:
    with open(jp, "r", encoding="utf-8") as fh:
        rep = json.load(fh)
except Exception as e:
    rerr = "%s: %s" % (type(e).__name__, str(e)[:120])
status, reason = "UNEVALUATED", ""
if mode == "off":
    status, reason = "DISABLED", "ZEROGRAD_GATE=off in the manifest -- the gate did not run"
elif not isinstance(rep, dict) or rep.get("gate") != "zerograd_gate":
    reason = "the gate wrote no readable report (exit %d; %s)" % (rc, rerr or "not a zerograd_gate report")
else:
    v = rep.get("verdict")
    if v == "PASS" and rc == 0:
        status = "PASS"
    elif v == "FAIL" and rc == 1:
        status = "FAIL"
        reason = "untrained parameter(s) or a failed positive control"
    elif v == "ERROR" and rc == 2:
        if rep.get("error_kind") == "no_optimizer_state":
            status, reason = "NOT_APPLICABLE", "not applicable (no optimizer state): %s" % str(rep.get("error"))[:200]
        else:
            reason = "the gate could not evaluate the checkpoint (%s): %s" % (rep.get("error_kind"), str(rep.get("error"))[:300])
    else:
        reason = "the gate's verdict %r disagrees with its exit code %d" % (v, rc)
opt = (rep or {}).get("optimizer") or {}
off = []
for r in ((rep or {}).get("offenders") or []):
    off.append({"names": r.get("names"), "case": r.get("case"), "numel_lo": r.get("numel_lo"), "numel_hi": r.get("numel_hi")})
ckstep = opt.get("ckpt_step")
summary = {
    "status": status, "reason": reason, "gate_exit_code": rc, "gate_verdict": (rep or {}).get("verdict"),
    "gate_version": (rep or {}).get("gate_version"), "ckpt": ckpt, "ckpt_step": ckstep,
    "final_step": int(final_step), "steps_requested": int(steps),
    "ckpt_step_is_final": (ckstep is not None and int(ckstep) >= int(steps)),
    "n_registered": opt.get("n_registered"), "n_trained": opt.get("n_trained"),
    "n_untrained": len(off), "untrained": off,
    "failed_controls": [c.get("pattern") for c in ((rep or {}).get("controls") or []) if not c.get("ok")],
    "allow": _flags("--allow"), "unused_allow": (rep or {}).get("unused_allow") or [],
    "control": _flags("--control"), "exclude_entry": _flags("--exclude-entry"),
    "param_names_source": ((rep or {}).get("alignment") or {}).get("param_names_source"),
    "gate_report": jp,
}
with open(outp, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
sys.stdout.write(status)
PYEOF
}

# Appends `--FLAG pattern` to ZG_ARGS for every pattern in "$@" (a bash array element, or a string holding
# one regex per line). Empty lines are skipped.
zg_add_patterns() {
  local flag="$1" entry line
  shift
  for entry in "$@"; do
    while IFS= read -r line; do
      line="${line%$'\r'}"
      [ -n "$line" ] && ZG_ARGS+=("$flag" "$line")
    done <<< "$entry"
  done
}

# Runs the gate on the final checkpoint. Sets ZG_STATUS and writes ${ZG_SUMMARY}; never exits.
zerograd_check() {   # $1 = final step
  local fs="$1" zrc=0 line i
  ZG_STATUS=""
  rm -f "$ZG_JSON" "$ZG_TXT" "$ZG_SUMMARY" 200>&-
  ZG_ARGS=()
  zg_add_patterns --allow ${ZEROGRAD_ALLOW[@]+"${ZEROGRAD_ALLOW[@]}"}
  zg_add_patterns --control ${ZEROGRAD_CONTROL[@]+"${ZEROGRAD_CONTROL[@]}"}
  zg_add_patterns --exclude-entry ${ZEROGRAD_EXCLUDE[@]+"${ZEROGRAD_EXCLUDE[@]}"}
  if [ "$ZG_MODE" = "off" ]; then
    log "⚠️ ZEROGRAD GATE DISABLED (ZEROGRAD_GATE=off in the manifest): untrained parameters would NOT be detected."
  else
    for ((i = 0; i < ${#ZG_ARGS[@]}; i += 2)); do
      log "zerograd gate ${ZG_ARGS[i]#--} pattern: ${ZG_ARGS[i+1]}"
    done
    log "zerograd gate: ${PYTHON} ${ZG_SCRIPT} --ckpt ${ZG_CKPT} (CPU, reads the optimizer state; no GPU, no data)"
    "$PYTHON" "$ZG_SCRIPT" --ckpt "$ZG_CKPT" --json-out "$ZG_JSON" --sha256 ${ZG_ARGS[@]+"${ZG_ARGS[@]}"} > "$ZG_TXT" 2>&1 200>&-
    zrc=$?
  fi
  ZG_STATUS="$(zg_decide "$ZG_MODE" "$zrc" "$ZG_JSON" "$ZG_CKPT" "$ZG_SUMMARY" "$fs" "$STEPS" ${ZG_ARGS[@]+"${ZG_ARGS[@]}"})"
  [ -n "$ZG_STATUS" ] || ZG_STATUS=UNEVALUATED
  if grep -q '"ckpt_step_is_final": false' "$ZG_SUMMARY" 2>/dev/null 200>&-; then
    log "⚠️ the gate read a checkpoint whose step is BELOW the requested ${STEPS} (or carries none): its verdict is"
    log "   about that earlier state, not the final one -- see ckpt_step in ${ZG_SUMMARY}."
  fi
  if [ -s "$ZG_TXT" ]; then
    grep -E 'UNTRAINED|WARN|control|VERDICT|ERROR' "$ZG_TXT" 200>&- | while IFS= read -r line; do log "   gate: ${line%$'\r'}"; done
  fi
  echo "ZZZG-${ARM}-${ZG_STATUS}ZZ" | tee -a "$SUPLOG" > /dev/null 200>&-
}

write_zerograd_fail_marker() {   # $1 = final step
  "$PYTHON" - "$ZG_FAIL_MARKER" "$ARM" "$1" "$STEPS" "$OUT_DIR" "$ZG_SUMMARY" "$ZG_JSON" <<'PYEOF' 200>&-
import json, sys, time
path, arm, final_step, steps, out_dir, zg_summary, zg_json = sys.argv[1:8]
def _load(p):
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None
summary = _load(zg_summary) or {"status": "UNEVALUATED", "reason": "no zerograd summary was written"}
with open(path, "w", encoding="utf-8") as fh:
    json.dump({
        "zerograd_fail": True, "status": summary.get("status"), "reason": summary.get("reason"), "arm": arm,
        "final_step": int(final_step), "steps_requested": int(steps), "out_dir": out_dir,
        "zerograd_summary": summary, "gate_report": _load(zg_json),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": ("Written by sup_refcv7.sh INSTEAD of the done-marker. Its presence makes every later supervisor "
                 "start exit 6 without launching or re-gating. The trainer's own summary.json is untouched: a "
                 "consumer that keys on it alone is NOT protected -- require this file to be ABSENT."),
    }, fh, indent=2)
PYEOF
}

# G5: gate the final checkpoint, then write EITHER the done-marker OR ZEROGRAD_FAIL.json. Returns 0 or 6;
# callers `exit` on it -- there is no path from here back into the relaunch loop.
finalize_run() {   # $1 = final step
  local fs="$1"
  zerograd_check "$fs"   # ZG-HOOK
  case "$ZG_STATUS" in
    PASS|NOT_APPLICABLE|DISABLED)
      write_done_marker "$fs"
      log "done-marker written at ${DONE_MARKER} (zerograd gate: ${ZG_STATUS}); supervisor exiting cleanly."
      return 0 ;;
    *)
      write_zerograd_fail_marker "$fs"
      log "⛔ ZEROGRAD GATE ${ZG_STATUS}: NO done-marker written. ${ZG_FAIL_MARKER} holds the gate report."
      log "   The supervisor exits 6 and does NOT relaunch; the trainer's own summary.json is untouched."
      echo "ZZZGFAIL-${ARM}ZZ" | tee -a "$SUPLOG" > /dev/null 200>&-
      return 6 ;;
  esac
}

marker_step() {   # the step recorded in an existing done-marker (trainer-written: "step"), or -1
  "$PYTHON" - "$DONE_MARKER" <<'PYEOF' 200>&-
import json, sys
try:
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        d = json.load(fh)
    s = d.get("step", d.get("final_step"))
    sys.stdout.write(str(int(s)))
except Exception:
    sys.stdout.write("-1")
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

# ---- G5: a complete run whose done-marker carries no zerograd record is gated, never relaunched --- #
if [ "$FINALIZE_ONLY" = "1" ]; then
  fs="$(final_step_from_metrics)"
  [ "$fs" != "-1" ] || fs="$(marker_step)"
  finalize_run "$fs"
  exit $?
fi

# ---- the supervise loop -------------------------------------------------------------------- #
launch=0
while [ "$launch" -lt "$MAX_RELAUNCH" ]; do
  launch=$((launch + 1))

  if run_is_complete; then
    fs="$(final_step_from_metrics)"
    log "metrics.jsonl already reaches step ${fs} >= ${STEPS} -- gating (zero-gradient), then writing done-marker."
    finalize_run "$fs"
    exit $?
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
    finalize_run "$fs"
    exit $?
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
