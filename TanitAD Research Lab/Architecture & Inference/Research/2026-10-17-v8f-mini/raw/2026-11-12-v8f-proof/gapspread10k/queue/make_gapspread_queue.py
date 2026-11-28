#!/usr/bin/env python
"""make_gapspread_queue.py -- card PREREG_V8F_PROOF_ADD134 (GAPSPREAD10K). Writes pod_gapspread10k_queue.sh from the queue of card ADD131
(boxfuse10k/queue/pod_boxfuse10k_queue.sh, sha256 prefix of its LF text 03be2485ff4d9c94) by asserted substitutions, the technique of make_boxfuse_queue.py.
The rollout function, the slot wait, the memory checks, the DRY mode, the BUSY/IDLE fragment and the marker line formats are kept as they are. What changes: the names of the
root, markers and log; the stop rule (3,600 s); the model seeds (0 to 3, the reader seed and map of seed m for m < 2 and of seed m - 2 for m >= 2, as in card ADD121); the gate
(the lead's test output must end in a pytest pass line, the refusal line starts with GATE); step 1 (record of card ADD134 and the file VGAPSP90.npz); the list of rollouts
(four, setting K0, tag I10RQ50KF5Hb20gSP90). The queue deletes, moves and stops nothing.

  python make_gapspread_queue.py [--src FILE] [--out FILE] [--print]
"""
import argparse
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "boxfuse10k", "queue", "pod_boxfuse10k_queue.sh"))
SRC_SHA16 = "03be2485ff4d9c94"
TAG = "ZZ MAKE-GAPSPREAD-QUEUE STOP: "
K0 = "READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5"
NAME = "I10RQ50KF5Hb20gSP90"
REFS = ["I10RLcKF5Hb20", "I10RQ50KF5Hb20", "I10RQ50KF5Hb20gGEc", "I10RQ50KF5Hb20gA90"]

HEADER = [
    "# pod_gapspread10k_queue.sh -- card ADD134 (GAPSPREAD10K): 4 rollouts, model seeds 0 to 3, through the wrapper variant rzgfhqetjv of card ADD102 (hash 28bc31f21ecb4ccb, unchanged):",
    "# the candidate form (reader-fed form RR with --vis-gap-file and --vis-closing, no label at inference) with the gap file VGAPSP90.npz (the base gap times the margin placed by the spread",
    "# of the three fit files) at readout K with no pre-roll (setting K0; run name I10RQ50KF5Hb20gSP90; --rule-every 5), pinned grid. Each run is called as the banked A90 run of the same model seed",
    "# (I10RQ50KF5Hb20gA90) with the gap file exchanged; the reader seed and map are those of model seed m for m < 2 and of seed m - 2 for m >= 2. A changed gap input of the rule is a study arm.",
    "# Step 0 gate: exit 4 (a line starting GATE) unless $GE/lead_tests/lead_pytest_queue.out exists and its last non-empty line is a pytest pass line (the test of this kit passed in the lead's run).",
    "# Step 1 builds no gap file: the queue stops (no rollout) unless $GF holds GAPSPREAD10K_GAPFILE_RECORD.json and VGAPSP90.npz, the record has card PREREG_V8F_PROOF_ADD134, pass_p0 true",
    "# and files.VGAPSP90.sha256 equal to the sha256 of the file beside it.",
    "# Step 2: the four rollouts, one after the other, each started only with fewer than 3 GPU compute processes and more than 28 GB free (memory.max minus anon); a run folder that exists is refused.",
    "# Time rule: before each start the seconds since the first line of $GE/GAPSPREAD10K_START.txt must be at most GS_STOP_S (default 3600, 1.0 pod-hours), else $GE/GAPSPREAD10K_STOP.txt. A failed rollout writes the stop marker.",
    "# Shared pod state file FORK_STATE_FILE (default /workspace/FORK_STATE.txt; empty = no write): a BUSY line before the first rollout and an IDLE line on every exit path after it.",
    "# Done marker $GE/GAPSPREAD10K_DONE.txt. Log $GE/gapspread10k_queue.log. DRY=1 prints every command and creates nothing. RESUME=1 continues after a stop marker. Nothing is removed or moved.",
    "# Banked runs needed (all four model seeds, T1_DONE): " + " ".join(REFS) + ". Written by make_gapspread_queue.py from the queue of card ADD131 (sha256 prefix %s). No training." % SRC_SHA16,
]

GATE_OK = r'''gate_ok() { # the lead's test output exists and its last non-empty line is a pytest pass line
  local f="$GE/lead_tests/lead_pytest_queue.out" last
  [ -f "$f" ] || return 1
  last=$(grep -v '^[[:space:]]*$' "$f" | tail -n 1 | tr -d '\r')
  [[ "$last" =~ [0-9]+\ passed ]] || return 1
  [[ "$last" =~ (failed|error) ]] && return 1
  return 0
}
'''

STEP = r'''step_inputs() { # step 1: the gap file of card ADD134 and its record are checked; nothing is built
  if [ "$DRY" = 1 ]; then
    echo "CHECK: $GF holds GAPSPREAD10K_GAPFILE_RECORD.json and VGAPSP90.npz"
    echo "CHECK: $GSREC has card PREREG_V8F_PROOF_ADD134, pass_p0 true and files.VGAPSP90.sha256 equal to the sha256 of the gap file"
    return 0
  fi
  [ -f "$GSREC" ] || stop "NO-GAP-RECORD $GSREC"
  [ -f "$GF/VGAPSP90.npz" ] || stop "NO-GAP-FILE $GF/VGAPSP90.npz"
  local adm
  adm=$("$PY" - "$GSREC" "$GF" <<'PYEOF'
import hashlib, json, os, sys
rec_path, gf = sys.argv[1:3]
rec = json.load(open(rec_path, encoding="utf-8"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if rec.get("card") != "PREREG_V8F_PROOF_ADD134":
    print("RECORD-IS-NOT-OF-THIS-CARD"); sys.exit(1)
if rec.get("pass_p0") is not True:
    print("PASS_P0-NOT-PASS"); sys.exit(1)
p = os.path.join(gf, "VGAPSP90.npz")
if not os.path.isfile(p) or ((rec.get("files") or {}).get("VGAPSP90") or {}).get("sha256") != sha(p):
    print("GAP-FILE-IS-NOT-THE-RECORDED-ONE VGAPSP90"); sys.exit(1)
print("ADMIT-OK gap file")
PYEOF
) || stop "INPUT-ADMISSION $adm"
  log "$adm"
}
'''

ROLLS = ('for k in "${ARMS[@]}"; do\n'
         '  r=$(( k >= 2 ? k - 2 : k ))\n'
         '  ' + K0 + ' QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${r}_q50.json" VGAP="$GF/VGAPSP90.npz" VCLOSE=1 rollout ' + NAME + ' RR "$k" "$r"\n'
         'done\n')


def build(src_text):
    state = {"t": src_text.replace("\r\n", "\n")}

    def once(a, b):
        if state["t"].count(a) != 1:
            raise SystemExit(TAG + "expected once, found %d: %r" % (state["t"].count(a), a[:90]))
        state["t"] = state["t"].replace(a, b)

    def span(a, b, new):
        s = state["t"]
        if s.count(a) != 1 or s.count(b) != 1 or not s.index(a) < s.index(b):
            raise SystemExit(TAG + "anchors: %r (%d) / %r (%d)" % (a[:60], s.count(a), b[:60], s.count(b)))
        state["t"] = s[:s.index(a)] + new + s[s.index(b):]

    def line_from(a, new):
        s = state["t"]
        if s.count(a) != 1:
            raise SystemExit(TAG + "line anchor found %d times: %r" % (s.count(a), a[:60]))
        i = s.index(a)
        state["t"] = s[:i] + new + s[s.index("\n", i):]

    lines = state["t"].split("\n")
    if not (lines[0] == "#!/usr/bin/env bash" and lines[13] == "set -u" and all(x.startswith("#") for x in lines[1:13])):
        raise SystemExit(TAG + "the head of the source is not the expected one")
    state["t"] = "\n".join([lines[0]] + HEADER + lines[13:])
    once("GE=${BF_ROOT:-/workspace/v8proof/boxfuse10k}", "GE=${GS_ROOT:-/workspace/v8proof/gapspread10k}")
    once("BFREC=$GF/BOXFUSE10K_GAPFILES_RECORD.json\n", "GSREC=$GF/GAPSPREAD10K_GAPFILE_RECORD.json\n")
    once("KQ=${BF_KIT:-$GE/kit_queue}", "KQ=${GS_KIT:-$GE/kit_queue}")
    once("SM=$GP/small_boxfuse10k", "SM=$GP/small_gapspread10k")
    once("STOPS=${BF_STOP_S:-4320}", "STOPS=${GS_STOP_S:-3600}")
    once("ARMS=(0 1); PS=\"\"", "ARMS=(0 1 2 3); PS=\"\"")
    once("LOG=$GP/boxfuse10k_queue.log", "LOG=$GP/gapspread10k_queue.log")
    line_from('log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD131 (BOXFUSE10K)',
              'log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD134 (GAPSPREAD10K): the candidate form with the gap file VGAPSP90.npz, '
              'at readout K with no pre-roll, model seeds ${ARMS[*]}"')
    once('gate_ok() { [ -f "$GE/lead_tests/lead_pytest_queue.out" ]; }\n', GATE_OK)
    once("# step 0 gate: the test of this kit passed in the lead's run (control d4)", "# step 0 gate: the test of this kit passed in the lead's run (control p4)")
    once('echo "GATE: needs $GE/lead_tests/lead_pytest_queue.out (exit 4 without it)"',
         'echo "GATE: needs $GE/lead_tests/lead_pytest_queue.out ending in a pytest pass line (exit 4 without it)"')
    once('echo "REFUSED: step 0 gate: $GE/lead_tests/lead_pytest_queue.out is missing"; exit 4',
         'echo "GATE: REFUSED, step 0 gate: $GE/lead_tests/lead_pytest_queue.out is missing or does not end in a pytest pass line"; exit 4')
    span("step_inputs() {", '[ "$DRY" = 1 ] || [ -f "$GRID" ] || stop "NO-PINNED-GRID"', STEP)
    line_from("  for nm in I10RLcKF5Hb20 ", "  for nm in %s; do" % " ".join(REFS))
    s = state["t"]
    if s.count("BUSY card ADD131 rollouts of the v8proof fork") != 2:
        raise SystemExit(TAG + "BUSY text count")
    state["t"] = s.replace("BUSY card ADD131 rollouts", "BUSY card ADD134 rollouts")
    span('# fork-state end\nfor k in', 'if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi', "# fork-state end\n" + ROLLS)
    once('NAMES="I10RQ50KF5Hb20gJEc I10RQ50KF5Hb20gJA90 I10RQ50Hb20gJEc"', 'NAMES="%s"' % NAME)
    t = state["t"].replace("BOXFUSE10K", "GAPSPREAD10K").replace("boxfuse10k", "gapspread10k")
    for bad in ("BF_", "BFREC", "boxfuse", "BOXFUSE", "JEc", "JA90", "VGAPJE", "pass_j", "I10RQ50Hb20"):
        if bad in t:
            raise SystemExit(TAG + "left over: " + bad)
    if (t.count(" rollout I10") != 1 or t.count("VCLOSE=1 rollout") != 1 or t.count("GAPSPREAD10K_START.txt") < 4
            or t.count('echo "ZZ GAPSPREAD10K DONE $(ts) all steps done" > "$GP/GAPSPREAD10K_DONE.txt"') != 1):
        raise SystemExit(TAG + "counts")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(HERE, "pod_gapspread10k_queue.sh"))
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    raw = io.open(a.src, "rb").read().replace(b"\r\n", b"\n")
    got = hashlib.sha256(raw).hexdigest()[:16]
    if got != SRC_SHA16:
        raise SystemExit(TAG + "source sha256 prefix %s, expected %s" % (got, SRC_SHA16))
    t = build(raw.decode("utf-8"))
    if a.print:
        sys.stdout.write(t)
        return 0
    with io.open(a.out, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(t)
    print("ZZ MAKE-GAPSPREAD-QUEUE DONE %s sha256 prefix %s" % (os.path.basename(a.out), hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
