#!/usr/bin/env python
"""make_presattrc_queue.py -- card PREREG_V8F_PROOF_ADD115 (PRESATTRC10K). Writes pod_presattrc10k_queue.sh from the queue of card ADD114
(presattr/queue/pod_presattr10k_queue.sh, sha256 prefix de04e5b848230293) by asserted substitutions. The rollout function, the slot wait, the stop rule and
the marker line formats are kept as they are. Step 1 checks the two gap files and admits the planted join of card ADD114 as it is (this queue builds no join);
step 2 runs four rollouts at the convention (pinned readout, no pre-roll, pinned grid, the rule every 10 steps).

  python make_presattrc_queue.py [--src FILE] [--out FILE] [--print]
"""
import argparse
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "presattr", "queue", "pod_presattr10k_queue.sh"))
SRC_SHA16 = "de04e5b848230293"
QM = '"$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"'
C = "READOUT=pinned KAPPA=0.2"
RUNS = [("%s QMAP_FILE=%s VGAP=\"$LAF\" VCLOSE=1" % (C, QM), "I10RQ50Hb20aLA", "RR"),
        ("%s RJOIN=\"$PJ/JLE.npz\"" % C, "I10RLcHb20pLE", "RLc")]
REFS = ["I10RLcHb20", "I10RQ50Hb20", "I10RQ50Hb20gGEc", "I10R0Hb20"]
TAG = "ZZ MAKE-PRESATTRC-QUEUE STOP: "

HEADER = [
    "# pod_presattrc10k_queue.sh -- card ADD115 (PRESATTRC10K): 4 rollouts, model seeds 0 and 1, through the wrapper variant rzgfhqetjv of card ADD102 (hash 28bc31f21ecb4ccb, unchanged),",
    "# at the convention (pinned readout, no pre-roll, pinned grid, the rule every 10 steps):",
    "# LA (I10RQ50Hb20aLA: the reader-fed form RR with --vis-gap-file <VGAPA_LA.npz of card ADD110> and --vis-closing) and PL (I10RLcHb20pLE: the label form RLc with --rule-join <JLE.npz>,",
    "# the planted join of card ADD114 with the camera gap of card ADD106 on the label rows); the scoring keeps the evaluation join. Cells with a label input are attribution arms and never candidates.",
    "# Step 0 gate: exit 4 unless $GE/lead_tests/lead_pytest_queue.out exists (the test of this kit passed in the lead's run before the queue start).",
    "# Step 1: the base gap file (sha256 prefix 55de2f7c) and the LA gap file (the hash in the record of card ADD111) are checked; the join folder of card ADD114 has to exist and is not built here;",
    "# the queue stops unless JLE_RECORD.json has pass_e0 true and the hashes of the join, of the base file and of the evaluation join.",
    "# Step 2: the four rollouts, one after the other, each started only with fewer than 3 GPU compute processes and more than 28 GB free (memory.max minus anon); a run with its T1_DONE is not run again.",
    "# Time rule: before each start the seconds since the first line of $GE/PRESATTRC10K_START.txt must be at most PC_STOP_S (default 7200), else $GE/PRESATTRC10K_STOP.txt. A failed rollout writes the stop marker.",
    "# Done marker $GE/PRESATTRC10K_DONE.txt. Log $GE/presattrc_queue.log. DRY=1 prints every command and creates nothing. RESUME=1 continues after a stop marker. Nothing is removed or moved.",
    "# Banked runs needed (both seeds, T1_DONE): " + " ".join(REFS) + ". Written by make_presattrc_queue.py from the queue of card ADD114 (sha256 prefix %s). No training." % SRC_SHA16,
]

DRY_OLD = '    echo "CMD: [only when $PJ is absent] CUDA_VISIBLE_DEVICES=-1 $PY $KQ/make_planted_join_le.py --kit-gap $KITG --join $JOINE --base $BASEF --out-dir $PJ"\n'
DRY_NEW = '    echo "CHECK: $PJ is the join folder of card ADD114; it has to exist and is not built here"\n'
BUILD_A = '  if [ ! -d "$PJ" ]; then\n    check_time "planted join"\n'
BUILD_B = '  [ -f "$PJ/JLE_RECORD.json" ] || stop "NO-JOIN-RECORD"'
BUILD_NEW = '  [ -d "$PJ" ] || stop "NO-JOIN-FOLDER-OF-CARD-ADD114 $PJ"\n'


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
    if not (lines[0] == "#!/usr/bin/env bash" and lines[12] == "set -u" and all(x.startswith("#") for x in lines[1:12])):
        raise SystemExit(TAG + "the head of the source is not the expected one")
    state["t"] = "\n".join([lines[0]] + HEADER + lines[12:])
    once("GE=${PA_ROOT:-/workspace/v8proof/presattr}", "GE=${PC_ROOT:-/workspace/v8proof/presattrc}")
    once("KQ=${PA_KIT:-$GE/kit_queue}", "KQ=${PC_KIT:-$GE/kit_queue}")
    once("PJ=$GE/joins\n", "PJ=${PC_JOINS:-/workspace/v8proof/presattr/joins}\n")
    once("SM=$GP/small_pa", "SM=$GP/small_pc")
    once("STOPS=${PA_STOP_S:-10800}", "STOPS=${PC_STOP_S:-7200}")
    once("LOG=$GP/presattr_queue.log", "LOG=$GP/presattrc_queue.log")
    line_from('log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD114',
              'log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD115 (PRESATTRC10K): LA (reader presence, label gap) and PL (label presence, camera gap through the '
              'planted join of card ADD114) at the convention, model seeds ${ARMS[*]}"')
    once("step_inputs() { # step 1: the two gap files as they are, and the planted join (made on the CPU when $PJ is absent)",
         "step_inputs() { # step 1: the two gap files as they are, and the planted join of card ADD114 as it is")
    once(DRY_OLD, DRY_NEW)
    span(BUILD_A, BUILD_B, BUILD_NEW)
    once('[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"', '[ "$DRY" = 1 ] || [ -f "$GRID" ] || stop "NO-PINNED-GRID"')
    line_from("  for nm in I10RLcKF5Hb20 ", "  for nm in %s; do" % " ".join(REFS))
    blocks = ['for k in "${ARMS[@]}"; do\n  %s rollout %s %s "$k" "$k"\ndone\n' % (env, name, form) for env, name, form in RUNS]
    span('for k in "${ARMS[@]}"; do\n  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE=', 'if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi', "".join(blocks))
    once('NAMES="I10RQ50KF5Hb20aLA I10RLcKF5Hb20pLE I10RLcKF5P30Hb20pLE"', 'NAMES="%s"' % " ".join(n for _, n, _ in RUNS))
    t = state["t"].replace("PRESATTR10K", "PRESATTRC10K")
    for bad in ("KF5", "P30Hb20", "PA_ROOT", "PA_KIT", "PA_STOP_S", "small_pa", "presattr_queue.log", "EVERY=5", "make_planted_join_le.py", "ADD114 (PRESATTR", "causal_g ACCGAIN"):
        if bad in t:
            raise SystemExit(TAG + "left over: " + bad)
    if t.count(" rollout I10") != 2 or t.count('RJOIN="$PJ/JLE.npz"') != 1 or t.count('VGAP="$LAF" VCLOSE=1') != 1 or t.count("PRESATTRC10K_START.txt") < 4:
        raise SystemExit(TAG + "counts")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(HERE, "pod_presattrc10k_queue.sh"))
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
    print("ZZ MAKE-PRESATTRC-QUEUE DONE %s sha256 prefix %s" % (os.path.basename(a.out), hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
