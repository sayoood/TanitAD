#!/usr/bin/env python
"""make_candset_queue.py -- card PREREG_V8F_PROOF_ADD113 (CANDSET10K). Writes pod_candset10k_queue.sh from the queue of card ADD111
(gaperr/gap/pod_gaperr10k_queue.sh, sha256 prefix 477923be1acf0804) by asserted substitutions. The rollout function, the slot wait, the
stop rule and the marker line formats are kept as they are; the gap-file build step is taken out (the base gap file of card ADD106 is
used as it is) and the eight rollouts are replaced by six: the candidate form at the settings CONV, K0 and K60 of the card.

  python make_candset_queue.py [--src FILE] [--out FILE] [--print]
"""
import argparse
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "gaperr", "gap", "pod_gaperr10k_queue.sh"))
SRC_SHA16 = "477923be1acf0804"
QM = '"$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"'
RUNS = [("CONV", 'READOUT=pinned KAPPA=0.2', "I10RQ50Hb20gGEc"),
        ("K0", 'READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5', "I10RQ50KF5Hb20gGEc"),
        ("K60", 'RGRID=$G60 PREROLL=60 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5', "I10RQ50KF5P60Hb20gGEc")]
REFS = ["I10RQ50Hb20", "I10RLcHb20", "I10R0Hb20", "I10RQ50KF5Hb20", "I10RLcKF5Hb20", "I10R0KHb20", "I10RQ50KF5P60Hb20", "I10RLcKF5P60Hb20",
        "I10R0KP60Hb20", "I10RQ50KF5P30Hb20gGEc", "I10RQ50KF5P30Hb20", "I10RLcKF5P30Hb20"]

HEADER = [
    "# pod_candset10k_queue.sh -- card ADD113 (CANDSET10K): 6 rollouts of the candidate form (reader-fed form RR through the wrapper variant rzgfhqetjv of card ADD102,",
    "# hash 28bc31f21ecb4ccb, unchanged, with --vis-gap-file <base gap file of card ADD106> and --vis-closing, as the banked run I10RQ50KF5P30Hb20gGEc) at three settings,",
    "# model seeds 0 and 1: CONV (no pre-roll, pinned readout; I10RQ50Hb20gGEc), K0 (no pre-roll, readout K, --rule-every 5; I10RQ50KF5Hb20gGEc) and K60 (pre-roll 60 steps = 12 s,",
    "# grid_p60, readout K, --rule-every 5; I10RQ50KF5P60Hb20gGEc). Every other argument is the one of the banked reference run RR of the setting. No label enters at inference.",
    "# Step 0 gate: exit 4 unless $GE/lead_tests/lead_pytest_queue.out exists (the test of this kit passed in the lead's run before the queue start).",
    "# Step 1: the base gap file must have the sha256 prefix 55de2f7c; no gap file is built.",
    "# Step 2: the six rollouts, one after the other, each started only with fewer than 3 GPU compute processes and more than 28 GB free (memory.max minus anon); a run with its T1_DONE is not run again.",
    "# Time rule: before each start the seconds since the first line of $GE/CANDSET10K_START.txt must be at most CS_STOP_S (default 10800), else $GE/CANDSET10K_STOP.txt. A failed rollout writes the stop marker.",
    "# Done marker $GE/CANDSET10K_DONE.txt. Log $GE/candset_queue.log. DRY=1 prints every command and creates nothing. RESUME=1 continues after a stop marker. Nothing is removed or moved.",
    "# Banked runs needed (both seeds, T1_DONE): " + " ".join(REFS) + ".",
    "# Written by make_candset_queue.py from the queue of card ADD111 (sha256 prefix %s). No training." % SRC_SHA16,
]


def build(src_text):
    t = src_text.replace("\r\n", "\n")
    state = {"t": t}

    def once(a, b):
        if state["t"].count(a) != 1:
            raise SystemExit("ZZ MAKE-CANDSET-QUEUE STOP: expected once, found %d: %r" % (state["t"].count(a), a[:90]))
        state["t"] = state["t"].replace(a, b)

    def between(a, b, new):
        s = state["t"]
        if s.count(a) != 1 or s.count(b) != 1:
            raise SystemExit("ZZ MAKE-CANDSET-QUEUE STOP: anchors not found once: %r / %r" % (a[:60], b[:60]))
        i, j = s.index(a), s.index(b)
        if not i < j:
            raise SystemExit("ZZ MAKE-CANDSET-QUEUE STOP: anchors out of order")
        state["t"] = s[:i] + new + s[j:]

    lines = state["t"].split("\n")
    if not (lines[0] == "#!/usr/bin/env bash" and lines[11] == "set -u" and all(x.startswith("#") for x in lines[1:11])):
        raise SystemExit("ZZ MAKE-CANDSET-QUEUE STOP: the head of the source is not the expected one")
    state["t"] = "\n".join([lines[0]] + HEADER + lines[11:])

    once("GE=${GE_ROOT:-/workspace/v8proof/gaperr}", "GE=${CS_ROOT:-/workspace/v8proof/candset}")
    once("SM=$GP/small_ge", "SM=$GP/small_cs")
    once("STOPS=${GE_STOP_S:-10800}", "STOPS=${CS_STOP_S:-10800}")
    once("LOG=$GP/gaperr_queue.log", "LOG=$GP/candset_queue.log")
    # gate
    once("# step 0 gate: the tests of both kits passed in the lead's run (control a4); nothing is created before this point",
         "# step 0 gate: the test of this kit passed in the lead's run (control d4); nothing is created before this point")
    once('gate_ok() { [ -f "$KG/lead_pytest.out" ] && [ -f "$KR/lead_pytest.out" ]; }', 'gate_ok() { [ -f "$GE/lead_tests/lead_pytest_queue.out" ]; }')
    once('  echo "GATE: needs $KG/lead_pytest.out and $KR/lead_pytest.out (exit 4 without them)"', '  echo "GATE: needs $GE/lead_tests/lead_pytest_queue.out (exit 4 without it)"')
    once('  gate_ok || { echo "REFUSED: step 0 gate: $KG/lead_pytest.out or $KR/lead_pytest.out is missing"; exit 4; }',
         '  gate_ok || { echo "REFUSED: step 0 gate: $GE/lead_tests/lead_pytest_queue.out is missing"; exit 4; }')
    # the start line of the log
    s = state["t"]
    a = 'log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD111'
    if s.count(a) != 1:
        raise SystemExit("ZZ MAKE-CANDSET-QUEUE STOP: log start line")
    i = s.index(a)
    j = s.index("\n", i)
    state["t"] = (s[:i] + 'log "queue start (DRY=$DRY, resume $RESUME, stop rule $STOPS s): card ADD113 (CANDSET10K): the candidate form (reader-fed rule RR, base gap file of card ADD106, '
                  'closing input of the run rule) at the settings CONV, K0 and K60, model seeds ${ARMS[*]}"' + s[j:])
    # the gap-file step goes out; the base file is checked instead
    between("admit() {", '[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"',
            'step_base() { # step 1: the base gap file of card ADD106 as it is\n'
            '  if [ "$DRY" = 1 ]; then echo "CHECK: sha256 of $BASEF begins 55de2f7c"; return 0; fi\n'
            '  [ -f "$BASEF" ] || stop "NO-BASE-GAP-FILE $BASEF"\n'
            '  local got; got=$(sha256sum "$BASEF" | cut -c1-8); [ "$got" = "55de2f7c" ] || stop "BASE-GAP-FILE-SHA $got"\n'
            '  log "base gap file sha256 prefix $got"\n'
            '}\n')
    once('[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"', '[ "$DRY" = 1 ] || [ -f "$G60" ] || stop "NO-GRID-P60"')
    between('  for nm in I10RLcKF5P30Hb20 I10RQ50KF5P30Hb20; do', "step_gapfiles\n",
            '  for nm in %s; do\n    for k in "${ARMS[@]}"; do [ -f "$W/t1/${nm}_s$k/T1_DONE" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done\n  done\nfi\n' % " ".join(REFS))
    once("step_gapfiles\n", "step_base\n")
    blocks = ['for k in "${ARMS[@]}"; do\n  %s QMAP_FILE=%s VGAP="$BASEF" VCLOSE=1 rollout %s RR "$k" "$k"\ndone\n' % (env, QM, name) for _, env, name in RUNS]
    s = state["t"]
    a4, b4 = 'for k in "${ARMS[@]}"; do\n  RGRID=$G30 PREROLL=30', 'if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi'
    if s.count(a4) != 4 or s.count(b4) != 1 or not s.index(a4) < s.index(b4):
        raise SystemExit("ZZ MAKE-CANDSET-QUEUE STOP: the four rollout loops of the source were not found")
    state["t"] = s[:s.index(a4)] + "".join(blocks) + s[s.index(b4):]
    once('NAMES="I10RQ50KF5P30Hb20bNR I10RQ50KF5P30Hb20bBR I10RQ50KF5P30Hb20bSM I10RQ50KF5P30Hb20bSL"', 'NAMES="%s"' % " ".join(n for _, _, n in RUNS))
    t = state["t"].replace("GAPERR10K", "CANDSET10K")
    for bad in ("GAPERR", "gaperr", "make_gaperr", "VGAPA_", "$KG", "$KR", "$LAF", "step_gapfiles", "admit"):
        if bad in t.replace("KG=$GE/kit_gap", "").replace("KR=$GE/kit_read", "").replace("LAF=/workspace/v8proof/fallattr/gapfiles/VGAPA_LA.npz", ""):
            raise SystemExit("ZZ MAKE-CANDSET-QUEUE STOP: left over: " + bad)
    if t.count(" rollout I10") != 3 or t.count('VGAP="$BASEF" VCLOSE=1') != 3 or t.count("CANDSET10K_START.txt") < 4:
        raise SystemExit("ZZ MAKE-CANDSET-QUEUE STOP: counts")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(HERE, "pod_candset10k_queue.sh"))
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args(argv)
    raw = io.open(a.src, "rb").read().replace(b"\r\n", b"\n")
    got = hashlib.sha256(raw).hexdigest()[:16]
    if got != SRC_SHA16:
        raise SystemExit("ZZ MAKE-CANDSET-QUEUE STOP: source sha256 prefix %s, expected %s" % (got, SRC_SHA16))
    t = build(raw.decode("utf-8"))
    if a.print:
        sys.stdout.write(t)
        return 0
    with io.open(a.out, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(t)
    print("ZZ MAKE-CANDSET-QUEUE DONE %s sha256 prefix %s" % (os.path.basename(a.out), hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
