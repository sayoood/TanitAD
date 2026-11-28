#!/usr/bin/env python
"""make_handsrc60_queue.py -- writes pod_handsrc60_queue.sh (card PREREG_V8F_PROOF_ADD74) from the queue of card ADD73 (pod_handsrc10k_queue.sh, sha256 prefix e4f3a446b45aee05).

Every line of that queue is kept but: the header, the start line of the log, the grid and pre-roll of the six rollouts (P = 60 grid, 60 steps), the run names (P60 for P30), the banked
own run, the marker, log and small-folder names, and one added start condition: the read of card ADD73 exists with every control of its part 1 passed.
Usage: python make_handsrc60_queue.py  (run in integ10k/).
"""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "pod_handsrc10k_queue.sh")
OUT = os.path.join(HERE, "pod_handsrc60_queue.sh")
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != "e4f3a446b45aee05":
    sys.exit("REFUSED: the source queue is not the one of card ADD73")
text = raw.decode("utf-8")
assert "\r" not in text
L = text.split("\n")
assert L[0] == "#!/usr/bin/env bash" and L[1].startswith("# pod_handsrc10k_queue.sh") and L[5].startswith("# Markers START_HANDSRC10K.txt") and L[6].startswith("set -u"), "anchors of the header"
head = ["#!/usr/bin/env bash",
        "# pod_handsrc60_queue.sh -- card PREREG_V8F_PROOF_ADD74 (HANDSRC60): six rollouts on the P = 60 grid of card ADD70 with a pre-roll of 60 steps under readout K (form R0,",
        "# --kappa-max 0.2, model seeds 0 and 1) with the wrapper v8f_t1_eval_reader_rule_rzgfhqs.py and --t0-state logged (I10R0KP60LgHb20), speed_logged (I10R0KP60SlHb20),",
        "# path_logged (I10R0KP60PlHb20). Written by make_handsrc60_queue.py from the queue of card ADD73; starts only when the read of card ADD73 passed every control of its part 1.",
        "# No training. One GPU job at a time. No pgrep, no pkill, no rm, no mv, no kill.",
        "# Markers START_HANDSRC60.txt / HANDSRC60_DONE.txt / HANDSRC60_STOP.txt in $W; step markers ro_<arm> in $W/markers. DRY=1 prints every command and runs nothing."]
text = "\n".join(head + L[6:])


def once(a, b, n=1):
    global text
    assert text.count(a) == n, (a, text.count(a))
    text = text.replace(a, b)


once('log "queue start (DRY=$DRY): card ADD73 (HANDSRC10K): six rollouts with --t0-state logged, speed_logged, path_logged on the P = 30 grid"',
     'log "queue start (DRY=$DRY): card ADD74 (HANDSRC60): six rollouts with --t0-state logged, speed_logged, path_logged on the P = 60 grid"')
once('[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"', '[ "$DRY" = 1 ] || [ -f "$G60" ] || stop "NO-GRID-P60"')
once('  [ -f "$CHK" ] && grep -q "^ZZ CHECK-HANDOVER-SRC PASS" "$CHK" || stop "TOOL-CHECK-HANDOVER-SRC-NOT-PASSED"\n',
     '  [ -f "$CHK" ] && grep -q "^ZZ CHECK-HANDOVER-SRC PASS" "$CHK" || stop "TOOL-CHECK-HANDOVER-SRC-NOT-PASSED"\n'
     '  RD73=/workspace/v8proof/podread/out/read_handsrc_decomp_n3/READ_HANDSRC_DECOMP.json\n'
     '  [ -f "$RD73" ] || stop "NO-READ-OF-CARD-ADD73"\n'
     '  "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))[\'controls_all_pass\'] is True else 1)" "$RD73" || stop "READ-OF-CARD-ADD73-PART-1-NOT-PASSED"\n')
once("  for nm in I10R0KHb20 I10R0KP30Hb20 I10R0Hb20; do", "  for nm in I10R0KHb20 I10R0KP60Hb20 I10R0Hb20; do")
once("RGRID=$G30 PREROLL=30 ", "RGRID=$G60 PREROLL=60 ", 3)
for a in ("Lg", "Sl", "Pl"):
    text = text.replace("I10R0KP30%sHb20" % a, "I10R0KP60%sHb20" % a)
for a, b in (("HANDSRC10K", "HANDSRC60"), ("queue_handsrc.log", "queue_handsrc60.log"), ("small_hs", "small_hs60")):
    text = text.replace(a, b)
assert "P30" not in text.replace("NO-GRID-P30", "") and "HANDSRC10K" not in text and "$G30 " not in text, "a name of card ADD73 is left"
assert text.count("--t0-state") >= 1 and text.count("PREROLL=60") == 3 and text.count("I10R0KP60") >= 7
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n")))
