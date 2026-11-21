#!/usr/bin/env python
"""make_handsrc_queue.py -- writes pod_handsrc10k_queue.sh (card PREREG_V8F_PROOF_ADD73) from the queue of card ADD71 (pod_agerule10k_queue.sh, sha256 prefix d83227f960a2c96b).

The helper functions, the start checks and the rollout function are taken line for line from that queue; new are the header, the wrapper (the generation with --t0-state), one line of the
rollout function that passes --t0-state, the check that the pod check of the wrapper passed, the marker names, the log name, the small folder and the list of rollouts.
Usage: python make_handsrc_queue.py  (run in integ10k/).
"""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "pod_agerule10k_queue.sh")
OUT = os.path.join(HERE, "pod_handsrc10k_queue.sh")
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != "d83227f960a2c96b":
    sys.exit("REFUSED: the source queue is not the one of card ADD71")
L = raw.decode("utf-8").split("\n")
assert L[0] == "#!/usr/bin/env bash" and L[1].startswith("# pod_agerule10k_queue.sh") and L[5].startswith("# Markers START_AGERULE10K.txt") and L[6].startswith("set -u"), "anchors of the header"
assert L[84].startswith('log "queue start (DRY=$DRY): card ADD71'), "anchor of the start line"
assert L[107].startswith("rollout() {") and L[148] == "}" and L[150].startswith('[ "$DRY" = 1 ] || [ -f "$G30" ]'), "anchors of the rollout function"
head = ["#!/usr/bin/env bash",
        "# pod_handsrc10k_queue.sh -- card PREREG_V8F_PROOF_ADD73 (HANDSRC10K): six rollouts on the P = 30 grid of card ADD70 with a pre-roll of 30 steps under readout K (form R0,",
        "# --kappa-max 0.2, model seeds 0 and 1) with the wrapper generation v8f_t1_eval_reader_rule_rzgfhqs.py and --t0-state logged (I10R0KP30LgHb20), speed_logged (I10R0KP30SlHb20),",
        "# path_logged (I10R0KP30PlHb20). Written by make_handsrc_queue.py from the queue of card ADD71: helper functions, start checks and the rollout function are its lines, plus one",
        "# line that passes --t0-state. Starts only when the pod check of the wrapper passed. No training. One GPU job at a time. No pgrep, no pkill, no rm, no mv, no kill.",
        "# Markers START_HANDSRC10K.txt / HANDSRC10K_DONE.txt / HANDSRC10K_STOP.txt in $W; step markers ro_<arm> in $W/markers. DRY=1 prints every command and runs nothing."]
body = L[6:84] + ['log "queue start (DRY=$DRY): card ADD73 (HANDSRC10K): six rollouts with --t0-state logged, speed_logged, path_logged on the P = 30 grid"'] + L[85:149]
text = "\n".join(body)


def once(a, b):
    global text
    assert text.count(a) == 1, (a, text.count(a))
    text = text.replace(a, b)


once("WRAP=$B/v8f_t1_eval_reader_rule_rzgfhq.py", "WRAP=$B/v8f_t1_eval_reader_rule_rzgfhqs.py")
once("for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzgfhq.py ", "for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzgfhqs.py v8f_t1_eval_reader_rule_rzgfhq.py ")
once('  extra+=(--preroll-steps "$pre")\n', '  extra+=(--preroll-steps "$pre")\n  [ -n "${T0STATE:-}" ] && extra+=(--t0-state "$T0STATE")\n')
tail = ["",
        '[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"',
        'CHK="$W/handsrc_test/CHECK_DONE_${CHECKTAG:-chk1}.txt"',
        'if [ "$DRY" != 1 ]; then',
        '  [ -f "$CHK" ] && grep -q "^ZZ CHECK-HANDOVER-SRC PASS" "$CHK" || stop "TOOL-CHECK-HANDOVER-SRC-NOT-PASSED"',
        "  for nm in I10R0KHb20 I10R0KP30Hb20 I10R0Hb20; do",
        '    for k in "${ARMS[@]}"; do [ -d "$W/t1/${nm}_s$k/dump_${nm}_s${k}_m10_causal_heads" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done',
        "  done",
        "fi",
        'for k in "${ARMS[@]}"; do',
        '  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 T0STATE=logged rollout I10R0KP30LgHb20 R0 "$k" "$k"',
        "done",
        'for k in "${ARMS[@]}"; do',
        '  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 T0STATE=speed_logged rollout I10R0KP30SlHb20 R0 "$k" "$k"',
        "done",
        'for k in "${ARMS[@]}"; do',
        '  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 T0STATE=path_logged rollout I10R0KP30PlHb20 R0 "$k" "$k"',
        "done",
        'if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi',
        'cp -f "$LOG" "$SM/" 2>/dev/null',
        'for nm in I10R0KP30LgHb20 I10R0KP30SlHb20 I10R0KP30PlHb20; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done',
        'echo "$(ts) all steps done" > "$W/HANDSRC10K_DONE.txt"',
        'log "HANDSRC10K_DONE"',
        ""]
text = "\n".join(head) + "\n" + text + "\n" + "\n".join(tail)
for a, b in (("AGERULE10K", "HANDSRC10K"), ("queue_agerule.log", "queue_handsrc.log"), ("small_ar", "small_hs")):
    text = text.replace(a, b)
assert "AGERULE" not in text and "LONGROLL" not in text and "agerule" not in text, "a name of another card is left"
assert "\r" not in text
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n")))
