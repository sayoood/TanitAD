#!/usr/bin/env python
"""make_agedose_queue.py -- writes pod_agedose10k_queue.sh (card PREREG_V8F_PROOF_ADD72) from the queue of card ADD71 (pod_agerule10k_queue.sh, sha256 prefix d83227f960a2c96b).

The helper functions, the start checks and the rollout function are taken line for line from that queue (it ran the six rollouts of card ADD71); only the header, the marker names,
the log name, the small folder and the list of rollouts are new. No wait for another queue. Usage: python make_agedose_queue.py  (run in integ10k/).
"""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "pod_agerule10k_queue.sh")
OUT = os.path.join(HERE, "pod_agedose10k_queue.sh")
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != "d83227f960a2c96b":
    sys.exit("REFUSED: the source queue is not the one of card ADD71")
L = raw.decode("utf-8").split("\n")
assert L[0] == "#!/usr/bin/env bash" and L[1].startswith("# pod_agerule10k_queue.sh") and L[5].startswith("# Markers START_AGERULE10K.txt") and L[6].startswith("set -u"), "anchors of the header"
assert L[84].startswith('log "queue start (DRY=$DRY): card ADD71'), "anchor of the start line"
assert L[107].startswith("rollout() {") and L[148] == "}" and L[150].startswith('[ "$DRY" = 1 ] || [ -f "$G30" ]'), "anchors of the rollout function"
head = ["#!/usr/bin/env bash",
        "# pod_agedose10k_queue.sh -- card PREREG_V8F_PROOF_ADD72 (AGEDOSE10K): twelve rollouts on the P = 30 grid of card ADD70 with a pre-roll of 5, 10 and 20 steps (2.0 s cycle,",
        "# wrapper of record of card ADD70, --kappa-max 0.2, model seeds 0 and 1): first I10R0KP5Hb20, I10R0KP10Hb20, I10R0KP20Hb20 (readout K), then I10RLcKF5P5Hb20, I10RLcKF5P10Hb20,",
        "# I10RLcKF5P20Hb20 (label-fed rule with its second look under readout K). Written by make_agedose_queue.py from the queue of card ADD71: helper functions, start checks and the",
        "# rollout function are its lines. No training, no map fit, no wait for another queue. One GPU job at a time, a slot wait before every rollout. No pgrep, no pkill, no rm, no mv, no kill.",
        "# Markers START_AGEDOSE10K.txt / AGEDOSE10K_DONE.txt / AGEDOSE10K_STOP.txt in $W; step markers ro_<arm> in $W/markers. DRY=1 prints every command and runs nothing."]
body = L[6:84] + ['log "queue start (DRY=$DRY): card ADD72 (AGEDOSE10K): twelve rollouts with a pre-roll of 5, 10 and 20 steps on the P = 30 grid"'] + L[85:149]
tail = ["",
        '[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"',
        'if [ "$DRY" != 1 ]; then',
        "  for nm in I10R0KHb20 I10R0KP30Hb20 I10RLcKF5Hb20 I10RLcKF5P30Hb20 I10R0Hb20; do",
        '    for k in "${ARMS[@]}"; do [ -d "$W/t1/${nm}_s$k/dump_${nm}_s${k}_m10_causal_heads" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done',
        "  done",
        "fi",
        'for p in 5 10 20; do for k in "${ARMS[@]}"; do',
        '  RGRID=$G30 PREROLL=$p READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 rollout "I10R0KP${p}Hb20" R0 "$k" "$k"',
        "done; done",
        'for p in 5 10 20; do for k in "${ARMS[@]}"; do',
        '  RGRID=$G30 PREROLL=$p READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 rollout "I10RLcKF5P${p}Hb20" RLc "$k" "$k"',
        "done; done",
        'if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi',
        'cp -f "$LOG" "$SM/" 2>/dev/null',
        'for p in 5 10 20; do for nm in "I10R0KP${p}Hb20" "I10RLcKF5P${p}Hb20"; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done; done',
        'echo "$(ts) all steps done" > "$W/AGEDOSE10K_DONE.txt"',
        'log "AGEDOSE10K_DONE"',
        ""]
text = "\n".join(head + body + tail)
for a, b in (("AGERULE10K", "AGEDOSE10K"), ("queue_agerule.log", "queue_agedose.log"), ("small_ar", "small_ad")):
    text = text.replace(a, b)
assert "AGERULE" not in text and "LONGROLL" not in text and "agerule" not in text, "a name of another card is left"
assert "\r" not in text
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n")))
