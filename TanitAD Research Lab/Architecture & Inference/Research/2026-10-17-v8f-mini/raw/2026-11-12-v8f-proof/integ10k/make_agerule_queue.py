#!/usr/bin/env python
"""make_agerule_queue.py -- writes pod_agerule10k_queue.sh (card PREREG_V8F_PROOF_ADD71, part A) from the queue of card ADD70 (pod_longroll10k_queue.sh, sha256 prefix 9e0e935d187729b4).

The helper functions, the start checks and the rollout function are taken line for line from that queue (it ran the 23 rollouts of card ADD70); only the marker names, the log name, the
small folder and the list of rollouts are new. No wait for another queue. Usage: python make_agerule_queue.py  (run in integ10k/).
"""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "pod_longroll10k_queue.sh")
OUT = os.path.join(HERE, "pod_agerule10k_queue.sh")
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != "9e0e935d187729b4":
    sys.exit("REFUSED: the source queue is not the one of card ADD70")
L = raw.decode("utf-8").split("\n")


def rng(a, b):                                                                               # 1-based, inclusive
    return L[a - 1:b]


assert L[14].startswith("set -u") and L[79].startswith("grid_counts()") and L[85] == "}", "anchors of the helper block"
assert L[88].startswith('if [ "$DRY" != 1 ]; then') and L[92] == "fi", "anchors of the start block"
assert L[94].startswith('if [ "$DRY" != 1 ]; then') and L[102].startswith('  for k in "${ARMS[@]}"; do') and L[105] == "fi", "anchors of the banked-run check"
assert L[106].startswith("for f in cmp_dumps.py") and L[119] == "fi", "anchors of the input checks"
assert L[142].startswith("rollout() {") and L[183] == "}", "anchors of the rollout function"
head = ["#!/usr/bin/env bash",
        "# pod_agerule10k_queue.sh -- card PREREG_V8F_PROOF_ADD71 (AGERULE10K), part A: six rollouts at a rollout age of 6 to 12 s on the P = 30 grid of card ADD70 (pre-roll 30 steps,",
        "# 2.0 s cycle, wrapper of record of card ADD70, --kappa-max 0.2, model seeds 0 and 1): I10RLcP30Hb20, I10RQ50P30Hb20 (pinned readout, rule at every replan) and I10R0C10XP30Hb20",
        "# (causal readout on the executed path with the own acceleration). Written by make_agerule_queue.py from the queue of card ADD70: helper functions, start checks and the rollout",
        "# function are its lines. No training, no map fit, no wait for another queue. One GPU job at a time, a slot wait before every rollout. No pgrep, no pkill, no rm, no mv, no kill.",
        "# Markers START_AGERULE10K.txt / AGERULE10K_DONE.txt / AGERULE10K_STOP.txt in $W; step markers ro_<arm> in $W/markers. DRY=1 prints every command and runs nothing."]
body = rng(15, 86) + [""] + rng(89, 93) + ['log "queue start (DRY=$DRY): card ADD71 (AGERULE10K), part A: six rollouts with a pre-roll of 30 steps on the P = 30 grid"'] + [L[94]] + rng(103, 106) + rng(107, 120) + [""] + rng(141, 184)
tail = ["",
        '[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"',
        'if [ "$DRY" != 1 ]; then',
        "  for nm in I10RLcKF5P30Hb20 I10RQ50KF5P30Hb20 I10R0KP30Hb20 I10R0P30Hb20 I10RLcHb20 I10RQ50Hb20 I10R0C10XHb20 I10R0KHb20 I10RLcKF5Hb20 I10RQ50KF5Hb20; do",
        '    for k in "${ARMS[@]}"; do [ -d "$W/t1/${nm}_s$k/dump_${nm}_s${k}_m10_causal_heads" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done',
        "  done",
        "fi",
        'for k in "${ARMS[@]}"; do',
        '  Q="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"',
        '  RGRID=$G30 PREROLL=30 READOUT=pinned KAPPA=0.2 rollout I10RLcP30Hb20 RLc "$k" "$k"',
        '  RGRID=$G30 PREROLL=30 READOUT=pinned KAPPA=0.2 QMAP_FILE="$Q" rollout I10RQ50P30Hb20 RR "$k" "$k"',
        '  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=1 PATHEXEC=1 KAPPA=0.2 rollout I10R0C10XP30Hb20 R0 "$k" "$k"',
        "done",
        'if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi',
        'cp -f "$LOG" "$SM/" 2>/dev/null',
        'for nm in I10RLcP30Hb20 I10RQ50P30Hb20 I10R0C10XP30Hb20; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done',
        'echo "$(ts) all steps done" > "$W/AGERULE10K_DONE.txt"',
        'log "AGERULE10K_DONE"',
        ""]
text = "\n".join(head + body + tail)
for a, b in (("LONGROLL10K", "AGERULE10K"), ("queue_longroll.log", "queue_agerule.log"), ("small_lr", "small_ar")):
    text = text.replace(a, b)
assert "LONGROLL" not in text and "CYCLEBOUND" not in text, "a name of another card is left"
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n")))
