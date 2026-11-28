#!/usr/bin/env python
"""make_closrc_queue.py -- writes pod_closrc10k_queue.sh (card PREREG_V8F_PROOF_ADD75) from the queue of card ADD71 (pod_agerule10k_queue.sh, sha256 prefix d83227f960a2c96b).

The helper functions, the start checks and the rollout function are taken line for line from that queue (wrapper of record v8f_t1_eval_reader_rule_rzgfhq.py); new are the header, the
start line of the log, two lines of the rollout function (the forms RRc and RRh get the reader and the map as RR does), the marker names, the log name, the small folder and the list
of rollouts.
Usage: python make_closrc_queue.py  (run in integ10k/).
"""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "pod_agerule10k_queue.sh")
OUT = os.path.join(HERE, "pod_closrc10k_queue.sh")
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != "d83227f960a2c96b":
    sys.exit("REFUSED: the source queue is not the one of card ADD71")
L = raw.decode("utf-8").split("\n")
assert L[0] == "#!/usr/bin/env bash" and L[1].startswith("# pod_agerule10k_queue.sh") and L[5].startswith("# Markers START_AGERULE10K.txt") and L[6].startswith("set -u"), "anchors of the header"
assert L[84].startswith('log "queue start (DRY=$DRY): card ADD71'), "anchor of the start line"
assert L[107].startswith("rollout() {") and L[148] == "}" and L[150].startswith('[ "$DRY" = 1 ] || [ -f "$G30" ]'), "anchors of the rollout function"
head = ["#!/usr/bin/env bash",
        "# pod_closrc10k_queue.sh -- card PREREG_V8F_PROOF_ADD75 (CLOSRC10K): ten rollouts on the P = 30 grid of card ADD70 with a pre-roll of 30 steps under readout K (--kappa-max 0.2,",
        "# the rule at the replan ticks, model seeds 0 and 1, q50 map) in the forms RR (I10RQ50KP30Hb20), RRc (I10RQ50cKP30Hb20), RRh (I10RQ50hKP30Hb20), RLb (I10RLbKP30Hb20) and",
        "# RLc (I10RLcKP30Hb20), wrapper of record v8f_t1_eval_reader_rule_rzgfhq.py. Written by make_closrc_queue.py from the queue of card ADD71.",
        "# No training. No pgrep, no pkill, no rm, no mv, no kill.",
        "# Markers START_CLOSRC10K.txt / CLOSRC10K_DONE.txt / CLOSRC10K_STOP.txt in $W; step markers ro_<arm> in $W/markers. DRY=1 prints every command and runs nothing."]
body = L[6:84] + ['log "queue start (DRY=$DRY): card ADD75 (CLOSRC10K): ten rollouts at age under readout K in the forms RR, RRc, RRh, RLb, RLc"'] + L[85:149]
text = "\n".join(body)


def once(a, b):
    global text
    assert text.count(a) == 1, (a, text.count(a))
    text = text.replace(a, b)


once('    RR) extra+=(--reader "$W/lr16/TK16_s$r/ckpt_last.pt") ;;', '    RR|RRc|RRh) extra+=(--reader "$W/lr16/TK16_s$r/ckpt_last.pt") ;;')
once('  case "$form" in RR) extra+=(--qmap "${QMAP_FILE:?}") ;; esac', '  case "$form" in RR|RRc|RRh) extra+=(--qmap "${QMAP_FILE:?}") ;; esac')
K = 'RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2'
tail = ["",
        '[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"',
        'if [ "$DRY" != 1 ]; then',
        "  for nm in I10R0KHb20 I10R0KP30Hb20 I10R0Hb20; do",
        '    for k in "${ARMS[@]}"; do [ -d "$W/t1/${nm}_s$k/dump_${nm}_s${k}_m10_causal_heads" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done',
        "  done",
        "fi",
        'for k in "${ARMS[@]}"; do',
        '  Q="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"',
        '  [ "$DRY" = 1 ] || [ -f "$Q" ] || stop "NO-MAP-Q50 s$k"',
        "  " + K + ' QMAP_FILE="$Q" rollout I10RQ50KP30Hb20 RR "$k" "$k"',
        "  " + K + ' rollout I10RLcKP30Hb20 RLc "$k" "$k"',
        "done",
        'for k in "${ARMS[@]}"; do',
        '  Q="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"',
        "  " + K + ' QMAP_FILE="$Q" rollout I10RQ50cKP30Hb20 RRc "$k" "$k"',
        "  " + K + ' rollout I10RLbKP30Hb20 RLb "$k" "$k"',
        "done",
        'for k in "${ARMS[@]}"; do',
        '  Q="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"',
        "  " + K + ' QMAP_FILE="$Q" rollout I10RQ50hKP30Hb20 RRh "$k" "$k"',
        "done",
        'if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi',
        'cp -f "$LOG" "$SM/" 2>/dev/null',
        'for nm in I10RQ50KP30Hb20 I10RQ50cKP30Hb20 I10RQ50hKP30Hb20 I10RLbKP30Hb20 I10RLcKP30Hb20; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done',
        'echo "$(ts) all steps done" > "$W/CLOSRC10K_DONE.txt"',
        'log "CLOSRC10K_DONE"',
        ""]
text = "\n".join(head) + "\n" + text + "\n" + "\n".join(tail)
for a, b in (("AGERULE10K", "CLOSRC10K"), ("queue_agerule.log", "queue_closrc.log"), ("small_ar", "small_cl")):
    text = text.replace(a, b)
assert "AGERULE" not in text and "LONGROLL" not in text and "agerule" not in text, "a name of another card is left"
assert "\r" not in text
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n")))
