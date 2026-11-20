#!/usr/bin/env python
"""make_midlook_queue.py -- writes pod_midlook10k_queue.sh from the queue of card ADD75 (pod_closrc10k_queue.sh, sha256 prefix a7dd037c306f4315).

The helper functions, the start checks and the rollout function are taken line for line from that queue; new are the header, the wrapper (the second-look generation
v8f_t1_eval_reader_rule_rzgfhqc.py, admitted only with its pod check in the form of record: the marker midlook_test/CHECK_DONE_chk1n1.txt has to say PASS and the wrapper has to carry
the checked hash), the start line of the log, the marker names, the log name, the small folder and the list of rollouts: for every form given, both model seeds, with --rule-every 5
under readout K (--acc-gain 0 --path-exec --kappa-max 0.2, q50 map), once at age (P = 30 grid, pre-roll 30) and once fresh (pinned grid, no pre-roll).
Usage: python make_midlook_queue.py CARD FORM[,FORM...]   with forms out of RRc, RRh   (run in integ10k/).
"""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "pod_closrc10k_queue.sh")
OUT = os.path.join(HERE, "pod_midlook10k_queue.sh")
WRAP_SHA = "8732329b4476347d"
NAME = {"RRc": "I10RQ50cKF5", "RRh": "I10RQ50hKF5"}
if len(sys.argv) != 3:
    sys.exit("usage: make_midlook_queue.py CARD FORM[,FORM...]")
CARD, FORMS = sys.argv[1], sys.argv[2].split(",")
if not FORMS or any(f not in NAME for f in FORMS) or len(set(FORMS)) != len(FORMS):
    sys.exit("REFUSED: forms out of RRc, RRh, each once")
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != "a7dd037c306f4315":
    sys.exit("REFUSED: the source queue is not the one of card ADD75")
if hashlib.sha256(io.open(os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqc.py"), "rb").read()).hexdigest()[:16] != WRAP_SHA:
    sys.exit("REFUSED: the local wrapper is not the checked one")
L = raw.decode("utf-8").split("\n")
i_set = L.index("set -u")
i_roll_end = max(i for i, x in enumerate(L) if x == "}")
i_tail = next(i for i, x in enumerate(L) if x.startswith('[ "$DRY" = 1 ] || [ -f "$G30" ]'))
assert L[0] == "#!/usr/bin/env bash" and i_set == 6 and i_tail == i_roll_end + 2, "anchors"
text = "\n".join(L[i_set:i_roll_end + 1])


def once(a, b):
    global text
    assert text.count(a) == 1, (a, text.count(a))
    text = text.replace(a, b)


once("WRAP=$B/v8f_t1_eval_reader_rule_rzgfhq.py", "WRAP=$B/v8f_t1_eval_reader_rule_rzgfhqc.py")
once('log "queue start (DRY=$DRY): card ADD75 (CLOSRC10K): ten rollouts at age under readout K in the forms RR, RRc, RRh, RLb, RLc"',
     'log "queue start (DRY=$DRY): card %s (MIDLOOK10K): %d rollouts with the second look under readout K in the forms %s, at age and fresh"' % (CARD, 4 * len(FORMS), ", ".join(FORMS)))
once("for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzgfhq.py ", "for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzgfhqc.py v8f_t1_eval_reader_rule_rzgfhq.py ")
head = ["#!/usr/bin/env bash",
        "# pod_midlook10k_queue.sh -- card %s (MIDLOOK10K): %d rollouts with --rule-every 5 under readout K (--acc-gain 0 --path-exec --kappa-max 0.2, model seeds 0 and 1," % (CARD, 4 * len(FORMS)),
        "# q50 map) in the forms %s, each at age (P = 30 grid of card ADD70, pre-roll 30) and fresh (pinned grid), wrapper v8f_t1_eval_reader_rule_rzgfhqc.py" % ", ".join(FORMS),
        "# (second-look generation; admitted only with its pod check in the form of record). Written by make_midlook_queue.py from the queue of card ADD75.",
        "# No training. No pgrep, no pkill, no rm, no mv, no kill.",
        "# Markers START_MIDLOOK10K.txt / MIDLOOK10K_DONE.txt / MIDLOOK10K_STOP.txt in $W; step markers ro_<arm> in $W/markers. DRY=1 prints every command and runs nothing."]
K = 'READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5'
names = []
tail = ["",
        '[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"',
        'if [ "$DRY" != 1 ]; then',
        '  grep -q "^ZZ CHECK-MIDLOOK-CLOSING PASS" "$W/midlook_test/CHECK_DONE_chk1n1.txt" 2>/dev/null || stop "NO-PASS-OF-THE-WRAPPER-CHECK"',
        '  got=$(sha256sum "$WRAP" | cut -c1-16); [ "$got" = "%s" ] || stop "WRAPPER-SHA $got"' % WRAP_SHA,
        "  for nm in I10RQ50KF5P30Hb20 I10RLcKF5P30Hb20 I10RQ50KF5Hb20 I10RLcKF5Hb20 I10R0KP30Hb20 I10R0KHb20 I10R0Hb20; do",
        '    for k in "${ARMS[@]}"; do [ -d "$W/t1/${nm}_s$k/dump_${nm}_s${k}_m10_causal_heads" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done',
        "  done",
        "fi"]
for f in FORMS:
    tail += ['for k in "${ARMS[@]}"; do',
             '  Q="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"',
             "  RGRID=$G30 PREROLL=30 " + K + ' QMAP_FILE="$Q" rollout %sP30Hb20 %s "$k" "$k"' % (NAME[f], f),
             "done"]
    names.append(NAME[f] + "P30Hb20")
for f in FORMS:
    tail += ['for k in "${ARMS[@]}"; do',
             '  Q="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"',
             "  " + K + ' QMAP_FILE="$Q" rollout %sHb20 %s "$k" "$k"' % (NAME[f], f),
             "done"]
    names.append(NAME[f] + "Hb20")
tail += ['if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi',
         'cp -f "$LOG" "$SM/" 2>/dev/null',
         'for nm in %s; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done' % " ".join(names),
         'echo "$(ts) all steps done" > "$W/MIDLOOK10K_DONE.txt"',
         'log "MIDLOOK10K_DONE"',
         ""]
text = "\n".join(head) + "\n" + text + "\n" + "\n".join(tail)
for a, b in (("CLOSRC10K", "MIDLOOK10K"), ("queue_closrc.log", "queue_midlook.log"), ("small_cl", "small_ml")):
    text = text.replace(a, b)
assert "CLOSRC" not in text and "closrc" not in text and "AGERULE" not in text, "a name of another card is left"
assert "\r" not in text
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d rollouts %d" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n"), 4 * len(FORMS)))
