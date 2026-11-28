"""make_estclose_queue.py -- writes pod_estclose10k_queue.sh from the queue of card ADD76 (pod_midlook10k_queue.sh, sha256 prefix 4dc50ac3df15dafd).

The helper functions (slot rule: fewer than 3 GPU compute processes and more than 28 GB of container memory free, two probes 45 s apart), the start checks and the rollout function are taken
line for line from that queue; new are the header, the wrapper (the estimate generation v8f_t1_eval_reader_rule_rzgfhqe.py, admitted only with its pod check in the form of record: the marker
estclose_test/CHECK_DONE_chk1.txt has to say PASS, the check record has to carry the wrapper hash, and the wrapper has to have that hash), the marker names, the log name, the small folder, the
estimator files and the list of rollouts of card ADD79 section 3: form RRe, readout K (--acc-gain 0 --path-exec --kappa-max 0.2), --rule-every 5, q50 map, model seeds 0 and 1 (reader seed k,
estimator seed k), at age (P = 30 grid, pre-roll 30) and fresh (pinned grid), estimators N10 and L25:
  I10RQ50eN10KF5P30Hb20  I10RQ50eL25KF5P30Hb20  I10RQ50eN10KF5Hb20  I10RQ50eL25KF5Hb20      (each _s0, _s1)
Usage: python make_estclose_queue.py CARD   (run in estclose/; CARD = the card name written into the header, for example ADD79). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "integ10k", "pod_midlook10k_queue.sh"))
OUT = os.path.join(HERE, "pod_estclose10k_queue.sh")
WRAP_FILE = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqe.py")
SRC_SHA = "4dc50ac3df15dafd"
ESTS = ("N10", "L25")
if len(sys.argv) != 2:
    sys.exit("usage: make_estclose_queue.py CARD")
CARD = sys.argv[1]
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != SRC_SHA:
    sys.exit("REFUSED: the source queue is not the one of card ADD76")
if not os.path.isfile(WRAP_FILE):
    sys.exit("REFUSED: the wrapper %s does not exist (run make_estclose_wrapper.py first)" % WRAP_FILE)
WRAP_SHA = hashlib.sha256(io.open(WRAP_FILE, "rb").read()).hexdigest()[:16]
if os.path.exists(OUT) and os.environ.get("EC_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (set EC_OVERWRITE_OWN=1 to rewrite it)" % OUT)
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


once("WRAP=$B/v8f_t1_eval_reader_rule_rzgfhqc.py", "WRAP=$B/v8f_t1_eval_reader_rule_rzgfhqe.py")
once('log "queue start (DRY=$DRY): card ADD76 (MIDLOOK10K): 8 rollouts with the second look under readout K in the forms RRc, RRh, at age and fresh"',
     'log "queue start (DRY=$DRY): card %s (ESTCLOSE10K): 8 rollouts of form RRe (closing speed estimated from the reader reads) with the second look under readout K, estimators N10 and L25, at age and fresh"' % CARD)
once("for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzgfhqc.py ", "for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzgfhqe.py closest.py v8f_t1_eval_reader_rule_rzgfhqc.py ")
once('    RR|RRc|RRh) extra+=(--reader "$W/lr16/TK16_s$r/ckpt_last.pt") ;;', '    RR|RRc|RRh|RRe) extra+=(--reader "$W/lr16/TK16_s$r/ckpt_last.pt") ;;')
once('  case "$form" in RR|RRc|RRh) extra+=(--qmap "${QMAP_FILE:?}") ;; esac\n',
     '  case "$form" in RR|RRc|RRh|RRe) extra+=(--qmap "${QMAP_FILE:?}") ;; esac\n'
     '  if [ "$form" = RRe ]; then\n'
     '    [ "$DRY" = 1 ] || [ -f "${EST:-}" ] || stop "NO-ESTIMATOR-FILE ${EST:-unset} $arm"\n'
     '    extra+=(--closing-est "${EST:?}")\n'
     '  fi\n')
once('started $(ts)" > "$out/ticks/run_$arm.txt"', 'started $(ts)${EST:+ closing-est $EST est sha256 $(sha256sum "$EST" | cut -c1-64)}" > "$out/ticks/run_$arm.txt"')
head = ["#!/usr/bin/env bash",
        "# pod_estclose10k_queue.sh -- card %s (ESTCLOSE10K): 8 rollouts of form RRe (the closing-speed input of the lead rule is an ESTIMATE computed from the token reader's own reads over the last" % CARD,
        "# H frames; estimator files /workspace/v8proof/closeread/est/closest_<N10|L25>_s<k>.npz), --rule-every 5 under readout K (--acc-gain 0 --path-exec --kappa-max 0.2), q50 map, model seeds 0 and 1",
        "# (reader seed k, estimator seed k), each at age (P = 30 grid of card ADD70, pre-roll 30) and fresh (pinned grid); wrapper v8f_t1_eval_reader_rule_rzgfhqe.py (estimate generation, hash %s)," % WRAP_SHA,
        "# admitted only with its pod check in the form of record (estclose_test/CHECK_DONE_chk1.txt says PASS and its record carries the wrapper hash). Written by make_estclose_queue.py from the queue of card ADD76.",
        "# No training. No process listing or stopping, no file removal or move: the queue only writes new files.",
        "# Markers START_ESTCLOSE10K.txt / ESTCLOSE10K_DONE.txt / ESTCLOSE10K_STOP.txt in $W; step markers ro_<arm> in $W/markers. DRY=1 prints every command and runs nothing."]
K = "READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5"
NAME = {e: "I10RQ50e%sKF5" % e for e in ESTS}
tail = ["",
        '[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"',
        'if [ "$DRY" != 1 ]; then',
        '  CK="$W/estclose_test/CHECK_DONE_chk1.txt"',
        '  grep -q "^ZZ CHECK-ESTCLOSE PASS" "$CK" 2>/dev/null || stop "NO-PASS-OF-THE-WRAPPER-CHECK"',
        '  got=$(sha256sum "$WRAP" | cut -c1-16); [ "$got" = "%s" ] || stop "WRAPPER-SHA $got"' % WRAP_SHA,
        '  grep -q "%s" "$W/estclose_test/CHECK_ESTCLOSE_chk1.txt.hashes" 2>/dev/null || stop "CHECK-RECORD-HAS-NOT-THIS-WRAPPER"' % WRAP_SHA,
        "  for nm in I10RQ50KF5P30Hb20 I10RLcKF5P30Hb20 I10RQ50KF5Hb20 I10RLcKF5Hb20 I10R0KP30Hb20 I10R0KHb20 I10R0Hb20; do",
        '    for k in "${ARMS[@]}"; do [ -d "$W/t1/${nm}_s$k/dump_${nm}_s${k}_m10_causal_heads" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done',
        "  done",
        "  for e in %s; do" % " ".join(ESTS),
        '    for k in "${ARMS[@]}"; do [ -f "$V/closeread/est/closest_${e}_s$k.npz" ] || stop "NO-ESTIMATOR closest_${e}_s$k"; done',
        "  done",
        "fi"]
names = []
for setting, pre in (("P30Hb20", "RGRID=$G30 PREROLL=30 "), ("Hb20", "")):
    for e in ESTS:
        tail += ['for k in "${ARMS[@]}"; do',
                 '  Q="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"',
                 "  " + pre + K + ' QMAP_FILE="$Q" EST="$V/closeread/est/closest_%s_s${k}.npz" rollout %s%s RRe "$k" "$k"' % (e, NAME[e], setting),
                 "done"]
        names.append(NAME[e] + setting)
tail += ['if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi',
         'cp -f "$LOG" "$SM/" 2>/dev/null',
         'for nm in %s; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done' % " ".join(names),
         'echo "$(ts) all steps done" > "$W/ESTCLOSE10K_DONE.txt"',
         'log "ESTCLOSE10K_DONE"',
         ""]
text = "\n".join(head) + "\n" + text + "\n" + "\n".join(tail)
for a, b in (("MIDLOOK10K", "ESTCLOSE10K"), ("queue_midlook.log", "queue_estclose.log"), ("small_ml", "small_ec")):
    text = text.replace(a, b)
assert "MIDLOOK" not in text and "midlook" not in text.lower() and "CLOSRC" not in text, "a name of another card is left"
assert "\r" not in text
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d rollouts %d (wrapper hash %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n"), 2 * len(ESTS) * 2, WRAP_SHA))
