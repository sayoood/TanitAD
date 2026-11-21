"""make_shuf_queue.py -- writes pod_shufclose10k_queue.sh from the queue of card ADD79 (estclose/pod_estclose10k_queue.sh, sha256 prefix b68b73b66c9ead6f) by counted text edits.

Kept line for line: the slot rule (fewer than 3 GPU compute processes and more than 28 GB of container memory free, two probes 45 s apart), the start checks and the rollout function.
New: the header, the wrapper (the table generation v8f_t1_eval_reader_rule_rzgfhqet.py, admitted only with its pod check of control u0: the marker shufclose_test/CHECK_SHUF_DONE_sk1.txt has to
say PASS, the check record has to carry the wrapper hash, and the wrapper has to have that hash; and with SURROGATE_RECORD.json of make_surrogate.py saying pass), the marker names, the log
name, the small folder, the closing source (a table surr_<E>_s<k>.npz of closeread/est_shuf instead of an estimator file) and the eight rollouts of card ADD83 section 3, in the order
AGE SL25, AGE SN10, FRESH SL25, FRESH SN10 (form RRe, readout K --acc-gain 0 --path-exec --kappa-max 0.2, --rule-every 5, q50 map, model seeds 0 and 1, reader seed k):
  I10RQ50eSL25KF5P30Hb20  I10RQ50eSN10KF5P30Hb20  I10RQ50eSL25KF5Hb20  I10RQ50eSN10KF5Hb20      (each _s0, _s1)
The queue requires the banked runs of card ADD79 (the runs of the estimates themselves) and the runs of rows 137.
Usage: python make_shuf_queue.py CARD   (run in shufclose/; CARD = the card name written into the header, for example ADD83). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "estclose", "pod_estclose10k_queue.sh"))
OUT = os.path.join(HERE, "pod_shufclose10k_queue.sh")
WRAP_FILE = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqet.py")
SRC_SHA = "b68b73b66c9ead6f"
ESTS = ("L25", "N10")
if len(sys.argv) != 2:
    sys.exit("usage: make_shuf_queue.py CARD")
CARD = sys.argv[1]
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != SRC_SHA:
    sys.exit("REFUSED: the source queue is not the one of card ADD79")
if not os.path.isfile(WRAP_FILE):
    sys.exit("REFUSED: the wrapper %s does not exist (run make_shuf_wrapper.py first)" % WRAP_FILE)
WRAP_SHA = hashlib.sha256(io.open(WRAP_FILE, "rb").read()).hexdigest()[:16]
if os.path.exists(OUT) and os.environ.get("SC_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (set SC_OVERWRITE_OWN=1 to rewrite it)" % OUT)
L = raw.decode("utf-8").split("\n")
i_set = L.index("set -u")
i_roll_end = max(i for i, x in enumerate(L) if x == "}")
i_tail = next(i for i, x in enumerate(L) if x.startswith('[ "$DRY" = 1 ] || [ -f "$G30" ]'))
assert L[0] == "#!/usr/bin/env bash" and i_set == 7 and i_tail == i_roll_end + 2, ("anchors", i_set, i_tail, i_roll_end)
text = "\n".join(L[i_set:i_roll_end + 1])
N_EDITS = 0


def once(a, b, n=1):
    global text, N_EDITS
    assert text.count(a) == n, (a, text.count(a), n)
    text = text.replace(a, b)
    N_EDITS += 1


once("WRAP=$B/v8f_t1_eval_reader_rule_rzgfhqe.py", "WRAP=$B/v8f_t1_eval_reader_rule_rzgfhqet.py")
once("V=/workspace/v8proof\n", "V=/workspace/v8proof\nTABLES=${I10_TABLES:-$V/closeread/est_shuf}\n")
once("LOG=$W/queue_estclose.log", "LOG=$W/queue_shufclose.log")
once("SM=$W/small_ec", "SM=$W/small_sc")
once("ESTCLOSE10K_STOP.txt", "SHUFCLOSE10K_STOP.txt")
once("START_ESTCLOSE10K.txt", "START_SHUFCLOSE10K.txt", 2)
once('log "queue start (DRY=$DRY): card ADD79 (ESTCLOSE10K): 8 rollouts of form RRe (closing speed estimated from the reader reads) with the second look under readout K, estimators N10 and L25, at age and fresh"',
     'log "queue start (DRY=$DRY): card %s (SHUFCLOSE10K): 8 rollouts of form RRe whose closing input is the estimate of ANOTHER clip (surrogate tables), estimators L25 and N10, at age and fresh"' % CARD)
once("for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzgfhqe.py closest.py ", "for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzgfhqet.py v8f_t1_eval_reader_rule_rzgfhqe.py closest.py ")
once('  if [ "$form" = RRe ]; then\n'
     '    [ "$DRY" = 1 ] || [ -f "${EST:-}" ] || stop "NO-ESTIMATOR-FILE ${EST:-unset} $arm"\n'
     '    extra+=(--closing-est "${EST:?}")\n'
     '  fi\n',
     '  if [ "$form" = RRe ]; then\n'
     '    [ "$DRY" = 1 ] || [ -f "${TAB:-}" ] || stop "NO-TABLE-FILE ${TAB:-unset} $arm"\n'
     '    extra+=(--closing-table "${TAB:?}")\n'
     '  fi\n')
once('${EST:+ closing-est $EST est sha256 $(sha256sum "$EST" | cut -c1-64)}', '${TAB:+ closing-table $TAB table sha256 $(sha256sum "$TAB" | cut -c1-64)}')
for bad in ("ESTCLOSE10K", "closing-est", "queue_estclose", "small_ec"):
    assert bad not in text, bad

head = ["#!/usr/bin/env bash",
        "# pod_shufclose10k_queue.sh -- card %s (SHUFCLOSE10K): 8 rollouts of form RRe whose closing-speed input is the ESTIMATE OF ANOTHER CLIP: the surrogate tables" % CARD,
        "# /workspace/v8proof/closeread/est_shuf/surr_<L25|N10>_s<k>.npz of make_surrogate.py (clip c at frame j gets the estimate of clip pi(c) at frame j), read through --closing-table;",
        "# --rule-every 5 under readout K (--acc-gain 0 --path-exec --kappa-max 0.2), q50 map, model seeds 0 and 1 (reader seed k, surrogate of seed k), each at age (P = 30 grid of card ADD70,",
        "# pre-roll 30) and fresh (pinned grid); wrapper v8f_t1_eval_reader_rule_rzgfhqet.py (table generation, hash %s)," % WRAP_SHA,
        "# admitted only with its pod check of control u0 (shufclose_test/CHECK_SHUF_DONE_sk1.txt says PASS and its record carries the wrapper hash) and with SURROGATE_RECORD.json saying pass.",
        "# Written by make_shuf_queue.py from the queue of card ADD79 (estclose/pod_estclose10k_queue.sh).",
        "# No training. No process listing or stopping, no file removal or move: the queue only writes new files.",
        "# Markers START_SHUFCLOSE10K.txt / SHUFCLOSE10K_DONE.txt / SHUFCLOSE10K_STOP.txt in $W; step markers ro_<arm> in $W/markers. DRY=1 prints every command and runs nothing."]
K = "READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5"
NAME = {e: "I10RQ50eS%sKF5" % e for e in ESTS}
tail = ["",
        '[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"',
        'if [ "$DRY" != 1 ]; then',
        '  CK="$W/shufclose_test/CHECK_SHUF_DONE_sk1.txt"',
        '  grep -q "^ZZ CHECK-SHUFCLOSE PASS" "$CK" 2>/dev/null || stop "NO-PASS-OF-THE-WRAPPER-CHECK"',
        '  got=$(sha256sum "$WRAP" | cut -c1-16); [ "$got" = "%s" ] || stop "WRAPPER-SHA $got"' % WRAP_SHA,
        '  grep -q "%s" "$W/shufclose_test/CHECK_SHUFCLOSE_sk1.txt.hashes" 2>/dev/null || stop "CHECK-RECORD-HAS-NOT-THIS-WRAPPER"' % WRAP_SHA,
        '  grep -q \'^ "pass": true\' "$TABLES/SURROGATE_RECORD.json" 2>/dev/null || stop "NO-PASS-OF-THE-SURROGATE-RECORD"',
        "  for nm in I10RQ50KF5P30Hb20 I10RLcKF5P30Hb20 I10RQ50KF5Hb20 I10RLcKF5Hb20 I10R0KP30Hb20 I10R0KHb20 I10R0Hb20 I10RQ50eN10KF5P30Hb20 I10RQ50eL25KF5P30Hb20 I10RQ50eN10KF5Hb20 I10RQ50eL25KF5Hb20; do",
        '    for k in "${ARMS[@]}"; do [ -d "$W/t1/${nm}_s$k/dump_${nm}_s${k}_m10_causal_heads" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done',
        "  done",
        "  for e in %s; do" % " ".join(ESTS),
        '    for k in "${ARMS[@]}"; do [ -f "$TABLES/surr_${e}_s$k.npz" ] || stop "NO-TABLE surr_${e}_s$k"; done',
        "  done",
        "fi"]
names = []
for setting, pre in (("P30Hb20", "RGRID=$G30 PREROLL=30 "), ("Hb20", "")):
    for e in ESTS:
        tail += ['for k in "${ARMS[@]}"; do',
                 '  Q="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"',
                 "  " + pre + K + ' QMAP_FILE="$Q" TAB="$TABLES/surr_%s_s${k}.npz" rollout %s%s RRe "$k" "$k"' % (e, NAME[e], setting),
                 "done"]
        names.append(NAME[e] + setting)
tail += ['if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi',
         'cp -f "$LOG" "$SM/" 2>/dev/null',
         'for nm in %s; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done' % " ".join(names),
         'echo "$(ts) all steps done" > "$W/SHUFCLOSE10K_DONE.txt"',
         'log "SHUFCLOSE10K_DONE"',
         ""]
assert "ESTCLOSE" not in text and "estclose" not in text.lower(), "a name of another card is left in the body"
text = "\n".join(head) + "\n" + text + "\n" + "\n".join(tail)
assert "\r" not in text
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d rollouts %d edits %d (wrapper hash %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n"), 2 * len(ESTS) * 2, N_EDITS, WRAP_SHA))
