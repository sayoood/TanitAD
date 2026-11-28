"""make_constclose_queue.py -- writes pod_constclose10k_queue.sh from the queue of card ADD79 (estclose/pod_estclose10k_queue.sh, sha256 prefix b68b73b66c9ead6f) by counted text edits.

Card ADD80 (CONSTCLOSE10K) section 2: twelve rollouts of form RRe of the same wrapper generation (v8f_t1_eval_reader_rule_rzgfhqe.py, hash a747abeb51ef041f), the estimator file being a constant
(history of one frame; /workspace/v8proof/closeread/est_const/const_C<cc>_H1.npz, the same file for both seeds), values 1.0 / 2.0 / 4.0 m/s, in the order
  AGE C20 (s0, s1), AGE C10, AGE C40, FRESH C20, FRESH C10, FRESH C40
with the run names I10RQ50e<C10|C20|C40>KF5P30Hb20 (AGE) and I10RQ50e<C10|C20|C40>KF5Hb20 (FRESH), each _s0 and _s1.
Changed against the parent: markers START_CONSTCLOSE10K.txt / CONSTCLOSE10K_DONE.txt / CONSTCLOSE10K_STOP.txt, log queue_constclose.log, small folder small_cc, the admission of the wrapper by the pod check
of this card (constclose_test/CHECK_CONST_DONE_<tag>.txt says PASS, the record of the check holds the wrapper hash and the hash of const_C20_H1.npz), the list of rollouts, the checks of the constant files.
Everything else (memory and GPU slot wait, banked-run checks, DRY=1, the rollout function) is the text of the parent.
Usage: python make_constclose_queue.py CARD   (CARD = the card name written into the header, for example ADD80). Standard library only. Refuses to overwrite its output (CC_OVERWRITE_OWN=1 rewrites it)."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("CC_SRC") or os.path.normpath(os.path.join(HERE, "..", "estclose", "pod_estclose10k_queue.sh"))
OUT = os.environ.get("CC_OUT") or os.path.join(HERE, "pod_constclose10k_queue.sh")
WRAP_SHA = "a747abeb51ef041f"
SRC_SHA = "b68b73b66c9ead6f"
VALUES = ("C10", "C20", "C40")
ORDER = (("P30Hb20", "C20"), ("P30Hb20", "C10"), ("P30Hb20", "C40"), ("Hb20", "C20"), ("Hb20", "C10"), ("Hb20", "C40"))
if len(sys.argv) != 2:
    sys.exit("usage: make_constclose_queue.py CARD")
CARD = sys.argv[1]
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != SRC_SHA:
    sys.exit("REFUSED: the source queue is not the one of card ADD79")
if os.path.exists(OUT) and os.environ.get("CC_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (set CC_OVERWRITE_OWN=1 to rewrite it)" % OUT)
L = raw.decode("utf-8").split("\n")
i_tail = next(i for i, x in enumerate(L) if x.startswith('[ "$DRY" = 1 ] || [ -f "$G30" ]'))
assert L[0] == "#!/usr/bin/env bash" and L[7] == "set -u" and L[i_tail - 2] == "}" and i_tail == 156 - 1, "anchors"
body = "\n".join(L[7:i_tail - 1])                                   # from "set -u" to the closing brace of rollout()


def once(a, b):
    global body
    assert body.count(a) == 1, (a, body.count(a))
    body = body.replace(a, b)


once('LOG=$W/queue_estclose.log', 'LOG=$W/queue_constclose.log')
once('SM=$W/small_ec', 'SM=$W/small_cc')
once('echo "$(ts) $*" > "$W/ESTCLOSE10K_STOP.txt"', 'echo "$(ts) $*" > "$W/CONSTCLOSE10K_STOP.txt"')
once('( set -C; echo "$(ts) pid $$" > "$W/START_ESTCLOSE10K.txt" ) 2>/dev/null || { echo "REFUSED: $W/START_ESTCLOSE10K.txt exists (a queue was started)"; exit 2; }',
     '( set -C; echo "$(ts) pid $$" > "$W/START_CONSTCLOSE10K.txt" ) 2>/dev/null || { echo "REFUSED: $W/START_CONSTCLOSE10K.txt exists (a queue was started)"; exit 2; }')
once('log "queue start (DRY=$DRY): card ADD79 (ESTCLOSE10K): 8 rollouts of form RRe (closing speed estimated from the reader reads) with the second look under readout K, estimators N10 and L25, at age and fresh"',
     'log "queue start (DRY=$DRY): card %s (CONSTCLOSE10K): 12 rollouts of form RRe with a constant closing input (1.0, 2.0, 4.0 m/s) and the second look under readout K, at age and fresh"' % CARD)
once('V=/workspace/v8proof\n', 'V=/workspace/v8proof\nESTC=${I10_ESTC:-$V/closeread/est_const}\nCKTAG=${CC_CHECK_TAG:-ck1}\n')

head = ["#!/usr/bin/env bash",
        "# pod_constclose10k_queue.sh -- card %s (CONSTCLOSE10K): 12 rollouts of form RRe in which the closing-speed input of the lead rule is a CONSTANT (estimator files of history one frame," % CARD,
        "# $ESTC/const_<C10|C20|C40>_H1.npz = 1.0 / 2.0 / 4.0 m/s, the same file for both seeds), --rule-every 5 under readout K (--acc-gain 0 --path-exec --kappa-max 0.2), q50 map, model seeds 0 and 1",
        "# (reader seed k), at age (P = 30 grid of card ADD70, pre-roll 30) and fresh (pinned grid); order: AGE C20, AGE C10, AGE C40, FRESH C20, FRESH C10, FRESH C40; wrapper",
        "# v8f_t1_eval_reader_rule_rzgfhqe.py (hash %s), admitted only with the pod check of this card (constclose_test/CHECK_CONST_DONE_<tag>.txt says PASS and the record carries the wrapper hash" % WRAP_SHA,
        "# and the hash of const_C20_H1.npz). Written by make_constclose_queue.py from the queue of card ADD79 (b68b73b66c9ead6f). No training. No process listing or stopping, no file removal or move.",
        "# Markers START_CONSTCLOSE10K.txt / CONSTCLOSE10K_DONE.txt / CONSTCLOSE10K_STOP.txt in $W; step markers ro_<arm> in $W/markers. DRY=1 prints every command and runs nothing."]
tail = ["",
        '[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"',
        'if [ "$DRY" != 1 ]; then',
        '  CK="$W/constclose_test/CHECK_CONST_DONE_$CKTAG.txt"',
        '  grep -q "^ZZ CHECK-CONSTCLOSE PASS" "$CK" 2>/dev/null || stop "NO-PASS-OF-THE-CONSTCLOSE-CHECK"',
        '  got=$(sha256sum "$WRAP" | cut -c1-16); [ "$got" = "%s" ] || stop "WRAPPER-SHA $got"' % WRAP_SHA,
        '  grep -q "%s" "$W/constclose_test/CHECK_CONSTCLOSE_$CKTAG.txt.hashes" 2>/dev/null || stop "CHECK-RECORD-HAS-NOT-THIS-WRAPPER"' % WRAP_SHA,
        '  for c in %s; do' % " ".join(VALUES),
        '    [ -f "$ESTC/const_${c}_H1.npz" ] || stop "NO-CONST-FILE const_${c}_H1.npz"',
        '  done',
        '  got=$(sha256sum "$ESTC/const_C20_H1.npz" | cut -c1-16)',
        '  grep -q "$got" "$W/constclose_test/CHECK_CONSTCLOSE_$CKTAG.txt.hashes" 2>/dev/null || stop "CHECK-RECORD-HAS-NOT-THIS-CONST-FILE $got"',
        "  for nm in I10RQ50KF5P30Hb20 I10RQ50cKF5P30Hb20 I10RQ50hKF5P30Hb20 I10RLcKF5P30Hb20 I10RQ50KF5Hb20 I10RQ50cKF5Hb20 I10RQ50hKF5Hb20 I10RLcKF5Hb20 I10R0KP30Hb20 I10R0KHb20 I10R0Hb20; do",
        '    for k in "${ARMS[@]}"; do [ -d "$W/t1/${nm}_s$k/dump_${nm}_s${k}_m10_causal_heads" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done',
        "  done",
        "fi"]
K = "READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5"
names = []
for setting, c in ORDER:
    pre = "RGRID=$G30 PREROLL=30 " if setting == "P30Hb20" else ""
    nm = "I10RQ50e%sKF5%s" % (c, setting)
    tail += ['for k in "${ARMS[@]}"; do',
             '  Q="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"',
             "  " + pre + K + ' QMAP_FILE="$Q" EST="$ESTC/const_%s_H1.npz" rollout %s RRe "$k" "$k"' % (c, nm),
             "done"]
    names.append(nm)
assert len(names) == 6 and len(set(names)) == 6
tail += ['if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi',
         'cp -f "$LOG" "$SM/" 2>/dev/null',
         'for nm in %s; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done' % " ".join(names),
         'echo "$(ts) all steps done" > "$W/CONSTCLOSE10K_DONE.txt"',
         'log "CONSTCLOSE10K_DONE"',
         ""]
text = "\n".join(head) + "\n" + body + "\n" + "\n".join(tail)
assert "ESTCLOSE10K" not in text and "estclose_test" not in text and "queue_estclose" not in text and "small_ec" not in text and "START_ESTCLOSE" not in text and "N10" not in text and "L25" not in text, "a name of the parent card is left"
assert text.count("rollout I10RQ50e") == 6 and text.count("RRe") >= 6
assert "\r" not in text
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d rollouts %d (wrapper hash %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n"), 2 * len(names), WRAP_SHA))
