"""make_negconst_queue.py -- writes pod_negconst10k_queue.sh from the queue of card ADD80 (constclose/pod_constclose10k_queue.sh, sha256 prefix 493d6361af2ec932) by counted text edits.

Card ADD82 (NEGCONST10K) section 2: twelve rollouts of form RRe of the same wrapper generation (v8f_t1_eval_reader_rule_rzgfhqe.py, hash a747abeb51ef041f), the estimator file being a constant of the
CAUTIOUS sign (history of one frame; /workspace/v8proof/closeread/est_negconst/const_M<cc>_H1.npz, the same file for both seeds), values -1.0 / -2.0 / -4.0 m/s, in the order
  AGE M10 (s0, s1), AGE M20, AGE M40, FRESH M10, FRESH M20, FRESH M40
with the run names I10RQ50e<M10|M20|M40>KF5P30Hb20 (AGE) and I10RQ50e<M10|M20|M40>KF5Hb20 (FRESH), each _s0 and _s1.
Changed against the parent: markers START_NEGCONST10K.txt / NEGCONST10K_DONE.txt / NEGCONST10K_STOP.txt, log queue_negconst.log, small folder small_nc, the admission of the wrapper by the pod check
of this card (negconst_test/CHECK_NEG_DONE_<tag>.txt says PASS, tag nk1, the record of the check holds the wrapper hash and the hash of const_M20_H1.npz), the list of rollouts, the checks of the constant
files, and the banked runs of cards ADD79 (RQ50eN10, RQ50eL25) and ADD80 (RQ50eC10, C20, C40, both settings, seeds 0 and 1) that are required beside the earlier banked ones.
Everything else (memory and GPU slot wait, DRY=1, the rollout function) is the text of the parent.
Usage: python make_negconst_queue.py [CARD]   (CARD = the card name written into the header, default ADD82). Standard library only. Refuses to overwrite its output (NC_OVERWRITE_OWN=1 rewrites it)."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("NC_SRC") or os.path.normpath(os.path.join(HERE, "..", "constclose", "pod_constclose10k_queue.sh"))
OUT = os.environ.get("NC_OUT") or os.path.join(HERE, "pod_negconst10k_queue.sh")
WRAP_SHA = "a747abeb51ef041f"
SRC_SHA = "493d6361af2ec932"
VALUES = ("M10", "M20", "M40")
ORDER = (("P30Hb20", "M10"), ("P30Hb20", "M20"), ("P30Hb20", "M40"), ("Hb20", "M10"), ("Hb20", "M20"), ("Hb20", "M40"))
CARD = sys.argv[1] if len(sys.argv) > 1 else "ADD82"
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != SRC_SHA:
    sys.exit("REFUSED: the source queue is not the one of card ADD80")
if os.path.exists(OUT) and os.environ.get("NC_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (set NC_OVERWRITE_OWN=1 to rewrite it)" % OUT)
L = raw.decode("utf-8").split("\n")
i_tail = next(i for i, x in enumerate(L) if x.startswith('[ "$DRY" = 1 ] || [ -f "$G30" ]'))
assert L[0] == "#!/usr/bin/env bash" and L[7] == "set -u" and L[i_tail - 2] == "}", "anchors"
body = "\n".join(L[7:i_tail - 1])                                   # from "set -u" to the closing brace of rollout()
n_edits = 0


def once(a, b):
    global body, n_edits
    assert body.count(a) == 1, (a, body.count(a))
    body = body.replace(a, b)
    n_edits += 1


once('LOG=$W/queue_constclose.log', 'LOG=$W/queue_negconst.log')
once('SM=$W/small_cc', 'SM=$W/small_nc')
once('echo "$(ts) $*" > "$W/CONSTCLOSE10K_STOP.txt"', 'echo "$(ts) $*" > "$W/NEGCONST10K_STOP.txt"')
once('( set -C; echo "$(ts) pid $$" > "$W/START_CONSTCLOSE10K.txt" ) 2>/dev/null || { echo "REFUSED: $W/START_CONSTCLOSE10K.txt exists (a queue was started)"; exit 2; }',
     '( set -C; echo "$(ts) pid $$" > "$W/START_NEGCONST10K.txt" ) 2>/dev/null || { echo "REFUSED: $W/START_NEGCONST10K.txt exists (a queue was started)"; exit 2; }')
once('log "queue start (DRY=$DRY): card ADD80 (CONSTCLOSE10K): 12 rollouts of form RRe with a constant closing input (1.0, 2.0, 4.0 m/s) and the second look under readout K, at age and fresh"',
     'log "queue start (DRY=$DRY): card %s (NEGCONST10K): 12 rollouts of form RRe with a constant closing input of the cautious sign (-1.0, -2.0, -4.0 m/s) and the second look under readout K, at age and fresh"' % CARD)
once('ESTC=${I10_ESTC:-$V/closeread/est_const}\nCKTAG=${CC_CHECK_TAG:-ck1}\n', 'ESTC=${I10_ESTC:-$V/closeread/est_negconst}\nCKTAG=${NC_CHECK_TAG:-nk1}\n')

head = ["#!/usr/bin/env bash",
        "# pod_negconst10k_queue.sh -- card %s (NEGCONST10K): 12 rollouts of form RRe in which the closing-speed input of the lead rule is a CONSTANT OF THE CAUTIOUS SIGN (estimator files of history one frame," % CARD,
        "# $ESTC/const_<M10|M20|M40>_H1.npz = -1.0 / -2.0 / -4.0 m/s, the same file for both seeds), --rule-every 5 under readout K (--acc-gain 0 --path-exec --kappa-max 0.2), q50 map, model seeds 0 and 1",
        "# (reader seed k), at age (P = 30 grid of card ADD70, pre-roll 30) and fresh (pinned grid); order: AGE M10, AGE M20, AGE M40, FRESH M10, FRESH M20, FRESH M40; wrapper",
        "# v8f_t1_eval_reader_rule_rzgfhqe.py (hash %s), admitted only with the pod check of this card (negconst_test/CHECK_NEG_DONE_<tag>.txt says PASS, tag nk1, and the record carries the wrapper hash" % WRAP_SHA,
        "# and the hash of const_M20_H1.npz); needs the banked runs of cards ADD79 and ADD80 beside the earlier ones. Written by make_negconst_queue.py from the queue of card ADD80 (493d6361af2ec932). No training.",
        "# No process listing or stopping, no file removal or move. Markers START_NEGCONST10K.txt / NEGCONST10K_DONE.txt / NEGCONST10K_STOP.txt in $W; step markers ro_<arm> in $W/markers. DRY=1 prints every command and runs nothing."]
BANKED = ("I10RQ50KF5P30Hb20 I10RQ50cKF5P30Hb20 I10RQ50hKF5P30Hb20 I10RLcKF5P30Hb20 I10RQ50KF5Hb20 I10RQ50cKF5Hb20 I10RQ50hKF5Hb20 I10RLcKF5Hb20 I10R0KP30Hb20 I10R0KHb20 I10R0Hb20 "
          "I10RQ50eN10KF5P30Hb20 I10RQ50eL25KF5P30Hb20 I10RQ50eN10KF5Hb20 I10RQ50eL25KF5Hb20 "
          "I10RQ50eC10KF5P30Hb20 I10RQ50eC20KF5P30Hb20 I10RQ50eC40KF5P30Hb20 I10RQ50eC10KF5Hb20 I10RQ50eC20KF5Hb20 I10RQ50eC40KF5Hb20")
assert len(BANKED.split()) == 21 and len(set(BANKED.split())) == 21
tail = ["",
        '[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"',
        'if [ "$DRY" != 1 ]; then',
        '  CK="$W/negconst_test/CHECK_NEG_DONE_$CKTAG.txt"',
        '  grep -q "^ZZ CHECK-NEGCONST PASS" "$CK" 2>/dev/null || stop "NO-PASS-OF-THE-NEGCONST-CHECK"',
        '  got=$(sha256sum "$WRAP" | cut -c1-16); [ "$got" = "%s" ] || stop "WRAPPER-SHA $got"' % WRAP_SHA,
        '  grep -q "%s" "$W/negconst_test/CHECK_NEGCONST_$CKTAG.txt.hashes" 2>/dev/null || stop "CHECK-RECORD-HAS-NOT-THIS-WRAPPER"' % WRAP_SHA,
        '  for c in %s; do' % " ".join(VALUES),
        '    [ -f "$ESTC/const_${c}_H1.npz" ] || stop "NO-CONST-FILE const_${c}_H1.npz"',
        '  done',
        '  got=$(sha256sum "$ESTC/const_M20_H1.npz" | cut -c1-16)',
        '  grep -q "$got" "$W/negconst_test/CHECK_NEGCONST_$CKTAG.txt.hashes" 2>/dev/null || stop "CHECK-RECORD-HAS-NOT-THIS-CONST-FILE $got"',
        "  for nm in %s; do" % BANKED,
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
         'echo "$(ts) all steps done" > "$W/NEGCONST10K_DONE.txt"',
         'log "NEGCONST10K_DONE"',
         ""]
text = "\n".join(head) + "\n" + body + "\n" + "\n".join(tail)
for bad in ("CONSTCLOSE", "constclose", "const_C", "ESTCLOSE", "small_cc", "small_ec", "CHECK_CONST_", "ck1"):
    assert bad not in text, "a name of the parent card is left: " + bad
assert text.count("rollout I10RQ50e") == 6 and text.count("RRe") >= 6
assert "\r" not in text
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d edits %d rollouts %d (wrapper hash %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n"), n_edits, 2 * len(names), WRAP_SHA))
