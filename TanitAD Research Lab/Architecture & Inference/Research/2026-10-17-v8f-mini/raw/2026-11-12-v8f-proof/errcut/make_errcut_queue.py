"""make_errcut_queue.py -- writes pod_errcut10k_queue.sh from ../gaprule/pod_gaprule10k_queue.sh (sha256 prefix below) by asserted, counted text edits (card ADD96, ERRCUT10K).
Kept byte for byte: everything from `set -u` down to the end of the function rollout() except the lines named below (the rollout function itself is not edited at all), the slot rule, the start
marker with set -C, the DRY=1 mode, the timeout, the wrapper (generation rzgfhqetj, hash cbf7063d867c9f4e), the banked-run checks.
New: root $GP default /workspace/v8proof/errcut10k (env GR_ROOT keeps working), check folder default /workspace/v8proof/gaprule10kb/check_j2 (the passed check record of card ADD95; env GR_CHECK),
markers START_ERRCUT10K_s<k>.txt, ERRCUT10K_DONE_s<k>.txt, ERRCUT10K_STOP_s<k>.txt, log queue_errcut_s<k>.log, the admission (MIX_JOINS_RECORD.json: e0 and e1 PASS, sha256 of every join file and of
the evaluation join), the eight rollouts (arms K100 K50 K25 K12, run folders $W/t1/I10RLcKF5P30Hb20m<ARM>_s<m>, each with --rule-join $PJ/<ARM>_s<m>.npz of its own model seed).
Usage: python make_errcut_queue.py   (run in errcut/). Refuses to overwrite (EC_OVERWRITE_OWN=1 rewrites it). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "gaprule", "pod_gaprule10k_queue.sh"))
OUT = os.path.join(HERE, "pod_errcut10k_queue.sh")
SRC_SHA_PREFIX = "4873249749264b43"
WRAP_SHA = "cbf7063d867c9f4e"
ARMS = ("K100", "K50", "K25", "K12")
raw = io.open(SRC, "rb").read()
src_sha = hashlib.sha256(raw).hexdigest()[:16]
if src_sha != SRC_SHA_PREFIX:
    sys.exit("REFUSED: the source queue is not the one of card ADD95 (%s)" % src_sha)
if os.path.exists(OUT) and os.environ.get("EC_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (EC_OVERWRITE_OWN=1 rewrites it)" % OUT)
text = raw.decode("utf-8")
assert "\r" not in text
N = 0


def once(a, b, n=1):
    global text, N
    assert text.count(a) == n, (a, text.count(a), n)
    text = text.replace(a, b)
    N += 1


# the model's rollout function, kept to compare after the edits
i0 = text.index("rollout() {\n")
i1 = text.index("\n}\n", i0) + 3
ROLL_BEFORE = text[i0:i1]

# ---- head (comment block before `set -u`)
j = text.index("set -u\n")
head = ["#!/usr/bin/env bash",
        "# pod_errcut10k_queue.sh -- card ADD96 (ERRCUT10K): 8 rollouts of the label-fed form with closing speed (form RLc, the arguments of the banked arm I10RLcKF5P30Hb20: pre-roll 30 steps = 6 s, readout K,",
        "# --rule-every 5, --kappa-max 0.2, heading bound 0.2 1/m, model seeds 0 and 1) whose lead rule reads its lead from a MIXED JOIN given with --rule-join; the scoring keeps the evaluation join.",
        "# Arms K100, K50, K25, K12 (joins $GP/joins/<ARM>_s<m>.npz from make_mix_joins.py, one per arm and model seed, model seed m uses reader seed m); run names I10RLcKF5P30Hb20m<ARM>_s<m>.",
        "# Wrapper v8f_t1_eval_reader_rule_rzgfhqetj.py (hash %s), admitted only with the pod check of control j2 of card ADD95 (CHECK_WRAPPER_J_DONE.txt last line PASS, its .hashes holding the wrapper hash)" % WRAP_SHA,
        "# and with MIX_JOINS_RECORD.json saying e0 PASS and e1 PASS with the hash of every join file and of the evaluation join. Banked runs needed: I10RLcKF5P30Hb20 and I10RQ50KF5P30Hb20 (RR), both seeds.",
        "# GR_SEED=0|1|ALL (default ALL): one model seed per process with the markers START_ERRCUT10K_s<k>.txt, ERRCUT10K_DONE_s<k>.txt, ERRCUT10K_STOP_s<k>.txt in $GP and the log queue_errcut_s<k>.log.",
        "# DRY=1 prints every command and runs nothing. A rollout starts only with more than 28 GB free (memory.max minus anon) and fewer than 3 GPU compute processes.",
        "# Written by make_errcut_queue.py from the queue of card ADD95 (gaprule/pod_gaprule10k_queue.sh). No training. No process listing or stopping, no file removal or move."]
text = "\n".join(head) + "\n" + text[j:]
N += 1

once("GP=${GR_ROOT:-/workspace/v8proof/gaprule10k}\n", "GP=${GR_ROOT:-/workspace/v8proof/errcut10k}\n")
once("CHK=${GR_CHECK:-$GP/check_j}\n", "CHK=${GR_CHECK:-/workspace/v8proof/gaprule10kb/check_j2}\n")
once("SM=$GP/small_gr", "SM=$GP/small_ec")
once("LOG=$GP/queue_gaprule$PS.log", "LOG=$GP/queue_errcut$PS.log")
once('"$GP/GAPRULE10K_STOP$PS.txt"', '"$GP/ERRCUT10K_STOP$PS.txt"')
once('$GP/START_GAPRULE10K$PS.txt', '$GP/START_ERRCUT10K$PS.txt', 2)
once('log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD94 (GAPRULE10K): label-fed rule with closing speed, lead read from a planted join (G0, GR1, GR2, GW05, GC10, GC20), scoring on the evaluation join, at age"',
     'log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD96 (ERRCUT10K): label-fed rule with closing speed, lead read from a mixed join (K100, K50, K25, K12), scoring on the evaluation join, at age"')
once("PLANTED_JOINS_RECORD.json", "MIX_JOINS_RECORD.json", 2)
once("NO-PLANTED-JOINS-RECORD", "NO-MIX-JOINS-RECORD")

# ---- admission
a0 = text.index("admit() {")
a1 = text.index("\n}\n", a0) + 3
ADMIT_PY = r'''import hashlib, json, os, sys
rec_path, jdir, ejoin, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
rec = json.load(open(rec_path, encoding="utf-8"))
for c in ("e0", "e1"):
    if not isinstance(rec.get(c), dict) or rec[c].get("PASS") is not True:
        print(c.upper() + "-NOT-PASS"); sys.exit(1)
h = hashlib.sha256(open(ejoin, "rb").read()).hexdigest()
if rec.get("eval_join_sha256") != h:
    print("RECORD-IS-FOR-ANOTHER-EVAL-JOIN"); sys.exit(1)
for a in names:
    fn = os.path.join(jdir, a + ".npz")
    if not os.path.isfile(fn):
        print("NO-JOIN " + a); sys.exit(1)
    if hashlib.sha256(open(fn, "rb").read()).hexdigest() != (rec.get("arms", {}).get(a) or {}).get("sha256"):
        print("JOIN-HASH-DIFFERS " + a); sys.exit(1)
print("ADMIT-OK joins %d" % len(names))
'''
admit_new = ('admit() { # RECORD JOINDIR EVALJOIN NAMES -> prints why on failure\n  "$PY" - "$@" <<\'PYEOF\'\n' + ADMIT_PY.rstrip("\n") + "\nPYEOF\n}\n")
text = text[:a0] + admit_new + text[a1:]
N += 1
JOINNAMES = ",".join("%s_s%d" % (a, s) for a in ARMS for s in (0, 1))
once('adm=$(admit "$PJ/MIX_JOINS_RECORD.json" "$PJ" "$JOINE" "G0,GR1,GR2,GW05,GC10,GC20")', 'adm=$(admit "$PJ/MIX_JOINS_RECORD.json" "$PJ" "$JOINE" "%s")' % JOINNAMES)

# ---- the rollouts and the end
r0 = text.index('for k in "${ARMS[@]}"; do\n  RGRID=')
r1 = text.index('if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi')
blocks = []
for arm in ARMS:
    blocks.append('for k in "${ARMS[@]}"; do\n  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 RJOIN="$PJ/%s_s$k.npz" rollout I10RLcKF5P30Hb20m%s RLc "$k" "$k"\ndone\n' % (arm, arm))
text = text[:r0] + "".join(blocks) + text[r1:]
N += 1
once('NAMES="I10RLcKF5P30Hb20jG0 I10RLcKF5P30Hb20jGR1 I10RLcKF5P30Hb20jGR2 I10RLcKF5P30Hb20jGW05 I10RLcKF5P30Hb20jGC10 I10RLcKF5P30Hb20jGC20"',
     'NAMES="%s"' % " ".join("I10RLcKF5P30Hb20m" + a for a in ARMS))
once('echo "$(ts) all steps done (seeds $SEEDSW)" > "$GP/GAPRULE10K_DONE$PS.txt"', 'echo "$(ts) all steps done (seeds $SEEDSW)" > "$GP/ERRCUT10K_DONE$PS.txt"')
once('log "GAPRULE10K_DONE seeds $SEEDSW"', 'log "ERRCUT10K_DONE seeds $SEEDSW"')

# ---- guards
i0 = text.index("rollout() {\n")
i1 = text.index("\n}\n", i0) + 3
assert text[i0:i1] == ROLL_BEFORE, "the rollout function was edited"
for bad in ("GAPRULE", "PLANTED", "planted", "gaprule10k/", "GR1", "GW05", "GC10", "GC20", "jG0"):
    assert bad not in text, bad
outside = text[:i0] + text[i1:]
assert all(x.startswith("#") for x in outside.split("\n") if "--rule-join" in x or "--eval-join" in x), "the rule join may be named outside the rollout function only in comments"
assert text[i0:i1].count("--rule-join") == 1 and text[i0:i1].count("--eval-join $JOINE") == 1 and text[i0:i1].count('--eval-join "$JOINE"') == 1
assert "\r" not in text
text = text if text.endswith("\n") else text + "\n"
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d rollouts 8 edits %d (source %s, wrapper hash %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n"), N, src_sha, WRAP_SHA))
