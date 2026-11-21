"""make_croprule_queue.py -- writes pod_croprule10k_queue.sh from ../smoothgap/pod_smoothgap10k_queue.sh (sha256 prefix below) by asserted, counted text edits (card ADD101, CROPRULE10K).
The queue of card ADD97 is itself the counted edit of the queue of card ADD96. Kept byte for byte: everything from `set -u` down to the end of the function rollout() except the lines named below
(the rollout function itself is not edited at all and is asserted equal to the one of the queue of card ADD97, of card ADD96 and of the model queue of card ADD95), the slot rule, the start marker with
set -C, the DRY=1 mode, the timeout, the wrapper (generation rzgfhqetj, hash cbf7063d867c9f4e), the admission code, the banked-run checks including the banked K100 runs.
New: root $GP default /workspace/v8proof/croprule10k (env GR_ROOT keeps working), markers START_CROPRULE10K_s<k>.txt, CROPRULE10K_DONE_s<k>.txt, CROPRULE10K_STOP_s<k>.txt, log queue_croprule_s<k>.log,
small folder $GP/small_cr, the record CROP_JOINS_RECORD.json (q0 and q1 PASS, sha256 of every join file and of the evaluation join), the eight rollouts (arms KB5 KB1 KD5 KD1, run folders
$W/t1/I10RLcKF5P30Hb20p<ARM>_s<m>, each with --rule-join $PJ/<ARM>_s<m>.npz of its own model seed).
Usage: python make_croprule_queue.py   (run in croprule/). Refuses to overwrite (CR_OVERWRITE_OWN=1 rewrites it). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "smoothgap", "pod_smoothgap10k_queue.sh"))
ERR = os.path.normpath(os.path.join(HERE, "..", "errcut", "pod_errcut10k_queue.sh"))
MODEL = os.path.normpath(os.path.join(HERE, "..", "gaprule", "pod_gaprule10k_queue.sh"))
OUT = os.path.join(HERE, "pod_croprule10k_queue.sh")
SRC_SHA_PREFIX = "cef2a3414b240499"
WRAP_SHA = "cbf7063d867c9f4e"
ARMS = ("KB5", "KB1", "KD5", "KD1")
raw = io.open(SRC, "rb").read()
src_sha = hashlib.sha256(raw).hexdigest()[:16]
if src_sha != SRC_SHA_PREFIX:
    sys.exit("REFUSED: the source queue is not the one of card ADD97 (%s)" % src_sha)
if os.path.exists(OUT) and os.environ.get("CR_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (CR_OVERWRITE_OWN=1 rewrites it)" % OUT)
text = raw.decode("utf-8")
assert "\r" not in text
N = 0


def once(a, b, n=1):
    global text, N
    assert text.count(a) == n, (a, text.count(a), n)
    text = text.replace(a, b)
    N += 1


def fn_of(t):
    i0 = t.index("rollout() {\n")
    return t[i0:t.index("\n}\n", i0) + 3]


ROLL_BEFORE = fn_of(text)
assert ROLL_BEFORE == fn_of(io.open(MODEL, "rb").read().decode("utf-8")) == fn_of(io.open(ERR, "rb").read().decode("utf-8")), "the rollout function of the source is not the one of the model queues"

# ---- head (comment block before `set -u`)
j = text.index("set -u\n")
head = ["#!/usr/bin/env bash",
        "# pod_croprule10k_queue.sh -- card ADD101 (CROPRULE10K): 8 rollouts of the label-fed form with closing speed (form RLc, the arguments of the banked arm I10RLcKF5P30Hb20: pre-roll 30 steps = 6 s, readout K,",
        "# --rule-every 5, --kappa-max 0.2, heading bound 0.2 1/m, model seeds 0 and 1) whose lead rule reads its lead from a JOIN WITH THE PILOT'S GAP given with --rule-join; the scoring keeps the evaluation join.",
        "# Arms KB5 (bound arm, head H5), KB1 (bound, H1), KD5 (fixed windows, H5), KD1 (fixed windows, H1) (joins $GP/joins/<ARM>_s<m>.npz from make_crop_joins.py, one per arm and model seed); run names I10RLcKF5P30Hb20p<ARM>_s<m>.",
        "# Wrapper v8f_t1_eval_reader_rule_rzgfhqetj.py (hash %s), admitted only with the pod check of control j2 of card ADD95 (CHECK_WRAPPER_J_DONE.txt last line PASS, its .hashes holding the wrapper hash)" % WRAP_SHA,
        "# and with CROP_JOINS_RECORD.json saying q0 PASS and q1 PASS with the hash of every join file and of the evaluation join. Banked runs needed: I10RLcKF5P30Hb20 and I10RQ50KF5P30Hb20 (RR), both seeds,",
        "# and I10RLcKF5P30Hb20mK100_s0 and _s1 (arm K100 of card ADD96, with T1_DONE).",
        "# GR_SEED=0|1|ALL (default ALL): one model seed per process with the markers START_CROPRULE10K_s<k>.txt, CROPRULE10K_DONE_s<k>.txt, CROPRULE10K_STOP_s<k>.txt in $GP and the log queue_croprule_s<k>.log.",
        "# DRY=1 prints every command and runs nothing. A rollout starts only with more than 28 GB free (memory.max minus anon) and fewer than 3 GPU compute processes.",
        "# Written by make_croprule_queue.py from the queue of card ADD97 (sha256 prefix %s). No training. No process listing or stopping, no file removal or move." % SRC_SHA_PREFIX]
text = "\n".join(head) + "\n" + text[j:]
N += 1

once("GP=${GR_ROOT:-/workspace/v8proof/smoothgap10k}\n", "GP=${GR_ROOT:-/workspace/v8proof/croprule10k}\n")
once("SM=$GP/small_sg", "SM=$GP/small_cr")
once("LOG=$GP/queue_smoothgap$PS.log", "LOG=$GP/queue_croprule$PS.log")
once('"$GP/SMOOTHGAP10K_STOP$PS.txt"', '"$GP/CROPRULE10K_STOP$PS.txt"')
once('$GP/START_SMOOTHGAP10K$PS.txt', '$GP/START_CROPRULE10K$PS.txt', 2)
once('log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD97 (SMOOTHGAP10K): label-fed rule with closing speed, lead read from a processed join (E, S50, S25, ES25), scoring on the evaluation join, at age"',
     'log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD101 (CROPRULE10K): label-fed rule with closing speed, lead read from a join with the pilot\'s gap (KB5, KB1, KD5, KD1), scoring on the evaluation join, at age"')
once("SMOOTH_JOINS_RECORD.json", "CROP_JOINS_RECORD.json", 2)
once("NO-SMOOTH-JOINS-RECORD", "NO-CROP-JOINS-RECORD")
once('for c in ("p0", "p1"):', 'for c in ("q0", "q1"):')
once('"E_s0,E_s1,S50_s0,S50_s1,S25_s0,S25_s1,ES25_s0,ES25_s1"', '"%s"' % ",".join("%s_s%d" % (a, s) for a in ARMS for s in (0, 1)))

# ---- the rollouts and the end
r0 = text.index('for k in "${ARMS[@]}"; do\n  RGRID=')
r1 = text.index('if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi')
blocks = []
for arm in ARMS:
    blocks.append('for k in "${ARMS[@]}"; do\n  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 RJOIN="$PJ/%s_s$k.npz" rollout I10RLcKF5P30Hb20p%s RLc "$k" "$k"\ndone\n' % (arm, arm))
text = text[:r0] + "".join(blocks) + text[r1:]
N += 1
once('NAMES="I10RLcKF5P30Hb20pE I10RLcKF5P30Hb20pS50 I10RLcKF5P30Hb20pS25 I10RLcKF5P30Hb20pES25"', 'NAMES="%s"' % " ".join("I10RLcKF5P30Hb20p" + a for a in ARMS))
once('echo "$(ts) all steps done (seeds $SEEDSW)" > "$GP/SMOOTHGAP10K_DONE$PS.txt"', 'echo "$(ts) all steps done (seeds $SEEDSW)" > "$GP/CROPRULE10K_DONE$PS.txt"')
once('log "SMOOTHGAP10K_DONE seeds $SEEDSW"', 'log "CROPRULE10K_DONE seeds $SEEDSW"')

# ---- guards
i0 = text.index("rollout() {\n")
i1 = text.index("\n}\n", i0) + 3
assert text[i0:i1] == ROLL_BEFORE, "the rollout function was edited"
assert fn_of(io.open(SRC, "rb").read().decode("utf-8")) == text[i0:i1] == fn_of(io.open(MODEL, "rb").read().decode("utf-8")) == fn_of(io.open(ERR, "rb").read().decode("utf-8"))
for bad in ("SMOOTH", "smooth", "ERRCUT", "errcut", "MIX_JOINS", "K50", "K25", "K12", "GAPRULE", "PLANTED", "planted", "small_sg", "S50", "ES25", 'RJOIN="$PJ/K1'):
    assert bad not in text, bad
assert text.count("rollout I10RLcKF5P30Hb20p") == 4 and "rollout I10RLcKF5P30Hb20m" not in text
outside = text[:i0] + text[i1:]
assert all(x.startswith("#") for x in outside.split("\n") if "--rule-join" in x or "--eval-join" in x), "the rule join may be named outside the rollout function only in comments"
assert text[i0:i1].count("--rule-join") == 1 and text[i0:i1].count("--eval-join $JOINE") == 1 and text[i0:i1].count('--eval-join "$JOINE"') == 1
assert "\r" not in text
text = text if text.endswith("\n") else text + "\n"
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d rollouts 8 edits %d (source %s, wrapper hash %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n"), N, src_sha, WRAP_SHA))
