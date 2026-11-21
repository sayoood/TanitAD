"""make_visrep_queue.py -- writes pod_visrep10k_queue.sh from ../visrule/pod_visrule10k_queue.sh (card ADD102, sha256 prefix below) by asserted, counted text edits (card ADD103, VISREP10K).
Kept byte for byte: everything from `set -u` down to the end of the function rollout() (the function is asserted equal to the one of the ADD102 queue, which carries the two arguments of the wrapper variant),
the slot rule, the start marker with set -C, the DRY=1 mode, the timeout, the checks of the build files, of the overlay tree, of the inputs and of the banked runs of the earlier cards.
Changed: root $GP /workspace/v8proof/visrep10k; the pod check record of card ADD102 ($CHK = /workspace/v8proof/visrule10k/check_v) and the wrapper hash 28bc31f21ecb4ccb are kept; the gap files of card ADD102 are
read from $G2 (/workspace/v8proof/visrule10k/gapfiles) and admitted with that card's record; the new gap files $GF/VGAP1_s<m>.npz are admitted with VISREP_GAPFILES_RECORD.json (v0, build, add102 and w0 PASS,
sha256 of both files and of the evaluation join); the banked runs of card ADD102 (V0, VD5, VD5c) join RR, RLc and KD5 as runs that must exist; the rollouts are VD5cx (closing switch, gap file of the other fit seed:
model seed k reads VGAP_s<1-k>), VRc (closing switch, no gap file) and VD1c (closing switch, gap file VGAP1_s<k>) of form RR, run names I10RQ50KF5P30Hb20w<ARM>_s<k>, markers START_VISREP10K_s<k>.txt,
VISREP10K_DONE_s<k>.txt, VISREP10K_STOP_s<k>.txt, log queue_visrep_s<k>.log, small folder $GP/small_vp.
Usage: python make_visrep_queue.py   (run in visrep/). Refuses to overwrite (VQ_OVERWRITE_OWN=1 rewrites it). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "visrule", "pod_visrule10k_queue.sh"))
WRAPPER = os.path.normpath(os.path.join(HERE, "..", "visrule", "v8f_t1_eval_reader_rule_rzgfhqetjv.py"))
OUT = os.path.join(HERE, "pod_visrep10k_queue.sh")
SRC_SHA_PREFIX = "6321d4e6e12c0f47"
WRAP_SHA = "28bc31f21ecb4ccb"
ARMS = ("VD5cx", "VRc", "VD1c")
if hashlib.sha256(io.open(WRAPPER, "rb").read()).hexdigest()[:16] != WRAP_SHA:
    sys.exit("REFUSED: the wrapper file on disk is not %s" % WRAP_SHA)
raw = io.open(SRC, "rb").read()
src_sha = hashlib.sha256(raw).hexdigest()[:16]
if src_sha != SRC_SHA_PREFIX:
    sys.exit("REFUSED: the source queue is not the one of card ADD102 (%s)" % src_sha)
if os.path.exists(OUT) and os.environ.get("VQ_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (VQ_OVERWRITE_OWN=1 rewrites it)" % OUT)
text = raw.decode("utf-8")
assert "\r" not in text
N = 0


def once(a, b, n=1):
    global text, N
    assert text.count(a) == n, (a[:90], text.count(a), n)
    text = text.replace(a, b)
    N += 1


def fn_of(t):
    i0 = t.index("rollout() {\n")
    return t[i0:t.index("\n}\n", i0) + 3]


ROLL_BEFORE = fn_of(text)
assert WRAP_SHA in text

# ---- head (comment block before `set -u`)
j = text.index("set -u\n")
head = ["#!/usr/bin/env bash",
        "# pod_visrep10k_queue.sh -- card ADD103 (VISREP10K): 6 rollouts of the reader-fed form RR (the arguments of the banked arm I10RQ50KF5P30Hb20: pre-roll 30 steps = 6 s, readout K, --rule-every 5, --kappa-max 0.2,",
        "# heading bound 0.2 1/m, model seeds 0 and 1) with the wrapper variant rzgfhqetjv of card ADD102 (hash %s, unchanged, admitted with the pod check record of card ADD102): per model seed k" % WRAP_SHA,
        "# VD5cx (the gap file of card ADD102 of the other fit seed, VGAP_s<1-k>, and --vis-closing), VRc (--vis-closing only: the rule receives the present reader's own gap and the slope of it as closing input) and",
        "# VD1c (the gap file VGAP1_s<k> of the head on one frame and --vis-closing). Run names I10RQ50KF5P30Hb20w<ARM>_s<k>. Gap files of card ADD102 in $G2, admitted with that card's VISRULE_GAPFILES_RECORD.json",
        "# (v0 and build PASS, sha256 of both files and of the evaluation join); new gap files $GP/gapfiles/VGAP1_s<k>.npz admitted with VISREP_GAPFILES_RECORD.json (v0, build, add102 and w0 PASS, the same hash checks).",
        "# Banked runs needed with T1_DONE or a dump folder: RR (I10RQ50KF5P30Hb20), RLc (I10RLcKF5P30Hb20), KD5 of card ADD101 and V0, VD5, VD5c of card ADD102 (I10RQ50KF5P30Hb20vV0, vVD5, vVD5c), both seeds.",
        "# GR_SEED=0|1|ALL (default ALL): one model seed per process with the markers START_VISREP10K_s<k>.txt, VISREP10K_DONE_s<k>.txt, VISREP10K_STOP_s<k>.txt in $GP and the log queue_visrep_s<k>.log.",
        "# DRY=1 prints every command and runs nothing. A rollout starts only with more than 28 GB free (memory.max minus anon) and fewer than 3 GPU compute processes.",
        "# Written by make_visrep_queue.py from the queue of card ADD102 (sha256 prefix %s). No training. No process listing or stopping, no file removal or move." % SRC_SHA_PREFIX]
text = "\n".join(head) + "\n" + text[j:]
N += 1

once("GP=${GR_ROOT:-/workspace/v8proof/visrule10k}\n", "GP=${GR_ROOT:-/workspace/v8proof/visrep10k}\n")
once("GF=${GR_GAPFILES:-$GP/gapfiles}\n", "GF=${GR_GAPFILES:-$GP/gapfiles}\nG2=${GR_GAP102:-/workspace/v8proof/visrule10k/gapfiles}\n")
once("CHK=${GR_CHECK:-$GP/check_v}\n", "CHK=${GR_CHECK:-/workspace/v8proof/visrule10k/check_v}\n")
once("SM=$GP/small_vr", "SM=$GP/small_vp")
once("LOG=$GP/queue_visrule$PS.log", "LOG=$GP/queue_visrep$PS.log")
once('"$GP/VISRULE10K_STOP$PS.txt"', '"$GP/VISREP10K_STOP$PS.txt"')
once('$GP/START_VISRULE10K$PS.txt', '$GP/START_VISREP10K$PS.txt', 2)
once('log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD102 (VISRULE10K): reader-fed rule RR with the fixed-window head\'s gap and the closing input of the label source\'s run rule (V0, VD5, VD5c), at age"',
     'log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD103 (VISREP10K): reader-fed rule RR with the closing input of the run rule: the gap of the other fit seed of the head on five frames (VD5cx), the reader\'s own gap (VRc), the gap of the head on one frame (VD1c), at age"')

# ---- the admission: records with the keys they must carry, the pod check of card ADD102 (path changed above), the banked runs
once("admit() { # RECORD JOINDIR EVALJOIN NAMES -> prints why on failure", "admit() { # RECORD JOINDIR EVALJOIN NAMES KEYS -> prints why on failure")
once('rec_path, jdir, ejoin, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")', 'rec_path, jdir, ejoin, names, keys = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","), sys.argv[5].split(",")')
once('for c in ("v0", "build"):', "for c in keys:")
once('  [ -f "$GF/VISRULE_GAPFILES_RECORD.json" ] || stop "NO-GAPFILES-RECORD"\n  adm=$(admit "$GF/VISRULE_GAPFILES_RECORD.json" "$GF" "$JOINE" "VGAP_s0,VGAP_s1") || stop "GAPFILE-ADMISSION $adm"\n  log "$adm"\n',
     '  [ -f "$G2/VISRULE_GAPFILES_RECORD.json" ] || stop "NO-GAPFILES-RECORD-ADD102"\n'
     '  adm=$(admit "$G2/VISRULE_GAPFILES_RECORD.json" "$G2" "$JOINE" "VGAP_s0,VGAP_s1" "v0,build") || stop "GAPFILE-ADMISSION-ADD102 $adm"\n'
     '  log "$adm"\n'
     '  [ -f "$GF/VISREP_GAPFILES_RECORD.json" ] || stop "NO-GAPFILES-RECORD"\n'
     '  adm=$(admit "$GF/VISREP_GAPFILES_RECORD.json" "$GF" "$JOINE" "VGAP1_s0,VGAP1_s1" "v0,build,add102,w0") || stop "GAPFILE-ADMISSION $adm"\n'
     '  log "$adm"\n')
once('  for k in "${ARMS[@]}"; do\n    [ -f "$W/t1/I10RLcKF5P30Hb20pKD5_s$k/T1_DONE" ] || stop "NO-BANKED-KD5 s$k"\n  done\n',
     '  for k in "${ARMS[@]}"; do\n    [ -f "$W/t1/I10RLcKF5P30Hb20pKD5_s$k/T1_DONE" ] || stop "NO-BANKED-KD5 s$k"\n  done\n'
     '  for nm in I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20vVD5 I10RQ50KF5P30Hb20vVD5c; do\n'
     '    for k in "${ARMS[@]}"; do [ -f "$W/t1/${nm}_s$k/T1_DONE" ] || stop "NO-BANKED-ADD102-RUN ${nm}_s$k"; done\n'
     '  done\n')

# ---- the rollouts and the end
r0 = text.index('for k in "${ARMS[@]}"; do\n  RGRID=')
r1 = text.index('if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi')
blocks = []
for arm in ARMS:
    env = 'RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"'
    if arm == "VD5cx":
        env += ' VGAP="$G2/VGAP_s$((1 - k)).npz"'
    if arm == "VD1c":
        env += ' VGAP="$GF/VGAP1_s$k.npz"'
    env += " VCLOSE=1"
    blocks.append('for k in "${ARMS[@]}"; do\n  %s rollout I10RQ50KF5P30Hb20w%s RR "$k" "$k"\ndone\n' % (env, arm))
text = text[:r0] + "".join(blocks) + text[r1:]
N += 1
once('NAMES="I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20vVD5c I10RQ50KF5P30Hb20vVD5"', 'NAMES="%s"' % " ".join("I10RQ50KF5P30Hb20w" + a for a in ARMS))
once('echo "$(ts) all steps done (seeds $SEEDSW)" > "$GP/VISRULE10K_DONE$PS.txt"', 'echo "$(ts) all steps done (seeds $SEEDSW)" > "$GP/VISREP10K_DONE$PS.txt"')
once('log "VISRULE10K_DONE seeds $SEEDSW"', 'log "VISREP10K_DONE seeds $SEEDSW"')

# ---- guards
assert fn_of(text) == ROLL_BEFORE, "the rollout function is not the one of the ADD102 queue"
for bad in ("START_VISRULE10K", "VISRULE10K_STOP", "VISRULE10K_DONE", "queue_visrule", "small_vr", "KB5", "KD1", "K100", "CROPRULE", "croprule", "RJOIN=", "$PJ", "rollout I10RQ50KF5P30Hb20v"):
    assert bad not in text, bad
assert text.count("rollout I10RQ50KF5P30Hb20w") == 3 and "rollout I10RLc" not in text
assert text.count("VCLOSE=1") == 3 and text.count("VGAP=") == 2
assert WRAP_SHA in text and text.count("28bc31f21ecb4ccb") >= 3
for w in ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop"):
    assert w not in text, w
assert "\r" not in text
text = text if text.endswith("\n") else text + "\n"
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d rollouts 6 edits %d (source %s, wrapper hash %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n"), N, src_sha, WRAP_SHA))
