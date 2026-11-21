"""make_visseed_queue.py -- writes pod_visseed10k_queue.sh from ../visrep/pod_visrep10k_queue.sh (card ADD103, sha256 prefix below) by asserted, counted text edits (card ADD104, VISSEED10K).
Kept byte for byte: everything from `set -u` down to the end of the function rollout() (asserted equal to the one of the ADD103 queue, which is the one of the ADD102 queue), the slot rule, the start marker with set -C,
the DRY=1 mode, the timeout, the checks of the build files, of the overlay tree, of the inputs, the wrapper admission of card ADD102 (hash 28bc31f21ecb4ccb and the pod check record in /workspace/v8proof/visrule10k/check_v).
Changed: root $GP /workspace/v8proof/visseed10k; the gap files of card ADD102 and ADD103 are not used (no $G2); the new gap files $GF/VGAPF_s<S>.npz (S = 2, 3, 4) are admitted with VISSEED_GAPFILES_RECORD.json (x0 and x1
PASS, sha256 of the three files and of the evaluation join); the banked runs that must exist with T1_DONE are V0 and VD5c of card ADD102 and VD5cx of card ADD103 (RR and RLc as dump folders, as before); the rollouts are
the VD5c call of the ADD102 queue (same environment, --vis-closing) with the gap file of fit seed S for each model seed k: names I10RQ50KF5P30Hb20fF<S>_s<k>; markers START_VISSEED10K_s<k>.txt, VISSEED10K_DONE_s<k>.txt,
VISSEED10K_STOP_s<k>.txt, log queue_visseed_s<k>.log, small folder $GP/small_vs.
Usage: python make_visseed_queue.py   (run in visseed/). Refuses to overwrite (VQ_OVERWRITE_OWN=1 rewrites it). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "visrep", "pod_visrep10k_queue.sh"))
SRC102 = os.path.normpath(os.path.join(HERE, "..", "visrule", "pod_visrule10k_queue.sh"))
WRAPPER = os.path.normpath(os.path.join(HERE, "..", "visrule", "v8f_t1_eval_reader_rule_rzgfhqetjv.py"))
OUT = os.path.join(HERE, "pod_visseed10k_queue.sh")
SRC_SHA_PREFIX = "ac22955967ea3cb7"
SRC102_SHA_PREFIX = "6321d4e6e12c0f47"
WRAP_SHA = "28bc31f21ecb4ccb"
FITS = (2, 3, 4)
if hashlib.sha256(io.open(WRAPPER, "rb").read()).hexdigest()[:16] != WRAP_SHA:
    sys.exit("REFUSED: the wrapper file on disk is not %s" % WRAP_SHA)
raw = io.open(SRC, "rb").read()
src_sha = hashlib.sha256(raw).hexdigest()[:16]
if src_sha != SRC_SHA_PREFIX:
    sys.exit("REFUSED: the source queue is not the one of card ADD103 (%s)" % src_sha)
raw102 = io.open(SRC102, "rb").read()
if hashlib.sha256(raw102).hexdigest()[:16] != SRC102_SHA_PREFIX:
    sys.exit("REFUSED: the queue of card ADD102 is not the expected file")
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
assert ROLL_BEFORE == fn_of(raw102.decode("utf-8"))
assert WRAP_SHA in text

# ---- head (comment block before `set -u`)
j = text.index("set -u\n")
head = ["#!/usr/bin/env bash",
        "# pod_visseed10k_queue.sh -- card ADD104 (VISSEED10K): 6 rollouts of the reader-fed form RR (the arguments of the banked arm I10RQ50KF5P30Hb20: pre-roll 30 steps = 6 s, readout K, --rule-every 5, --kappa-max 0.2,",
        "# heading bound 0.2 1/m, model seeds 0 and 1) with the wrapper variant rzgfhqetjv of card ADD102 (hash %s, unchanged, admitted with the pod check record of card ADD102): per model seed k" % WRAP_SHA,
        "# the three arms F2, F3, F4 = the VD5c call of the ADD102 queue (--vis-closing) with the gap file VGAPF_s<S> of the head on five frames of fit seed S = 2, 3, 4 (the same file for both model seeds).",
        "# Run names I10RQ50KF5P30Hb20fF<S>_s<k>. The gap files $GP/gapfiles/VGAPF_s<S>.npz are admitted with VISSEED_GAPFILES_RECORD.json (x0 and x1 PASS, sha256 of the three files and of the evaluation join).",
        "# Banked runs needed with T1_DONE or a dump folder: RR (I10RQ50KF5P30Hb20), RLc (I10RLcKF5P30Hb20), V0 and VD5c of card ADD102 (I10RQ50KF5P30Hb20vV0, vVD5c), VD5cx of card ADD103 (I10RQ50KF5P30Hb20wVD5cx), both seeds.",
        "# GR_SEED=0|1|ALL (default ALL): one model seed per process with the markers START_VISSEED10K_s<k>.txt, VISSEED10K_DONE_s<k>.txt, VISSEED10K_STOP_s<k>.txt in $GP and the log queue_visseed_s<k>.log.",
        "# DRY=1 prints every command and runs nothing. A rollout starts only with more than 28 GB free (memory.max minus anon) and fewer than 3 GPU compute processes.",
        "# Written by make_visseed_queue.py from the queue of card ADD103 (sha256 prefix %s). No training. No process listing or stopping, no file removal or move." % SRC_SHA_PREFIX]
text = "\n".join(head) + "\n" + text[j:]
N += 1

once("GP=${GR_ROOT:-/workspace/v8proof/visrep10k}\n", "GP=${GR_ROOT:-/workspace/v8proof/visseed10k}\n")
once("G2=${GR_GAP102:-/workspace/v8proof/visrule10k/gapfiles}\n", "")
once("SM=$GP/small_vp", "SM=$GP/small_vs")
once("LOG=$GP/queue_visrep$PS.log", "LOG=$GP/queue_visseed$PS.log")
once('"$GP/VISREP10K_STOP$PS.txt"', '"$GP/VISSEED10K_STOP$PS.txt"')
once('$GP/START_VISREP10K$PS.txt', '$GP/START_VISSEED10K$PS.txt', 2)
once('log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD103 (VISREP10K): reader-fed rule RR with the closing input of the run rule: the gap of the other fit seed of the head on five frames (VD5cx), the reader\'s own gap (VRc), the gap of the head on one frame (VD1c), at age"',
     'log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD104 (VISSEED10K): reader-fed rule RR with the closing input of the run rule and the gap of the head on five frames of fit seeds 2, 3 and 4 (F2, F3, F4), at age"')

# ---- the admission
once('  [ -f "$G2/VISRULE_GAPFILES_RECORD.json" ] || stop "NO-GAPFILES-RECORD-ADD102"\n'
     '  adm=$(admit "$G2/VISRULE_GAPFILES_RECORD.json" "$G2" "$JOINE" "VGAP_s0,VGAP_s1" "v0,build") || stop "GAPFILE-ADMISSION-ADD102 $adm"\n'
     '  log "$adm"\n'
     '  [ -f "$GF/VISREP_GAPFILES_RECORD.json" ] || stop "NO-GAPFILES-RECORD"\n'
     '  adm=$(admit "$GF/VISREP_GAPFILES_RECORD.json" "$GF" "$JOINE" "VGAP1_s0,VGAP1_s1" "v0,build,add102,w0") || stop "GAPFILE-ADMISSION $adm"\n'
     '  log "$adm"\n',
     '  [ -f "$GF/VISSEED_GAPFILES_RECORD.json" ] || stop "NO-GAPFILES-RECORD"\n'
     '  adm=$(admit "$GF/VISSEED_GAPFILES_RECORD.json" "$GF" "$JOINE" "VGAPF_s2,VGAPF_s3,VGAPF_s4" "x0,x1") || stop "GAPFILE-ADMISSION $adm"\n'
     '  log "$adm"\n')
once('  for k in "${ARMS[@]}"; do\n    [ -f "$W/t1/I10RLcKF5P30Hb20pKD5_s$k/T1_DONE" ] || stop "NO-BANKED-KD5 s$k"\n  done\n', "")
once('  for nm in I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20vVD5 I10RQ50KF5P30Hb20vVD5c; do\n    for k in "${ARMS[@]}"; do [ -f "$W/t1/${nm}_s$k/T1_DONE" ] || stop "NO-BANKED-ADD102-RUN ${nm}_s$k"; done\n',
     '  for nm in I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20vVD5c I10RQ50KF5P30Hb20wVD5cx; do\n    for k in "${ARMS[@]}"; do [ -f "$W/t1/${nm}_s$k/T1_DONE" ] || stop "NO-BANKED-RUN-OF-ADD102-OR-ADD103 ${nm}_s$k"; done\n')

# ---- the rollouts and the end
r0 = text.index('for k in "${ARMS[@]}"; do\n  RGRID=')
r1 = text.index('if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi')
env0 = 'RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"'
base102 = [x for x in raw102.decode("utf-8").splitlines() if "rollout I10RQ50KF5P30Hb20vVD5c RR" in x]
assert len(base102) == 1 and base102[0].strip().startswith(env0 + ' VGAP="$GF/VGAP_s$k.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20vVD5c RR'), base102
blocks = ['for k in "${ARMS[@]}"; do\n  %s VGAP="$GF/VGAPF_s%d.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20fF%d RR "$k" "$k"\ndone\n' % (env0, s, s) for s in FITS]
text = text[:r0] + "".join(blocks) + text[r1:]
N += 1
once('NAMES="I10RQ50KF5P30Hb20wVD5cx I10RQ50KF5P30Hb20wVRc I10RQ50KF5P30Hb20wVD1c"', 'NAMES="%s"' % " ".join("I10RQ50KF5P30Hb20fF%d" % s for s in FITS))
once('echo "$(ts) all steps done (seeds $SEEDSW)" > "$GP/VISREP10K_DONE$PS.txt"', 'echo "$(ts) all steps done (seeds $SEEDSW)" > "$GP/VISSEED10K_DONE$PS.txt"')
once('log "VISREP10K_DONE seeds $SEEDSW"', 'log "VISSEED10K_DONE seeds $SEEDSW"')

# ---- guards
assert fn_of(text) == ROLL_BEFORE, "the rollout function is not the one of the ADD103 queue"
for bad in ("VISREP", "visrep", "VISRULE10K_", "queue_visrule", "$G2", "G2=", "VGAP1", "VGAP_s", "KD5", "pKD5", "VD1c", "VRc", "RJOIN=", "$PJ", "rollout I10RQ50KF5P30Hb20v", "rollout I10RQ50KF5P30Hb20w"):
    assert bad not in text, bad
assert text.count("rollout I10RQ50KF5P30Hb20fF") == 3 and "rollout I10RLc" not in text
assert text.count("VCLOSE=1") == 3 and text.count("VGAP=") == 3
assert WRAP_SHA in text and text.count("28bc31f21ecb4ccb") >= 3
for w in ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop"):
    assert w not in text, w
assert "\r" not in text
text = text if text.endswith("\n") else text + "\n"
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d rollouts 6 edits %d (source %s, wrapper hash %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n"), N, src_sha, WRAP_SHA))
