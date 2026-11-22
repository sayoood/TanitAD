"""make_cropscale_queue.py -- writes pod_cropscale10k_queue.sh from ../visseed/pod_visseed10k_queue.sh (card ADD104, sha256 prefix below) by asserted, counted text edits (card ADD106, CROPSCALE10K).
Kept byte for byte: everything from `set -u` down to the end of the function rollout() (asserted equal to the one of the ADD104 queue, of the ADD105 queue and of the ADD102 queue), the slot rule, the start marker
with set -C, the DRY=1 mode, the timeout, the checks of the build files, of the overlay tree, of the inputs, the wrapper admission of card ADD102 (hash 28bc31f21ecb4ccb and the pod check record in
/workspace/v8proof/visrule10k/check_v), the check of the banked runs RR and RLc (dump folders).
Changed: root $GP /workspace/v8proof/cropscale; the gap files $GF/VGAPL_s2, VGAPL_s3, VGAPL_s4, VGAPLE, VGAPSE3 are admitted with CROPSCALE_GAPFILES_RECORD.json (g2 and g3 PASS, sha256 of the five files and of the
evaluation join); the banked runs that must exist with T1_DONE are V0 and VD5c of card ADD102, F2, F3, F4 of card ADD104 and E5c of card ADD105 (both model seeds); ten rollouts = the VD5c call of the ADD102 queue
(same environment, --vis-closing) with one gap file each, names I10RQ50KF5P30Hb20g<ARM>_s<k>; markers START_CROPSCALE10K_s<k>.txt, CROPSCALE10K_DONE_s<k>.txt, CROPSCALE10K_STOP_s<k>.txt, log queue_cropscale_s<k>.log,
small folder $GP/small_cs.
Usage: python make_cropscale_queue.py   (run in cropscale/fit/). Refuses to overwrite (VQ_OVERWRITE_OWN=1 rewrites it). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "visseed", "pod_visseed10k_queue.sh"))
SRC105 = os.path.normpath(os.path.join(HERE, "..", "..", "visens", "pod_visens10k_queue.sh"))
SRC102 = os.path.normpath(os.path.join(HERE, "..", "..", "visrule", "pod_visrule10k_queue.sh"))
WRAPPER = os.path.normpath(os.path.join(HERE, "..", "..", "visrule", "v8f_t1_eval_reader_rule_rzgfhqetjv.py"))
OUT = os.path.join(HERE, "pod_cropscale10k_queue.sh")
SRC_SHA_PREFIX = "ba9c4bf35430e42f"
SRC105_SHA_PREFIX = "ee9d4f9d232b04c2"
SRC102_SHA_PREFIX = "6321d4e6e12c0f47"
WRAP_SHA = "28bc31f21ecb4ccb"
ARMS = (("G2", "VGAPL_s2"), ("G3", "VGAPL_s3"), ("G4", "VGAPL_s4"), ("GEc", "VGAPLE"), ("E3c", "VGAPSE3"))
for p, pre, what in ((WRAPPER, WRAP_SHA, "the wrapper file"), (SRC, SRC_SHA_PREFIX, "the source queue (card ADD104)"), (SRC105, SRC105_SHA_PREFIX, "the queue of card ADD105"), (SRC102, SRC102_SHA_PREFIX, "the queue of card ADD102")):
    if hashlib.sha256(io.open(p, "rb").read()).hexdigest()[:16] != pre:
        sys.exit("REFUSED: %s on disk is not %s" % (what, pre))
raw = io.open(SRC, "rb").read()
src_sha = hashlib.sha256(raw).hexdigest()[:16]
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
assert ROLL_BEFORE == fn_of(io.open(SRC102, "rb").read().decode("utf-8")) == fn_of(io.open(SRC105, "rb").read().decode("utf-8"))
assert WRAP_SHA in text

# ---- head (comment block before `set -u`)
j = text.index("set -u\n")
head = ["#!/usr/bin/env bash",
        "# pod_cropscale10k_queue.sh -- card ADD106 (CROPSCALE10K): 10 rollouts of the reader-fed form RR (the arguments of the banked arm I10RQ50KF5P30Hb20: pre-roll 30 steps = 6 s, readout K, --rule-every 5, --kappa-max 0.2,",
        "# heading bound 0.2 1/m, model seeds 0 and 1) with the wrapper variant rzgfhqetjv of card ADD102 (hash %s, unchanged, admitted with the pod check record of card ADD102): per model seed k" % WRAP_SHA,
        "# the five arms G2, G3, G4, GEc, E3c in this order = the VD5c call of the ADD102 queue (--vis-closing) with the gap file VGAPL_s2, VGAPL_s3, VGAPL_s4 (the dumped gap of the head on five frames fit on the large fit set,",
        "# fit seeds 2, 3, 4), VGAPLE (the mean of those three) and VGAPSE3 (the mean of the three 500-clip fits of seeds 2, 3, 4); the same file for both model seeds.",
        "# Run names I10RQ50KF5P30Hb20g<ARM>_s<k>. The gap files $GP/gapfiles/<name>.npz are admitted with CROPSCALE_GAPFILES_RECORD.json (g2 and g3 PASS, sha256 of the five files and of the evaluation join).",
        "# Banked runs needed with T1_DONE or a dump folder: RR (I10RQ50KF5P30Hb20), RLc (I10RLcKF5P30Hb20), V0 and VD5c of card ADD102 (I10RQ50KF5P30Hb20vV0, vVD5c), F2, F3, F4 of card ADD104 (I10RQ50KF5P30Hb20fF<S>),",
        "# E5c of card ADD105 (I10RQ50KF5P30Hb20eE5c), both seeds.",
        "# GR_SEED=0|1|ALL (default ALL): one model seed per process with the markers START_CROPSCALE10K_s<k>.txt, CROPSCALE10K_DONE_s<k>.txt, CROPSCALE10K_STOP_s<k>.txt in $GP and the log queue_cropscale_s<k>.log.",
        "# DRY=1 prints every command and runs nothing. A rollout starts only with more than 28 GB free (memory.max minus anon) and fewer than 3 GPU compute processes.",
        "# Written by make_cropscale_queue.py from the queue of card ADD104 (sha256 prefix %s). No training. No process listing or stopping, no file removal or move." % SRC_SHA_PREFIX]
text = "\n".join(head) + "\n" + text[j:]
N += 1

once("GP=${GR_ROOT:-/workspace/v8proof/visseed10k}\n", "GP=${GR_ROOT:-/workspace/v8proof/cropscale}\n")
once("SM=$GP/small_vs", "SM=$GP/small_cs")
once("LOG=$GP/queue_visseed$PS.log", "LOG=$GP/queue_cropscale$PS.log")
once('"$GP/VISSEED10K_STOP$PS.txt"', '"$GP/CROPSCALE10K_STOP$PS.txt"')
once('$GP/START_VISSEED10K$PS.txt', '$GP/START_CROPSCALE10K$PS.txt', 2)
once('log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD104 (VISSEED10K): reader-fed rule RR with the closing input of the run rule and the gap of the head on five frames of fit seeds 2, 3 and 4 (F2, F3, F4), at age"',
     'log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD106 (CROPSCALE10K): reader-fed rule RR with the closing input of the run rule and the gap of the head on five frames fit on the large fit set (G2, G3, G4, GEc) and the mean of the 500-clip fits of seeds 2, 3, 4 (E3c), at age"')

# ---- the admission
once('  [ -f "$GF/VISSEED_GAPFILES_RECORD.json" ] || stop "NO-GAPFILES-RECORD"\n'
     '  adm=$(admit "$GF/VISSEED_GAPFILES_RECORD.json" "$GF" "$JOINE" "VGAPF_s2,VGAPF_s3,VGAPF_s4" "x0,x1") || stop "GAPFILE-ADMISSION $adm"\n',
     '  [ -f "$GF/CROPSCALE_GAPFILES_RECORD.json" ] || stop "NO-GAPFILES-RECORD"\n'
     '  adm=$(admit "$GF/CROPSCALE_GAPFILES_RECORD.json" "$GF" "$JOINE" "VGAPL_s2,VGAPL_s3,VGAPL_s4,VGAPLE,VGAPSE3" "g2,g3") || stop "GAPFILE-ADMISSION $adm"\n')
once('  for nm in I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20vVD5c I10RQ50KF5P30Hb20wVD5cx; do\n    for k in "${ARMS[@]}"; do [ -f "$W/t1/${nm}_s$k/T1_DONE" ] || stop "NO-BANKED-RUN-OF-ADD102-OR-ADD103 ${nm}_s$k"; done\n',
     '  for nm in I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20vVD5c I10RQ50KF5P30Hb20fF2 I10RQ50KF5P30Hb20fF3 I10RQ50KF5P30Hb20fF4 I10RQ50KF5P30Hb20eE5c; do\n'
     '    for k in "${ARMS[@]}"; do [ -f "$W/t1/${nm}_s$k/T1_DONE" ] || stop "NO-BANKED-RUN-OF-ADD102-ADD104-OR-ADD105 ${nm}_s$k"; done\n')

# ---- the rollouts and the end
r0 = text.index('for k in "${ARMS[@]}"; do\n  RGRID=')
r1 = text.index('if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi')
env0 = 'RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"'
base102 = [x for x in io.open(SRC102, "rb").read().decode("utf-8").splitlines() if "rollout I10RQ50KF5P30Hb20vVD5c RR" in x]
assert len(base102) == 1 and base102[0].strip().startswith(env0 + ' VGAP="$GF/VGAP_s$k.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20vVD5c RR'), base102
assert text[r0:r1].count("rollout I10RQ50KF5P30Hb20fF") == 3
blocks = ['for k in "${ARMS[@]}"; do\n  %s VGAP="$GF/%s.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20g%s RR "$k" "$k"\ndone\n' % (env0, gf, a) for a, gf in ARMS]
text = text[:r0] + "".join(blocks) + text[r1:]
N += 1
once('NAMES="I10RQ50KF5P30Hb20fF2 I10RQ50KF5P30Hb20fF3 I10RQ50KF5P30Hb20fF4"', 'NAMES="%s"' % " ".join("I10RQ50KF5P30Hb20g%s" % a for a, _ in ARMS))
once('echo "$(ts) all steps done (seeds $SEEDSW)" > "$GP/VISSEED10K_DONE$PS.txt"', 'echo "$(ts) all steps done (seeds $SEEDSW)" > "$GP/CROPSCALE10K_DONE$PS.txt"')
once('log "VISSEED10K_DONE seeds $SEEDSW"', 'log "CROPSCALE10K_DONE seeds $SEEDSW"')

# ---- guards
assert fn_of(text) == ROLL_BEFORE, "the rollout function is not the one of the ADD104 queue"
for bad in ("VISSEED", "visseed", "VISREP", "visrep", "VISRULE10K_", "queue_visrule", "$G2", "G2=", "VGAP1", "VGAP_s", "VGAPF", "wVD5cx", "KD5", "pKD5", "VD1c", "VRc", "RJOIN=", "$PJ",
            "rollout I10RQ50KF5P30Hb20v", "rollout I10RQ50KF5P30Hb20w", "rollout I10RQ50KF5P30Hb20f", "rollout I10RQ50KF5P30Hb20e"):
    assert bad not in text, bad
assert text.count("rollout I10RQ50KF5P30Hb20g") == 5 and "rollout I10RLc" not in text
assert text.count("VCLOSE=1") == 5 and text.count("VGAP=") == 5 and text.count("VGAPL_s2") >= 2
assert WRAP_SHA in text and text.count(WRAP_SHA) >= 3
for w in ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop"):
    assert w not in text, w
assert "\r" not in text
text = text if text.endswith("\n") else text + "\n"
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d rollouts 10 edits %d (source %s, wrapper hash %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n"), N, src_sha, WRAP_SHA))
