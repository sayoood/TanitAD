"""make_visrule_queue.py -- writes pod_visrule10k_queue.sh from ../croprule/pod_croprule10k_queue.sh (sha256 prefix below) by asserted, counted text edits (card ADD102, VISRULE10K).
The queue of card ADD101 is itself the counted edit of the queue of card ADD97. Kept byte for byte: everything from `set -u` down to the end of the function rollout() except the lines named below,
the slot rule, the start marker with set -C, the DRY=1 mode, the timeout, the admission code, the banked-run checks.
Changed in rollout(): three inserted lines and one longer run-text line, all for the new arguments of the variant (--vis-gap-file from env VGAP, --vis-closing from env VCLOSE=1; both default off); the
function with those lines taken out is asserted equal to the function of the ADD101 queue, which is asserted equal to the ones of the ADD97, ADD96 and ADD95 queues.
New: root $GP default /workspace/v8proof/visrule10k, wrapper v8f_t1_eval_reader_rule_rzgfhqetjv.py (hash read from the file next to this script at the time of the edit and asserted in the queue), its pod check
$GP/check_v/CHECK_WRAPPER_V_DONE.txt, the gap files $GP/gapfiles/VGAP_s<m>.npz with VISRULE_GAPFILES_RECORD.json (v0 and build PASS, sha256 of both files and of the evaluation join), markers START_VISRULE10K_s<k>.txt,
VISRULE10K_DONE_s<k>.txt, VISRULE10K_STOP_s<k>.txt, log queue_visrule_s<k>.log, small folder $GP/small_vr, six rollouts of form RR (per model seed V0, VD5c, VD5; run folders $W/t1/I10RQ50KF5P30Hb20v<ARM>_s<m>).
Banked runs needed: RR (I10RQ50KF5P30Hb20), RLc (I10RLcKF5P30Hb20) and KD5 of card ADD101 (I10RLcKF5P30Hb20pKD5, with T1_DONE), both seeds.
Usage: python make_visrule_queue.py   (run in visrule/). Refuses to overwrite (VQ_OVERWRITE_OWN=1 rewrites it). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "croprule", "pod_croprule10k_queue.sh"))
SM_Q = os.path.normpath(os.path.join(HERE, "..", "smoothgap", "pod_smoothgap10k_queue.sh"))
ERR = os.path.normpath(os.path.join(HERE, "..", "errcut", "pod_errcut10k_queue.sh"))
MODEL = os.path.normpath(os.path.join(HERE, "..", "gaprule", "pod_gaprule10k_queue.sh"))
WRAPPER = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqetjv.py")
OUT = os.path.join(HERE, "pod_visrule10k_queue.sh")
SRC_SHA_PREFIX = "267699a0b9f130ad"
OLD_WRAP_SHA = "cbf7063d867c9f4e"
if not os.path.isfile(WRAPPER):
    sys.exit("REFUSED: the wrapper %s is not there (run make_wrapper_v.py first)" % WRAPPER)
WRAP_SHA = hashlib.sha256(io.open(WRAPPER, "rb").read()).hexdigest()[:16]
ARMS = ("V0", "VD5c", "VD5")
raw = io.open(SRC, "rb").read()
src_sha = hashlib.sha256(raw).hexdigest()[:16]
if src_sha != SRC_SHA_PREFIX:
    sys.exit("REFUSED: the source queue is not the one of card ADD101 (%s)" % src_sha)
if os.path.exists(OUT) and os.environ.get("VQ_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (VQ_OVERWRITE_OWN=1 rewrites it)" % OUT)
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
assert ROLL_BEFORE == fn_of(io.open(MODEL, "rb").read().decode("utf-8")) == fn_of(io.open(ERR, "rb").read().decode("utf-8")) == fn_of(io.open(SM_Q, "rb").read().decode("utf-8")), "the rollout function of the source is not the one of the earlier queues"

# ---- head (comment block before `set -u`)
j = text.index("set -u\n")
head = ["#!/usr/bin/env bash",
        "# pod_visrule10k_queue.sh -- card ADD102 (VISRULE10K): 6 rollouts of the reader-fed form RR (the arguments of the banked arm I10RQ50KF5P30Hb20: pre-roll 30 steps = 6 s, readout K, --rule-every 5, --kappa-max 0.2,",
        "# heading bound 0.2 1/m, model seeds 0 and 1) with the wrapper variant rzgfhqetjv: per model seed V0 (the new arguments off: control v1), VD5 (--vis-gap-file: where the reader marks a lead and the fixed-window head has an",
        "# output the rule receives the head's gap), VD5c (the same and --vis-closing: the closing input is the slope of the gap the rule receives by the run rule of the label source). Run names I10RQ50KF5P30Hb20v<ARM>_s<m>.",
        "# Gap files $GP/gapfiles/VGAP_s<m>.npz from make_vis_gapfile.py (one per model seed). Wrapper hash %s, admitted only with the pod check of the variant (CHECK_WRAPPER_V_DONE.txt last line PASS, its .hashes holding the" % WRAP_SHA,
        "# wrapper hash) and with VISRULE_GAPFILES_RECORD.json saying v0 PASS and build PASS with the hash of both gap files and of the evaluation join. Banked runs needed: I10RQ50KF5P30Hb20 (RR) and I10RLcKF5P30Hb20 (RLc),",
        "# both seeds, and I10RLcKF5P30Hb20pKD5_s0 and _s1 (arm KD5 of card ADD101, with T1_DONE).",
        "# GR_SEED=0|1|ALL (default ALL): one model seed per process with the markers START_VISRULE10K_s<k>.txt, VISRULE10K_DONE_s<k>.txt, VISRULE10K_STOP_s<k>.txt in $GP and the log queue_visrule_s<k>.log.",
        "# DRY=1 prints every command and runs nothing. A rollout starts only with more than 28 GB free (memory.max minus anon) and fewer than 3 GPU compute processes.",
        "# Written by make_visrule_queue.py from the queue of card ADD101 (sha256 prefix %s). No training. No process listing or stopping, no file removal or move." % SRC_SHA_PREFIX]
text = "\n".join(head) + "\n" + text[j:]
N += 1

once("GP=${GR_ROOT:-/workspace/v8proof/croprule10k}\n", "GP=${GR_ROOT:-/workspace/v8proof/visrule10k}\n")
once("WRAP=$B/v8f_t1_eval_reader_rule_rzgfhqetj.py\n", "WRAP=$B/v8f_t1_eval_reader_rule_rzgfhqetjv.py\n")
once("PJ=${GR_JOINS:-$GP/joins}\n", "GF=${GR_GAPFILES:-$GP/gapfiles}\n")
once("CHK=${GR_CHECK:-/workspace/v8proof/gaprule10kb/check_j2}\n", "CHK=${GR_CHECK:-$GP/check_v}\n")
once("SM=$GP/small_cr", "SM=$GP/small_vr")
once("LOG=$GP/queue_croprule$PS.log", "LOG=$GP/queue_visrule$PS.log")
once('"$GP/CROPRULE10K_STOP$PS.txt"', '"$GP/VISRULE10K_STOP$PS.txt"')
once('$GP/START_CROPRULE10K$PS.txt', '$GP/START_VISRULE10K$PS.txt', 2)
once('log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD101 (CROPRULE10K): label-fed rule with closing speed, lead read from a join with the pilot\'s gap (KB5, KB1, KD5, KD1), scoring on the evaluation join, at age"',
     'log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD102 (VISRULE10K): reader-fed rule RR with the fixed-window head\'s gap and the closing input of the label source\'s run rule (V0, VD5, VD5c), at age"')
once("for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzgfhqetj.py v8f_t1_eval_reader_rule_rzgfhqet.py", "for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzgfhqetjv.py v8f_t1_eval_reader_rule_rzgfhqetj.py v8f_t1_eval_reader_rule_rzgfhqet.py")

# ---- rollout(): the new arguments (three inserted lines and one longer run-text line)
INS1 = '  [ -n "${EVERY:-}" ] && extra+=(--rule-every "$EVERY")\n'
INS1_NEW = INS1
once('  if [ -n "${RJOIN:-}" ]; then\n', '  if [ -n "${VGAP:-}" ]; then\n    [ "$DRY" = 1 ] || [ -f "$VGAP" ] || stop "NO-GAP-FILE $VGAP $arm"\n    extra+=(--vis-gap-file "$VGAP")\n  fi\n'
     '  [ "${VCLOSE:-0}" = 1 ] && extra+=(--vis-closing)\n  if [ -n "${RJOIN:-}" ]; then\n')
once('${RJOIN:+ rule-join $RJOIN rule join sha256 $(sha256sum "$RJOIN" | cut -c1-64)}"',
     '${RJOIN:+ rule-join $RJOIN rule join sha256 $(sha256sum "$RJOIN" | cut -c1-64)}${VGAP:+ vis-gap-file $VGAP gap file sha256 $(sha256sum "$VGAP" | cut -c1-64) vis-closing ${VCLOSE:-0}}"')

# ---- the admission: record, join-hash checks and the check of the wrapper
once('for c in ("q0", "q1"):', 'for c in ("v0", "build"):')
once('(rec.get("arms", {}).get(a) or {}).get("sha256")', '(rec.get("files", {}).get(a) or {}).get("file_sha256")')
once('  CK="$CHK/CHECK_WRAPPER_J_DONE.txt"\n  [ "$(tail -n 1 "$CK" 2>/dev/null)" = "ZZ CHECK-WRAPPER-J PASS" ]', '  CK="$CHK/CHECK_WRAPPER_V_DONE.txt"\n  [ "$(tail -n 1 "$CK" 2>/dev/null)" = "ZZ CHECK-WRAPPER-V PASS" ]')
once('[ "$got" = "cbf7063d867c9f4e" ] || stop "WRAPPER-SHA $got"', '[ "$got" = "%s" ] || stop "WRAPPER-SHA $got"' % WRAP_SHA)
once('grep -q "cbf7063d867c9f4e" "$CHK/CHECK_WRAPPER_J.hashes"', 'grep -q "%s" "$CHK/CHECK_WRAPPER_V.hashes"' % WRAP_SHA)
once('  [ -f "$PJ/CROP_JOINS_RECORD.json" ] || stop "NO-CROP-JOINS-RECORD"\n', '  [ -f "$GF/VISRULE_GAPFILES_RECORD.json" ] || stop "NO-GAPFILES-RECORD"\n')
once('adm=$(admit "$PJ/CROP_JOINS_RECORD.json" "$PJ" "$JOINE" "KB5_s0,KB5_s1,KB1_s0,KB1_s1,KD5_s0,KD5_s1,KD1_s0,KD1_s1") || stop "JOIN-ADMISSION $adm"',
     'adm=$(admit "$GF/VISRULE_GAPFILES_RECORD.json" "$GF" "$JOINE" "VGAP_s0,VGAP_s1") || stop "GAPFILE-ADMISSION $adm"')
once('  for k in "${ARMS[@]}"; do\n    [ -f "$W/t1/I10RLcKF5P30Hb20mK100_s$k/T1_DONE" ] || stop "NO-BANKED-K100 s$k"\n  done\n',
     '  for k in "${ARMS[@]}"; do\n    [ -f "$W/t1/I10RLcKF5P30Hb20pKD5_s$k/T1_DONE" ] || stop "NO-BANKED-KD5 s$k"\n  done\n')

# ---- the rollouts and the end
r0 = text.index('for k in "${ARMS[@]}"; do\n  RGRID=')
r1 = text.index('if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi')
blocks = []
for arm in ARMS:
    env = 'RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"'
    if arm != "V0":
        env += ' VGAP="$GF/VGAP_s$k.npz"'
    if arm == "VD5c":
        env += " VCLOSE=1"
    blocks.append('for k in "${ARMS[@]}"; do\n  %s rollout I10RQ50KF5P30Hb20v%s RR "$k" "$k"\ndone\n' % (env, arm))
text = text[:r0] + "".join(blocks) + text[r1:]
N += 1
once('NAMES="I10RLcKF5P30Hb20pKB5 I10RLcKF5P30Hb20pKB1 I10RLcKF5P30Hb20pKD5 I10RLcKF5P30Hb20pKD1"', 'NAMES="%s"' % " ".join("I10RQ50KF5P30Hb20v" + a for a in ARMS))
once('echo "$(ts) all steps done (seeds $SEEDSW)" > "$GP/CROPRULE10K_DONE$PS.txt"', 'echo "$(ts) all steps done (seeds $SEEDSW)" > "$GP/VISRULE10K_DONE$PS.txt"')
once('log "CROPRULE10K_DONE seeds $SEEDSW"', 'log "VISRULE10K_DONE seeds $SEEDSW"')

# ---- guards
i0 = text.index("rollout() {\n")
i1 = text.index("\n}\n", i0) + 3
back = text[i0:i1]
for a_, b_ in (('  if [ -n "${VGAP:-}" ]; then\n    [ "$DRY" = 1 ] || [ -f "$VGAP" ] || stop "NO-GAP-FILE $VGAP $arm"\n    extra+=(--vis-gap-file "$VGAP")\n  fi\n  [ "${VCLOSE:-0}" = 1 ] && extra+=(--vis-closing)\n', ""),
               ('${VGAP:+ vis-gap-file $VGAP gap file sha256 $(sha256sum "$VGAP" | cut -c1-64) vis-closing ${VCLOSE:-0}}', "")):
    assert back.count(a_) == 1
    back = back.replace(a_, b_)
assert back == ROLL_BEFORE, "the rollout function differs from the ADD101 one beyond the new arguments"
for bad in ("CROPRULE", "croprule", "SMOOTH", "smooth", "ERRCUT", "errcut", "KB5", "KB1", "KD1", "K100", "GAPRULE", "PLANTED", "planted", "small_cr", "CROP_JOINS", "make_crop_joins", "$PJ"):
    assert bad not in text, bad
assert "rollout I10RLc" not in text and text.count("rollout I10RQ50KF5P30Hb20v") == 3
assert text[i0:i1].count("--rule-join") == 1 and text[i0:i1].count("--eval-join $JOINE") == 1 and text[i0:i1].count('--eval-join "$JOINE"') == 1
assert "RJOIN=" not in text
assert text.count("--vis-gap-file") == 2 and text[i0:i1].count("--vis-closing") == 1, "the new arguments are named in the rollout function only (and the header)"
assert "\r" not in text
text = text if text.endswith("\n") else text + "\n"
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d rollouts 6 edits %d (source %s, wrapper hash %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n"), N, src_sha, WRAP_SHA))
