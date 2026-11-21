"""make_gaprule_queue.py -- writes pod_gaprule10k_queue.sh from ../smoothreq/pod_smoothreq10k_queue.sh (sha256 prefix below) by counted text edits (card ADD94, GAPRULE10K).
Kept line for line: the slot rule (two probes 45 s apart, fewer than 3 GPU compute processes, container memory free above 28 GB = memory.max minus the anon line), the start marker with set -C,
the DRY=1 mode, the timeout, the stop and done markers and the rollout function. New: root $GP = /workspace/v8proof/gaprule10k (planted joins, record, markers, logs, small folder; the rollouts
themselves go to $W/t1 like the model queue so that the pull of the reader is the same), the wrapper of the j generation, the new environment variable RJOIN (adds --rule-join), the admission
(planted joins record with j1 PASS and the hash of every file, the pod check of the wrapper j2 with the hash of the wrapper), the banked runs, and the twelve rollouts
(arm names I10RLcKF5P30Hb20j<ARM>; the command of the banked arm I10RLcKF5P30Hb20 of the queue of card ADD70 plus --rule-join).
Usage: python make_gaprule_queue.py   (run in gaprule/). Refuses to overwrite (GR_OVERWRITE_OWN=1 rewrites it). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "smoothreq", "pod_smoothreq10k_queue.sh"))
OUT = os.path.join(HERE, "pod_gaprule10k_queue.sh")
WRAPJ = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqetj.py")
SRC_SHA = "c9622ba2c86eb428"
ARMS = ("G0", "GR1", "GR2", "GW05", "GC10", "GC20")
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != SRC_SHA:
    sys.exit("REFUSED: the source queue is not the one of card ADD85 (%s)" % hashlib.sha256(raw).hexdigest()[:16])
if not os.path.isfile(WRAPJ):
    sys.exit("REFUSED: run make_wrapper_j.py first")
WRAPJ_SHA = hashlib.sha256(io.open(WRAPJ, "rb").read()).hexdigest()[:16]
if os.path.exists(OUT) and os.environ.get("GR_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (GR_OVERWRITE_OWN=1 rewrites it)" % OUT)
L = raw.decode("utf-8").split("\n")
i_set = L.index("set -u")
i_roll = next(i for i, x in enumerate(L) if x.startswith("rollout() {"))
i_end = next(i for i in range(i_roll, len(L)) if L[i] == "}")
text = "\n".join(L[i_set:i_end + 1])
N = 0


def once(a, b, n=1):
    global text, N
    assert text.count(a) == n, (a, text.count(a), n)
    text = text.replace(a, b)
    N += 1


once("W=${I10_W:-/workspace/v8proof/integ10k}\n", "W=${I10_W:-/workspace/v8proof/integ10k}\nGP=${GR_ROOT:-/workspace/v8proof/gaprule10k}\n")
once("TABLES=${I10_TABLES:-$V/closeread/est_smooth}\n", "PJ=${GR_JOINS:-$GP/joins}\nCHK=${GR_CHECK:-$GP/check_j}\n")
once("WRAP=$B/v8f_t1_eval_reader_rule_rzgfhqet.py", "WRAP=$B/v8f_t1_eval_reader_rule_rzgfhqetj.py")
once("SM=$W/small_sr", "SM=$GP/small_gr")
once("SR_SEED", "GR_SEED", 2)
once("SMOOTHREQ10K", "GAPRULE10K", 4)
once("queue_smoothreq", "queue_gaprule")
once('"$W/GAPRULE10K_STOP$PS.txt"', '"$GP/GAPRULE10K_STOP$PS.txt"')
once('LOG=$W/queue_gaprule$PS.log', 'LOG=$GP/queue_gaprule$PS.log')
once('$W/START_GAPRULE10K$PS.txt', '$GP/START_GAPRULE10K$PS.txt', 2)
once('"$W/markers/$1.done"', '"$GP/markers/$1.done"', 2)
once('mkdir -p "$W/markers" "$W/t1" "$SM"', 'mkdir -p "$GP/markers" "$W/t1" "$SM"')
once('  [ -d "$W" ] || { echo "REFUSED: no $W"; exit 2; }\n', '  [ -d "$W" ] || { echo "REFUSED: no $W"; exit 2; }\n  mkdir -p "$GP" || { echo "REFUSED: cannot make $GP"; exit 2; }\n')
once('log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD85 (GAPRULE10K): rollouts of form RRe whose closing input is a table (smooth and white planted errors, Z00, STKM, TKD), at age"',
     'log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD94 (GAPRULE10K): label-fed rule with closing speed, lead read from a planted join (G0, GR1, GR2, GW05, GC10, GC20), scoring on the evaluation join, at age"')
once("v8f_t1_eval_reader_rule_rzgfhqet.py v8f_t1_eval_reader_rule_rzgfhqe.py", "v8f_t1_eval_reader_rule_rzgfhqetj.py v8f_t1_eval_reader_rule_rzgfhqet.py v8f_t1_eval_reader_rule_rzgfhqe.py")
once('    [ -f "$W/lr16/TK16_s$k/ckpt_last.pt" ] || stop "NO-READER TK16_s$k"\n    [ -f "$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" ] || stop "NO-MAP-Q50 s$k"\n', "")
once('  [ "$form" = R0 ] || extra+=(--rule-log "$out/ticks/rule_log_$arm.npz")\n',
     '  [ "$form" = R0 ] || extra+=(--rule-log "$out/ticks/rule_log_$arm.npz")\n'
     '  if [ -n "${RJOIN:-}" ]; then\n    [ "$DRY" = 1 ] || [ -f "$RJOIN" ] || stop "NO-RULE-JOIN $RJOIN $arm"\n    extra+=(--rule-join "$RJOIN")\n  fi\n')
once('${TAB:+ closing-table $TAB table sha256 $(sha256sum "$TAB" | cut -c1-64)}', '${RJOIN:+ rule-join $RJOIN rule join sha256 $(sha256sum "$RJOIN" | cut -c1-64)}')
for bad in ("smooth", "SMOOTH", "ADD85", "tabclose"):
    assert bad not in text, bad

ADMIT_PY = r'''import hashlib, json, os, sys
rec_path, jdir, ejoin, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
rec = json.load(open(rec_path, encoding="utf-8"))
if not isinstance(rec.get("j1"), dict) or rec["j1"].get("PASS") is not True:
    print("J1-NOT-PASS"); sys.exit(1)
h = hashlib.sha256(open(ejoin, "rb").read()).hexdigest()
if rec.get("eval_join_sha256") != h:
    print("RECORD-IS-FOR-ANOTHER-EVAL-JOIN"); sys.exit(1)
for a in arms:
    fn = os.path.join(jdir, a + ".npz")
    if not os.path.isfile(fn):
        print("NO-JOIN " + a); sys.exit(1)
    if hashlib.sha256(open(fn, "rb").read()).hexdigest() != (rec.get("arms", {}).get(a) or {}).get("sha256"):
        print("JOIN-HASH-DIFFERS " + a); sys.exit(1)
print("ADMIT-OK joins %d" % len(arms))
'''
head = ["#!/usr/bin/env bash",
        "# pod_gaprule10k_queue.sh -- card ADD94 (GAPRULE10K): 12 rollouts of the label-fed form with closing speed (form RLc, the arguments of the banked arm I10RLcKF5P30Hb20: pre-roll 30 steps = 6 s, readout K,",
        "# --rule-every 5, --kappa-max 0.2, heading bound 0.2 1/m, model seeds 0 and 1) whose lead rule reads its lead from a PLANTED JOIN given with --rule-join; the scoring keeps the evaluation join.",
        "# Arms G0, GR1, GR2, GW05, GC10, GC20 (joins $GP/joins/<ARM>.npz from make_planted_joins.py); run names I10RLcKF5P30Hb20j<ARM>_s<k>.",
        "# Wrapper v8f_t1_eval_reader_rule_rzgfhqetj.py (hash %s), admitted only with the pod check of control j2 (CHECK_WRAPPER_J_DONE.txt last line PASS, its .hashes holding the wrapper hash)" % WRAPJ_SHA,
        "# and with PLANTED_JOINS_RECORD.json saying j1 PASS with the hash of every join file and of the evaluation join. Banked runs needed: I10RLcKF5P30Hb20 and I10RQ50KF5P30Hb20 (RR), both seeds.",
        "# GR_SEED=0|1|ALL (default ALL): one model seed per process with the markers START_GAPRULE10K_s<k>.txt, GAPRULE10K_DONE_s<k>.txt, GAPRULE10K_STOP_s<k>.txt in $GP and the log queue_gaprule_s<k>.log.",
        "# DRY=1 prints every command and runs nothing. A rollout starts only with more than 28 GB free (memory.max minus anon) and fewer than 3 GPU compute processes.",
        "# Written by make_gaprule_queue.py from the queue of card ADD85 (smoothreq/pod_smoothreq10k_queue.sh). No training. No process listing or stopping, no file removal or move."]
tail = ["",
        "admit() { # RECORD JOINDIR EVALJOIN ARMS -> prints why on failure",
        '  "$PY" - "$@" <<\'PYEOF\'',
        ADMIT_PY.rstrip("\n"),
        "PYEOF",
        "}",
        '[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"',
        'if [ "$DRY" != 1 ]; then',
        '  CK="$CHK/CHECK_WRAPPER_J_DONE.txt"',
        '  [ "$(tail -n 1 "$CK" 2>/dev/null)" = "ZZ CHECK-WRAPPER-J PASS" ] || stop "NO-PASS-OF-THE-WRAPPER-CHECK"',
        '  got=$(sha256sum "$WRAP" | cut -c1-16); [ "$got" = "%s" ] || stop "WRAPPER-SHA $got"' % WRAPJ_SHA,
        '  grep -q "%s" "$CHK/CHECK_WRAPPER_J.hashes" 2>/dev/null || stop "CHECK-RECORD-HAS-NOT-THIS-WRAPPER"' % WRAPJ_SHA,
        '  [ -f "$PJ/PLANTED_JOINS_RECORD.json" ] || stop "NO-PLANTED-JOINS-RECORD"',
        '  adm=$(admit "$PJ/PLANTED_JOINS_RECORD.json" "$PJ" "$JOINE" "%s") || stop "JOIN-ADMISSION $adm"' % ",".join(ARMS),
        '  log "$adm"',
        '  for nm in I10RLcKF5P30Hb20 I10RQ50KF5P30Hb20; do',
        '    for k in "${ARMS[@]}"; do [ -d "$W/t1/${nm}_s$k/dump_${nm}_s${k}_m10_causal_heads" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done',
        "  done",
        "fi"]
names = []
for arm in ARMS:
    nm = "I10RLcKF5P30Hb20j%s" % arm
    tail += ['for k in "${ARMS[@]}"; do',
             "  RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 " + 'RJOIN="$PJ/%s.npz" rollout %s RLc "$k" "$k"' % (arm, nm),
             "done"]
    names.append(nm)
tail += ['if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi',
         'cp -f "$LOG" "$SM/" 2>/dev/null',
         'NAMES="%s"' % " ".join(names),
         'for nm in $NAMES; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done',
         'echo "$(ts) all steps done (seeds $SEEDSW)" > "$GP/GAPRULE10K_DONE$PS.txt"',
         'log "GAPRULE10K_DONE seeds $SEEDSW"',
         ""]
out = "\n".join(head) + "\n" + text + "\n" + "\n".join(tail)
assert "\r" not in out
io.open(OUT, "w", encoding="utf-8", newline="\n").write(out)
print("wrote %s sha256 %s lines %d rollouts 12 edits %d (wrapper j hash %s)" % (os.path.basename(OUT), hashlib.sha256(out.encode("utf-8")).hexdigest()[:16], out.count("\n"), N, WRAPJ_SHA))
