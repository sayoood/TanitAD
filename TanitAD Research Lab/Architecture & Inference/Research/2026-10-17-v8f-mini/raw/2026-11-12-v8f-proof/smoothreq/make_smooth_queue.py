"""make_smooth_queue.py -- writes pod_smoothreq10k_queue.sh from the queue of card ADD84 (tabclose/pod_tabclose10k_queue.sh, sha256 prefix recorded below) by counted text edits.

Kept line for line: the slot rule, the start checks and the rollout function (closing table path, form RRe).
New: header, markers START_SMOOTHREQ10K[_s<k>].txt / SMOOTHREQ10K_DONE[_s<k>].txt / SMOOTHREQ10K_STOP[_s<k>].txt, log queue_smoothreq[_s<k>].log, small folder small_sr, tables from est_smooth, the seed
switch SR_SEED=0|1|ALL, the admission (pod check of card ADD83 + SMOOTH_TABLES_RECORD.json + table hashes) and the 14 rollouts (age only).
Usage: python make_smooth_queue.py   (run in smoothreq/). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "tabclose", "pod_tabclose10k_queue.sh"))
OUT = os.path.join(HERE, "pod_smoothreq10k_queue.sh")
WRAP_FILE = os.path.normpath(os.path.join(HERE, "..", "shufclose", "v8f_t1_eval_reader_rule_rzgfhqet.py"))
SRC_SHA = os.environ.get("SR_SRC_SHA", "cd0a0c2903436bae")
WRAP_SHA = "02a58e655e5c1979"
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != SRC_SHA:
    sys.exit("REFUSED: the source queue is not the one of card ADD84 (%s)" % hashlib.sha256(raw).hexdigest()[:16])
if hashlib.sha256(io.open(WRAP_FILE, "rb").read()).hexdigest()[:16] != WRAP_SHA:
    sys.exit("REFUSED: the wrapper is not the one of card ADD83")
if os.path.exists(OUT) and os.environ.get("SR_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (set SR_OVERWRITE_OWN=1 to rewrite it)" % OUT)
L = raw.decode("utf-8").split("\n")
i_set = L.index("set -u")
i_roll = next(i for i, x in enumerate(L) if x.startswith("rollout() {"))
i_roll_end = next(i for i in range(i_roll, len(L)) if L[i] == "}")
i_admit = next(i for i, x in enumerate(L) if x.startswith("admit() {"))
assert L[0] == "#!/usr/bin/env bash" and i_set == 10 and i_admit > i_roll_end, (i_set, i_admit, i_roll_end)
text = "\n".join(L[i_set:i_roll_end + 1])
N_EDITS = 0


def once(a, b, n=1):
    global text, N_EDITS
    assert text.count(a) == n, (a, text.count(a), n)
    text = text.replace(a, b)
    N_EDITS += 1


once("TABLES=${I10_TABLES:-$V/closeread/est_tab}", "TABLES=${I10_TABLES:-$V/closeread/est_smooth}")
once("LOG=$W/queue_tabclose.log",
     'SEEDSW=${SR_SEED:-ALL}\n'
     'case "$SEEDSW" in 0|1|ALL) ;; *) echo "REFUSED: SR_SEED must be 0, 1 or ALL (got $SEEDSW)"; exit 2;; esac\n'
     'ARMS=(0 1); PS=""\n'
     '[ "$SEEDSW" = ALL ] || { ARMS=("$SEEDSW"); PS="_s$SEEDSW"; }\n'
     'LOG=$W/queue_smoothreq$PS.log')
once("SM=$W/small_tb", "SM=$W/small_sr")
once("TABCLOSE10K_STOP.txt", "SMOOTHREQ10K_STOP$PS.txt", 1)
once("START_TABCLOSE10K$PS.txt", "START_SMOOTHREQ10K$PS.txt", 2)
once('ARMS=(0 1)\nPART=${TB_PART:-ALL}\ncase "$PART" in A|B|ALL) ;; *) echo "REFUSED: TB_PART must be A, B or ALL (got $PART)"; exit 2;; esac\nPS=""; [ "$PART" = ALL ] || PS="_$PART"\n'
     'want() { [ "$PART" = ALL ] || [ "$PART" = "$1" ]; }\n', "")
once('log "queue start (DRY=$DRY, part $PART): card ADD84 (TABCLOSE10K): rollouts of form RRe whose closing input is a table (token head TK, its surrogate STK, planted-quality arms), at age and fresh"',
     'log "queue start (DRY=$DRY, seeds $SEEDSW): card ADD85 (SMOOTHREQ10K): rollouts of form RRe whose closing input is a table (smooth and white planted errors, Z00, STKM, TKD), at age"')
for bad in ("TABCLOSE10K", "tabclose", "small_tb", "ADD84", "PART"):
    assert bad not in text.replace("ADD85", ""), bad

ADMIT_PY = r'''import hashlib, json, os, sys
rec_path, tdir, arms = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
rec = json.load(open(rec_path, encoding="utf-8"))
if rec.get("pass") is not True:
    print("RECORD-PASS-NOT-TRUE"); sys.exit(1)
for c in ("b0", "b1", "b2"):
    d = rec.get(c)
    if not isinstance(d, dict) or d.get("PASS") is not True:
        print("CONTROL-NOT-PASS " + c); sys.exit(1)
tabs = rec.get("tables")
if not isinstance(tabs, dict):
    print("RECORD-HAS-NO-TABLES"); sys.exit(1)
for a in arms:
    for k in (0, 1):
        fn = "tab_%s_s%d.npz" % (a, k)
        p = os.path.join(tdir, fn)
        if not os.path.isfile(p):
            print("NO-TABLE " + fn); sys.exit(1)
        if fn not in tabs:
            print("TABLE-NOT-IN-RECORD " + fn); sys.exit(1)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if h != tabs[fn]:
            print("TABLE-HASH-DIFFERS " + fn); sys.exit(1)
print("ADMIT-OK tables %d" % (2 * len(arms)))
'''

head = ["#!/usr/bin/env bash",
        "# pod_smoothreq10k_queue.sh -- card ADD85 (SMOOTHREQ10K): 14 rollouts of form RRe whose closing-speed input is a TABLE read through --closing-table, at age only (pre-roll 30 steps = 6 s):",
        "# STKM, TKD, Z00, S90, S50, S7B, W7B; tables /workspace/v8proof/closeread/est_smooth/tab_<ARM>_s<k>.npz; run names I10RQ50e<ARM>KF5P30Hb20_s<k>;",
        "# --rule-every 5 under readout K (--acc-gain 0 --path-exec --kappa-max 0.2), q50 map, model seeds 0 and 1 (reader seed k);",
        "# wrapper v8f_t1_eval_reader_rule_rzgfhqet.py (hash %s), admitted only with the pod check of card ADD83 (CHECK_SHUF_DONE_<tag>.txt says PASS and its record carries the wrapper hash)" % WRAP_SHA,
        "# and with SMOOTH_TABLES_RECORD.json saying pass (controls b0, b1, b2 PASS) and every table file equal to its record entry; the banked runs the reader needs (including the 18 runs of card ADD84) have to exist.",
        "# SR_SEED=0|1|ALL (default ALL): with 0 or 1 only the seven rollouts of that model seed run, with the markers START_SMOOTHREQ10K_s<k>.txt, SMOOTHREQ10K_DONE_s<k>.txt, SMOOTHREQ10K_STOP_s<k>.txt and the log",
        "# queue_smoothreq_s<k>.log, so that the two seeds can run side by side as two processes. DRY=1 prints every command and runs nothing.",
        "# Written by make_smooth_queue.py from the queue of card ADD84 (tabclose/pod_tabclose10k_queue.sh).",
        "# No training. No process listing or stopping, no file removal or move: the queue only writes new files.",
        "# Markers START_SMOOTHREQ10K.txt / SMOOTHREQ10K_DONE.txt / SMOOTHREQ10K_STOP.txt in $W; step markers ro_<arm> in $W/markers."]
K = "READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5"
ORDER = ["STKM", "TKD", "Z00", "S90", "S50", "S7B", "W7B"]
BANKED_OLD = ("I10RQ50KF5P30Hb20 I10RLcKF5P30Hb20 I10RQ50KF5Hb20 I10RLcKF5Hb20 I10R0KP30Hb20 I10R0KHb20 I10R0Hb20 "
              "I10RQ50eN10KF5P30Hb20 I10RQ50eL25KF5P30Hb20 I10RQ50eN10KF5Hb20 I10RQ50eL25KF5Hb20")
BANKED_84 = " ".join("I10RQ50e%sKF5P30Hb20" % a for a in ("TK", "STK", "L100", "W00", "W50", "W70", "W90", "S70")) + " I10RQ50eTKKF5Hb20"
assert len(BANKED_84.split()) == 9
tail = ["",
        "admit() { # RECORD TABLEDIR ARMS -> prints why on failure",
        '  "$PY" - "$@" <<\'PYEOF\'',
        ADMIT_PY.rstrip("\n"),
        "PYEOF",
        "}",
        '[ "$DRY" = 1 ] || [ -f "$G30" ] || stop "NO-GRID-P30"',
        'if [ "$DRY" != 1 ]; then',
        '  TAG=${SC_TAG:-sk1}',
        '  CK="$W/shufclose_test/CHECK_SHUF_DONE_$TAG.txt"',
        '  grep -q "^ZZ CHECK-SHUFCLOSE PASS" "$CK" 2>/dev/null || stop "NO-PASS-OF-THE-WRAPPER-CHECK"',
        '  got=$(sha256sum "$WRAP" | cut -c1-16); [ "$got" = "%s" ] || stop "WRAPPER-SHA $got"' % WRAP_SHA,
        '  grep -q "%s" "$W/shufclose_test/CHECK_SHUFCLOSE_$TAG.txt.hashes" 2>/dev/null || stop "CHECK-RECORD-HAS-NOT-THIS-WRAPPER"' % WRAP_SHA,
        '  [ -f "$TABLES/SMOOTH_TABLES_RECORD.json" ] || stop "NO-SMOOTH-TABLES-RECORD"',
        '  adm=$(admit "$TABLES/SMOOTH_TABLES_RECORD.json" "$TABLES" "%s") || stop "TABLE-ADMISSION $adm"' % ",".join(ORDER),
        '  log "$adm"',
        "  for nm in %s %s; do" % (BANKED_OLD, BANKED_84),
        '    for k in "${ARMS[@]}"; do [ -d "$W/t1/${nm}_s$k/dump_${nm}_s${k}_m10_causal_heads" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done',
        "  done",
        '  for k in "${ARMS[@]}"; do [ -d "$W/t1/I10RQ50cKF5P30Hb20_s$k" ] || stop "NO-BANKED-RUN I10RQ50cKF5P30Hb20_s$k"; done',
        "fi"]
names = []
for arm in ORDER:
    nm = "I10RQ50e%sKF5P30Hb20" % arm
    tail += ['for k in "${ARMS[@]}"; do',
             '  Q="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"',
             "  RGRID=$G30 PREROLL=30 " + K + ' QMAP_FILE="$Q" TAB="$TABLES/tab_%s_s${k}.npz" rollout %s RRe "$k" "$k"' % (arm, nm),
             "done"]
    names.append(nm)
assert len(names) * 2 == 14
tail += ['if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi',
         'cp -f "$LOG" "$SM/" 2>/dev/null',
         'NAMES="%s"' % " ".join(names),
         'for nm in $NAMES; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done',
         'echo "$(ts) all steps done (seeds $SEEDSW)" > "$W/SMOOTHREQ10K_DONE$PS.txt"',
         'log "SMOOTHREQ10K_DONE seeds $SEEDSW"',
         ""]
text = "\n".join(head) + "\n" + text + "\n" + "\n".join(tail)
assert "\r" not in text
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d rollouts 14 edits %d (wrapper hash %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n"), N_EDITS, WRAP_SHA))
