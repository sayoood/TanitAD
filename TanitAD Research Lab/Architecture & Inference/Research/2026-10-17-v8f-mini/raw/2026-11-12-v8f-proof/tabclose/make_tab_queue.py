"""make_tab_queue.py -- writes pod_tabclose10k_queue.sh from the queue of card ADD83 (shufclose/pod_shufclose10k_queue.sh, sha256 prefix 5cac167387fe4654) by counted text edits.

Kept line for line: the slot rule, the start checks and the rollout function (closing table path, form RRe).
New: header, markers (START_TABCLOSE10K[_part].txt, TABCLOSE10K_DONE[_part].txt, TABCLOSE10K_STOP.txt), log queue_tabclose.log, small folder small_tb, tables from est_tab,
the part switch TB_PART=A|B|ALL, the admission (pod check of card ADD83 + TAB_TABLES_RECORD.json + table hashes) and the 18 rollouts.
Usage: python make_tab_queue.py   (run in tabclose/). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "shufclose", "pod_shufclose10k_queue.sh"))
OUT = os.path.join(HERE, "pod_tabclose10k_queue.sh")
WRAP_FILE = os.path.normpath(os.path.join(HERE, "..", "shufclose", "v8f_t1_eval_reader_rule_rzgfhqet.py"))
SRC_SHA = "5cac167387fe4654"
WRAP_SHA = "02a58e655e5c1979"
raw = io.open(SRC, "rb").read()
if hashlib.sha256(raw).hexdigest()[:16] != SRC_SHA:
    sys.exit("REFUSED: the source queue is not the one of card ADD83")
if hashlib.sha256(io.open(WRAP_FILE, "rb").read()).hexdigest()[:16] != WRAP_SHA:
    sys.exit("REFUSED: the wrapper is not the one of card ADD83")
if os.path.exists(OUT) and os.environ.get("TB_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (set TB_OVERWRITE_OWN=1 to rewrite it)" % OUT)
L = raw.decode("utf-8").split("\n")
i_set = L.index("set -u")
i_roll_end = max(i for i, x in enumerate(L) if x == "}")
i_tail = next(i for i, x in enumerate(L) if x.startswith('[ "$DRY" = 1 ] || [ -f "$G30" ]'))
assert L[0] == "#!/usr/bin/env bash" and i_set == 9 and i_tail == i_roll_end + 2, ("anchors", i_set, i_tail, i_roll_end)
text = "\n".join(L[i_set:i_roll_end + 1])
N_EDITS = 0


def once(a, b, n=1):
    global text, N_EDITS
    assert text.count(a) == n, (a, text.count(a), n)
    text = text.replace(a, b)
    N_EDITS += 1


once("TABLES=${I10_TABLES:-$V/closeread/est_shuf}", "TABLES=${I10_TABLES:-$V/closeread/est_tab}")
once("LOG=$W/queue_shufclose.log", "LOG=$W/queue_tabclose.log")
once("SM=$W/small_sc", "SM=$W/small_tb")
once("SHUFCLOSE10K_STOP.txt", "TABCLOSE10K_STOP.txt")
once("START_SHUFCLOSE10K.txt", "START_TABCLOSE10K$PS.txt", 2)
once("ARMS=(0 1)\n",
     'ARMS=(0 1)\n'
     'PART=${TB_PART:-ALL}\n'
     'case "$PART" in A|B|ALL) ;; *) echo "REFUSED: TB_PART must be A, B or ALL (got $PART)"; exit 2;; esac\n'
     'PS=""; [ "$PART" = ALL ] || PS="_$PART"\n'
     'want() { [ "$PART" = ALL ] || [ "$PART" = "$1" ]; }\n')
once('log "queue start (DRY=$DRY): card ADD83 (SHUFCLOSE10K): 8 rollouts of form RRe whose closing input is the estimate of ANOTHER clip (surrogate tables), estimators L25 and N10, at age and fresh"',
     'log "queue start (DRY=$DRY, part $PART): card ADD84 (TABCLOSE10K): rollouts of form RRe whose closing input is a table (token head TK, its surrogate STK, planted-quality arms), at age and fresh"')
for bad in ("SHUFCLOSE10K", "shufclose", "surr_", "small_sc", "ADD83"):
    assert bad not in text, bad

ADMIT_PY = r'''import hashlib, json, os, sys
rec_path, tdir, part, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
rec = json.load(open(rec_path, encoding="utf-8"))
if part == "ALL" and rec.get("pass") is not True:
    print("RECORD-PASS-NOT-TRUE"); sys.exit(1)
need = []
if part in ("A", "ALL"):
    need += ["a4_u1", "a4_u4"]
if part in ("B", "ALL"):
    need += ["a0", "a1"]
for c in need:
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
print("ADMIT-OK part %s tables %d" % (part, 2 * len(arms)))
'''

head = ["#!/usr/bin/env bash",
        "# pod_tabclose10k_queue.sh -- card ADD84 (TABCLOSE10K): 18 rollouts of form RRe whose closing-speed input is a TABLE read through --closing-table:",
        "# the token-head table TK, its surrogate STK (part A) and the planted-quality arms L100, W00, W50, W70, W90, S70 (part B); tables /workspace/v8proof/closeread/est_tab/tab_<ARM>_s<k>.npz;",
        "# --rule-every 5 under readout K (--acc-gain 0 --path-exec --kappa-max 0.2), q50 map, model seeds 0 and 1 (reader seed k), at age (P = 30 grid, pre-roll 30 steps = 6 s) and, for TK, fresh (pinned grid);",
        "# wrapper v8f_t1_eval_reader_rule_rzgfhqet.py (hash %s), admitted only with the pod check of card ADD83 (CHECK_SHUF_DONE_<tag>.txt says PASS and its record carries the wrapper hash)" % WRAP_SHA,
        "# and with TAB_TABLES_RECORD.json saying pass and every table file of the part equal to its record entry.",
        "# TB_PART=A (TK AGE, STK AGE, TK FRESH: 6 rollouts) | B (the other 12) | ALL (default); markers carry the suffix _A / _B with a part switch. DRY=1 prints every command and runs nothing.",
        "# Written by make_tab_queue.py from the queue of card ADD83 (shufclose/pod_shufclose10k_queue.sh).",
        "# No training. No process listing or stopping, no file removal or move: the queue only writes new files.",
        "# Markers START_TABCLOSE10K.txt / TABCLOSE10K_DONE.txt / TABCLOSE10K_STOP.txt in $W; step markers ro_<arm> in $W/markers."]
K = "READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5"
# (arm, setting, part)
PLAN = [("TK", "AGE", "A"), ("STK", "AGE", "A"), ("L100", "AGE", "B"), ("W70", "AGE", "B"), ("W50", "AGE", "B"), ("W90", "AGE", "B"),
        ("W00", "AGE", "B"), ("S70", "AGE", "B"), ("TK", "FRESH", "A")]
assert len(PLAN) * 2 == 18
A_ARMS = ["TK", "STK"]
B_ARMS = ["L100", "W00", "W50", "W70", "W90", "S70"]
BANKED = ("I10RQ50KF5P30Hb20 I10RLcKF5P30Hb20 I10RQ50KF5Hb20 I10RLcKF5Hb20 I10R0KP30Hb20 I10R0KHb20 I10R0Hb20 "
          "I10RQ50eN10KF5P30Hb20 I10RQ50eL25KF5P30Hb20 I10RQ50eN10KF5Hb20 I10RQ50eL25KF5Hb20")
tail = ["",
        "admit() { # RECORD TABLEDIR PART ARMS -> prints why on failure",
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
        '  [ -f "$TABLES/TAB_TABLES_RECORD.json" ] || stop "NO-TAB-TABLES-RECORD"',
        '  ARMLIST=""; want A && ARMLIST="%s"; want B && ARMLIST="${ARMLIST:+$ARMLIST,}%s"' % (",".join(A_ARMS), ",".join(B_ARMS)),
        '  adm=$(admit "$TABLES/TAB_TABLES_RECORD.json" "$TABLES" "$PART" "$ARMLIST") || stop "TABLE-ADMISSION $adm"',
        '  log "$adm"',
        "  for nm in %s; do" % BANKED,
        '    for k in "${ARMS[@]}"; do [ -d "$W/t1/${nm}_s$k/dump_${nm}_s${k}_m10_causal_heads" ] || stop "NO-BANKED-RUN ${nm}_s$k"; done',
        "  done",
        "fi"]
names = {"A": [], "B": []}
for arm, setting, part in PLAN:
    pre = "RGRID=$G30 PREROLL=30 " if setting == "AGE" else ""
    nm = "I10RQ50e%sKF5%s" % (arm, "P30Hb20" if setting == "AGE" else "Hb20")
    tail += ["if want %s; then" % part,
             '  for k in "${ARMS[@]}"; do',
             '    Q="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"',
             "    " + pre + K + ' QMAP_FILE="$Q" TAB="$TABLES/tab_%s_s${k}.npz" rollout %s RRe "$k" "$k"' % (arm, nm),
             "  done",
             "fi"]
    names[part].append(nm)
assert len(names["A"]) == 3 and len(names["B"]) == 6
tail += ['if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi',
         'cp -f "$LOG" "$SM/" 2>/dev/null',
         'NAMES=""; want A && NAMES="%s"; want B && NAMES="$NAMES %s"' % (" ".join(names["A"]), " ".join(names["B"])),
         'for nm in $NAMES; do for k in "${ARMS[@]}"; do cp -f "$W/t1/${nm}_s$k/ticks/run_${nm}_s$k.txt" "$W/t1/${nm}_s$k/ticks/preroll_log_${nm}_s$k.json" "$SM/" 2>/dev/null; done; done',
         'echo "$(ts) all steps done (part $PART)" > "$W/TABCLOSE10K_DONE$PS.txt"',
         'log "TABCLOSE10K_DONE part $PART"',
         ""]
text = "\n".join(head) + "\n" + text + "\n" + "\n".join(tail)
assert "\r" not in text
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s lines %d rollouts 18 edits %d (wrapper hash %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n"), N_EDITS, WRAP_SHA))
