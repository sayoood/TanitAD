"""make_pod_read_tabclose.py -- writes podread_tabclose10k.sh and chain_read_tabclose10k.sh from shufclose/podread_shufclose10k.sh (338f2c2fd183acca) and shufclose/chain_read_shufclose10k.sh (39e98b463d712faf)
by counted edits. Usage: python make_pod_read_tabclose.py   (run in tabclose/). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SH = os.path.normpath(os.path.join(HERE, "..", "shufclose"))
N = [0]


def load(fn, sha):
    raw = io.open(os.path.join(SH, fn), "rb").read()
    if hashlib.sha256(raw).hexdigest()[:16] != sha:
        sys.exit("REFUSED: %s is not the parent of card ADD83" % fn)
    return raw.decode("utf-8")


def edit(t, a, b, n=1):
    assert t.count(a) == n, (a, t.count(a), n)
    N[0] += 1
    return t.replace(a, b)


def put(fn, text):
    out = os.path.join(HERE, fn)
    if os.path.exists(out) and os.environ.get("TB_OVERWRITE_OWN") != "1":
        sys.exit("REFUSED: %s exists (set TB_OVERWRITE_OWN=1 to rewrite it)" % fn)
    assert "\r" not in text and "shuf" not in text.lower() and "surr" not in text.lower() and "ADD83" not in text
    io.open(out, "w", encoding="utf-8", newline="\n").write(text)
    print("wrote %s sha256 %s lines %d" % (fn, hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n")))


# ------------------------------------------------------------------------------------------------ podread
t = load("podread_shufclose10k.sh", "338f2c2fd183acca")
L = t.split("\n")
i_set = L.index("set -u")
assert L[0] == "#!/usr/bin/env bash" and all(x.startswith("#") for x in L[1:i_set])
head = ["#!/usr/bin/env bash",
        "# podread_tabclose10k.sh -- pod-side read of card PREREG_V8F_PROOF_ADD84 (TABCLOSE10K): the token-head table and the planted-quality tables fed to the rule (form RRe, --closing-table), at age and fresh.",
        "# New runs by part (TB_PART / --part): A = I10RQ50e{TK,STK}KF5P30Hb20 and I10RQ50eTKKF5Hb20; B = I10RQ50e{L100,W70,W50,W90,W00,S70}KF5P30Hb20; ALL = both. Banked: the runs of the estimates (card ADD79), RR, RRc, RLc, R0K at both settings, R0Hb20.",
        "# CPU only, no rollout. Never reads twice into one folder. It deletes, moves and stops nothing.",
        "# Usage: bash podread_tabclose10k.sh --pull PULL --code CODE --grid-dir DIR --fresh-grid FILE --midlook-json FILE --estclose-json FILE --tab-dir DIR --w W --tree TREE --out OUTDIR --wrapper-sha HEX16",
        "#          [--part A|B|ALL] [--read-dir DIR] [--py PYTHON]",
        "#   --read-dir: folder that holds read_tabclose10k.py, read_estclose10k_n2.py and the engine modules; default $PWD.  --w: queue folder; it must hold TABCLOSE10K_DONE[_part].txt and not TABCLOSE10K_STOP.txt.",
        "#   --tab-dir: folder of the tables and TAB_TABLES_RECORD.json (closeread/est_tab).  --estclose-json: READ_ESTCLOSE10K.json of card ADD79 (required, controls_all_pass true).",
        "# Output folder OUT: READ_TABCLOSE10K.json (+ .txt), read_tabclose10k.log, selftest_pod.log, READER_HASHES.txt.",
        "# Last line: ZZ PODREAD-TABCLOSE10K DONE or ZZ PODREAD-TABCLOSE10K STOP <why>."]
t = "\n".join(head + L[i_set:])
t = edit(t, 'PULL=""; CODE=""; GRIDDIR=""; FG=""; MIDJ=""; ESTJ=""; SHUFDIR=""; W=""; TREE=""; OUT=""; WSHA=""; RD="$PWD"; PY=/workspace/venv/bin/python',
         'PULL=""; CODE=""; GRIDDIR=""; FG=""; MIDJ=""; ESTJ=""; TABDIR=""; W=""; TREE=""; OUT=""; WSHA=""; RD="$PWD"; PY=/workspace/venv/bin/python; PART=ALL')
t = edit(t, "    --shuf-dir) SHUFDIR=$2; shift 2;;\n", "    --tab-dir) TABDIR=$2; shift 2;;\n    --part) PART=$2; shift 2;;\n")
t = edit(t, 'for v in PULL CODE GRIDDIR FG MIDJ ESTJ SHUFDIR W TREE OUT WSHA; do eval "val=\\${$v}"; [ -n "$val" ] || stop "MISSING-ARGUMENT $v"; done\n',
         'for v in PULL CODE GRIDDIR FG MIDJ ESTJ TABDIR W TREE OUT WSHA; do eval "val=\\${$v}"; [ -n "$val" ] || stop "MISSING-ARGUMENT $v"; done\n'
         'case "$PART" in A|B|ALL) ;; *) stop "PART must be A, B or ALL";; esac\n'
         'PS=""; [ "$PART" = ALL ] || PS="_$PART"\n')
t = edit(t, '[ -f "$W/SHUFCLOSE10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"', '[ -f "$W/TABCLOSE10K_DONE$PS.txt" ] || stop "QUEUE-NOT-DONE"')
t = edit(t, '[ -e "$W/SHUFCLOSE10K_STOP.txt" ] && stop "QUEUE-STOPPED"', '[ -e "$W/TABCLOSE10K_STOP.txt" ] && stop "QUEUE-STOPPED"')
t = edit(t, 'read_shufclose10k.py read_estclose10k_n2.py', 'read_tabclose10k.py read_estclose10k_n2.py')
t = edit(t, '"$SHUFDIR/SURROGATE_RECORD.json"; do [ -f "$f" ]', '"$TABDIR/TAB_TABLES_RECORD.json"; do [ -f "$f" ]')
t = edit(t, 'grep -q \'^ "pass": true\' "$SHUFDIR/SURROGATE_RECORD.json" || stop "SURROGATE-RECORD-NOT-PASS"\n', '')
t = edit(t, 'for e in N10 L25; do for s in 0 1; do [ -f "$SHUFDIR/surr_${e}_s$s.npz" ] || stop "MISSING-INPUT $SHUFDIR/surr_${e}_s$s.npz"; done; done\n',
         'ARMS_N=""; [ "$PART" != B ] && ARMS_N="TK STK"; [ "$PART" != A ] && ARMS_N="$ARMS_N L100 W70 W50 W90 W00 S70"\n'
         'for x in $ARMS_N; do for s in 0 1; do [ -f "$TABDIR/tab_${x}_s$s.npz" ] || stop "MISSING-INPUT $TABDIR/tab_${x}_s$s.npz"; done; done\n'
         'NEWRUNS=""; [ "$PART" != B ] && NEWRUNS="RQ50eTKKF5P30Hb20 RQ50eSTKKF5P30Hb20 RQ50eTKKF5Hb20"\n'
         '[ "$PART" != A ] && NEWRUNS="$NEWRUNS RQ50eL100KF5P30Hb20 RQ50eW70KF5P30Hb20 RQ50eW50KF5P30Hb20 RQ50eW90KF5P30Hb20 RQ50eW00KF5P30Hb20 RQ50eS70KF5P30Hb20"\n')
t = edit(t, "for r in R0Hb20 R0KHb20 R0KP30Hb20 RQ50KF5P30Hb20 RLcKF5P30Hb20 RQ50hKF5P30Hb20 RQ50cKF5P30Hb20 RQ50eN10KF5P30Hb20 RQ50eL25KF5P30Hb20 RQ50eSN10KF5P30Hb20 RQ50eSL25KF5P30Hb20 RQ50KF5Hb20 RLcKF5Hb20 RQ50hKF5Hb20 RQ50cKF5Hb20 RQ50eN10KF5Hb20 RQ50eL25KF5Hb20 RQ50eSN10KF5Hb20 RQ50eSL25KF5Hb20; do",
         "for r in R0Hb20 R0KHb20 R0KP30Hb20 RQ50KF5P30Hb20 RLcKF5P30Hb20 RQ50cKF5P30Hb20 RQ50eN10KF5P30Hb20 RQ50eL25KF5P30Hb20 RQ50KF5Hb20 RLcKF5Hb20 RQ50cKF5Hb20 RQ50eN10KF5Hb20 RQ50eL25KF5Hb20 $NEWRUNS; do")
t = edit(t, '"$SHUFDIR"/SURROGATE_RECORD.json "$SHUFDIR"/surr_*.npz', '"$TABDIR"/TAB_TABLES_RECORD.json "$TABDIR"/tab_*.npz')
t = edit(t, '--shuf-dir "$SHUFDIR"', '--tab-dir "$TABDIR" --part "$PART"')
for a, b in (("read_shufclose10k.py", "read_tabclose10k.py"), ("READ_SHUFCLOSE10K.json", "READ_TABCLOSE10K.json"), ("read_shufclose10k.log", "read_tabclose10k.log"), ("READ-SHUFCLOSE", "READ-TABCLOSE")):
    n = t.count(a)
    assert n >= 1, a
    t = edit(t, a, b, n)
put("podread_tabclose10k.sh", t)

# ------------------------------------------------------------------------------------------------ chain
c = load("chain_read_shufclose10k.sh", "39e98b463d712faf")
L = c.split("\n")
i_set = L.index("set -u")
assert all(x.startswith("#") for x in L[1:i_set])
head = ["#!/usr/bin/env bash",
        "# chain_read_tabclose10k.sh -- pod: waits for the queue of card ADD84 (TABCLOSE10K_DONE.txt, or TABCLOSE10K_DONE_<part>.txt with TB_PART=A|B; TABCLOSE10K_STOP.txt) AND for the go file GO_READ_TABCLOSE10K.txt",
        "# (written by the lead once the hold that carries the card is landed), links the new run folders into the pull folder of the readers and runs podread_tabclose10k.sh once",
        "# (output folder read_tabclose10k, or read_tabclose10k_<part> with a part switch).",
        "# CPU only. No process listing or stopping, no file removal or move. Refuses to run twice (START_READ_TABCLOSE10K[_part].txt). The wrapper hash of the queue is checked here as well.",
        "# Last line of the log: ZZ CHAIN-READ-TABCLOSE10K END <what>."]
c = "\n".join(head + L[i_set:])
c = edit(c, "set -u\n", 'set -u\nPART=${TB_PART:-ALL}\ncase "$PART" in A|B|ALL) ;; *) echo "REFUSED: TB_PART must be A, B or ALL"; exit 2;; esac\nPS=""; [ "$PART" = ALL ] || PS="_$PART"\n')
c = edit(c, "OUT=$P/out/read_shufclose10k\n", "OUT=$P/out/read_tabclose10k$PS\n")
c = edit(c, "SD=/workspace/v8proof/closeread/est_shuf\n", "SD=/workspace/v8proof/closeread/est_tab\n")
c = edit(c, '"$W/START_READ_SHUFCLOSE10K.txt" ) 2>/dev/null || { echo "REFUSED: START_READ_SHUFCLOSE10K.txt exists"; exit 2; }',
         '"$W/START_READ_TABCLOSE10K$PS.txt" ) 2>/dev/null || { echo "REFUSED: START_READ_TABCLOSE10K$PS.txt exists"; exit 2; }')
c = edit(c, '[ -e "$W/SHUFCLOSE10K_STOP.txt" ] && { echo "ZZ CHAIN-READ-SHUFCLOSE10K END QUEUE-STOPPED $(cut -c1-200 "$W/SHUFCLOSE10K_STOP.txt")"; exit 3; }',
         '[ -e "$W/TABCLOSE10K_STOP.txt" ] && { echo "ZZ CHAIN-READ-TABCLOSE10K END QUEUE-STOPPED $(cut -c1-200 "$W/TABCLOSE10K_STOP.txt")"; exit 3; }')
c = edit(c, '[ -f "$W/SHUFCLOSE10K_DONE.txt" ] && [ -f "$W/GO_READ_SHUFCLOSE10K.txt" ] && break', '[ -f "$W/TABCLOSE10K_DONE$PS.txt" ] && [ -f "$W/GO_READ_TABCLOSE10K.txt" ] && break')
c = edit(c, 'queue done: $(cut -c1-120 "$W/SHUFCLOSE10K_DONE.txt"); go: $(cut -c1-160 "$W/GO_READ_SHUFCLOSE10K.txt")', 'queue done: $(cut -c1-120 "$W/TABCLOSE10K_DONE$PS.txt"); go: $(cut -c1-160 "$W/GO_READ_TABCLOSE10K.txt")')
c = edit(c, 'for d in "$W"/t1/I10RQ50eS*_s[01]; do', 'for d in "$W"/t1/I10RQ50e{TK,STK,L100,W00,W50,W70,W90,S70}*_s[01]; do')
c = edit(c, "grep -c '^I10RQ50eS'", "grep -c '^I10RQ50e'")
c = edit(c, 'bash "$P/podread_shufclose10k.sh"', 'bash "$P/podread_tabclose10k.sh"')
c = edit(c, '--shuf-dir "$SD"', '--tab-dir "$SD" --part "$PART"')
n = c.count("ZZ CHAIN-READ-SHUFCLOSE10K")
assert n >= 1
c = edit(c, "ZZ CHAIN-READ-SHUFCLOSE10K", "ZZ CHAIN-READ-TABCLOSE10K", n)
put("chain_read_tabclose10k.sh", c)
print("edits", N[0])
