"""make_pod_read_smooth.py -- writes podread_smoothreq10k.sh and chain_read_smoothreq10k.sh from tabclose/podread_tabclose10k.sh and tabclose/chain_read_tabclose10k.sh (sha256 prefixes below) by counted edits.
Usage: python make_pod_read_smooth.py   (run in smoothreq/). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TB = os.path.normpath(os.path.join(HERE, "..", "tabclose"))
N = [0]
SHA_POD = os.environ.get("SR_POD_SHA", "85a5e181627245e4")
SHA_CHAIN = os.environ.get("SR_CHAIN_SHA", "4cf6572c12e5dee4")


def load(fn, sha):
    raw = io.open(os.path.join(TB, fn), "rb").read()
    if hashlib.sha256(raw).hexdigest()[:16] != sha:
        sys.exit("REFUSED: %s is not the parent of card ADD84 (%s)" % (fn, hashlib.sha256(raw).hexdigest()[:16]))
    return raw.decode("utf-8")


def edit(t, a, b, n=1):
    assert t.count(a) == n, (a, t.count(a), n)
    N[0] += 1
    return t.replace(a, b)


def rename(t):
    for a, b in (("TABCLOSE10K", "SMOOTHREQ10K"), ("tabclose10k", "smoothreq10k"), ("READ-TABCLOSE", "READ-SMOOTHREQ")):
        t = t.replace(a, b)
    return t


def put(fn, text, allow=""):
    out = os.path.join(HERE, fn)
    if os.path.exists(out) and os.environ.get("SR_OVERWRITE_OWN") != "1":
        sys.exit("REFUSED: %s exists (set SR_OVERWRITE_OWN=1 to rewrite it)" % fn)
    chk = text.replace(allow, "") if allow else text
    assert "\r" not in text and "TABCLOSE" not in chk and "tabclose10k" not in chk, "a token of the parent is left"
    io.open(out, "w", encoding="utf-8", newline="\n").write(text)
    print("wrote %s sha256 %s lines %d" % (fn, hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], text.count("\n")))


# ------------------------------------------------------------------------------------------------ podread
t = load("podread_tabclose10k.sh", SHA_POD)
L = t.split("\n")
i_set = L.index("set -u")
head = ["#!/usr/bin/env bash",
        "# podread_smoothreq10k.sh -- pod-side read of card PREREG_V8F_PROOF_ADD85 (SMOOTHREQ10K): tables with a smooth or white planted error, a smooth spread alone, the matched surrogate and the head without its mean bias",
        "# fed to the rule (form RRe, --closing-table), at age. New runs: I10RQ50e{STKM,TKD,Z00,S90,S50,S7B,W7B}KF5P30Hb20. Banked: the 18 runs of card ADD84, RR, RRc, RLc, R0K, the runs of card ADD79, R0Hb20.",
        "# CPU only, no rollout. Never reads twice into one folder. It deletes, moves and stops nothing.",
        "# Usage: bash podread_smoothreq10k.sh --pull PULL --code CODE --grid-dir DIR --fresh-grid FILE --midlook-json FILE --estclose-json FILE --tabclose-json FILE --tab-dir DIR --w W --tree TREE --out OUTDIR --wrapper-sha HEX16",
        "#          [--read-dir DIR] [--py PYTHON]",
        "#   --w: queue folder; it must hold SMOOTHREQ10K_DONE.txt (or both SMOOTHREQ10K_DONE_s0.txt and _s1.txt) and no SMOOTHREQ10K_STOP*.txt.  --tab-dir: folder of the tables and SMOOTH_TABLES_RECORD.json (closeread/est_smooth).",
        "#   --tabclose-json: the result file of the read of card ADD84 (required).  --estclose-json: READ_ESTCLOSE10K.json of card ADD79 (required, controls_all_pass true).",
        "# Output folder OUT: READ_SMOOTHREQ10K.json (+ .txt), read_smoothreq10k.log, selftest_pod.log, READER_HASHES.txt.",
        "# Last line: ZZ PODREAD-SMOOTHREQ10K DONE or ZZ PODREAD-SMOOTHREQ10K STOP <why>."]
t = "\n".join(head + L[i_set:])
t = edit(t, 'ESTJ=""; TABDIR=""; W=""', 'ESTJ=""; TBJ=""; TABDIR=""; W=""')
t = edit(t, 'PY=/workspace/venv/bin/python; PART=ALL', 'PY=/workspace/venv/bin/python')
t = edit(t, "    --part) PART=$2; shift 2;;\n", "    --tabclose-json) TBJ=$2; shift 2;;\n")
t = edit(t, "for v in PULL CODE GRIDDIR FG MIDJ ESTJ TABDIR W TREE OUT WSHA;", "for v in PULL CODE GRIDDIR FG MIDJ ESTJ TBJ TABDIR W TREE OUT WSHA;")
t = edit(t, 'case "$PART" in A|B|ALL) ;; *) stop "PART must be A, B or ALL";; esac\nPS=""; [ "$PART" = ALL ] || PS="_$PART"\n', "")
t = edit(t, '[ -f "$W/TABCLOSE10K_DONE$PS.txt" ] || stop "QUEUE-NOT-DONE"', '{ [ -f "$W/SMOOTHREQ10K_DONE.txt" ] || { [ -f "$W/SMOOTHREQ10K_DONE_s0.txt" ] && [ -f "$W/SMOOTHREQ10K_DONE_s1.txt" ]; }; } || stop "QUEUE-NOT-DONE"')
t = edit(t, '[ -e "$W/TABCLOSE10K_STOP.txt" ] && stop "QUEUE-STOPPED"', 'for sf in "$W"/SMOOTHREQ10K_STOP*.txt; do [ -e "$sf" ] && stop "QUEUE-STOPPED"; done')
t = edit(t, 'MODS="read_tabclose10k.py ', 'MODS="read_smoothreq10k.py ')
t = edit(t, '"$TABDIR/TAB_TABLES_RECORD.json"; do [ -f "$f" ]', '"$TABDIR/SMOOTH_TABLES_RECORD.json" "$TBJ"; do [ -f "$f" ]')
t = edit(t, 'grep -q \'PREREG_V8F_PROOF_ADD79\' "$ESTJ" || stop "ESTCLOSE-JSON-IS-NOT-THE-READ-OF-ADD79"\n',
         'grep -q \'PREREG_V8F_PROOF_ADD79\' "$ESTJ" || stop "ESTCLOSE-JSON-IS-NOT-THE-READ-OF-ADD79"\ngrep -q \'PREREG_V8F_PROOF_ADD84\' "$TBJ" || stop "ADD84-JSON-IS-NOT-THE-READ-OF-ADD84"\n')
t = edit(t, 'ARMS_N=""; [ "$PART" != B ] && ARMS_N="TK STK"; [ "$PART" != A ] && ARMS_N="$ARMS_N L100 W70 W50 W90 W00 S70"\n', 'ARMS_N="STKM TKD Z00 S90 S50 S7B W7B"\n')
t = edit(t, 'NEWRUNS=""; [ "$PART" != B ] && NEWRUNS="RQ50eTKKF5P30Hb20 RQ50eSTKKF5P30Hb20 RQ50eTKKF5Hb20"\n[ "$PART" != A ] && NEWRUNS="$NEWRUNS RQ50eL100KF5P30Hb20 RQ50eW70KF5P30Hb20 RQ50eW50KF5P30Hb20 RQ50eW90KF5P30Hb20 RQ50eW00KF5P30Hb20 RQ50eS70KF5P30Hb20"\n',
         'NEWRUNS=""; for x in $ARMS_N; do NEWRUNS="$NEWRUNS RQ50e${x}KF5P30Hb20"; done\nBANK84=""; for x in TK STK L100 W00 W50 W70 W90 S70; do BANK84="$BANK84 RQ50e${x}KF5P30Hb20"; done\n')
t = edit(t, "for r in R0Hb20 R0KHb20 R0KP30Hb20 RQ50KF5P30Hb20 RLcKF5P30Hb20 RQ50cKF5P30Hb20 RQ50eN10KF5P30Hb20 RQ50eL25KF5P30Hb20 RQ50KF5Hb20 RLcKF5Hb20 RQ50cKF5Hb20 RQ50eN10KF5Hb20 RQ50eL25KF5Hb20 $NEWRUNS; do",
         "for r in R0Hb20 R0KP30Hb20 RQ50KF5P30Hb20 RLcKF5P30Hb20 RQ50cKF5P30Hb20 RQ50eN10KF5P30Hb20 RQ50eL25KF5P30Hb20 $BANK84 $NEWRUNS; do")
t = edit(t, '"$TABDIR"/TAB_TABLES_RECORD.json "$TABDIR"/tab_*.npz', '"$TBJ" "$TABDIR"/SMOOTH_TABLES_RECORD.json "$TABDIR"/tab_*.npz')
t = edit(t, '--tab-dir "$TABDIR" --part "$PART" --w', '--tabclose-json "$TBJ" --tab-dir "$TABDIR" --w')
t = rename(t)
assert "PART" not in t and "$PS" not in t and "TAB_TABLES" not in t
put("podread_smoothreq10k.sh", t)

# ------------------------------------------------------------------------------------------------ chain
c = load("chain_read_tabclose10k.sh", SHA_CHAIN)
L = c.split("\n")
i_set = L.index("set -u")
head = ["#!/usr/bin/env bash",
        "# chain_read_smoothreq10k.sh -- pod: waits for the queue of card ADD85 (SMOOTHREQ10K_DONE.txt, or both SMOOTHREQ10K_DONE_s0.txt and SMOOTHREQ10K_DONE_s1.txt; any SMOOTHREQ10K_STOP*.txt ends it) AND for the go file",
        "# GO_READ_SMOOTHREQ10K.txt in $W (written by the lead once the hold that carries the card is landed), links the run folders into the pull folder of the readers and runs podread_smoothreq10k.sh once.",
        "# CPU only. No process listing or stopping, no file removal or move. Refuses to run twice (START_READ_SMOOTHREQ10K.txt). The wrapper hash of the queue is checked here as well.",
        "# Last line of the log: ZZ CHAIN-READ-SMOOTHREQ10K END <what>."]
c = "\n".join(head + L[i_set:])
c = edit(c, 'set -u\nPART=${TB_PART:-ALL}\ncase "$PART" in A|B|ALL) ;; *) echo "REFUSED: TB_PART must be A, B or ALL"; exit 2;; esac\nPS=""; [ "$PART" = ALL ] || PS="_$PART"\n', "set -u\n")
c = edit(c, "OUT=$P/out/read_tabclose10k$PS\n", "OUT=$P/out/read_smoothreq10k\n")
c = edit(c, "SD=/workspace/v8proof/closeread/est_tab\n", "SD=/workspace/v8proof/closeread/est_smooth\nTBJ=@@TBJ@@\n")
c = edit(c, '[ -e "$W/TABCLOSE10K_STOP.txt" ] && { echo "ZZ CHAIN-READ-TABCLOSE10K END QUEUE-STOPPED $(cut -c1-200 "$W/TABCLOSE10K_STOP.txt")"; exit 3; }',
         'for sf in "$W"/SMOOTHREQ10K_STOP*.txt; do [ -e "$sf" ] && { echo "ZZ CHAIN-READ-TABCLOSE10K END QUEUE-STOPPED $(cut -c1-200 "$sf")"; exit 3; }; done')
c = edit(c, '[ -f "$W/TABCLOSE10K_DONE$PS.txt" ] && [ -f "$W/GO_READ_TABCLOSE10K.txt" ] && break',
         '{ [ -f "$W/SMOOTHREQ10K_DONE.txt" ] || { [ -f "$W/SMOOTHREQ10K_DONE_s0.txt" ] && [ -f "$W/SMOOTHREQ10K_DONE_s1.txt" ]; }; } && [ -f "$W/GO_READ_TABCLOSE10K.txt" ] && break')
c = edit(c, 'queue done: $(cut -c1-120 "$W/TABCLOSE10K_DONE$PS.txt"); go:', 'queue done (done markers: $(ls "$W" | grep -c "^SMOOTHREQ10K_DONE")); go:')
c = edit(c, 'for d in "$W"/t1/I10RQ50e{TK,STK,L100,W00,W50,W70,W90,S70}*_s[01]; do', 'for d in "$W"/t1/I10RQ50e{STKM,TKD,Z00,S90,S50,S7B,W7B,TK,STK,L100,W00,W50,W70,W90,S70}KF5P30Hb20_s[01]; do')
c = edit(c, '--tab-dir "$SD" --part "$PART"', '--tabclose-json "$TBJ" --tab-dir "$SD"')
c = edit(c, "START_READ_TABCLOSE10K$PS.txt", "START_READ_TABCLOSE10K.txt", 2)
c = c.replace("GO_READ_TABCLOSE10K", "GO_READ_SMOOTHREQ10K")
c = rename(c).replace("TBJ=@@TBJ@@", 'TBJ=${SR_TBJSON:-$P/out/read_tabclose10k/READ_TABCLOSE10K.json}')
assert "PART" not in c and "$PS" not in c
c = c.replace("ls \"$PL\" | grep -c '^I10RQ50e'", "ls \"$PL\" | grep -c '^I10RQ50e'")
put("chain_read_smoothreq10k.sh", c, allow="$P/out/read_tabclose10k/READ_TABCLOSE10K.json")
print("edits", N[0])
