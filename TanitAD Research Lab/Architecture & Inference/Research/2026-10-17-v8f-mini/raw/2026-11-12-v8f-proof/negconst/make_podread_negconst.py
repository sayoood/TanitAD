"""make_podread_negconst.py -- writes podread_negconst10k.sh (pod-side read of card ADD82) and chain_read_negconst10k.sh from constclose/podread_constclose10k_n1.sh (sha256 prefix 07a4eb718a1e9415) and
constclose/chain_read_constclose10k.sh by counted, asserted text edits. Refuses to overwrite its outputs (NC_OVERWRITE_OWN=1 rewrites them).
Usage: bash ops/pycap.sh make_podread_negconst.py"""
import hashlib
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.environ.get("NC_CCDIR") or os.path.normpath(os.path.join(HERE, "..", "constclose"))
OUT_P = os.environ.get("NC_OUT_P") or os.path.join(HERE, "podread_negconst10k.sh")
OUT_C = os.environ.get("NC_OUT_C") or os.path.join(HERE, "chain_read_negconst10k.sh")
for o in (OUT_P, OUT_C):
    if os.path.exists(o) and os.environ.get("NC_OVERWRITE_OWN") != "1":
        sys.exit("REFUSED: %s exists" % o)
n_edits = 0


def load(name, sha):
    raw = io.open(os.path.join(CC, name), "rb").read()
    if sha:
        assert hashlib.sha256(raw).hexdigest()[:16] == sha, name + " changed"
    return raw.decode("utf-8")


class Ed:
    def __init__(self, text):
        self.T = text

    def once(self, a, b, n=1):
        global n_edits
        assert self.T.count(a) == n, (a[:90], self.T.count(a))
        self.T = self.T.replace(a, b)
        n_edits += 1

    def rx(self, pat, rep, minimum=1):
        global n_edits
        self.T, n = re.subn(pat, rep, self.T)
        assert n >= minimum, (pat, n)
        n_edits += 1
        return n


# ================================================================================================ the pod read
P = Ed(load("podread_constclose10k_n1.sh", "07a4eb718a1e9415"))
L = P.T.split("\n")
i_set = L.index("set -u")
assert L[0] == "#!/usr/bin/env bash" and i_set > 5
HEAD = """#!/usr/bin/env bash
# podread_negconst10k.sh -- pod-side read of card PREREG_V8F_PROOF_ADD82 (NEGCONST10K): the reader-fed lead rule with a constant closing input of the cautious sign (form RRe, -1.0 / -2.0 / -4.0 m/s), at age and fresh.
# New runs I10RQ50e<M10|M20|M40>KF5P30Hb20 and I10RQ50e<M10|M20|M40>KF5Hb20; banked (row 137 and card ADD76): RR, RRc, RRh, RLc, no rule R0K at both settings, spread R0Hb20. REQUIRED beside them, with their result files:
# the runs RQ50eN10 and RQ50eL25 of card ADD79 (--estclose-json, controls_all_pass true) and the runs RQ50eC10, RQ50eC20, RQ50eC40 of card ADD80 (--constclose-json, controls_all_pass true), both settings.
# A missing or not admitted result file stops this script (it is a failed control k3 of the reader, so no name could be written anyway).
# CPU only, no rollout. Never reads twice into one folder. It deletes, moves and stops nothing. Derived from the read script of card ADD80 by make_podread_negconst.py.
# Usage: bash podread_negconst10k.sh --pull PULL --code CODE --grid-dir DIR --fresh-grid FILE --midlook-json FILE --const-dir DIR --w W --tree TREE --out OUTDIR --wrapper-sha HEX16 --estclose-json FILE --constclose-json FILE [--read-dir DIR] [--py PYTHON]
#   --read-dir: folder that holds read_negconst10k.py, read_estclose10k.py, closest.py and the engine modules (read_handsrc_decomp.py, read_closrc10k.py, read_agedose_decomp.py, read_agerule_decomp.py,
#               read_longroll10k*.py, strat_roll_read.py); default $PWD
#   --w: queue folder; it must hold NEGCONST10K_DONE.txt and not NEGCONST10K_STOP.txt.  --pull: folder of the run folders I10<run>_s<k>.  --const-dir: the folder of the three constant files of the queue (est_negconst).
# Output folder OUT: READ_NEGCONST10K.json, read_negconst10k.log, selftest_pod.log, READER_HASHES.txt.
# Last line: ZZ PODREAD-NEGCONST10K DONE or ZZ PODREAD-NEGCONST10K STOP <why>."""
P.T = HEAD + "\n" + "\n".join(L[i_set:])
n_edits += 1
P.once('PULL=""; CODE=""; GRIDDIR=""; FG=""; MIDJ=""; CONSTDIR=""; W=""; TREE=""; OUT=""; WSHA=""; ESTJ=""; RD="$PWD"', 'PULL=""; CODE=""; GRIDDIR=""; FG=""; MIDJ=""; CONSTDIR=""; W=""; TREE=""; OUT=""; WSHA=""; ESTJ=""; CONSTJ=""; RD="$PWD"')
P.once("    --estclose-json) ESTJ=$2; shift 2;;\n", "    --estclose-json) ESTJ=$2; shift 2;;\n    --constclose-json) CONSTJ=$2; shift 2;;\n")
P.rx(r"CONSTCLOSE10K", "NEGCONST10K", 8)
P.once("read_constclose10k_n1.py", "read_negconst10k.py", P.T.count("read_constclose10k_n1.py"))
P.once("read_constclose10k.log", "read_negconst10k.log", P.T.count("read_constclose10k.log"))
P.once("^ZZ READ-CONSTCLOSE|", "^ZZ READ-NEGCONST|")
P.once("for v in PULL CODE GRIDDIR FG MIDJ CONSTDIR W TREE OUT WSHA; do", "for v in PULL CODE GRIDDIR FG MIDJ CONSTDIR W TREE OUT WSHA ESTJ CONSTJ; do")
P.once("for c in C10 C20 C40; do [ -f", "for c in M10 M20 M40; do [ -f")
a0 = P.T.index('EARGS=""; ERUNS=""')
a1 = P.T.index('for r in R0Hb20')
P.T = P.T[:a0] + '''[ -f "$ESTJ" ] || stop "MISSING-INPUT $ESTJ"
[ -f "$CONSTJ" ] || stop "MISSING-INPUT $CONSTJ"
grep -q '"controls_all_pass": true' "$ESTJ" || stop "ADD79-READ-NOT-ADMITTED"
grep -q '"controls_all_pass": true' "$CONSTJ" || stop "ADD80-READ-NOT-ADMITTED"
grep -q 'PREREG_V8F_PROOF_ADD79' "$ESTJ" || stop "ADD79-JSON-IS-NOT-THE-RESULT-OF-ADD79"
grep -q 'PREREG_V8F_PROOF_ADD80' "$CONSTJ" || stop "ADD80-JSON-IS-NOT-THE-RESULT-OF-ADD80"
EARGS="--estclose-json $ESTJ --constclose-json $CONSTJ"
ERUNS="RQ50eN10KF5P30Hb20 RQ50eL25KF5P30Hb20 RQ50eN10KF5Hb20 RQ50eL25KF5Hb20"
OLDRUNS="RQ50eC10KF5P30Hb20 RQ50eC20KF5P30Hb20 RQ50eC40KF5P30Hb20 RQ50eC10KF5Hb20 RQ50eC20KF5Hb20 RQ50eC40KF5Hb20"
echo "[$(ts)] the reads of cards ADD79 and ADD80 are admitted: the pairs with N10 and L25 and the runs of +1, +2, +4 m/s are read"
''' + P.T[a1:]
n_edits += 1
P.rx(r"RQ50eC(10|20|40)KF5", r"RQ50eM\1KF5", 6)
# the run list: the new runs, then ERUNS and OLDRUNS (the OLDRUNS were written above after the rename, so they keep their names)
P.once("RQ50eM40KF5Hb20 $ERUNS; do", "RQ50eM40KF5Hb20 $ERUNS $OLDRUNS; do")
# the OLDRUNS and the ERUNS block above were inserted before the rename; put the old names back
P.once('OLDRUNS="RQ50eM10KF5P30Hb20 RQ50eM20KF5P30Hb20 RQ50eM40KF5P30Hb20 RQ50eM10KF5Hb20 RQ50eM20KF5Hb20 RQ50eM40KF5Hb20"', 'OLDRUNS="RQ50eC10KF5P30Hb20 RQ50eC20KF5P30Hb20 RQ50eC40KF5P30Hb20 RQ50eC10KF5Hb20 RQ50eC20KF5Hb20 RQ50eC40KF5Hb20"')
P.once('"$CONSTDIR"/const_C*_H1.npz $ESTJ )', '"$CONSTDIR"/const_M*_H1.npz $ESTJ $CONSTJ )')
for bad in ("CONSTCLOSE", "constclose10k", "const_C"):
    assert bad not in P.T, bad
assert "ESTJ" in P.T and "$EARGS" in P.T and "\r" not in P.T and P.T.count("RQ50eM") == 6 and P.T.count("RQ50eC") == 9, (P.T.count("RQ50eM"), P.T.count("RQ50eC"))   # 6 in OLDRUNS, 3 in the header
io.open(OUT_P, "w", encoding="utf-8", newline="\n").write(P.T)
print("wrote %s sha256 %s" % (os.path.basename(OUT_P), hashlib.sha256(P.T.encode("utf-8")).hexdigest()[:16]))

# ================================================================================================ the chain
C = Ed(load("chain_read_constclose10k.sh", None))
LC = C.T.split("\n")
i_set = LC.index("set -u")
HEADC = """#!/usr/bin/env bash
# chain_read_negconst10k.sh -- pod: waits for the queue of card ADD82 (NEGCONST10K_DONE.txt / NEGCONST10K_STOP.txt) AND for the go file GO_READ_NEGCONST10K.txt (written by the
# lead once the hold that carries the card is landed), links the new run folders into the pull folder of the readers and runs podread_negconst10k.sh once. The read results of cards ADD79
# (read_estclose10k_n2/READ_ESTCLOSE10K.json) and ADD80 (read_constclose10k/READ_CONSTCLOSE10K.json) are passed as required inputs; the runs of those cards must already be linked in the pull folder.
# CPU only. No process listing or stopping, no file removal or move. Refuses to run twice (START_READ_NEGCONST10K.txt).
# Last line of the log: ZZ CHAIN-READ-NEGCONST10K END <what>."""
C.T = HEADC + "\n" + "\n".join(LC[i_set:])
n_edits += 1
C.rx(r"CONSTCLOSE10K", "NEGCONST10K", 6)
C.once("OUT=$P/out/read_constclose10k\n", "OUT=$P/out/read_negconst10k\n")
C.once("EJ=$P/out/read_estclose10k_n2/READ_ESTCLOSE10K.json\n", "EJ=$P/out/read_estclose10k_n2/READ_ESTCLOSE10K.json\nCJ=$P/out/read_constclose10k/READ_CONSTCLOSE10K.json\n")
C.once('for d in "$W"/t1/I10RQ50eC*_s[01]; do', 'for d in "$W"/t1/I10RQ50eM*_s[01]; do')
C.once("""echo "[$(ts)] links of the new runs: $(ls "$PL" | grep -c '^I10RQ50eC')\"""", """echo "[$(ts)] links of the new runs: $(ls "$PL" | grep -c '^I10RQ50eM')\"""")
C.once('bash "$P/podread_constclose10k_n1.sh"', 'bash "$P/podread_negconst10k.sh"')
C.once("--const-dir /workspace/v8proof/closeread/est_const ", "--const-dir /workspace/v8proof/closeread/est_negconst ")
C.once('--estclose-json "$EJ" --read-dir "$RD"', '--estclose-json "$EJ" --constclose-json "$CJ" --read-dir "$RD"')
rest = C.T
for bad in ("CONSTCLOSE10K", "est_const/", "podread_constclose", "I10RQ50eC"):
    assert bad not in rest.replace("READ_CONSTCLOSE10K.json", "").replace("read_constclose10k/", ""), bad
assert "\r" not in C.T
io.open(OUT_C, "w", encoding="utf-8", newline="\n").write(C.T)
print("wrote %s sha256 %s edits %d" % (os.path.basename(OUT_C), hashlib.sha256(C.T.encode("utf-8")).hexdigest()[:16], n_edits))
