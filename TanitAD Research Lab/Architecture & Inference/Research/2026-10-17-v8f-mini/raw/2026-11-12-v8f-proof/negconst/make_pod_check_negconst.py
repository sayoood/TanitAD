"""make_pod_check_negconst.py -- writes pod_check_negconst.sh from constclose/pod_check_constclose.sh (sha256 prefix d7e6a17fc819efe1) by counted, asserted text edits.
Refuses to overwrite its output (NC_OVERWRITE_OWN=1 rewrites it). Usage: bash ops/pycap.sh make_pod_check_negconst.py"""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("NC_SRC") or os.path.normpath(os.path.join(HERE, "..", "constclose", "pod_check_constclose.sh"))
OUT = os.environ.get("NC_OUT") or os.path.join(HERE, "pod_check_negconst.sh")
raw = io.open(SRC, "rb").read()
assert hashlib.sha256(raw).hexdigest()[:16] == "d7e6a17fc819efe1", "parent changed"
if os.path.exists(OUT) and os.environ.get("NC_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists" % OUT)
T = raw.decode("utf-8")
n_edits = 0


def once(a, b, n=1):
    global T, n_edits
    assert T.count(a) == n, (a[:70], T.count(a))
    T = T.replace(a, b)
    n_edits += 1


L = T.split("\n")
i_set = L.index("set -u")
assert L[0] == "#!/usr/bin/env bash" and 1 <= i_set <= 12, i_set
HEAD = """#!/usr/bin/env bash
# pod_check_negconst.sh -- the pod check of card ADD82 (NEGCONST10K), control k1: the constant reaches the rule with its sign. Runs check_negconst.py: ONE small rollout of the wrapper
# v8f_t1_eval_reader_rule_rzgfhqe.py in form RRe with const_M20_H1.npz (-2.0 m/s) on the first three clips of the pinned grid (model seed 0, readout K, --rule-every 5, q50 map), and the checks on its logs,
# among them the sign check K1e (v_L = max(0, v_l - 2.0), below v_l wherever v_l is above zero). Derived from the check script of card ADD80 by make_pod_check_negconst.py.
# Writes only into $W/negconst_test. No process listing or stopping, no file removal or move. Refuses to run twice with the same tag. One other GPU compute process is admitted (never more than three GPU jobs).
# Environment (all optional): TAG (default nk1; the marker and record names), RUNTAG (the run folder, default = TAG), ANALYZE=1 (only read the run folder already there; give a new TAG and the RUNTAG of the
# finished rollout), NCLIPS (default 3), CONSTDIR (default /workspace/v8proof/closeread/est_negconst; the files written by make_negconst_files.py).
# Markers: $W/negconst_test/CHECK_NEG_DONE_$TAG.txt (the last line of the check: ZZ CHECK-NEGCONST PASS ... or FAIL ...) or CHECK_NEG_STOP_$TAG.txt.
# Record: $W/negconst_test/CHECK_NEGCONST_$TAG.txt (output), .hashes (sha256 prefixes of the files: the check, the wrapper, closest.py and the three constant files), .rc."""
T = HEAD + "\n" + "\n".join(L[i_set:])
n_edits += 1
once("constclose_test", "negconst_test", T.count("constclose_test"))
once("CHECK_CONST_STOP_", "CHECK_NEG_STOP_", T.count("CHECK_CONST_STOP_"))
once("CHECK_CONST_DONE_", "CHECK_NEG_DONE_", T.count("CHECK_CONST_DONE_"))
once("CHECK_CONSTCLOSE_", "CHECK_NEGCONST_", T.count("CHECK_CONSTCLOSE_"))
once("check_constclose.py", "check_negconst.py", T.count("check_constclose.py"))
once("ZZ POD-CHECK-CONSTCLOSE", "ZZ POD-CHECK-NEGCONST", T.count("ZZ POD-CHECK-CONSTCLOSE"))
once("est_const", "est_negconst", T.count("est_const"))
once("TAG=${TAG:-ck1}", "TAG=${TAG:-nk1}")
for c in ("C10", "C20", "C40"):
    once(c, "M" + c[1:], T.count(c))
assert "CONSTCLOSE" not in T and "constclose" not in T and "const_C" not in T and "CHECK_CONST" not in T, "a name of the parent card is left"
assert "-2.0" in T and "const_M20_H1.npz" in T and "\r" not in T
io.open(OUT, "w", encoding="utf-8", newline="\n").write(T)
print("wrote %s sha256 %s edits %d" % (os.path.basename(OUT), hashlib.sha256(T.encode("utf-8")).hexdigest()[:16], n_edits))
