"""make_negconst_gen.py -- writes make_negconst_files.py (pod tool of card ADD82, control k0) from constclose/make_const_files.py by counted, asserted text edits (parent hash printed).
Usage: bash ops/pycap.sh make_negconst_gen.py"""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "constclose", "make_const_files.py"))
OUT = os.path.join(HERE, "make_negconst_files.py")
raw = io.open(SRC, "rb").read()
PARENT_SHA = hashlib.sha256(raw).hexdigest()[:16]
if os.path.exists(OUT) and os.environ.get("NC_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists" % OUT)
T = raw.decode("utf-8")
n_edits = 0


def once(a, b, n=1):
    global T, n_edits
    assert T.count(a) == n, (a[:70], T.count(a))
    T = T.replace(a, b)
    n_edits += 1


end = T.index('python make_const_files.py --selftest') 
end = T.index('"""', end) + 3
DOC = '''"""make_negconst_files.py -- pod tool of card ADD82 (NEGCONST10K): writes the three constant closing-input estimator files of the CAUTIOUS sign
  const_M10_H1.npz (-1.0 m/s)   const_M20_H1.npz (-2.0 m/s)   const_M40_H1.npz (-4.0 m/s)      history H = 1 frame
with closest.save_planted(path, "const", 1, value) (card ADD82 section 2). closest.py is found through the environment variable CLOSEST_DIR or in the folder of this script.

Control k0 (card ADD82 section 4), per file: loaded with ClosingEstimator.load, the file returns exactly its value for random inputs of shape [1, 3] and [n, 1, 3]; need_frames() equals 5
(H + reader history 5 - 1). The output folder must not exist (it is created here); nothing is removed or moved.
Prints one line per file: "FILE <name> value <v> sha256 <64 hex>"; last line "ZZ NEGCONSTFILES PASS" or "ZZ NEGCONSTFILES FAIL <names>".

  python make_negconst_files.py [--out-dir /workspace/v8proof/closeread/est_negconst]
  python make_negconst_files.py --selftest      (writes into a fresh temporary folder, checks the planted failures of k0 are seen)
Derived from the tool of card ADD80 (sha256 prefix %s) by make_negconst_gen.py.
"""''' % PARENT_SHA
T = DOC + T[end:]
n_edits += 1
once('VALUES = (("C10", 1.0), ("C20", 2.0), ("C40", 4.0))', 'VALUES = (("M10", -1.0), ("M20", -2.0), ("M40", -4.0))')
once('DEFAULT_OUT = "/workspace/v8proof/closeread/est_const"', 'DEFAULT_OUT = "/workspace/v8proof/closeread/est_negconst"')
once("ZZ CONSTFILES", "ZZ NEGCONSTFILES", 6)
once('tempfile.mkdtemp(prefix="tmp_cf_", dir=HERE)', 'tempfile.mkdtemp(prefix="tmp_nf_", dir=HERE)')
once('chk("the values of the card: 1.0, 2.0 and 4.0 m/s for C10, C20, C40, history one frame, five frames", [(t, v) for t, v in VALUES] == [("C10", 1.0), ("C20", 2.0), ("C40", 4.0)] and H == 1 and NEED == 5)',
     'chk("the values of the card: -1.0, -2.0 and -4.0 m/s for M10, M20, M40, history one frame, five frames", [(t, v) for t, v in VALUES] == [("M10", -1.0), ("M20", -2.0), ("M40", -4.0)] and H == 1 and NEED == 5)')
once('CL.save_planted(p, "const", H, 2.5)', 'CL.save_planted(p, "const", H, 2.0)')
once('chk("planted wrong value is seen",', 'chk("planted value of the wrong sign (+2.0 instead of the card value) is seen",')
once('''    def reader_closing(p, tag):''', '''    def flipped_sign(p, tag):
        CL.save_planted(p, "const", H, -dict(VALUES)[tag])
    lf = []
    chk("planted flipped sign of every value is seen", run(os.path.join(tmp, "f"), CL=CL, emit=lf.append, tamper=flipped_sign) == 1 and lf[-1].startswith("ZZ NEGCONSTFILES FAIL M10,M20,M40"))

    def reader_closing(p, tag):''')
assert "CONSTFILES PASS" not in T.replace("NEGCONSTFILES PASS", "")
assert "const_C" not in T and '"C10"' not in T and "est_const" not in T.replace("est_negconst", "")
io.open(OUT, "w", encoding="utf-8", newline="\n").write(T)
print("wrote %s sha256 %s edits %d parent %s" % (os.path.basename(OUT), hashlib.sha256(T.encode("utf-8")).hexdigest()[:16], n_edits, PARENT_SHA))
