"""make_const_files.py -- pod tool of card ADD80 (CONSTCLOSE10K): writes the three constant closing-input estimator files
  const_C10_H1.npz (1.0 m/s)   const_C20_H1.npz (2.0 m/s)   const_C40_H1.npz (4.0 m/s)      history H = 1 frame
with closest.save_planted(path, "const", 1, value) (card ADD80 section 2). closest.py is found through the environment variable CLOSEST_DIR or in the folder of this script.

Control k0 (card ADD80 section 4), per file: loaded with ClosingEstimator.load, the file returns exactly its value for random inputs of shape [1, 3] and [n, 1, 3]; need_frames() equals 5
(H + reader history 5 - 1). The output folder must not exist (it is created here); nothing is removed or moved.
Prints one line per file: "FILE <name> value <v> sha256 <64 hex>"; last line "ZZ CONSTFILES PASS" or "ZZ CONSTFILES FAIL <names>".

  python make_const_files.py [--out-dir /workspace/v8proof/closeread/est_const]
  python make_const_files.py --selftest      (writes into a fresh temporary folder, checks the planted failures of k0 are seen)
"""
import argparse
import hashlib
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VALUES = (("C10", 1.0), ("C20", 2.0), ("C40", 4.0))
H = 1
NEED = 5
DEFAULT_OUT = "/workspace/v8proof/closeread/est_const"


def import_closest():
    for d in (os.environ.get("CLOSEST_DIR", ""), HERE):
        if d and os.path.isfile(os.path.join(d, "closest.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            import closest
            return closest
    raise SystemExit("REFUSED: closest.py not found (set CLOSEST_DIR)")


def file_name(tag):
    return "const_%s_H1.npz" % tag


def k0_check(CL, path, value, seed=0, n=64):
    """-> list of failed sub-checks (empty = pass)"""
    bad = []
    rng = np.random.default_rng(seed)
    E = CL.ClosingEstimator.load(path)
    if E.H != H:
        bad.append("H %d" % E.H)
    if E.need_frames() != NEED:
        bad.append("need_frames %d" % E.need_frames())
    for shape in ((1, 3), (n, 1, 3)):
        scale = rng.choice([0.01, 1.0, 100.0])
        X = rng.normal(0.0, scale, shape) + rng.normal(0.0, 5.0, shape)
        try:
            out = E.apply(X)
        except ValueError as e:
            bad.append("apply refused shape %s" % (shape,))
            continue
        if out.shape != (shape[0],) or not np.all(out == value):
            bad.append("value on shape %s" % (shape,))
    return bad


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run(out_dir, values=VALUES, CL=None, emit=print, tamper=None):
    CL = CL or import_closest()
    if os.path.exists(out_dir):
        emit("REFUSED: %s exists" % out_dir)
        emit("ZZ CONSTFILES FAIL output-exists")
        return 3
    os.makedirs(out_dir)
    failed = []
    for tag, v in values:
        p = os.path.join(out_dir, file_name(tag))
        CL.save_planted(p, "const", H, v)
        if tamper:
            tamper(p, tag)
        bad = k0_check(CL, p, v)
        emit("FILE %s value %.1f sha256 %s%s" % (file_name(tag), v, sha(p), (" k0 FAIL " + ",".join(bad)) if bad else " k0 ok"))
        if bad:
            failed.append(tag)
    emit("ZZ CONSTFILES " + ("PASS" if not failed else "FAIL " + ",".join(failed)))
    return 0 if not failed else 1


def selftest():
    CL = import_closest()
    fails = []

    def chk(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)
    tmp = tempfile.mkdtemp(prefix="tmp_cf_", dir=HERE)
    lines = []
    rc = run(os.path.join(tmp, "a"), CL=CL, emit=lines.append)
    chk("the values of the card: 1.0, 2.0 and 4.0 m/s for C10, C20, C40, history one frame, five frames", [(t, v) for t, v in VALUES] == [("C10", 1.0), ("C20", 2.0), ("C40", 4.0)] and H == 1 and NEED == 5)
    chk("three files, rc 0, PASS line", rc == 0 and lines[-1] == "ZZ CONSTFILES PASS" and len([x for x in lines if x.startswith("FILE ")]) == 3)
    chk("each value as the card has it", all(("value %.1f" % v) in l for (t, v), l in zip(VALUES, lines)))
    lines2 = []
    rc2 = run(os.path.join(tmp, "a"), CL=CL, emit=lines2.append)
    chk("existing output folder is refused", rc2 == 3 and lines2[-1].startswith("ZZ CONSTFILES FAIL"))

    def wrong_value(p, tag):
        CL.save_planted(p, "const", H, 2.5)
    l3 = []
    rc3 = run(os.path.join(tmp, "b"), CL=CL, emit=l3.append, tamper=wrong_value)
    chk("planted wrong value is seen", rc3 == 1 and l3[-1].startswith("ZZ CONSTFILES FAIL"))

    def off_by_a_hair(p, tag):
        CL.save_planted(p, "const", H, dict(VALUES)[tag] + 1e-9)
    lh = []
    chk("planted value off by 1e-9 is seen (k0 asks for exactly the value)", run(os.path.join(tmp, "e"), CL=CL, emit=lh.append, tamper=off_by_a_hair) == 1)

    def wrong_hist(p, tag):
        CL.save_planted(p, "const", 2, dict(VALUES)[tag])
    l4 = []
    rc4 = run(os.path.join(tmp, "c"), CL=CL, emit=l4.append, tamper=wrong_hist)
    chk("planted history of two frames is seen", rc4 == 1)

    def reader_closing(p, tag):
        CL.save_planted(p, "reader_closing", H)
    l5 = []
    rc5 = run(os.path.join(tmp, "d"), CL=CL, emit=l5.append, tamper=reader_closing)
    chk("planted input-dependent file is seen", rc5 == 1)
    print("ZZ CONSTFILES-SELFTEST rc %d (temporary folder %s)" % (0 if not fails else 1, tmp))
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else run(a.out_dir))
