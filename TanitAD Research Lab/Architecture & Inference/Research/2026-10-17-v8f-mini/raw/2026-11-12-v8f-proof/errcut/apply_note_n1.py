"""apply_note_n1.py -- card PREREG_V8F_PROOF_ADD96, note N1: asserted edits of the kit by the lead, before any join of the card is built.
The clause of control e1 that no mixed value is below 0 becomes a recorded count (n_mixed_below_0 per arm and reader seed); such a value is left as it is. Run once.
Usage: python apply_note_n1.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def rd(name):
    with open(os.path.join(HERE, name), encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


def sub(src, old, new, count, what):
    n = src.count(old)
    if n != count:
        sys.exit("REFUSED: %s: expected %d of [%s], found %d" % (what, count, old[:70], n))
    return src.replace(old, new)


def wr(name, text):
    with open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("rewrote", name, len(text.splitlines()), "lines")


m = rd("make_mix_joins.py")
if "n_mixed_below_0" in m:
    sys.exit("REFUSED: the note is applied already")
m = sub(m, '"min_mixed": float(mixed_saved.min()) if mixed_saved.size else None,',
        '"min_mixed": float(mixed_saved.min()) if mixed_saved.size else None, "n_mixed_below_0": int((mixed_saved < 0.0).sum()),', 1, "record count")
m = sub(m, '            if a["min_mixed"] is not None and a["min_mixed"] < 0.0:\n'
           '                bad.append("%s: a mixed value is below 0 (%g)" % (nm, a["min_mixed"]))\n',
        '            if not (isinstance(a.get("n_mixed_below_0"), int) and a["n_mixed_below_0"] >= 0):          # note N1: counted and left as it is, no bar\n'
        '                bad.append("%s: no count of the mixed values below 0 in the record" % nm)\n', 1, "e1 clause")
r = rd("read_errcut10k.py")
r = sub(r, '"min_mixed": x.get("min_mixed")}', '"min_mixed": x.get("min_mixed"), "n_mixed_below_0": x.get("n_mixed_below_0")}', 1, "reader payload")
t = rd("test_errcut.py")
t = sub(t, ' and a["n_mixed_above_60"] >= 0 and a["min_mixed"] >= 0.0\n', ' and a["n_mixed_above_60"] >= 0 and a["n_mixed_below_0"] >= 0\n', 1, "test clean build")
t = sub(t, '    assert MMJ.e1_check(with_("K25_s1", min_mixed=-0.01))["PASS"] is False\n',
        '    assert MMJ.e1_check(with_("K25_s1", min_mixed=-0.01, n_mixed_below_0=3))["PASS"] is True          # note N1: a value below 0 is counted, no bar\n'
        '    assert MMJ.e1_check(with_("K25_s1", n_mixed_below_0=None))["PASS"] is False\n', 1, "test clause")
t += '''

def test_note_n1_a_negative_reader_gap_is_counted_and_left(tmp_path):
    J, dumps, qmaps = MMJ.synthetic_all()
    meta = MMJ.synthetic_meta()
    d0 = dict(dumps[0])
    g = np.array(d0["gap"], np.float64)
    ok = np.flatnonzero(np.asarray(d0["valid"], bool) & np.isfinite(g) & np.asarray(d0["lab_present"], bool))
    g[ok[:7]] = -4.0
    d0["gap"] = g
    rec = MMJ.build(J, {0: d0, 1: dumps[1]}, qmaps, str(tmp_path / "joins"), meta=meta)
    a = rec["arms"]["K100_s0"]
    assert a["n_mixed_below_0"] >= 7 and a["min_mixed"] <= -4.0 + 1e-6
    assert rec["e1"]["PASS"] is True
'''
d = rd("ERRCUT_DEFS.md")
d = d.rstrip("\n") + ("\n\n## Note N1 of the card (applied by the lead with apply_note_n1.py before any join was built)\n"
                      "The clause of e1 that no mixed value is below 0 is a recorded count: `n_mixed_below_0` per arm and reader seed in the record and in the reader's payload. "
                      "Such a value is left as it is; the rule's label source takes a gap as a lead only from 0 to 60 m, so the rule sees no lead on that frame.\n")
wr("make_mix_joins.py", m)
wr("read_errcut10k.py", r)
wr("test_errcut.py", t)
wr("ERRCUT_DEFS.md", d)
print("ZZ APPLY-NOTE-N1 DONE")
