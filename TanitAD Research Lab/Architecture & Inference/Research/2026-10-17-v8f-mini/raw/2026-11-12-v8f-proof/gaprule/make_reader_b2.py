"""make_reader_b2.py -- card PREREG_V8F_PROOF_ADD95: the repaired reader and its pod script, by asserted edits.
The first controls read on the pod stopped in the engine's grid read before any value existed: the engine reads the attribute n_boot_control of the argument namespace, which the reader
of card ADD85 sets to 300 after parsing and the reader of this kit did not set. read_gaprule10kb2.py sets it the same way; nothing else changes. podread_gaprule10kb2.sh runs it into the
new folder read2. Refuses to write over an existing file. Usage: python make_reader_b2.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def rd(name):
    with open(os.path.join(HERE, name), encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


def sub(src, old, new, count, what):
    n = src.count(old)
    if n != count:
        sys.exit("REFUSED: %s: expected %d of [%s], found %d" % (what, count, old[:60], n))
    return src.replace(old, new)


def wr(name, text):
    p = os.path.join(HERE, name)
    if os.path.exists(p):
        sys.exit("REFUSED: %s exists" % name)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote", name, len(text.splitlines()), "lines")


r = rd("read_gaprule10kb.py")
r = sub(r, '"""read_gaprule10kb.py -- ', '"""read_gaprule10kb2.py (read_gaprule10kb.py with the engine attribute n_boot_control set to 300 after parsing, as the reader of card ADD85 sets it) -- ', 1, "head")
r = sub(r, "    a = ap.parse_args(argv)\n", "    a = ap.parse_args(argv)\n    a.n_boot_control = 300\n", 1, "attribute")
wr("read_gaprule10kb2.py", r)
p = rd("podread_gaprule10kb.sh")
p = sub(p, "# podread_gaprule10kb.sh -- ", "# podread_gaprule10kb2.sh (the repaired reader read_gaprule10kb2.py, output folder read2) -- ", 1, "pod head")
p = sub(p, "OUT=$GP/read;", "OUT=$GP/read2;", 1, "pod out")
p = sub(p, "read_gaprule10kb.py", "read_gaprule10kb2.py", p.count("read_gaprule10kb.py"), "pod reader name")
if p.count("read_gaprule10kb2.py") < 5:
    sys.exit("REFUSED: reader name replaced fewer than 5 times")
wr("podread_gaprule10kb2.sh", p)
print("ZZ MAKE-READER-B2 DONE")
