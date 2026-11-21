# make_read_q1b.py -- derives read_q1b.py from read_q1.py with ONE plumbing fix, asserted to match exactly once.
# Control q2 of note PREREG_V8F_PROOF_ADD40_N7 reads: with the quantile head replaced by a constant of 0 m the 6 s position
# error rises against cand-cv. read_q1.sh computes that pair (IPcq0 against IPc); read_q1.py asked for IPcq0 against IPcq,
# a pair that no reader computes, so the control read "pair missing". The fix makes the reader ask for the pair of the note.
# No rule, bar or name changes. Written by the lead before any outcome difference was printed (names were withheld).
import io
import sys

s = io.open("read_q1.py", encoding="utf-8", newline="").read()
a = 'e = E6("IPcq0", "IPcq")'
b = 'e = E6("IPcq0", "IPc")'
if s.count(a) != 1:
    print("ZZ MAKE-READ-Q1B STOP: expected one match, found", s.count(a))
    sys.exit(2)
io.open("read_q1b.py", "w", encoding="utf-8", newline="").write(s.replace(a, b))
print("ok edits 1")
