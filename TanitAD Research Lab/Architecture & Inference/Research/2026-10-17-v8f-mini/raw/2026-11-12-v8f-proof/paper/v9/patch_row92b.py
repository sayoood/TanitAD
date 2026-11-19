# patch_row92b.py -- lead's second patch of paper v9 (review item 1 of the build agent): the three contact-rate differences of row 90 get their reads.
# One replacement in tanitad_4b_paper.tex, asserted to match exactly once. Runs after patch_row92.py.
import io
import sys

P = "tanitad_4b_paper.tex"
s = io.open(P, encoding="utf-8", newline="").read()
a = "in the order the row lists them)"
b = "for LS with the first fit, LSD with all training clips and LS with all training clips)"
c = s.count(a)
if c != 1:
    print("ZZ PATCH-ROW92B STOP: expected one match, found", c)
    sys.exit(2)
io.open(P, "w", encoding="utf-8", newline="").write(s.replace(a, b))
print("ok edits 1")
