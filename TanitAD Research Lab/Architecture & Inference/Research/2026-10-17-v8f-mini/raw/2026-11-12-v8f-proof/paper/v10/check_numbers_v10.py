# check_numbers_v10.py -- every number token of the version-10 fragments must occur in the register rows it is taken from.
# Usage: python check_numbers_v10.py <row file> [<row file> ...]   (fragments: frag_*.tex in this folder)
# A token is a decimal or comma-formatted number, or an integer of two or more digits. Signs and LaTeX spacing are dropped before the comparison.
# Prints the tokens that are in no row file; the author reviews each (a year, a card id, a count stated in words are expected there).
import glob
import io
import re
import sys

rows = ""
for p in sys.argv[1:]:
    rows += io.open(p, encoding="utf-8").read() + "\n"
rows_tok = set(re.findall(r"\d[\d,]*\.\d+|\d[\d,]*\d|\d", rows))
bad = {}
tot = 0
for f in sorted(glob.glob("frag_*.tex")):
    t = io.open(f, encoding="utf-8").read()
    t = t.replace("\\,", " ").replace("$", " ")
    for tok in re.findall(r"\d[\d,]*\.\d+|\d[\d,]*\d", t):
        tot += 1
        if tok not in rows_tok and tok.rstrip("0").rstrip(".") not in rows_tok:
            bad.setdefault(f, []).append(tok)
print("tokens checked", tot)
for f in bad:
    print(f, "NOT IN ROWS:", " ".join(sorted(set(bad[f]))))
print("ZZ NUMBERS-V10 DONE", sum(len(v) for v in bad.values()))
