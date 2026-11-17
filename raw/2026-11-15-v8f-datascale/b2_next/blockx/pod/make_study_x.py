"""make_study_x.py -- writes blockx/study/dscale10k_check.py = block3/study/dscale10k_check.py with TWO X-only edits, each asserted to match exactly once.  The block-3 copy (MM-ruled) is read, never edited.
  1. G-LIVE kinds for the X arm = ("tenk",): an arm that trains on the 10k alone has no corpus or mixed batch (the control, a corpus TL-positive batch, is still required and is taken from U_s0's reach table, same code and release).
  2. C-JOIN-EQ for the X arm: the corpus join has no X clip, so the corpus comparison cannot apply.  The X rule: the arm's own join (written by the dry build with --fit-join) equals the UNION join (U_s0's, fitted on 14,512 clips)
     on every 10k clip, array by array, bit for bit (join_eq_problems(Ju = union join, Jc = X join)).  A mismatch is a REFUSAL.  UNVERIFIED on real data: whether --fit-join output is per-clip identical across the two sets.
Prints the unified diff.  Pod, CPU."""
import difflib
import os

D = "/root/dscale10k"
SRC = D + "/block3/study/dscale10k_check.py"
DST = D + "/blockx/study/dscale10k_check.py"
s = open(SRC, encoding="utf-8").read()
a1 = '    kinds = ("corpus",) if spec["set"] == "B" else ("corpus", "tenk", "mixed")\n'
b1 = '    kinds = ("corpus",) if spec["set"] == "B" else (("tenk",) if spec["set"] == "X" else ("corpus", "tenk", "mixed"))      # X (blockx): the 10k alone\n'
assert s.count(a1) == 1
s = s.replace(a1, b1)
a2 = '        if a.corpus_join and a.union_join and os.path.isfile(a.corpus_join) and os.path.isfile(a.union_join):\n'
b2 = ('        if spec["set"] == "X":                                                                  # blockx: the X join must equal the union join on the 10k clips (no corpus clip is in it)\n'
      '            if a.union_join and a.corpus_join and os.path.isfile(a.union_join) and os.path.isfile(a.corpus_join):\n'
      '                put("C-JOIN-EQ", join_eq_problems(dict(np.load(a.corpus_join, allow_pickle=True)), dict(np.load(a.union_join, allow_pickle=True))),\n'
      '                    rule="X: the arm join (--union-join) against the union join (--corpus-join slot) on the X clips; the X clip list is the arm join clip list")\n'
      '            else:\n'
      '                put("C-JOIN-EQ", ["X arm: --corpus-join (the UNION join) / --union-join (the X join) are needed"])\n'
      '        elif a.corpus_join and a.union_join and os.path.isfile(a.corpus_join) and os.path.isfile(a.union_join):\n')
assert s.count(a2) == 1
s = s.replace(a2, b2)
os.makedirs(os.path.dirname(DST), exist_ok=True)
open(DST, "w", encoding="utf-8", newline="\n").write(s)
for l in difflib.unified_diff(open(SRC, encoding="utf-8").read().splitlines(), s.splitlines(), SRC, DST, lineterm="", n=1):
    print(l)
print("ZZ STUDY-X DONE ZZ")
