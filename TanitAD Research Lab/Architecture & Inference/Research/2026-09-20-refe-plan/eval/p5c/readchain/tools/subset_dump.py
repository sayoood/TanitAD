"""MM 2026-11-17: REHEARSAL ONLY. Cut a stage-2 proposal dump (npz with a `token` array; per-token arrays share its first dim) to the tokens of a
rehearsal plans file, so p5c_tactical's dump-vs-seam token-set check compares like with like (its --tokens-from restricts the seams, not the dump).
Scalars and arrays whose first dim is not the token count are copied unchanged. The real read never calls this (dump and seams are both 12,146).
usage: subset_dump.py <dump.npz> <tokens-from npz (has `token`)> <out.npz>"""
import sys

import numpy as np

src, tok_from, out = sys.argv[1:4]
z = np.load(src, allow_pickle=False)
tz = np.load(tok_from, allow_pickle=False)
want = [str(t) for t in tz["tokens" if "tokens" in tz.files else "token"]]          # plans npz use `tokens` (as p5c_tactical reads them)
tok = [str(t) for t in z["token"]]
pos = {t: i for i, t in enumerate(tok)}
missing = [t for t in want if t not in pos]
assert not missing, f"{len(missing)} rehearsal tokens are not in the dump, e.g. {missing[:3]}"
idx = np.array([pos[t] for t in want], dtype=np.int64)
n = len(tok)
res = {k: (z[k][idx] if z[k].ndim and z[k].shape[0] == n else z[k]) for k in z.files}
np.savez(out, **res)
print("ZZSUBSET_DUMP", len(idx), "of", n, "->", out)
