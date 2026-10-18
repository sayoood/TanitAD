#!/usr/bin/env python3
"""Regenerate PICKCON_FILE_REVERSALS (refe/test_rl_s1b.py) = the (new, old) text pairs that reverse the PICKCON edits of refe/scorer_finetune.py back to the 0a5958a trainer
(git blob e063928807b7e0df79b1c1ddfe192596ca1ff601).  usage: python gen_pickcon_reversals.py <base scorer_finetune.py> <patched scorer_finetune.py> <test_rl_s1b.py to splice into>
Each pair is grown with leading context lines until the NEW snippet is unique in the patched file; the reversed file must be byte-identical to the base."""
import difflib, hashlib, pprint, sys
base_p, new_p, test_p = sys.argv[1:4]
BASE_BLOB = "e063928807b7e0df79b1c1ddfe192596ca1ff601"
old = open(base_p, encoding="utf-8", newline="").read()
new = open(new_p, encoding="utf-8", newline="").read()
ol, nl = old.splitlines(keepends=True), new.splitlines(keepends=True)
pairs = []
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ol, nl, autojunk=False).get_opcodes():
    if tag == "equal":
        continue
    c = 1
    while True:
        nsn, osn = "".join(nl[max(0, j1 - c):j2]), "".join(ol[max(0, i1 - c):i2])
        if new.count(nsn) == 1:
            break
        c += 1
        assert c < 12
    pairs.append((nsn, osn))
s = new
for nsn, osn in pairs:
    assert s.count(nsn) == 1
    s = s.replace(nsn, osn)
blob = lambda t: hashlib.sha1(b"blob %d\0" % len(t.encode()) + t.encode()).hexdigest()  # noqa: E731
assert s == old and blob(s) == BASE_BLOB, (blob(s), BASE_BLOB)
body = "# <<<GEN PICKCON_FILE_REVERSALS\nPICKCON_FILE_REVERSALS = " + pprint.pformat(tuple(pairs), width=190) + "\n# >>>GEN PICKCON_FILE_REVERSALS\n"
t = open(test_p, encoding="utf-8", newline="").read()
a, b = t.find("# <<<GEN PICKCON_FILE_REVERSALS"), t.find("# >>>GEN PICKCON_FILE_REVERSALS")
assert a >= 0 and b > a
t = t[:a] + body.rstrip("\n") + t[b + len("# >>>GEN PICKCON_FILE_REVERSALS"):]
open(test_p, "w", encoding="utf-8", newline="").write(t)
print(f"{len(pairs)} pairs spliced; patched blob {blob(new)}; reversal -> {blob(s)}")
