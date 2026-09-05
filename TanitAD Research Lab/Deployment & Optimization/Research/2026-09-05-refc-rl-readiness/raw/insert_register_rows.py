"""Insert the D-RL-READY rows into GOALS_AND_CLAIMS.md — INSERT, never rewrite.
Re-reads the register immediately before editing; refuses if any row id already
exists; inserts the block right BEFORE the `## D-REFCV4B-EGODROP2` section header
(i.e. after the D-REFC-DDAUDIT block), or at EOF if that header moved.
Usage: python insert_register_rows.py <register.md> <rows.md>"""
import re
import sys

reg, rows = sys.argv[1], sys.argv[2]
block = open(rows, encoding="utf-8").read().rstrip("\n") + "\n\n"
ids = re.findall(r"^\| ([A-Z][A-Z0-9-]+) \|", block, flags=re.M)
assert ids, "no rows found in the rows file"
s = open(reg, encoding="utf-8").read()                        # re-read NOW
present = [i for i in ids if re.search(rf"^\| {re.escape(i)} \|", s, flags=re.M)]
if present:
    raise SystemExit(f"REFUSED: rows already present in the register: {present}")
anchor = "## D-REFCV4B-EGODROP2"
i = s.find(anchor)
if i < 0:
    new = s.rstrip("\n") + "\n\n" + block
    where = "EOF"
else:
    new = s[:i] + block + s[i:]
    where = f"before {anchor!r} at char {i}"
assert len(new) == len(s) + len(block), "insertion changed more than the inserted block"
open(reg, "w", encoding="utf-8", newline="\n").write(new)
back = open(reg, encoding="utf-8").read()
missing = [i for i in ids if not re.search(rf"^\| {re.escape(i)} \|", back, flags=re.M)]
assert not missing, f"rows not found after write: {missing}"
print(f"inserted {ids} {where}; register {len(s)} -> {len(back)} chars")
