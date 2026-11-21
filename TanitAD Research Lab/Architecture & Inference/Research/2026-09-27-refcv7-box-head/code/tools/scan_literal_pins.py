"""Literal source pins my rewrite could break (the Master Mind's item 4, 2026-09-27; NEW-2's gate found one the hard way).

For every test file in the TIP tree (stack/tests, taniteval/tests, plus any tests/ dir), every string constant of
>= 6 characters (ast) is checked against each SHARED file this landing rewrites: a literal that occurs in the tip's
file but NOT in the landed file is a pin the landing breaks -- reported with the test file and line. A literal that
never occurred in the tip file is not a pin on it (the tests pass or fail on their own). Also reported: tests that
read one of the shared files by NAME, so a reader can see the full exposure.

usage: python scan_literal_pins.py <tip tree> <landed tree> <shared repo path>...
"""
import ast
import sys
from pathlib import Path

tip, landed = Path(sys.argv[1]), Path(sys.argv[2])
shared = sys.argv[3:]
texts = {p: ((tip / p).read_text(encoding="utf-8"), (landed / p).read_text(encoding="utf-8")) for p in shared}
tests = sorted(x for d in ("stack/tests", "taniteval/tests") for x in (tip / d).rglob("test_*.py"))
broken, readers = [], {}
for t in tests:
    src = t.read_text(encoding="utf-8", errors="replace")
    for p in shared:
        if Path(p).name in src:
            readers.setdefault(p, []).append(t.relative_to(tip).as_posix())
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) >= 6:
            lit = node.value
            for p, (a, b) in texts.items():
                if Path(p).name not in src:        # a SOURCE pin needs the test to read the file by name
                    continue
                if lit in a and lit not in b:
                    broken.append((t.relative_to(tip).as_posix(), node.lineno, p, lit[:100]))
print(f"scanned {len(tests)} test files x {len(shared)} shared files")
for p in shared:
    print(f"  tests naming {Path(p).name}: {len(readers.get(p, []))}")
if broken:
    print(f"LITERALS PRESENT AT THE TIP BUT GONE AFTER THE LANDING: {len(broken)}")
    for row in broken:
        print("  ", row)
else:
    print("no test literal that occurs in a tip shared file is removed by the landing")
