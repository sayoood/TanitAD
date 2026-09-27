"""map_progress_to_ids.py <pytest -q log> <collect-only list> <reference log> [<reference log> ...] -> the FAILED/ERROR
ids of a run whose log carries only progress characters (pytest -q -rs prints no FAILED/ERROR lines).

The progress characters are mapped 1:1 onto the collected ids IN ORDER -- refused (exit 2) unless the two counts are
EQUAL (a teardown error prints a second character for one test and would shift every later id). Each F/E id is then
looked up in the reference logs' FAILED/ERROR lines (e.g. the Thor tip AND landed logs of the same test list); a
dev-box rootdir of ``stack/`` prints ``tests/...``, normalised to ``stack/tests/...``; ids outside the rootdir print
as ``::name`` and are matched by the test name. Prints one line per F/E id and the count found in NO reference."""
import re
import sys
from collections import Counter
from pathlib import Path

log, collect, refs = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
ids = [ln.strip() for ln in collect.read_text(encoding="utf-8").splitlines() if "::" in ln]
chars = []
for ln in log.read_text(encoding="utf-8", errors="replace").splitlines():
    m = re.match(r"^([.FEsxX]+)\s*(\[\s*\d+%\])?\s*$", ln)
    if m:
        chars.extend(m.group(1))
if len(ids) != len(chars):
    print(f"REFUSED: {len(chars)} progress characters vs {len(ids)} collected ids -- not a 1:1 map")
    sys.exit(2)
ref = set()
for r in refs:
    for ln in r.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^(FAILED|ERROR) (\S+)", ln)
        if m:
            ref.add(m.group(2))
ref_names = {t.split("::")[-1] for t in ref}


def norm(x):
    return x if x.startswith(("stack/", "taniteval/", "::")) else "stack/" + x


bad = [(c, norm(i)) for c, i in zip(chars, ids) if c in "FE"]
alone = 0
for c, i in bad:
    hit = i in ref or (i.startswith("::") and i.split("::")[-1] in ref_names)
    alone += not hit
    print(f"{c} {i}  {'(in the reference logs)' if hit else '(IN NO REFERENCE LOG)'}")
print(f"{len(ids)} ids, {Counter(c for c, _ in bad)}; in no reference log: {alone}")
sys.exit(1 if alone else 0)
