"""Superset check of a package document against its LANDED blob -- the lander's rule, run here
first: every `code span` and every number of the landed text must survive in the new text (a
rewrite that drops one is a loss the lander stops on, 2026-09-27 at the batch-2 landing).

    python result_superset_check.py <tip-commit> [<package-relative path>=RESULT.md]

Exit 0 == nothing dropped; 1 == the dropped spans / numbers are printed. The comparison is on
SETS (a span quoted twice and now once is not a loss), and it reads the landed blob by
`git cat-file` from the lander repo (40-char asserted).
"""
import pathlib
import re
import subprocess
import sys

TIP = sys.argv[1]
REL_DOC = sys.argv[2] if len(sys.argv) > 2 else "RESULT.md"
PK = pathlib.Path(__file__).resolve().parents[1]
REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-declared-vs-built"
GIT = ["git", "--git-dir=C:/Users/Admin/tanitad-push/.git"]

b = subprocess.run(GIT + ["rev-parse", "--verify", "--quiet", f"{TIP}:{REL}/{REL_DOC}"],
                   capture_output=True, text=True).stdout.strip()
if len(b) != 40:
    raise SystemExit(f"INCONCLUSIVE: no 40-char blob for {REL_DOC} at {TIP} ({b!r})")
old = subprocess.run(GIT + ["cat-file", "-p", b], capture_output=True,
                     check=True).stdout.decode("utf-8")
new = (PK / REL_DOC).read_text(encoding="utf-8")
assert old.strip() and new.strip(), "an empty read is not a comparison"


def spans(t):
    return set(re.findall(r"`([^`\n]+)`", t))


def numbers(t):
    return set(re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", t))


lost_s = sorted(spans(old) - spans(new))
lost_n = sorted(numbers(old) - numbers(new))
print(f"landed blob {b[:12]}: {len(spans(old))} spans / {len(numbers(old))} numbers; "
      f"new: {len(spans(new))} / {len(numbers(new))}")
print(f"dropped spans: {len(lost_s)}  dropped numbers: {len(lost_n)}")
for s in lost_s[:40]:
    print("  span:", s)
for n in lost_n[:40]:
    print("  number:", n)
sys.exit(1 if (lost_s or lost_n) else 0)
