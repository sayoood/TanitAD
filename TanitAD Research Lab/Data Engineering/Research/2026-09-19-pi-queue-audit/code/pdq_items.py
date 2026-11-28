import re
import sys

L = open("C:/Users/Admin/AppData/Local/Temp/claude/pdq_tip.md", encoding="utf-8").read().split("\n")
HDR = re.compile(r"^(#{2,3}) .*?(?:ITEM\s+|^#{2,3}\s+)(\d{1,2})\b")


def item_start(n):
    pats = [re.compile(r"^## %d\. " % n), re.compile(r"^#{2,3} .*NEW ITEM %d\b" % n)]
    return [i for i, l in enumerate(L) if any(p.search(l) for p in pats)]


def section(i):
    j = i + 1
    while j < len(L) and not L[j].startswith("## "):
        j += 1
    return i, j


KEY = re.compile(r"DEFAULT|[Dd]efault|RULING|CLOSED|ANSWERED|UPDATE|DECIDE|verbatim|ruled|⇒", re.I)
for n in [int(x) for x in sys.argv[1:]]:
    for s in item_start(n):
        a, b = section(s)
        print("\n======== ITEM %d  lines %d-%d ========" % (n, a + 1, b))
        shown = 0
        for k in range(a, b):
            l = L[k]
            if k < a + 6 or KEY.search(l):
                print("%5d| %s" % (k + 1, l[:240]))
                shown += 1
            if shown > 22:
                print("      ... (truncated)")
                break
