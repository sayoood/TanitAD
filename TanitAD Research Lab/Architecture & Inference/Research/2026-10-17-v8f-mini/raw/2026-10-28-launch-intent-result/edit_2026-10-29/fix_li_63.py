"""Fix the stale M-LAUNCH lines of RESULT_V8F_LAUNCH_INTENT.md (summary bullet + section 6.3 table): M-LAUNCH was read NOT MOVED
(section 6.2), so the rows say so; a dated note under the table discloses the edit. Exact replacements, each asserted to occur once."""
import subprocess, sys, hashlib
TIP = sys.argv[1]
P = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/RESULT_V8F_LAUNCH_INTENT.md"
b = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*", "show", f"{TIP}:{P}"], capture_output=True, check=True).stdout
assert b"\r" not in b
open("tip.md", "wb").write(b)
R = [
 ("NEXT-LAUNCH (PREREG section 11) routes by M-LAUNCH, which is built separately; its slot is open in section 6.2.",
  "NEXT-LAUNCH (PREREG section 11) routes by M-LAUNCH, built separately; section 6.2 holds its result, **NOT MOVED**, which routes to LAUNCH-INTENT-HI (its own PI go)."),
 ("| M-LAUNCH (the section 11 routing variable) | NOT BUILT in the reader; being built separately | its build; it reads the tick dumps, which are single-copy on F: (section 8, item 6) |",
  "| M-LAUNCH (the section 11 routing variable) | built separately and read: **NOT MOVED** (section 6.2) | nothing; it reads the tick dumps, which are single-copy on F: (section 8, item 6) |"),
 ("| LOOP-STATE (zero training) | routed from M-LAUNCH *moved* | M-LAUNCH |",
  "| LOOP-STATE (zero training) | routed from M-LAUNCH *moved*; M-LAUNCH read *not moved*, so this route is not taken | – |"),
 ("| LAUNCH-INTENT-HI (W 4.17852) | routed from M-LAUNCH *not moved* | M-LAUNCH, and its own PI go for box 2 |",
  "| LAUNCH-INTENT-HI (W 4.17852) | the route M-LAUNCH selected (*not moved*) | its own PI go for box 2 |"),
]
t = b.decode("utf-8")
for o, n in R:
    assert t.count(o) == 1, o[:60]
    t = t.replace(o, n)
ANCH = "| anti-echo block; M2 / L2 launch subset | NOT BUILT | build items for the reader |\n"
assert t.count(ANCH) == 1
NOTE = ("\n*(Edited 2026-10-29 by the Master Mind: the M-LAUNCH, LOOP-STATE and LAUNCH-INTENT-HI rows above, and the Next bullet at the top, now"
        " state the M-LAUNCH result that section 6.2 holds. As landed, they said M-LAUNCH was being built. No number changed.)*\n")
i = t.index(ANCH) + len(ANCH)
t = t[:i] + NOTE + t[i:]
for w in ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn"):
    assert w not in "".join(n for _, n in R).lower() + NOTE.lower(), w
nb = t.encode("utf-8")
open("new.md", "wb").write(nb)
print("tip", hashlib.sha256(b).hexdigest()[:12], "new", hashlib.sha256(nb).hexdigest()[:12], len(b), len(nb))
