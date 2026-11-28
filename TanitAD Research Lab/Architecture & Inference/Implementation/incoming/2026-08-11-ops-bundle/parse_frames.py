"""Reassemble @@NNN-framed lines pulled over the PTY into the original text."""
import re
import sys

src = open(sys.argv[1]).read()
lines = {}
for m in re.finditer(r"@@(\d{3}) (.*?)##", src):
    lines[int(m.group(1))] = m.group(2)
if not lines:
    print("NO FRAMES")
    sys.exit(1)
mx = max(lines)
missing = [i for i in range(mx + 1) if i not in lines]
print("missing:", missing)
print("\n".join(lines[i] for i in sorted(lines)))
