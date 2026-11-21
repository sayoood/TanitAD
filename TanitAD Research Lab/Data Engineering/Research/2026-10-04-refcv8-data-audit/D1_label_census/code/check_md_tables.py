"""Markdown table sanity: every row of a table has the same number of UNESCAPED pipes as its header."""
import re
import sys
from pathlib import Path

txt = Path(sys.argv[1]).read_text(encoding="utf-8")
pipe = re.compile(r"(?<!\\)\|")
lines = txt.split("\n")
bad = []
n_tables = 0
i = 0
while i < len(lines):
    if lines[i].startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-"):
        n_tables += 1
        n = len(pipe.findall(lines[i]))
        j = i
        while j < len(lines) and lines[j].startswith("|"):
            if len(pipe.findall(lines[j])) != n:
                bad.append((j + 1, lines[j][:90]))
            j += 1
        i = j
    else:
        i += 1
print("tables", n_tables, "rows with a wrong pipe count", len(bad))
for b in bad[:12]:
    print(b)
