"""Merge @@NNN-framed pulls (best-of across files), report remaining holes."""
import json
import re
import sys

lines = {}
for path in sys.argv[1:]:
    for m in re.finditer(r"@@(\d{3}) (.*?)##", open(path).read()):
        i, txt = int(m.group(1)), m.group(2)
        # later files are targeted refetches — they override earlier captures
        lines[i] = txt
mx = max(lines)
missing = [i for i in range(mx + 1) if i not in lines]
print("still missing:", missing, file=sys.stderr)
text = "\n".join(lines[i] for i in sorted(lines))
try:
    obj = json.loads(text)
    print("JSON VALID", file=sys.stderr)
    print(json.dumps(obj, indent=1, sort_keys=True))
except Exception as e:
    print(f"JSON INVALID: {e}", file=sys.stderr)
    print(text)
