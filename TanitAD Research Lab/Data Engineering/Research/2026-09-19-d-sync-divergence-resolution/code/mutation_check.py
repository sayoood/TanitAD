"""Mutation check for resolve_divergence.classify — "mutate, do not inspect" (CLAUDE.md).

A classifier that reports "0 files to merge" is only evidence if it CAN report
otherwise. Each mutation below plants genuine backup-only content — the kind a merge
would have to preserve — into a REAL backup text, and the REAL classifier must then
return UNEXPLAINED. The unmutated input must reproduce its original verdict (control).

Mutations, one per verdict class that claims "backup adds nothing":
  M1 REDACTION-ONLY  -> append a new code line to the backup
  M2 REDACTION-ONLY  -> change one numeric value inside the backup JSON
  M3 HEADER+BACKUP   -> edit one line in the backup body
  M4 STUB-FILLED     -> insert a real line just before the SEE_MANIFEST stub
  M5 semantic JSON   -> add a key to one object in the backup JSON
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import resolve_divergence as RD  # noqa: E402

files = json.load(open(sys.argv[1], encoding="utf-8"))["untracked_differ"]
RD.pref.update(RD.build_pref(files))


def pick(suffix):
    return next(p for p in files if p.endswith(suffix))


def load(p):
    bb = RD.git("rev-parse", "%s:%s" % (RD.B, p)).strip().decode()
    hb = RD.git("rev-parse", "HEAD:%s" % p).strip().decode()
    return RD.text(RD.B, p), RD.text("HEAD", p), bb, hb


cases = []
p = pick("code/temporal_filter.py"); b, h, bb, hb = load(p)
cases.append(("control", p, b, h, bb, hb, "REDACTION-ONLY"))
cases.append(("M1 append a code line", p, b + "EXTRA_THRESHOLD = 0.42\n", h, bb, hb, "UNEXPLAINED"))

p = pick("raw/sam3_paint_ours.json"); b, h, bb, hb = load(p)
i = next(k for k, ch in enumerate(b) if ch.isdigit())
mut = b[:i] + ("7" if b[i] != "7" else "3") + b[i + 1:]
cases.append(("control", p, b, h, bb, hb, "REDACTION-ONLY"))
cases.append(("M2 change one number", p, mut, h, bb, hb, "UNEXPLAINED"))

p = pick("code/build_frames.py"); b, h, bb, hb = load(p)
lines = b.splitlines(True); j = len(lines) // 2
lines[j] = lines[j].rstrip("\n") + "  # edited in the backup only\n"
cases.append(("control", p, b, h, bb, hb, "HEADER+BACKUP"))
cases.append(("M3 edit a body line", p, "".join(lines), h, bb, hb, "UNEXPLAINED"))

p = pick("sam3-only-road-map/RESULT.md"); b, h, bb, hb = load(p)
k = b.rstrip("\n").rfind("\n") + 1
cases.append(("control", p, b, h, bb, hb, "STUB-FILLED"))
cases.append(("M4 line before the stub", p, b[:k] + "A finding only the backup had.\n" + b[k:],
              h, bb, hb, "UNEXPLAINED"))

p = pick("code/sample_plan.json"); b, h, bb, hb = load(p)
o = json.loads(b)
first = o[0] if isinstance(o, list) else next(iter(o.values()))
if isinstance(first, list):
    first = first[0]
first["note"] = "backup-only field"
cases.append(("control", p, b, h, bb, hb, "REDACTION-ONLY"))
cases.append(("M5 add a JSON key", p, json.dumps(o), h, bb, hb, "UNEXPLAINED"))

ok = True
rows = []
for name, p, b, h, bb, hb, want in cases:
    got = RD.classify(p, b, h, bb, hb)
    passed = got.startswith(want)
    ok &= passed
    rows.append({"case": name, "file": p.split("Research/")[1], "want": want, "got": got,
                 "pass": passed})
    print("%-4s %-26s %-60s got %s" % ("PASS" if passed else "FAIL", name,
                                       p.split("Research/")[1][:60], got))
json.dump({"all_pass": ok, "cases": rows}, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
print("\nALL MUTATIONS DETECTED, ALL CONTROLS REPRODUCED" if ok else "\nA CHECK FAILED")
sys.exit(0 if ok else 1)
