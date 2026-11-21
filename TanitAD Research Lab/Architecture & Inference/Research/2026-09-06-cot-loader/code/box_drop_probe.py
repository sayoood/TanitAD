"""The 10 dropped grounding rows the new counter surfaced -- what are they?

⛔ THIS EXISTS BECAUSE MY OWN CLAIM WAS WRONG. RESULT.md §1.5 first said "all
four swallow counters read 0". They do not: `box_json_failed` is 10. I had read
only the counter that `coverage()` happens to expose -- absence found at ONE
location, exactly the error the operating standard names, committed inside the
instrument built to prevent it.

⚠️ WHY IT MATTERS RATHER THAN BEING COSMETIC: `alpamayo_records.py`'s own rule is
that grounding can CONFIRM a token and can NEVER REFUTE one, because a missing
box overwhelmingly means the question was not asked. A box row that FAILED TO
PARSE is indistinguishable from that -- so a dropped row silently converts
"perception we hold" into "perception we were never offered".

ASCII-only stdout.
usage: box_drop_probe.py <out.json>
"""
import json
import sys

import pandas as pd

from tanitad.data import alpamayo_records as AR

df = pd.read_parquet(AR.records_path())
g = df[df["task"] == "grounding_via_vqa"]
bad, ok = [], 0
for _, r in g.iterrows():
    try:
        d = json.loads(r["raw_json"])
    except Exception as exc:
        bad.append({"clip_id": r["clip_id"], "stage": "raw_json",
                    "error": "%s: %s" % (type(exc).__name__, exc)})
        continue
    raw = d.get("box")
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if isinstance(raw, str):
        raw = raw.strip()
    raw = raw or None
    if raw is None:
        continue
    try:
        json.loads(raw)
        ok += 1
    except Exception as exc:
        bad.append({"clip_id": r["clip_id"], "stage": "box",
                    "error": "%s: %s" % (type(exc).__name__, exc),
                    "head": raw[:180], "len": len(raw)})

print("grounding_via_vqa rows        : %d" % len(g))
print("  box payload parsed OK       : %d" % ok)
print("  box payload UNPARSEABLE     : %d   <- silently dropped before today"
      % len(bad))
print("  as a fraction of the task   : %d / %d = %.3f %%"
      % (len(bad), len(g), 100.0 * len(bad) / len(g)))
print()
for b in bad:
    print("  %s  %s" % (b["clip_id"][:8], b["error"]))
    if "head" in b:
        print("      len=%d  head=%s"
              % (b["len"], b["head"].encode("ascii", "replace").decode()))
json.dump({"n_grounding_rows": int(len(g)), "box_ok": ok, "box_bad": len(bad),
           "bad": bad}, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print("WROTE %s" % sys.argv[1])
