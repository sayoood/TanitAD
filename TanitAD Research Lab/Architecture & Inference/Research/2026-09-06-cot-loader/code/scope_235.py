"""Why 219 and not 235: the QUESTION column is not the corpus SPEAKING.

`probe_speedlimit.py` searches `raw_json_text + "  ||Q|| " + question`;
`classify_limits.py` searches `raw_json_text` alone. The 235 therefore counts
clips where Alpamayo was *ASKED ABOUT* a speed limit as well as clips where it
*MENTIONED* one -- and asking is not stating. ASCII-only.
"""
import json
import re
import sys

import pandas as pd

REC = "C:/Users/Admin/tanitad-data/alpamayo/records.parquet"
df = pd.read_parquet(REC)
N = int(df["clip_id"].nunique())

PAT_PROBE = re.compile(r"speed[\s\-_]?limit", re.I)   # probe_speedlimit.py's
PAT_MINE = re.compile(r"speed[\s\-]?limit", re.I)     # mine


def texts_of(raw):
    out = []
    try:
        d = json.loads(raw)
    except Exception:
        return [str(raw)]

    def walk(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            [walk(x) for x in v]
        elif isinstance(v, dict):
            [walk(x) for x in v.values()]
    walk(d)
    return out


text_only, q_only, either = set(), set(), set()
for _, r in df.iterrows():
    blob = "  ".join(texts_of(r["raw_json"]))
    q = r["question"] if isinstance(r["question"], str) else ""
    t_hit = bool(PAT_PROBE.search(blob))
    q_hit = bool(PAT_PROBE.search(q))
    if t_hit:
        text_only.add(r["clip_id"])
    if q_hit:
        q_only.add(r["clip_id"])
    if t_hit or q_hit:
        either.add(r["clip_id"])

print("CORPUS records.parquet  n_clips=%d  n_rows=%d" % (N, len(df)))
print()
print("  phrase in raw_json TEXT only (the model SPEAKING) : %4d / %d = %.2f %%"
      % (len(text_only), N, 100.0 * len(text_only) / N))
print("  phrase in the QUESTION column (the model ASKED)   : %4d / %d = %.2f %%"
      % (len(q_only), N, 100.0 * len(q_only) / N))
print("  EITHER (what probe_speedlimit.py reported as 235) : %4d / %d = %.2f %%"
      % (len(either), N, 100.0 * len(either) / N))
print("  question-only, never mentioned in any answer      : %4d"
      % len(q_only - text_only))
print()
# a CONTROL that must read a known value: the two regexes must agree on text
alt = set()
for _, r in df.iterrows():
    if PAT_MINE.search("  ".join(texts_of(r["raw_json"]))):
        alt.add(r["clip_id"])
print("  CONTROL: my regex vs the probe's on the SAME text  : %d vs %d  (equal: %s)"
      % (len(alt), len(text_only), alt == text_only))
json.dump({"n_clips": N, "text_only": len(text_only), "question_only": len(q_only),
           "either": len(either), "question_not_text": len(q_only - text_only),
           "regexes_agree_on_text": bool(alt == text_only)},
          open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print("WROTE %s" % sys.argv[1])
