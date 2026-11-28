"""P1 - Is there a POSTED SPEED LIMIT signal in the Alpamayo augmentation?

Searches EVERY text field of EVERY task. Reports fractions with the corpus
named. ASCII-only output (cp1252 dev box).
"""
import json
import re
import sys
from collections import Counter

import pandas as pd

REC = "C:/Users/Admin/tanitad-data/alpamayo/records.parquet"
df = pd.read_parquet(REC)
N_CLIPS = df["clip_id"].nunique()
print("CORPUS: records.parquet  rows=%d  distinct_clips=%d" % (len(df), N_CLIPS))
print()

# ---------------------------------------------------------------- 1. VQA bank
vq = df[df["task"] == "vqa"]
print("=" * 72)
print("1. VQA CATEGORY BANK  (rows=%d, distinct clips=%d)"
      % (len(vq), vq["clip_id"].nunique()))
cats = vq["vqa_category"].value_counts()
print("distinct categories: %d" % len(cats))
for k, v in cats.items():
    print("   %-34s %5d  (%.1f%% of %d clips)" % (k, v, 100.0 * v / N_CLIPS, N_CLIPS))
print()
qs = vq["question"].dropna().unique()
print("distinct QUESTIONS in the bank: %d" % len(qs))
for q in sorted(qs):
    print("   - %s" % q[:150])
print()

# ------------------------------------------------- 2. pull every text field out
def texts_of(raw):
    """Every string the raw_json holds, flattened."""
    out = []
    try:
        d = json.loads(raw)
    except Exception:
        return [str(raw)]
    def walk(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
    walk(d)
    return out

# ------------------------------------------------------------- 3. the patterns
PATS = {
    "speed limit (phrase)": re.compile(r"speed[\s\-_]?limit", re.I),
    "posted":               re.compile(r"\bposted\b", re.I),
    "limit is/of N":        re.compile(r"\blimit\s+(?:is|of|at)\s+\d", re.I),
    "N mph":                re.compile(r"\b\d{1,3}\s*mph\b", re.I),
    "N km/h|kph":           re.compile(r"\b\d{1,3}\s*(?:km/?h|kph|kmh)\b", re.I),
    "speed sign":           re.compile(r"speed\s+sign|limit\s+sign", re.I),
    "sign (any)":           re.compile(r"\bsigns?\b", re.I),
    "school zone":          re.compile(r"school\s+zone", re.I),
    "regulatory":           re.compile(r"\bregulator", re.I),
}

hits = {k: Counter() for k in PATS}          # pattern -> task -> n rows
clip_hits = {k: set() for k in PATS}         # pattern -> set(clip_id)
examples = {k: [] for k in PATS}

for _, r in df.iterrows():
    blob = "  ".join(texts_of(r["raw_json"]))
    q = r["question"] if isinstance(r["question"], str) else ""
    full = blob + "  ||Q|| " + q
    for name, pat in PATS.items():
        m = pat.search(full)
        if m:
            hits[name][r["task"]] += 1
            clip_hits[name].add(r["clip_id"])
            if len(examples[name]) < 6:
                a = max(0, m.start() - 110)
                examples[name].append(
                    (r["task"], r["clip_id"][:8], full[a:m.end() + 110].replace("\n", " ")))

print("=" * 72)
print("2. SPEED-LIMIT PATTERN SEARCH over ALL 5 tasks, ALL text fields")
print("   denominator for clip fractions = %d distinct clips" % N_CLIPS)
print()
print("   %-22s %8s  %-52s" % ("pattern", "clips", "rows by task"))
for name in PATS:
    tot = sum(hits[name].values())
    bytask = ", ".join("%s=%d" % (t, n) for t, n in sorted(hits[name].items()))
    print("   %-22s %4d/%d  %s" % (name, len(clip_hits[name]), N_CLIPS, bytask or "-"))
print()

# --------------------------------------------------- 4. is the ANSWER numeric?
print("=" * 72)
print("3. EXAMPLES (context around each match)")
for name in PATS:
    if not examples[name]:
        continue
    print("-" * 72)
    print("PATTERN: %s   [%d clips]" % (name, len(clip_hits[name])))
    for t, cid, ctx in examples[name]:
        try:
            print("   [%s %s] ...%s..." % (t, cid, ctx[:300]))
        except UnicodeEncodeError:
            print("   [%s %s] <non-ascii context>" % (t, cid))
print()

# ------------------------- 5. THE decisive one: does any NUMBER come with it?
sl = PATS["speed limit (phrase)"]
NUM_NEAR = re.compile(r"speed[\s\-_]?limit[^.]{0,80}?(\d{1,3})", re.I)
n_with_num = set()
num_ex = []
for _, r in df.iterrows():
    blob = "  ".join(texts_of(r["raw_json"]))
    m = NUM_NEAR.search(blob)
    if m:
        n_with_num.add(r["clip_id"])
        if len(num_ex) < 10:
            num_ex.append((r["task"], r["clip_id"][:8], m.group(0)[:200]))
print("=" * 72)
print("4. DECISIVE: 'speed limit' WITH A NUMBER within 80 chars")
print("   clips = %d / %d" % (len(n_with_num), N_CLIPS))
for t, cid, s in num_ex:
    try:
        print("   [%s %s] %s" % (t, cid, s.replace("\n", " ")))
    except UnicodeEncodeError:
        print("   [%s %s] <non-ascii>" % (t, cid))

json.dump({"n_clips": int(N_CLIPS),
           "categories": {k: int(v) for k, v in cats.items()},
           "questions": sorted(qs.tolist()),
           "pattern_clip_counts": {k: len(v) for k, v in clip_hits.items()},
           "pattern_rows_by_task": {k: dict(v) for k, v in hits.items()},
           "speed_limit_with_number_clips": sorted(n_with_num)},
          open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print()
print("WROTE %s" % sys.argv[1])
