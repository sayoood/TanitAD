"""P1 step 2 - split the speed-limit mentions into READ-OFF-A-SIGN vs
LANGUAGE-PRIOR GUESS, and measure the grounded share.

A prior-based guess ("residential areas typically have 25-30 mph") is the VLM's
world knowledge about road types, NOT perception. Admitting it would be the
`road_class` circularity in a new costume. ASCII-only output.
"""
import json
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

REC = "C:/Users/Admin/tanitad-data/alpamayo/records.parquet"
df = pd.read_parquet(REC)
N = df["clip_id"].nunique()


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


# a value ASSERTED as read off a sign / gantry
READ = re.compile(
    r"(?:speed[\s\-]?limit\s+sign|limit\s+sign|gantry\s+signs?|sign)\s*"
    r"(?:[a-z,\s]{0,40}?)(?:indicat\w+|display\w*|read\w*|shows?|of|is|posted|:)?\s*"
    r"(\d{1,3})\s*(km/?h|kph|mph)?"
    r"|(?:a|an)\s+(\d{1,3})\s*(km/?h|kph|mph)?\s*speed[\s\-]?limit\s+sign"
    r"|speed[\s\-]?limit\s+(?:of|is|at)\s+(\d{1,3})\s*(km/?h|kph|mph)?", re.I)

# hedging that marks a LANGUAGE PRIOR rather than a reading
HEDGE = re.compile(
    r"\btypically\b|\busually\b|\bgenerally\b|\boften\b|\blikely\b|\bprobabl\w+"
    r"|\bcould\s+(?:range|be)\b|\bwould\s+(?:be|likely)\b|\bapproximately\b"
    r"|\bestimate\w*\b|\bassum\w+|\bif\s+not\b|\bdepending\s+on\s+local\b"
    r"|\bnot\s+visible\b|\bnot\s+observed\b|\bexact\s+speed", re.I)

NEG = re.compile(
    r"\b(?:no|not)\b[^.]{0,40}?speed[\s\-]?limit"
    r"|speed[\s\-]?limit[^.]{0,40}?\bnot\s+(?:visible|observed|present|shown)",
    re.I)

rows = defaultdict(dict)          # clip -> task -> blob
for _, r in df.iterrows():
    rows[r["clip_id"]][r["task"]] = "  ".join(texts_of(r["raw_json"]))

read_clips, hedge_clips, neg_clips = {}, {}, {}
values = Counter()
units = Counter()
per_task_read = Counter()

for cid, byt in rows.items():
    for task, blob in byt.items():
        if "speed limit" not in blob.lower() and "speed-limit" not in blob.lower():
            continue
        neg = bool(NEG.search(blob))
        hed = bool(HEDGE.search(blob))
        m = READ.search(blob)
        val = None
        if m:
            g = [x for x in m.groups() if x and x.isdigit()]
            u = [x for x in m.groups() if x and not x.isdigit()]
            if g:
                val = int(g[0])
        # a READ requires: a number attached to a sign, NOT hedged, NOT negated
        if val is not None and not hed and not neg:
            read_clips.setdefault(cid, (task, val, (u[0] if u else None),
                                        m.group(0)[:120]))
            per_task_read[task] += 1
            values[val] += 1
            units[(u[0].lower() if u else "NONE")] += 1
        elif hed:
            hedge_clips.setdefault(cid, (task, m.group(0)[:80] if m else ""))
        elif neg:
            neg_clips.setdefault(cid, task)

print("CORPUS records.parquet  distinct clips = %d" % N)
print()
print("=" * 72)
print("SPEED-LIMIT MENTION, CLASSIFIED  (clip-level, any task)")
print("  READ off a sign, unhedged, with a VALUE : %4d / %d" % (len(read_clips), N))
print("  HEDGED (language prior / 'typically')   : %4d / %d" % (len(hedge_clips), N))
print("  NEGATED ('not visible')                 : %4d / %d" % (len(neg_clips), N))
print()
print("  rows contributing a READ, by task: %s" % dict(per_task_read))
print()
print("VALUE DISTRIBUTION of the READ set:")
for v, n in sorted(values.items()):
    print("    %4d -> %d clip-rows" % (v, n))
print("UNIT distribution: %s" % dict(units))
print()
print("=" * 72)
print("SAMPLE of READ clips (clip, task, value, unit, matched text):")
for i, (cid, (task, val, u, txt)) in enumerate(sorted(read_clips.items())):
    if i >= 25:
        break
    try:
        print("  %s %-14s %4s %-5s | %s" % (cid[:8], task, val, u or "-",
                                            txt.replace("\n", " ")[:95]))
    except UnicodeEncodeError:
        print("  %s %-14s %4s <non-ascii>" % (cid[:8], task, val))
print()

# ------------------------------------------------ the dedicated VQA questions
vq = df[df["task"] == "vqa"]
print("=" * 72)
print("DEDICATED VQA QUESTIONS ABOUT SPEED-LIMIT SIGNS")
for pat in ("speed limit sign", "safe maximum speed", "school zone"):
    sub = vq[vq["question"].fillna("").str.contains(pat, case=False)]
    print("  question contains %-22r : %4d / %d clips" % (pat, len(sub), N))
    for q, n in sub["question"].value_counts().items():
        ans = []
        for _, r in sub[sub["question"] == q].head(400).iterrows():
            t = texts_of(r["raw_json"])
            ans.append(t[0].split("<|im_end|>")[0].strip() if t else "")
        yes = sum(1 for a in ans if a.lower().startswith("yes"))
        no = sum(1 for a in ans if a.lower().startswith("no"))
        withnum = sum(1 for a in ans if re.search(r"\d", a))
        print("     %-70s n=%3d  yes=%d no=%d with_digit=%d"
              % (q[:70], n, yes, no, withnum))
print()

# ------------------------------------------------------- grounding: sign boxes
g = df[df["task"] == "grounding_via_vqa"]
labs = Counter()
sign_q = 0
for _, r in g.iterrows():
    q = r["question"] if isinstance(r["question"], str) else ""
    if re.search(r"sign", q, re.I):
        sign_q += 1
    for t in texts_of(r["raw_json"]):
        for m in re.finditer(r'"label"\s*:\s*"([^"]+)"', t):
            labs[m.group(1)] += 1
print("=" * 72)
print("GROUNDING (grounding_via_vqa): rows=%d" % len(g))
print("  questions mentioning 'sign' : %d" % sign_q)
print("  box LABEL census (all rows, deduped per row not applied):")
for k, v in labs.most_common(25):
    print("     %-30s %d" % (k, v))

json.dump({"n_clips": int(N),
           "read": {k: {"task": v[0], "value": v[1], "unit": v[2],
                        "text": v[3]} for k, v in read_clips.items()},
           "n_read": len(read_clips), "n_hedged": len(hedge_clips),
           "n_negated": len(neg_clips),
           "values": {str(k): v for k, v in values.items()},
           "units": {str(k): v for k, v in units.items()},
           "grounding_sign_questions": sign_q,
           "box_labels": dict(labs)},
          open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print()
print("WROTE %s" % sys.argv[1])
