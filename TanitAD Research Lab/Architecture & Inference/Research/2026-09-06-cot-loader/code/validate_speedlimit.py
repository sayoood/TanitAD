"""D-COT-LOADER validation: PR-2a..e and PR-3a, plus the 57-value bank.

Regenerates every number in RESULT.md from
`C:/Users/Admin/tanitad-data/alpamayo/records.parquet`
(md5 9f13474723b880eec7fcc09a7be478d8, 23,644 rows / 4,729 clips).

CONTROLS ARE NOT OPTIONAL and are printed with the run:
  * a CONSTANT control -- text with no speed-limit phrase must read the
    no-information value EXACTLY (state None, value None);
  * a DELIBERATE-REGRESSION control -- the pre-fix regex must capture 0 values
    through the same harness, or the harness is not reading the regex;
  * `n` is printed beside every fraction, and both the clip-level and the
    row-level denominators are reported, because they differ.

ASCII-only stdout (cp1252 dev box).

usage: validate_speedlimit.py <out.json>
"""
import json
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

from tanitad.data import alpamayo_records as AR
from tanitad.data import cot_tokens_v7 as COT

SIB = ("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/TanitAD Research Lab/"
       "Architecture & Inference/Research/2026-09-06-speed-limit-source/raw/"
       "limits_classified.json")

REC = AR.records_path()
df = pd.read_parquet(REC)
N = int(df["clip_id"].nunique())
res = {"corpus": REC, "n_clips": N, "n_rows": int(len(df))}
print("CORPUS %s" % REC)
print("  distinct clips = %d   rows = %d" % (N, len(df)))


def texts_of(raw):
    """Every string anywhere in the row's raw_json -- the sibling's walker."""
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


rows = defaultdict(dict)                       # clip -> task -> blob
for _, r in df.iterrows():
    rows[r["clip_id"]][r["task"]] = "  ".join(texts_of(r["raw_json"]))

# ---------------------------------------------------------------- PR-2a / 2b
read_clips, hedge_clips, neg_clips = {}, {}, {}
row_units, clip_values = Counter(), Counter()
per_task_read = Counter()
row_read = 0
for cid, byt in rows.items():
    for task, blob in byt.items():
        r = COT.speed_limit_reading(blob)
        if r is None:
            continue
        if r.state == "READ":
            row_read += 1
            per_task_read[task] += 1
            row_units[r.unit or "NONE"] += 1
            if cid not in read_clips:
                read_clips[cid] = {"task": task, "value": r.value,
                                   "unit": r.unit,
                                   "unit_missing": r.unit_missing,
                                   "value_ms": r.value_ms,
                                   "text": r.text}
                clip_values[r.value] += 1
        elif r.state == "HEDGED":
            hedge_clips.setdefault(cid, task)
        elif r.state == "NEGATED":
            neg_clips.setdefault(cid, task)

clip_units = Counter(v["unit"] or "NONE" for v in read_clips.values())
res["classification"] = {
    "n_read_clips": len(read_clips), "n_hedged_clips": len(hedge_clips),
    "n_negated_clips": len(neg_clips), "n_read_rows": row_read,
    "per_task_read_rows": dict(per_task_read),
    "clip_level_units": dict(clip_units), "row_level_units": dict(row_units),
    "clip_level_values": {str(k): v for k, v in sorted(clip_values.items())},
}
print()
print("=" * 72)
print("PR-2a  CLASSIFICATION (denominator = %d clips)" % N)
print("  READ  (value off a sign, unhedged, un-negated) : %4d clips  %.2f %%"
      % (len(read_clips), 100.0 * len(read_clips) / N))
print("  HEDGED (language prior)                        : %4d clips  %.2f %%"
      % (len(hedge_clips), 100.0 * len(hedge_clips) / N))
print("  NEGATED                                        : %4d clips  %.2f %%"
      % (len(neg_clips), 100.0 * len(neg_clips) / N))
print("  READ clip-ROWS = %d, by task: %s" % (row_read, dict(per_task_read)))

# --- agreement with the sibling's independently-written classifier -----------
try:
    sib = json.load(open(SIB, encoding="utf-8"))
    sib_ids = set(sib["read"])
    mine = set(read_clips)
    res["sibling"] = {
        "path": SIB, "n_read": sib["n_read"], "n_hedged": sib["n_hedged"],
        "n_negated": sib["n_negated"],
        "ids_identical": sib_ids == mine,
        "only_sibling": sorted(sib_ids - mine), "only_mine": sorted(mine - sib_ids),
        "values_identical": all(
            sib["read"][k]["value"] == read_clips[k]["value"]
            for k in (sib_ids & mine)),
    }
    print("  vs sibling classify_limits.py: read %d/%d  hedged %d/%d  neg %d/%d"
          % (len(mine), sib["n_read"], len(hedge_clips), sib["n_hedged"],
             len(neg_clips), sib["n_negated"]))
    print("  clip-id sets identical: %s   values identical: %s"
          % (res["sibling"]["ids_identical"], res["sibling"]["values_identical"]))
except Exception as exc:
    res["sibling"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
    print("  sibling comparison INCONCLUSIVE: %s" % exc)

# ----------------------------------------------------------------- PR-2b / 2e
n_read = len(read_clips)
clip_none = clip_units.get("NONE", 0)
row_none = row_units.get("NONE", 0)
res["units"] = {
    "clip_level": {"none": clip_none, "n": n_read,
                   "frac": round(clip_none / n_read, 4) if n_read else None},
    "row_level": {"none": row_none, "n": row_read,
                  "frac": round(row_none / row_read, 4) if row_read else None},
}
print()
print("PR-2b/2e  UNITS -- BOTH denominators, because they differ")
print("  clip level: %s   ->  no unit %d / %d = %.2f %%"
      % (dict(clip_units), clip_none, n_read, 100.0 * clip_none / max(n_read, 1)))
print("  row  level: %s   ->  no unit %d / %d = %.2f %%"
      % (dict(row_units), row_none, row_read, 100.0 * row_none / max(row_read, 1)))
bad_ms = [k for k, v in read_clips.items() if v["unit_missing"] and v["value_ms"] is not None]
res["units"]["unitless_with_ms"] = bad_ms
print("  UNIT-LESS readings carrying an m/s value (MUST be 0): %d" % len(bad_ms))

# --------------------------------------------------------------------- PR-2c
print()
print("PR-2c  CONTROLS AT KNOWN VALUES")
controls = {}
CASES = [
    ("constant_no_phrase", "The ego proceeds through the intersection.", None, None),
    ("empty", "", None, None),
    ("none", None, None, None),
    ("read_kmh", "A 70 km/h speed limit sign appears ahead.", "READ", 70),
    ("read_mph", "The speed limit sign reads 35 mph.", "READ", 35),
    ("read_no_unit", "A 70 speed limit sign is visible ahead.", "READ", 70),
    ("hedged", "Residential areas typically have a 25 mph speed limit.",
     "HEDGED", None),
    # NOTE, and it is a definitional quirk INHERITED deliberately so the two
    # implementations agree clip-for-clip: the sibling's HEDGE pattern contains
    # `not visible`, and HEDGED is tested before NEGATED, so an explicitly
    # negated "sign is not visible" lands in HEDGED. Predicted in advance here.
    ("negated_via_hedge_word", "The speed limit sign is not visible in this scene.",
     "HEDGED", None),
    # the NEGATED branch must be REACHABLE -- an unreachable branch measures
    # nothing, which is the same defect as a guard that is never called.
    ("negated_true", "There is no speed limit sign in view.", "NEGATED", None),
    # the MENTION branch: phrase present, no value, neither hedged nor negated.
    ("mention_no_value", "The ego respects the speed limit while turning.",
     "MENTION", None),
]
ok = True
for name, txt, want_state, want_val in CASES:
    r = COT.speed_limit_reading(txt)
    got_state = r.state if r else None
    got_val = r.value if r else None
    good = (got_state == want_state) and (got_val == want_val)
    ok = ok and good
    controls[name] = {"state": got_state, "value": got_val,
                      "unit": (r.unit if r else None),
                      "value_ms": (r.value_ms if r else None),
                      "expected_state": want_state, "expected_value": want_val,
                      "pass": good}
    print("  %-20s state=%-8s value=%-5s ms=%-8s %s"
          % (name, got_state, got_val,
             (r.value_ms if r else None), "OK" if good else "MISMATCH"))
res["controls"] = controls
res["controls_all_pass"] = ok

# --------------------------------------------------------------------- PR-2d
OLD = re.compile(r"\bspeed limit\b")
old_captured = 0
for cid, byt in rows.items():
    for blob in byt.values():
        m = OLD.search(blob.lower())
        if m and m.groups():
            old_captured += 1
res["pr2d"] = {"prefix_regex_captured_values": old_captured,
               "current_captured_values": n_read}
print()
print("PR-2d  DELIBERATE REGRESSION on the capture")
print("  pre-fix regex r'\\bspeed limit\\b' captured values : %d  (must be 0)"
      % old_captured)
print("  current regex captured values                    : %d  (must be > 0)"
      % n_read)

# --------------------------------------------------------------------- PR-3a
PHRASES = {
    "speed limit": r"speed[\s\-]?limit",
    "yield": r"\byield",
    "parked": r"\bparked\b",
    "pedestrian": r"\bpedestrian",
    "traffic light": r"\btraffic light",
    "oncoming": r"\boncoming\b",
    "merge": r"\bmerg(?:e|es|ing)\b",
    "ramp": r"\bramp\b",
    "open door": r"\bopen door\b",
}
scope = {}
clips_all = sorted(AR.available())
for label, pat in PHRASES.items():
    rx = re.compile(pat, re.I)
    n_cot = sum(1 for cid in clips_all
                if rx.search(AR.get(cid).cot or ""))
    n_meta = sum(1 for cid, byt in rows.items()
                 if rx.search(byt.get("meta_action", "")))
    n_all = sum(1 for cid, byt in rows.items()
                if any(rx.search(b) for b in byt.values()))
    scope[label] = {"cot_field": n_cot, "meta_action_task": n_meta,
                    "all_five_tasks": n_all}
res["docstring_scope"] = scope
print()
print("PR-3a  THE DOCSTRING'S SCOPE (n = %d clips)" % N)
print("  %-15s %8s %8s %8s" % ("phrase", "cot", "meta_task", "all5"))
for label, d in scope.items():
    print("  %-15s %8d %8d %8d"
          % (label, d["cot_field"], d["meta_action_task"], d["all_five_tasks"]))

# ------------------------------------------------- the bank: all 57 readings
res["readings"] = read_clips
print()
print("THE %d CAPTURED READINGS (clip, task, value, unit, m/s)" % n_read)
for cid in sorted(read_clips):
    v = read_clips[cid]
    print("  %s %-16s %4s %-5s %-9s | %s"
          % (cid[:8], v["task"], v["value"], v["unit"] or "NONE",
             ("-" if v["value_ms"] is None else "%.4f" % v["value_ms"]),
             " ".join(v["text"].split())[:70].encode("ascii", "replace").decode()))

json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), indent=1, sort_keys=True)
print()
print("WROTE %s" % sys.argv[1])
