"""v7.2 tactical label census. ASCII-only output (cp1252 dev box)."""
import gzip, json, sys, collections

path = sys.argv[1]
recs = []
with gzip.open(path, "rt", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            recs.append(json.loads(line))

print("BLOB:", path)
print("n_records:", len(recs))
print("schema_version:", collections.Counter(r.get("schema_version") for r in recs).most_common())
print("vocab:", collections.Counter(r.get("vocab") for r in recs).most_common())
print()

# --- top-level key census ---
topkeys = collections.Counter()
for r in recs:
    topkeys.update(r.keys())
print("TOP-LEVEL KEYS (key -> n_records carrying it, of %d):" % len(recs))
for k, n in sorted(topkeys.items()):
    print("   %-24s %d" % (k, n))
print()

# --- g_tac structure ---
gt_keys = collections.Counter()
n_gtac = 0
for r in recs:
    g = r.get("g_tac")
    if isinstance(g, dict):
        n_gtac += 1
        gt_keys.update(g.keys())
print("g_tac present on %d/%d records" % (n_gtac, len(recs)))
print("g_tac SUBKEYS:")
for k, n in sorted(gt_keys.items()):
    print("   %-24s %d" % (k, n))
print()

# --- the tactical GOAL SET token census ---
goal_tok = collections.Counter()
n_with_goals = 0
for r in recs:
    g = (r.get("g_tac") or {}).get("goals") or {}
    if g:
        n_with_goals += 1
        goal_tok.update(g.keys())
print("records carrying a NON-EMPTY g_tac.goals set: %d/%d" % (n_with_goals, len(recs)))
print("TACTICAL GOAL TOKEN FREQUENCY (token -> n_records):")
for k, n in goal_tok.most_common():
    print("   %-32s %d" % (k, n))
print()

# --- traffic light specifically ---
tl = {k: v for k, v in goal_tok.items() if "TRAFFIC_LIGHT" in k}
tl_total = 0
for r in recs:
    g = (r.get("g_tac") or {}).get("goals") or {}
    if any("TRAFFIC_LIGHT" in k for k in g):
        tl_total += 1
print("TRAFFIC LIGHT: %d/%d records carry any TRAFFIC_LIGHT_* tactical goal" % (tl_total, len(recs)))
for k in sorted(tl):
    print("   %-32s %d/%d" % (k, tl[k], len(recs)))
print()

# --- one full example record carrying a traffic light token ---
ex = None
for r in recs:
    g = (r.get("g_tac") or {}).get("goals") or {}
    if any("TRAFFIC_LIGHT_REACT_RED" in k for k in g):
        ex = r
        break
if ex is None:
    for r in recs:
        g = (r.get("g_tac") or {}).get("goals") or {}
        if any("TRAFFIC_LIGHT" in k for k in g):
            ex = r
            break
print("EXAMPLE RECORD WITH A TRAFFIC-LIGHT GOAL:")
if ex is None:
    print("   NONE FOUND")
else:
    print(json.dumps(ex, indent=1, sort_keys=True)[:5000])
