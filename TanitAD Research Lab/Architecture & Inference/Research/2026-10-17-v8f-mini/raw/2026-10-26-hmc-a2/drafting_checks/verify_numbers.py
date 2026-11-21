import json, re, sys
sys.stdout.reconfigure(encoding='utf-8')
doc_path, *json_paths = sys.argv[1:]
txt = open(doc_path, encoding='utf-8').read().replace('\u2212', '-')
floats = []
def walk(o):
    if isinstance(o, bool): return
    if isinstance(o, (int, float)): floats.append(float(o))
    elif isinstance(o, dict):
        for v in o.values(): walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
for p in json_paths: walk(json.load(open(p, encoding='utf-8')))
# strip hex hashes / long identifiers so they are not tokenised
t2 = re.sub(r'\b[0-9a-f]{8,}\b', ' ', txt)
toks = re.findall(r'(?<![\w.])[+\-]?\d+\.\d{3,}(?:e[+\-]?\d+)?', t2)
seen = set(); bad = []
for s in toks:
    if s in seen: continue
    seen.add(s)
    if 'e' in s.lower():
        v = float(s); d = len(s.lower().split('e')[0].split('.')[1])
        ok = any(abs(f) > 0 and abs(f - v) <= abs(v) * 10 ** -(d - 0) for f in floats if abs(f) < 1e-3 or abs(f) > 1e3)
        ok = ok or any(abs(f - v) <= abs(v) * 10 ** -d for f in floats)
    else:
        d = len(s.split('.')[1]); v = float(s)
        ok = any(abs(round(f, d) - v) < 10 ** -(d + 3) or abs(abs(round(f, d)) - abs(v)) < 10 ** -(d + 3) and False for f in floats)
    if not ok: bad.append(s)
print(doc_path.split('/')[-1], 'distinct numeric tokens (>=3 decimals):', len(seen), ' not found among JSON floats:', len(bad))
print(bad)
