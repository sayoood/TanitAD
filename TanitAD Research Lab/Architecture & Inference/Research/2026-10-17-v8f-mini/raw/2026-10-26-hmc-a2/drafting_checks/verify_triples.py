import json, re, sys
sys.stdout.reconfigure(encoding='utf-8')
doc_path, *json_paths = sys.argv[1:]
txt = open(doc_path, encoding='utf-8').read().replace('\u2212', '-')
triples = []   # (mean_or_delta, lo, hi) from every dict in every JSON
def walk(o, src):
    if isinstance(o, dict):
        ks = o.keys()
        m = o.get('mean', o.get('delta'))
        if m is not None and 'lo' in ks and 'hi' in ks and all(isinstance(x, (int, float)) for x in (m, o['lo'], o['hi'])):
            triples.append((float(m), float(o['lo']), float(o['hi']), src))
        for v in o.values(): walk(v, src)
    elif isinstance(o, list):
        for v in o: walk(v, src)
for p in json_paths:
    walk(json.load(open(p, encoding='utf-8')), p.split('/')[-1])
num = r'[+\-]?\d+\.\d+'
pat = re.compile(r'(' + num + r')\s*\[\s*(' + num + r')\s*,\s*(' + num + r')\s*\]')
def dec(s): return len(s.split('.')[1])
found = 0; bad = []
for m in pat.finditer(txt):
    a, b, c = m.group(1), m.group(2), m.group(3)
    found += 1
    ok = False
    for (x, lo, hi, src) in triples:
        if all(abs(round(v, dec(s)) - float(s)) < 10 ** -(dec(s) + 3) for v, s in ((x, a), (lo, b), (hi, c))):
            ok = True; break
    if not ok:
        line = txt.count('\n', 0, m.start()) + 1
        bad.append((line, m.group(0)))
print('doc:', doc_path)
print('json triples loaded:', len(triples), 'from', [p.split('/')[-1] for p in json_paths])
print('point [low, high] patterns found in doc:', found)
print('matched in one JSON object:', found - len(bad))
print('NOT matched:', len(bad))
for line, s in bad: print('  line', line, ':', s)
