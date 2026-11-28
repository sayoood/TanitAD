import json, os, sys
A9 = 'E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-01-goal-trigger/a9'
cur = json.load(open(sys.argv[1]))
base = lambda x: x.replace('\\', '/').rsplit('/', 1)[-1]
n = bad = 0
seen = set()
for fn in ['inputs_adopted_rest.json'] + sorted(f for f in os.listdir(A9) if f.startswith('inputs_pdm_route')):
    rec = json.load(open(A9 + '/' + fn))
    for t, r in rec.items():
        if t in cur and t not in seen:
            seen.add(t)
            n += 1
            ok = (r['ego'] == cur[t]['ego'] and r['goal'] == cur[t]['goal'] and [base(x) for x in r['frames']] == [base(x) for x in cur[t]['frames']])
            if not ok:
                bad += 1
                if bad < 4:
                    print(fn, t, r['ego'] == cur[t]['ego'], r['goal'] == cur[t]['goal'], r['goal'][:4], cur[t]['goal'][:4])
print('compared', n, 'bad', bad, 'of', len(cur))
t = next(iter(cur))
print(t, {k: v for k, v in cur[t].items()})
