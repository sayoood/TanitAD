import json
rows=[]
for l in open('E:/refcv8_eval_kit/thor_reads/metrics_20261017T173755.jsonl', encoding='utf-8'):
    l=l.strip()
    if l.startswith('{'):
        try:
            r=json.loads(l)
            if 'elapsed_s' in r and 'traj' in r: rows.append((r['step'],r['elapsed_s']))
        except Exception: pass
rows.sort()
for a,b in [(15000,16000),(20000,21000),(21000,22000)]:
    s=[x for x in rows if a<=x[0]<=b]
    if len(s)>1: print(a,b, (s[-1][1]-s[0][1])/(s[-1][0]-s[0][0]), 's/step (elapsed_s diff; restarts would inflate)')
print(rows[-1])
