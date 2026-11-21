import re, collections, sys, json
sys.path.insert(0, 'C:/Users/Admin/par18/s0probe')
from common_s0 import *
logs, tok_log, by = s0()
L = set(logs)
g = collections.defaultdict(set)
for line in open('C:/Users/Admin/par18/s0probe/logdirs_find.txt', encoding='utf-8'):
    p = line.strip().replace(chr(92), '/')
    if not p:
        continue
    base = p.rsplit('/', 1)[1]
    par = p.rsplit('/', 1)[0]
    name = re.sub(r'_CAM_[A-Z0-9]+\.zip$', '', base)
    g[par].add(name)
rows = []
for par, ns in g.items():
    k = len(ns & L)
    if k:
        rows.append((k, len(ns), par))
for k, n, par in sorted(rows, reverse=True):
    print(k, n, par)
json.dump([{'S0_logs_present': k, 'entries': n, 'dir': p} for k, n, p in sorted(rows, reverse=True)], open('C:/Users/Admin/par18/s0probe/logdir_overlap.json', 'w'), indent=1)
