import json, os, sys, glob, collections, hashlib
sys.path.insert(0, 'C:/Users/Admin/par18/s0probe')
from common_s0 import *
logs, tok_log, by = s0()
T = set(tok_log)
B = SR + '/run/batches'
tot = 0
seen = set()
roots = collections.Counter()
units = {}
for u in sorted(os.listdir(B)):
    p = os.path.join(B, u, 'batch_tokens.json')
    if not os.path.exists(p):
        units[u] = None
        continue
    j = json.load(open(p))
    toks = set(j['tokens'])
    units[u] = {'n_tokens': len(toks), 'in_S0': len(toks & T), 'overlap_with_prior_units': len(toks & seen), 'sha256': hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]}
    seen |= toks
    for t, r in j['token_root'].items():
        roots[os.path.basename(r.replace(chr(92), '/'))] += 1
print('plan files (batch_tokens.json) units:', sum(1 for v in units.values() if v), 'tokens covered (union):', len(seen), 'in S0:', len(seen & T))
print(roots.most_common(6))
for u, v in units.items():
    print(u, v)
# the master plan (single json) and sh5 plans
mp = SR + '/run/plan/master_plan.json'
print('master_plan', os.path.getsize(mp))
sh5 = ['plan_full', 'plan_partial_5776']
for s in sh5:
    p = 'D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/%s/plan.json' % s
    j = json.load(open(p))
    ks = list(j.keys())
    toks = j.get('tokens') or []
    print(s, ks[:8], len(toks), len(set(toks) & T))
json.dump({'units': units, 'union_tokens': len(seen), 'root_counter': roots.most_common(20)}, open('C:/Users/Admin/par18/s0probe/plans_cov.json', 'w'), indent=1)
