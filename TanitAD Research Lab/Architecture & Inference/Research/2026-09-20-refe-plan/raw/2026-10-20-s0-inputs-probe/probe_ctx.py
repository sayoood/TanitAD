import numpy as np, json, sys, os, collections
sys.path.insert(0, 'C:/Users/Admin/par18/s0probe')
from common_s0 import *
logs, tok_log, by = s0()
T = set(tok_log)
DD = 'D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19'
man = json.load(open(DD + '/ctx_local/ctx_manifest.json'))
print('manifest keys', list(man.keys()))
toks = man['tokens']
print('n tokens', len(toks), 'in S0', len(set(toks) & T), 'logs', len({tok_log[t] for t in toks if t in T}))
done = np.load(DD + '/ctx_local/ctx_done.npy')
print('ctx_done', done.shape, done.dtype, int(done.sum()))
ctx = np.load(DD + '/ctx_local/ctx.npy', mmap_mode='r')
print('ctx', ctx.shape, ctx.dtype)
for k, v in man.items():
    if k != 'tokens':
        print(k, json.dumps(v)[:300])
fid = [json.loads(l) for l in open(DD + '/ctx_local/fid.jsonl', encoding='utf-8') if l.strip()]
print('fid rows', len(fid), fid[-1] if fid else None)
# which logs do these tokens belong to, and do the done tokens cover them
dt = [t for t, d in zip(toks, done) if d]
print('done tokens in S0', len(set(dt) & T))
lc = collections.Counter(tok_log[t] for t in dt if t in T)
print('logs with ctx done', len(lc))
json.dump({'n_manifest_tokens': len(toks), 'n_done': int(done.sum()), 'n_done_in_S0': len(set(dt) & T), 'n_logs_done': len(lc), 'ctx_shape': list(ctx.shape), 'ctx_dtype': str(ctx.dtype),
           'logs_done': sorted(lc), 'tokens_done': sorted(dt)}, open('C:/Users/Admin/par18/s0probe/ctx_cov.json', 'w'))
