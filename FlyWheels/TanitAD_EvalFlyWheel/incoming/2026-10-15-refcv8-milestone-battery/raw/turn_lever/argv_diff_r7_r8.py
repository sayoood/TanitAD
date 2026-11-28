"""argv + effective-weight diff refcv7-r101-s0 vs refcv8-r101-s0 (read-only; prints to stdout)."""
import json, sys
p7 = sys.argv[1] if len(sys.argv) > 1 else 'E:/refcv7_eval_kit/ckpt/config.json'
p8 = sys.argv[2] if len(sys.argv) > 2 else 'E:/refcv8_eval_kit/ckpt/config.json'
c7, c8 = json.load(open(p7)), json.load(open(p8))
def parse(a):
    d, k = {}, None
    for x in a:
        if x.startswith('--'):
            k = x; d[k] = []
        else:
            d[k].append(x)
    return {k: ' '.join(v) for k, v in d.items()}
a7, a8 = parse(c7['argv']), parse(c8['argv'])
print('ONLY refcv7:', [k for k in a7 if k not in a8])
print('ONLY refcv8:'); [print('  ', k, a8[k]) for k in a8 if k not in a7]
print('DIFFERENT VALUE:'); [print('  ', k, '| r7 =', a7[k], '| r8 =', a8[k]) for k in a7 if k in a8 and a7[k] != a8[k]]
def ew(c): return {t['flag']: t['effective'] for t in c['effective_weights']['terms']}
e7, e8 = ew(c7), ew(c8)
print('EFFECTIVE WEIGHTS (nonzero in either):')
for f in sorted(set(e7) | set(e8)):
    if e7.get(f) or e8.get(f):
        print('  ', f, 'r7 =', e7.get(f), 'r8 =', e8.get(f))
print('anchors sha256 r7/r8:', c7['anchors']['sha256_installed'][:16], c8['anchors']['sha256_installed'][:16])
print('refcv8 block:', json.dumps({k: c8['refcv8'][k] for k in ('modulate_base','base_constraints','n_alloc','alloc_emit','lat_prior_dropout','cond_dropout','tf_start','tf_end','w_cons','w_sat','w_alloc_l1','w_listwise','w_subscore','emit_start','no_rc','nav_args_dropout')}))
print('init_from:', json.dumps(c8['init_from']))
print('ema:', json.dumps({k: c8['ema'][k] for k in ('decay','init')}))
