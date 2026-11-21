import pickle, sys, numpy as np, time
t=time.time()
det = pickle.load(open(sys.argv[1], 'rb'))
print('load s', round(time.time()-t,1), 'n rec', len(det))
r = det[0]
print({k: (type(v).__name__) for k, v in r.items()})
b = r['box3d']
print(type(b), len(b))
pk = b[0]
print({k: (getattr(v,'shape',None), getattr(v,'dtype',None)) for k, v in pk.items()})
nb = sum(1 for r in det if r.get('box3d'))
print('windows with box3d', nb, 'with agent', sum(1 for r in det if r.get('agent')))
print('slots', pk['logit'].shape, 'n pos', sum(int(r['box3d'][0]['pos'].sum()) for r in det if r.get('box3d')))
print('keys rec', list(r.keys()))
sh = sorted({r['sha12'] for r in det}); print('n clips', len(sh))
