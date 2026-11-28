import sys, time, numpy as np
sys.path.insert(0, 'C:/Users/Admin/refcv8_pf/refcv9_mapsel/code'); sys.path.insert(0, 'C:/Users/Admin/refcv8_pf/refcv9_l2/code')
import mapsel_data as D, l2_common as C
t=time.time(); E=C.load_eval_pf(0); print('E loaded', len(E['ws']), round(time.time()-t,1),'s')
keys=list(zip(E['clip'].tolist(), E['ws'].tolist()))
h,v=D.human_plans(keys); print('human', h.shape, v.mean())
# parity: S6 g at 1..6 s  <-> human slots idx [1,3,4,5,6,7]
d=np.abs(h[:,[1,3,4,5,6,7]]-E['g'].astype(np.float32))
print('parity max abs', float(d[v].max()), 'p99', float(np.quantile(d[v],0.99)))
r=D.GTReader(); print('has gt', sum(r.has.values()), 'of', len(r.has))
t=time.time(); c=r.codes(0, keys[0][1]); print('codes', None if c is None else (c.shape,c.dtype, np.unique(c)), round(time.time()-t,2))
