import json,numpy as np
T=np.load('F:/TanitAD-scratch/datascale/navtok/navtok_table.npz'); V=list(T['vocab'])
g=json.load(open('F:/TanitAD-scratch/strategic/proto/eval_grid_windows.json'))
def side(c):
    n=V[c]
    if n.startswith('TURN') or n.startswith('UTURN'): pass
    return n
turn_ids={i:('L' if n.endswith('_L') or n.startswith('TURN_L') or 'LEFT' in n else 'R' if n.endswith('_R') or n.startswith('TURN_R') or 'RIGHT' in n else '?') for i,n in enumerate(V) if n.startswith('TURN') or n.startswith('UTURN')}
print([n for n in V if n.startswith(('TURN','UTURN'))])
clip2i={c:i for i,c in enumerate(T['eval_clip'].tolist())}
def rows(clips_steps):
    out=[]
    for c,j in clips_steps:
        i=clip2i[c]; 
        if j>=T['eval_n'][i]: out.append(None); continue
        out.append(T['eval_off'][i]+j)
    return out
# eval141 grid
cs=[(c,t) for c in g['order'] for t in g['windows'][c]]
r=rows(cs); ok=[x for x in r if x is not None]
print('grid windows',len(cs),'in range',len(ok))
r=np.array(ok); fl=T['eval_flag'][r]; print('flag==0 (text row ok)',(fl==0).sum(), 'flag values',np.unique(fl,return_counts=True))
cls=T['eval_cls_id'][r]; d=T['eval_dist_m'][r]; okm=(fl==0)
isturn=np.array([V[c].startswith('TURN') for c in cls]); anyman=np.array([d_==d_ and d_>=0 for d_ in d])
print('text TURN_* windows',int((isturn&okm).sum()))
for lo,hi in [(0,30),(30,60),(60,150),(150,1e9)]:
    m=okm&isturn&(d>=lo)&(d<hi); print(' turn dist [%g,%g) m: n=%d eps=%d'%(lo,hi,m.sum(),len({cs[k][0] for k in np.where(m)[0]} if False else [])))
# episodes per band
epi=np.array([cs[k][0] for k in range(len(cs)) if r is not None][:len(r)])
for lo,hi in [(0,30),(30,60),(60,150),(150,1e9)]:
    m=okm&isturn&(d>=lo)&(d<hi); print('  eps',len(set(epi[m].tolist())))
# ACTSENS
z=np.load('F:/TanitAD-scratch/actsens/read/out/NULL_s3/win/windows.npz',allow_pickle=True)
ingrid=0; tot=0
for e,n in zip(z['eid'],z['now']):
    tot+=1; ingrid+= int((int(n)+3) in set(g['windows'][str(e)]))
print('ACTSENS windows with now+3 in grid',ingrid,'of',tot)
rr=rows([(str(e),int(n)+3) for e,n in zip(z['eid'],z['now'])]); 
rr=np.array([x if x is not None else 0 for x in rr]); fl=T['eval_flag'][rr]; cls=T['eval_cls_id'][rr]; d=T['eval_dist_m'][rr]; v0=z['v0']
isturn=np.array([V[c].startswith('TURN') for c in cls])&(fl==0)
print('ACTSENS text ok',int((fl==0).sum()),'turn',int(isturn.sum()), 'eps with turn',len(set(z['eid'][isturn].tolist())))
tt=d/np.maximum(v0,1.0)
for lo,hi in [(0,6),(6,12),(12,30),(30,1e9)]:
    m=isturn&(tt>=lo)&(tt<hi); print(' turn time-to [%g,%g) s: n=%d eps=%d'%(lo,hi,m.sum(),len(set(z['eid'][m].tolist()))))
np.save('F:/TanitAD-scratch/strategic/proto/actsens_rows.npy',rr)
