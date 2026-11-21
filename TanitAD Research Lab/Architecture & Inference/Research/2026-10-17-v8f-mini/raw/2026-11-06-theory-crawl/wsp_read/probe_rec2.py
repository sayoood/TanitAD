import json, numpy as np, sys
d=sys.argv[1]
recs=[json.loads(l) for l in open(d+"/records.jsonl")]
for r in recs[:2]:
    ex=r['extra']; tr=np.array(r['track']); cl=np.array(r['cl'])
    print('f0',r['f0'],'t0_step',r['t0_step'],'n_steps',r['n_steps'],'track',tr.shape)
    for e in ex:
        s=e['step']
        print('step',s,'v0_ms',round(e['v0_ms'],4),'track[2s]v',round(tr[2*s][3],4),'track[s]v',round(tr[s][3],4), 'bwd diff 2s', round(float(np.hypot(*(tr[2*s,:2]-tr[2*s-1,:2]))/0.1),4))
    # cl vs track
    t0=2*r['t0_step']
    p0=tr[t0,:2]; 
    print('cl[0]',cl[0],'track[t0+2]-p0',tr[t0+2,:2]-p0,'track[t0+1]-p0',tr[t0+1,:2]-p0)
    print('last cl',cl[-1], 'track[t0+60]-p0', tr[t0+60,:2]-p0, 'raw', tr[-1,:2])
    # planned speeds
    P=np.array(r['plans'])
    for k in range(3):
        p=P[k]
        sp=np.hypot(*(np.diff(np.vstack([[0,0],p]),axis=0)).T)/0.2
        print('tick',k,'plan speeds (first 12 at 0.2s):',np.round(sp[:12],2))
        print('   exec track v every 0.2s from tick:', np.round(tr[2*ex[k]['step']:2*ex[k]['step']+25:2,3],2))
