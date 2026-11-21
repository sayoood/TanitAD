import json, numpy as np, sys
d=sys.argv[1]
recs=[json.loads(l) for l in open(d+"/records.jsonl")]
for r in recs[:2]:
    print(r['f0'], r['start_frame'], r['t0_step'], r['n_steps'], r['v0'], r['schedule'], r['purpose'])
    ex=r['extra']
    print([ (e['step'],e['a_s'],e['a_t'],e['steps_since_t0'],round(e['v0_ms'],3)) for e in ex])
    cl=np.array(r['cl']); g=np.array(r['g']); tr=np.array(r['track']); P=np.array(r['plans'])
    print('cl',cl.shape, cl[:3].tolist(), cl[-1].tolist())
    print('g',g[:3].tolist(), g[-1].tolist())
    print('track',tr.shape, tr[0].tolist(), tr[18].tolist(), tr[-1].tolist())
    print('plan0 first', P[0,:3].tolist(), 'plan0 last', P[0,-1].tolist(), 'plan1 last', P[1,-1].tolist(), 'plan2 last', P[2,-1].tolist())
    print(ex[0]['s_ckpt'], ex[0]['s_pose'])
    print(r['calls'][0], r['calls'][-1])
    print(list(ex[0]['composite'].keys()) if isinstance(ex[0]['composite'],dict) else ex[0]['composite'])
    print(len(r['steps']))
