import json, numpy as np, glob
R="C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/rollouts"
for arm in ("P10","C0C_s0"):
    root = R if arm=="P10" else "C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/rollouts"
    d=[x for x in sorted(glob.glob(f"{root}/*/objects/T2/{arm}")) if __import__("os").path.isfile(x+"/records.jsonl")][3:5]
    print(arm, d[:2])
    for dd in d[:1]:
        for line in open(dd+"/records.jsonl"):
            r=json.loads(line); tr=np.array(r['track']); ex=r['extra']
            print(' keys', sorted(r.keys())[:40]); print(' extra0 keys', sorted(ex[0].keys()))
            print(' f0',r['f0'],'v0',r['v0'],'t0_step',r['t0_step'],'track',tr.shape, 'plans', np.array(r['plans']).shape)
            for k,e in enumerate(ex[:3]):
                s=e['step']; p=np.array(r['plans'][k])
                sp=np.hypot(*np.diff(np.vstack([[0,0],p]),axis=0).T)/0.2
                print('  k',k,'step',s,'v0_ms',round(e['v0_ms'],3),'track v@2s', round(tr[2*s][3],3) if 2*s<len(tr) else None,'plan speeds 0.2..2.4:', np.round(sp[:12],1), 'track v +2s', round(tr[2*s+20][3],2) if 2*s+20<len(tr) else None)
            break
