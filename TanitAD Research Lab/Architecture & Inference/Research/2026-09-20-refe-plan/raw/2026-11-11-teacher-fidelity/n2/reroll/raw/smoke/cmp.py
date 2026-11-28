import json,sys,numpy as np
sys.path.insert(0,'E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n2')
import tfn2_probe as P
bank=P.load_keys('F:/tanitad_scratch/tfn2_reroll/w')
base={}
for arm in ('R0','RJ5','RG8','R5HZ'):
    rows=[json.loads(l) for l in open(f'F:/tanitad_scratch/tfn2_reroll/run/smoke/{arm}.jsonl')]
    for r in rows:
        if 'error' in r: print(arm,r['error']); continue
        t=np.array(r['traj20']); k=(r['log'],r['token'])
        if arm=='R0': base[k]=t
        b=bank[k][0]
        print(arm,r['token'][:8],'vs bank max dxy %.2e'%np.abs(t[:,:2]-b[:,:2]).max(),'vs R0 max dxy %.3f'%np.abs(t[:,:2]-base[k][:,:2]).max(),'end x %.2f'%t[-1,0], 'knobs',{a:b for a,b in r['knobs'].items() if a!='max_jerk_long'},'sec',r['sec'])
