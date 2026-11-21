import gzip, json, numpy as np, collections as C
R=[json.loads(l) for l in gzip.open('E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot/raw_states.jsonl.gz','rt')]
T=0.5
def brake_dist(v,a=2.0):
    ts=v/a
    return v*T-0.5*a*T*T if ts>=T else v*v/(2*a)
def recompute(logs,v0,limit,variant):
    tot=0.0; vprev=v0; per=[]
    ref_v=v0
    for c in logs:
        norm=limit*0.1*5
        d=c['progress']*norm   # dist (approx; clipped 0..1.5)
        if variant=='orig': p=c['progress']
        elif variant=='A_percycle':
            p=max(0.0,d-brake_dist(vprev))/norm
        elif variant=='B_episode_ref':
            # reference: -2 m/s2 stop from episode-start speed; distance in this cycle
            t0=c['cycle']*T; 
            s=lambda t: (v0*min(t,v0/2)-min(t,v0/2)**2) if v0>0 else 0.0
            ref=s(t0+T)-s(t0)
            p=max(0.0,d-ref)/norm
        r=0.0 if c['hard'] else p*c['qprod']
        tot+=r; per.append(r); vprev=c['v']
    return tot,per
lim={s['idx']:max(s['limit'],2.0) for s in R}
import sys
for variant in ('orig','A_percycle','B_episode_ref'):
    pd=[];stop=[];sl=[];ft=[]
    for s in R:
        L=lim[s['idx']]
        a=s['arms'][f"log:{s['pdm_best']}"]; pd.append(recompute(a['logs'],s['v0'],L,variant)[0])
        if s['v0']>0.5 and 'stop:log' in s['reg']:
            lg=s['reg']['stop:log']['logs']; t,_=recompute(lg,s['v0'],L,variant); stop.append(t/len(lg))
        if 'slowest:log' in s['reg']: sl.append(recompute(s['reg']['slowest:log']['logs'],s['v0'],L,variant)[0])
        # all-arm mean return
    allr=[recompute(a['logs'],s['v0'],lim[s['idx']],variant)[0] for s in R for k,a in s['arms'].items() if k.startswith('log:')]
    print(variant,'PDMbest log mean ret',round(np.mean(pd),3),'| do-nothing mean reward/cycle (moving)',round(np.mean(stop),4),'| slowest ret',round(np.mean(sl),3),'| all-24-arms mean ret',round(np.mean(allr),3))
